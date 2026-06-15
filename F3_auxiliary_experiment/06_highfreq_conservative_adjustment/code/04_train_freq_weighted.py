import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2] / "code"))

from common import DATA_DIR, DT, LOG_DIR, SOURCE_DATA_DIR, ensure_dirs
from losses import BandwidthLoss
from model import UNetCBAM


class ZeroMeanResidualModel(nn.Module):
    def __init__(self, base_c=32):
        super().__init__()
        self.net = UNetCBAM(base_c=base_c)

    def forward(self, x):
        out = self.net(x)
        return out - out.mean(dim=(-2, -1), keepdim=True)


class ResidualPatchDataset(Dataset):
    def __init__(self, split, augment=False):
        self.inputs = np.load(SOURCE_DATA_DIR / f"{split}_inputs.npy", mmap_mode="r")
        self.labels = np.load(SOURCE_DATA_DIR / f"{split}_labels.npy", mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = np.asarray(self.inputs[idx], dtype=np.float32)
        y = np.asarray(self.labels[idx], dtype=np.float32)
        r = y - x
        if self.augment:
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
                r = np.flip(r, axis=1)
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=0)
                y = np.flip(y, axis=0)
                r = np.flip(r, axis=0)
        return (
            torch.from_numpy(x.copy()).unsqueeze(0),
            torch.from_numpy(y.copy()).unsqueeze(0),
            torch.from_numpy(r.copy()).unsqueeze(0),
        )


class BiasCorrectedFreqWeightedLoss(nn.Module):
    def __init__(self, mean_weight=0.1, hf_loss_weight=0.5, hf_boost_factor=2.0):
        super().__init__()
        self.wide_loss = BandwidthLoss(lambda_mae=1.0, lambda_freq=0.5, lambda_phase=0.3)
        self.residual_loss = nn.L1Loss()
        self.mean_weight = float(mean_weight)
        self.hf_loss_weight = float(hf_loss_weight)
        self.hf_boost_factor = float(hf_boost_factor)

    def high_freq_weighted_amp_loss(self, pred, target):
        nt = pred.shape[-2]
        pred_spec = torch.fft.rfft(pred, dim=-2)
        target_spec = torch.fft.rfft(target, dim=-2)
        freqs = torch.fft.rfftfreq(nt, d=DT, device=pred.device)
        weights = torch.ones_like(freqs)
        weights[freqs >= 35.0] = self.hf_boost_factor
        weights = weights.view(1, 1, -1, 1)
        amp_diff = torch.abs(torch.abs(pred_spec) - torch.abs(target_spec))
        return torch.mean(amp_diff * weights)

    def forward(self, x, residual_pred, wide_target, residual_target):
        wide_pred = x + residual_pred
        loss_wide, loss_dict = self.wide_loss(wide_pred, wide_target)
        loss_residual = self.residual_loss(residual_pred, residual_target)
        loss_mean = torch.abs(residual_pred.mean(dim=(-2, -1))).mean()
        loss_hf = self.high_freq_weighted_amp_loss(wide_pred, wide_target)
        total = (
            loss_wide
            + 0.2 * loss_residual
            + self.mean_weight * loss_mean
            + self.hf_loss_weight * loss_hf
        )
        loss_dict["total"] = float(total.detach().cpu())
        loss_dict["residual"] = float(loss_residual.detach().cpu())
        loss_dict["mean"] = float(loss_mean.detach().cpu())
        loss_dict["hf_weighted"] = float(loss_hf.detach().cpu())
        return total, loss_dict


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "total": 0.0,
        "mae": 0.0,
        "freq": 0.0,
        "phase": 0.0,
        "residual": 0.0,
        "mean": 0.0,
        "hf_weighted": 0.0,
    }
    for x, y, residual in loader:
        x = x.to(device)
        y = y.to(device)
        residual = residual.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            residual_pred = model(x)
            loss, loss_dict = criterion(x, residual_pred, y, residual)
            if is_train:
                loss.backward()
                optimizer.step()
        for key in totals:
            totals[key] += loss_dict[key]
    return {key: val / max(1, len(loader)) for key, val in totals.items()}


def plot_history(history):
    keys = ["total", "mae", "freq", "phase", "residual", "mean", "hf_weighted"]
    fig, axes = plt.subplots(1, len(keys), figsize=(28, 4))
    epochs = np.asarray(history["epoch"])
    for ax, key in zip(axes, keys):
        ax.plot(epochs, history[f"train_{key}"], label="Train")
        ax.plot(epochs, history[f"val_{key}"], label="Val")
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(LOG_DIR / "freq_weighted_training_curves.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-c", type=int, default=32)
    parser.add_argument("--mean-weight", type=float, default=0.1)
    parser.add_argument("--hf-loss-weight", type=float, default=0.5)
    parser.add_argument("--hf-boost-factor", type=float, default=2.0)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=300)
    args = parser.parse_args()

    ensure_dirs()
    (DATA_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader = DataLoader(
        ResidualPatchDataset("train", augment=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ResidualPatchDataset("val", augment=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroMeanResidualModel(base_c=args.base_c).to(device)
    criterion = BiasCorrectedFreqWeightedLoss(
        mean_weight=args.mean_weight,
        hf_loss_weight=args.hf_loss_weight,
        hf_boost_factor=args.hf_boost_factor,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience, factor=0.5
    )

    keys = ["total", "mae", "freq", "phase", "residual", "mean", "hf_weighted"]
    history = {f"{split}_{key}": [] for split in ("train", "val") for key in keys}
    history["epoch"] = []
    history["lr"] = []
    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, criterion, device, optimizer)
        val = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val["total"])
        lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["lr"].append(lr)
        for key in keys:
            history[f"train_{key}"].append(train[key])
            history[f"val_{key}"].append(val[key])

        print(
            f"Epoch {epoch}: train={train['total']:.4f}, val={val['total']:.4f}, "
            f"hf={val['hf_weighted']:.4f}, lr={lr:.2e}",
            flush=True,
        )

        if val["total"] < best_val:
            best_val = val["total"]
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val,
                "base_c": args.base_c,
                "mean_weight": args.mean_weight,
                "hf_loss_weight": args.hf_loss_weight,
                "hf_boost_factor": args.hf_boost_factor,
            }, DATA_DIR / "checkpoints" / "best_freq_weighted_model.pth")
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val["total"],
            "base_c": args.base_c,
            "mean_weight": args.mean_weight,
            "hf_loss_weight": args.hf_loss_weight,
            "hf_boost_factor": args.hf_boost_factor,
        }, DATA_DIR / "checkpoints" / "last_freq_weighted_model.pth")

    np.save(LOG_DIR / "freq_weighted_training_history.npy", history)
    plot_history(history)
    print(f"Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
