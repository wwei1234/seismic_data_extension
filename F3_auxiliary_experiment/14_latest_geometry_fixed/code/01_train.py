"""
Train a residual prediction network with residual-band-constrained loss.

Key difference from the freq_weighted experiment (06):
    Instead of applying frequency weighting on the full wide-band prediction,
    the frequency-domain losses (amplitude + phase, energy ratio) operate
    directly on the residual, which naturally has power only in the infill band
    (25-75 Hz).  This avoids the low-frequency dominance problem.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    DATA_DIR,
    DT,
    FIGURE_DIR,
    LOG_DIR,
    SOURCE_DATA_DIR,
    ensure_dirs,
    residual_band_loss,
    residual_energy_ratio_loss,
)
from model import UNetCBAM


# ── Model ────────────────────────────────────────────────────────────────────

class ZeroMeanResidualModel(nn.Module):
    """UNet-CBAM whose output is forced to zero-mean per patch."""

    def __init__(self, base_c=32):
        super().__init__()
        self.net = UNetCBAM(base_c=base_c)

    def forward(self, x):
        out = self.net(x)
        return out - out.mean(dim=(-2, -1), keepdim=True)


# ── Dataset ──────────────────────────────────────────────────────────────────


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


# ── Loss ─────────────────────────────────────────────────────────────────────


class ResidualBandConstrainedLoss(nn.Module):
    """
    Total loss with residual-band frequency constraints.

    Loss components:
        loss_wide    – L1 on the reconstructed wide-band prediction
        loss_res_l1  – L1 on the residual directly
        loss_mean    – penalise non-zero residual mean
        loss_rb      – amplitude + phase matching in the 25-75 Hz residual band
        loss_er      – asymmetric energy-ratio loss (penalise high-freq deficit)
    """

    def __init__(self, mean_weight=0.1):
        super().__init__()
        self.mean_weight = float(mean_weight)

    def forward(self, x, residual_pred, wide_target, residual_target):
        wide_pred = x + residual_pred

        loss_wide = F.l1_loss(wide_pred, wide_target)
        loss_res_l1 = F.l1_loss(residual_pred, residual_target)
        loss_mean = torch.abs(residual_pred.mean(dim=(-2, -1))).mean()
        loss_rb = residual_band_loss(residual_pred, residual_target)
        loss_er = residual_energy_ratio_loss(residual_pred, residual_target)

        total = (
            1.0 * loss_wide
            + 0.2 * loss_res_l1
            + self.mean_weight * loss_mean
            + 1.0 * loss_rb
            + 0.5 * loss_er
        )

        loss_dict = {
            "total": float(total.detach().cpu()),
            "wide": float(loss_wide.detach().cpu()),
            "residual": float(loss_res_l1.detach().cpu()),
            "mean": float(loss_mean.detach().cpu()),
            "rb": float(loss_rb.detach().cpu()),
            "er": float(loss_er.detach().cpu()),
        }
        return total, loss_dict


# ── Training loop ────────────────────────────────────────────────────────────


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"total": 0.0, "wide": 0.0, "residual": 0.0, "mean": 0.0, "rb": 0.0, "er": 0.0}
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
    """
    Plot training curves.

    Left panel  – total loss
    Right panel – wide, rb, er losses (the three main constraint terms)
    """
    keys_loss = ["wide", "rb", "er", "residual", "mean"]
    epochs = np.asarray(history["epoch"])

    # total
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].plot(epochs, history["train_total"], label="Train total")
    axes[0].plot(epochs, history["val_total"], label="Val total")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # wide, rb, er
    colors = {"wide": "C0", "rb": "C1", "er": "C2", "residual": "C3", "mean": "C4"}
    for key in keys_loss:
        axes[1].plot(epochs, history[f"val_{key}"], label=f"val_{key}", color=colors[key])
    axes[1].set_title("Validation loss components")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(LOG_DIR / "training_curves.png", dpi=300)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-c", type=int, default=32)
    parser.add_argument("--mean-weight", type=float, default=0.1)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=300)
    args = parser.parse_args()

    ensure_dirs()
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
    criterion = ResidualBandConstrainedLoss(mean_weight=args.mean_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience, factor=0.5
    )

    keys = ["total", "wide", "residual", "mean", "rb", "er"]
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
            f"Epoch {epoch:3d}: train={train['total']:.4f}, val={val['total']:.4f}, "
            f"wide={val['wide']:.4f}, rb={val['rb']:.4f}, er={val['er']:.4f}, "
            f"mean={val['mean']:.6f}, lr={lr:.2e}",
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
            }, DATA_DIR / "checkpoints" / "best_model.pth")
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
        }, DATA_DIR / "checkpoints" / "last_model.pth")

    np.save(LOG_DIR / "training_history.npy", history)
    plot_history(history)
    print(f"Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
