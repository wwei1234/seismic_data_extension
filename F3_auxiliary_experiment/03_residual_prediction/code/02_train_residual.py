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

from common import CHECKPOINT_DIR, DATA_DIR, LOG_DIR, ensure_dirs
from losses import BandwidthLoss
from model import UNetCBAM


class ResidualPatchDataset(Dataset):
    def __init__(self, input_path, label_path, residual_path, augment=False):
        self.inputs = np.load(input_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        self.residuals = np.load(residual_path, mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = np.asarray(self.inputs[idx], dtype=np.float32)
        y = np.asarray(self.labels[idx], dtype=np.float32)
        r = np.asarray(self.residuals[idx], dtype=np.float32)
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


class ResidualWideLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide_loss = BandwidthLoss(lambda_mae=1.0, lambda_freq=0.5, lambda_phase=0.3)
        self.residual_loss = nn.L1Loss()

    def forward(self, x, residual_pred, wide_target, residual_target):
        wide_pred = x + residual_pred
        loss_wide, loss_dict = self.wide_loss(wide_pred, wide_target)
        loss_residual = self.residual_loss(residual_pred, residual_target)
        total = loss_wide + 0.2 * loss_residual
        loss_dict["total"] = float(total.detach().cpu())
        loss_dict["residual"] = float(loss_residual.detach().cpu())
        return total, loss_dict


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"total": 0.0, "mae": 0.0, "freq": 0.0, "phase": 0.0, "residual": 0.0}
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
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    keys = ["total", "mae", "freq", "phase", "residual"]
    epochs = np.asarray(
        history.get("epoch", np.arange(1, len(history["train_total"]) + 1))
    )
    for ax, key in zip(axes, keys):
        ax.plot(epochs, history[f"train_{key}"], label="Train")
        ax.plot(epochs, history[f"val_{key}"], label="Val")
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(LOG_DIR / "residual_training_curves.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-c", type=int, default=32)
    parser.add_argument("--early-stop-patience", type=int, default=300)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    ensure_dirs()
    torch.manual_seed(42)
    np.random.seed(42)

    train_ds = ResidualPatchDataset(
        DATA_DIR / "train_inputs.npy",
        DATA_DIR / "train_labels.npy",
        DATA_DIR / "train_residuals.npy",
        augment=True,
    )
    val_ds = ResidualPatchDataset(
        DATA_DIR / "val_inputs.npy",
        DATA_DIR / "val_labels.npy",
        DATA_DIR / "val_residuals.npy",
        augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetCBAM(base_c=args.base_c).to(device)
    criterion = ResidualWideLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience, factor=0.5
    )

    history = {f"{split}_{key}": [] for split in ("train", "val")
               for key in ("total", "mae", "freq", "phase", "residual")}
    history["epoch"] = []
    history["lr"] = []
    best_val = float("inf")
    no_improve = 0
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val = float(checkpoint.get("val_loss", best_val))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history_path = LOG_DIR / "residual_training_history.npy"
        if history_path.exists():
            saved_history = np.load(history_path, allow_pickle=True).item()
            saved_epochs = saved_history.get("epoch")
            if saved_epochs is not None and len(saved_epochs) == start_epoch - 1:
                history = saved_history
            else:
                print(
                    "Existing history does not match the resume epoch; "
                    "recording only the resumed segment.",
                    flush=True,
                )
        print(
            f"Resuming from {args.resume}: start_epoch={start_epoch}, "
            f"best_val={best_val:.6f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train = run_epoch(model, train_loader, criterion, device, optimizer)
        val = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val["total"])
        lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        for key in ("total", "mae", "freq", "phase", "residual"):
            history[f"train_{key}"].append(train[key])
            history[f"val_{key}"].append(val[key])
        history["lr"].append(lr)
        print(
            f"Epoch {epoch}: train={train['total']:.4f}, val={val['total']:.4f}, lr={lr:.2e}",
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
            }, CHECKPOINT_DIR / "best_residual_model.pth")
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
        }, CHECKPOINT_DIR / "last_residual_model.pth")

    np.save(LOG_DIR / "residual_training_history.npy", history)
    plot_history(history)
    print(f"Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
