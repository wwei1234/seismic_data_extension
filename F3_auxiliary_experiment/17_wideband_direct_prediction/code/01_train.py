"""Train a direct low-pass-to-wideband prediction model."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from config import DATA_DIR, DT, RANDOM_SEED
from wideband_training import WidebandCompositeLoss, WidebandModel


class WidebandPatchDataset(Dataset):
    def __init__(self, split, augment=False):
        self.inputs = np.load(DATA_DIR / f"{split}_inputs.npy", mmap_mode="r")
        self.labels = np.load(DATA_DIR / f"{split}_labels.npy", mmap_mode="r")
        self.augment = bool(augment)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = np.asarray(self.inputs[idx], dtype=np.float32)
        y = np.asarray(self.labels[idx], dtype=np.float32)
        if self.augment and np.random.rand() < 0.5:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)
        return (
            torch.from_numpy(x.copy()).unsqueeze(0),
            torch.from_numpy(y.copy()).unsqueeze(0),
        )


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    keys = ("total", "waveform", "spectrum", "phase", "gradient", "low_frequency")
    totals = {key: 0.0 for key in keys}
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            prediction = model(x)
            loss, parts = criterion(prediction, y)
            if is_train:
                loss.backward()
                optimizer.step()
        for key in keys:
            totals[key] += parts[key]
    return {key: value / max(len(loader), 1) for key, value in totals.items()}


def plot_history(history, output_path):
    epochs = np.asarray(history["epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(epochs, history["train_total"], label="Train")
    axes[0].plot(epochs, history["val_total"], label="Validation")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    for key in ("waveform", "spectrum", "phase", "gradient", "low_frequency"):
        axes[1].plot(epochs, history[f"val_{key}"], label=key)
    axes[1].set_title("Validation loss components")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-c", type=int, default=32)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DATA_DIR / "模型检查点"),
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = DATA_DIR.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    train_loader = DataLoader(
        WidebandPatchDataset("train", augment=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WidebandPatchDataset("val", augment=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WidebandModel(base_c=args.base_c).to(device)
    criterion = WidebandCompositeLoss(dt=DT)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    keys = ("total", "waveform", "spectrum", "phase", "gradient", "low_frequency")
    history = {f"{split}_{key}": [] for split in ("train", "val") for key in keys}
    history.update({"epoch": [], "lr": [], "device": str(device)})
    best_val = float("inf")

    print(f"Device: {device}", flush=True)
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

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val["total"],
            "base_c": args.base_c,
            "target_type": "wide_band",
            "normalization": "per_section_p99_abs_clean_narrow",
        }
        torch.save(checkpoint, checkpoint_dir / "last_model.pth")
        if val["total"] < best_val:
            best_val = val["total"]
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")

        np.save(log_dir / "training_history.npy", history)
        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train={train['total']:.6f}, val={val['total']:.6f}, "
            f"wave={val['waveform']:.6f}, spec={val['spectrum']:.6f}, "
            f"phase={val['phase']:.6f}, grad={val['gradient']:.6f}, "
            f"low={val['low_frequency']:.6f}, lr={lr:.2e}",
            flush=True,
        )

    plot_history(history, log_dir / "training_curves.png")
    print(f"Best val loss: {best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()
