import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from config import CHECKPOINT_DIR, DATA_DIR, LOG_DIR, RANDOM_SEED
from dataset import NpyPatchDataset
from losses import BandwidthLoss
from model import UNetCBAM


def run_epoch(model, loader, criterion, device, optimizer=None, desc="train"):
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"total": 0.0, "mae": 0.0, "freq": 0.0}

    iterator = tqdm(loader, desc=desc)
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss, loss_dict = criterion(pred, y)
            if is_train:
                loss.backward()
                optimizer.step()
        for key in totals:
            totals[key] += loss_dict[key]
        iterator.set_postfix(loss=f"{loss_dict['total']:.4f}")

    return {key: val / max(1, len(loader)) for key, val in totals.items()}


def plot_history(history):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys = [("total", "Total Loss"), ("mae", "MAE"), ("freq", "Frequency")]
    epochs = np.arange(1, len(history["train_total"]) + 1)
    for ax, (key, title) in zip(axes, keys):
        ax.plot(epochs, history[f"train_{key}"], label="Train")
        ax.plot(epochs, history[f"val_{key}"], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(LOG_DIR / "training_curves.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-c", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = NpyPatchDataset(DATA_DIR / "train_inputs.npy", DATA_DIR / "train_labels.npy")
    val_ds = NpyPatchDataset(DATA_DIR / "val_inputs.npy", DATA_DIR / "val_labels.npy")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetCBAM(base_c=args.base_c).to(device)
    criterion = BandwidthLoss(lambda_mae=1.0, lambda_freq=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {k: [] for k in [
        "train_total", "train_mae", "train_freq",
        "val_total", "val_mae", "val_freq",
    ]}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, criterion, device, optimizer, f"Epoch {epoch} train")
        val = run_epoch(model, val_loader, criterion, device, None, f"Epoch {epoch} val")
        for key in ["total", "mae", "freq"]:
            history[f"train_{key}"].append(train[key])
            history[f"val_{key}"].append(val[key])
        print(f"Epoch {epoch}: train={train['total']:.4f}, val={val['total']:.4f}")

        if val["total"] < best_val:
            best_val = val["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val,
                "base_c": args.base_c,
            }, CHECKPOINT_DIR / "best_model.pth")

    np.save(LOG_DIR / "training_history.npy", history)
    plot_history(history)
    print(f"Best val loss: {best_val:.6f}")
    print(f"Saved best model to {CHECKPOINT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
