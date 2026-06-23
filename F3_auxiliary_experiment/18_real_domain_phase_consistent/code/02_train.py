"""Train experiment 18 with guarded real F3 and synthetic residual patches."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from config import (
    CHECKPOINT_DIR,
    DT,
    LOG_DIR,
    RANDOM_SEED,
    ensure_dirs,
)
from hybrid_dataset import HybridResidualDataset
from phase_loss import PhaseConsistentLoss
from phase_model import PhaseConsistentResidualModel
from training import train_step, validation_step


LOSS_KEYS = ("total", "residual", "stft", "correlation", "lateral", "leakage")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    totals = {key: 0.0 for key in LOSS_KEYS}
    real_count = 0
    sample_count = 0
    for inputs, targets, source_ids in loader:
        if optimizer is None:
            parts = validation_step(model, criterion, inputs, targets, device)
        else:
            parts = train_step(
                model,
                criterion,
                optimizer,
                inputs,
                targets,
                device,
                scaler,
            )
        batch_size = inputs.shape[0]
        for key in LOSS_KEYS:
            totals[key] += parts[key] * batch_size
        real_count += int(source_ids.sum())
        sample_count += batch_size
    result = {
        key: value / max(sample_count, 1)
        for key, value in totals.items()
    }
    result["real_fraction"] = real_count / max(sample_count, 1)
    return result


def plot_history(history):
    epochs = np.asarray(history["epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(epochs, history["train_total"], label="Train")
    axes[0].plot(epochs, history["val_total"], label="Validation")
    axes[0].set_title("Experiment 18 total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    for key in LOSS_KEYS[1:]:
        axes[1].plot(epochs, history[f"val_{key}"], label=key)
    axes[1].set_title("Validation components")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    output = LOG_DIR / "training_curves.png"
    fig.savefig(output, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--base-c", type=int, default=32)
    parser.add_argument("--real-probability", type=float, default=0.7)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    args = parser.parse_args()

    ensure_dirs()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    train_dataset = HybridResidualDataset(
        "train",
        real_probability=args.real_probability,
        augment=True,
    )
    val_dataset = HybridResidualDataset(
        "val",
        real_probability=0.85,
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseConsistentResidualModel(base_c=args.base_c, dt=DT).to(device)
    criterion = PhaseConsistentLoss(dt=DT)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=8,
        factor=0.5,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda"
        else None
    )

    history = {
        f"{split}_{key}": []
        for split in ("train", "val")
        for key in (*LOSS_KEYS, "real_fraction")
    }
    history.update({"epoch": [], "lr": [], "device": str(device)})
    best_val = float("inf")
    no_improve = 0
    print(
        f"Device={device}, train={len(train_dataset)}, val={len(val_dataset)}, "
        f"real_probability={args.real_probability}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        val = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val["total"])
        lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["lr"].append(lr)
        for key in (*LOSS_KEYS, "real_fraction"):
            history[f"train_{key}"].append(train[key])
            history[f"val_{key}"].append(val[key])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val["total"],
            "base_c": args.base_c,
            "target_type": "projected_high_frequency_residual",
            "normalization": "p99_abs_narrow_unclipped",
            "real_probability": args.real_probability,
            "projector": {
                "low_stop": 22.0,
                "low_pass": 28.0,
                "high_pass": 85.0,
                "high_stop": 100.0,
            },
        }
        torch.save(checkpoint, CHECKPOINT_DIR / "last_model.pth")
        if val["total"] < best_val:
            best_val = val["total"]
            no_improve = 0
            torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pth")
        else:
            no_improve += 1

        np.save(LOG_DIR / "training_history.npy", history)
        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train={train['total']:.6f}, val={val['total']:.6f}, "
            f"res={val['residual']:.6f}, stft={val['stft']:.6f}, "
            f"corr={val['correlation']:.6f}, lat={val['lateral']:.6f}, "
            f"leak={val['leakage']:.8f}, lr={lr:.2e}",
            flush=True,
        )
        if no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    plot_history(history)
    print(f"Best val loss: {best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()
