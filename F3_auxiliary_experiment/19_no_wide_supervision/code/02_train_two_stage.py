"""Two-stage training without using F3 wide-band targets."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from config import (
    CHECKPOINT_DIR,
    DT,
    FINAL_PROJECTOR,
    LOG_DIR,
    PRETRAIN_PROJECTOR,
    RANDOM_SEED,
    SELF_SUPERVISED_DIR,
    SYNTHETIC_DATA_DIR,
    ensure_dirs,
)
from leakage_guard import (
    assert_training_paths_are_safe,
    create_model_lock,
)
from phase_loss import PhaseConsistentLoss
from phase_model import PhaseConsistentResidualModel, project_frequency_band


LOSS_KEYS = ("total", "residual", "stft", "correlation", "lateral", "leakage")


class ResidualDataset(Dataset):
    def __init__(self, input_path, label_path, augment=False):
        assert_training_paths_are_safe([input_path, label_path])
        self.inputs = np.load(input_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        self.augment = bool(augment)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        x = np.asarray(self.inputs[index], dtype=np.float32)
        label = np.asarray(self.labels[index], dtype=np.float32)
        if self.augment and np.random.rand() < 0.5:
            x = np.flip(x, axis=1)
            label = np.flip(label, axis=1)
        return (
            torch.from_numpy(x.copy()).unsqueeze(0),
            torch.from_numpy(label.copy()).unsqueeze(0),
        )


def run_epoch(model, loader, criterion, optimizer, device, projector, scaler):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in LOSS_KEYS}
    count = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = project_frequency_band(labels, dt=DT, **projector)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                _, prediction = model.forward_with_residual(inputs)
            loss, parts = criterion(prediction.float(), labels.float())
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch = inputs.shape[0]
        count += batch
        for key in LOSS_KEYS:
            totals[key] += parts[key] * batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_stage(
    name,
    model,
    train_paths,
    val_paths,
    projector,
    epochs,
    learning_rate,
    batch_size,
    device,
    save_best,
):
    model.projector = dict(projector)
    train_loader = DataLoader(
        ResidualDataset(*train_paths, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ResidualDataset(*val_paths, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    criterion = PhaseConsistentLoss(dt=DT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8, factor=0.5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        train = run_epoch(
            model, train_loader, criterion, optimizer, device, projector, scaler
        )
        val = run_epoch(
            model, val_loader, criterion, None, device, projector, scaler
        )
        scheduler.step(val["total"])
        row = {
            "epoch": epoch,
            "stage": name,
            "train": train,
            "val": val,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if save_best and val["total"] < best:
            best = val["total"]
            torch.save({
                "epoch": epoch,
                "stage": name,
                "model_state_dict": model.state_dict(),
                "base_c": 32,
                "val_loss": best,
                "projector": dict(projector),
                "target_type": "synthetic_high_frequency_residual",
                "uses_f3_wide_target": False,
            }, CHECKPOINT_DIR / "best_model.pth")
        print(
            f"{name} {epoch:3d}/{epochs}: train={train['total']:.6f}, "
            f"val={val['total']:.6f}, corr={val['correlation']:.6f}, "
            f"stft={val['stft']:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
    np.save(LOG_DIR / f"{name}_history.npy", np.asarray(history, dtype=object))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--synthetic-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    ensure_dirs()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseConsistentResidualModel(
        base_c=32,
        dt=DT,
        projector=PRETRAIN_PROJECTOR,
    ).to(device)

    pretrain_paths = (
        (
            SELF_SUPERVISED_DIR / "train_inputs.npy",
            SELF_SUPERVISED_DIR / "train_labels.npy",
        ),
        (
            SELF_SUPERVISED_DIR / "val_inputs.npy",
            SELF_SUPERVISED_DIR / "val_labels.npy",
        ),
    )
    synthetic_paths = (
        (
            SYNTHETIC_DATA_DIR / "train_inputs.npy",
            SYNTHETIC_DATA_DIR / "train_labels.npy",
        ),
        (
            SYNTHETIC_DATA_DIR / "val_inputs.npy",
            SYNTHETIC_DATA_DIR / "val_labels.npy",
        ),
    )
    assert_training_paths_are_safe([
        *pretrain_paths[0], *pretrain_paths[1],
        *synthetic_paths[0], *synthetic_paths[1],
    ])

    train_stage(
        "narrow_pretrain",
        model,
        pretrain_paths[0],
        pretrain_paths[1],
        PRETRAIN_PROJECTOR,
        args.pretrain_epochs,
        5e-4,
        args.batch_size,
        device,
        save_best=False,
    )
    best = train_stage(
        "synthetic_finetune",
        model,
        synthetic_paths[0],
        synthetic_paths[1],
        FINAL_PROJECTOR,
        args.synthetic_epochs,
        3e-4,
        args.batch_size,
        device,
        save_best=True,
    )
    lock = create_model_lock(
        CHECKPOINT_DIR / "best_model.pth",
        CHECKPOINT_DIR / "model_lock.json",
        {
            "experiment": 19,
            "uses_f3_wide_target": False,
            "selection_metric": "synthetic_validation_loss",
            "best_synthetic_val_loss": best,
            "pretrain_epochs": args.pretrain_epochs,
            "synthetic_epochs": args.synthetic_epochs,
            "final_projector": FINAL_PROJECTOR,
        },
    )
    print(f"MODEL_LOCKED sha256={lock['sha256']}", flush=True)


if __name__ == "__main__":
    main()
