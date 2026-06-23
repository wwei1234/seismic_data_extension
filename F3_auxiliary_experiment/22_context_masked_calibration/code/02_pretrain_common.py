"""Train the common experiment 21 initializer without F3 wide targets."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    COMMON_DATA_DIR,
    PROJECTOR,
    RANDOM_SEED,
    WORKSPACE_ROOT,
    ensure_dirs,
)
from datasets import F3MaskedDataset, SyntheticResidualDataset  # noqa: E402
from leakage_guard import create_common_lock  # noqa: E402
from phase_model import PhaseConsistentResidualModel  # noqa: E402
from training_utils import make_loader, train_batches, validate  # noqa: E402


def domains_for_epoch(epoch):
    if epoch <= 60:
        return ["f3"]
    if epoch <= 180:
        return ["f3", "f3", "synthetic"]
    return ["f3", "synthetic"]


def learning_rate_for_epoch(epoch):
    if epoch <= 60:
        return 5e-4
    if epoch <= 180:
        return 3e-4
    return 1e-4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--base-c", type=int, default=32)
    args = parser.parse_args()
    ensure_dirs()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f3_root = (
        WORKSPACE_ROOT / "20_curriculum_multiband" / "data" / "F3多频带自监督"
    )
    synthetic_root = (
        WORKSPACE_ROOT / "20_curriculum_multiband" / "data" / "测井合成样本"
    )
    datasets = {
        "f3": F3MaskedDataset(f3_root, "train", RANDOM_SEED),
        "synthetic": SyntheticResidualDataset(
            synthetic_root, "train", RANDOM_SEED
        ),
    }
    val_datasets = {
        "f3": F3MaskedDataset(
            f3_root, "val", RANDOM_SEED + 1, augment=False
        ),
        "synthetic": SyntheticResidualDataset(
            synthetic_root, "val", RANDOM_SEED + 1, augment=False
        ),
    }
    loaders = {
        key: make_loader(value, BATCH_SIZE, True)
        for key, value in datasets.items()
    }
    val_loaders = {
        key: make_loader(value, BATCH_SIZE, False)
        for key, value in val_datasets.items()
    }
    model = PhaseConsistentResidualModel(
        base_c=args.base_c,
        projector=PROJECTOR,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    optimizer = None
    current_lr = None
    history = []
    best_score = None
    best_metadata = None
    output = CHECKPOINT_DIR / "pretrain"
    checkpoint = output / "common_model.pth"
    for epoch in range(1, args.epochs + 1):
        for dataset in datasets.values():
            dataset.set_epoch(epoch)
        lr = learning_rate_for_epoch(epoch)
        if lr != current_lr:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=1e-4
            )
            current_lr = lr
        train = train_batches(
            model,
            loaders,
            domains_for_epoch(epoch),
            optimizer,
            scaler,
            device,
            args.steps_per_epoch,
        )
        f3_val = validate(
            model, val_loaders["f3"], "f3", device, args.validation_batches
        )
        synthetic_val = validate(
            model,
            val_loaders["synthetic"],
            "synthetic",
            device,
            args.validation_batches,
        )
        row = {
            "epoch": epoch,
            "train": train,
            "f3_val": f3_val,
            "synthetic_val": synthetic_val,
            "uses_f3_wide_target": False,
        }
        history.append(row)
        score = (f3_val["correlation"], f3_val["phase"])
        if best_score is None or score > best_score:
            best_score = score
            best_metadata = row
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "base_c": args.base_c,
                "projector": PROJECTOR,
                "f3_validation": f3_val,
                "synthetic_validation": synthetic_val,
                "uses_f3_wide_target": False,
            }, checkpoint)
        print(
            f"epoch {epoch:3d}/{args.epochs} "
            f"f3_corr={f3_val['correlation']:.4f} "
            f"f3_phase={f3_val['phase']:.4f} "
            f"syn_corr={synthetic_val['correlation']:.4f}",
            flush=True,
        )
    (output / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    create_common_lock(
        checkpoint,
        COMMON_DATA_DIR / "common_data_manifest.json",
        output / "common_lock.json",
        {
            "experiment": 21,
            "stage": "common_pretraining",
            "best_epoch": best_metadata["epoch"],
            "f3_validation": best_metadata["f3_val"],
            "synthetic_validation": best_metadata["synthetic_val"],
        },
    )


if __name__ == "__main__":
    main()
