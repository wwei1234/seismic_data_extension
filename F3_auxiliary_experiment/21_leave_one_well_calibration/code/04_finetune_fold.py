"""Fine-tune one experiment 21 fold with identical settings."""

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
    FOLDS,
    LOCAL_DATA_DIR,
    PROJECTOR,
    RANDOM_SEED,
    WORKSPACE_ROOT,
    ensure_dirs,
)
from datasets import (  # noqa: E402
    F3MaskedDataset,
    LocalCalibrationDataset,
    SyntheticResidualDataset,
    domain_cycle,
)
from leakage_guard import create_fold_lock, sha256_file  # noqa: E402
from phase_model import PhaseConsistentResidualModel  # noqa: E402
from training_utils import make_loader, train_batches, validate  # noqa: E402


def set_encoder_frozen(model, frozen):
    children = list(model.net.children())
    for child in children[:2]:
        for parameter in child.parameters():
            parameter.requires_grad = not frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--validation-batches", type=int, default=16)
    args = parser.parse_args()
    ensure_dirs()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common_root = CHECKPOINT_DIR / "pretrain"
    common_checkpoint = torch.load(
        common_root / "common_model.pth", map_location="cpu"
    )
    common_lock = json.loads(
        (common_root / "common_lock.json").read_text(encoding="utf-8")
    )
    base_c = int(common_checkpoint["base_c"])
    model = PhaseConsistentResidualModel(
        base_c=base_c,
        projector=common_checkpoint["projector"],
    ).to(device)
    model.load_state_dict(common_checkpoint["model_state_dict"])
    local_root = LOCAL_DATA_DIR / args.fold
    f3_root = (
        WORKSPACE_ROOT / "20_curriculum_multiband" / "data" / "F3多频带自监督"
    )
    synthetic_root = (
        WORKSPACE_ROOT / "20_curriculum_multiband" / "data" / "测井合成样本"
    )
    train_datasets = {
        "local_wide": LocalCalibrationDataset(
            local_root, "train", RANDOM_SEED
        ),
        "synthetic": SyntheticResidualDataset(
            synthetic_root, "train", RANDOM_SEED
        ),
        "f3": F3MaskedDataset(f3_root, "train", RANDOM_SEED),
    }
    val_datasets = {
        "local_wide": LocalCalibrationDataset(
            local_root, "val", RANDOM_SEED + 1, augment=False
        ),
        "f3": F3MaskedDataset(
            f3_root, "val", RANDOM_SEED + 1, augment=False
        ),
    }
    loaders = {
        key: make_loader(value, BATCH_SIZE, True)
        for key, value in train_datasets.items()
    }
    val_loaders = {
        key: make_loader(value, BATCH_SIZE, False)
        for key, value in val_datasets.items()
    }
    common_f3 = float(common_lock["f3_validation"]["correlation"])
    output = CHECKPOINT_DIR / args.fold
    best_path = output / "best_model.pth"
    last_path = output / "last_model.pth"
    best_score = None
    best_row = None
    history = []
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    optimizer = None
    stage = None
    for epoch in range(1, args.epochs + 1):
        for dataset in train_datasets.values():
            dataset.set_epoch(epoch)
        current_stage = "A" if epoch <= 60 else "B"
        if current_stage != stage:
            stage = current_stage
            lr = 1e-4 if stage == "A" else 3e-5
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=1e-4
            )
        set_encoder_frozen(model, epoch <= 10)
        domains = domain_cycle(RANDOM_SEED + epoch, size=10)
        train = train_batches(
            model,
            loaders,
            domains,
            optimizer,
            scaler,
            device,
            args.steps_per_epoch,
        )
        local_val = validate(
            model,
            val_loaders["local_wide"],
            "local_wide",
            device,
            args.validation_batches,
        )
        f3_val = validate(
            model, val_loaders["f3"], "f3", device, args.validation_batches
        )
        row = {
            "epoch": epoch,
            "stage": stage,
            "train": train,
            "local_val": local_val,
            "f3_val": f3_val,
        }
        history.append(row)
        gate = (
            local_val["correlation"] > 0
            and local_val["phase"] > 0
            and f3_val["correlation"] >= 0.95 * common_f3
            and local_val["leakage"] <= 0.01
        )
        score = (local_val["correlation"], local_val["phase"])
        if gate and (best_score is None or score > best_score):
            best_score = score
            best_row = row
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "base_c": base_c,
                "projector": PROJECTOR,
                "fold": args.fold,
                "local_validation": local_val,
                "f3_validation": f3_val,
                "common_checkpoint_sha256": common_lock[
                    "checkpoint_sha256"
                ],
                "uses_heldout_well_wide_target": False,
            }, best_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "base_c": base_c,
            "projector": PROJECTOR,
        }, last_path)
        print(
            f"{args.fold} epoch {epoch:3d}/{args.epochs} "
            f"local_corr={local_val['correlation']:.4f} "
            f"phase={local_val['phase']:.4f} "
            f"f3_corr={f3_val['correlation']:.4f} gate={gate}",
            flush=True,
        )
    (output / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if best_row is None:
        raise RuntimeError(f"{args.fold} did not pass the fold gate.")
    manifest_path = local_root / "fold_manifest.json"
    create_fold_lock(
        best_path,
        manifest_path,
        output / "fold_lock.json",
        common_checkpoint_sha256=common_lock["checkpoint_sha256"],
    )
    print(
        f"LOCKED {args.fold} epoch={best_row['epoch']} "
        f"sha={sha256_file(best_path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
