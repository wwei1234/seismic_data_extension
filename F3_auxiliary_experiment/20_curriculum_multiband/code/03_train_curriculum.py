"""Train experiment 20 with gated F3/synthetic curriculum learning."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CURRICULUM_STAGES,
    DT,
    F3_PATCH_DIR,
    FINAL_PROJECTOR,
    LOG_DIR,
    RANDOM_SEED,
    SYNTHETIC_DIR,
    TOTAL_EPOCHS,
    ensure_dirs,
)
from curriculum import GatedCheckpointSelector, domain_cycle  # noqa: E402
from datasets import F3MaskedDataset, SyntheticResidualDataset  # noqa: E402
from leakage_guard import create_model_lock  # noqa: E402
from phase_loss import (  # noqa: E402
    DomainAwarePhaseLoss,
    low_frequency_leakage_fraction,
    project_batch_frequency_band,
)
from phase_model import PhaseConsistentResidualModel  # noqa: E402


@dataclass
class TrainingResult:
    history: list
    best_epoch: int | None
    gate_passed: bool
    checkpoint_path: Path | None


def stage_for_epoch(epoch):
    for stage in CURRICULUM_STAGES:
        if stage["start"] <= epoch <= stage["end"]:
            return stage
    return CURRICULUM_STAGES[-1]


def _next_batch(loaders, iterators, domain):
    try:
        return next(iterators[domain])
    except (StopIteration, KeyError):
        iterators[domain] = iter(loaders[domain])
        return next(iterators[domain])


def _move_batch(batch, device):
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def _correlation_score(prediction, target):
    pred = prediction - prediction.mean(dim=-2, keepdim=True)
    truth = target - target.mean(dim=-2, keepdim=True)
    numerator = (pred * truth).sum(dim=-2)
    denominator = torch.sqrt(
        pred.square().sum(dim=-2) * truth.square().sum(dim=-2) + 1e-8
    )
    weights = truth.square().mean(dim=-2)
    score = (numerator / denominator * weights).sum() / (weights.sum() + 1e-8)
    return float(score.detach().cpu())


def _phase_score(prediction, target):
    pred_spectrum = torch.fft.rfft(prediction, dim=-2)
    target_spectrum = torch.fft.rfft(target, dim=-2)
    cross = pred_spectrum * target_spectrum.conj()
    weight = target_spectrum.abs()
    cosine = cross.real / (cross.abs() + 1e-8)
    score = (cosine * weight).sum() / (weight.sum() + 1e-8)
    return float(score.detach().cpu())


def _run_validation(model, loader, criterion, device, domain, maximum_batches):
    model.eval()
    totals = {
        "loss": 0.0,
        "correlation": 0.0,
        "phase": 0.0,
        "leakage": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= maximum_batches:
                break
            batch = _move_batch(batch, device)
            raw_residual = model.net(batch["input"])
            loss, parts = criterion(
                raw_residual,
                batch["label"],
                batch["input"],
                batch["target"],
                domain=domain,
                projector=batch["projector"],
            )
            prediction = project_batch_frequency_band(
                raw_residual,
                batch["projector"],
                DT,
            )
            target = project_batch_frequency_band(
                batch["label"],
                batch["projector"],
                DT,
            )
            batch_size = batch["input"].shape[0]
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["correlation"] += _correlation_score(prediction, target) * batch_size
            totals["phase"] += _phase_score(prediction, target) * batch_size
            emitted_leakage = low_frequency_leakage_fraction(
                prediction,
                batch["projector"],
                DT,
            )
            totals["leakage"] += float(emitted_leakage.detach().cpu()) * batch_size
            count += batch_size
    divisor = max(count, 1)
    result = {key: value / divisor for key, value in totals.items()}
    if domain == "synthetic":
        result["residual_correlation"] = result["correlation"]
    return result


def _make_loaders(
    f3_train,
    f3_val,
    synthetic_train,
    synthetic_val,
    batch_size,
):
    return {
        "f3": DataLoader(
            f3_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        ),
        "synthetic": DataLoader(
            synthetic_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        ),
        "f3_val": DataLoader(
            f3_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "synthetic_val": DataLoader(
            synthetic_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }


def run_training(
    epochs,
    f3_train,
    f3_val,
    synthetic_train,
    synthetic_val,
    output_dir,
    device,
    batch_size=4,
    steps_per_epoch=64,
    validation_batches=16,
    base_c=32,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    model = PhaseConsistentResidualModel(
        base_c=base_c,
        dt=DT,
        projector=FINAL_PROJECTOR,
    ).to(device)
    criterion = DomainAwarePhaseLoss(dt=DT)
    loaders = _make_loaders(
        f3_train,
        f3_val,
        synthetic_train,
        synthetic_val,
        batch_size,
    )
    selector = GatedCheckpointSelector()
    history = []
    optimizer = None
    scheduler = None
    current_stage = None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_path = output_dir / "best_model.pth"

    for epoch in range(1, int(epochs) + 1):
        f3_train.set_epoch(epoch)
        synthetic_train.set_epoch(epoch)
        stage = stage_for_epoch(epoch)
        if stage["name"] != current_stage:
            current_stage = stage["name"]
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(stage["lr"]),
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=8,
                factor=0.5,
                min_lr=1e-6,
            )

        model.train()
        iterators = {}
        cycle = domain_cycle(epoch)
        accumulators = {
            "f3": {"loss": 0.0, "correlation": 0.0, "count": 0},
            "synthetic": {"loss": 0.0, "correlation": 0.0, "count": 0},
        }
        for step in range(int(steps_per_epoch)):
            domain = cycle[step % len(cycle)]
            batch = _move_batch(_next_batch(loaders, iterators, domain), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                raw_residual = model.net(batch["input"])
            loss, parts = criterion(
                raw_residual.float(),
                batch["label"].float(),
                batch["input"].float(),
                batch["target"].float(),
                domain=domain,
                projector=batch["projector"],
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size_actual = batch["input"].shape[0]
            accumulators[domain]["loss"] += parts["total"] * batch_size_actual
            accumulators[domain]["correlation"] += (
                1.0 - parts["correlation"]
            ) * batch_size_actual
            accumulators[domain]["count"] += batch_size_actual

        f3_validation = _run_validation(
            model,
            loaders["f3_val"],
            criterion,
            device,
            "f3",
            validation_batches,
        )
        synthetic_validation = _run_validation(
            model,
            loaders["synthetic_val"],
            criterion,
            device,
            "synthetic",
            validation_batches,
        )
        scheduler.step(f3_validation["loss"] + synthetic_validation["loss"])

        train_metrics = {}
        for domain, values in accumulators.items():
            divisor = max(values["count"], 1)
            train_metrics[domain] = {
                "loss": values["loss"] / divisor,
                "correlation": values["correlation"] / divisor,
                "count": values["count"],
            }
        row = {
            "epoch": epoch,
            "stage": current_stage,
            "f3_train": train_metrics["f3"],
            "synthetic_train": train_metrics["synthetic"],
            "f3_val": f3_validation,
            "synthetic_val": synthetic_validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "uses_f3_wide_target": False,
        }
        history.append(row)

        if selector.consider(epoch, f3_validation, synthetic_validation):
            torch.save({
                "epoch": epoch,
                "stage": current_stage,
                "model_state_dict": model.state_dict(),
                "base_c": base_c,
                "projector": FINAL_PROJECTOR,
                "f3_validation": f3_validation,
                "synthetic_validation": synthetic_validation,
                "uses_f3_wide_target": False,
            }, best_path)

        print(
            f"epoch {epoch:3d}/{epochs} {current_stage} "
            f"f3_corr={f3_validation['correlation']:.4f} "
            f"f3_phase={f3_validation['phase']:.4f} "
            f"f3_leak={f3_validation['leakage']:.4f} "
            f"syn_corr={synthetic_validation['residual_correlation']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    gate_passed = selector.best_epoch is not None
    if gate_passed:
        create_model_lock(
            best_path,
            output_dir / "model_lock.json",
            {
                "experiment": 20,
                "uses_f3_wide_target": False,
                "selection_metric": "gated_f3_then_synthetic_correlation",
                "best_epoch": selector.best_epoch,
                "best_f3_validation": selector.best_f3,
                "best_synthetic_validation": selector.best_synthetic,
                "final_projector": FINAL_PROJECTOR,
                "total_epochs": int(epochs),
            },
        )
    else:
        failure = {
            "experiment": 20,
            "gate_passed": False,
            "uses_f3_wide_target": False,
            "last_f3_validation": history[-1]["f3_val"],
            "last_synthetic_validation": history[-1]["synthetic_val"],
        }
        (output_dir / "training_failed_gate.json").write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
    return TrainingResult(
        history=history,
        best_epoch=selector.best_epoch,
        gate_passed=gate_passed,
        checkpoint_path=best_path if gate_passed else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_training(
        epochs=args.epochs,
        f3_train=F3MaskedDataset(F3_PATCH_DIR, "train", seed=RANDOM_SEED),
        f3_val=F3MaskedDataset(
            F3_PATCH_DIR,
            "val",
            seed=RANDOM_SEED + 1,
            augment=False,
        ),
        synthetic_train=SyntheticResidualDataset(
            SYNTHETIC_DIR,
            "train",
            seed=RANDOM_SEED,
        ),
        synthetic_val=SyntheticResidualDataset(
            SYNTHETIC_DIR,
            "val",
            seed=RANDOM_SEED + 1,
            augment=False,
        ),
        output_dir=CHECKPOINT_DIR,
        device=device,
        batch_size=args.batch_size,
        steps_per_epoch=1 if args.smoke else args.steps_per_epoch,
        validation_batches=1 if args.smoke else args.validation_batches,
        base_c=4 if args.smoke else 32,
    )
    print(
        f"TRAINING_COMPLETE gate_passed={result.gate_passed} "
        f"best_epoch={result.best_epoch}",
        flush=True,
    )


if __name__ == "__main__":
    main()
