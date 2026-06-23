import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DT
from phase_loss import (
    DomainAwarePhaseLoss,
    low_frequency_leakage_fraction,
    project_batch_frequency_band,
    supervised_traces,
)


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def correlation_score(prediction, target):
    pred = prediction - prediction.mean(dim=-2, keepdim=True)
    truth = target - target.mean(dim=-2, keepdim=True)
    numerator = (pred * truth).sum(dim=-2)
    denominator = torch.sqrt(
        pred.square().sum(dim=-2) * truth.square().sum(dim=-2) + 1e-8
    )
    weights = truth.square().mean(dim=-2)
    score = (numerator / denominator * weights).sum() / (weights.sum() + 1e-8)
    return float(score.detach().cpu())


def phase_score(prediction, target):
    pred_spectrum = torch.fft.rfft(prediction, dim=-2)
    target_spectrum = torch.fft.rfft(target, dim=-2)
    cross = pred_spectrum * target_spectrum.conj()
    weight = target_spectrum.abs()
    cosine = cross.real / (cross.abs() + 1e-8)
    return float(((cosine * weight).sum() / (weight.sum() + 1e-8)).cpu())


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=shuffle,
    )


def next_batch(loaders, iterators, domain):
    try:
        return next(iterators[domain])
    except (KeyError, StopIteration):
        iterators[domain] = iter(loaders[domain])
        return next(iterators[domain])


def validate(model, loader, domain, device, maximum_batches=16):
    criterion = DomainAwarePhaseLoss(dt=DT)
    totals = {"loss": 0.0, "correlation": 0.0, "phase": 0.0, "leakage": 0.0}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= maximum_batches:
                break
            batch = move_batch(batch, device)
            raw = model.net(batch["input"])
            loss, _ = criterion(
                raw.float(),
                batch["label"].float(),
                batch["input"].float(),
                batch["target"].float(),
                domain,
                batch["projector"],
                supervision_mask=batch.get("mask"),
            )
            prediction = project_batch_frequency_band(
                raw.float(), batch["projector"], DT
            )
            target = project_batch_frequency_band(
                batch["label"].float(), batch["projector"], DT
            )
            if "mask" in batch:
                prediction = supervised_traces(prediction, batch["mask"])
                target = supervised_traces(target, batch["mask"])
            n = batch["input"].shape[0]
            totals["loss"] += float(loss.cpu()) * n
            totals["correlation"] += correlation_score(prediction, target) * n
            totals["phase"] += phase_score(prediction, target) * n
            totals["leakage"] += float(
                low_frequency_leakage_fraction(
                    prediction, batch["projector"], DT
                ).cpu()
            ) * n
            count += n
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_batches(
    model,
    loaders,
    domains,
    optimizer,
    scaler,
    device,
    steps,
):
    criterion = DomainAwarePhaseLoss(dt=DT)
    iterators = {}
    model.train()
    totals = {}
    for step in range(steps):
        domain = domains[step % len(domains)]
        batch = move_batch(next_batch(loaders, iterators, domain), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            raw = model.net(batch["input"])
        loss, parts = criterion(
            raw.float(),
            batch["label"].float(),
            batch["input"].float(),
            batch["target"].float(),
            domain,
            batch["projector"],
            supervision_mask=batch.get("mask"),
        )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        totals.setdefault(domain, []).append(parts["total"])
    return {
        domain: float(np.mean(values))
        for domain, values in totals.items()
    }
