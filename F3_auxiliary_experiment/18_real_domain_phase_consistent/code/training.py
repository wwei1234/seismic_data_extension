from contextlib import nullcontext

import torch


def train_step(
    model,
    criterion,
    optimizer,
    inputs,
    targets,
    device,
    scaler=None,
):
    model.train()
    inputs = inputs.to(device)
    targets = targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None and device.type == "cuda"
    amp_context = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else nullcontext()
    )
    with amp_context:
        _, residual_prediction = model.forward_with_residual(inputs)
    loss, parts = criterion(residual_prediction.float(), targets.float())
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return parts


def validation_step(model, criterion, inputs, targets, device):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, residual_prediction = model.forward_with_residual(inputs)
        _, parts = criterion(residual_prediction.float(), targets.float())
    return parts
