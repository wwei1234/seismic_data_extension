import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_model import project_frequency_band


PROJECTOR_KEYS = ("low_stop", "low_pass", "high_pass", "high_stop")


def project_batch_frequency_band(signal, projectors, dt):
    if projectors.ndim == 1:
        projectors = projectors.unsqueeze(0).expand(signal.shape[0], -1)
    projected = []
    for index in range(signal.shape[0]):
        values = {
            key: float(projectors[index, position].detach().cpu())
            for position, key in enumerate(PROJECTOR_KEYS)
        }
        projected.append(project_frequency_band(signal[index:index + 1], dt=dt, **values))
    return torch.cat(projected, dim=0)


def correlation_loss(prediction, target):
    pred = prediction - prediction.mean(dim=-2, keepdim=True)
    truth = target - target.mean(dim=-2, keepdim=True)
    numerator = (pred * truth).sum(dim=-2)
    denominator = torch.sqrt(
        pred.square().sum(dim=-2) * truth.square().sum(dim=-2) + 1e-8
    )
    correlation = numerator / denominator
    weights = truth.square().mean(dim=-2)
    return 1.0 - (correlation * weights).sum() / (weights.sum() + 1e-8)


def lateral_gradient_loss(prediction, target):
    pred_gradient = prediction[..., 1:] - prediction[..., :-1]
    target_gradient = target[..., 1:] - target[..., :-1]
    return F.smooth_l1_loss(pred_gradient, target_gradient, beta=0.05)


def low_frequency_leakage_fraction(prediction, projectors, dt):
    spectrum = torch.fft.rfft(prediction, dim=-2)
    frequencies = torch.fft.rfftfreq(
        prediction.shape[-2],
        d=dt,
        device=prediction.device,
    )
    fractions = []
    for index in range(prediction.shape[0]):
        low_stop = projectors[index, 0]
        low_energy = spectrum[index, :, frequencies < low_stop, :].abs().square().sum()
        total_energy = spectrum[index].abs().square().sum()
        fractions.append(low_energy / (total_energy + 1e-8))
    return torch.stack(fractions).mean()


def complex_stft_loss(prediction, target, windows=(32, 64, 128)):
    time_size = prediction.shape[-2]
    losses = []
    pred_traces = prediction.permute(0, 1, 3, 2).reshape(-1, time_size)
    target_traces = target.permute(0, 1, 3, 2).reshape(-1, time_size)
    for requested in windows:
        window_size = min(int(requested), time_size)
        if window_size < 8:
            continue
        hop = max(window_size // 4, 1)
        window = torch.hann_window(
            window_size,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        pred_stft = torch.stft(
            pred_traces,
            n_fft=window_size,
            hop_length=hop,
            win_length=window_size,
            window=window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target_traces,
            n_fft=window_size,
            hop_length=hop,
            win_length=window_size,
            window=window,
            return_complex=True,
        )
        scale = target_stft.abs().mean().detach() + 1e-6
        complex_error = (pred_stft - target_stft).abs().mean() / scale
        amplitude_error = F.l1_loss(
            torch.log1p(pred_stft.abs()),
            torch.log1p(target_stft.abs()),
        )
        losses.append(complex_error + 0.25 * amplitude_error)
    return torch.stack(losses).mean()


class DomainAwarePhaseLoss(nn.Module):
    def __init__(self, dt=0.004):
        super().__init__()
        self.dt = float(dt)
        self.weights = {
            "f3": {
                "residual": 0.25,
                "correlation": 0.35,
                "stft": 0.25,
                "wide": 0.0,
                "lateral": 0.10,
                "leakage": 0.05,
            },
            "synthetic": {
                "residual": 0.20,
                "correlation": 0.30,
                "stft": 0.25,
                "wide": 0.15,
                "lateral": 0.05,
                "leakage": 0.05,
            },
        }

    def forward(
        self,
        residual_prediction,
        residual_target,
        input_data,
        target_wide,
        domain,
        projector,
    ):
        if domain not in self.weights:
            raise ValueError(f"Unknown training domain: {domain}")
        projected_prediction = project_batch_frequency_band(
            residual_prediction,
            projector,
            self.dt,
        )
        projected_target = project_batch_frequency_band(
            residual_target,
            projector,
            self.dt,
        )
        residual = F.smooth_l1_loss(
            projected_prediction,
            projected_target,
            beta=0.05,
        )
        correlation = correlation_loss(projected_prediction, projected_target)
        stft = complex_stft_loss(projected_prediction, projected_target)
        lateral = lateral_gradient_loss(projected_prediction, projected_target)
        reconstructed = input_data + projected_prediction
        wide = F.smooth_l1_loss(reconstructed, target_wide, beta=0.05)
        leakage = low_frequency_leakage_fraction(
            residual_prediction,
            projector,
            self.dt,
        )
        weights = self.weights[domain]
        total = (
            weights["residual"] * residual
            + weights["correlation"] * correlation
            + weights["stft"] * stft
            + weights["wide"] * wide
            + weights["lateral"] * lateral
            + weights["leakage"] * leakage
        )
        parts = {
            "total": float(total.detach().cpu()),
            "residual": float(residual.detach().cpu()),
            "correlation": float(correlation.detach().cpu()),
            "stft": float(stft.detach().cpu()),
            "wide": float(wide.detach().cpu()),
            "lateral": float(lateral.detach().cpu()),
            "leakage": float(leakage.detach().cpu()),
        }
        return total, parts
