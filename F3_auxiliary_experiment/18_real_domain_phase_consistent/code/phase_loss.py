import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_bandpass(signal, dt, low_hz, high_hz):
    spectrum = torch.fft.rfft(signal, dim=-2)
    frequencies = torch.fft.rfftfreq(signal.shape[-2], d=dt, device=signal.device)
    mask = ((frequencies >= low_hz) & (frequencies <= high_hz)).to(signal.dtype)
    return torch.fft.irfft(
        spectrum * mask.view(1, 1, -1, 1),
        n=signal.shape[-2],
        dim=-2,
    )


def tracewise_correlation_loss(prediction, target, dt=0.004):
    prediction = temporal_bandpass(prediction, dt, 25.0, 80.0)
    target = temporal_bandpass(target, dt, 25.0, 80.0)
    prediction = prediction - prediction.mean(dim=-2, keepdim=True)
    target = target - target.mean(dim=-2, keepdim=True)
    numerator = (prediction * target).sum(dim=-2)
    denominator = torch.sqrt(
        prediction.square().sum(dim=-2) * target.square().sum(dim=-2) + 1e-8
    )
    correlation = numerator / denominator
    weights = target.square().mean(dim=-2)
    return 1.0 - (correlation * weights).sum() / (weights.sum() + 1e-8)


def lateral_gradient_loss(prediction, target):
    pred_gradient = prediction[..., 1:] - prediction[..., :-1]
    target_gradient = target[..., 1:] - target[..., :-1]
    return F.l1_loss(pred_gradient, target_gradient)


def multi_resolution_stft_loss(
    prediction,
    target,
    dt=0.004,
    windows=(64, 128, 256),
    trace_stride=4,
):
    pred_traces = prediction[..., ::trace_stride].permute(0, 1, 3, 2).reshape(
        -1, prediction.shape[-2]
    )
    target_traces = target[..., ::trace_stride].permute(0, 1, 3, 2).reshape(
        -1, target.shape[-2]
    )
    losses = []
    for window_size in windows:
        window = torch.hann_window(
            window_size,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        hop = max(window_size // 4, 1)
        pred_stft = torch.stft(
            pred_traces,
            n_fft=window_size,
            hop_length=hop,
            win_length=window_size,
            window=window,
            return_complex=True,
            center=True,
        )
        target_stft = torch.stft(
            target_traces,
            n_fft=window_size,
            hop_length=hop,
            win_length=window_size,
            window=window,
            return_complex=True,
            center=True,
        )
        frequencies = torch.fft.rfftfreq(window_size, d=dt, device=prediction.device)
        mask = (frequencies >= 25.0) & (frequencies <= 80.0)
        pred_band = pred_stft[:, mask]
        target_band = target_stft[:, mask]
        scale = target_band.abs().mean(dim=(-2, -1), keepdim=True) + 1e-5
        complex_loss = ((pred_band - target_band).abs() / scale).mean()
        log_amp_loss = F.l1_loss(
            torch.log1p(pred_band.abs()),
            torch.log1p(target_band.abs()),
        )
        losses.append(complex_loss + 0.25 * log_amp_loss)
    return torch.stack(losses).mean()


def low_frequency_leakage_loss(prediction, dt=0.004):
    low = temporal_bandpass(prediction, dt, 0.0, 22.0)
    return low.abs().mean()


class PhaseConsistentLoss(nn.Module):
    def __init__(
        self,
        dt=0.004,
        stft_windows=(64, 128, 256),
        residual_weight=1.0,
        stft_weight=0.5,
        correlation_weight=0.3,
        lateral_weight=0.2,
        leakage_weight=0.1,
    ):
        super().__init__()
        self.dt = float(dt)
        self.stft_windows = tuple(int(value) for value in stft_windows)
        self.weights = {
            "residual": float(residual_weight),
            "stft": float(stft_weight),
            "correlation": float(correlation_weight),
            "lateral": float(lateral_weight),
            "leakage": float(leakage_weight),
        }

    def forward(self, residual_prediction, residual_target):
        residual = F.smooth_l1_loss(residual_prediction, residual_target, beta=0.05)
        stft = multi_resolution_stft_loss(
            residual_prediction,
            residual_target,
            dt=self.dt,
            windows=self.stft_windows,
        )
        correlation = tracewise_correlation_loss(
            residual_prediction,
            residual_target,
            dt=self.dt,
        )
        lateral = lateral_gradient_loss(residual_prediction, residual_target)
        leakage = low_frequency_leakage_loss(residual_prediction, dt=self.dt)
        total = (
            self.weights["residual"] * residual
            + self.weights["stft"] * stft
            + self.weights["correlation"] * correlation
            + self.weights["lateral"] * lateral
            + self.weights["leakage"] * leakage
        )
        parts = {
            "total": float(total.detach().cpu()),
            "residual": float(residual.detach().cpu()),
            "stft": float(stft.detach().cpu()),
            "correlation": float(correlation.detach().cpu()),
            "lateral": float(lateral.detach().cpu()),
            "leakage": float(leakage.detach().cpu()),
        }
        return total, parts
