import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from model import UNetCBAM


class WidebandModel(nn.Module):
    def __init__(self, base_c=32):
        super().__init__()
        self.net = UNetCBAM(base_c=base_c)

    def forward(self, x):
        return self.net(x)


class WidebandCompositeLoss(nn.Module):
    def __init__(
        self,
        dt=0.004,
        spectrum_weight=0.5,
        phase_weight=0.2,
        gradient_weight=0.2,
        low_frequency_weight=0.5,
    ):
        super().__init__()
        self.dt = float(dt)
        self.spectrum_weight = float(spectrum_weight)
        self.phase_weight = float(phase_weight)
        self.gradient_weight = float(gradient_weight)
        self.low_frequency_weight = float(low_frequency_weight)

    def forward(self, prediction, target):
        waveform = F.l1_loss(prediction, target)

        nt = prediction.shape[-2]
        freqs = torch.fft.rfftfreq(nt, d=self.dt, device=prediction.device)
        pred_spec = torch.fft.rfft(prediction, dim=-2)
        target_spec = torch.fft.rfft(target, dim=-2)

        high_mask = (freqs >= 25.0) & (freqs <= 80.0)
        pred_amp = pred_spec.abs()[:, :, high_mask, :]
        target_amp = target_spec.abs()[:, :, high_mask, :]
        pred_amp = pred_amp / (pred_amp.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        target_amp = target_amp / (target_amp.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        spectrum = F.l1_loss(pred_amp, target_amp)

        phase_delta = (
            torch.angle(pred_spec[:, :, high_mask, :])
            - torch.angle(target_spec[:, :, high_mask, :])
        )
        phase = (1.0 - torch.cos(phase_delta)).mean()

        pred_gradient = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_gradient = target[..., 1:, :] - target[..., :-1, :]
        gradient = F.l1_loss(pred_gradient, target_gradient)

        low_mask = ((freqs >= 3.0) & (freqs <= 35.0)).to(prediction.dtype)
        low_shape = (1, 1, low_mask.numel(), 1)
        pred_low = torch.fft.irfft(
            pred_spec * low_mask.view(low_shape), n=nt, dim=-2
        )
        target_low = torch.fft.irfft(
            target_spec * low_mask.view(low_shape), n=nt, dim=-2
        )
        low_frequency = F.l1_loss(pred_low, target_low)

        total = (
            waveform
            + self.spectrum_weight * spectrum
            + self.phase_weight * phase
            + self.gradient_weight * gradient
            + self.low_frequency_weight * low_frequency
        )
        parts = {
            "total": float(total.detach().cpu()),
            "waveform": float(waveform.detach().cpu()),
            "spectrum": float(spectrum.detach().cpu()),
            "phase": float(phase.detach().cpu()),
            "gradient": float(gradient.detach().cpu()),
            "low_frequency": float(low_frequency.detach().cpu()),
        }
        return total, parts
