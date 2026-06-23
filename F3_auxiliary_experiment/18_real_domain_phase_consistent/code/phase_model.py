import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from model import UNetCBAM


def cosine_frequency_mask(
    nt,
    dt,
    low_stop=22.0,
    low_pass=28.0,
    high_pass=85.0,
    high_stop=100.0,
    device=None,
    dtype=None,
):
    frequencies = torch.fft.rfftfreq(nt, d=dt, device=device)
    mask = torch.zeros_like(frequencies, dtype=dtype or torch.float32)

    low_ramp = (frequencies > low_stop) & (frequencies < low_pass)
    if low_pass > low_stop:
        phase = (frequencies[low_ramp] - low_stop) / (low_pass - low_stop)
        values = 0.5 - 0.5 * torch.cos(torch.pi * phase)
        mask[low_ramp] = values.to(mask.dtype)

    passband = (frequencies >= low_pass) & (frequencies <= high_pass)
    mask[passband] = 1.0

    high_ramp = (frequencies > high_pass) & (frequencies < high_stop)
    if high_stop > high_pass:
        phase = (frequencies[high_ramp] - high_pass) / (high_stop - high_pass)
        values = 0.5 + 0.5 * torch.cos(torch.pi * phase)
        mask[high_ramp] = values.to(mask.dtype)
    return mask


def project_frequency_band(
    signal,
    dt=0.004,
    low_stop=22.0,
    low_pass=28.0,
    high_pass=85.0,
    high_stop=100.0,
):
    spectrum = torch.fft.rfft(signal, dim=-2)
    mask = cosine_frequency_mask(
        signal.shape[-2],
        dt,
        low_stop=low_stop,
        low_pass=low_pass,
        high_pass=high_pass,
        high_stop=high_stop,
        device=signal.device,
        dtype=signal.dtype,
    )
    return torch.fft.irfft(
        spectrum * mask.view(1, 1, -1, 1),
        n=signal.shape[-2],
        dim=-2,
    )


def project_numpy_frequency_band(
    signal,
    dt=0.004,
    low_stop=22.0,
    low_pass=28.0,
    high_pass=85.0,
    high_stop=100.0,
):
    signal = np.asarray(signal, dtype=np.float32)
    frequencies = np.fft.rfftfreq(signal.shape[0], d=dt)
    mask = np.zeros_like(frequencies)
    low_ramp = (frequencies > low_stop) & (frequencies < low_pass)
    phase = (frequencies[low_ramp] - low_stop) / (low_pass - low_stop)
    mask[low_ramp] = 0.5 - 0.5 * np.cos(np.pi * phase)
    mask[(frequencies >= low_pass) & (frequencies <= high_pass)] = 1.0
    high_ramp = (frequencies > high_pass) & (frequencies < high_stop)
    phase = (frequencies[high_ramp] - high_pass) / (high_stop - high_pass)
    mask[high_ramp] = 0.5 + 0.5 * np.cos(np.pi * phase)
    spectrum = np.fft.rfft(signal, axis=0)
    return np.fft.irfft(
        spectrum * mask[:, None],
        n=signal.shape[0],
        axis=0,
    ).astype(np.float32)


class PhaseConsistentResidualModel(nn.Module):
    def __init__(self, base_c=32, dt=0.004):
        super().__init__()
        self.net = UNetCBAM(base_c=base_c)
        self.dt = float(dt)

    def forward_with_residual(self, narrow):
        raw_residual = self.net(narrow)
        residual = project_frequency_band(raw_residual.float(), dt=self.dt)
        return narrow + residual, residual

    def forward(self, narrow):
        wide, _ = self.forward_with_residual(narrow)
        return wide
