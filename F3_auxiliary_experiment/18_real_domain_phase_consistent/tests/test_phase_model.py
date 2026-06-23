import sys
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from phase_model import (  # noqa: E402
    PhaseConsistentResidualModel,
    project_frequency_band,
)


DT = 0.004


def sinusoid(frequency_hz, nt=256):
    time = torch.arange(nt, dtype=torch.float32) * DT
    return torch.sin(2.0 * torch.pi * frequency_hz * time)[None, None, :, None]


def band_energy(signal, low_hz, high_hz):
    spectrum = torch.fft.rfft(signal, dim=-2)
    frequencies = torch.fft.rfftfreq(signal.shape[-2], d=DT)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    return spectrum[:, :, mask, :].abs().square().sum()


def test_frequency_projector_removes_low_frequency_and_keeps_passband():
    signal = sinusoid(10.0) + sinusoid(50.0)

    projected = project_frequency_band(signal, dt=DT)

    assert band_energy(projected, 8.0, 12.0) < band_energy(signal, 8.0, 12.0) * 1e-4
    assert band_energy(projected, 48.0, 52.0) > band_energy(signal, 48.0, 52.0) * 0.90


def test_phase_consistent_model_preserves_input_low_band():
    torch.manual_seed(7)
    model = PhaseConsistentResidualModel(base_c=4, dt=DT)
    input_patch = torch.randn(1, 1, 256, 32)

    wide, residual = model.forward_with_residual(input_patch)

    low_mask = torch.fft.rfftfreq(input_patch.shape[-2], d=DT) <= 20.0
    input_low = torch.fft.rfft(input_patch, dim=-2)[:, :, low_mask, :]
    wide_low = torch.fft.rfft(wide, dim=-2)[:, :, low_mask, :]
    assert torch.allclose(wide_low, input_low, atol=2e-5, rtol=1e-5)
    assert band_energy(residual, 0.0, 20.0) < band_energy(residual, 28.0, 80.0) * 1e-5
