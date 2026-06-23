import sys
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from phase_loss import (  # noqa: E402
    PhaseConsistentLoss,
    lateral_gradient_loss,
)


DT = 0.004


def sinusoid_tensor(frequency_hz, phase=0.0, nt=256, nx=16):
    time = torch.arange(nt, dtype=torch.float32) * DT
    trace = torch.sin(2.0 * torch.pi * frequency_hz * time + phase)
    return trace[None, None, :, None].repeat(1, 1, 1, nx)


def coherent_dipping_event(nt=256, nx=32):
    time = torch.arange(nt, dtype=torch.float32) * DT
    traces = []
    for ix in range(nx):
        phase = ix * 0.05
        traces.append(torch.sin(2.0 * torch.pi * 45.0 * time + phase))
    return torch.stack(traces, dim=-1)[None, None]


def test_loss_prefers_correct_phase_over_reversed_phase():
    target = sinusoid_tensor(50.0)
    criterion = PhaseConsistentLoss(dt=DT, stft_windows=(64,))

    aligned_loss, _ = criterion(target, target)
    reversed_loss, _ = criterion(-target, target)

    assert aligned_loss < reversed_loss
    assert reversed_loss > 0.5


def test_lateral_gradient_penalizes_trace_scrambling():
    target = coherent_dipping_event()
    scrambled = target[..., torch.randperm(target.shape[-1])]

    aligned = lateral_gradient_loss(target, target)
    disrupted = lateral_gradient_loss(scrambled, target)

    assert aligned < 1e-7
    assert disrupted > aligned + 1e-3
