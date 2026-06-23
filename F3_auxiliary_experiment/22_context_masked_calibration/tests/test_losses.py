import sys
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from phase_loss import DomainAwarePhaseLoss  # noqa: E402
from phase_model import PhaseConsistentResidualModel  # noqa: E402


def test_model_accepts_local_and_common_patch_widths():
    model = PhaseConsistentResidualModel(base_c=4)
    for width in (32, 256):
        output = model(torch.randn(1, 1, 256, width))
        assert output.shape == (1, 1, 256, width)


def test_perfect_local_residual_has_lower_loss_than_phase_inversion():
    criterion = DomainAwarePhaseLoss()
    target = torch.randn(1, 1, 64, 16) * 0.05
    inputs = torch.randn(1, 1, 64, 16) * 0.1
    projector = torch.tensor([[22.0, 28.0, 85.0, 100.0]])
    good, _ = criterion(
        target,
        target,
        inputs,
        inputs + target,
        "local_wide",
        projector,
    )
    bad, _ = criterion(
        -target,
        target,
        inputs,
        inputs + target,
        "local_wide",
        projector,
    )
    assert good < bad


def test_local_loss_ignores_prediction_errors_outside_supervision_mask():
    criterion = DomainAwarePhaseLoss()
    criterion.weights["local_wide"]["leakage"] = 0.0
    target = torch.randn(1, 1, 64, 32) * 0.05
    inputs = torch.randn(1, 1, 64, 32) * 0.1
    mask = torch.zeros_like(target)
    mask[..., 8:16] = 1
    projector = torch.tensor([[22.0, 28.0, 85.0, 100.0]])
    outside_error = target.clone()
    outside_error[mask == 0] += 10.0

    good, _ = criterion(
        target,
        target,
        inputs,
        inputs + target,
        "local_wide",
        projector,
        supervision_mask=mask,
    )
    outside, _ = criterion(
        outside_error,
        target,
        inputs,
        inputs + target,
        "local_wide",
        projector,
        supervision_mask=mask,
    )

    torch.testing.assert_close(good, outside)
