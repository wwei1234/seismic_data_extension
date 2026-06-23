import sys
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from phase_loss import DomainAwarePhaseLoss, project_batch_frequency_band  # noqa: E402


def test_batch_projection_supports_different_sample_bands():
    torch.manual_seed(2)
    signal = torch.randn(2, 1, 128, 8)
    projectors = torch.tensor([
        [10.0, 12.0, 20.0, 22.0],
        [25.0, 28.0, 60.0, 70.0],
    ])
    projected = project_batch_frequency_band(signal, projectors, dt=0.004)
    assert projected.shape == signal.shape
    assert not torch.allclose(projected[0], projected[1])


def test_matching_prediction_has_lower_f3_loss():
    torch.manual_seed(3)
    target = torch.randn(2, 1, 128, 8)
    inputs = torch.randn_like(target)
    projectors = torch.tensor([
        [10.0, 12.0, 30.0, 35.0],
        [10.0, 12.0, 30.0, 35.0],
    ])
    criterion = DomainAwarePhaseLoss(dt=0.004)
    matching, _ = criterion(
        target,
        target,
        inputs,
        inputs + target,
        domain="f3",
        projector=projectors,
    )
    wrong, _ = criterion(
        -target,
        target,
        inputs,
        inputs + target,
        domain="f3",
        projector=projectors,
    )
    assert matching < wrong


def test_leakage_metric_is_bounded_energy_fraction():
    torch.manual_seed(9)
    prediction = torch.randn(2, 1, 128, 8)
    target = torch.randn_like(prediction) * 0.1
    inputs = torch.randn_like(prediction) * 0.1
    projectors = torch.tensor([
        [10.0, 12.0, 30.0, 35.0],
        [10.0, 12.0, 30.0, 35.0],
    ])
    criterion = DomainAwarePhaseLoss(dt=0.004)
    _, parts = criterion(
        prediction,
        target,
        inputs,
        inputs + target,
        domain="f3",
        projector=projectors,
    )
    assert 0.0 <= parts["leakage"] <= 1.0
