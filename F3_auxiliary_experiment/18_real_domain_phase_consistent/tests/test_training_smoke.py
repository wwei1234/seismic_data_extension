import sys
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from phase_loss import PhaseConsistentLoss  # noqa: E402
from phase_model import PhaseConsistentResidualModel  # noqa: E402
from training import train_step  # noqa: E402


def test_one_training_step_is_finite_and_updates_parameters():
    torch.manual_seed(9)
    model = PhaseConsistentResidualModel(base_c=4)
    criterion = PhaseConsistentLoss(stft_windows=(64,))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(1, 1, 64, 32)
    targets = torch.randn(1, 1, 64, 32) * 0.1
    before = next(model.parameters()).detach().clone()

    parts = train_step(
        model,
        criterion,
        optimizer,
        inputs,
        targets,
        device=torch.device("cpu"),
        scaler=None,
    )

    after = next(model.parameters()).detach()
    assert torch.isfinite(torch.tensor(parts["total"]))
    assert not torch.allclose(before, after)
