import sys
from pathlib import Path

import numpy as np
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from config import SYNTHETIC_DATA_DIR  # noqa: E402
from leakage_guard import (  # noqa: E402
    assert_training_paths_are_safe,
    create_model_lock,
    verify_model_lock,
)
from self_supervised_samples import make_narrow_self_supervised_pair  # noqa: E402


def test_training_paths_reject_wide_reference_and_experiment18():
    safe = [
        SYNTHETIC_DATA_DIR / "train_inputs.npy",
        Path("19_no_wide_supervision/data/窄频自监督/train_labels.npy"),
    ]
    assert_training_paths_are_safe(safe)

    with pytest.raises(ValueError):
        assert_training_paths_are_safe([
            Path("18_real_domain_phase_consistent/data/真实样本/train_labels.npy")
        ])
    with pytest.raises(ValueError):
        assert_training_paths_are_safe([Path("wide_reference.npy")])


def test_self_supervised_target_contains_only_known_narrow_band():
    dt = 0.004
    time = np.arange(256, dtype=np.float32) * dt
    known_narrow = (
        np.sin(2 * np.pi * 12 * time)
        + 0.5 * np.sin(2 * np.pi * 28 * time)
    )[:, None]

    extra_low, known_residual, scale = make_narrow_self_supervised_pair(
        known_narrow,
        dt=dt,
    )

    assert scale > 0
    assert np.allclose(extra_low + known_residual, known_narrow / scale, atol=2e-5)


def test_model_lock_detects_checkpoint_change(tmp_path):
    checkpoint = tmp_path / "best_model.pth"
    lock = tmp_path / "model_lock.json"
    checkpoint.write_bytes(b"locked checkpoint")
    create_model_lock(checkpoint, lock, {"experiment": 19})
    verify_model_lock(checkpoint, lock)

    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError):
        verify_model_lock(checkpoint, lock)
