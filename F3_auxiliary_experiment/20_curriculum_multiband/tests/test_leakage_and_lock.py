import sys
from pathlib import Path

import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from leakage_guard import (  # noqa: E402
    assert_training_paths_are_safe,
    create_model_lock,
    verify_model_lock,
)


def test_training_paths_reject_f3_wide_and_experiment18():
    assert_training_paths_are_safe([
        Path("20_curriculum_multiband/data/F3/train_clean_narrow.npy"),
        Path("shared_data/well_reflectivities.npy"),
    ])
    with pytest.raises(ValueError):
        assert_training_paths_are_safe([
            Path("18_real_domain_phase_consistent/data/真实样本/train_labels.npy")
        ])
    with pytest.raises(ValueError):
        assert_training_paths_are_safe([Path("wide_reference.npy")])


def test_model_lock_detects_checkpoint_change(tmp_path):
    checkpoint = tmp_path / "best_model.pth"
    lock = tmp_path / "model_lock.json"
    checkpoint.write_bytes(b"locked")
    create_model_lock(
        checkpoint,
        lock,
        {"experiment": 20, "uses_f3_wide_target": False},
    )
    verify_model_lock(checkpoint, lock)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_model_lock(checkpoint, lock)
