import json
import sys
from pathlib import Path

import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from leakage_guard import (  # noqa: E402
    authorize_heldout_reference,
    create_fold_lock,
)


def test_fold_lock_authorizes_only_heldout_sections(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"model")
    manifest = {
        "fold": "fold_well1",
        "heldout_well": "well1",
        "heldout_inline": 244,
        "heldout_crossline": 336,
        "calibration_wells": ["well2", "well3", "well4"],
        "uses_heldout_well_wide_target": False,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    lock_path = tmp_path / "fold_lock.json"
    create_fold_lock(
        checkpoint,
        manifest_path,
        lock_path,
        common_checkpoint_sha256="a" * 64,
    )
    metadata = authorize_heldout_reference(
        checkpoint,
        manifest_path,
        lock_path,
        "well1",
        "inline",
        244,
    )
    assert metadata["heldout_well"] == "well1"
    with pytest.raises(ValueError, match="section"):
        authorize_heldout_reference(
            checkpoint,
            manifest_path,
            lock_path,
            "well1",
            "inline",
            362,
        )


def test_lock_rejects_modified_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"model")
    manifest = {
        "fold": "fold_well1",
        "heldout_well": "well1",
        "heldout_inline": 244,
        "heldout_crossline": 336,
        "calibration_wells": ["well2", "well3", "well4"],
        "uses_heldout_well_wide_target": False,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    lock_path = tmp_path / "fold_lock.json"
    create_fold_lock(
        checkpoint,
        manifest_path,
        lock_path,
        common_checkpoint_sha256="a" * 64,
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        authorize_heldout_reference(
            checkpoint,
            manifest_path,
            lock_path,
            "well1",
            "crossline",
            336,
        )
