import sys
from pathlib import Path

import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from fold_geometry import (  # noqa: E402
    calibration_window,
    heldout_guard,
    plan_fold,
    regions_overlap,
    validate_fold_manifest,
)


def test_calibration_window_has_expected_bounds():
    assert calibration_window({"inline": 362, "crossline": 387}) == {
        "inline_min": 354,
        "inline_max": 370,
        "crossline_min": 371,
        "crossline_max": 403,
    }


def test_fold_manifest_excludes_heldout_guard_and_splits_planes():
    manifest = plan_fold("fold_well1")
    validate_fold_manifest(manifest)
    guard = heldout_guard({"inline": 244, "crossline": 336})
    for region in manifest["wide_sample_regions"]:
        assert not regions_overlap(region, guard)
        well_inline = region["well_inline"]
        assert region["train_inline_values"] == list(
            range(well_inline - 8, well_inline + 7)
        )
        assert region["val_inline_values"] == [well_inline + 7, well_inline + 8]


def test_manifest_rejects_heldout_as_calibration():
    manifest = plan_fold("fold_well1")
    manifest["calibration_wells"].append("well1")
    with pytest.raises(ValueError, match="held-out"):
        validate_fold_manifest(manifest)
