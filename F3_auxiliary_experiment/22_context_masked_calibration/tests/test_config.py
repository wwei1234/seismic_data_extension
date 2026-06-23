import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from config import (  # noqa: E402
    CALIBRATION_CROSSLINE_RADIUS,
    CALIBRATION_INLINE_RADIUS,
    FOLDS,
    HELDOUT_CROSSLINE_GUARD,
    HELDOUT_INLINE_GUARD,
    LOCAL_SPATIAL_PATCH,
    LOCAL_SUPERVISION_WIDTH,
    LOCAL_TIME_PATCH,
    WELLS,
)


def test_four_folds_leave_each_well_out_once():
    assert set(FOLDS) == {
        "fold_well1",
        "fold_well2",
        "fold_well3",
        "fold_well4",
    }
    assert {fold["heldout_well"] for fold in FOLDS.values()} == set(WELLS)
    for fold in FOLDS.values():
        assert len(fold["calibration_wells"]) == 3
        assert fold["heldout_well"] not in fold["calibration_wells"]


def test_fixed_window_and_patch_configuration():
    assert (CALIBRATION_INLINE_RADIUS, CALIBRATION_CROSSLINE_RADIUS) == (8, 16)
    assert (HELDOUT_INLINE_GUARD, HELDOUT_CROSSLINE_GUARD) == (16, 32)
    assert (LOCAL_TIME_PATCH, LOCAL_SPATIAL_PATCH) == (256, 256)
    assert LOCAL_SUPERVISION_WIDTH == 32
