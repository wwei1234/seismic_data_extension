import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from local_calibration_samples import (  # noqa: E402
    make_local_pair,
    pad_inside_window,
    plan_local_patches,
)


def test_pair_uses_narrow_p99_and_closes():
    rng = np.random.default_rng(4)
    wide = rng.normal(size=(256, 32)).astype(np.float32)
    narrow, residual, scale = make_local_pair(wide, dt=0.004)
    assert scale > 0
    assert np.max(np.abs(narrow + residual - wide / scale)) < 1e-6


def test_patch_coordinates_stay_inside_window_and_split():
    window = {
        "well_inline": 362,
        "inline_min": 354,
        "inline_max": 370,
        "crossline_min": 371,
        "crossline_max": 403,
    }
    train = plan_local_patches(window, time_size=462, split="train")
    val = plan_local_patches(window, time_size=462, split="val")
    assert train and val
    val_planes = {369, 370}
    assert all(not set(row["inline_values"]) & val_planes for row in train)
    assert all(set(row["inline_values"]) <= val_planes for row in val)


def test_padding_uses_only_values_inside_window():
    patch = np.arange(17, dtype=np.float32)[None, :].repeat(4, axis=0)
    padded, left, right = pad_inside_window(patch, 32)
    assert padded.shape == (4, 32)
    assert left + right == 15
    assert padded.min() >= patch.min()
    assert padded.max() <= patch.max()
