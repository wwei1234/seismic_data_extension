import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from real_f3_samples import (  # noqa: E402
    assign_plane_splits,
    make_real_pair,
    patch_is_guarded,
    patch_has_sufficient_coverage,
    plan_axis_candidates,
)


def test_patch_rejected_when_it_intersects_guarded_plane():
    assert patch_is_guarded(240, 496, [244], margin=8)
    assert not patch_is_guarded(260, 516, [244], margin=8)


def test_real_pair_closure_uses_unclipped_narrow_scale():
    rng = np.random.default_rng(3)
    wide = rng.normal(size=(256, 32)).astype(np.float32)

    narrow, residual, scale = make_real_pair(wide, dt=0.004)

    assert scale > 0.0
    assert np.max(np.abs(narrow)) > 1.0
    assert np.allclose(narrow + residual, wide / scale, atol=2e-5)


def test_candidate_planner_excludes_source_and_orthogonal_guards():
    axis_values = np.arange(100, 751)
    orthogonal_values = np.arange(300, 1251)

    candidates = plan_axis_candidates(
        source_axis="inline",
        axis_values=axis_values,
        orthogonal_values=orthogonal_values,
        source_guards=[244, 362, 442, 722],
        source_margin=8,
        orthogonal_guards=[336, 387, 848, 1007],
        orthogonal_margin=16,
        time_size=462,
        patch_size=256,
        spatial_size=256,
        stride=128,
    )

    assert candidates
    for candidate in candidates:
        assert abs(candidate["section_number"] - 244) > 8
        start_value = candidate["spatial_start_value"]
        stop_value = candidate["spatial_stop_value"]
        assert not patch_is_guarded(
            start_value,
            stop_value,
            [336, 387, 848, 1007],
            margin=16,
        )


def test_plane_split_never_places_one_section_in_both_sets():
    candidates = [
        {"source_axis": "inline", "section_number": section, "time_start": time}
        for section in range(100, 120)
        for time in (0, 128, 206)
    ]

    train, val = assign_plane_splits(candidates, val_fraction=0.2, seed=42)

    train_planes = {(row["source_axis"], row["section_number"]) for row in train}
    val_planes = {(row["source_axis"], row["section_number"]) for row in val}
    assert train_planes
    assert val_planes
    assert train_planes.isdisjoint(val_planes)


def test_patch_quality_rejects_missing_grid_but_keeps_dense_data():
    dense = np.ones((256, 256), dtype=np.float32)
    sparse = np.zeros((256, 256), dtype=np.float32)
    sparse[:, :2] = 1.0

    assert patch_has_sufficient_coverage(dense)
    assert not patch_has_sufficient_coverage(sparse)
