import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from context_masked_samples import (  # noqa: E402
    build_supervision_mask,
    make_context_pair,
    plan_centered_context,
)


def test_centered_context_contains_consecutive_coordinates_and_supervision():
    coordinates = np.arange(300, 1251)
    plan = plan_centered_context(coordinates, 371, 32, 256)

    assert len(plan["context_values"]) == 256
    assert np.all(np.diff(plan["context_values"]) == 1)
    assert plan["context_values"][plan["mask_start"]] == 371
    assert plan["context_values"][plan["mask_stop"] - 1] == 402


def test_mask_has_exactly_32_supervised_traces():
    mask = build_supervision_mask(256, 256, 112, 32)

    assert mask.shape == (256, 256)
    assert int(mask.sum()) == 256 * 32
    assert np.all(mask[:, 112:144] == 1)
    assert np.all(mask[:, :112] == 0)
    assert np.all(mask[:, 144:] == 0)


def test_context_pair_zeroes_target_outside_mask_and_closes_inside():
    rng = np.random.default_rng(7)
    wide = rng.normal(size=(256, 256)).astype(np.float32)
    mask = build_supervision_mask(256, 256, 80, 32)

    narrow, residual, target, scale = make_context_pair(wide, mask, 0.004)

    assert scale > 0
    assert np.count_nonzero(residual[mask == 0]) == 0
    assert np.count_nonzero(target[mask == 0]) == 0
    np.testing.assert_allclose(
        (narrow + residual)[mask == 1],
        (wide / scale)[mask == 1],
        atol=1e-6,
    )
