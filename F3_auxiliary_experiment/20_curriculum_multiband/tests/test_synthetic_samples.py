import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from synthetic_samples import (  # noqa: E402
    build_test_sample,
    grouped_section_split,
)


def test_synthetic_label_uses_clean_narrow_not_noisy_input():
    sample = build_test_sample(noise_level=0.03, seed=5)
    assert np.allclose(
        sample.clean_narrow_norm + sample.label_norm,
        sample.wide_norm,
        atol=2e-5,
    )
    assert not np.allclose(
        sample.input_norm + sample.label_norm,
        sample.wide_norm,
    )
    assert sample.scale_source == "p99_abs_clean_narrow"


def test_group_split_keeps_section_out_of_both_splits():
    metadata = [
        {"section_id": f"section_{section}", "patch": patch}
        for section in range(10)
        for patch in range(3)
    ]
    train, val = grouped_section_split(metadata, val_fraction=0.2, seed=42)
    train_sections = {row["section_id"] for row in train}
    val_sections = {row["section_id"] for row in val}
    assert train_sections.isdisjoint(val_sections)
