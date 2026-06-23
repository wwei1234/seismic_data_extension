import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from datasets import LocalCalibrationDataset, domain_cycle  # noqa: E402


def test_local_dataset_returns_clean_residual_pair(tmp_path):
    root = tmp_path
    inputs = np.zeros((2, 256, 32), dtype=np.float32)
    labels = np.ones_like(inputs) * 0.1
    np.save(root / "train_inputs.npy", inputs)
    np.save(root / "train_labels.npy", labels)
    np.save(root / "train_metadata.npy", np.asarray([
        {"well": "well2"},
        {"well": "well3"},
    ], dtype=object))
    dataset = LocalCalibrationDataset(root, "train", augment=False)
    row = dataset[0]
    assert row["domain"] == "local_wide"
    assert row["input"].shape == (1, 256, 32)
    assert np.isclose(float(row["target"].mean()), 0.1)


def test_domain_cycle_has_expected_counts():
    cycle = domain_cycle(seed=4, size=1000)
    assert abs(cycle.count("local_wide") - 600) <= 1
    assert abs(cycle.count("synthetic") - 200) <= 1
    assert abs(cycle.count("f3") - 200) <= 1
