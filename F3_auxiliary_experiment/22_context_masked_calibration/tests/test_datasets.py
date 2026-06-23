import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from datasets import LocalCalibrationDataset, domain_cycle  # noqa: E402


def test_local_dataset_returns_clean_residual_pair(tmp_path):
    root = tmp_path
    inputs = np.zeros((2, 256, 256), dtype=np.float32)
    labels = np.ones_like(inputs) * 0.1
    masks = np.zeros_like(inputs)
    masks[:, :, 112:144] = 1
    np.save(root / "train_inputs.npy", inputs)
    np.save(root / "train_labels.npy", labels)
    np.save(root / "train_masks.npy", masks)
    np.save(root / "train_metadata.npy", np.asarray([
        {"well": "well2"},
        {"well": "well3"},
    ], dtype=object))
    dataset = LocalCalibrationDataset(root, "train", augment=False)
    row = dataset[0]
    assert row["domain"] == "local_wide"
    assert row["input"].shape == (1, 256, 256)
    assert int(row["mask"].sum()) == 256 * 32
    assert np.isclose(float(row["target"].mean()), 0.1)


def test_local_dataset_flips_mask_with_input_and_label(tmp_path):
    inputs = np.zeros((1, 4, 6), dtype=np.float32)
    labels = np.zeros_like(inputs)
    masks = np.zeros_like(inputs)
    inputs[0, :, 1] = 1
    labels[0, :, 2] = 2
    masks[0, :, 1:3] = 1
    np.save(tmp_path / "train_inputs.npy", inputs)
    np.save(tmp_path / "train_labels.npy", labels)
    np.save(tmp_path / "train_masks.npy", masks)
    np.save(
        tmp_path / "train_metadata.npy",
        np.asarray([{"well": "well1"}], dtype=object),
    )
    dataset = LocalCalibrationDataset(tmp_path, "train", seed=1, augment=True)
    dataset.set_epoch(1)

    sample = dataset[0]

    input_columns = np.where(sample["input"].numpy()[0].max(axis=0) > 0)[0]
    mask_columns = np.where(sample["mask"].numpy()[0].max(axis=0) > 0)[0]
    assert int(sample["mask"].sum()) == 8
    assert input_columns[0] in mask_columns


def test_domain_cycle_has_expected_counts():
    cycle = domain_cycle(seed=4, size=1000)
    assert abs(cycle.count("local_wide") - 600) <= 1
    assert abs(cycle.count("synthetic") - 200) <= 1
    assert abs(cycle.count("f3") - 200) <= 1
