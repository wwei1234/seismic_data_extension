import inspect
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from datasets import F3MaskedDataset, grouped_plane_split  # noqa: E402


def save_patch_file(root, count=2):
    root.mkdir(parents=True, exist_ok=True)
    time = np.arange(256, dtype=np.float32) * 0.004
    base = np.sin(2 * np.pi * 18 * time)[:, None]
    patches = np.stack(
        [np.repeat(base * (index + 1), 256, axis=1) for index in range(count)]
    )
    np.save(root / "train_clean_narrow.npy", patches.astype(np.float32))
    np.save(
        root / "train_metadata.npy",
        np.asarray(
            [
                {"clean_id": index, "source_axis": "inline", "section_number": index}
                for index in range(count)
            ],
            dtype=object,
        ),
    )


def test_f3_dataset_changes_task_without_changing_clean_patch(tmp_path):
    save_patch_file(tmp_path, count=2)
    dataset = F3MaskedDataset(tmp_path, split="train", seed=42)
    first = dataset.sample_at(0, epoch=1)
    second = dataset.sample_at(0, epoch=2)
    assert first["clean_id"] == second["clean_id"]
    assert first["task_name"] != second["task_name"] or not np.allclose(
        first["input"].numpy(),
        second["input"].numpy(),
    )


def test_only_horizontal_flip_is_used():
    source = inspect.getsource(F3MaskedDataset)
    assert "axis=0" not in source
    assert "axis=1" in source


def test_grouped_split_keeps_planes_disjoint():
    rows = [
        {"source_axis": "inline", "section_number": number, "patch": patch}
        for number in range(10)
        for patch in range(2)
    ]
    train, val = grouped_plane_split(rows, val_fraction=0.2, seed=42)
    train_planes = {(row["source_axis"], row["section_number"]) for row in train}
    val_planes = {(row["source_axis"], row["section_number"]) for row in val}
    assert train_planes.isdisjoint(val_planes)
