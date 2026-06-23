import numpy as np
import torch
from torch.utils.data import Dataset

from config import DT, F3_MASK_TASKS, FINAL_PROJECTOR, NOISE_LEVELS
from leakage_guard import assert_training_paths_are_safe
from masking import make_masked_pair


PROJECTOR_KEYS = ("low_stop", "low_pass", "high_pass", "high_stop")


def grouped_plane_split(rows, val_fraction=0.2, seed=42):
    plane_keys = sorted(
        {(row["source_axis"], int(row["section_number"])) for row in rows}
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(plane_keys)
    val_count = max(1, int(round(len(plane_keys) * val_fraction)))
    val_planes = set(plane_keys[:val_count])
    train = []
    val = []
    for row in rows:
        key = (row["source_axis"], int(row["section_number"]))
        (val if key in val_planes else train).append(row)
    return train, val


def _projector_tensor(projector):
    return torch.tensor(
        [float(projector[key]) for key in PROJECTOR_KEYS],
        dtype=torch.float32,
    )


class F3MaskedDataset(Dataset):
    def __init__(self, root, split="train", seed=42, augment=None):
        self.root = root
        self.split = str(split)
        self.seed = int(seed)
        self.epoch = 1
        self.augment = self.split == "train" if augment is None else bool(augment)
        patch_path = root / f"{self.split}_clean_narrow.npy"
        metadata_path = root / f"{self.split}_metadata.npy"
        assert_training_paths_are_safe([patch_path, metadata_path])
        self.patches = np.load(patch_path, mmap_mode="r")
        self.metadata = np.load(metadata_path, allow_pickle=True)

    def __len__(self):
        return int(self.patches.shape[0])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def sample_at(self, index, epoch=None):
        current_epoch = self.epoch if epoch is None else int(epoch)
        rng = np.random.default_rng(
            self.seed + current_epoch * 1_000_003 + int(index) * 9_973
        )
        task_name = tuple(F3_MASK_TASKS)[int(rng.integers(0, len(F3_MASK_TASKS)))]
        noise_level = float(NOISE_LEVELS[int(rng.integers(0, len(NOISE_LEVELS)))])
        pair = make_masked_pair(
            np.asarray(self.patches[index], dtype=np.float32),
            dt=DT,
            task_name=task_name,
            rng=rng,
            noise_level=noise_level,
        )
        input_data = pair.input_norm
        label = pair.label_norm
        target = pair.target_norm
        if self.augment and rng.random() < 0.5:
            input_data = np.flip(input_data, axis=1)
            label = np.flip(label, axis=1)
            target = np.flip(target, axis=1)
        metadata = dict(self.metadata[index])
        return {
            "input": torch.from_numpy(input_data.copy()).unsqueeze(0),
            "label": torch.from_numpy(label.copy()).unsqueeze(0),
            "target": torch.from_numpy(target.copy()).unsqueeze(0),
            "projector": _projector_tensor(pair.projector),
            "domain": "f3",
            "task_name": task_name,
            "clean_id": int(metadata.get("clean_id", index)),
            "noise_level": noise_level,
        }

    def __getitem__(self, index):
        return self.sample_at(index)


class SyntheticResidualDataset(Dataset):
    def __init__(self, root, split="train", seed=42, augment=None):
        self.root = root
        self.split = str(split)
        self.seed = int(seed)
        self.epoch = 1
        self.augment = self.split == "train" if augment is None else bool(augment)
        paths = {
            "input": root / f"{split}_inputs.npy",
            "label": root / f"{split}_labels.npy",
            "clean": root / f"{split}_clean_narrow.npy",
            "wide": root / f"{split}_wide.npy",
            "metadata": root / f"{split}_metadata.npy",
        }
        assert_training_paths_are_safe(paths.values())
        self.inputs = np.load(paths["input"], mmap_mode="r")
        self.labels = np.load(paths["label"], mmap_mode="r")
        self.clean = np.load(paths["clean"], mmap_mode="r")
        self.wide = np.load(paths["wide"], mmap_mode="r")
        self.metadata = np.load(paths["metadata"], allow_pickle=True)

    def __len__(self):
        return int(self.inputs.shape[0])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + int(index) * 9_973
        )
        input_data = np.asarray(self.inputs[index], dtype=np.float32)
        label = np.asarray(self.labels[index], dtype=np.float32)
        clean = np.asarray(self.clean[index], dtype=np.float32)
        wide = np.asarray(self.wide[index], dtype=np.float32)
        if self.augment and rng.random() < 0.5:
            input_data = np.flip(input_data, axis=1)
            label = np.flip(label, axis=1)
            clean = np.flip(clean, axis=1)
            wide = np.flip(wide, axis=1)
        metadata = dict(self.metadata[index])
        return {
            "input": torch.from_numpy(input_data.copy()).unsqueeze(0),
            "label": torch.from_numpy(label.copy()).unsqueeze(0),
            "target": torch.from_numpy(wide.copy()).unsqueeze(0),
            "clean_input": torch.from_numpy(clean.copy()).unsqueeze(0),
            "projector": _projector_tensor(FINAL_PROJECTOR),
            "domain": "synthetic",
            "section_id": str(metadata["section_id"]),
        }
