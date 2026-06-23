import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    DT,
    F3_MASK_TASKS,
    LOCAL_RATIO,
    NOISE_LEVELS,
    PROJECTOR,
    SYNTHETIC_RATIO,
)
from masking import make_masked_pair


PROJECTOR_KEYS = ("low_stop", "low_pass", "high_pass", "high_stop")


def projector_tensor(projector):
    return torch.tensor(
        [float(projector[key]) for key in PROJECTOR_KEYS],
        dtype=torch.float32,
    )


def domain_cycle(seed, size=1000):
    counts = {
        "local_wide": int(round(size * LOCAL_RATIO)),
        "synthetic": int(round(size * SYNTHETIC_RATIO)),
    }
    counts["f3"] = size - counts["local_wide"] - counts["synthetic"]
    values = [
        domain
        for domain, count in counts.items()
        for _ in range(count)
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return values


class LocalCalibrationDataset(Dataset):
    def __init__(self, root, split="train", seed=42, augment=None):
        self.inputs = np.load(root / f"{split}_inputs.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.metadata = np.load(
            root / f"{split}_metadata.npy",
            allow_pickle=True,
        )
        self.seed = int(seed)
        self.epoch = 1
        self.augment = split == "train" if augment is None else bool(augment)

    def __len__(self):
        return int(self.inputs.shape[0])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        x = np.asarray(self.inputs[index], dtype=np.float32)
        label = np.asarray(self.labels[index], dtype=np.float32)
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index)
        if self.augment and rng.random() < 0.5:
            x = np.flip(x, axis=1)
            label = np.flip(label, axis=1)
        return {
            "input": torch.from_numpy(x.copy()).unsqueeze(0),
            "label": torch.from_numpy(label.copy()).unsqueeze(0),
            "target": torch.from_numpy((x + label).copy()).unsqueeze(0),
            "projector": projector_tensor(PROJECTOR),
            "domain": "local_wide",
            "well": dict(self.metadata[index]).get("well"),
        }


class SyntheticResidualDataset(Dataset):
    def __init__(self, root, split="train", seed=42, augment=None):
        self.inputs = np.load(root / f"{split}_inputs.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.seed = int(seed)
        self.epoch = 1
        self.augment = split == "train" if augment is None else bool(augment)

    def __len__(self):
        return int(self.inputs.shape[0])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        x = np.asarray(self.inputs[index], dtype=np.float32)
        label = np.asarray(self.labels[index], dtype=np.float32)
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index)
        if self.augment and rng.random() < 0.5:
            x = np.flip(x, axis=1)
            label = np.flip(label, axis=1)
        return {
            "input": torch.from_numpy(x.copy()).unsqueeze(0),
            "label": torch.from_numpy(label.copy()).unsqueeze(0),
            "target": torch.from_numpy((x + label).copy()).unsqueeze(0),
            "projector": projector_tensor(PROJECTOR),
            "domain": "synthetic",
        }


class F3MaskedDataset(Dataset):
    def __init__(self, root, split="train", seed=42, augment=None):
        self.patches = np.load(
            root / f"{split}_clean_narrow.npy",
            mmap_mode="r",
        )
        self.seed = int(seed)
        self.epoch = 1
        self.augment = split == "train" if augment is None else bool(augment)

    def __len__(self):
        return int(self.patches.shape[0])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        rng = np.random.default_rng(
            self.seed + self.epoch * 1000003 + index * 9973
        )
        task = tuple(F3_MASK_TASKS)[int(rng.integers(0, len(F3_MASK_TASKS)))]
        noise = float(NOISE_LEVELS[int(rng.integers(0, len(NOISE_LEVELS)))])
        pair = make_masked_pair(
            np.asarray(self.patches[index], dtype=np.float32),
            DT,
            task,
            rng,
            noise,
        )
        x, label, target = pair.input_norm, pair.label_norm, pair.target_norm
        if self.augment and rng.random() < 0.5:
            x = np.flip(x, axis=1)
            label = np.flip(label, axis=1)
            target = np.flip(target, axis=1)
        return {
            "input": torch.from_numpy(x.copy()).unsqueeze(0),
            "label": torch.from_numpy(label.copy()).unsqueeze(0),
            "target": torch.from_numpy(target.copy()).unsqueeze(0),
            "projector": projector_tensor(pair.projector),
            "domain": "f3",
        }
