import math

import numpy as np
import torch
from torch.utils.data import Dataset

from config import REAL_DATA_DIR, SYNTHETIC_DATA_DIR


def build_source_schedule(length, real_probability=0.7):
    real_count = int(round(length * real_probability))
    return ["real"] * real_count + ["synthetic"] * (length - real_count)


class HybridResidualDataset(Dataset):
    def __init__(
        self,
        split,
        real_probability=0.7,
        augment=False,
        epoch_size=None,
    ):
        self.real_inputs = np.load(
            REAL_DATA_DIR / f"{split}_inputs.npy",
            mmap_mode="r",
        )
        self.real_labels = np.load(
            REAL_DATA_DIR / f"{split}_labels.npy",
            mmap_mode="r",
        )
        self.synthetic_inputs = np.load(
            SYNTHETIC_DATA_DIR / f"{split}_inputs.npy",
            mmap_mode="r",
        )
        self.synthetic_labels = np.load(
            SYNTHETIC_DATA_DIR / f"{split}_labels.npy",
            mmap_mode="r",
        )
        self.real_probability = float(real_probability)
        self.augment = bool(augment)
        if epoch_size is None:
            epoch_size = max(
                len(self.real_inputs),
                int(math.ceil(len(self.real_inputs) / max(self.real_probability, 1e-6))),
            )
        self.schedule = build_source_schedule(epoch_size, self.real_probability)
        self.real_positions = []
        self.synthetic_positions = []
        real_index = 0
        synthetic_index = 0
        for source in self.schedule:
            if source == "real":
                self.real_positions.append(real_index % len(self.real_inputs))
                self.synthetic_positions.append(-1)
                real_index += 1
            else:
                self.real_positions.append(-1)
                self.synthetic_positions.append(
                    synthetic_index % len(self.synthetic_inputs)
                )
                synthetic_index += 1

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, index):
        source = self.schedule[index]
        if source == "real":
            data_index = self.real_positions[index]
            x = np.asarray(self.real_inputs[data_index], dtype=np.float32)
            residual = np.asarray(self.real_labels[data_index], dtype=np.float32)
            source_id = 1
        else:
            data_index = self.synthetic_positions[index]
            x = np.asarray(self.synthetic_inputs[data_index], dtype=np.float32)
            residual = np.asarray(self.synthetic_labels[data_index], dtype=np.float32)
            source_id = 0
        if self.augment and np.random.rand() < 0.5:
            x = np.flip(x, axis=1)
            residual = np.flip(residual, axis=1)
        return (
            torch.from_numpy(x.copy()).unsqueeze(0),
            torch.from_numpy(residual.copy()).unsqueeze(0),
            torch.tensor(source_id, dtype=torch.int64),
        )
