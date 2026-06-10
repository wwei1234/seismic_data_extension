from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyPatchDataset(Dataset):
    def __init__(self, input_path, label_path):
        self.inputs = np.load(Path(input_path), mmap_mode="r")
        self.labels = np.load(Path(label_path), mmap_mode="r")
        if self.inputs.shape != self.labels.shape:
            raise ValueError(f"Shape mismatch: {self.inputs.shape} vs {self.labels.shape}")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.inputs[idx], dtype=np.float32).copy()).unsqueeze(0)
        y = torch.from_numpy(np.asarray(self.labels[idx], dtype=np.float32).copy()).unsqueeze(0)
        return x, y
