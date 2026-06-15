import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from common import DATA_DIR, FIGURE_DIR, SOURCE_DATA_DIR, ensure_dirs


def save_split(split):
    inputs = np.load(SOURCE_DATA_DIR / f"{split}_inputs.npy").astype(np.float32)
    labels = np.load(SOURCE_DATA_DIR / f"{split}_labels.npy").astype(np.float32)
    residuals = (labels - inputs).astype(np.float32)
    np.save(DATA_DIR / f"{split}_inputs.npy", inputs)
    np.save(DATA_DIR / f"{split}_labels.npy", labels)
    np.save(DATA_DIR / f"{split}_residuals.npy", residuals)
    return inputs, labels, residuals


def main():
    ensure_dirs()
    train_inputs, train_labels, train_residuals = save_split("train")
    val_inputs, val_labels, val_residuals = save_split("val")

    np.save(DATA_DIR / "residual_dataset_metadata.npy", {
        "train_shape": train_inputs.shape,
        "val_shape": val_inputs.shape,
        "source": str(SOURCE_DATA_DIR),
        "residual_definition": "wide_label - narrow_input",
    })

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (train_inputs[0], "Narrow input"),
        (train_residuals[0], "Residual label"),
        (train_labels[0], "Wide label"),
    ]
    for ax, (data, title) in zip(axes, panels):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_residual_training_sample.png", dpi=300)
    plt.close(fig)

    print(f"train_inputs: {train_inputs.shape}")
    print(f"val_inputs: {val_inputs.shape}")
    print(f"Saved residual dataset to {DATA_DIR}")


if __name__ == "__main__":
    main()
