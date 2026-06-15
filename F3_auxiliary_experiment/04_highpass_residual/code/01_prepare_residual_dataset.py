import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from common import (
    DATA_DIR,
    DT,
    FIGURE_DIR,
    RESIDUAL_BAND,
    SOURCE_DATA_DIR,
    ensure_dirs,
    trapezoid_band,
)


def bandpass_residual_batch(residuals):
    residuals = np.asarray(residuals, dtype=np.float32)
    nt = residuals.shape[1]
    freqs = np.fft.rfftfreq(nt, DT)
    filt = trapezoid_band(freqs, *RESIDUAL_BAND)[None, :, None]
    spec = np.fft.rfft(residuals, axis=1)
    filtered = np.fft.irfft(spec * filt, n=nt, axis=1)
    return filtered.astype(np.float32)


def save_split(split, chunk_size=64):
    inputs = np.load(SOURCE_DATA_DIR / f"{split}_inputs.npy", mmap_mode="r")
    labels = np.load(SOURCE_DATA_DIR / f"{split}_labels.npy", mmap_mode="r")
    shape = inputs.shape

    input_out = np.lib.format.open_memmap(
        DATA_DIR / f"{split}_inputs.npy", mode="w+", dtype=np.float32, shape=shape
    )
    label_out = np.lib.format.open_memmap(
        DATA_DIR / f"{split}_labels.npy", mode="w+", dtype=np.float32, shape=shape
    )
    residual_out = np.lib.format.open_memmap(
        DATA_DIR / f"{split}_residuals.npy", mode="w+", dtype=np.float32, shape=shape
    )

    for start in range(0, shape[0], chunk_size):
        stop = min(shape[0], start + chunk_size)
        x = np.asarray(inputs[start:stop], dtype=np.float32)
        y = np.asarray(labels[start:stop], dtype=np.float32)
        residual = bandpass_residual_batch(y - x)
        input_out[start:stop] = x
        label_out[start:stop] = y
        residual_out[start:stop] = residual
        print(f"{split}: prepared {stop}/{shape[0]}", flush=True)

    input_out.flush()
    label_out.flush()
    residual_out.flush()
    return input_out, label_out, residual_out


def main():
    ensure_dirs()
    train_inputs, train_labels, train_residuals = save_split("train")
    val_inputs, val_labels, val_residuals = save_split("val")

    np.save(DATA_DIR / "residual_dataset_metadata.npy", {
        "train_shape": train_inputs.shape,
        "val_shape": val_inputs.shape,
        "source": str(SOURCE_DATA_DIR),
        "residual_definition": "bandpass(wide_label - narrow_input)",
        "residual_band_hz": RESIDUAL_BAND,
        "dt_s": DT,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (train_inputs[0], "Narrow input"),
        (train_residuals[0], "Band-limited residual label"),
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
    fig.savefig(FIGURE_DIR / "01_highpass_residual_training_sample.png", dpi=300)
    plt.close(fig)

    print(f"train_inputs: {train_inputs.shape}")
    print(f"val_inputs: {val_inputs.shape}")
    print(f"Residual band: {RESIDUAL_BAND} Hz")
    print(f"Saved high-pass residual dataset to {DATA_DIR}")


if __name__ == "__main__":
    main()
