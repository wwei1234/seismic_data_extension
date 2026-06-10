import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from config import DATA_DIR, DT, FIGURE_DIR, PATCH_SIZE, RANDOM_SEED
from signal_utils import convolve_reflectivity, normalize_minmax


def sample_structural_reflectivity(r1d, patch_size, rng):
    n = r1d.size
    margin = patch_size // 2 + 30
    center = rng.integers(margin, n - margin)
    x = np.arange(patch_size)
    xc = (patch_size - 1) / 2.0

    slope = rng.uniform(-0.25, 0.25)
    fold_amp = rng.uniform(-10.0, 10.0)
    fold_period = rng.uniform(60.0, 160.0)
    fold_phase = rng.uniform(0.0, 2.0 * np.pi)
    shifts = slope * (x - xc) + fold_amp * np.sin(2.0 * np.pi * x / fold_period + fold_phase)

    if rng.random() < 0.45:
        fault_x = rng.integers(patch_size // 4, patch_size * 3 // 4)
        fault_throw = rng.uniform(-12.0, 12.0)
        shifts[x >= fault_x] += fault_throw

    shifts += rng.normal(0.0, 1.0, size=patch_size)
    t = center + np.arange(-patch_size // 2, patch_size // 2)
    out = np.zeros((patch_size, patch_size), dtype=np.float32)
    idx_axis = np.arange(n)

    for ix, shift in enumerate(shifts):
        sample_pos = t + shift
        out[:, ix] = np.interp(sample_pos, idx_axis, r1d, left=0.0, right=0.0)

    return out


def make_pair(reflectivity, narrow_wavelet, wide_wavelet, noise_level, rng):
    narrow = convolve_reflectivity(reflectivity, narrow_wavelet)
    wide = convolve_reflectivity(reflectivity, wide_wavelet)
    if noise_level > 0:
        narrow = narrow + rng.normal(0.0, np.std(narrow) * noise_level, size=narrow.shape)
    return normalize_minmax(narrow), normalize_minmax(wide)


def split_arrays(inputs, labels, train_ratio=0.8, val_ratio=0.1):
    n = inputs.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train_inputs": inputs[:n_train],
        "train_labels": labels[:n_train],
        "val_inputs": inputs[n_train:n_train + n_val],
        "val_labels": labels[n_train:n_train + n_val],
        "test_inputs": inputs[n_train + n_val:],
        "test_labels": labels[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1200)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--noise-levels", default="0.05,0.10,0.15")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    reflectivities = np.load(DATA_DIR / "well_reflectivities.npy", allow_pickle=True).item()
    narrow_wavelet = np.load(DATA_DIR / "estimated_narrow_wavelet.npy")
    wide_wavelet = np.load(DATA_DIR / "estimated_wide_wavelet.npy")
    wells = list(reflectivities.keys())
    noise_levels = [float(x) for x in args.noise_levels.split(",")]

    inputs = np.zeros((args.num_samples, args.patch_size, args.patch_size), dtype=np.float32)
    labels = np.zeros_like(inputs)

    for i in range(args.num_samples):
        well_name = wells[i % len(wells)]
        r2d = sample_structural_reflectivity(reflectivities[well_name], args.patch_size, rng)
        noise_level = noise_levels[i % len(noise_levels)]
        inputs[i], labels[i] = make_pair(r2d, narrow_wavelet, wide_wavelet, noise_level, rng)

    order = rng.permutation(args.num_samples)
    inputs = inputs[order]
    labels = labels[order]

    arrays = split_arrays(inputs, labels)
    for name, arr in arrays.items():
        np.save(DATA_DIR / f"{name}.npy", arr)

    metadata = {
        "num_samples": args.num_samples,
        "patch_size": args.patch_size,
        "dt": DT,
        "noise_levels": noise_levels,
    }
    np.save(DATA_DIR / "synthetic_metadata.npy", metadata)

    idx = 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(inputs[idx], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("Synthetic narrow input")
    axes[1].imshow(labels[idx], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title("Synthetic wide label")
    for ax in axes:
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_synthetic_pair_example.png", dpi=300)
    plt.close(fig)

    print(f"Saved synthetic dataset to {DATA_DIR}")
    for name, arr in arrays.items():
        print(f"{name}: {arr.shape}")


if __name__ == "__main__":
    main()
