import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    DATA_DIR,
    DT,
    FIGURE_DIR,
    NOISE_LEVELS,
    PATCH_SIZE,
    PATCH_STRIDE,
    Q_FILTER_Q,
    Q_FILTER_STRENGTH,
    RANDOM_SEED,
    SYNTHETIC_SECTIONS_PER_WAVELET,
    SYNTHETIC_SECTION_WIDTH,
)
from signal_utils import apply_time_variant_q_filter_section, convolve_reflectivity


def patch_starts(n, patch_size, stride, cover_last=True):
    if n < patch_size:
        raise ValueError(f"Section size {n} is smaller than patch size {patch_size}.")
    starts = list(range(0, n - patch_size + 1, stride))
    last = n - patch_size
    if cover_last and starts[-1] != last:
        starts.append(last)
    return starts


def minmax_with_stats(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32), mn, mx
    out = 2.0 * (x - mn) / (mx - mn) - 1.0
    return out.astype(np.float32), mn, mx


def build_structural_reflectivity(r1d, width, rng):
    r1d = np.asarray(r1d, dtype=np.float32)
    n_time = r1d.size
    t_axis = np.arange(n_time, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    xc = (width - 1) / 2.0

    slope = rng.uniform(-0.12, 0.12)
    curvature = rng.uniform(-35.0, 35.0) * ((x - xc) / max(xc, 1.0)) ** 2
    shifts = slope * (x - xc) + curvature

    folds = []
    for _ in range(rng.integers(2, 5)):
        fold_amp = rng.uniform(6.0, 35.0)
        fold_period = rng.uniform(140.0, 650.0)
        fold_phase = rng.uniform(0.0, 2.0 * np.pi)
        shifts += fold_amp * np.sin(2.0 * np.pi * x / fold_period + fold_phase)
        folds.append({
            "amp_samples": float(fold_amp),
            "period_traces": float(fold_period),
            "phase_rad": float(fold_phase),
        })

    for _ in range(rng.integers(1, 4)):
        center_x = rng.uniform(0.15 * width, 0.85 * width)
        sigma_x = rng.uniform(70.0, 240.0)
        amp = rng.uniform(-28.0, 28.0)
        shifts += amp * np.exp(-0.5 * ((x - center_x) / sigma_x) ** 2)

    fault_info = []
    if rng.random() < 0.65:
        fault_x = int(rng.integers(width // 4, width * 3 // 4))
        fault_throw = float(rng.uniform(-28.0, 28.0))
        shifts[x >= fault_x] += fault_throw
        fault_info.append({"x": fault_x, "throw_samples": fault_throw})
    if rng.random() < 0.25:
        fault_x = int(rng.integers(width // 5, width * 4 // 5))
        fault_throw = float(rng.uniform(-16.0, 16.0))
        shifts[x >= fault_x] += fault_throw
        fault_info.append({"x": fault_x, "throw_samples": fault_throw})

    lateral_jitter = gaussian_filter1d(rng.normal(0.0, 3.5, size=width), sigma=7.0)
    shifts += lateral_jitter

    section = np.zeros((n_time, width), dtype=np.float32)
    index = np.arange(n_time, dtype=np.float32)
    for ix, shift in enumerate(shifts):
        vertical_jitter = gaussian_filter1d(rng.normal(0.0, 0.75, size=n_time), sigma=6.0)
        vertical_jitter += rng.uniform(-1.2, 1.2) * np.sin(
            2.0 * np.pi * t_axis / rng.uniform(90.0, 220.0) + rng.uniform(0.0, 2.0 * np.pi)
        )
        sample_pos = t_axis + shift + vertical_jitter
        section[:, ix] = np.interp(sample_pos, index, r1d, left=0.0, right=0.0)

    return section, {
        "slope_samples_per_trace": float(slope),
        "curvature_samples": float(curvature[-1] - curvature[int(xc)]),
        "folds": folds,
        "fault": fault_info,
    }


def add_noise_by_ratio(section, noise_level, rng):
    signal_std = float(np.std(section))
    if noise_level <= 0.0 or signal_std <= 1e-12:
        return section.astype(np.float32)
    noise = rng.normal(0.0, signal_std * noise_level, size=section.shape)
    return (section + noise).astype(np.float32)


def cut_patches(input_section, label_section, patch_size, stride):
    t_starts = patch_starts(input_section.shape[0], patch_size, stride)
    x_starts = patch_starts(input_section.shape[1], patch_size, stride)
    inputs = []
    labels = []
    locations = []
    for t0 in t_starts:
        for x0 in x_starts:
            inputs.append(input_section[t0:t0 + patch_size, x0:x0 + patch_size])
            labels.append(label_section[t0:t0 + patch_size, x0:x0 + patch_size])
            locations.append((t0, x0))
    return np.stack(inputs), np.stack(labels), locations


def split_train_val(inputs, labels, metadata, rng, train_ratio=0.8):
    order = rng.permutation(inputs.shape[0])
    n_train = int(round(inputs.shape[0] * train_ratio))
    train_idx = order[:n_train]
    val_idx = order[n_train:]
    return {
        "train_inputs": inputs[train_idx],
        "train_labels": labels[train_idx],
        "val_inputs": inputs[val_idx],
        "val_labels": labels[val_idx],
        "train_metadata": [metadata[i] for i in train_idx],
        "val_metadata": [metadata[i] for i in val_idx],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    parser.add_argument("--section-width", type=int, default=SYNTHETIC_SECTION_WIDTH)
    parser.add_argument("--sections-per-wavelet", type=int, default=SYNTHETIC_SECTIONS_PER_WAVELET)
    parser.add_argument("--noise-levels", default=",".join(str(x) for x in NOISE_LEVELS))
    parser.add_argument("--q", type=float, default=Q_FILTER_Q)
    parser.add_argument("--q-strength", type=float, default=Q_FILTER_STRENGTH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    reflectivities = np.load(DATA_DIR / "well_reflectivities.npy", allow_pickle=True).item()
    wavelet_pairs = np.load(DATA_DIR / "well_wavelet_pairs.npy", allow_pickle=True).item()
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]

    all_inputs = []
    all_labels = []
    patch_metadata = []
    section_metadata = []
    section_inputs = {}
    section_labels = {}
    reflectivity_sections = {}
    q_reflectivity_sections = {}
    synthetic_narrow_no_q = {}
    synthetic_wide_no_q = {}
    synthetic_narrow_q = {}
    synthetic_wide_q = {}

    for well_name, reflectivity in reflectivities.items():
        narrow_wavelet = wavelet_pairs[well_name]["narrow"]
        wide_wavelet = wavelet_pairs[well_name]["wide"]
        for section_idx in range(args.sections_per_wavelet):
            base_reflectivity, structure = build_structural_reflectivity(
                reflectivity, args.section_width, rng
            )
            q_reflectivity = apply_time_variant_q_filter_section(
                base_reflectivity, DT, q=args.q, strength=args.q_strength
            )
            clean_narrow_no_q = convolve_reflectivity(base_reflectivity, narrow_wavelet)
            clean_wide_no_q = convolve_reflectivity(base_reflectivity, wide_wavelet)
            clean_narrow = convolve_reflectivity(q_reflectivity, narrow_wavelet)
            clean_wide = convolve_reflectivity(q_reflectivity, wide_wavelet)
            noise_level = noise_levels[section_idx % len(noise_levels)]
            section_id = f"{well_name}_section_{section_idx:02d}_noise_{noise_level:.2f}"
            noisy_narrow = add_noise_by_ratio(clean_narrow, noise_level, rng)
            input_norm, input_min, input_max = minmax_with_stats(noisy_narrow)
            label_norm, label_min, label_max = minmax_with_stats(clean_wide)
            patches_x, patches_y, locations = cut_patches(
                input_norm, label_norm, args.patch_size, args.stride
            )

            start_index = len(all_inputs)
            all_inputs.extend(patches_x)
            all_labels.extend(patches_y)
            for patch_idx, (t0, x0) in enumerate(locations):
                patch_metadata.append({
                    "section_id": section_id,
                    "well": well_name,
                    "noise_level": noise_level,
                    "t0": int(t0),
                    "x0": int(x0),
                    "input_min": input_min,
                    "input_max": input_max,
                    "label_min": label_min,
                    "label_max": label_max,
                })

            section_inputs[section_id] = input_norm.astype(np.float32)
            section_labels[section_id] = label_norm.astype(np.float32)
            reflectivity_sections[section_id] = base_reflectivity.astype(np.float32)
            q_reflectivity_sections[section_id] = q_reflectivity.astype(np.float32)
            synthetic_narrow_no_q[section_id] = clean_narrow_no_q.astype(np.float32)
            synthetic_wide_no_q[section_id] = clean_wide_no_q.astype(np.float32)
            synthetic_narrow_q[section_id] = clean_narrow.astype(np.float32)
            synthetic_wide_q[section_id] = clean_wide.astype(np.float32)
            section_metadata.append({
                "section_id": section_id,
                "well": well_name,
                "section_index": int(section_idx),
                "noise_level": noise_level,
                "num_patches": int(len(locations)),
                "patch_index_start": int(start_index),
                "patch_index_end": int(start_index + len(locations)),
                "input_min": input_min,
                "input_max": input_max,
                "label_min": label_min,
                "label_max": label_max,
                "q": float(args.q),
                "q_strength": float(args.q_strength),
                "structure": structure,
            })

    inputs = np.stack(all_inputs).astype(np.float32)
    labels = np.stack(all_labels).astype(np.float32)
    split = split_train_val(inputs, labels, patch_metadata, rng, train_ratio=0.8)

    for name in ("train_inputs", "train_labels", "val_inputs", "val_labels"):
        np.save(DATA_DIR / f"{name}.npy", split[name])

    np.save(DATA_DIR / "synthetic_section_inputs.npy", section_inputs)
    np.save(DATA_DIR / "synthetic_section_labels.npy", section_labels)
    np.save(DATA_DIR / "synthetic_reflectivity_sections.npy", reflectivity_sections)
    np.save(DATA_DIR / "synthetic_q_reflectivity_sections.npy", q_reflectivity_sections)
    np.save(DATA_DIR / "synthetic_narrow_no_q_sections.npy", synthetic_narrow_no_q)
    np.save(DATA_DIR / "synthetic_wide_no_q_sections.npy", synthetic_wide_no_q)
    np.save(DATA_DIR / "synthetic_narrow_q_sections.npy", synthetic_narrow_q)
    np.save(DATA_DIR / "synthetic_wide_q_sections.npy", synthetic_wide_q)
    np.save(DATA_DIR / "synthetic_patch_metadata.npy", {
        "train": split["train_metadata"],
        "val": split["val_metadata"],
    })
    np.save(DATA_DIR / "synthetic_metadata.npy", {
        "dt": DT,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "section_width": args.section_width,
        "sections_per_wavelet": args.sections_per_wavelet,
        "noise_levels": noise_levels,
        "q": float(args.q),
        "q_strength": float(args.q_strength),
        "num_sections": len(section_metadata),
        "num_patches": int(inputs.shape[0]),
        "num_train": int(split["train_inputs"].shape[0]),
        "num_val": int(split["val_inputs"].shape[0]),
        "sections": section_metadata,
    })

    first_key = sorted(section_inputs)[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    q_panels = [
        (reflectivity_sections[first_key], "Reflectivity before Q"),
        (q_reflectivity_sections[first_key], "Reflectivity after Q"),
        (synthetic_wide_no_q[first_key], "Wide synthetic before Q"),
        (synthetic_wide_q[first_key], "Wide synthetic after Q"),
    ]
    for ax, (data, title) in zip(axes.ravel(), q_panels):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_q_filter_before_after_example.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(section_inputs[first_key], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title(f"{first_key} narrow input")
    axes[1].imshow(section_labels[first_key], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title(f"{first_key} wide label")
    for ax in axes:
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_synthetic_section_example.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(split["train_inputs"][0], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("Training patch input")
    axes[1].imshow(split["train_labels"][0], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title("Training patch label")
    for ax in axes:
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_synthetic_pair_example.png", dpi=300)
    plt.close(fig)

    print(f"Saved synthetic dataset to {DATA_DIR}")
    print(f"sections: {len(section_metadata)}")
    print(f"all patches: {inputs.shape}")
    print(f"train_inputs: {split['train_inputs'].shape}")
    print(f"val_inputs: {split['val_inputs'].shape}")


if __name__ == "__main__":
    main()
