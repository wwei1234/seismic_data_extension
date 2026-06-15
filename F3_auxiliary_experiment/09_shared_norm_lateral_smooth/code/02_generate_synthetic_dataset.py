"""Generate synthetic training dataset with improved realism.

Changes from the 01_basic_strategy version (see 08_agent instruction):
  1. Noise lowered: [0.0, 0.01, 0.02, 0.05], added AFTER normalisation
  2. Lateral smoothing (uniform_filter1d, size=5) to remove inter-trace jitter
  3. Low-frequency background removed via 8 Hz highpass on convolved traces
  4. Fold amplitude reduced: (3, 15) instead of (6, 35)
  5. Q filter applied to convolved traces (physically correct), not reflectivity
  6. Shared normalisation: clip to 99th-pctl of |wide| so input/label share scale
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    DATA_DIR,
    DT,
    FIGURE_DIR,
    LOWCUT_FREQ,
    NOISE_LEVELS,
    PATCH_SIZE,
    PATCH_STRIDE,
    Q_FILTER_Q,
    Q_FILTER_STRENGTH,
    RANDOM_SEED,
    SOURCE_DATA_DIR,
    SYNTHETIC_SECTIONS_PER_WAVELET,
    SYNTHETIC_SECTION_WIDTH,
)
from signal_utils import (
    apply_time_variant_q_filter_section,
    convolve_reflectivity,
    remove_lowfreq_background,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def patch_starts(n, patch_size, stride, cover_last=True):
    if n < patch_size:
        raise ValueError(f"Section size {n} is smaller than patch size {patch_size}.")
    starts = list(range(0, n - patch_size + 1, stride))
    last = n - patch_size
    if cover_last and starts[-1] != last:
        starts.append(last)
    return starts


# ── Structural reflectivity building ─────────────────────────────────────────


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
        fold_amp = rng.uniform(3.0, 15.0)          # ⇐ reduced from (6, 35)
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

    # ⇐ Modification 2: lateral smoothing to remove inter-trace jitter
    section = uniform_filter1d(section, size=5, axis=1)

    return section, {
        "slope_samples_per_trace": float(slope),
        "curvature_samples": float(curvature[-1] - curvature[int(xc)]),
        "folds": folds,
        "fault": fault_info,
    }


# ── Patch extraction ─────────────────────────────────────────────────────────


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


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    parser.add_argument("--section-width", type=int, default=SYNTHETIC_SECTION_WIDTH)
    parser.add_argument("--sections-per-wavelet", type=int, default=SYNTHETIC_SECTIONS_PER_WAVELET)
    parser.add_argument("--noise-levels", default=",".join(str(x) for x in NOISE_LEVELS))
    parser.add_argument("--q", type=float, default=Q_FILTER_Q)
    parser.add_argument("--q-strength", type=float, default=Q_FILTER_STRENGTH)
    parser.add_argument("--lowcut", type=float, default=LOWCUT_FREQ,
                        help="Highpass cutoff Hz. 0 disables low-freq removal.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    reflectivities = np.load(SOURCE_DATA_DIR / "well_reflectivities.npy", allow_pickle=True).item()
    wavelet_pairs = np.load(SOURCE_DATA_DIR / "well_wavelet_pairs.npy", allow_pickle=True).item()
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]

    all_inputs = []
    all_labels = []
    patch_metadata = []
    section_metadata = []
    section_inputs = {}
    section_labels = {}

    for well_name, reflectivity in reflectivities.items():
        narrow_wavelet = wavelet_pairs[well_name]["narrow"]
        wide_wavelet = wavelet_pairs[well_name]["wide"]
        for section_idx in range(args.sections_per_wavelet):
            base_reflectivity, structure = build_structural_reflectivity(
                reflectivity, args.section_width, rng
            )

            # Convolve with base (non-Q) reflectivity first — Modification 5
            clean_narrow = convolve_reflectivity(base_reflectivity, narrow_wavelet)
            clean_wide = convolve_reflectivity(base_reflectivity, wide_wavelet)

            # Apply Q filter to convolved traces — Modification 5 (moved from reflectivity)
            clean_narrow = apply_time_variant_q_filter_section(
                clean_narrow, DT, q=args.q, strength=args.q_strength
            )
            clean_wide = apply_time_variant_q_filter_section(
                clean_wide, DT, q=args.q, strength=args.q_strength
            )

            # Remove low-frequency background — Modification 3 (optional)
            if args.lowcut > 0:
                clean_narrow = remove_lowfreq_background(clean_narrow, dt=DT, f_cut=args.lowcut)
                clean_wide = remove_lowfreq_background(clean_wide, dt=DT, f_cut=args.lowcut)

            # Shared-scale normalisation — Modification 1
            shared_scale = float(np.percentile(np.abs(clean_wide), 99))
            shared_scale = max(shared_scale, 1e-8)
            input_norm = np.clip(clean_narrow / shared_scale, -1.0, 1.0).astype(np.float32)
            label_norm = np.clip(clean_wide / shared_scale, -1.0, 1.0).astype(np.float32)

            # Add noise AFTER normalisation — Modification 1
            noise_level = noise_levels[section_idx % len(noise_levels)]
            section_id = f"{well_name}_section_{section_idx:02d}_noise_{noise_level:.2f}"
            if noise_level > 0.0:
                noise = rng.normal(0.0, noise_level, size=input_norm.shape).astype(np.float32)
                input_norm = np.clip(input_norm + noise, -1.0, 1.0).astype(np.float32)
            input_min = label_min = -shared_scale
            input_max = label_max = shared_scale

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
        "lowcut_freq": LOWCUT_FREQ,
        "num_sections": len(section_metadata),
        "num_patches": int(inputs.shape[0]),
        "num_train": int(split["train_inputs"].shape[0]),
        "num_val": int(split["val_inputs"].shape[0]),
        "sections": section_metadata,
    })

    # ── Figures ──
    first_key = sorted(section_inputs)[0]

    # 1) Section-level comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(section_inputs[first_key], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title(f"{first_key} narrow input (shared norm, noise={args.noise_levels})")
    axes[1].imshow(section_labels[first_key], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title(f"{first_key} wide label")
    for ax in axes:
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_synthetic_section_example.png", dpi=300)
    plt.close(fig)

    # 2) Single trace overlay (pick middle trace from first section)
    mid_trace = args.section_width // 2
    trace_narrow = section_inputs[first_key][:, mid_trace]
    trace_label = section_labels[first_key][:, mid_trace]
    time_axis = np.arange(trace_narrow.size) * DT

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, trace_narrow, "b-", lw=1.2, label="Narrow input")
    ax.plot(time_axis, trace_label, "r--", lw=1.2, label="Wide label")
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"{first_key} — single trace overlay (mid trace)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_single_trace_overlay.png", dpi=300)
    plt.close(fig)

    # 3) Amplitude spectrum comparison
    from signal_utils import average_amplitude_spectrum
    freqs_n, amp_n = average_amplitude_spectrum(section_inputs[first_key], DT)
    freqs_w, amp_w = average_amplitude_spectrum(section_labels[first_key], DT)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs_n, amp_n, "b-", lw=2, label="Narrow input")
    ax.plot(freqs_w, amp_w, "r-", lw=2, label="Wide label")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"{first_key} — amplitude spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_spectrum_compare.png", dpi=300)
    plt.close(fig)

    # 4) Patch-level example
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
