"""Generate synthetic training dataset — 13_kriging.

Key changes from 11:
  1. Multi-well kriging interpolation replaces single-well lateral stretching.
  2. F3 amplitude envelope modulation adds realistic amplitude distribution.
  3. Structural perturbation separated from reflectivity building.
  4. Normalisation uses WIDE's p99 (not narrow's).
  5. 11 well-combinations × 2 profiles × 4 noise levels = 88 profiles.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    DATA_DIR, DT, ENVELOPE_SMOOTH_T, ENVELOPE_SMOOTH_X,
    FIGURE_DIR, KRIGING_NUGGET, KRIGING_SILL, LOWCUT_FREQ,
    NARROW_BAND, NOISE_LEVELS, PATCH_SIZE, PATCH_STRIDE,
    PROFILES_PER_COMBO, Q_FILTER_Q, Q_FILTER_STRENGTH,
    RANDOM_SEED, SEGY_PATH, SHOTNUM, SOURCE_DATA_DIR,
    SYNTHETIC_SECTION_WIDTH,
)
from signal_utils import (
    apply_amplitude_envelope, apply_time_variant_q_filter_section,
    average_amplitude_spectrum, build_multiwell_section_kriging,
    convolve_reflectivity, extract_amplitude_envelope,
    zero_phase_filter_section,
)
from segy_reader import read_segy
from itertools import combinations


# ── Helpers ──────────────────────────────────────────────────────────────────


def patch_starts(n, patch_size, stride, cover_last=True):
    if n < patch_size:
        raise ValueError(f"Section size {n} < patch size {patch_size}.")
    starts = list(range(0, n - patch_size + 1, stride))
    if cover_last and starts[-1] != n - patch_size:
        starts.append(n - patch_size)
    return starts


# ── Structural perturbation ──────────────────────────────────────────────────


def add_structural_perturbation(section, rng):
    """Apply tilt, folds, faults and jitter to an existing 2D reflectivity section."""
    n_time, width = section.shape
    x = np.arange(width, dtype=np.float32); xc = (width - 1) / 2.0
    t_axis = np.arange(n_time, dtype=np.float32)

    slope = rng.uniform(-0.08, 0.08)
    curvature = rng.uniform(-35.0, 35.0) * ((x - xc) / max(xc, 1.0)) ** 2
    shifts = slope * (x - xc) + curvature

    folds = []
    for _ in range(rng.integers(2, 5)):
        fa = rng.uniform(3.0, 15.0); fp = rng.uniform(140.0, 650.0)
        fph = rng.uniform(0.0, 2.0 * np.pi)
        shifts += fa * np.sin(2.0 * np.pi * x / fp + fph)
        folds.append({"amp": float(fa), "period": float(fp), "phase": float(fph)})

    for _ in range(rng.integers(1, 4)):
        cx = rng.uniform(0.15 * width, 0.85 * width); sx = rng.uniform(70.0, 240.0)
        amp = rng.uniform(-28.0, 28.0)
        shifts += amp * np.exp(-0.5 * ((x - cx) / sx) ** 2)

    # micro-undulations
    if rng.random() < 0.3:
        ma = rng.uniform(0.3, 1.0); mp = rng.uniform(10, 50)
        shifts += ma * np.sin(2.0 * np.pi * x / mp + rng.uniform(0, 2 * np.pi))

    lateral_jitter = gaussian_filter1d(rng.normal(0.0, 3.5, size=width), sigma=7.0)
    shifts += lateral_jitter

    # faults
    fault_info = []
    if rng.random() < 0.65:
        fx = int(rng.integers(width // 4, width * 3 // 4))
        ft = float(rng.uniform(4.0, 14.0) * rng.choice([-1, 1]))
        fault_info.append({"x": fx, "throw": ft})
        shifts[x >= fx] += ft
    if rng.random() < 0.25:
        fx = int(rng.integers(width // 5, width * 4 // 5))
        ft = float(rng.uniform(3.0, 8.0) * rng.choice([-1, 1]))
        fault_info.append({"x": fx, "throw": ft})
        shifts[x >= fx] += ft

    # apply shifts with nearest-neighbour (preserves sparsity)
    result = np.zeros_like(section)
    idx_map = np.arange(n_time, dtype=np.int32)
    for ix, shift in enumerate(shifts):
        sample_pos = t_axis + shift
        idx = np.rint(sample_pos).astype(np.int32)
        valid = (idx >= 0) & (idx < n_time)
        result[valid, ix] = section[idx[valid], ix]

    result = uniform_filter1d(result, size=3, axis=1)

    return result.astype(np.float32), {
        "slope": float(slope), "folds": folds, "faults": fault_info,
    }


# ── Patch extraction ─────────────────────────────────────────────────────────


def cut_patches(input_section, label_section, patch_size, stride):
    t_starts = patch_starts(input_section.shape[0], patch_size, stride)
    x_starts = patch_starts(input_section.shape[1], patch_size, stride)
    inputs, labels, locs = [], [], []
    for t0 in t_starts:
        for x0 in x_starts:
            inputs.append(input_section[t0:t0 + patch_size, x0:x0 + patch_size])
            labels.append(label_section[t0:t0 + patch_size, x0:x0 + patch_size])
            locs.append((t0, x0))
    return np.stack(inputs), np.stack(labels), locs


def split_train_val(inputs, labels, metadata, rng, train_ratio=0.8):
    order = rng.permutation(inputs.shape[0])
    n_train = int(round(inputs.shape[0] * train_ratio))
    t_idx = order[:n_train]; v_idx = order[n_train:]
    return {
        "train_inputs": inputs[t_idx], "train_labels": labels[t_idx],
        "val_inputs": inputs[v_idx], "val_labels": labels[v_idx],
        "train_metadata": [metadata[i] for i in t_idx],
        "val_metadata": [metadata[i] for i in v_idx],
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    parser.add_argument("--section-width", type=int, default=SYNTHETIC_SECTION_WIDTH)
    parser.add_argument("--profiles-per-combo", type=int, default=PROFILES_PER_COMBO)
    parser.add_argument("--noise-levels", default=",".join(str(x) for x in NOISE_LEVELS))
    parser.add_argument("--q", type=float, default=Q_FILTER_Q)
    parser.add_argument("--q-strength", type=float, default=Q_FILTER_STRENGTH)
    parser.add_argument("--lowcut", type=float, default=LOWCUT_FREQ)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    reflectivities = np.load(SOURCE_DATA_DIR / "well_reflectivities.npy", allow_pickle=True).item()
    well_names_all = sorted(reflectivities.keys())
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]

    # ── Load F3, extract envelope, estimate range ──
    print("Loading F3 for envelope and range estimation...")
    f3_cube = read_segy(SEGY_PATH, shotnum=SHOTNUM)
    mid = f3_cube.shape[0] // 2
    f3_section = f3_cube[mid].astype(np.float32)
    f3_lowpass = zero_phase_filter_section(f3_section, DT, NARROW_BAND)
    f3_envelope = extract_amplitude_envelope(f3_lowpass,
                                              smooth_t=ENVELOPE_SMOOTH_T,
                                              smooth_x=ENVELOPE_SMOOTH_X)
    # estimate range from correlation drop-off
    corr_profile = np.array([
        float(np.corrcoef(f3_lowpass[:, 0], f3_lowpass[:, lag])[0, 1])
        for lag in range(1, min(500, f3_lowpass.shape[1]))
    ])
    range_est = int(np.argmax(corr_profile < 0.5)) + 1
    range_est = max(100, min(range_est, 500))
    print(f"Estimated lateral correlation range: {range_est} traces")

    # ── Load raw wavelets ──
    raw_wavelets_arr = np.load(SOURCE_DATA_DIR / "well_estimated_wavelets.npy")
    raw_wavelets = {wn: raw_wavelets_arr[i] for i, wn in enumerate(well_names_all)}

    # ── Build well combinations ──
    combos_2 = list(combinations(well_names_all, 2))  # 6
    combos_3 = list(combinations(well_names_all, 3))  # 4
    combos_4 = list(combinations(well_names_all, 4))  # 1
    all_combos = [(list(c), len(c)) for c in combos_2 + combos_3 + combos_4]
    print(f"Well combinations: 2w={len(combos_2)}, 3w={len(combos_3)}, 4w={len(combos_4)}"
          f" → {len(all_combos)} total")

    all_inputs, all_labels, patch_meta, section_meta = [], [], [], []
    section_inputs, section_labels = {}, {}

    profile_idx = 0
    for combo_idx, (subset, n_wells) in enumerate(all_combos):
        # Pick one well's wavelet (use first well in subset for consistency)
        well_for_wavelet = subset[0]
        wavelet = raw_wavelets[well_for_wavelet]

        for rep in range(args.profiles_per_combo):
            # Build reflectivity by kriging between selected wells
            base_refl = build_multiwell_section_kriging(
                reflectivities, subset, args.section_width, rng,
                nugget=KRIGING_NUGGET, sill=KRIGING_SILL, range_=range_est,
            )
            # Add structural perturbation
            base_refl, structure = add_structural_perturbation(base_refl, rng)

            # Convolve → Q filter
            clean_wide = convolve_reflectivity(base_refl, wavelet)
            clean_narrow = zero_phase_filter_section(clean_wide, DT, NARROW_BAND)
            clean_narrow = apply_time_variant_q_filter_section(
                clean_narrow, DT, q=args.q, strength=args.q_strength)
            clean_wide = apply_time_variant_q_filter_section(
                clean_wide, DT, q=args.q, strength=args.q_strength)

            # Amplitude envelope modulation
            clean_wide, clean_narrow = apply_amplitude_envelope(
                clean_wide, clean_narrow, f3_envelope, rng,
            )

            # Normalise with WIDE's p99
            for nl in noise_levels:
                shared_scale = float(np.percentile(np.abs(clean_wide), 99))
                shared_scale = max(shared_scale, 1e-8)
                input_norm = np.clip(clean_narrow / shared_scale, -1.0, 1.0).astype(np.float32)
                label_norm = np.clip(clean_wide / shared_scale, -1.0, 1.0).astype(np.float32)

                sid = f"c{combo_idx:02d}_r{rep}_n{nl:.2f}"
                if nl > 0.0:
                    nz = rng.normal(0.0, nl, size=input_norm.shape).astype(np.float32)
                    input_norm = np.clip(input_norm + nz, -1.0, 1.0).astype(np.float32)
                imn = iln = -shared_scale; imx = ilx = shared_scale

                px, py, locs = cut_patches(input_norm, label_norm, args.patch_size, args.stride)
                start = len(all_inputs)
                all_inputs.extend(px); all_labels.extend(py)
                for pi, (t0, x0) in enumerate(locs):
                    patch_meta.append({
                        "section_id": sid, "wells": subset, "n_wells": n_wells,
                        "combo_idx": combo_idx, "noise_level": nl,
                        "t0": int(t0), "x0": int(x0),
                        "input_min": imn, "input_max": imx,
                        "label_min": iln, "label_max": ilx,
                    })
                section_inputs[sid] = input_norm.astype(np.float32)
                section_labels[sid] = label_norm.astype(np.float32)
                section_meta.append({
                    "section_id": sid, "wells": subset, "n_wells": n_wells,
                    "combo_idx": combo_idx, "noise_level": nl,
                    "num_patches": int(len(locs)),
                    "patch_index_start": int(start),
                    "patch_index_end": int(start + len(locs)),
                    "input_min": imn, "input_max": imx,
                    "label_min": iln, "label_max": ilx,
                    "q": float(args.q), "q_strength": float(args.q_strength),
                    "structure": structure,
                })
                profile_idx += 1

    inputs_arr = np.stack(all_inputs).astype(np.float32)
    labels_arr = np.stack(all_labels).astype(np.float32)
    split = split_train_val(inputs_arr, labels_arr, patch_meta, rng)

    for name in ("train_inputs", "train_labels", "val_inputs", "val_labels"):
        np.save(DATA_DIR / f"{name}.npy", split[name])
    np.save(DATA_DIR / "synthetic_section_inputs.npy", section_inputs)
    np.save(DATA_DIR / "synthetic_section_labels.npy", section_labels)
    np.save(DATA_DIR / "synthetic_patch_metadata.npy",
            {"train": split["train_metadata"], "val": split["val_metadata"]})
    np.save(DATA_DIR / "synthetic_metadata.npy", {
        "dt": DT, "patch_size": args.patch_size, "stride": args.stride,
        "section_width": args.section_width, "profiles_per_combo": args.profiles_per_combo,
        "noise_levels": noise_levels, "q": float(args.q),
        "q_strength": float(args.q_strength), "lowcut_freq": args.lowcut,
        "kriging_nugget": KRIGING_NUGGET, "kriging_sill": KRIGING_SILL,
        "kriging_range": range_est,
        "num_sections": len(section_meta), "num_patches": int(inputs_arr.shape[0]),
        "num_train": int(split["train_inputs"].shape[0]),
        "num_val": int(split["val_inputs"].shape[0]),
        "sections": section_meta,
    })

    # ── Figures ──
    first_key = sorted(section_inputs)[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(section_inputs[first_key], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title(f"{first_key} narrow input")
    axes[1].imshow(section_labels[first_key], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title(f"{first_key} wide label")
    for ax in axes: ax.set_xlabel("Trace"); ax.set_ylabel("Time sample")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "02_section_example.png", dpi=300); plt.close(fig)

    mid_t = args.section_width // 2
    t = np.arange(section_inputs[first_key].shape[0]) * DT
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, section_inputs[first_key][:, mid_t], "b-", lw=1.2, label="Narrow")
    ax.plot(t, section_labels[first_key][:, mid_t], "r--", lw=1.2, label="Wide")
    ax.invert_yaxis(); ax.set_title(f"{first_key} — mid trace"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "02_trace_overlay.png", dpi=300); plt.close(fig)

    fn, an = average_amplitude_spectrum(section_inputs[first_key], DT)
    fw, aw = average_amplitude_spectrum(section_labels[first_key], DT)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fn, an, "b-", lw=2, label="Narrow"); ax.plot(fw, aw, "r-", lw=2, label="Wide")
    ax.set_xlim(0, 120); ax.set_ylim(0, 1.05); ax.set_xlabel("Hz")
    ax.set_title(f"{first_key} — spectrum"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "02_spectrum.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(split["train_inputs"][0], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("Train patch input")
    axes[1].imshow(split["train_labels"][0], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title("Train patch label")
    for ax in axes: ax.set_xlabel("Trace"); ax.set_ylabel("Time")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "02_patch_example.png", dpi=300); plt.close(fig)

    print(f"Saved to {DATA_DIR}")
    print(f"Profiles: {profile_idx}, patches: {inputs_arr.shape}")
    print(f"train: {split['train_inputs'].shape}, val: {split['val_inputs'].shape}")


if __name__ == "__main__":
    main()
