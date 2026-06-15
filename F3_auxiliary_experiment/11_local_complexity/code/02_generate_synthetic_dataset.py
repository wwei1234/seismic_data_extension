"""Generate synthetic training dataset — 11_local_complexity.

Builds on 10 (raw wavelet, narrow-norm, dipping faults, stride 128) and adds
local-scale complexity to break up overly-smooth, continuous reflectors:
  1. Stratified chaos — different time segments get different perturbation levels.
  2. Micro-undulations — occasional (30% chance) tiny-amplitude short-wavelength folds.
  3. Lateral amplitude modulation — slow trace-wise gain variation.
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
    NARROW_BAND,
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
    zero_phase_filter_section,
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


# ── Dipping fault ────────────────────────────────────────────────────────────


def _apply_dipping_fault(section, fx, throw, angle_deg):
    """Through-going dipping fault: displace samples on the down-thrown side."""
    n_time, n_trace = section.shape
    angle_rad = np.radians(angle_deg)
    cot_a = 1.0 / np.tan(angle_rad)
    aspect = n_trace / n_time * 0.7

    throw_int = int(round(abs(throw)))
    if throw_int == 0:
        return

    src = section.copy()
    section.fill(0)

    t_mid = (n_time - 1) / 2.0
    t_idx = np.arange(n_time, dtype=np.float32)
    cutoff_x = fx + (t_idx - t_mid) * cot_a * aspect

    for t in range(n_time):
        cx = cutoff_x[t]
        src_t = t - throw_int
        valid_src = 0 <= src_t < n_time
        for ix in range(n_trace):
            on_down = (throw > 0 and ix >= cx) or (throw < 0 and ix <= cx)
            if on_down and valid_src:
                section[t, ix] = src[src_t, ix]
            elif not on_down:
                section[t, ix] = src[t, ix]


# ── Structural reflectivity with local complexity ────────────────────────────


def build_structural_reflectivity(r1d, width, rng):
    r1d = np.asarray(r1d, dtype=np.float32)
    n_time = r1d.size
    t_axis = np.arange(n_time, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    xc = (width - 1) / 2.0

    # ── large-scale deformation ──
    slope = rng.uniform(-0.12, 0.12)
    curvature = rng.uniform(-35.0, 35.0) * ((x - xc) / max(xc, 1.0)) ** 2
    shifts = slope * (x - xc) + curvature

    folds = []
    for _ in range(rng.integers(2, 5)):
        fold_amp = rng.uniform(3.0, 15.0)
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

    # Feature 2: micro-undulations — reduced to ~1/10 of original
    if rng.random() < 0.3:
        micro_amp = rng.uniform(0.3, 1.0)
        micro_period = rng.uniform(10, 50)
        micro_phase = rng.uniform(0, 2.0 * np.pi)
        shifts += micro_amp * np.sin(2.0 * np.pi * x / micro_period + micro_phase)

    lateral_jitter = gaussian_filter1d(rng.normal(0.0, 3.5, size=width), sigma=7.0)
    shifts += lateral_jitter

    # ── faults ──
    fault_info = []
    if rng.random() < 0.65:
        fx = int(rng.integers(width // 4, width * 3 // 4))
        ft = float(rng.uniform(4.0, 14.0) * rng.choice([-1, 1]))
        fa = float(rng.choice([75, 60, 45, 30]))
        fault_info.append({"x": fx, "throw_samples": ft, "angle_deg": fa})
    if rng.random() < 0.25:
        fx = int(rng.integers(width // 5, width * 4 // 5))
        ft = float(rng.uniform(3.0, 8.0) * rng.choice([-1, 1]))
        fa = float(rng.choice([75, 60, 45, 30]))
        fault_info.append({"x": fx, "throw_samples": ft, "angle_deg": fa})

    # ── build section ──
    section = np.zeros((n_time, width), dtype=np.float32)
    index = np.arange(n_time, dtype=np.float32)
    for ix, shift in enumerate(shifts):
        vertical_jitter = gaussian_filter1d(rng.normal(0.0, 0.75, size=n_time), sigma=6.0)
        vertical_jitter += rng.uniform(-1.2, 1.2) * np.sin(
            2.0 * np.pi * t_axis / rng.uniform(90.0, 220.0) + rng.uniform(0.0, 2.0 * np.pi)
        )
        sample_pos = t_axis + shift + vertical_jitter
        section[:, ix] = np.interp(sample_pos, index, r1d, left=0.0, right=0.0)

    for f in fault_info:
        _apply_dipping_fault(section, f["x"], f["throw_samples"], f["angle_deg"])

    # ── Feature 1: stratified chaos ──
    n_seg = rng.integers(3, 7)
    boundaries = [0] + sorted(rng.integers(10, n_time - 10, size=n_seg).tolist()) + [n_time]
    for si in range(len(boundaries) - 1):
        t0, t1 = boundaries[si], boundaries[si + 1]
        level = rng.uniform(0.0, 0.04)       # chaos level: 0 = clean, 0.04 = messy
        if level < 0.003:
            continue                            # some layers stay clean
        noise = rng.normal(0.0, level * np.std(r1d[r1d != 0]) * 2, (t1 - t0, width))
        section[t0:t1, :] += noise.astype(np.float32)

    # ── Feature 3: lateral amplitude modulation ──
    gain = gaussian_filter1d(rng.uniform(0.7, 1.3, size=width), sigma=25)
    section = section * gain.astype(np.float32)[None, :]

    # ── lateral smoothing (after all perturbations) ──
    section = uniform_filter1d(section, size=5, axis=1)

    return section, {
        "slope": float(slope),
        "folds": folds,
        "fault": fault_info,
        "n_chaos_segments": n_seg,
        "n_scattered": None,  # filled below
    }


# ── Patch extraction ─────────────────────────────────────────────────────────


def cut_patches(input_section, label_section, patch_size, stride):
    t_starts = patch_starts(input_section.shape[0], patch_size, stride)
    x_starts = patch_starts(input_section.shape[1], patch_size, stride)
    inputs = []
    labels = []
    locs = []
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
    parser.add_argument("--sections-per-wavelet", type=int, default=SYNTHETIC_SECTIONS_PER_WAVELET)
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
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]

    raw_wavelets_arr = np.load(SOURCE_DATA_DIR / "well_estimated_wavelets.npy")
    well_names = list(reflectivities.keys())
    raw_wavelets = {wn: raw_wavelets_arr[i] for i, wn in enumerate(well_names)}

    all_inputs, all_labels, patch_meta, section_meta = [], [], [], []
    section_inputs, section_labels = {}, {}

    for well_name, reflectivity in reflectivities.items():
        wavelet = raw_wavelets[well_name]
        for section_idx in range(args.sections_per_wavelet):
            base_refl, structure = build_structural_reflectivity(
                reflectivity, args.section_width, rng
            )

            clean_wide = convolve_reflectivity(base_refl, wavelet)
            clean_narrow = zero_phase_filter_section(clean_wide, DT, NARROW_BAND)
            clean_narrow = apply_time_variant_q_filter_section(
                clean_narrow, DT, q=args.q, strength=args.q_strength)
            clean_wide = apply_time_variant_q_filter_section(
                clean_wide, DT, q=args.q, strength=args.q_strength)
            if args.lowcut > 0:
                clean_narrow = remove_lowfreq_background(clean_narrow, dt=DT, f_cut=args.lowcut)
                clean_wide = remove_lowfreq_background(clean_wide, dt=DT, f_cut=args.lowcut)

            shared_scale = float(np.percentile(np.abs(clean_narrow), 99))
            shared_scale = max(shared_scale, 1e-8)
            input_norm = np.clip(clean_narrow / shared_scale, -1.0, 1.0).astype(np.float32)
            label_norm = np.clip(clean_wide / shared_scale, -1.0, 1.0).astype(np.float32)

            nl = noise_levels[section_idx % len(noise_levels)]
            sid = f"{well_name}_section_{section_idx:02d}_noise_{nl:.2f}"
            if nl > 0.0:
                nz = rng.normal(0.0, nl, size=input_norm.shape).astype(np.float32)
                input_norm = np.clip(input_norm + nz, -1.0, 1.0).astype(np.float32)
            imn = iln = -shared_scale
            imx = ilx = shared_scale

            px, py, locs = cut_patches(input_norm, label_norm, args.patch_size, args.stride)
            start = len(all_inputs)
            all_inputs.extend(px); all_labels.extend(py)
            for pi, (t0, x0) in enumerate(locs):
                patch_meta.append({"section_id": sid, "well": well_name,
                                   "noise_level": nl, "t0": int(t0), "x0": int(x0),
                                   "input_min": imn, "input_max": imx,
                                   "label_min": iln, "label_max": ilx})
            section_inputs[sid] = input_norm.astype(np.float32)
            section_labels[sid] = label_norm.astype(np.float32)
            section_meta.append({"section_id": sid, "well": well_name,
                                 "section_index": int(section_idx), "noise_level": nl,
                                 "num_patches": int(len(locs)),
                                 "patch_index_start": int(start),
                                 "patch_index_end": int(start + len(locs)),
                                 "input_min": imn, "input_max": imx,
                                 "label_min": iln, "label_max": ilx,
                                 "q": float(args.q), "q_strength": float(args.q_strength),
                                 "structure": structure})

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
        "section_width": args.section_width,
        "sections_per_wavelet": args.sections_per_wavelet,
        "noise_levels": noise_levels, "q": float(args.q),
        "q_strength": float(args.q_strength), "lowcut_freq": args.lowcut,
        "num_sections": len(section_meta), "num_patches": int(inputs_arr.shape[0]),
        "num_train": int(split["train_inputs"].shape[0]),
        "num_val": int(split["val_inputs"].shape[0]),
        "sections": section_meta,
    })

    # ── Figures ──
    # pick a section with faults for the preview figure if possible
    faulty_sections = [sid for sid, sec in section_inputs.items()
                       if any(s["section_id"] == sid and s["structure"].get("fault")
                              for s in section_meta)]
    first_key = faulty_sections[0] if faulty_sections else sorted(section_inputs)[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(section_inputs[first_key], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title(f"{first_key} narrow input")
    axes[1].imshow(section_labels[first_key], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title(f"{first_key} wide label")
    for ax in axes: ax.set_xlabel("Trace"); ax.set_ylabel("Time sample")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "02_section_example.png", dpi=300); plt.close(fig)

    mid = args.section_width // 2
    t = np.arange(section_inputs[first_key].shape[0]) * DT
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, section_inputs[first_key][:, mid], "b-", lw=1.2, label="Narrow")
    ax.plot(t, section_labels[first_key][:, mid], "r--", lw=1.2, label="Wide")
    ax.invert_yaxis(); ax.set_title(f"{first_key} — mid trace"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "02_trace_overlay.png", dpi=300); plt.close(fig)

    from signal_utils import average_amplitude_spectrum
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
    print(f"sections: {len(section_meta)}, patches: {inputs_arr.shape}")
    print(f"train: {split['train_inputs'].shape}, val: {split['val_inputs'].shape}")


if __name__ == "__main__":
    main()
