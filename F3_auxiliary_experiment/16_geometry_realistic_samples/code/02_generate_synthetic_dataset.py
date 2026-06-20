"""Generate geometry-realistic residual-label samples for experiment 16.

Outputs keep the same file names used by training, but train_labels/val_labels
are high-frequency residual targets: normalized wide - normalized low-pass.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))
sys.path.append(str(CODE_DIR))

from config import (
    BSPLINE_WAVELET_BANDS,
    CROSSLINE_MIN,
    DATA_DIR,
    DT,
    ENVELOPE_SMOOTH_T,
    ENVELOPE_SMOOTH_X,
    FIGURE_DIR,
    LOWCUT_FREQ,
    NARROW_BAND,
    NOISE_LEVELS,
    PATCH_SIZE,
    PATCH_STRIDE,
    PROFILES_PER_COMBO,
    Q_FILTER_Q,
    Q_FILTER_STRENGTH,
    RANDOM_SEED,
    SEGY_PATH,
    SHOTNUM,
    SOURCE_DATA_DIR,
    STRUCTURE_FULL_SAMPLE,
    STRUCTURE_START_SAMPLE,
    SYNTHETIC_SECTION_WIDTH,
    USE_Q_FILTER,
)
from signal_utils import (
    apply_amplitude_envelope,
    apply_time_variant_q_filter_section,
    average_amplitude_spectrum,
    convolve_reflectivity,
    extract_amplitude_envelope,
    zero_phase_filter_section,
)
from segy_reader import read_segy


def patch_starts(n, patch_size, stride, cover_last=True):
    if n < patch_size:
        raise ValueError(f"Section size {n} < patch size {patch_size}.")
    starts = list(range(0, n - patch_size + 1, stride))
    if cover_last and starts[-1] != n - patch_size:
        starts.append(n - patch_size)
    return starts


def normalise_wavelet(wavelet):
    wavelet = np.asarray(wavelet, dtype=np.float64)
    wavelet -= np.mean(wavelet)
    max_abs = np.max(np.abs(wavelet))
    if max_abs > 0:
        wavelet = wavelet / max_abs
    return wavelet.astype(np.float32)


def build_bspline_wavelet(length, dt, band):
    f1, f2, f3, f4 = band
    nyq = 0.5 / dt
    n_fft = 512
    freqs = np.fft.rfftfreq(n_fft, dt)
    x = np.array([0.0, f1, f2, (f2 + f3) * 0.5, f3, f4, nyq], dtype=np.float64)
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    spline = make_interp_spline(x, y, k=3, bc_type="clamped")
    amp = np.clip(spline(freqs), 0.0, 1.0)
    wavelet_full = np.fft.fftshift(np.fft.irfft(amp, n=n_fft))
    center = n_fft // 2
    half = length // 2
    return normalise_wavelet(wavelet_full[center - half:center + half + 1])


def build_wavelet_bank(raw_wavelets):
    reference = normalise_wavelet(np.mean(np.stack(list(raw_wavelets.values())), axis=0))
    bank = []
    for band in BSPLINE_WAVELET_BANDS:
        name = f"bspline_{int(band[2])}_{int(band[3])}hz"
        bank.append({
            "name": name,
            "wavelet": build_bspline_wavelet(reference.size, DT, band),
            "band": band,
        })
    return bank


def plot_wavelet_bank_spectra(bank):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    t = (np.arange(bank[0]["wavelet"].size) - bank[0]["wavelet"].size // 2) * DT
    for item in bank:
        wavelet = item["wavelet"]
        axes[0].plot(t, wavelet, lw=1.4, label=item["name"])
        freqs, amp = average_amplitude_spectrum(wavelet[:, None], DT)
        axes[1].plot(freqs, amp, lw=1.4, label=item["name"])
    axes[0].set_title("Wavelet bank")
    axes[0].set_xlabel("Time/s")
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Wavelet-bank spectra")
    axes[1].set_xlabel("Hz")
    axes[1].set_xlim(0, 120)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_wavelet_bank_spectra.png", dpi=300)
    plt.close(fig)


def build_multiwell_section_linear_projected(reflectivities, well_names_subset, matches, width):
    """Linear multi-well interpolation using actual F3 crossline projection."""
    ordered = sorted(well_names_subset, key=lambda w: matches[w]["crossline"])
    positions = np.array(
        [float(matches[w]["crossline"] - CROSSLINE_MIN) for w in ordered],
        dtype=np.float64,
    )
    positions = np.clip(positions, 0.0, float(width - 1))
    n_time = min(len(reflectivities[w]) for w in ordered)
    x_query = np.arange(width, dtype=np.float64)
    section = np.zeros((n_time, width), dtype=np.float32)
    for it in range(n_time):
        values = np.array([reflectivities[w][it] for w in ordered], dtype=np.float64)
        section[it] = np.interp(
            x_query, positions, values, left=values[0], right=values[-1]
        )
    return section, {
        "ordered_wells": ordered,
        "well_positions": [float(x) for x in positions],
        "well_crosslines": [int(matches[w]["crossline"]) for w in ordered],
    }


def structure_time_weight(n_time, start, full):
    t = np.arange(n_time, dtype=np.float32)
    if full <= start:
        return (t >= start).astype(np.float32)
    return np.clip((t - start) / float(full - start), 0.0, 1.0).astype(np.float32)


def add_structural_perturbation(section, rng):
    """Apply lower-section structures with linear interpolation, not nearest-neighbour."""
    n_time, width = section.shape
    x = np.arange(width, dtype=np.float32)
    xc = (width - 1) / 2.0
    t_axis = np.arange(n_time, dtype=np.float32)
    weight_t = structure_time_weight(n_time, STRUCTURE_START_SAMPLE, STRUCTURE_FULL_SAMPLE)

    shift_x = np.zeros(width, dtype=np.float32)
    slope = rng.uniform(-0.025, 0.025)
    curvature = rng.uniform(-12.0, 12.0) * ((x - xc) / max(xc, 1.0)) ** 2
    shift_x += slope * (x - xc) + curvature

    folds = []
    for _ in range(rng.integers(1, 4)):
        amp = rng.uniform(2.0, 10.0)
        period = rng.uniform(260.0, 750.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        shift_x += amp * np.sin(2.0 * np.pi * x / period + phase)
        folds.append({"amp": float(amp), "period": float(period), "phase": float(phase)})

    for _ in range(rng.integers(1, 3)):
        cx = rng.uniform(0.20 * width, 0.85 * width)
        sx = rng.uniform(100.0, 280.0)
        amp = rng.uniform(-12.0, 12.0)
        shift_x += amp * np.exp(-0.5 * ((x - cx) / sx) ** 2)

    lateral_jitter = gaussian_filter1d(rng.normal(0.0, 1.5, size=width), sigma=10.0)
    shift_x += lateral_jitter.astype(np.float32)
    shift_2d = weight_t[:, None] * shift_x[None, :]

    fault_info = []
    if rng.random() < 0.45:
        t0 = int(rng.integers(STRUCTURE_FULL_SAMPLE, max(STRUCTURE_FULL_SAMPLE + 1, n_time - 70)))
        x0 = float(rng.uniform(0.20 * width, 0.80 * width))
        dip = float(rng.uniform(-0.45, 0.45))
        throw = float(rng.uniform(3.0, 10.0) * rng.choice([-1, 1]))
        tt = np.arange(n_time, dtype=np.float32)
        fault_x = x0 + dip * (tt - t0)
        active = tt >= t0
        side = x[None, :] >= fault_x[:, None]
        taper = np.clip((tt - t0) / 60.0, 0.0, 1.0)
        shift_2d += (active[:, None] & side) * taper[:, None] * throw
        fault_info.append({"t0": t0, "x0": x0, "dip": dip, "throw": throw})

    result = np.zeros_like(section, dtype=np.float32)
    for ix in range(width):
        sample_pos = t_axis + shift_2d[:, ix]
        result[:, ix] = np.interp(
            sample_pos, t_axis, section[:, ix], left=0.0, right=0.0
        ).astype(np.float32)

    result = uniform_filter1d(result, size=5, axis=1)
    return result.astype(np.float32), {
        "slope": float(slope),
        "folds": folds,
        "faults": fault_info,
        "structure_start_sample": int(STRUCTURE_START_SAMPLE),
        "structure_full_sample": int(STRUCTURE_FULL_SAMPLE),
        "time_interpolation": "linear",
        "lateral_smoothing_size": 5,
    }


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


def plot_synthetic_section_examples(section_inputs, section_wide_labels, section_labels, section_meta):
    out_dir = FIGURE_DIR / "合成剖面"
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for meta in section_meta:
        n_wells = int(meta["n_wells"])
        if n_wells not in chosen:
            chosen[n_wells] = meta

    for n_wells in (2, 3, 4):
        if n_wells not in chosen:
            continue
        meta = chosen[n_wells]
        sid = meta["section_id"]
        narrow = section_inputs[sid]
        wide = section_wide_labels[sid]
        residual = section_labels[sid]
        panels = [
            (narrow, "Low-pass input"),
            (wide, "Wide label"),
            (residual, "Residual label"),
        ]
        shared_clip = max(
            float(np.percentile(np.abs(np.concatenate([p[0].ravel() for p in panels])), 99.0)),
            1e-8,
        )
        fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True)
        for ax, (data, title) in zip(axes, panels):
            ax.imshow(data, cmap="seismic", aspect="auto", vmin=-shared_clip, vmax=shared_clip)
            ax.set_title(title)
            ax.set_xlabel("Trace")
            ax.set_ylabel("Time sample")
            for well, xpos in zip(meta["ordered_wells"], meta["well_positions"]):
                ax.axvline(float(xpos), color="yellow", lw=1.4)
                ax.text(
                    float(xpos), 4, well,
                    color="black", fontsize=8, ha="center", va="top",
                    bbox={"facecolor": "yellow", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
                )
        fig.suptitle(
            f"{n_wells}-well synthetic section | {sid} | {meta['wavelet_name']} | noise={meta['noise_level']:.2f}"
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{n_wells}well_synthetic_section.png", dpi=300)
        plt.close(fig)


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
    well_matches = np.load(SOURCE_DATA_DIR / "well_trace_matches.npy", allow_pickle=True).item()
    well_names_all = sorted(reflectivities.keys())
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]

    print("Loading F3 for envelope estimation...")
    f3_cube = read_segy(SEGY_PATH, shotnum=SHOTNUM)
    f3_section = f3_cube[f3_cube.shape[0] // 2].astype(np.float32)
    f3_lowpass = zero_phase_filter_section(f3_section, DT, NARROW_BAND)
    f3_envelope = extract_amplitude_envelope(
        f3_lowpass, smooth_t=ENVELOPE_SMOOTH_T, smooth_x=ENVELOPE_SMOOTH_X
    )

    raw_wavelets_arr = np.load(SOURCE_DATA_DIR / "well_estimated_wavelets.npy")
    raw_wavelets = {wn: raw_wavelets_arr[i] for i, wn in enumerate(well_names_all)}
    wavelet_bank = build_wavelet_bank(raw_wavelets)
    plot_wavelet_bank_spectra(wavelet_bank)

    combos_2 = list(combinations(well_names_all, 2))
    combos_3 = list(combinations(well_names_all, 3))
    combos_4 = list(combinations(well_names_all, 4))
    all_combos = [(list(c), len(c)) for c in combos_2 + combos_3 + combos_4]
    print(
        f"Well combinations: 2w={len(combos_2)}, 3w={len(combos_3)}, "
        f"4w={len(combos_4)} -> {len(all_combos)} total"
    )

    expected_profiles_per_combo = len(wavelet_bank) * len(noise_levels)
    if args.profiles_per_combo != expected_profiles_per_combo:
        print(
            f"profiles_per_combo={args.profiles_per_combo} ignored; "
            f"using each B-spline wavelet x each noise level ({expected_profiles_per_combo})."
        )

    all_inputs, all_labels, patch_meta, section_meta = [], [], [], []
    section_inputs, section_clean_inputs, section_labels, section_wide_labels = {}, {}, {}, {}
    profile_idx = 0

    for combo_idx, (subset, n_wells) in enumerate(all_combos):
        for wavelet_idx, wavelet_item in enumerate(wavelet_bank):
            for noise_idx, noise_level in enumerate(noise_levels):
                rep = wavelet_idx * len(noise_levels) + noise_idx
                base_refl, interp_info = build_multiwell_section_linear_projected(
                    reflectivities, subset, well_matches, args.section_width
                )
                base_refl, structure = add_structural_perturbation(base_refl, rng)

                clean_wide = convolve_reflectivity(base_refl, wavelet_item["wavelet"])
                clean_narrow = zero_phase_filter_section(clean_wide, DT, NARROW_BAND)
                if USE_Q_FILTER:
                    clean_narrow = apply_time_variant_q_filter_section(
                        clean_narrow, DT, q=args.q, strength=args.q_strength
                    )
                    clean_wide = apply_time_variant_q_filter_section(
                        clean_wide, DT, q=args.q, strength=args.q_strength
                    )

                clean_wide, clean_narrow = apply_amplitude_envelope(
                    clean_wide, clean_narrow, f3_envelope, rng
                )

                shared_scale = float(np.percentile(np.abs(clean_narrow), 99))
                shared_scale = max(shared_scale, 1e-8)
                clean_input_norm = np.clip(clean_narrow / shared_scale, -1.0, 1.0).astype(np.float32)
                wide_norm = np.clip(clean_wide / shared_scale, -1.0, 1.0).astype(np.float32)
                label_norm = np.clip(wide_norm - clean_input_norm, -1.0, 1.0).astype(np.float32)
                input_norm = clean_input_norm.copy()
                if noise_level > 0.0:
                    noise = rng.normal(0.0, noise_level, size=input_norm.shape).astype(np.float32)
                    input_norm = np.clip(input_norm + noise, -1.0, 1.0).astype(np.float32)

                sid = f"c{combo_idx:02d}_w{wavelet_idx}_n{noise_level:.2f}"
                patches_x, patches_y, locs = cut_patches(
                    input_norm, label_norm, args.patch_size, args.stride
                )
                start = len(all_inputs)
                all_inputs.extend(patches_x)
                all_labels.extend(patches_y)
                for t0, x0 in locs:
                    patch_meta.append({
                        "section_id": sid,
                        "wells": subset,
                        "n_wells": n_wells,
                        "combo_idx": combo_idx,
                        "noise_level": noise_level,
                        "wavelet_name": wavelet_item["name"],
                        "t0": int(t0),
                        "x0": int(x0),
                        "input_min": -shared_scale,
                        "input_max": shared_scale,
                        "label_type": "high_frequency_residual",
                        "label_min": -shared_scale,
                        "label_max": shared_scale,
                    })

                section_inputs[sid] = input_norm.astype(np.float32)
                section_clean_inputs[sid] = clean_input_norm.astype(np.float32)
                section_wide_labels[sid] = wide_norm.astype(np.float32)
                section_labels[sid] = label_norm.astype(np.float32)
                section_meta.append({
                    "section_id": sid,
                    "wells": subset,
                    "n_wells": n_wells,
                    "combo_idx": combo_idx,
                    "noise_level": noise_level,
                    "num_patches": int(len(locs)),
                    "patch_index_start": int(start),
                    "patch_index_end": int(start + len(locs)),
                    "input_min": -shared_scale,
                    "input_max": shared_scale,
                    "label_type": "high_frequency_residual",
                    "label_min": -shared_scale,
                    "label_max": shared_scale,
                    "q": float(args.q),
                    "q_strength": float(args.q_strength),
                    "use_q_filter": bool(USE_Q_FILTER),
                    "wavelet_name": wavelet_item["name"],
                    "wavelet_band": wavelet_item["band"],
                    "ordered_wells": interp_info["ordered_wells"],
                    "well_positions": interp_info["well_positions"],
                    "well_crosslines": interp_info["well_crosslines"],
                    "interpolation": "actual_crossline_linear",
                    "structure": structure,
                })
                profile_idx += 1

    inputs_arr = np.stack(all_inputs).astype(np.float32)
    labels_arr = np.stack(all_labels).astype(np.float32)
    split = split_train_val(inputs_arr, labels_arr, patch_meta, rng)

    for name in ("train_inputs", "train_labels", "val_inputs", "val_labels"):
        np.save(DATA_DIR / f"{name}.npy", split[name])
    np.save(DATA_DIR / "synthetic_section_inputs.npy", section_inputs)
    np.save(DATA_DIR / "synthetic_section_clean_inputs.npy", section_clean_inputs)
    np.save(DATA_DIR / "synthetic_section_labels.npy", section_labels)
    np.save(DATA_DIR / "synthetic_section_wide_labels.npy", section_wide_labels)
    np.save(
        DATA_DIR / "synthetic_patch_metadata.npy",
        {"train": split["train_metadata"], "val": split["val_metadata"]},
    )
    np.save(DATA_DIR / "synthetic_metadata.npy", {
        "dt": DT,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "section_width": args.section_width,
        "crossline_min": CROSSLINE_MIN,
        "profiles_per_combo": args.profiles_per_combo,
        "actual_profiles_per_combo": expected_profiles_per_combo,
        "label_type": "high_frequency_residual",
        "normalization": "per_section_p99_abs_clean_narrow",
        "noise_levels": noise_levels,
        "q": float(args.q),
        "q_strength": float(args.q_strength),
        "use_q_filter": bool(USE_Q_FILTER),
        "lowcut_freq": args.lowcut,
        "interpolation": "actual_crossline_linear",
        "wavelet_bank": [
            {"name": item["name"], "band": item["band"], "wavelet": item["wavelet"]}
            for item in wavelet_bank
        ],
        "num_sections": len(section_meta),
        "num_patches": int(inputs_arr.shape[0]),
        "num_train": int(split["train_inputs"].shape[0]),
        "num_val": int(split["val_inputs"].shape[0]),
        "sections": section_meta,
    })

    plot_synthetic_section_examples(section_inputs, section_wide_labels, section_labels, section_meta)

    first_key = sorted(section_inputs)[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(section_inputs[first_key], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title(f"{first_key} narrow input")
    axes[1].imshow(section_labels[first_key], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title(f"{first_key} residual label")
    for ax in axes:
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_section_example.png", dpi=300)
    plt.close(fig)

    freqs_n, amp_n = average_amplitude_spectrum(section_inputs[first_key], DT)
    freqs_w, amp_w = average_amplitude_spectrum(section_wide_labels[first_key], DT)
    freqs_r, amp_r = average_amplitude_spectrum(section_labels[first_key], DT)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs_n, amp_n, "b-", lw=2, label="Narrow")
    ax.plot(freqs_w, amp_w, "r-", lw=2, label="Wide")
    ax.plot(freqs_r, amp_r, "g-", lw=2, label="Residual")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Hz")
    ax.set_title(f"{first_key} spectrum")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_spectrum.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(split["train_inputs"][0], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("Train patch input")
    axes[1].imshow(split["train_labels"][0], cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_title("Train patch label")
    for ax in axes:
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_patch_example.png", dpi=300)
    plt.close(fig)

    print(f"Saved to {DATA_DIR}")
    print(f"Profiles: {profile_idx}, patches: {inputs_arr.shape}")
    print(f"train: {split['train_inputs'].shape}, val: {split['val_inputs'].shape}")


if __name__ == "__main__":
    main()
