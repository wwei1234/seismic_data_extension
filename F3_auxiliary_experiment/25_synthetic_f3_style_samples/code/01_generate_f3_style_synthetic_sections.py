import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter1d
from scipy.signal import fftconvolve


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parents[0]
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from segy_reader import read_segy  # noqa: E402
from signal_utils import average_amplitude_spectrum, trapezoid_band, zero_phase_filter_section  # noqa: E402


DT = 0.004
SHOTNUM = 651
NARROW_BAND = (0.0, 0.0, 30.0, 35.0)
WIDE_BANDS = ((3.0, 6.0, 55.0, 70.0), (3.0, 6.0, 65.0, 80.0), (3.0, 6.0, 75.0, 90.0))
RESIDUAL_BAND = (35.0, 45.0, 70.0, 95.0)
CROSSLINE_MIN = 300
CROSSLINE_MAX = 1250
SECTION_WIDTH = CROSSLINE_MAX - CROSSLINE_MIN + 1
RANDOM_SEED = 42
STRUCTURE_START_SAMPLE = 120
STRUCTURE_FULL_SAMPLE = 240
F3_STAT_PATCH_COUNT = 24

SEGY_PATH = WORKSPACE_ROOT / "Rawdata" / "Seismic_data.sgy"
SOURCE_DATA_DIR = WORKSPACE_ROOT / "shared_data"
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"


def ensure_dirs():
    for path in (DATA_DIR, FIGURE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def normalize_wavelet(wavelet):
    wavelet = np.asarray(wavelet, dtype=np.float64)
    wavelet -= wavelet.mean()
    scale = np.max(np.abs(wavelet))
    if scale > 0:
        wavelet /= scale
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
    return normalize_wavelet(wavelet_full[center - half:center + half + 1])


def build_wavelet_bank():
    raw = np.load(SOURCE_DATA_DIR / "well_wide_wavelets.npy", allow_pickle=True)
    ref_len = raw.shape[1]
    return [
        {"name": f"bspline_{int(b[2])}_{int(b[3])}hz", "band": b, "wavelet": build_bspline_wavelet(ref_len, DT, b)}
        for b in WIDE_BANDS
    ]


def convolve_reflectivity(section, wavelet):
    out = np.zeros_like(section, dtype=np.float32)
    for ix in range(section.shape[1]):
        out[:, ix] = fftconvolve(section[:, ix], wavelet, mode="same")
    return out


def apply_shift_field(section, shift_2d):
    n_time, width = section.shape
    t_axis = np.arange(n_time, dtype=np.float32)
    result = np.zeros_like(section, dtype=np.float32)
    for ix in range(width):
        sample_pos = t_axis + shift_2d[:, ix]
        result[:, ix] = np.interp(sample_pos, t_axis, section[:, ix], left=0.0, right=0.0).astype(np.float32)
    return result


def section_metrics(section):
    section = np.asarray(section, dtype=np.float64)
    gx = np.gradient(section, axis=1)
    gt = np.gradient(section, axis=0)
    gx_abs = float(np.mean(np.abs(gx)))
    gt_abs = float(np.mean(np.abs(gt)))
    x0 = section[:, :-1].ravel()
    x1 = section[:, 1:].ravel()
    xlag1 = float(np.corrcoef(x0, x1)[0, 1]) if np.std(x0) > 0 and np.std(x1) > 0 else 0.0
    return {
        "std": float(np.std(section)),
        "p99_abs": float(np.percentile(np.abs(section), 99)),
        "xlag1_corr": xlag1,
        "grad_t_abs": gt_abs,
        "grad_x_abs": gx_abs,
        "grad_x_over_t": gx_abs / max(gt_abs, 1e-12),
    }


def median_metrics(metrics):
    keys = metrics[0].keys()
    return {key: float(np.median([m[key] for m in metrics])) for key in keys}


def build_multiwell_section(reflectivities, matches, subset, rng):
    ordered = sorted(subset, key=lambda w: matches[w]["crossline"])
    positions = np.array([matches[w]["crossline"] - CROSSLINE_MIN for w in ordered], dtype=np.float64)
    positions = np.clip(positions, 0.0, SECTION_WIDTH - 1.0)
    n_time = min(len(reflectivities[w]) for w in ordered)
    xq = np.arange(SECTION_WIDTH, dtype=np.float64)
    section = np.zeros((n_time, SECTION_WIDTH), dtype=np.float32)
    for it in range(n_time):
        vals = np.array([reflectivities[w][it] for w in ordered], dtype=np.float64)
        section[it] = np.interp(xq, positions, vals, left=vals[0], right=vals[-1])

    # Avoid hard vertical blocks outside/near sparse well control by blending in
    # weak, laterally coherent reflectivity texture. This keeps the section
    # synthetic while making its statistics closer to real F3 low-pass data.
    random_refl = rng.normal(0.0, 1.0, section.shape).astype(np.float32)
    random_refl = gaussian_filter(random_refl, sigma=(2.5, 30.0), mode="reflect")
    random_refl *= (0.25 * np.std(section) / (np.std(random_refl) + 1e-8))
    x_norm = np.arange(SECTION_WIDTH, dtype=np.float32)
    dist_to_well = np.min(np.abs(x_norm[:, None] - positions[None, :]), axis=1)
    sparse_weight = np.clip((dist_to_well - 35.0) / 220.0, 0.0, 0.45)
    left_edge_weight = np.clip((positions[0] + 30.0 - x_norm) / 120.0, 0.0, 0.55)
    right_edge_weight = np.clip((x_norm - positions[-1] + 30.0) / 160.0, 0.0, 0.55)
    sparse_weight = np.maximum(sparse_weight, np.maximum(left_edge_weight, right_edge_weight))
    section = (1.0 - sparse_weight[None, :]) * section + sparse_weight[None, :] * random_refl
    section = uniform_filter1d(section, size=3, axis=1)
    return section, {"ordered_wells": ordered, "positions": positions.tolist()}


def structure_time_weight(n_time, start, full):
    t = np.arange(n_time, dtype=np.float32)
    if full <= start:
        return (t >= start).astype(np.float32)
    return np.clip((t - start) / float(full - start), 0.0, 1.0).astype(np.float32)


def add_structural_perturbation(section, rng):
    """Add F3-like middle/deep structures using linear time interpolation."""
    n_time, width = section.shape
    x = np.arange(width, dtype=np.float32)
    xc = (width - 1) / 2.0
    t_axis = np.arange(n_time, dtype=np.float32)
    weight_t = structure_time_weight(n_time, STRUCTURE_START_SAMPLE, STRUCTURE_FULL_SAMPLE)

    shift_x = np.zeros(width, dtype=np.float32)
    slope = rng.uniform(-0.028, 0.028)
    curvature = rng.uniform(-14.0, 14.0) * ((x - xc) / max(xc, 1.0)) ** 2
    shift_x += slope * (x - xc) + curvature

    folds = []
    for _ in range(int(rng.integers(2, 5))):
        amp = rng.uniform(4.0, 14.0)
        period = rng.uniform(260.0, 820.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        shift_x += amp * np.sin(2.0 * np.pi * x / period + phase)
        folds.append({"amp": float(amp), "period": float(period), "phase": float(phase)})

    uplifts = []
    for _ in range(int(rng.integers(1, 3))):
        cx = rng.uniform(0.18 * width, 0.86 * width)
        sx = rng.uniform(90.0, 260.0)
        amp = rng.uniform(-14.0, 14.0)
        shift_x += amp * np.exp(-0.5 * ((x - cx) / sx) ** 2)
        uplifts.append({"center_x": float(cx), "sigma_x": float(sx), "amp": float(amp)})

    lateral_jitter = gaussian_filter1d(rng.normal(0.0, 1.2, size=width), sigma=12.0)
    shift_x += lateral_jitter.astype(np.float32)
    shift_2d = weight_t[:, None] * shift_x[None, :]

    fault_info = []
    fault_count = int(rng.integers(1, 3))
    for _ in range(fault_count):
        t0 = int(rng.integers(STRUCTURE_FULL_SAMPLE, max(STRUCTURE_FULL_SAMPLE + 1, n_time - 80)))
        x0 = float(rng.uniform(0.18 * width, 0.82 * width))
        dip = float(rng.uniform(-0.55, 0.55))
        throw = float(rng.uniform(3.0, 9.0) * rng.choice([-1, 1]))
        tt = np.arange(n_time, dtype=np.float32)
        fault_x = x0 + dip * (tt - t0)
        active = tt >= t0
        side = x[None, :] >= fault_x[:, None]
        taper = np.clip((tt - t0) / 70.0, 0.0, 1.0)
        shift_2d += (active[:, None] & side) * taper[:, None] * throw
        fault_info.append({"t0": t0, "x0": x0, "dip": dip, "throw": throw})

    result = apply_shift_field(section, shift_2d)

    result = uniform_filter1d(result, size=5, axis=1)
    return result.astype(np.float32), {
        "slope": float(slope),
        "curvature_min": float(np.min(curvature)),
        "curvature_max": float(np.max(curvature)),
        "folds": folds,
        "uplifts": uplifts,
        "faults": fault_info,
        "structure_start_sample": int(STRUCTURE_START_SAMPLE),
        "structure_full_sample": int(STRUCTURE_FULL_SAMPLE),
        "time_interpolation": "linear",
        "lateral_smoothing_size": 5,
    }


def add_seismic_horizon_step_offsets(section, rng, min_sample=300):
    """Add visible segmented small reflector offsets in seismic-image domain."""
    n_time, width = section.shape
    x = np.arange(width, dtype=np.float32)
    t_axis = np.arange(n_time, dtype=np.float32)
    shift_2d = np.zeros_like(section, dtype=np.float32)
    attenuation = np.ones_like(section, dtype=np.float32)

    start = min(max(min_sample, 0), n_time - 1)
    deep_abs = np.mean(np.abs(section[start:]), axis=1)
    candidate_t = np.where(deep_abs > np.percentile(deep_abs, 66))[0] + start
    step_info = []
    if candidate_t.size == 0:
        return section.astype(np.float32), step_info

    band_count = int(rng.integers(8, 13))
    selected_t = rng.choice(candidate_t, size=min(band_count, candidate_t.size), replace=False)
    for center_t in selected_t:
        band_half = float(rng.uniform(5.0, 12.0))
        band_gate = np.exp(-0.5 * ((t_axis - float(center_t)) / band_half) ** 2)
        segment_count = int(rng.integers(9, 17))
        for _ in range(segment_count):
            x0 = float(rng.uniform(0.06 * width, 0.94 * width))
            seg_width = float(rng.uniform(14.0, 52.0))
            throw = float(rng.uniform(1.2, 4.6) * rng.choice([-1, 1]))
            phase_jitter = float(rng.uniform(-1.4, 1.4))
            amp_drop = float(rng.uniform(0.08, 0.26))
            edge = float(rng.uniform(1.5, 4.0))
            left = 0.5 * (1.0 + np.tanh((x - (x0 - 0.5 * seg_width)) / edge))
            right = 0.5 * (1.0 - np.tanh((x - (x0 + 0.5 * seg_width)) / edge))
            seg_gate = left * right
            local_gate = band_gate[:, None] * seg_gate[None, :]
            shift_2d += local_gate * (throw + phase_jitter)
            attenuation *= (1.0 - amp_drop * local_gate).astype(np.float32)
            step_info.append({
                "center_t": int(center_t),
                "band_half": band_half,
                "x0": x0,
                "width": seg_width,
                "throw": throw,
                "phase_jitter": phase_jitter,
                "amp_drop": amp_drop,
            })

    shifted = apply_shift_field(section, shift_2d)
    shifted *= np.clip(attenuation, 0.55, 1.0)
    shifted = uniform_filter1d(shifted, size=2, axis=1)
    return shifted.astype(np.float32), step_info


def valid_f3_patch(cube, geometry, rng, target_shape):
    nt, nx = target_shape
    for _ in range(100):
        il_idx = int(rng.integers(0, cube.shape[0]))
        section = cube[il_idx].astype(np.float32)
        rms = np.sqrt(np.mean(section.astype(np.float64) ** 2, axis=0))
        valid = np.where(rms > 0.02 * np.percentile(rms[rms > 0], 90))[0] if np.any(rms > 0) else np.array([])
        if valid.size < nx:
            continue
        start_min = int(valid[0])
        start_max = int(valid[-1] - nx + 1)
        if start_max < start_min:
            continue
        x0 = int(rng.integers(start_min, start_max + 1))
        patch = section[:nt, x0:x0 + nx]
        if patch.shape == target_shape:
            low = zero_phase_filter_section(patch, DT, NARROW_BAND)
            return low, {
                "inline": int(geometry["inlines"][il_idx]),
                "trace_start": x0,
                "trace_stop": x0 + nx,
            }
    raise RuntimeError("Could not sample a valid F3 statistics patch.")


def sample_f3_style_statistics(cube, geometry, rng, target_shape, count=F3_STAT_PATCH_COUNT):
    patches = []
    infos = []
    metrics = []
    for _ in range(count):
        patch, info = valid_f3_patch(cube, geometry, rng, target_shape)
        patches.append(patch.astype(np.float32))
        infos.append(info)
        metrics.append(section_metrics(patch))
    return {
        "patches": patches,
        "sources": infos,
        "median_metrics": median_metrics(metrics),
    }


def f3_style_fields(f3_low, rng):
    gx = gaussian_filter(np.gradient(f3_low, axis=1), sigma=(2.0, 5.0))
    gt = gaussian_filter(np.gradient(f3_low, axis=0), sigma=(2.0, 5.0))
    dip = -gx / (np.abs(gt) + 0.08 * np.percentile(np.abs(gt), 90) + 1e-6)
    dip = np.clip(gaussian_filter(dip, sigma=(5.0, 16.0)), -0.75, 0.75).astype(np.float32)

    envelope = np.sqrt(gaussian_filter(f3_low.astype(np.float64) ** 2, sigma=(12.0, 32.0)))
    envelope /= np.percentile(envelope, 90) + 1e-8
    envelope = np.clip(envelope, 0.15, 1.7).astype(np.float32)

    coherency = np.sqrt(gx ** 2 + gt ** 2)
    coherency = gaussian_filter(coherency, sigma=(3.0, 10.0))
    coherency /= np.percentile(coherency, 90) + 1e-8
    coherency = np.clip(coherency, 0.25, 1.45).astype(np.float32)

    noise = f3_low - gaussian_filter(f3_low, sigma=(1.1, 2.2))
    noise = zero_phase_filter_section(noise, DT, (35.0, 45.0, 95.0, 115.0))
    noise /= np.percentile(np.abs(noise), 99) + 1e-8

    discontinuity = gaussian_filter(rng.random(f3_low.shape).astype(np.float32), sigma=(3.0, 9.0))
    discontinuity = np.where(discontinuity > np.percentile(discontinuity, 35), 1.0, 0.45).astype(np.float32)
    discontinuity = gaussian_filter(discontinuity, sigma=(1.0, 2.0))
    return dip, envelope, coherency, noise.astype(np.float32), discontinuity.astype(np.float32)


def f3_texture_from_patch(f3_low):
    fine = f3_low - gaussian_filter(f3_low, sigma=(2.2, 9.0))
    mid = gaussian_filter(f3_low, sigma=(0.6, 1.4)) - gaussian_filter(f3_low, sigma=(5.0, 18.0))
    texture = 0.55 * fine + 0.45 * mid
    texture /= np.percentile(np.abs(texture), 99) + 1e-8
    return texture.astype(np.float32)


def add_f3_statistical_texture(section, f3_low, target_metrics, rng):
    metrics_before = section_metrics(section)
    target_xlag = target_metrics["xlag1_corr"]
    target_grad_ratio = target_metrics["grad_x_over_t"]
    xlag_gap = max(0.0, metrics_before["xlag1_corr"] - target_xlag)
    grad_gap = max(0.0, target_grad_ratio - metrics_before["grad_x_over_t"])

    p99 = np.percentile(np.abs(section), 99) + 1e-8
    texture = f3_texture_from_patch(f3_low)
    random_texture = gaussian_filter(rng.normal(0.0, 1.0, section.shape), sigma=(0.8, 2.2))
    random_texture /= np.percentile(np.abs(random_texture), 99) + 1e-8

    strength = np.clip(0.10 + 0.75 * xlag_gap + 0.55 * grad_gap, 0.10, 0.32)
    shallow_gate = 0.65 + 0.35 * structure_time_weight(section.shape[0], 80, 240)
    work = section + p99 * strength * shallow_gate[:, None] * (0.72 * texture + 0.28 * random_texture)

    envelope = np.sqrt(gaussian_filter(f3_low.astype(np.float64) ** 2, sigma=(7.0, 22.0)))
    envelope /= np.percentile(envelope, 90) + 1e-8
    amp_mod = np.clip(0.82 + 0.28 * envelope, 0.65, 1.18)
    mottled = gaussian_filter(rng.normal(0.0, 1.0, section.shape), sigma=(8.0, 18.0))
    mottled /= np.percentile(np.abs(mottled), 95) + 1e-8
    amp_mod *= np.clip(1.0 + 0.10 * mottled, 0.78, 1.22)
    work *= amp_mod.astype(np.float32)

    metrics_after = section_metrics(work)
    return work.astype(np.float32), {
        "target_xlag1_corr": float(target_xlag),
        "target_grad_x_over_t": float(target_grad_ratio),
        "before": metrics_before,
        "after": metrics_after,
        "texture_strength": float(strength),
    }


def warp_along_dip(section, dip, rng, strength=1.0):
    nt, nx = section.shape
    t_axis = np.arange(nt, dtype=np.float32)
    out = np.zeros_like(section, dtype=np.float32)
    col_shift = np.cumsum(np.median(dip, axis=0))
    col_shift -= col_shift.mean()
    col_shift = gaussian_filter1d(col_shift, sigma=22.0)
    local = gaussian_filter(rng.normal(0.0, 1.0, section.shape), sigma=(16.0, 34.0))
    local *= 2.2 / (np.std(local) + 1e-8)
    shifts = np.clip(strength * (0.5 * dip + 0.25 * col_shift[None, :] + local), -9.0, 9.0)
    for ix in range(nx):
        out[:, ix] = np.interp(t_axis + shifts[:, ix], t_axis, section[:, ix], left=0.0, right=0.0)
    return out


def apply_f3_style_to_synthetic(reflectivity, f3_low, rng):
    dip, envelope, coherency, noise, discontinuity = f3_style_fields(f3_low, rng)
    styled_refl = warp_along_dip(reflectivity, dip, rng, strength=0.65)
    styled_refl = uniform_filter1d(styled_refl, size=7, axis=1)
    styled_refl *= envelope * (0.75 + 0.25 * coherency)
    styled_refl *= discontinuity
    broad_texture = gaussian_filter(rng.normal(0.0, 1.0, reflectivity.shape), sigma=(2.0, 18.0))
    broad_texture *= 0.035 * np.percentile(np.abs(styled_refl), 99) / (np.percentile(np.abs(broad_texture), 99) + 1e-8)
    styled_refl += broad_texture.astype(np.float32)
    styled_refl += 0.018 * np.percentile(np.abs(styled_refl), 99) * noise
    return styled_refl.astype(np.float32), {
        "envelope_p90": float(np.percentile(envelope, 90)),
        "coherency_p90": float(np.percentile(coherency, 90)),
        "noise_p99": float(np.percentile(np.abs(noise), 99)),
    }


def plot_section(out_path, title, panels):
    comparable = [panels["synthetic_lowpass"], panels["synthetic_wide"], panels["f3_lowpass_style_reference"]]
    clip = max(float(np.percentile(np.abs(np.concatenate([x.ravel() for x in comparable])), 99)), 1e-8)
    res_clip = max(float(np.percentile(np.abs(panels["residual_label"]), 99)), 1e-8)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    plot_items = [
        ("synthetic_lowpass", "Synthetic low-pass input", clip),
        ("synthetic_wide", "Synthetic wide label", clip),
        ("residual_label", "Residual label", res_clip),
        ("f3_lowpass_style_reference", "F3 low-pass style reference", clip),
    ]
    for ax, (key, label, limit) in zip(axes, plot_items):
        ax.imshow(panels[key], cmap="seismic", aspect="auto", vmin=-limit, vmax=limit)
        ax.set_title(label)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_deep_zoom(out_path, title, panels, t0=285, t1=455, x0=0, x1=951):
    comparable = [panels["synthetic_lowpass"][t0:t1, x0:x1], panels["synthetic_wide"][t0:t1, x0:x1]]
    clip = max(float(np.percentile(np.abs(np.concatenate([x.ravel() for x in comparable])), 99)), 1e-8)
    res_clip = max(float(np.percentile(np.abs(panels["residual_label"][t0:t1, x0:x1]), 99)), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2), sharey=True)
    for ax, (key, label, limit) in zip(axes, [
        ("synthetic_lowpass", "Low-pass zoom", clip),
        ("synthetic_wide", "Wide zoom", clip),
        ("residual_label", "Residual zoom", res_clip),
    ]):
        ax.imshow(panels[key][t0:t1, x0:x1], cmap="seismic", aspect="auto", vmin=-limit, vmax=limit)
        ax.set_title(label)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.suptitle(title + f" | zoom t={t0}:{t1}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_spectra(out_path, title, panels):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, style in [
        ("synthetic_lowpass", "Synthetic low-pass input", "b-"),
        ("synthetic_wide", "Synthetic wide label", "r-"),
        ("residual_label", "Residual label", "g-"),
        ("f3_lowpass_style_reference", "F3 low-pass style reference", "k--"),
    ]:
        freqs, amp = average_amplitude_spectrum(panels[key], DT)
        ax.plot(freqs, amp, style, label=label)
    ax.axvspan(35, 90, color="tab:red", alpha=0.10, label="35-90 Hz")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    for path in (DATA_DIR, FIGURE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    reflectivities = np.load(SOURCE_DATA_DIR / "well_reflectivities.npy", allow_pickle=True).item()
    matches = np.load(SOURCE_DATA_DIR / "well_trace_matches.npy", allow_pickle=True).item()
    wavelet_bank = build_wavelet_bank()
    cube, geometry = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)

    well_names = list(reflectivities)
    combos = [
        ("2well", list(combinations(well_names, 2))[0]),
        ("3well", list(combinations(well_names, 3))[1]),
        ("4well", tuple(well_names)),
    ]
    summary = {}
    for combo_name, subset in combos:
        reflectivity, info = build_multiwell_section(reflectivities, matches, subset, rng)
        reflectivity, structure_info = add_structural_perturbation(reflectivity, rng)
        f3_low, f3_info = valid_f3_patch(cube, geometry, rng, reflectivity.shape)
        f3_stats = sample_f3_style_statistics(cube, geometry, rng, reflectivity.shape)
        styled_refl, style_info = apply_f3_style_to_synthetic(reflectivity, f3_low, rng)
        wavelet = wavelet_bank[int(rng.integers(0, len(wavelet_bank)))]
        synthetic_wide = convolve_reflectivity(styled_refl, wavelet["wavelet"])
        synthetic_wide, texture_info = add_f3_statistical_texture(
            synthetic_wide, f3_low, f3_stats["median_metrics"], rng
        )
        synthetic_wide, horizon_step_info = add_seismic_horizon_step_offsets(synthetic_wide, rng, min_sample=300)
        synthetic_lowpass = zero_phase_filter_section(synthetic_wide, DT, NARROW_BAND)
        residual_label = zero_phase_filter_section(synthetic_wide - synthetic_lowpass, DT, RESIDUAL_BAND)

        scale = np.percentile(np.abs(synthetic_lowpass), 99) + 1e-8
        panels = {
            "synthetic_lowpass": np.clip(synthetic_lowpass / scale, -1, 1),
            "synthetic_wide": np.clip(synthetic_wide / scale, -1, 1),
            "residual_label": np.clip(residual_label / scale, -1, 1),
            "f3_lowpass_style_reference": np.clip(f3_low / (np.percentile(np.abs(f3_low), 99) + 1e-8), -1, 1),
        }
        np.save(DATA_DIR / f"{combo_name}_synthetic_lowpass.npy", panels["synthetic_lowpass"].astype(np.float32))
        np.save(DATA_DIR / f"{combo_name}_synthetic_wide.npy", panels["synthetic_wide"].astype(np.float32))
        np.save(DATA_DIR / f"{combo_name}_residual_label.npy", panels["residual_label"].astype(np.float32))
        np.save(DATA_DIR / f"{combo_name}_f3_lowpass_style_reference.npy", panels["f3_lowpass_style_reference"].astype(np.float32))

        title = f"F3-style synthetic sample | {combo_name} | wells={','.join(subset)} | {wavelet['name']}"
        plot_section(FIGURE_DIR / f"{combo_name}_f3_style_synthetic_section.png", title, panels)
        plot_deep_zoom(FIGURE_DIR / f"{combo_name}_f3_style_deep_step_zoom.png", title, panels)
        plot_spectra(FIGURE_DIR / f"{combo_name}_f3_style_synthetic_spectra.png", title, panels)
        summary[combo_name] = {
            "wells": list(subset),
            "well_info": info,
            "structure_info": structure_info,
            "horizon_step_offsets": horizon_step_info,
            "f3_style_source": f3_info,
            "f3_style_stat_sources": f3_stats["sources"],
            "f3_style_target_metrics": f3_stats["median_metrics"],
            "style_info": style_info,
            "texture_matching_info": texture_info,
            "final_metrics": {
                "synthetic_lowpass": section_metrics(panels["synthetic_lowpass"]),
                "synthetic_wide": section_metrics(panels["synthetic_wide"]),
                "residual_label": section_metrics(panels["residual_label"]),
                "f3_lowpass_style_reference": section_metrics(panels["f3_lowpass_style_reference"]),
            },
            "wavelet": wavelet["name"],
            "narrow_band": NARROW_BAND,
            "residual_band": RESIDUAL_BAND,
            "input_is_real_f3_lowpass": False,
            "uses_f3_lowpass_as_statistics_only": True,
        }
        print(f"generated {combo_name}")

    (FIGURE_DIR / "f3_style_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metric_rows = []
    for combo_name, combo_summary in summary.items():
        target = combo_summary["f3_style_target_metrics"]
        for panel_name, metrics in combo_summary["final_metrics"].items():
            row = {"combo": combo_name, "panel": panel_name}
            row.update(metrics)
            row["target_xlag1_corr"] = target["xlag1_corr"]
            row["target_grad_x_over_t"] = target["grad_x_over_t"]
            metric_rows.append(row)
    metric_path = FIGURE_DIR / "f3_style_metric_comparison.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)


if __name__ == "__main__":
    main()
