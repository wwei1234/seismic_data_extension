"""Rematch F3 wells to geometry-aware SEG-Y traces and estimate wavelets.

This script intentionally does not reuse cached well traces from older runs.
It scans SEG-Y trace headers, maps wellhead coordinates to the nearest trace,
extracts traces by inline/crossline, and re-estimates per-well wavelets from
the current geometry-correct reader.
"""

import os
import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

sys.path.append(str(Path(__file__).resolve().parent))

from config import DATA_DIR, DT, FIGURE_DIR, NARROW_BAND, SEGY_PATH
from segy_reader import (
    _read_segy_basic_info,
    _sample_dtype,
    read_trace_by_inline_crossline,
)
from signal_utils import average_amplitude_spectrum, trapezoid_band


WAVELET_LEN = 41
WELL_LOCATIONS = {
    "F02-1": (606554.0, 6080126.0),
    "F03-2": (619101.0, 6089491.0),
    "F03-4": (623256.0, 6082586.0),
    "F06-1": (607903.0, 6077213.0),
}


def _coord_scalar(raw_scalar):
    if raw_scalar == 0:
        return 1.0
    if raw_scalar > 0:
        return float(raw_scalar)
    return 1.0 / abs(float(raw_scalar))


def scan_trace_headers(segy_path):
    info = _read_segy_basic_info(segy_path)
    n = info["trace_num"]
    inlines = np.empty(n, dtype=np.int32)
    crosslines = np.empty(n, dtype=np.int32)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)

    with open(segy_path, "rb") as f:
        for trace_idx in range(n):
            offset = 3600 + trace_idx * info["trace_size"]
            f.seek(offset)
            header = f.read(240)
            scalar = _coord_scalar(struct.unpack(">h", header[70:72])[0])
            inlines[trace_idx] = struct.unpack(">i", header[188:192])[0]
            crosslines[trace_idx] = struct.unpack(">i", header[192:196])[0]
            xs[trace_idx] = struct.unpack(">i", header[180:184])[0] * scalar
            ys[trace_idx] = struct.unpack(">i", header[184:188])[0] * scalar

    return {
        "info": info,
        "inlines": inlines,
        "crosslines": crosslines,
        "x": xs,
        "y": ys,
    }


def nearest_trace(header_table, x, y):
    dx = header_table["x"] - float(x)
    dy = header_table["y"] - float(y)
    dist2 = dx * dx + dy * dy
    idx = int(np.argmin(dist2))
    return {
        "trace_index": idx,
        "inline": int(header_table["inlines"][idx]),
        "crossline": int(header_table["crosslines"][idx]),
        "x": float(header_table["x"][idx]),
        "y": float(header_table["y"][idx]),
        "distance_m": float(np.sqrt(dist2[idx])),
    }


def zero_phase_filter_1d(trace, dt, band):
    trace = np.asarray(trace, dtype=np.float64)
    freqs = np.fft.rfftfreq(trace.size, dt)
    spec = np.fft.rfft(trace - np.mean(trace))
    filt = trapezoid_band(freqs, *band)
    return np.fft.irfft(spec * filt, n=trace.size).astype(np.float32)


def center_crop_wavelet(wavelet, length=WAVELET_LEN):
    wavelet = np.asarray(wavelet, dtype=np.float64)
    peak = int(np.argmax(np.abs(wavelet)))
    half = length // 2
    padded = np.pad(wavelet, (length, length), mode="constant")
    center = peak + length
    cropped = padded[center - half:center + half + 1]
    cropped -= np.mean(cropped)
    max_abs = np.max(np.abs(cropped))
    if max_abs > 0:
        cropped = cropped / max_abs
    return cropped.astype(np.float32)


def estimate_wavelet(reflectivity, trace, eps=1e-3):
    r = np.asarray(reflectivity, dtype=np.float64)
    s = np.asarray(trace, dtype=np.float64)
    n = max(r.size, s.size)
    r = r[:n] - np.nanmean(r[:n])
    s = s[:n] - np.nanmean(s[:n])
    r = r / (np.std(r) + 1e-8)
    s = s / (np.std(s) + 1e-8)

    rf = np.fft.rfft(r)
    sf = np.fft.rfft(s)
    denom = np.abs(rf) ** 2 + eps * np.max(np.abs(rf) ** 2)
    wf = sf * np.conj(rf) / np.maximum(denom, 1e-12)
    wavelet = np.fft.irfft(wf, n=n)
    return center_crop_wavelet(np.fft.fftshift(wavelet), WAVELET_LEN)


def tie_score(reflectivity, wavelet, trace):
    synthetic = fftconvolve(reflectivity, wavelet, mode="same")
    a = synthetic - np.mean(synthetic)
    b = trace - np.mean(trace)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom), synthetic.astype(np.float32)


def orient_wavelet_to_trace(reflectivity, wavelet, trace):
    score, synthetic = tie_score(reflectivity, wavelet, trace)
    if score < 0.0:
        wavelet = -wavelet
        score, synthetic = tie_score(reflectivity, wavelet, trace)
    return wavelet.astype(np.float32), score, synthetic


def plot_well_diagnostics(well_name, reflectivity, trace, wavelet, narrow_wavelet, synthetic, score):
    t = np.arange(trace.size) * DT
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(trace / (np.max(np.abs(trace)) + 1e-8), t, "k", lw=1.0, label="trace")
    axes[0].plot(synthetic / (np.max(np.abs(synthetic)) + 1e-8), t, "r", lw=1.0, label="synthetic")
    axes[0].invert_yaxis()
    axes[0].set_title(f"{well_name} tie corr={score:.3f}")
    axes[0].set_xlabel("normalized amplitude")
    axes[0].set_ylabel("time/s")
    axes[0].legend()

    wt = (np.arange(wavelet.size) - wavelet.size // 2) * DT
    axes[1].plot(wt, wavelet, "r", lw=1.5, label="wide")
    axes[1].plot(wt, narrow_wavelet, "b", lw=1.5, label="narrow")
    axes[1].set_title("estimated wavelets")
    axes[1].set_xlabel("time/s")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    f_trace, a_trace = average_amplitude_spectrum(trace[:, None], DT)
    f_syn, a_syn = average_amplitude_spectrum(synthetic[:, None], DT)
    f_w, a_w = average_amplitude_spectrum(wavelet[:, None], DT)
    axes[2].plot(f_trace, a_trace, "k", lw=1.0, label="trace")
    axes[2].plot(f_syn, a_syn, "r", lw=1.0, label="synthetic")
    axes[2].plot(f_w, a_w, "C2", lw=1.0, label="wavelet")
    axes[2].set_xlim(0, 120)
    axes[2].set_title("spectra")
    axes[2].set_xlabel("Hz")
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"01_{well_name}_rematch_wavelet.png", dpi=300)
    plt.close(fig)


def plot_location_map(header_table, matches):
    fig, ax = plt.subplots(figsize=(7, 7))
    step = max(1, len(header_table["x"]) // 20000)
    ax.scatter(header_table["x"][::step], header_table["y"][::step], s=1, c="0.75", label="F3 traces")
    for well, match in matches.items():
        wx, wy = WELL_LOCATIONS[well]
        ax.plot(wx, wy, "rx", ms=8)
        ax.plot(match["x"], match["y"], "bo", ms=4)
        ax.plot([wx, match["x"]], [wy, match["y"]], "r-", lw=0.8)
        ax.text(wx, wy, well, fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Wellhead to nearest SEG-Y trace rematch")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_well_trace_rematch_map.png", dpi=300)
    plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    reflectivities = np.load(DATA_DIR / "well_reflectivities.npy", allow_pickle=True).item()
    well_names = sorted(reflectivities)
    header_table = scan_trace_headers(SEGY_PATH)

    well_traces = {}
    wavelet_pairs = {}
    tie_scores = {}
    matches = {}
    wide_wavelets = []
    narrow_wavelets = []

    for well in well_names:
        match = nearest_trace(header_table, *WELL_LOCATIONS[well])
        matches[well] = match
        trace = read_trace_by_inline_crossline(SEGY_PATH, match["inline"], match["crossline"])
        trace = trace.astype(np.float32)
        refl = np.asarray(reflectivities[well], dtype=np.float32)

        if trace.size != refl.size:
            n = min(trace.size, refl.size)
            trace = trace[:n]
            refl = refl[:n]

        wide_wavelet = estimate_wavelet(refl, trace)
        wide_wavelet, score, synthetic = orient_wavelet_to_trace(refl, wide_wavelet, trace)
        narrow_wavelet = zero_phase_filter_1d(wide_wavelet, DT, NARROW_BAND)
        narrow_wavelet = center_crop_wavelet(narrow_wavelet, WAVELET_LEN)

        well_traces[well] = trace
        wide_wavelets.append(wide_wavelet)
        narrow_wavelets.append(narrow_wavelet)
        wavelet_pairs[well] = {
            "wide": wide_wavelet,
            "narrow": narrow_wavelet,
            "estimated": wide_wavelet,
            "trace": trace,
            "reflectivity": refl,
            "match": match,
        }
        tie_scores[well] = {
            "correlation": score,
            **match,
        }
        plot_well_diagnostics(well, refl, trace, wide_wavelet, narrow_wavelet, synthetic, score)

        print(
            f"{well}: inline={match['inline']}, crossline={match['crossline']}, "
            f"distance={match['distance_m']:.1f} m, corr={score:.3f}",
            flush=True,
        )

    wide_wavelets = np.stack(wide_wavelets).astype(np.float32)
    narrow_wavelets = np.stack(narrow_wavelets).astype(np.float32)
    np.save(DATA_DIR / "well_traces.npy", well_traces)
    np.save(DATA_DIR / "well_estimated_wavelets.npy", wide_wavelets)
    np.save(DATA_DIR / "well_wide_wavelets.npy", wide_wavelets)
    np.save(DATA_DIR / "well_narrow_wavelets.npy", narrow_wavelets)
    np.save(DATA_DIR / "estimated_wide_wavelet.npy", np.mean(wide_wavelets, axis=0))
    np.save(DATA_DIR / "estimated_narrow_wavelet.npy", np.mean(narrow_wavelets, axis=0))
    np.save(DATA_DIR / "well_wavelet_pairs.npy", wavelet_pairs)
    np.save(DATA_DIR / "well_wavelet_tie_scores.npy", tie_scores)
    np.save(DATA_DIR / "well_trace_matches.npy", matches)
    plot_location_map(header_table, matches)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
