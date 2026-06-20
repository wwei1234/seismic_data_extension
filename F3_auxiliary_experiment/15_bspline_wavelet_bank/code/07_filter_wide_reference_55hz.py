"""
Filter the actual wide-band F3 reference data around/above 55 Hz.

Outputs two filtered cubes:
  - 55-75 Hz band-pass result
  - >=55 Hz high-pass result
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"
DT = 0.004


def trapezoid_band(freqs, f1, f2, f3, f4):
    filt = np.zeros_like(freqs, dtype=np.float64)
    up = (freqs >= f1) & (freqs < f2)
    keep = (freqs >= f2) & (freqs <= f3)
    down = (freqs > f3) & (freqs <= f4)
    if f2 > f1:
        filt[up] = (freqs[up] - f1) / (f2 - f1)
    filt[keep] = 1.0
    if f4 > f3:
        filt[down] = (f4 - freqs[down]) / (f4 - f3)
    return np.clip(filt, 0.0, 1.0)


def highpass(freqs, cut1, cut2):
    filt = np.ones_like(freqs, dtype=np.float64)
    filt[freqs < cut1] = 0.0
    ramp = (freqs >= cut1) & (freqs < cut2)
    if cut2 > cut1:
        filt[ramp] = (freqs[ramp] - cut1) / (cut2 - cut1)
    return np.clip(filt, 0.0, 1.0)


def filter_section(section, dt, filt):
    spec = np.fft.rfft(np.asarray(section, dtype=np.float64), axis=0)
    return np.fft.irfft(spec * filt[:, None], n=section.shape[0], axis=0).astype(np.float32)


def average_spectrum_from_cube(path, dt=DT):
    cube = np.load(path, mmap_mode="r")
    n_inline, nt, _ = cube.shape
    amp_sum = None
    trace_count = 0
    for idx in range(n_inline):
        section = np.asarray(cube[idx], dtype=np.float64)
        section = section - section.mean(axis=0, keepdims=True)
        amp = np.abs(np.fft.rfft(section, axis=0)).sum(axis=1)
        amp_sum = amp if amp_sum is None else amp_sum + amp
        trace_count += section.shape[1]
    amp = amp_sum / max(trace_count, 1)
    amp = amp / (np.max(amp) + 1e-12)
    freqs = np.fft.rfftfreq(nt, dt)
    return freqs, amp


def plot_sections(prefix, wide, band_cube, high_cube, out_path):
    mid = wide.shape[0] // 2
    panels = [
        (wide[mid], "Wide-band reference"),
        (band_cube[mid], "Band-pass 55-75 Hz"),
        (high_cube[mid], "High-pass >=55 Hz"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (data, title) in zip(axes, panels):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.suptitle(prefix)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_spectra(series, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, path, color in series:
        freqs, amp = average_spectrum_from_cube(path, DT)
        ax.plot(freqs, amp, color=color, lw=2, label=label)
    ax.axvspan(55, 75, color="tab:red", alpha=0.12, label="55-75 Hz")
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.set_title("Filtered wide-band reference spectra")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="kriging_v1")
    parser.add_argument("--band-low", type=float, default=55.0)
    parser.add_argument("--band-high", type=float, default=75.0)
    parser.add_argument("--transition", type=float, default=3.0)
    args = parser.parse_args()

    wide_path = DATA_DIR / f"{args.prefix}_wide_reference.npy"
    wide = np.load(wide_path, mmap_mode="r")
    n_inline, nt, nx = wide.shape
    freqs = np.fft.rfftfreq(nt, DT)

    band_filter = trapezoid_band(
        freqs,
        args.band_low - args.transition,
        args.band_low,
        args.band_high,
        args.band_high + args.transition,
    )
    high_filter = highpass(freqs, args.band_low - args.transition, args.band_low)

    band_path = DATA_DIR / f"{args.prefix}_wide_reference_bp55_75.npy"
    high_path = DATA_DIR / f"{args.prefix}_wide_reference_hp55.npy"
    band_cube = np.lib.format.open_memmap(
        band_path, mode="w+", dtype=np.float32, shape=(n_inline, nt, nx)
    )
    high_cube = np.lib.format.open_memmap(
        high_path, mode="w+", dtype=np.float32, shape=(n_inline, nt, nx)
    )

    for il_idx in range(n_inline):
        band_cube[il_idx] = filter_section(wide[il_idx], DT, band_filter)
        high_cube[il_idx] = filter_section(wide[il_idx], DT, high_filter)
        print(f"Filtered inline {il_idx + 1}/{n_inline}", flush=True)

    band_cube.flush()
    high_cube.flush()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    section_png = FIGURE_DIR / f"{args.prefix}_wide_reference_55hz_filtered_sections.png"
    spectrum_png = FIGURE_DIR / f"{args.prefix}_wide_reference_55hz_filtered_spectra.png"
    plot_sections(args.prefix, wide, band_cube, high_cube, section_png)
    plot_spectra(
        [
            ("Original wide reference", wide_path, "black"),
            ("Band-pass 55-75 Hz", band_path, "tab:red"),
            ("High-pass >=55 Hz", high_path, "tab:purple"),
        ],
        spectrum_png,
    )

    print(f"Saved band-pass cube: {band_path}")
    print(f"Saved high-pass cube: {high_path}")
    print(f"Saved section figure: {section_png}")
    print(f"Saved spectrum figure: {spectrum_png}")


if __name__ == "__main__":
    main()
