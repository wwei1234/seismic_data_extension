"""
Low-pass filter the actual wide-band F3 reference below 55 Hz.

Filter:
  - keep frequencies <= 55 Hz
  - taper 55-58 Hz to zero by default
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"
DT = 0.004


def lowpass(freqs, keep_high, stop_high):
    filt = np.ones_like(freqs, dtype=np.float64)
    filt[freqs > stop_high] = 0.0
    ramp = (freqs > keep_high) & (freqs <= stop_high)
    if stop_high > keep_high:
        filt[ramp] = (stop_high - freqs[ramp]) / (stop_high - keep_high)
    return np.clip(filt, 0.0, 1.0)


def filter_section(section, filt):
    spec = np.fft.rfft(np.asarray(section, dtype=np.float64), axis=0)
    return np.fft.irfft(spec * filt[:, None], n=section.shape[0], axis=0).astype(np.float32)


def average_spectrum_from_cube(path):
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
    freqs = np.fft.rfftfreq(nt, DT)
    return freqs, amp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="kriging_v1")
    parser.add_argument("--cutoff", type=float, default=55.0)
    parser.add_argument("--transition", type=float, default=3.0)
    args = parser.parse_args()

    wide_path = DATA_DIR / f"{args.prefix}_wide_reference.npy"
    wide = np.load(wide_path, mmap_mode="r")
    n_inline, nt, nx = wide.shape
    freqs = np.fft.rfftfreq(nt, DT)
    filt = lowpass(freqs, args.cutoff, args.cutoff + args.transition)

    out_path = DATA_DIR / f"{args.prefix}_wide_reference_lp55.npy"
    low_cube = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(n_inline, nt, nx)
    )

    for il_idx in range(n_inline):
        low_cube[il_idx] = filter_section(wide[il_idx], filt)
        print(f"Filtered inline {il_idx + 1}/{n_inline}", flush=True)
    low_cube.flush()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    section_png = FIGURE_DIR / f"{args.prefix}_wide_reference_lp55_section_compare.png"
    spectrum_png = FIGURE_DIR / f"{args.prefix}_wide_reference_lp55_spectrum_compare.png"

    mid = n_inline // 2
    panels = [
        (wide[mid], "Original wide-band reference"),
        (low_cube[mid], "Low-pass <=55 Hz"),
        (wide[mid] - low_cube[mid], "Removed >55 Hz component"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (data, title) in zip(axes, panels):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(section_png, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for label, path, color in [
        ("Original wide-band reference", wide_path, "black"),
        ("Low-pass <=55 Hz", out_path, "tab:blue"),
    ]:
        sf, amp = average_spectrum_from_cube(path)
        ax.plot(sf, amp, color=color, lw=2, label=label)
    ax.axvline(args.cutoff, color="tab:red", lw=1.5, linestyle="--", label="55 Hz")
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.set_title("Wide-band reference low-pass spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(spectrum_png, dpi=300)
    plt.close(fig)

    print(f"Saved low-pass cube: {out_path}")
    print(f"Saved section figure: {section_png}")
    print(f"Saved spectrum figure: {spectrum_png}")


if __name__ == "__main__":
    main()
