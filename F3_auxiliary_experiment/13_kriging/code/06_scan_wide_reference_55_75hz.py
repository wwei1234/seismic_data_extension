"""
Scan the actual wide-band F3 reference spectrum in the 55-75 Hz range.

Outputs:
  - focused amplitude spectrum plot
  - per-frequency CSV inside the target band
  - summary CSV with peak and band energy ratios
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"
DT = 0.004


def average_cube_spectrum(path, dt=DT):
    cube = np.load(path, mmap_mode="r")
    n_inline, nt, _ = cube.shape
    amp_sum = None
    trace_count = 0
    for idx in range(n_inline):
        section = np.asarray(cube[idx], dtype=np.float64)
        section = section - section.mean(axis=0, keepdims=True)
        spec = np.fft.rfft(section, axis=0)
        amp = np.abs(spec).sum(axis=1)
        amp_sum = amp if amp_sum is None else amp_sum + amp
        trace_count += section.shape[1]
    amp = amp_sum / max(trace_count, 1)
    freqs = np.fft.rfftfreq(nt, dt)
    amp = amp / (np.max(amp) + 1e-12)
    return freqs, amp


def band_ratio(freqs, amp, low, high, base_low=1.0, base_high=120.0):
    band = (freqs >= low) & (freqs <= high)
    base = (freqs >= base_low) & (freqs <= base_high)
    return float(np.sum(amp[band]) / (np.sum(amp[base]) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="kriging_v1")
    parser.add_argument("--low", type=float, default=55.0)
    parser.add_argument("--high", type=float, default=75.0)
    parser.add_argument("--context-low", type=float, default=35.0)
    parser.add_argument("--context-high", type=float, default=90.0)
    args = parser.parse_args()

    wide_path = DATA_DIR / f"{args.prefix}_wide_reference.npy"
    if not wide_path.exists():
        raise FileNotFoundError(wide_path)

    freqs, amp = average_cube_spectrum(wide_path, DT)
    band = (freqs >= args.low) & (freqs <= args.high)
    context = (freqs >= args.context_low) & (freqs <= args.context_high)
    if not np.any(band):
        raise ValueError("No FFT bins found in the requested band.")

    band_freqs = freqs[band]
    band_amp = amp[band]
    peak_idx = int(np.argmax(band_amp))
    peak_freq = float(band_freqs[peak_idx])
    peak_amp = float(band_amp[peak_idx])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.prefix}_wide_reference_{args.low:g}_{args.high:g}hz"
    plot_path = FIGURE_DIR / f"{tag}_scan.png"
    csv_path = FIGURE_DIR / f"{tag}_spectrum.csv"
    stats_path = FIGURE_DIR / f"{tag}_stats.csv"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(freqs[context], amp[context], color="black", lw=2)
    axes[0].axvspan(args.low, args.high, color="tab:red", alpha=0.14, label=f"{args.low:g}-{args.high:g} Hz")
    axes[0].scatter([peak_freq], [peak_amp], color="tab:red", zorder=3)
    axes[0].set_title("Wide-band reference spectrum context")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Normalized average amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(band_freqs, band_amp, color="tab:red", lw=2)
    axes[1].fill_between(band_freqs, band_amp, color="tab:red", alpha=0.16)
    axes[1].scatter([peak_freq], [peak_amp], color="black", zorder=3)
    axes[1].annotate(
        f"peak {peak_freq:.2f} Hz\namp {peak_amp:.3f}",
        xy=(peak_freq, peak_amp),
        xytext=(peak_freq + 1.0, peak_amp * 0.92),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    axes[1].set_xlim(args.low, args.high)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title(f"Focused {args.low:g}-{args.high:g} Hz")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "normalized_average_amplitude"])
        for freq, value in zip(band_freqs, band_amp):
            writer.writerow([float(freq), float(value)])

    rows = [
        ["wide_reference_path", str(wide_path)],
        ["band_low_hz", args.low],
        ["band_high_hz", args.high],
        ["band_peak_frequency_hz", peak_freq],
        ["band_peak_amplitude", peak_amp],
        ["band_mean_amplitude", float(np.mean(band_amp))],
        ["band_max_minus_min_amplitude", float(np.max(band_amp) - np.min(band_amp))],
        ["ratio_55_75_over_1_120", band_ratio(freqs, amp, args.low, args.high, 1.0, 120.0)],
        ["ratio_55_75_over_35_80", band_ratio(freqs, amp, args.low, args.high, 35.0, 80.0)],
    ]
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    print(f"Saved plot: {plot_path}")
    print(f"Saved band spectrum CSV: {csv_path}")
    print(f"Saved stats CSV: {stats_path}")
    print(f"55-75 Hz peak: {peak_freq:.2f} Hz, amplitude={peak_amp:.4f}")
    print(f"55-75 / 1-120 ratio: {rows[-2][1]:.4f}")
    print(f"55-75 / 35-80 ratio: {rows[-1][1]:.4f}")


if __name__ == "__main__":
    main()
