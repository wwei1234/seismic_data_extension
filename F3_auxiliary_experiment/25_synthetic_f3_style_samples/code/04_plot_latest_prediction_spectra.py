"""
Plot spectra for the latest F3 prediction outputs.

Inputs are the four arrays written by 02_predict_f3.py:
    - residual prediction
    - final wide prediction
    - low-pass/narrow input
    - wide-band reference
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"
DT = 0.004
HIGH_BAND = (50.0, 90.0)


SERIES = (
    ("residual_prediction", "Prediction residual", "tab:orange", "-"),
    ("wide_prediction", "Final prediction", "tab:red", "-"),
    ("narrow_input", "Low-pass data", "tab:blue", "-"),
    ("wide_reference", "Wide-band data", "black", "--"),
)


def spectrum_stats(freqs, amp):
    peak_idx = int(np.argmax(amp))
    return {
        "peak_frequency_hz": float(freqs[peak_idx]),
        "high_freq_ratio_50_90": high_freq_ratio(freqs, amp),
    }


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


def high_freq_ratio(freqs, amp, band=HIGH_BAND):
    mask_high = (freqs >= band[0]) & (freqs <= band[1])
    mask_all = (freqs >= 1.0) & (freqs <= band[1])
    return float(np.sum(amp[mask_high]) / (np.sum(amp[mask_all]) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="kriging_v1")
    parser.add_argument("--fmax", type=float, default=120.0)
    args = parser.parse_args()

    spectra = {}
    stats = {}
    for suffix, label, color, style in SERIES:
        path = DATA_DIR / f"{args.prefix}_{suffix}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        freqs, amp = average_cube_spectrum(path, DT)
        spectra[suffix] = {
            "path": path,
            "label": label,
            "color": color,
            "style": style,
            "freqs": freqs,
            "amp": amp,
        }
        stats[suffix] = spectrum_stats(freqs, amp)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGURE_DIR / f"{args.prefix}_four_spectra.png"
    out_csv = FIGURE_DIR / f"{args.prefix}_four_spectra.csv"
    out_stats = FIGURE_DIR / f"{args.prefix}_four_spectra_stats.csv"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for item in spectra.values():
        ax.plot(
            item["freqs"],
            item["amp"],
            linestyle=item["style"],
            color=item["color"],
            lw=2.0,
            label=item["label"],
        )
    ax.axvspan(50.0, 90.0, color="tab:red", alpha=0.10, label="50-90 Hz")
    ax.set_xlim(0, args.fmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.set_title("Latest prediction spectra")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    common_freqs = spectra[SERIES[0][0]]["freqs"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", *[label for _, label, _, _ in SERIES]])
        for i, freq in enumerate(common_freqs):
            writer.writerow([float(freq), *[float(spectra[suffix]["amp"][i]) for suffix, _, _, _ in SERIES]])

    with out_stats.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "path", "peak_frequency_hz", "high_freq_ratio_50_90"])
        for suffix, label, _, _ in SERIES:
            writer.writerow([
                label,
                spectra[suffix]["path"],
                stats[suffix]["peak_frequency_hz"],
                stats[suffix]["high_freq_ratio_50_90"],
            ])

    print(f"Saved figure: {out_png}")
    print(f"Saved spectra CSV: {out_csv}")
    print(f"Saved stats CSV: {out_stats}")
    for suffix, label, _, _ in SERIES:
        print(
            f"{label}: peak={stats[suffix]['peak_frequency_hz']:.2f} Hz, "
            f"50-90Hz ratio={stats[suffix]['high_freq_ratio_50_90']:.4f}"
        )


if __name__ == "__main__":
    main()
