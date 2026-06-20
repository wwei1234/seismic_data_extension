"""Plot spectra for the 2/3/4-well synthetic section examples."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"
OUT_DIR = FIGURE_DIR / "合成剖面"
DT = 0.004


def average_amplitude_spectrum(section, dt=DT):
    section = np.asarray(section, dtype=np.float64)
    work = section - section.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(work, axis=0)
    amp = np.abs(spec).mean(axis=1)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    amp = amp / (np.max(amp) + 1e-12)
    return freqs, amp


def band_mean(freqs, amp, low, high):
    mask = (freqs >= low) & (freqs < high)
    return float(np.mean(amp[mask])) if np.any(mask) else 0.0


def high_freq_ratio(freqs, amp, low=35.0, high=80.0):
    high_mask = (freqs >= low) & (freqs <= high)
    all_mask = (freqs >= 1.0) & (freqs <= high)
    return float(np.sum(amp[high_mask]) / (np.sum(amp[all_mask]) + 1e-12))


def first_section_by_well_count(metadata, n_wells):
    return next(s for s in metadata["sections"] if int(s["n_wells"]) == n_wells)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = np.load(DATA_DIR / "synthetic_section_inputs.npy", allow_pickle=True).item()
    labels = np.load(DATA_DIR / "synthetic_section_labels.npy", allow_pickle=True).item()
    metadata = np.load(DATA_DIR / "synthetic_metadata.npy", allow_pickle=True).item()

    rows = []
    combined = {}
    for n_wells in (2, 3, 4):
        meta = first_section_by_well_count(metadata, n_wells)
        sid = meta["section_id"]
        series = {
            "lowpass_input": inputs[sid],
            "wide_label": labels[sid],
            "residual_label": labels[sid] - inputs[sid],
        }
        spectra = {}
        for name, section in series.items():
            freqs, amp = average_amplitude_spectrum(section)
            spectra[name] = (freqs, amp)
            peak = float(freqs[int(np.argmax(amp))])
            row = {
                "n_wells": n_wells,
                "section_id": sid,
                "wavelet_name": meta["wavelet_name"],
                "noise_level": meta["noise_level"],
                "series": name,
                "peak_hz": peak,
                "ratio_35_80": high_freq_ratio(freqs, amp),
                "mean_0_10": band_mean(freqs, amp, 0, 10),
                "mean_10_25": band_mean(freqs, amp, 10, 25),
                "mean_25_35": band_mean(freqs, amp, 25, 35),
                "mean_35_55": band_mean(freqs, amp, 35, 55),
                "mean_55_80": band_mean(freqs, amp, 55, 80),
                "mean_80_120": band_mean(freqs, amp, 80, 120),
            }
            rows.append(row)
        combined[n_wells] = (meta, spectra)

        fig, ax = plt.subplots(figsize=(10, 5))
        styles = {
            "lowpass_input": ("tab:blue", "-"),
            "wide_label": ("tab:red", "-"),
            "residual_label": ("tab:green", "-"),
        }
        for name, (freqs, amp) in spectra.items():
            color, style = styles[name]
            ax.plot(freqs, amp, color=color, linestyle=style, lw=2.0, label=name)
        ax.axvspan(35, 80, color="tab:red", alpha=0.10, label="35-80 Hz")
        ax.set_xlim(0, 120)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized average amplitude")
        ax.set_title(f"{n_wells}-well synthetic spectra | {sid} | {meta['wavelet_name']}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{n_wells}well_synthetic_spectra.png", dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5), sharey=True)
    for ax, n_wells in zip(axes, (2, 3, 4)):
        meta, spectra = combined[n_wells]
        for name, (freqs, amp) in spectra.items():
            color, style = {
                "lowpass_input": ("tab:blue", "-"),
                "wide_label": ("tab:red", "-"),
                "residual_label": ("tab:green", "-"),
            }[name]
            ax.plot(freqs, amp, color=color, linestyle=style, lw=1.8, label=name)
        ax.axvspan(35, 80, color="tab:red", alpha=0.10)
        ax.set_xlim(0, 120)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{n_wells}-well | {meta['wavelet_name']}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Normalized average amplitude")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "synthetic_section_spectra_compare.png", dpi=300)
    plt.close(fig)

    csv_path = OUT_DIR / "synthetic_section_spectra_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['n_wells']}well {row['series']}: "
            f"peak={row['peak_hz']:.2f}Hz, 35-80 ratio={row['ratio_35_80']:.4f}, "
            f"55-80 mean={row['mean_55_80']:.4f}"
        )
    print(f"Saved spectra to {OUT_DIR}")


if __name__ == "__main__":
    main()
