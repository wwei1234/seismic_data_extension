"""
High-pass filter the predicted residual and add it back to the low-pass input.

This is a post-processing experiment for kriging_v1:
    residual_hp = highpass(residual_prediction)
    wide_hp = narrow_input + residual_hp
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


def highpass_section(section, dt, f1, f2):
    section = np.asarray(section, dtype=np.float64)
    spec = np.fft.rfft(section, axis=0)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    filt = np.ones_like(freqs, dtype=np.float64)
    filt[freqs < f1] = 0.0
    ramp = (freqs >= f1) & (freqs < f2)
    if f2 > f1:
        filt[ramp] = (freqs[ramp] - f1) / (f2 - f1)
    return np.fft.irfft(spec * filt[:, None], n=section.shape[0], axis=0).astype(np.float32)


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


def high_freq_ratio(freqs, amp, low=50.0, high=90.0):
    mask_high = (freqs >= low) & (freqs <= high)
    mask_all = (freqs >= 1.0) & (freqs <= high)
    return float(np.sum(amp[mask_high]) / (np.sum(amp[mask_all]) + 1e-12))


def save_stats(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "peak_frequency_hz", "high_freq_ratio_50_90"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="kriging_v1")
    parser.add_argument("--cut1", type=float, default=40.0)
    parser.add_argument("--cut2", type=float, default=50.0)
    parser.add_argument("--fmax", type=float, default=120.0)
    args = parser.parse_args()

    residual_path = DATA_DIR / f"{args.prefix}_residual_prediction.npy"
    narrow_path = DATA_DIR / f"{args.prefix}_narrow_input.npy"
    original_pred_path = DATA_DIR / f"{args.prefix}_wide_prediction.npy"
    reference_path = DATA_DIR / f"{args.prefix}_wide_reference.npy"

    residual = np.load(residual_path, mmap_mode="r")
    narrow = np.load(narrow_path, mmap_mode="r")
    n_inline, nt, nx = residual.shape

    tag = f"hp{args.cut2:g}"
    residual_hp_path = DATA_DIR / f"{args.prefix}_{tag}_residual_prediction.npy"
    wide_hp_path = DATA_DIR / f"{args.prefix}_{tag}_wide_prediction.npy"

    residual_hp = np.lib.format.open_memmap(
        residual_hp_path, mode="w+", dtype=np.float32, shape=(n_inline, nt, nx)
    )
    wide_hp = np.lib.format.open_memmap(
        wide_hp_path, mode="w+", dtype=np.float32, shape=(n_inline, nt, nx)
    )

    for il_idx in range(n_inline):
        hp = highpass_section(residual[il_idx], DT, args.cut1, args.cut2)
        residual_hp[il_idx] = hp
        wide_hp[il_idx] = narrow[il_idx] + hp
        print(f"Processed inline {il_idx + 1}/{n_inline}", flush=True)

    residual_hp.flush()
    wide_hp.flush()

    spectra = {}
    series = [
        ("Low-pass data", narrow_path, "tab:blue", "-"),
        ("Original prediction", original_pred_path, "tab:red", "-"),
        ("High-pass residual recombined", wide_hp_path, "tab:green", "-"),
        ("Wide-band data", reference_path, "black", "--"),
        ("Original residual", residual_path, "tab:orange", "-"),
        ("High-pass residual", residual_hp_path, "tab:purple", "-"),
    ]
    for label, path, color, style in series:
        freqs, amp = average_cube_spectrum(path, DT)
        spectra[label] = (freqs, amp, color, style)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    spectrum_png = FIGURE_DIR / f"{args.prefix}_{tag}_spectrum_compare.png"
    section_png = FIGURE_DIR / f"{args.prefix}_{tag}_section_compare.png"
    stats_csv = FIGURE_DIR / f"{args.prefix}_{tag}_spectrum_stats.csv"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for label in ("Low-pass data", "Original prediction", "High-pass residual recombined", "Wide-band data"):
        freqs, amp, color, style = spectra[label]
        axes[0].plot(freqs, amp, linestyle=style, color=color, lw=2, label=label)
    axes[0].axvspan(35, 80, color="tab:red", alpha=0.10)
    axes[0].set_xlim(0, args.fmax)
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("Final prediction spectrum")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Normalized average amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for label in ("Original residual", "High-pass residual"):
        freqs, amp, color, style = spectra[label]
        axes[1].plot(freqs, amp, linestyle=style, color=color, lw=2, label=label)
    axes[1].axvspan(35, 80, color="tab:red", alpha=0.10)
    axes[1].set_xlim(0, args.fmax)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Residual spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(spectrum_png, dpi=300)
    plt.close(fig)

    original_pred = np.load(original_pred_path, mmap_mode="r")
    wide_ref = np.load(reference_path, mmap_mode="r")
    mid = n_inline // 2
    panels = [
        (narrow[mid], "Low-pass data"),
        (original_pred[mid], "Original prediction"),
        (wide_hp[mid], "High-pass residual recombined"),
        (wide_ref[mid], "Wide-band data"),
        (wide_hp[mid] - original_pred[mid], "New - original prediction"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(24, 5))
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

    rows = []
    for label, (freqs, amp, _, _) in spectra.items():
        peak = float(freqs[int(np.argmax(amp))])
        rows.append([label, peak, high_freq_ratio(freqs, amp)])
        print(f"{label}: peak={peak:.2f} Hz, 50-90Hz ratio={rows[-1][2]:.4f}")
    save_stats(stats_csv, rows)

    print(f"Saved high-pass residual: {residual_hp_path}")
    print(f"Saved recombined prediction: {wide_hp_path}")
    print(f"Saved section figure: {section_png}")
    print(f"Saved spectrum figure: {spectrum_png}")
    print(f"Saved stats CSV: {stats_csv}")


if __name__ == "__main__":
    main()
