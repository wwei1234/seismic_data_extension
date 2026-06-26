"""
Evaluate the residual-band-constrained prediction on F3 data.

Metrics:
    Corr  – Pearson correlation coefficient
    MAE   – mean absolute error
    RMSE  – root mean square error
    50-90 Hz high-frequency energy ratio
    Spectrum L1 distance
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    DATA_DIR,
    DT,
    FIGURE_DIR,
    HIGH_BAND,
    average_cube_spectrum,
    ensure_dirs,
    high_freq_ratio,
    spectrum_l1,
)


def streaming_metrics(pred, target):
    """Compute metrics streamingly to handle large memmap arrays."""
    n = pred.shape[0]
    sum_abs = 0.0
    sum_sq = 0.0
    sum_p = 0.0
    sum_t = 0.0
    sum_p2 = 0.0
    sum_t2 = 0.0
    sum_pt = 0.0
    count = 0
    peak = 1e-8
    for idx in range(n):
        p = np.asarray(pred[idx], dtype=np.float64)
        t = np.asarray(target[idx], dtype=np.float64)
        d = p - t
        sum_abs += float(np.sum(np.abs(d)))
        sum_sq += float(np.sum(d ** 2))
        sum_p += float(np.sum(p))
        sum_t += float(np.sum(t))
        sum_p2 += float(np.sum(p ** 2))
        sum_t2 += float(np.sum(t ** 2))
        sum_pt += float(np.sum(p * t))
        count += p.size
        peak = max(peak, float(np.max(np.abs(p))), float(np.max(np.abs(t))))
    mae = sum_abs / count
    mse = sum_sq / count
    rmse = float(np.sqrt(mse))
    cov = sum_pt - sum_p * sum_t / count
    var_p = sum_p2 - sum_p ** 2 / count
    var_t = sum_t2 - sum_t ** 2 / count
    corr = float(cov / (np.sqrt(max(var_p, 1e-12) * max(var_t, 1e-12))))
    psnr = float(20.0 * np.log10(peak / max(rmse, 1e-12)))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "PSNR": psnr, "Correlation": corr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="residual_band_f3")
    args = parser.parse_args()

    ensure_dirs()

    pred_path = DATA_DIR / f"{args.prefix}_wide_prediction.npy"
    narrow_path = DATA_DIR / f"{args.prefix}_narrow_input.npy"
    target_path = DATA_DIR / f"{args.prefix}_wide_reference.npy"

    pred = np.load(pred_path, mmap_mode="r")
    narrow = np.load(narrow_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")

    # ── spectra ──
    freq_p, amp_p = average_cube_spectrum(pred_path, DT)
    freq_n, amp_n = average_cube_spectrum(narrow_path, DT)
    freq_t, amp_t = average_cube_spectrum(target_path, DT)

    # ── metrics ──
    results = {
        "narrow_vs_reference": streaming_metrics(narrow, target),
        "prediction_vs_reference": streaming_metrics(pred, target),
        "spectrum": {
            "narrow_high_freq_ratio_50_90": high_freq_ratio(freq_n, amp_n),
            "prediction_high_freq_ratio_50_90": high_freq_ratio(freq_p, amp_p),
            "reference_high_freq_ratio_50_90": high_freq_ratio(freq_t, amp_t),
            "prediction_spectrum_l1": spectrum_l1(freq_p, amp_p, freq_t, amp_t),
            "narrow_spectrum_l1": spectrum_l1(freq_n, amp_n, freq_t, amp_t),
        },
    }
    np.save(DATA_DIR / f"{args.prefix}_metrics.npy", results)

    # ── print ──
    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print()
    print("Low-pass input vs F3 reference:")
    for k, v in results["narrow_vs_reference"].items():
        print(f"  {k}: {v:.4f}")
    print()
    print("Residual-band prediction vs F3 reference:")
    for k, v in results["prediction_vs_reference"].items():
        print(f"  {k}: {v:.4f}")
    print()
    print("50-90 Hz high-frequency energy ratio:")
    for k, v in results["spectrum"].items():
        if "ratio" in k:
            print(f"  {k}: {v:.4f}")
    print()
    print("Spectrum L1 distance (vs F3 reference):")
    for k, v in results["spectrum"].items():
        if "l1" in k.lower():
            print(f"  {k}: {v:.4f}")

    # ── spectrum figure ──
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(freq_n, amp_n, "b-", lw=2, label="Low-pass input")
    ax.plot(freq_p, amp_p, "r-", lw=2, label="Residual-band prediction")
    ax.plot(freq_t, amp_t, "k--", lw=2, label="F3 reference")
    ax.axvspan(HIGH_BAND[0], HIGH_BAND[1], color="red", alpha=0.12, label="50-90 Hz")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Residual-band-constrained prediction spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_spectrum_compare.png", dpi=300)
    plt.close(fig)

    # ── section comparison figure ──
    mid = pred.shape[0] // 2
    panels = [
        (narrow[mid], "Low-pass input"),
        (pred[mid], "Residual-band prediction"),
        (target[mid], "F3 reference"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (data, title) in zip(axes, panels):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_section_compare.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
