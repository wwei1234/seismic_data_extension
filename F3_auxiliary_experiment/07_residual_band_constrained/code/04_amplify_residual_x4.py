import argparse
import sys
from pathlib import Path

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
    corr = float(cov / np.sqrt(max(var_p, 1e-12) * max(var_t, 1e-12)))
    psnr = float(20.0 * np.log10(peak / max(rmse, 1e-12)))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "PSNR": psnr, "Correlation": corr}


def write_amplified_prediction(prefix, output_prefix, gain):
    narrow_path = DATA_DIR / f"{prefix}_narrow_input.npy"
    residual_path = DATA_DIR / f"{prefix}_residual_prediction.npy"
    reference_path = DATA_DIR / f"{prefix}_wide_reference.npy"

    narrow = np.load(narrow_path, mmap_mode="r")
    residual = np.load(residual_path, mmap_mode="r")

    out_residual_path = DATA_DIR / f"{output_prefix}_residual_prediction.npy"
    out_prediction_path = DATA_DIR / f"{output_prefix}_wide_prediction.npy"
    out_residual = np.lib.format.open_memmap(
        out_residual_path, mode="w+", dtype=np.float32, shape=residual.shape
    )
    out_prediction = np.lib.format.open_memmap(
        out_prediction_path, mode="w+", dtype=np.float32, shape=narrow.shape
    )

    for idx in range(narrow.shape[0]):
        amplified = np.asarray(residual[idx], dtype=np.float32) * gain
        out_residual[idx] = amplified
        out_prediction[idx] = np.asarray(narrow[idx], dtype=np.float32) + amplified
        if (idx + 1) % 50 == 0 or idx == 0 or idx + 1 == narrow.shape[0]:
            print(f"Amplified inline {idx + 1}/{narrow.shape[0]}")

    del out_residual
    del out_prediction
    return narrow_path, out_prediction_path, reference_path


def evaluate_and_plot(prefix, output_prefix, narrow_path, prediction_path, reference_path, gain):
    narrow = np.load(narrow_path, mmap_mode="r")
    prediction = np.load(prediction_path, mmap_mode="r")
    reference = np.load(reference_path, mmap_mode="r")
    original_prediction_path = DATA_DIR / f"{prefix}_wide_prediction.npy"

    freq_n, amp_n = average_cube_spectrum(narrow_path, DT)
    freq_p, amp_p = average_cube_spectrum(prediction_path, DT)
    freq_t, amp_t = average_cube_spectrum(reference_path, DT)
    freq_o, amp_o = average_cube_spectrum(original_prediction_path, DT)

    results = {
        "gain": gain,
        "narrow_vs_reference": streaming_metrics(narrow, reference),
        "x4_prediction_vs_reference": streaming_metrics(prediction, reference),
        "spectrum": {
            "narrow_high_freq_ratio_35_80": high_freq_ratio(freq_n, amp_n),
            "x4_prediction_high_freq_ratio_35_80": high_freq_ratio(freq_p, amp_p),
            "reference_high_freq_ratio_35_80": high_freq_ratio(freq_t, amp_t),
            "x4_prediction_spectrum_l1": spectrum_l1(freq_p, amp_p, freq_t, amp_t),
            "narrow_spectrum_l1": spectrum_l1(freq_n, amp_n, freq_t, amp_t),
        },
    }
    np.save(DATA_DIR / f"{output_prefix}_metrics.npy", results)

    print("=" * 70)
    print(f"Residual x{gain:g} Evaluation Results")
    print("=" * 70)
    print()
    print("Low-pass input vs F3 reference:")
    for key, value in results["narrow_vs_reference"].items():
        print(f"  {key}: {value:.4f}")
    print()
    print(f"Residual x{gain:g} prediction vs F3 reference:")
    for key, value in results["x4_prediction_vs_reference"].items():
        print(f"  {key}: {value:.4f}")
    print()
    print("35-80 Hz high-frequency energy ratio:")
    for key, value in results["spectrum"].items():
        if "ratio" in key:
            print(f"  {key}: {value:.4f}")
    print()
    print("Spectrum L1 distance (vs F3 reference):")
    for key, value in results["spectrum"].items():
        if "l1" in key.lower():
            print(f"  {key}: {value:.4f}")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(freq_n, amp_n, "b-", lw=2, label="Low-pass input")
    ax.plot(freq_o, amp_o, color="orange", lw=2, label="Original prediction")
    ax.plot(freq_p, amp_p, "r-", lw=2, label=f"Residual x{gain:g} prediction")
    ax.plot(freq_t, amp_t, "k--", lw=2, label="F3 reference")
    ax.axvspan(HIGH_BAND[0], HIGH_BAND[1], color="red", alpha=0.12, label="35-80 Hz")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"Residual x{gain:g} prediction spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"04_x{gain:g}_spectrum_compare.png", dpi=300)
    plt.close(fig)

    mid = prediction.shape[0] // 2
    original = np.load(original_prediction_path, mmap_mode="r")
    panels = [
        (narrow[mid], "Low-pass input"),
        (original[mid], "Original prediction"),
        (prediction[mid], f"Residual x{gain:g} prediction"),
        (reference[mid], "F3 reference"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (data, title) in zip(axes, panels):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"04_x{gain:g}_section_compare.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="residual_band_f3")
    parser.add_argument("--output-prefix", default="residual_band_f3_x4")
    parser.add_argument("--gain", type=float, default=4.0)
    args = parser.parse_args()

    ensure_dirs()
    paths = write_amplified_prediction(args.prefix, args.output_prefix, args.gain)
    evaluate_and_plot(args.prefix, args.output_prefix, *paths, args.gain)


if __name__ == "__main__":
    main()
