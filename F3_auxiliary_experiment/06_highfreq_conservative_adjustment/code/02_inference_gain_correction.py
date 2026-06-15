import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from common import (
    BIAS_DATA_DIR,
    DATA_DIR,
    DT,
    FIGURE_DIR,
    HIGH_BAND,
    RESIDUAL_GAIN_BAND,
    average_cube_spectrum,
    ensure_dirs,
    high_freq_ratio,
    spectrum_l1,
    trapezoid_band,
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
    corr = float(cov / (np.sqrt(max(var_p, 1e-12) * max(var_t, 1e-12))))
    psnr = float(20.0 * np.log10(peak / max(rmse, 1e-12)))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "PSNR": psnr, "Correlation": corr}


def residual_gain_correction(residual, narrow, target_ratio, max_gain=3.0):
    wide = narrow + residual
    freqs = np.fft.rfftfreq(wide.shape[0], DT)
    spec_wide = np.fft.rfft(wide - wide.mean(axis=0, keepdims=True), axis=0)
    amp = np.mean(np.abs(spec_wide), axis=1)
    current_ratio = high_freq_ratio(freqs, amp)

    if current_ratio <= 1e-12 or current_ratio >= target_ratio:
        return residual.astype(np.float32), 1.0, current_ratio

    gain = min(float(target_ratio / current_ratio), max_gain)
    residual_spec = np.fft.rfft(residual, axis=0)
    band = trapezoid_band(freqs, 30.0, RESIDUAL_GAIN_BAND[0], RESIDUAL_GAIN_BAND[1], 85.0)
    gain_curve = 1.0 + (gain - 1.0) * band
    corrected = np.fft.irfft(residual_spec * gain_curve[:, None], n=residual.shape[0], axis=0)
    corrected -= corrected.mean(axis=0, keepdims=True)
    return corrected.astype(np.float32), gain, current_ratio


def run_for_ratio(target_ratio, max_gain):
    residual_src = np.load(BIAS_DATA_DIR / "bias_corrected_f3_full_residual_prediction.npy", mmap_mode="r")
    narrow_src = np.load(BIAS_DATA_DIR / "bias_corrected_f3_full_narrow_input.npy", mmap_mode="r")
    target_src = np.load(BIAS_DATA_DIR / "bias_corrected_f3_full_wide_reference.npy", mmap_mode="r")

    shape = residual_src.shape
    tag = f"gain_ratio_{target_ratio:.2f}".replace(".", "p")
    pred_path = DATA_DIR / f"{tag}_wide_prediction.npy"
    residual_path = DATA_DIR / f"{tag}_residual_prediction.npy"

    pred_out = np.lib.format.open_memmap(pred_path, mode="w+", dtype=np.float32, shape=shape)
    residual_out = np.lib.format.open_memmap(residual_path, mode="w+", dtype=np.float32, shape=shape)

    gains = []
    current_ratios = []
    for idx in range(shape[0]):
        residual = np.asarray(residual_src[idx], dtype=np.float32)
        narrow = np.asarray(narrow_src[idx], dtype=np.float32)
        corrected, gain, current_ratio = residual_gain_correction(
            residual, narrow, target_ratio=target_ratio, max_gain=max_gain
        )
        residual_out[idx] = corrected
        pred_out[idx] = narrow + corrected
        gains.append(gain)
        current_ratios.append(current_ratio)
        if idx % 25 == 0 or idx == shape[0] - 1:
            print(
                f"{tag}: corrected inline {idx + 1}/{shape[0]}, "
                f"gain={gain:.3f}, current_ratio={current_ratio:.3f}",
                flush=True,
            )

    pred_out.flush()
    residual_out.flush()

    pred_freq, pred_amp = average_cube_spectrum(pred_path, DT)
    f3_freq, f3_amp = average_cube_spectrum(
        BIAS_DATA_DIR / "bias_corrected_f3_full_wide_reference.npy", DT
    )
    narrow_freq, narrow_amp = average_cube_spectrum(
        BIAS_DATA_DIR / "bias_corrected_f3_full_narrow_input.npy", DT
    )
    bias_freq, bias_amp = average_cube_spectrum(
        BIAS_DATA_DIR / "bias_corrected_f3_full_wide_prediction.npy", DT
    )

    result_metrics = {
        "target_ratio": target_ratio,
        "max_gain": max_gain,
        "mean_applied_gain": float(np.mean(gains)),
        "median_applied_gain": float(np.median(gains)),
        "mean_pre_gain_high_ratio": float(np.mean(current_ratios)),
        "prediction_vs_reference": streaming_metrics(pred_out, target_src),
        "spectrum": {
            "prediction_high_freq_ratio_35_80": high_freq_ratio(pred_freq, pred_amp),
            "reference_high_freq_ratio_35_80": high_freq_ratio(f3_freq, f3_amp),
            "bias_prediction_high_freq_ratio_35_80": high_freq_ratio(bias_freq, bias_amp),
            "narrow_high_freq_ratio_35_80": high_freq_ratio(narrow_freq, narrow_amp),
            "prediction_spectrum_l1": spectrum_l1(pred_freq, pred_amp, f3_freq, f3_amp),
            "bias_prediction_spectrum_l1": spectrum_l1(bias_freq, bias_amp, f3_freq, f3_amp),
        },
    }
    np.save(DATA_DIR / f"{tag}_metrics.npy", result_metrics)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(narrow_freq, narrow_amp, "b-", lw=1.8, label="F3 low-pass")
    ax.plot(bias_freq, bias_amp, color="orange", lw=2.0, label="Bias-corrected prediction")
    ax.plot(pred_freq, pred_amp, "r-", lw=2.2, label=f"Gain corrected {target_ratio:.2f}")
    ax.plot(f3_freq, f3_amp, "k--", lw=2.2, label="F3 reference")
    ax.axvspan(HIGH_BAND[0], HIGH_BAND[1], color="red", alpha=0.12, label="35-80 Hz")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"Inference spectral gain correction, target ratio={target_ratio:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"02_{tag}_spectrum_compare.png", dpi=300)
    plt.close(fig)

    mid = shape[0] // 2
    panels = [
        (narrow_src[mid], "Low-pass input"),
        (residual_out[mid], "Gain corrected residual"),
        (pred_out[mid], "Gain corrected prediction"),
        (target_src[mid], "F3 reference"),
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
    fig.savefig(FIGURE_DIR / f"02_{tag}_section_compare.png", dpi=300)
    plt.close(fig)

    print(f"Metrics for {tag}:")
    print(result_metrics)
    return result_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.30])
    parser.add_argument("--max-gain", type=float, default=3.0)
    args = parser.parse_args()

    ensure_dirs()
    all_metrics = {}
    for ratio in args.ratios:
        all_metrics[f"{ratio:.2f}"] = run_for_ratio(ratio, args.max_gain)
    np.save(DATA_DIR / "gain_correction_all_metrics.npy", all_metrics)


if __name__ == "__main__":
    main()
