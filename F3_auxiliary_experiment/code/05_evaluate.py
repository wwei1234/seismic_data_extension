import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from config import DT, FIGURE_DIR, PREDICTION_DIR
from signal_utils import average_amplitude_spectrum


def metrics(pred, target):
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    diff = pred - target
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    corr = np.corrcoef(pred.ravel(), target.ravel())[0, 1]
    peak = max(np.max(np.abs(pred)), np.max(np.abs(target)), 1e-8)
    psnr = 20 * np.log10(peak / max(rmse, 1e-12))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "PSNR": psnr, "Correlation": corr}


def high_freq_ratio(freqs, amp, band=(35.0, 80.0)):
    mask_high = (freqs >= band[0]) & (freqs <= band[1])
    mask_all = (freqs >= 1.0) & (freqs <= band[1])
    return float(np.sum(amp[mask_high]) / (np.sum(amp[mask_all]) + 1e-12))


def cube_to_time_trace_matrix(cube):
    return cube.transpose(1, 0, 2).reshape(cube.shape[1], -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="f3_prediction")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pred = np.load(PREDICTION_DIR / f"{args.prefix}_output.npy")
    narrow = np.load(PREDICTION_DIR / f"{args.prefix}_narrow_input.npy")
    target = np.load(PREDICTION_DIR / f"{args.prefix}_wide_reference.npy")

    metric_dict = {
        "narrow_vs_reference": metrics(narrow, target),
        "prediction_vs_reference": metrics(pred, target),
    }

    freq_n, amp_n = average_amplitude_spectrum(cube_to_time_trace_matrix(narrow), DT)
    freq_p, amp_p = average_amplitude_spectrum(cube_to_time_trace_matrix(pred), DT)
    freq_t, amp_t = average_amplitude_spectrum(cube_to_time_trace_matrix(target), DT)

    metric_dict["spectrum"] = {
        "narrow_high_freq_ratio_35_80": high_freq_ratio(freq_n, amp_n),
        "prediction_high_freq_ratio_35_80": high_freq_ratio(freq_p, amp_p),
        "reference_high_freq_ratio_35_80": high_freq_ratio(freq_t, amp_t),
        "prediction_spectrum_l1": float(np.mean(np.abs(amp_p - amp_t))),
        "narrow_spectrum_l1": float(np.mean(np.abs(amp_n - amp_t))),
    }
    np.save(PREDICTION_DIR / f"{args.prefix}_metrics.npy", metric_dict)

    print("Metrics:")
    for group, vals in metric_dict.items():
        print(group)
        for key, val in vals.items():
            print(f"  {key}: {val:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_n, amp_n, "b-", lw=2, label="Low-pass input")
    ax.plot(freq_p, amp_p, "r-", lw=2, label="Prediction")
    ax.plot(freq_t, amp_t, "k--", lw=2, label="F3 reference")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.prefix}_spectrum_compare.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
