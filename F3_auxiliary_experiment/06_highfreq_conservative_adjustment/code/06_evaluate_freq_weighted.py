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
    corr = float(cov / (np.sqrt(max(var_p, 1e-12) * max(var_t, 1e-12))))
    psnr = float(20.0 * np.log10(peak / max(rmse, 1e-12)))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "PSNR": psnr, "Correlation": corr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="freq_weighted_f3_full")
    args = parser.parse_args()

    ensure_dirs()
    pred = np.load(DATA_DIR / f"{args.prefix}_wide_prediction.npy", mmap_mode="r")
    narrow = np.load(DATA_DIR / f"{args.prefix}_narrow_input.npy", mmap_mode="r")
    target = np.load(DATA_DIR / f"{args.prefix}_wide_reference.npy", mmap_mode="r")

    freq_p, amp_p = average_cube_spectrum(DATA_DIR / f"{args.prefix}_wide_prediction.npy", DT)
    freq_n, amp_n = average_cube_spectrum(DATA_DIR / f"{args.prefix}_narrow_input.npy", DT)
    freq_t, amp_t = average_cube_spectrum(DATA_DIR / f"{args.prefix}_wide_reference.npy", DT)

    metrics = {
        "narrow_vs_reference": streaming_metrics(narrow, target),
        "freq_weighted_prediction_vs_reference": streaming_metrics(pred, target),
        "spectrum": {
            "narrow_high_freq_ratio_35_80": high_freq_ratio(freq_n, amp_n),
            "prediction_high_freq_ratio_35_80": high_freq_ratio(freq_p, amp_p),
            "reference_high_freq_ratio_35_80": high_freq_ratio(freq_t, amp_t),
            "prediction_spectrum_l1": spectrum_l1(freq_p, amp_p, freq_t, amp_t),
            "narrow_spectrum_l1": spectrum_l1(freq_n, amp_n, freq_t, amp_t),
        },
    }
    np.save(DATA_DIR / f"{args.prefix}_metrics.npy", metrics)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(freq_n, amp_n, "b-", lw=2, label="Low-pass input")
    ax.plot(freq_p, amp_p, "r-", lw=2, label="Frequency-weighted prediction")
    ax.plot(freq_t, amp_t, "k--", lw=2, label="F3 reference")
    ax.axvspan(HIGH_BAND[0], HIGH_BAND[1], color="red", alpha=0.12, label="35-80 Hz")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Frequency-weighted prediction spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"06_{args.prefix}_spectrum_compare.png", dpi=300)
    plt.close(fig)

    print("Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
