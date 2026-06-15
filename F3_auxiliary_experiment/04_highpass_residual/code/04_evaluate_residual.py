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
    average_amplitude_spectrum,
    average_phase_spectrum,
    cube_to_time_trace_matrix,
    ensure_dirs,
    high_freq_ratio,
    metrics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="highpass_residual_f3_full")
    args = parser.parse_args()

    ensure_dirs()
    pred = np.load(DATA_DIR / f"{args.prefix}_wide_prediction.npy", mmap_mode="r")
    residual = np.load(DATA_DIR / f"{args.prefix}_residual_prediction.npy", mmap_mode="r")
    narrow = np.load(DATA_DIR / f"{args.prefix}_narrow_input.npy", mmap_mode="r")
    target = np.load(DATA_DIR / f"{args.prefix}_wide_reference.npy", mmap_mode="r")

    metric_dict = {
        "narrow_vs_reference": metrics(narrow, target),
        "highpass_residual_prediction_vs_reference": metrics(pred, target),
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
    np.save(DATA_DIR / f"{args.prefix}_metrics.npy", metric_dict)

    print("Metrics:")
    for group, vals in metric_dict.items():
        print(group)
        for key, val in vals.items():
            print(f"  {key}: {val:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_n, amp_n, "b-", lw=2, label="Low-pass input")
    ax.plot(freq_p, amp_p, "r-", lw=2, label="High-pass residual prediction")
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

    freq_p_ph, phase_p, valid_p = average_phase_spectrum(cube_to_time_trace_matrix(pred), DT)
    freq_t_ph, phase_t, valid_t = average_phase_spectrum(cube_to_time_trace_matrix(target), DT)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_p_ph[valid_p], phase_p[valid_p], "r.", ms=3, label="High-pass residual prediction")
    ax.plot(freq_t_ph[valid_t], phase_t[valid_t], "k.", ms=3, label="F3 reference")
    ax.set_xlim(0, 100)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.prefix}_phase_compare.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
