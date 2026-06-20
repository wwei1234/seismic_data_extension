"""Evaluate direct and high-pass residual recombination outputs."""

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


def load_cube(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def evaluate_prediction(pred_path, target_path):
    pred = load_cube(pred_path)
    target = load_cube(target_path)
    freq_p, amp_p = average_cube_spectrum(pred_path, DT)
    freq_t, amp_t = average_cube_spectrum(target_path, DT)
    result = streaming_metrics(pred, target)
    result.update({
        "high_freq_ratio_35_80": high_freq_ratio(freq_p, amp_p),
        "spectrum_l1_vs_reference": spectrum_l1(freq_p, amp_p, freq_t, amp_t),
    })
    return result, (freq_p, amp_p)


def format_metrics(title, metrics):
    lines = [title]
    for key in ("MAE", "RMSE", "MSE", "PSNR", "Correlation",
                "high_freq_ratio_35_80", "spectrum_l1_vs_reference"):
        lines.append(f"  {key}: {metrics[key]:.6f}")
    return "\n".join(lines)


def plot_spectra(prefix, spectra):
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, (freqs, amp, color, style) in spectra.items():
        ax.plot(freqs, amp, color=color, linestyle=style, lw=2.0, label=label)
    ax.axvspan(35.0, 80.0, color="tab:red", alpha=0.10, label="35-80 Hz")
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.set_title("Direct vs high-pass residual recombination spectra")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIGURE_DIR / f"{prefix}_direct_vs_highpass_spectra.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_sections(prefix, narrow_path, direct_path, highpass_path, target_path):
    narrow = load_cube(narrow_path)
    direct = load_cube(direct_path)
    highpass = load_cube(highpass_path)
    target = load_cube(target_path)
    mid = direct.shape[0] // 2
    panels = [
        (narrow[mid], "Low-pass input"),
        (direct[mid], "Direct recombination"),
        (highpass[mid], "High-pass residual recombination"),
        (target[mid], "F3 reference"),
        (highpass[mid] - direct[mid], "High-pass - direct"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(24, 5))
    for ax, (data, title) in zip(axes, panels):
        clip = max(float(np.percentile(np.abs(data), 99.0)), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    out = FIGURE_DIR / f"{prefix}_direct_vs_highpass_sections.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="bspline_v2")
    args = parser.parse_args()

    ensure_dirs()
    direct_path = DATA_DIR / f"{args.prefix}_direct_wide_prediction.npy"
    highpass_path = DATA_DIR / f"{args.prefix}_highpass_wide_prediction.npy"
    narrow_path = DATA_DIR / f"{args.prefix}_narrow_input.npy"
    target_path = DATA_DIR / f"{args.prefix}_wide_reference.npy"

    direct_metrics, direct_spec = evaluate_prediction(direct_path, target_path)
    highpass_metrics, highpass_spec = evaluate_prediction(highpass_path, target_path)
    narrow_metrics, narrow_spec = evaluate_prediction(narrow_path, target_path)
    ref_freq, ref_amp = average_cube_spectrum(target_path, DT)

    spectra = {
        "Low-pass input": (*narrow_spec, "tab:blue", "-"),
        "Direct recombination": (*direct_spec, "tab:red", "-"),
        "High-pass residual recombination": (*highpass_spec, "tab:green", "-"),
        "F3 reference": (ref_freq, ref_amp, "black", "--"),
    }
    spectrum_png = plot_spectra(args.prefix, spectra)
    section_png = plot_sections(args.prefix, narrow_path, direct_path, highpass_path, target_path)

    report = [
        f"Evaluation report for prefix: {args.prefix}",
        "",
        format_metrics("Low-pass input vs reference", narrow_metrics),
        "",
        format_metrics("Direct residual + low-pass vs reference", direct_metrics),
        "",
        format_metrics("High-pass residual + low-pass vs reference", highpass_metrics),
        "",
        "Output files:",
        f"  Direct prediction: {direct_path}",
        f"  High-pass prediction: {highpass_path}",
        f"  Spectrum figure: {spectrum_png}",
        f"  Section figure: {section_png}",
    ]
    out_txt = FIGURE_DIR / f"{args.prefix}_evaluation_report.txt"
    out_txt.write_text("\n".join(report), encoding="utf-8")
    np.save(DATA_DIR / f"{args.prefix}_direct_vs_highpass_metrics.npy", {
        "narrow": narrow_metrics,
        "direct": direct_metrics,
        "highpass": highpass_metrics,
    })
    print("\n".join(report))
    print(f"Saved report: {out_txt}")


if __name__ == "__main__":
    main()
