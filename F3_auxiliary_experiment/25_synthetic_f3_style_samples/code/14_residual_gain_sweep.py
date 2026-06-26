"""Sweep gain levels for predicted residual recombination.

The model output is a high-frequency residual. This script tests whether simple
amplitude scaling of that residual improves the final F3 inline reconstruction:

    wide_gain = lowpass_input + gain * predicted_residual

Both the raw predicted residual and the 35 Hz high-pass residual are evaluated.
"""

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from common import (  # noqa: E402
    DATA_DIR,
    DT,
    FIGURE_DIR,
    average_amplitude_spectrum,
    high_freq_ratio,
    metrics,
    spectrum_l1,
)


PREFIX = "residual25_f3style_inline"
GAIN_LEVELS = (0.5, 1.0, 1.5, 2.0, 3.0)
OUTPUT_DIR = FIGURE_DIR / "残差增幅实验"


def load(name):
    return np.load(DATA_DIR / f"{PREFIX}_{name}.npy", allow_pickle=True)


def spectra_for(section):
    return average_amplitude_spectrum(section.reshape(-1, section.shape[-1]), DT)


def cube_spectrum(cube):
    return average_amplitude_spectrum(cube.reshape(-1, cube.shape[-1]), DT)


def prediction_metrics(pred, reference):
    freqs_p, amp_p = cube_spectrum(pred)
    freqs_r, amp_r = cube_spectrum(reference)
    result = metrics(pred, reference)
    result["high_freq_ratio_35_90"] = high_freq_ratio(freqs_p, amp_p)
    result["spectrum_l1_vs_reference"] = spectrum_l1(freqs_p, amp_p, freqs_r, amp_r)
    return result


def plot_gain_sections(out_path, narrow, reference, residual, inline_values, mode_name):
    predictions = [(gain, narrow + gain * residual) for gain in GAIN_LEVELS]
    n_rows = narrow.shape[0]
    n_cols = len(predictions) + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for row_idx, inline in enumerate(inline_values):
        panels = [("Low-pass", narrow[row_idx])]
        panels.extend((f"gain {gain:g}", pred[row_idx]) for gain, pred in predictions)
        panels.append(("F3 wide ref", reference[row_idx]))
        clip = max(float(np.percentile(np.abs(np.concatenate([x.ravel() for _, x in panels])), 99.0)), 1e-8)
        for ax, (title, data) in zip(axes[row_idx], panels):
            ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
            ax.set_title(title if row_idx == 0 else "")
            ax.set_xlabel("Trace")
            ax.set_ylabel(f"Inline {int(inline)}\nTime")

    fig.suptitle(f"Residual gain sweep: {mode_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_gain_zoom(out_path, narrow, reference, residual, inline_values, mode_name, t0=285, t1=455):
    predictions = [(gain, narrow + gain * residual) for gain in GAIN_LEVELS]
    n_rows = narrow.shape[0]
    n_cols = len(predictions) + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.6 * n_rows), squeeze=False)

    for row_idx, inline in enumerate(inline_values):
        panels = [("Low-pass", narrow[row_idx, t0:t1])]
        panels.extend((f"gain {gain:g}", pred[row_idx, t0:t1]) for gain, pred in predictions)
        panels.append(("F3 wide ref", reference[row_idx, t0:t1]))
        clip = max(float(np.percentile(np.abs(np.concatenate([x.ravel() for _, x in panels])), 99.0)), 1e-8)
        for ax, (title, data) in zip(axes[row_idx], panels):
            ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
            ax.set_title(title if row_idx == 0 else "")
            ax.set_xlabel("Trace")
            ax.set_ylabel(f"Inline {int(inline)}\n{t0}:{t1}")

    fig.suptitle(f"Deep zoom residual gain sweep: {mode_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_gain_spectra(out_path, narrow, reference, residual, mode_name):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, data, color, style, lw in [
        ("Low-pass", narrow, "tab:blue", "-", 2.0),
        ("F3 wide ref", reference, "black", "--", 2.2),
    ]:
        freqs, amp = cube_spectrum(data)
        ax.plot(freqs, amp, color=color, linestyle=style, lw=lw, label=label)

    colors = ["#9467bd", "#d62728", "#ff7f0e", "#2ca02c", "#8c564b"]
    for gain, color in zip(GAIN_LEVELS, colors):
        pred = narrow + gain * residual
        freqs, amp = cube_spectrum(pred)
        ax.plot(freqs, amp, color=color, lw=1.8, label=f"gain {gain:g}")

    ax.axvspan(35.0, 90.0, color="tab:red", alpha=0.10, label="35-90 Hz")
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.set_title(f"Residual gain spectra: {mode_name}")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_summary(path, rows):
    best_corr = max(rows, key=lambda r: r["Correlation"])
    best_spectrum = min(rows, key=lambda r: r["spectrum_l1_vs_reference"])
    lines = [
        "25号实验预测残差增幅重组测试",
        "",
        "公式：wide_gain = lowpass_input + gain * predicted_residual",
        f"增幅档位：{', '.join(str(x) for x in GAIN_LEVELS)}",
        "",
        "按相关性最优：",
        (
            f"  mode={best_corr['mode']}, gain={best_corr['gain']}, "
            f"Correlation={best_corr['Correlation']:.6f}, RMSE={best_corr['RMSE']:.6f}, "
            f"HF ratio={best_corr['high_freq_ratio_35_90']:.6f}"
        ),
        "按频谱L1最优：",
        (
            f"  mode={best_spectrum['mode']}, gain={best_spectrum['gain']}, "
            f"Spectrum L1={best_spectrum['spectrum_l1_vs_reference']:.6f}, "
            f"Correlation={best_spectrum['Correlation']:.6f}, RMSE={best_spectrum['RMSE']:.6f}"
        ),
        "",
        "说明：如果增幅后高频占比提高但相关性/RMSE变差，说明当前残差主要问题是相位与空间位置不匹配，单纯放大不能真正修正拓频质量。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    narrow = load("narrow_input").astype(np.float32)
    reference = load("wide_reference").astype(np.float32)
    raw_residual = load("residual_prediction").astype(np.float32)
    hp_residual = load("highpass_residual_prediction").astype(np.float32)
    meta = load("inline_metadata").item()
    inline_values = meta.get("section_numbers") or meta.get("inline_numbers") or list(range(narrow.shape[0]))

    residual_modes = [
        ("raw_pred_residual", raw_residual, "raw predicted residual"),
        ("hp35_pred_residual", hp_residual, "35 Hz high-pass predicted residual"),
    ]

    rows = []
    for mode_key, residual, mode_name in residual_modes:
        plot_gain_sections(
            OUTPUT_DIR / f"{PREFIX}_{mode_key}_gain_sections.png",
            narrow,
            reference,
            residual,
            inline_values,
            mode_name,
        )
        plot_gain_zoom(
            OUTPUT_DIR / f"{PREFIX}_{mode_key}_gain_deep_zoom.png",
            narrow,
            reference,
            residual,
            inline_values,
            mode_name,
        )
        plot_gain_spectra(
            OUTPUT_DIR / f"{PREFIX}_{mode_key}_gain_spectra.png",
            narrow,
            reference,
            residual,
            mode_name,
        )

        for gain in GAIN_LEVELS:
            pred = narrow + gain * residual
            row = {
                "mode": mode_key,
                "gain": gain,
                **prediction_metrics(pred, reference),
            }
            rows.append(row)
            np.save(OUTPUT_DIR / f"{PREFIX}_{mode_key}_gain_{gain:g}_wide_prediction.npy", pred.astype(np.float32))

    csv_path = OUTPUT_DIR / f"{PREFIX}_residual_gain_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_summary(OUTPUT_DIR / f"{PREFIX}_residual_gain_summary.txt", rows)
    print(f"Saved residual gain sweep outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
