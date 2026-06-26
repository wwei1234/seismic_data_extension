import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from common import DATA_DIR, DT, FIGURE_DIR, average_amplitude_spectrum, metrics  # noqa: E402


PREFIX = "residual25_f3style_inline"


def load(name):
    return np.load(DATA_DIR / f"{PREFIX}_{name}.npy", allow_pickle=True)


def spectrum_ratio(section, low=35.0, high=90.0):
    freqs, amp = average_amplitude_spectrum(section, DT)
    mask = (freqs >= low) & (freqs <= high)
    base = (freqs >= 1.0) & (freqs <= high)
    return float(np.sum(amp[mask]) / (np.sum(amp[base]) + 1e-12))


def plot_four_residual_sections(out_path, actual, pred, pred_hp, inline_values):
    diff = pred_hp - actual
    shared = max(
        float(np.percentile(np.abs(actual), 99)),
        float(np.percentile(np.abs(pred), 99)),
        float(np.percentile(np.abs(pred_hp), 99)),
        1e-8,
    )
    diff_clip = max(float(np.percentile(np.abs(diff), 99)), 1e-8)
    fig, axes = plt.subplots(len(inline_values), 4, figsize=(22, 13), sharex=True, sharey=True)
    cols = [
        ("Actual residual", actual, shared),
        ("Pred residual", pred, shared),
        ("High-pass pred residual", pred_hp, shared),
        ("HP pred - actual", diff, diff_clip),
    ]
    for i, inline in enumerate(inline_values):
        for j, (title, cube, clip) in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(cube[i], cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
            if i == 0:
                ax.set_title(title)
            ax.set_ylabel(f"Inline {inline}\nTime")
            ax.set_xlabel("Trace")
    fig.suptitle("Predicted high-frequency residual vs actual residual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_zoom(out_path, actual, pred_hp, inline_values, t0=285, t1=455):
    shared = max(
        float(np.percentile(np.abs(actual[:, t0:t1]), 99)),
        float(np.percentile(np.abs(pred_hp[:, t0:t1]), 99)),
        1e-8,
    )
    fig, axes = plt.subplots(len(inline_values), 3, figsize=(18, 12), sharex=True, sharey=True)
    for i, inline in enumerate(inline_values):
        diff = pred_hp[i, t0:t1] - actual[i, t0:t1]
        diff_clip = max(float(np.percentile(np.abs(diff), 99)), 1e-8)
        panels = [
            ("Actual residual zoom", actual[i, t0:t1], shared),
            ("HP predicted residual zoom", pred_hp[i, t0:t1], shared),
            ("Difference zoom", diff, diff_clip),
        ]
        for j, (title, data, clip) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
            if i == 0:
                ax.set_title(title)
            ax.set_ylabel(f"Inline {inline}\nTime {t0}:{t1}")
            ax.set_xlabel("Trace")
    fig.suptitle("Deep high-frequency residual zoom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def plot_spectra(out_path, actual, pred, pred_hp):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data, style in [
        ("Actual residual", actual, "k-"),
        ("Pred residual", pred, "r-"),
        ("High-pass pred residual", pred_hp, "b-"),
    ]:
        freqs, amp = average_amplitude_spectrum(data.reshape(-1, data.shape[-1]), DT)
        ax.plot(freqs, amp, style, lw=2, label=label)
    ax.axvspan(35, 90, color="tab:red", alpha=0.10, label="35-90 Hz")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.set_title("Residual spectra")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    wide = load("wide_reference")
    narrow = load("narrow_input")
    pred = load("residual_prediction")
    pred_hp = load("highpass_residual_prediction")
    meta = load("inline_metadata").item()
    inline_values = meta.get("section_numbers") or meta.get("inline_numbers") or [244, 362, 442, 722]

    actual = wide - narrow
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_four_residual_sections(
        FIGURE_DIR / f"{PREFIX}_residual_prediction_vs_actual.png",
        actual, pred, pred_hp, inline_values,
    )
    plot_zoom(
        FIGURE_DIR / f"{PREFIX}_residual_prediction_vs_actual_deep_zoom.png",
        actual, pred_hp, inline_values,
    )
    plot_spectra(
        FIGURE_DIR / f"{PREFIX}_residual_prediction_vs_actual_spectra.png",
        actual, pred, pred_hp,
    )

    rows = []
    for i, inline in enumerate(inline_values):
        for name, data in [("pred", pred[i]), ("highpass_pred", pred_hp[i])]:
            row = {"inline": int(inline), "residual": name}
            row.update(metrics(data, actual[i]))
            row["actual_35_90_ratio"] = spectrum_ratio(actual[i])
            row["pred_35_90_ratio"] = spectrum_ratio(data)
            rows.append(row)
    with (FIGURE_DIR / f"{PREFIX}_residual_prediction_vs_actual_stats.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved residual comparison figures and stats.")


if __name__ == "__main__":
    main()
