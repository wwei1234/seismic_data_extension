"""Evaluate and aggregate all experiment 22 held-out folds."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from config import (  # noqa: E402
    DT,
    EVALUATION_DIR,
    FIGURE_DIR,
    FOLDS,
    PREDICTION_DIR,
    ensure_dirs,
)
from phase_metrics import (  # noqa: E402
    bandpass_correlation,
    low_frequency_metrics,
    residual_high_frequency_metrics,
    safe_correlation,
)


def spatial_metrics(prediction, reference):
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    peak = max(float(np.max(np.abs(reference))), 1e-8)
    return {
        "MAE": float(np.mean(np.abs(difference))),
        "RMSE": rmse,
        "PSNR": float(20 * np.log10(peak / max(rmse, 1e-12))),
        "Correlation": safe_correlation(prediction, reference),
    }


def average_spectrum(section):
    amplitude = np.abs(np.fft.rfft(section, axis=0)).mean(axis=1)
    amplitude /= amplitude.max() + 1e-12
    return np.fft.rfftfreq(section.shape[0], DT), amplitude


def evaluate(prediction, reference, narrow):
    result = spatial_metrics(prediction, reference)
    frequencies, amplitude = average_spectrum(prediction)
    ref_frequencies, ref_amplitude = average_spectrum(reference)
    high = (frequencies >= 35) & (frequencies <= 80)
    total = (frequencies >= 1) & (frequencies <= 80)
    common = frequencies <= 100
    result.update({
        "high_freq_ratio_35_80": float(
            amplitude[high].sum() / (amplitude[total].sum() + 1e-12)
        ),
        "spectrum_l1_vs_reference": float(np.mean(np.abs(
            amplitude[common]
            - np.interp(frequencies[common], ref_frequencies, ref_amplitude)
        ))),
        "full_bandpass_correlation_25_80": bandpass_correlation(
            prediction, reference, DT
        ),
    })
    residual = residual_high_frequency_metrics(
        prediction, reference, narrow, DT
    )
    low = low_frequency_metrics(prediction, narrow, DT)
    result.update({
        "residual_bandpass_correlation_25_80": residual[
            "bandpass_correlation"
        ],
        "residual_weighted_phase_score_25_80": residual["phase_score"],
        "residual_envelope_correlation_25_80": residual[
            "envelope_correlation"
        ],
        "low_frequency_correlation_0_22": low["correlation"],
        "low_frequency_nrmse_0_22": low["nrmse"],
    })
    return result


def plot_section(path, title, arrays):
    compared = [arrays[key] for key in (
        "narrow", "direct", "highpass", "reference"
    )]
    clip = max(float(np.percentile(
        np.abs(np.concatenate([item.ravel() for item in compared])), 99
    )), 1e-8)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    panels = (
        (arrays["narrow"], "Low-pass input", clip),
        (arrays["direct"], "Direct prediction", clip),
        (arrays["highpass"], "Highpass prediction", clip),
        (arrays["reference"], "F3 wide reference", clip),
        (
            arrays["direct"] - arrays["reference"],
            "Direct error",
            max(float(np.percentile(np.abs(
                arrays["direct"] - arrays["reference"]
            ), 99)), 1e-8),
        ),
    )
    for axis, (data, name, limit) in zip(axes, panels):
        axis.imshow(
            data, cmap="seismic", aspect="auto", vmin=-limit, vmax=limit
        )
        axis.set_title(name)
        axis.set_xlabel("Trace")
        axis.set_ylabel("Time sample")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_spectra(path, title, arrays):
    fig, axis = plt.subplots(figsize=(11, 5))
    for key, label, style in (
        ("narrow", "Low-pass input", "b-"),
        ("direct", "Direct prediction", "r-"),
        ("highpass", "Highpass prediction", "g-"),
        ("reference", "F3 wide reference", "k--"),
    ):
        frequencies, amplitude = average_spectrum(arrays[key])
        axis.plot(frequencies, amplitude, style, label=label)
    axis.axvspan(35, 80, color="tab:red", alpha=0.1)
    axis.set_xlim(0, 120)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalized average amplitude")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def mean_metrics(rows, name):
    keys = rows[0][name].keys()
    return {
        key: float(np.mean([row[name][key] for row in rows]))
        for key in keys
    }


def passes_success_gate(aggregate, direct_wins):
    return (
        aggregate["inline"]["direct"]["Correlation"]
        > aggregate["inline"]["baseline"]["Correlation"]
        and aggregate["crossline"]["direct"]["Correlation"]
        > aggregate["crossline"]["baseline"]["Correlation"]
        and aggregate["inline"]["direct"][
            "residual_bandpass_correlation_25_80"
        ] > 0
        and aggregate["crossline"]["direct"][
            "residual_bandpass_correlation_25_80"
        ] > 0
        and direct_wins >= 6
    )


def main():
    ensure_dirs()
    rows = []
    figure_dir = FIGURE_DIR / "预测评价"
    for fold in FOLDS:
        for axis in ("inline", "crossline"):
            prefix = f"{fold}_{axis}"
            arrays = {
                "narrow": np.load(
                    PREDICTION_DIR / f"{prefix}_narrow_input.npy"
                ),
                "direct": np.load(
                    PREDICTION_DIR / f"{prefix}_direct_prediction.npy"
                ),
                "highpass": np.load(
                    PREDICTION_DIR / f"{prefix}_highpass_prediction.npy"
                ),
                "reference": np.load(
                    PREDICTION_DIR / f"{prefix}_wide_reference.npy"
                ),
            }
            metadata = np.load(
                PREDICTION_DIR / f"{prefix}_metadata.npy",
                allow_pickle=True,
            ).item()
            row = {
                **metadata,
                "baseline": evaluate(
                    arrays["narrow"], arrays["reference"], arrays["narrow"]
                ),
                "direct": evaluate(
                    arrays["direct"], arrays["reference"], arrays["narrow"]
                ),
                "highpass": evaluate(
                    arrays["highpass"], arrays["reference"], arrays["narrow"]
                ),
            }
            rows.append(row)
            plot_section(
                figure_dir / f"{prefix}_sections.png",
                f"{fold} held-out {axis} {metadata['section_number']}",
                arrays,
            )
            plot_spectra(
                figure_dir / f"{prefix}_spectra.png",
                f"{fold} held-out {axis} {metadata['section_number']}",
                arrays,
            )
    aggregate = {}
    for axis in ("inline", "crossline"):
        selected = [row for row in rows if row["section_axis"] == axis]
        aggregate[axis] = {
            name: mean_metrics(selected, name)
            for name in ("baseline", "direct", "highpass")
        }
    direct_wins = sum(
        row["direct"]["Correlation"] >= row["baseline"]["Correlation"]
        for row in rows
    )
    success = passes_success_gate(aggregate, direct_wins)
    result = {
        "experiment": 22,
        "rows": rows,
        "aggregate": aggregate,
        "direct_sections_not_below_baseline": direct_wins,
        "success": success,
    }
    np.save(EVALUATION_DIR / "leave_one_well_aggregate_metrics.npy", result)
    lines = [
        "21号留一井局部宽频标定最终评价",
        "================================",
        "",
        "数据条件：每折仅使用另外三口井 ±8 inline、±16 crossline 局部宽频窗口。",
        "留出井及其扩大保护区未参与训练、验证或checkpoint选择。",
        "",
    ]
    for axis in ("inline", "crossline"):
        lines.append(f"{axis} aggregate")
        for name in ("baseline", "direct", "highpass"):
            metrics = aggregate[axis][name]
            lines.append(
                f"  {name}: corr={metrics['Correlation']:.6f}, "
                f"rmse={metrics['RMSE']:.6f}, "
                f"res_corr={metrics['residual_bandpass_correlation_25_80']:.6f}, "
                f"phase={metrics['residual_weighted_phase_score_25_80']:.6f}, "
                f"spectrum_l1={metrics['spectrum_l1_vs_reference']:.6f}"
            )
        lines.append("")
    lines.extend([
        f"Direct sections not below baseline: {direct_wins}/8",
        f"Predeclared success: {success}",
        "",
        "Experiment 18 is an all-area wide-supervision upper bound and is not the same data condition.",
    ])
    report = figure_dir / "leave_one_well_final_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
