"""Blindly evaluate locked experiment 19 against the withheld F3 reference."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from config import DT, EVALUATION_DIR, FIGURE_DIR, PREDICTION_DIR, ensure_dirs
from phase_metrics import (
    bandpass_correlation,
    envelope_correlation,
    low_frequency_metrics,
    residual_high_frequency_metrics,
    safe_correlation,
    weighted_phase_score,
)


def spatial_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    difference = prediction - target
    mse = float(np.mean(difference ** 2))
    rmse = float(np.sqrt(mse))
    peak = max(float(np.max(np.abs(prediction))), float(np.max(np.abs(target))), 1e-8)
    return {
        "MAE": float(np.mean(np.abs(difference))),
        "RMSE": rmse,
        "PSNR": float(20.0 * np.log10(peak / max(rmse, 1e-12))),
        "Correlation": safe_correlation(prediction, target),
    }


def average_spectrum(cube):
    spectrum = np.abs(np.fft.rfft(np.asarray(cube, dtype=np.float64), axis=-2))
    amplitude = spectrum.mean(axis=tuple(index for index in range(spectrum.ndim) if index != spectrum.ndim - 2))
    amplitude /= np.max(amplitude) + 1e-12
    frequencies = np.fft.rfftfreq(cube.shape[-2], d=DT)
    return frequencies, amplitude


def spectrum_metrics(prediction, target):
    frequencies, amplitude = average_spectrum(prediction)
    target_frequencies, target_amplitude = average_spectrum(target)
    high = (frequencies >= 35.0) & (frequencies <= 80.0)
    total = (frequencies >= 1.0) & (frequencies <= 80.0)
    common = frequencies <= 100.0
    target_interp = np.interp(frequencies[common], target_frequencies, target_amplitude)
    return {
        "high_freq_ratio_35_80": float(
            amplitude[high].sum() / (amplitude[total].sum() + 1e-12)
        ),
        "spectrum_l1_vs_reference": float(
            np.mean(np.abs(amplitude[common] - target_interp))
        ),
    }, (frequencies, amplitude)


def evaluate(prediction, reference, narrow):
    result = spatial_metrics(prediction, reference)
    spectral, spectrum = spectrum_metrics(prediction, reference)
    low = low_frequency_metrics(prediction, narrow, DT)
    residual_high = residual_high_frequency_metrics(
        prediction,
        reference,
        narrow,
        DT,
    )
    result.update(spectral)
    result.update({
        "full_bandpass_correlation_25_80": bandpass_correlation(
            prediction, reference, DT
        ),
        "residual_bandpass_correlation_25_80": residual_high["bandpass_correlation"],
        "residual_weighted_phase_score_25_80": residual_high["phase_score"],
        "residual_envelope_correlation_25_80": residual_high["envelope_correlation"],
        "low_frequency_correlation_0_22": low["correlation"],
        "low_frequency_nrmse_0_22": low["nrmse"],
    })
    return result, spectrum


def format_metrics(title, metrics):
    lines = [title]
    for key, value in metrics.items():
        lines.append(f"  {key}: {value:.6f}")
    return "\n".join(lines)


def plot_sections(path, axis, numbers, narrow, prediction, reference):
    rows = prediction.shape[0]
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4 * rows), squeeze=False)
    for row in range(rows):
        panels = (
            (narrow[row], "Low-pass input"),
            (prediction[row], "Experiment 19 blind prediction"),
            (reference[row], "F3 wide reference"),
            (prediction[row] - reference[row], "Prediction error"),
        )
        shared_clip = max(
            float(np.percentile(np.abs(np.concatenate([
                narrow[row].ravel(),
                prediction[row].ravel(),
                reference[row].ravel(),
            ])), 99)),
            1e-8,
        )
        for column, (data, title) in enumerate(panels):
            clip = shared_clip if column < 3 else max(
                float(np.percentile(np.abs(data), 99)), 1e-8
            )
            axes[row, column].imshow(
                data,
                cmap="seismic",
                aspect="auto",
                vmin=-clip,
                vmax=clip,
            )
            axes[row, column].set_title(f"{axis} {numbers[row]} | {title}")
            axes[row, column].set_xlabel("Trace")
            axes[row, column].set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_spectra(path, spectra):
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, (frequencies, amplitude, color, style) in spectra.items():
        ax.plot(frequencies, amplitude, color=color, linestyle=style, label=label)
    ax.axvspan(35, 80, color="tab:red", alpha=0.1)
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    ensure_dirs()
    narrow = np.load(PREDICTION_DIR / f"{args.prefix}_narrow_input.npy", mmap_mode="r")
    prediction = np.load(PREDICTION_DIR / f"{args.prefix}_wide_prediction.npy", mmap_mode="r")
    reference = np.load(PREDICTION_DIR / f"{args.prefix}_wide_reference.npy", mmap_mode="r")
    metadata = np.load(
        PREDICTION_DIR / f"{args.prefix}_metadata.npy",
        allow_pickle=True,
    ).item()
    baseline, baseline_spectrum = evaluate(narrow, reference, narrow)
    predicted, predicted_spectrum = evaluate(prediction, reference, narrow)
    reference_frequencies, reference_amplitude = average_spectrum(reference)

    per_section = []
    for index, number in enumerate(metadata["section_numbers"]):
        section_baseline, _ = evaluate(
            narrow[index:index + 1],
            reference[index:index + 1],
            narrow[index:index + 1],
        )
        section_prediction, _ = evaluate(
            prediction[index:index + 1],
            reference[index:index + 1],
            narrow[index:index + 1],
        )
        per_section.append({
            "section_number": int(number),
            "baseline": section_baseline,
            "prediction": section_prediction,
        })

    result = {
        "prefix": args.prefix,
        "section_axis": metadata["section_axis"],
        "section_numbers": metadata["section_numbers"],
        "baseline": baseline,
        "prediction": predicted,
        "per_section": per_section,
    }
    np.save(EVALUATION_DIR / f"{args.prefix}_metrics.npy", result)

    section_path = FIGURE_DIR / "预测评价" / f"{args.prefix}_sections.png"
    spectrum_path = FIGURE_DIR / "预测评价" / f"{args.prefix}_spectra.png"
    plot_sections(
        section_path,
        metadata["section_axis"],
        metadata["section_numbers"],
        narrow,
        prediction,
        reference,
    )
    plot_spectra(spectrum_path, {
        "Low-pass input": (*baseline_spectrum, "tab:blue", "-"),
        "Experiment 19 blind prediction": (*predicted_spectrum, "tab:red", "-"),
        "F3 wide reference": (
            reference_frequencies,
            reference_amplitude,
            "black",
            "--",
        ),
    })

    report = [
        f"Evaluation prefix: {args.prefix}",
        f"Section axis: {metadata['section_axis']}",
        f"Section numbers: {', '.join(str(v) for v in metadata['section_numbers'])}",
        "",
        format_metrics("Low-pass baseline vs reference", baseline),
        "",
        format_metrics("Experiment 18 prediction vs reference", predicted),
        "",
        "Per-section key metrics:",
        "  number, baseline_corr, prediction_corr, prediction_band_corr, "
        "prediction_phase_score, prediction_spectrum_l1",
    ]
    for row in per_section:
        report.append(
            f"  {row['section_number']}, "
            f"{row['baseline']['Correlation']:.6f}, "
            f"{row['prediction']['Correlation']:.6f}, "
            f"{row['prediction']['residual_bandpass_correlation_25_80']:.6f}, "
            f"{row['prediction']['residual_weighted_phase_score_25_80']:.6f}, "
            f"{row['prediction']['spectrum_l1_vs_reference']:.6f}"
        )
    report.extend([
        "",
        f"Section figure: {section_path}",
        f"Spectrum figure: {spectrum_path}",
    ])
    report_path = FIGURE_DIR / "预测评价" / f"{args.prefix}_evaluation_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
