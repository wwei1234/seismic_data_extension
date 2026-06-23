"""Evaluate experiment 20 low-pass, direct and highpass recombinations."""

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
    low_frequency_metrics,
    residual_high_frequency_metrics,
    safe_correlation,
)


def spatial_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    difference = prediction - target
    mse = float(np.mean(difference ** 2))
    rmse = float(np.sqrt(mse))
    peak = max(
        float(np.max(np.abs(prediction))),
        float(np.max(np.abs(target))),
        1e-8,
    )
    return {
        "MAE": float(np.mean(np.abs(difference))),
        "RMSE": rmse,
        "PSNR": float(20.0 * np.log10(peak / max(rmse, 1e-12))),
        "Correlation": safe_correlation(prediction, target),
    }


def average_spectrum(cube):
    cube = np.asarray(cube, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(cube, axis=-2))
    axes = tuple(index for index in range(spectrum.ndim) if index != spectrum.ndim - 2)
    amplitude = spectrum.mean(axis=axes)
    amplitude /= np.max(amplitude) + 1e-12
    frequencies = np.fft.rfftfreq(cube.shape[-2], d=DT)
    return frequencies, amplitude


def spectrum_metrics(prediction, reference):
    frequencies, amplitude = average_spectrum(prediction)
    reference_frequencies, reference_amplitude = average_spectrum(reference)
    high = (frequencies >= 35.0) & (frequencies <= 80.0)
    total = (frequencies >= 1.0) & (frequencies <= 80.0)
    common = frequencies <= 100.0
    reference_interp = np.interp(
        frequencies[common],
        reference_frequencies,
        reference_amplitude,
    )
    return {
        "high_freq_ratio_35_80": float(
            amplitude[high].sum() / (amplitude[total].sum() + 1e-12)
        ),
        "spectrum_l1_vs_reference": float(
            np.mean(np.abs(amplitude[common] - reference_interp))
        ),
    }, (frequencies, amplitude)


def evaluate(prediction, reference, narrow):
    result = spatial_metrics(prediction, reference)
    spectral, spectrum = spectrum_metrics(prediction, reference)
    low = low_frequency_metrics(prediction, narrow, DT)
    residual = residual_high_frequency_metrics(
        prediction,
        reference,
        narrow,
        DT,
    )
    result.update(spectral)
    result.update({
        "full_bandpass_correlation_25_80": bandpass_correlation(
            prediction,
            reference,
            DT,
        ),
        "residual_bandpass_correlation_25_80": residual["bandpass_correlation"],
        "residual_weighted_phase_score_25_80": residual["phase_score"],
        "residual_envelope_correlation_25_80": residual["envelope_correlation"],
        "low_frequency_correlation_0_22": low["correlation"],
        "low_frequency_nrmse_0_22": low["nrmse"],
    })
    return result, spectrum


def format_metrics(title, metrics):
    lines = [title]
    lines.extend(f"  {key}: {value:.6f}" for key, value in metrics.items())
    return "\n".join(lines)


def provenance_lines(metadata):
    lines = [f"Checkpoint SHA256: {metadata['lock_sha256']}"]
    if metadata.get("diagnostic_ungated_evaluation"):
        lines.extend([
            "This diagnostic checkpoint did not pass the prelocked training gate.",
            "Ungated real-F3 evaluation was explicitly authorized by the user.",
            "F3 wide reference was read only after diagnostic checkpoint verification.",
        ])
    else:
        lines.append("F3 wide reference was read only after model-lock verification.")
    return lines


def plot_sections(path, axis_name, numbers, narrow, direct, highpass, reference):
    rows = len(numbers)
    fig, axes = plt.subplots(rows, 5, figsize=(22, 4 * rows), squeeze=False)
    for row, number in enumerate(numbers):
        compared = (narrow[row], direct[row], highpass[row], reference[row])
        shared_clip = max(
            float(np.percentile(np.abs(np.concatenate([
                item.ravel() for item in compared
            ])), 99)),
            1e-8,
        )
        panels = (
            (narrow[row], "Low-pass input", shared_clip),
            (direct[row], "Direct prediction", shared_clip),
            (highpass[row], "Highpass prediction", shared_clip),
            (reference[row], "F3 wide reference", shared_clip),
            (
                direct[row] - reference[row],
                "Direct error",
                max(
                    float(np.percentile(np.abs(direct[row] - reference[row]), 99)),
                    1e-8,
                ),
            ),
        )
        for column, (data, title, clip) in enumerate(panels):
            axes[row, column].imshow(
                data,
                cmap="seismic",
                aspect="auto",
                vmin=-clip,
                vmax=clip,
            )
            axes[row, column].set_title(f"{axis_name} {number} | {title}")
            axes[row, column].set_xlabel("Trace")
            axes[row, column].set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_spectra(path, spectra):
    fig, axis = plt.subplots(figsize=(12, 5))
    for label, (frequencies, amplitude, style) in spectra.items():
        axis.plot(frequencies, amplitude, style, linewidth=1.8, label=label)
    axis.axvspan(35, 80, color="tab:red", alpha=0.1)
    axis.set_xlim(0, 120)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalized average amplitude")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    ensure_dirs()
    arrays = {
        name: np.load(
            PREDICTION_DIR / f"{args.prefix}_{name}.npy",
            mmap_mode="r",
        )
        for name in (
            "narrow_input",
            "direct_prediction",
            "highpass_prediction",
            "wide_reference",
        )
    }
    metadata = np.load(
        PREDICTION_DIR / f"{args.prefix}_metadata.npy",
        allow_pickle=True,
    ).item()
    results = {}
    spectra = {}
    for name, data in (
        ("baseline", arrays["narrow_input"]),
        ("direct", arrays["direct_prediction"]),
        ("highpass", arrays["highpass_prediction"]),
    ):
        results[name], spectra[name] = evaluate(
            data,
            arrays["wide_reference"],
            arrays["narrow_input"],
        )
    reference_spectrum = average_spectrum(arrays["wide_reference"])

    per_section = []
    for index, number in enumerate(metadata["section_numbers"]):
        row = {"section_number": int(number)}
        for name, data in (
            ("baseline", arrays["narrow_input"]),
            ("direct", arrays["direct_prediction"]),
            ("highpass", arrays["highpass_prediction"]),
        ):
            row[name], _ = evaluate(
                data[index:index + 1],
                arrays["wide_reference"][index:index + 1],
                arrays["narrow_input"][index:index + 1],
            )
        per_section.append(row)

    result = {
        "prefix": args.prefix,
        "section_axis": metadata["section_axis"],
        "section_numbers": metadata["section_numbers"],
        **results,
        "per_section": per_section,
    }
    np.save(EVALUATION_DIR / f"{args.prefix}_metrics.npy", result)

    output_dir = FIGURE_DIR / "预测评价"
    section_path = output_dir / f"{args.prefix}_sections.png"
    spectrum_path = output_dir / f"{args.prefix}_spectra.png"
    plot_sections(
        section_path,
        metadata["section_axis"],
        metadata["section_numbers"],
        arrays["narrow_input"],
        arrays["direct_prediction"],
        arrays["highpass_prediction"],
        arrays["wide_reference"],
    )
    plot_spectra(spectrum_path, {
        "Low-pass input": (*spectra["baseline"], "b-"),
        "Direct prediction": (*spectra["direct"], "r-"),
        "Highpass prediction": (*spectra["highpass"], "g-"),
        "F3 wide reference": (*reference_spectrum, "k--"),
    })

    report = [
        f"Evaluation prefix: {args.prefix}",
        f"Section axis: {metadata['section_axis']}",
        f"Section numbers: {', '.join(map(str, metadata['section_numbers']))}",
        *provenance_lines(metadata),
        "",
        format_metrics("Low-pass baseline vs reference", results["baseline"]),
        "",
        format_metrics("Experiment 20 direct vs reference", results["direct"]),
        "",
        format_metrics("Experiment 20 highpass vs reference", results["highpass"]),
        "",
        "Per-section key metrics:",
        "  number, baseline_corr, direct_corr, direct_residual_corr, "
        "direct_phase, highpass_corr, highpass_residual_corr, highpass_phase",
    ]
    for row in per_section:
        report.append(
            f"  {row['section_number']}, "
            f"{row['baseline']['Correlation']:.6f}, "
            f"{row['direct']['Correlation']:.6f}, "
            f"{row['direct']['residual_bandpass_correlation_25_80']:.6f}, "
            f"{row['direct']['residual_weighted_phase_score_25_80']:.6f}, "
            f"{row['highpass']['Correlation']:.6f}, "
            f"{row['highpass']['residual_bandpass_correlation_25_80']:.6f}, "
            f"{row['highpass']['residual_weighted_phase_score_25_80']:.6f}"
        )
    report.extend([
        "",
        "Spectral improvement alone is insufficient; correlation, phase and error "
        "must also improve.",
        f"Section figure: {section_path}",
        f"Spectrum figure: {spectrum_path}",
    ])
    report_path = output_dir / f"{args.prefix}_evaluation_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
