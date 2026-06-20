"""Evaluate direct wideband predictions against F3 wideband references."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from config import DATA_DIR, DT


def metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    difference = prediction - target
    mse = float(np.mean(difference ** 2))
    rmse = float(np.sqrt(mse))
    peak = max(float(np.max(np.abs(prediction))), float(np.max(np.abs(target))), 1e-8)
    return {
        "MAE": float(np.mean(np.abs(difference))),
        "RMSE": rmse,
        "MSE": mse,
        "PSNR": float(20.0 * np.log10(peak / max(rmse, 1e-12))),
        "Correlation": float(np.corrcoef(prediction.ravel(), target.ravel())[0, 1]),
    }


def average_spectrum(cube):
    amp_sum = None
    trace_count = 0
    for section in cube:
        work = np.asarray(section, dtype=np.float64)
        work -= work.mean(axis=0, keepdims=True)
        amp = np.abs(np.fft.rfft(work, axis=0)).sum(axis=1)
        amp_sum = amp if amp_sum is None else amp_sum + amp
        trace_count += work.shape[1]
    amplitude = amp_sum / max(trace_count, 1)
    amplitude /= np.max(amplitude) + 1e-12
    frequencies = np.fft.rfftfreq(cube.shape[1], DT)
    return frequencies, amplitude


def high_frequency_ratio(frequencies, amplitude):
    high = (frequencies >= 35.0) & (frequencies <= 80.0)
    total = (frequencies >= 1.0) & (frequencies <= 80.0)
    return float(amplitude[high].sum() / (amplitude[total].sum() + 1e-12))


def spectrum_l1(freq_a, amp_a, freq_b, amp_b):
    common = freq_a[freq_a <= 100.0]
    other = np.interp(common, freq_b, amp_b)
    return float(np.mean(np.abs(amp_a[:common.size] - other)))


def evaluate_cube(cube, target):
    result = metrics(cube, target)
    frequencies, amplitude = average_spectrum(cube)
    target_freq, target_amp = average_spectrum(target)
    result.update({
        "high_freq_ratio_35_80": high_frequency_ratio(frequencies, amplitude),
        "spectrum_l1_vs_reference": spectrum_l1(
            frequencies, amplitude, target_freq, target_amp
        ),
    })
    return result, (frequencies, amplitude)


def format_metrics(title, result):
    lines = [title]
    for key in (
        "MAE", "RMSE", "MSE", "PSNR", "Correlation",
        "high_freq_ratio_35_80", "spectrum_l1_vs_reference",
    ):
        lines.append(f"  {key}: {result[key]:.6f}")
    return "\n".join(lines)


def plot_sections(output, section_axis, section_numbers, narrow, prediction, reference):
    rows = prediction.shape[0]
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4.0 * rows), squeeze=False)
    for row in range(rows):
        error = prediction[row] - reference[row]
        panels = [
            (narrow[row], "Low-pass input"),
            (prediction[row], "Direct wideband prediction"),
            (reference[row], "F3 wideband reference"),
            (error, "Prediction error"),
        ]
        clip = max(
            float(np.percentile(np.abs(np.concatenate([
                narrow[row].ravel(), prediction[row].ravel(), reference[row].ravel()
            ])), 99.0)),
            1e-8,
        )
        for ax, (data, title) in zip(axes[row], panels):
            image = ax.imshow(
                data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip
            )
            ax.set_title(
                f"{section_axis.title()} {int(section_numbers[row])} | {title}"
            )
            ax.set_xlabel("Trace")
            ax.set_ylabel("Time sample")
        fig.colorbar(image, ax=axes[row].tolist(), fraction=0.012, pad=0.01)
    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.05, top=0.97, wspace=0.22, hspace=0.32)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_spectra(output, spectra):
    fig, ax = plt.subplots(figsize=(11, 5))
    styles = {
        "Low-pass input": ("tab:blue", "-"),
        "Direct wideband prediction": ("tab:red", "-"),
        "F3 wideband reference": ("black", "--"),
    }
    for label, (frequencies, amplitude) in spectra.items():
        color, style = styles[label]
        ax.plot(frequencies, amplitude, color=color, linestyle=style, lw=2, label=label)
    ax.axvspan(35, 80, color="tab:red", alpha=0.10, label="35-80 Hz")
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    prediction_dir = DATA_DIR / "预测结果"
    evaluation_dir = DATA_DIR / "评价结果"
    figure_dir = DATA_DIR.parent / "figures" / "预测评价"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    narrow = np.load(prediction_dir / f"{args.prefix}_narrow_input.npy", mmap_mode="r")
    prediction = np.load(
        prediction_dir / f"{args.prefix}_wide_prediction.npy", mmap_mode="r"
    )
    reference = np.load(
        prediction_dir / f"{args.prefix}_wide_reference.npy", mmap_mode="r"
    )
    metadata = np.load(
        prediction_dir / f"{args.prefix}_metadata.npy", allow_pickle=True
    ).item()
    section_axis = metadata["section_axis"]
    section_numbers = metadata["section_numbers"]

    baseline, baseline_spectrum = evaluate_cube(narrow, reference)
    predicted, predicted_spectrum = evaluate_cube(prediction, reference)
    reference_freq, reference_amp = average_spectrum(reference)
    per_section = []
    for idx, number in enumerate(section_numbers):
        per_section.append({
            "section_number": int(number),
            "baseline": metrics(narrow[idx], reference[idx]),
            "prediction": metrics(prediction[idx], reference[idx]),
        })

    result = {
        "prefix": args.prefix,
        "section_axis": section_axis,
        "section_numbers": section_numbers,
        "baseline": baseline,
        "prediction": predicted,
        "per_section": per_section,
    }
    np.save(evaluation_dir / f"{args.prefix}_metrics.npy", result)

    section_png = figure_dir / f"{args.prefix}_sections.png"
    spectrum_png = figure_dir / f"{args.prefix}_spectra.png"
    plot_sections(
        section_png, section_axis, section_numbers, narrow, prediction, reference
    )
    plot_spectra(spectrum_png, {
        "Low-pass input": baseline_spectrum,
        "Direct wideband prediction": predicted_spectrum,
        "F3 wideband reference": (reference_freq, reference_amp),
    })

    report = [
        f"Evaluation prefix: {args.prefix}",
        f"Section axis: {section_axis}",
        f"Section numbers: {', '.join(str(x) for x in section_numbers)}",
        "",
        format_metrics("Low-pass input vs reference", baseline),
        "",
        format_metrics("Direct wideband prediction vs reference", predicted),
        "",
        f"Per-{section_axis} spatial metrics:",
        f"  {section_axis}, baseline_MAE, prediction_MAE, baseline_RMSE, "
        "prediction_RMSE, baseline_Correlation, prediction_Correlation",
    ]
    for row in per_section:
        base = row["baseline"]
        pred = row["prediction"]
        report.append(
            f"  {row['section_number']}, {base['MAE']:.6f}, {pred['MAE']:.6f}, "
            f"{base['RMSE']:.6f}, {pred['RMSE']:.6f}, "
            f"{base['Correlation']:.6f}, {pred['Correlation']:.6f}"
        )
    report.extend([
        "",
        f"Section figure: {section_png}",
        f"Spectrum figure: {spectrum_png}",
    ])
    report_path = figure_dir / f"{args.prefix}_evaluation_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
