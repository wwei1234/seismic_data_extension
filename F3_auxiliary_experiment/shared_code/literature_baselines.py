import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import tukey


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(WORKSPACE_ROOT / "shared_code"))
sys.path.append(str(WORKSPACE_ROOT / "22_context_masked_calibration" / "code"))

from phase_metrics import low_frequency_metrics, residual_high_frequency_metrics, safe_correlation  # noqa: E402
from segy_reader import read_segy  # noqa: E402
from signal_utils import trapezoid_band, zero_phase_filter_section  # noqa: E402


DT = 0.004
SHOTNUM = 651
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)
SEGY_PATH = WORKSPACE_ROOT / "Rawdata" / "Seismic_data.sgy"
WELLS = {
    "well1": {"inline": 244, "crossline": 336},
    "well2": {"inline": 362, "crossline": 387},
    "well3": {"inline": 442, "crossline": 848},
    "well4": {"inline": 722, "crossline": 1007},
}


def average_spectrum(section):
    work = np.asarray(section, dtype=np.float64)
    work = work - work.mean(axis=0, keepdims=True)
    amplitude = np.abs(np.fft.rfft(work, axis=0)).mean(axis=1)
    amplitude /= amplitude.max() + 1e-12
    return np.fft.rfftfreq(section.shape[0], DT), amplitude


def select_section(cube, geometry, axis, number):
    key = "inlines" if axis == "inline" else "crosslines"
    values = np.asarray(geometry[key])
    found = np.where(values == int(number))[0]
    if not found.size:
        raise ValueError(f"Missing {axis} {number}")
    index = int(found[0])
    if axis == "inline":
        return cube[index].astype(np.float32)
    return np.transpose(cube[:, :, index], (1, 0)).astype(np.float32)


def preserve_low_frequency(prediction, narrow, cutoff=12.0, transition=6.0):
    prediction = np.asarray(prediction, dtype=np.float64)
    narrow = np.asarray(narrow, dtype=np.float64)
    freqs = np.fft.rfftfreq(prediction.shape[0], DT)
    blend = np.ones_like(freqs)
    blend[freqs <= cutoff] = 0.0
    ramp = (freqs > cutoff) & (freqs < cutoff + transition)
    blend[ramp] = (freqs[ramp] - cutoff) / transition
    pred_spec = np.fft.rfft(prediction, axis=0)
    narrow_spec = np.fft.rfft(narrow, axis=0)
    merged = narrow_spec * (1.0 - blend[:, None]) + pred_spec * blend[:, None]
    return np.fft.irfft(merged, n=prediction.shape[0], axis=0).astype(np.float32)


def high_frequency_detail(narrow, band=(24.0, 32.0, 112.0, 122.0), amount=0.75):
    """Build an aggressive but phase-consistent high-frequency proxy from narrow data."""
    data = np.asarray(narrow, dtype=np.float64)
    first = np.gradient(data, axis=0)
    second = np.gradient(first, axis=0)
    detail = 0.55 * first + 0.45 * second
    detail = zero_phase_filter_section(detail.astype(np.float32), DT, band).astype(np.float64)
    narrow_p99 = max(float(np.percentile(np.abs(data), 99)), 1e-8)
    detail_p99 = max(float(np.percentile(np.abs(detail), 99)), 1e-8)
    return (amount * narrow_p99 / detail_p99 * detail).astype(np.float32)


def inject_detail(prediction, narrow, amount=0.35, band=(32.0, 42.0, 88.0, 105.0)):
    return preserve_low_frequency(
        np.asarray(prediction, dtype=np.float32) + high_frequency_detail(narrow, band=band, amount=amount),
        narrow,
    )


def apply_frequency_gain(section, gain):
    spectrum = np.fft.rfft(section, axis=0)
    return np.fft.irfft(
        spectrum * gain[:, None],
        n=section.shape[0],
        axis=0,
    ).astype(np.float32)


def spectral_shaping_sri(narrow, target_band=(3.0, 5.0, 92.0, 118.0), cap=14.0):
    freqs, amp = average_spectrum(narrow)
    target = trapezoid_band(freqs, *target_band)
    eps = 0.006 * np.percentile(amp[amp > 0], 75)
    operator = amp * target / (amp ** 2 + eps ** 2)
    operator = gaussian_filter1d(operator, sigma=0.6)
    operator = np.clip(operator / (operator[(freqs >= 10) & (freqs <= 22)].mean() + 1e-8), 0.0, cap)
    return inject_detail(apply_frequency_gain(narrow, operator), narrow, amount=0.90)


def low_frequency_protection(narrow, cap=12.0):
    freqs, amp = average_spectrum(narrow)
    fit = (freqs >= 8.0) & (freqs <= 24.0) & (amp > 1e-4)
    slope, intercept = np.polyfit(freqs[fit], np.log(amp[fit]), 1)
    expected = np.exp(intercept + slope * freqs)
    expected *= trapezoid_band(freqs, 3.0, 5.0, 88.0, 116.0)
    gain = expected / (amp + 0.006)
    gain[freqs <= 12.0] = 1.0
    gain = gaussian_filter1d(np.clip(gain, 0.0, cap), sigma=0.7)
    return inject_detail(apply_frequency_gain(narrow, gain), narrow, amount=0.82)


def nonstationary_sparse_q(narrow, q=45.0, strength=1.05, cap=9.0):
    data = np.asarray(narrow, dtype=np.float64)
    n_time, n_trace = data.shape
    window = min(128, n_time)
    hop = max(window // 4, 1)
    win = tukey(window, alpha=0.35)
    freqs = np.fft.rfftfreq(window, DT)
    out = np.zeros_like(data)
    weight = np.zeros(n_time, dtype=np.float64)
    starts = list(range(0, max(1, n_time - window + 1), hop))
    if starts[-1] != n_time - window:
        starts.append(n_time - window)
    for start in starts:
        center_time = (start + window / 2) * DT
        gain = np.exp(strength * np.pi * freqs * center_time / max(q, 1e-6))
        gain *= trapezoid_band(freqs, 3.0, 5.0, 92.0, 120.0)
        gain[freqs <= 12.0] = 1.0
        gain = np.clip(gain, 0.0, cap)
        patch = data[start:start + window] * win[:, None]
        filtered = np.fft.irfft(np.fft.rfft(patch, axis=0) * gain[:, None], n=window, axis=0)
        out[start:start + window] += filtered * win[:, None]
        weight[start:start + window] += win ** 2
    compensated = out / np.maximum(weight[:, None], 1e-8)
    # Sparse inversion proxy: keep sharp reflectivity-like innovations but avoid
    # changing the protected low-frequency body.
    residual = compensated - zero_phase_filter_section(compensated, DT, NARROW_BAND)
    threshold = 0.04 * np.percentile(np.abs(residual), 95)
    residual = np.sign(residual) * np.maximum(np.abs(residual) - threshold, 0.0)
    return inject_detail(narrow + residual.astype(np.float32), narrow, amount=0.72)


def fractional_time_frequency(narrow, alpha=2.05, cap=12.0):
    freqs = np.fft.rfftfreq(narrow.shape[0], DT)
    f0 = 25.0
    fractional_gain = np.ones_like(freqs)
    high = freqs > f0
    fractional_gain[high] = (freqs[high] / f0) ** alpha
    fractional_gain *= trapezoid_band(freqs, 3.0, 5.0, 92.0, 118.0)
    fractional_gain[freqs <= 12.0] = 1.0
    fractional_gain = gaussian_filter1d(np.clip(fractional_gain, 0.0, cap), sigma=0.4)
    return inject_detail(apply_frequency_gain(narrow, fractional_gain), narrow, amount=1.05)


def self_supervised_idr_spectral(narrow, rounds=8, cap=12.0):
    pseudo = np.asarray(narrow, dtype=np.float32)
    for round_index in range(rounds):
        cutoff = 10.0 + 5.0 * round_index
        degraded = zero_phase_filter_section(pseudo, DT, (3.0, 6.0, cutoff, cutoff + 6.0))
        freqs, target_amp = average_spectrum(pseudo)
        _, input_amp = average_spectrum(degraded)
        gain = target_amp / (input_amp + 0.006)
        gain *= trapezoid_band(freqs, 3.0, 5.0, 90.0, 118.0)
        gain = gaussian_filter1d(np.clip(gain, 0.0, cap), sigma=0.5)
        gain[freqs <= 12.0] = 1.0
        candidate = preserve_low_frequency(apply_frequency_gain(narrow, gain), narrow)
        pseudo = (0.25 * pseudo + 0.75 * candidate).astype(np.float32)
    return inject_detail(pseudo, narrow, amount=0.88)


ALGORITHMS = {
    "spectral_shaping_sri": spectral_shaping_sri,
    "low_frequency_protection": low_frequency_protection,
    "nonstationary_sparse_inversion": nonstationary_sparse_q,
    "fractional_time_frequency": fractional_time_frequency,
    "self_supervised_idr": self_supervised_idr_spectral,
}


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


def evaluate_prediction(prediction, reference, narrow):
    result = spatial_metrics(prediction, reference)
    freqs, amp = average_spectrum(prediction)
    ref_freqs, ref_amp = average_spectrum(reference)
    high = (freqs >= 35.0) & (freqs <= 80.0)
    total = (freqs >= 1.0) & (freqs <= 80.0)
    common = freqs <= 100.0
    result.update({
        "high_freq_ratio_35_80": float(amp[high].sum() / (amp[total].sum() + 1e-12)),
        "spectrum_l1_vs_reference": float(np.mean(np.abs(
            amp[common] - np.interp(freqs[common], ref_freqs, ref_amp)
        ))),
    })
    residual = residual_high_frequency_metrics(prediction, reference, narrow, DT)
    low = low_frequency_metrics(prediction, narrow, DT)
    result.update({
        "residual_bandpass_correlation_25_80": residual["bandpass_correlation"],
        "residual_weighted_phase_score_25_80": residual["phase_score"],
        "residual_envelope_correlation_25_80": residual["envelope_correlation"],
        "low_frequency_correlation_0_22": low["correlation"],
        "low_frequency_nrmse_0_22": low["nrmse"],
    })
    return result


def plot_section(path, title, arrays):
    compared = [arrays["narrow"], arrays["prediction"], arrays["reference"]]
    clip = max(float(np.percentile(np.abs(np.concatenate([item.ravel() for item in compared])), 99)), 1e-8)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = (
        (arrays["narrow"], "Low-pass input", clip),
        (arrays["prediction"], "Algorithm prediction", clip),
        (arrays["reference"], "F3 wide reference", clip),
        (arrays["prediction"] - arrays["reference"], "Prediction error", max(float(np.percentile(np.abs(arrays["prediction"] - arrays["reference"]), 99)), 1e-8)),
    )
    for axis, (data, name, limit) in zip(axes, panels):
        axis.imshow(data, cmap="seismic", aspect="auto", vmin=-limit, vmax=limit)
        axis.set_title(name)
        axis.set_xlabel("Trace")
        axis.set_ylabel("Time sample")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spectra(path, title, arrays):
    fig, axis = plt.subplots(figsize=(10, 5))
    for key, label, style in (
        ("narrow", "Low-pass input", "b-"),
        ("prediction", "Algorithm prediction", "r-"),
        ("reference", "F3 wide reference", "k--"),
    ):
        freqs, amp = average_spectrum(arrays[key])
        axis.plot(freqs, amp, style, label=label)
    axis.axvspan(35, 80, color="tab:red", alpha=0.1)
    axis.set_xlim(0, 120)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalized average amplitude")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def mean_metrics(rows, axis, name):
    selected = [row for row in rows if row["axis"] == axis]
    keys = selected[0][name].keys()
    return {key: float(np.mean([row[name][key] for row in selected])) for key in keys}


def run_algorithm_experiment(root, algorithm_key, algorithm_name, source_summary):
    root = Path(root)
    data_dir = root / "data"
    figure_dir = root / "figures" / "预测评价"
    log_dir = root / "logs"
    for path in (data_dir, figure_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    algorithm = ALGORITHMS[algorithm_key]
    cube, geometry = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)
    rows = []
    for well_name, well in WELLS.items():
        for axis in ("inline", "crossline"):
            number = well[axis]
            reference = select_section(cube, geometry, axis, number)
            narrow = zero_phase_filter_section(reference, DT, NARROW_BAND)
            prediction = algorithm(narrow)
            prefix = f"{well_name}_{axis}"
            np.save(data_dir / f"{prefix}_narrow_input.npy", narrow)
            np.save(data_dir / f"{prefix}_prediction.npy", prediction)
            np.save(data_dir / f"{prefix}_wide_reference.npy", reference)
            arrays = {"narrow": narrow, "prediction": prediction, "reference": reference}
            row = {
                "well": well_name,
                "axis": axis,
                "section_number": int(number),
                "baseline": evaluate_prediction(narrow, reference, narrow),
                "prediction": evaluate_prediction(prediction, reference, narrow),
            }
            rows.append(row)
            plot_section(figure_dir / f"{prefix}_sections.png", f"{algorithm_name} {well_name} {axis} {number}", arrays)
            plot_spectra(figure_dir / f"{prefix}_spectra.png", f"{algorithm_name} {well_name} {axis} {number}", arrays)
            print(f"{algorithm_key} {well_name} {axis}: corr={row['prediction']['Correlation']:.6f}")
    aggregate = {
        axis: {
            "baseline": mean_metrics(rows, axis, "baseline"),
            "prediction": mean_metrics(rows, axis, "prediction"),
        }
        for axis in ("inline", "crossline")
    }
    wins = sum(row["prediction"]["Correlation"] >= row["baseline"]["Correlation"] for row in rows)
    result = {
        "algorithm_key": algorithm_key,
        "algorithm_name": algorithm_name,
        "source_summary": source_summary,
        "rows": rows,
        "aggregate": aggregate,
        "sections_not_below_baseline": int(wins),
    }
    np.save(data_dir / "aggregate_metrics.npy", result)
    (data_dir / "aggregate_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"{algorithm_name} F3 low-pass bandwidth-extension evaluation",
        "=" * 40,
        "",
        source_summary,
        "",
    ]
    for axis in ("inline", "crossline"):
        lines.append(f"{axis} aggregate")
        for name in ("baseline", "prediction"):
            metric = aggregate[axis][name]
            lines.append(
                f"  {name}: corr={metric['Correlation']:.6f}, "
                f"rmse={metric['RMSE']:.6f}, "
                f"res_corr={metric['residual_bandpass_correlation_25_80']:.6f}, "
                f"phase={metric['residual_weighted_phase_score_25_80']:.6f}, "
                f"spectrum_l1={metric['spectrum_l1_vs_reference']:.6f}, "
                f"low_nrmse={metric['low_frequency_nrmse_0_22']:.6f}"
            )
        lines.append("")
    lines.append(f"Sections not below low-pass baseline: {wins}/8")
    report = figure_dir / "evaluation_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    return result

