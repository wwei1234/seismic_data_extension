import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

sys.path.append(str(Path(__file__).resolve().parent))

from config import DATA_DIR, DT, FIGURE_DIR, NARROW_BAND, WAVELET_LENGTH_MS, WELL_POSITIONS
from signal_utils import (
    average_amplitude_spectrum,
    estimate_wavelet_frequency_domain,
    normalize_max_abs,
    shape_wavelet_to_target_spectrum,
    zero_phase_filter_trace,
)
from well_utils import compute_ai, load_well_reflectivity, load_well_trace


def corrcoef(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return 0.0
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def shift_trace(trace, shift):
    trace = np.asarray(trace, dtype=np.float64)
    out = np.zeros_like(trace)
    if shift > 0:
        out[shift:] = trace[:-shift]
    elif shift < 0:
        out[:shift] = trace[-shift:]
    else:
        out[:] = trace
    return out


def search_best_polarity_and_shift(seismic_trace, synthetic_trace, max_shift_samples=40):
    best = {
        "correlation": -np.inf,
        "polarity": 1,
        "shift_samples": 0,
        "shift_ms": 0.0,
        "aligned_synthetic": synthetic_trace,
    }

    for polarity in (1, -1):
        signed = polarity * synthetic_trace
        for shift in range(-max_shift_samples, max_shift_samples + 1):
            shifted = shift_trace(signed, shift)
            cc = corrcoef(seismic_trace, shifted)
            if np.isfinite(cc) and cc > best["correlation"]:
                best = {
                    "correlation": cc,
                    "polarity": polarity,
                    "shift_samples": shift,
                    "shift_ms": shift * DT * 1000.0,
                    "aligned_synthetic": shifted,
                }
    return best


def plot_well_estimation_steps(well_name, reflectivity, seismic_trace, wavelet, inline, crossline):
    n_time = min(reflectivity.size, seismic_trace.size)
    reflectivity = reflectivity[:n_time]
    seismic_trace = seismic_trace[:n_time]
    synthetic = fftconvolve(reflectivity, wavelet, mode="same")

    r_norm = normalize_max_abs(reflectivity)
    s_norm = normalize_max_abs(seismic_trace)
    syn_norm = normalize_max_abs(synthetic)
    wavelet_norm = normalize_max_abs(wavelet)
    raw_cc = corrcoef(s_norm, syn_norm)
    tie = search_best_polarity_and_shift(s_norm, syn_norm, max_shift_samples=40)
    aligned_syn = normalize_max_abs(tie["aligned_synthetic"])

    time_s = np.arange(n_time) * DT
    wavelet_time_ms = (np.arange(wavelet.size) - wavelet.size // 2) * DT * 1000.0

    freqs_s, amp_s = average_amplitude_spectrum(s_norm[:, None], DT)
    freqs_syn, amp_syn = average_amplitude_spectrum(syn_norm[:, None], DT)
    freqs_w, amp_w = average_amplitude_spectrum(wavelet_norm[:, None], DT)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"{well_name}: wavelet estimation from reflectivity and well-tie seismic trace\n"
        f"Seismic trace location: inline {inline}, crossline {crossline}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].plot(r_norm, time_s, "k-", lw=1.2)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("1. Time-domain reflectivity R(t)")
    axes[0, 0].set_xlabel("Normalized amplitude")
    axes[0, 0].set_ylabel("Time (s)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(s_norm, time_s, "b-", lw=1.2)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title("2. Well-tie seismic trace S(t)")
    axes[0, 1].set_xlabel("Normalized amplitude")
    axes[0, 1].set_ylabel("Time (s)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(wavelet_time_ms, wavelet_norm, "r-", lw=2)
    axes[0, 2].set_title("3. Estimated wavelet W(t)")
    axes[0, 2].set_xlabel("Time (ms)")
    axes[0, 2].set_ylabel("Normalized amplitude")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(s_norm, time_s, "b-", lw=1.2, label="Seismic trace")
    axes[1, 0].plot(aligned_syn, time_s, "r--", lw=1.2, label="Aligned synthetic")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title(
        "4. Tie check after polarity/shift search\n"
        f"raw corr={raw_cc:.3f}, best corr={tie['correlation']:.3f}, "
        f"pol={tie['polarity']:+d}, shift={tie['shift_ms']:.0f} ms"
    )
    axes[1, 0].set_xlabel("Normalized amplitude")
    axes[1, 0].set_ylabel("Time (s)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(freqs_s, amp_s, "b-", lw=2, label="Seismic trace")
    axes[1, 1].plot(freqs_syn, amp_syn, "r--", lw=2, label="Synthetic")
    axes[1, 1].set_title("5. Trace spectrum check")
    axes[1, 1].set_xlim(0, 125)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Normalized amplitude")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].plot(freqs_w, amp_w, "r-", lw=2)
    axes[1, 2].set_title("6. Estimated wavelet spectrum")
    axes[1, 2].set_xlim(0, 125)
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].set_xlabel("Frequency (Hz)")
    axes[1, 2].set_ylabel("Normalized amplitude")
    axes[1, 2].grid(True, alpha=0.3)

    fig.text(
        0.02,
        0.01,
        "Regularized deconvolution: W(f) = S(f) * conj(R(f)) / (|R(f)|^2 + water_level). "
        "Then synthetic trace is computed as R(t) * W(t). "
        "Tie check searches polarity +/- and integer time shifts to maximize correlation.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    out_path = FIGURE_DIR / f"01_wavelet_estimation_steps_{well_name}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return raw_cc, tie, out_path


def plot_ai_reflectivity(well_logs):
    n = len(well_logs)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n), squeeze=False)
    for row, (well_name, logs) in enumerate(well_logs.items()):
        depth = logs["depth"]
        ai = logs["ai"]
        reflectivity = logs["reflectivity_time"]
        time_s = np.arange(reflectivity.size) * DT
        axes[row, 0].plot(ai, depth, "k-", lw=1.0)
        axes[row, 0].invert_yaxis()
        axes[row, 0].set_title(f"{well_name} acoustic impedance")
        axes[row, 0].set_xlabel("AI")
        axes[row, 0].set_ylabel("Depth")
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].plot(normalize_max_abs(reflectivity), time_s, "r-", lw=1.0)
        axes[row, 1].invert_yaxis()
        axes[row, 1].set_title(f"{well_name} time reflectivity")
        axes[row, 1].set_xlabel("Normalized amplitude")
        axes[row, 1].set_ylabel("Time (s)")
        axes[row, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = FIGURE_DIR / "01_ai_and_reflectivity.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    wavelets = []
    wide_wavelets = {}
    narrow_wavelets = {}
    reflectivities = {}
    traces = {}
    well_logs = {}
    tie_scores = {}

    for well_name in WELL_POSITIONS:
        reflectivity, curves, _ = load_well_reflectivity(well_name)
        trace = load_well_trace(well_name)
        wavelet = estimate_wavelet_frequency_domain(
            reflectivity,
            trace,
            dt=DT,
            wavelet_length_ms=WAVELET_LENGTH_MS,
            water_level=0.03,
            smooth_sigma=1.2,
        )
        if wavelet[len(wavelet) // 2] < 0:
            wavelet = -wavelet
        wavelets.append(wavelet)
        lowpass_trace = zero_phase_filter_trace(trace, DT, NARROW_BAND)
        wide_wavelets[well_name] = shape_wavelet_to_target_spectrum(wavelet, trace, DT)
        narrow_wavelets[well_name] = shape_wavelet_to_target_spectrum(
            wide_wavelets[well_name], lowpass_trace, DT
        )
        reflectivities[well_name] = reflectivity
        traces[well_name] = trace
        well_logs[well_name] = {
            "depth": curves["DEPTH"],
            "ai": compute_ai(curves),
            "reflectivity_time": reflectivity,
            "well_trace": trace,
            "well_trace_lowpass": lowpass_trace,
        }
        pos = WELL_POSITIONS[well_name]
        raw_cc, tie, out_path = plot_well_estimation_steps(
            well_name,
            reflectivity,
            trace,
            wavelet,
            pos["inline"],
            pos["crossline"],
        )
        tie_scores[well_name] = {
            "inline": pos["inline"],
            "crossline": pos["crossline"],
            "raw_zero_lag_correlation": raw_cc,
            "best_correlation": tie["correlation"],
            "best_polarity": tie["polarity"],
            "best_shift_samples": tie["shift_samples"],
            "best_shift_ms": tie["shift_ms"],
            "figure": str(out_path),
        }

    estimated = np.mean(np.stack(wavelets, axis=0), axis=0)
    if estimated[len(estimated) // 2] < 0:
        estimated = -estimated
    estimated /= np.max(np.abs(estimated)) + 1e-8

    wide = np.mean(np.stack([wide_wavelets[name] for name in WELL_POSITIONS], axis=0), axis=0)
    narrow = np.mean(np.stack([narrow_wavelets[name] for name in WELL_POSITIONS], axis=0), axis=0)
    wide = normalize_max_abs(wide)
    narrow = normalize_max_abs(narrow)

    np.save(DATA_DIR / "well_reflectivities.npy", reflectivities)
    np.save(DATA_DIR / "well_traces.npy", traces)
    np.save(DATA_DIR / "well_ai_logs.npy", well_logs)
    np.save(DATA_DIR / "well_estimated_wavelets.npy", np.stack(wavelets, axis=0))
    np.save(DATA_DIR / "well_wide_wavelets.npy", wide_wavelets)
    np.save(DATA_DIR / "well_narrow_wavelets.npy", narrow_wavelets)
    np.save(DATA_DIR / "well_wavelet_pairs.npy", {
        well_name: {
            "wide": wide_wavelets[well_name],
            "narrow": narrow_wavelets[well_name],
        }
        for well_name in WELL_POSITIONS
    })
    np.save(DATA_DIR / "well_wavelet_tie_scores.npy", tie_scores)
    np.save(DATA_DIR / "estimated_wide_wavelet.npy", wide)
    np.save(DATA_DIR / "estimated_narrow_wavelet.npy", narrow)
    ai_fig = plot_ai_reflectivity(well_logs)

    time_ms = (np.arange(wide.size) - wide.size // 2) * DT * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for well_name in WELL_POSITIONS:
        axes[0].plot(time_ms, wide_wavelets[well_name], lw=1.2, alpha=0.55, label=f"{well_name} wide")
        axes[0].plot(time_ms, narrow_wavelets[well_name], lw=1.0, alpha=0.35, ls="--")
    axes[0].plot(time_ms, wide, "r-", lw=2.5, label="wide target")
    axes[0].plot(time_ms, narrow, "b-", lw=2.5, label="narrow input")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    freqs_w, amp_w = average_amplitude_spectrum(wide[:, None], DT)
    freqs_n, amp_n = average_amplitude_spectrum(narrow[:, None], DT)
    axes[1].plot(freqs_w, amp_w, "r-", lw=2.5, label="wide target")
    axes[1].plot(freqs_n, amp_n, "b-", lw=2.5, label="narrow input")
    axes[1].set_xlim(0, 125)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Normalized amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_wavelets_from_wells.png", dpi=300)
    plt.close(fig)

    print(f"Saved wavelets to {DATA_DIR}")
    print(f"Saved figure to {FIGURE_DIR / '01_wavelets_from_wells.png'}")
    print(f"Saved AI/reflectivity figure to {ai_fig}")
    for well_name, score in tie_scores.items():
        print(
            f"{well_name}: inline={score['inline']}, crossline={score['crossline']}, "
            f"raw_corr={score['raw_zero_lag_correlation']:.3f}, "
            f"best_corr={score['best_correlation']:.3f}, "
            f"polarity={score['best_polarity']:+d}, "
            f"shift={score['best_shift_ms']:.0f} ms, "
            f"figure={score['figure']}"
        )


if __name__ == "__main__":
    main()
