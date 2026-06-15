import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "predictions"
DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"

DT = 0.004


def trapezoid_band(freqs, f1, f2, f3, f4):
    shape = np.zeros_like(freqs, dtype=np.float64)
    up = (freqs >= f1) & (freqs < f2)
    keep = (freqs >= f2) & (freqs <= f3)
    down = (freqs > f3) & (freqs <= f4)
    if f2 > f1:
        shape[up] = (freqs[up] - f1) / (f2 - f1)
    shape[keep] = 1.0
    if f4 > f3:
        shape[down] = (f4 - freqs[down]) / (f4 - f3)
    return np.clip(shape, 0.0, 1.0)


def zero_phase_filter_section(section, dt, band):
    section = np.asarray(section, dtype=np.float64)
    spec = np.fft.rfft(section, axis=0)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    filt = trapezoid_band(freqs, *band)[:, None]
    return np.fft.irfft(spec * filt, n=section.shape[0], axis=0).astype(np.float32)


def highpass_section(section, dt, cutoff, taper=6.0):
    f1 = max(0.0, cutoff - taper)
    f2 = cutoff
    nyquist = 0.5 / dt
    return zero_phase_filter_section(section, dt, (f1, f2, nyquist, nyquist))


def average_amplitude_spectrum(section, dt):
    section = np.asarray(section, dtype=np.float64)
    work = section - np.mean(section, axis=0, keepdims=True)
    spec = np.fft.rfft(work, axis=0)
    amp = np.mean(np.abs(spec), axis=1)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    if np.max(amp) > 0:
        amp = amp / np.max(amp)
    return freqs, amp


def average_phase_spectrum(section, dt, amp_threshold=0.05, fmax=100.0):
    section = np.asarray(section, dtype=np.float64)
    work = section - np.mean(section, axis=0, keepdims=True)
    spec = np.fft.rfft(work, axis=0)
    mean_spec = np.mean(spec, axis=1)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    amp = np.mean(np.abs(spec), axis=1)
    if np.max(amp) > 0:
        amp = amp / np.max(amp)
    phase = np.angle(mean_spec)
    valid = (freqs >= 1.0) & (freqs <= fmax) & (amp >= amp_threshold)
    return freqs, phase, valid


def metrics(pred, target):
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(pred.ravel(), target.ravel())[0, 1])
    peak = max(float(np.max(np.abs(pred))), float(np.max(np.abs(target))), 1e-8)
    psnr = float(20 * np.log10(peak / max(rmse, 1e-12)))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "PSNR": psnr, "Correlation": corr}


def high_freq_ratio(freqs, amp, band=(35.0, 80.0)):
    mask_high = (freqs >= band[0]) & (freqs <= band[1])
    mask_all = (freqs >= 1.0) & (freqs <= band[1])
    return float(np.sum(amp[mask_high]) / (np.sum(amp[mask_all]) + 1e-12))


def cube_to_time_trace_matrix(cube):
    return cube.transpose(1, 0, 2).reshape(cube.shape[1], -1)


def apply_correction(pred_path, narrow_path, target_path, out_prefix, cutoff, taper):
    pred = np.load(pred_path, mmap_mode="r")
    narrow = np.load(narrow_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")

    out_path = DATA_DIR / f"{out_prefix}_corrected_cutoff_{cutoff:g}.npy"
    corrected = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=pred.shape
    )

    for idx in range(pred.shape[0]):
        pred_high = highpass_section(pred[idx], DT, cutoff=cutoff, taper=taper)
        corrected[idx] = narrow[idx].astype(np.float32) + pred_high
        if (idx + 1) % 50 == 0 or idx == pred.shape[0] - 1:
            print(f"Corrected inline {idx + 1}/{pred.shape[0]}")
    corrected.flush()
    return out_path


def evaluate_and_plot(corrected_path, pred_path, narrow_path, target_path, out_prefix, cutoff):
    corrected = np.load(corrected_path, mmap_mode="r")
    pred = np.load(pred_path, mmap_mode="r")
    narrow = np.load(narrow_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")

    metric_dict = {
        "narrow_vs_reference": metrics(narrow, target),
        "network_vs_reference": metrics(pred, target),
        "corrected_vs_reference": metrics(corrected, target),
    }

    freq_n, amp_n = average_amplitude_spectrum(cube_to_time_trace_matrix(narrow), DT)
    freq_p, amp_p = average_amplitude_spectrum(cube_to_time_trace_matrix(pred), DT)
    freq_c, amp_c = average_amplitude_spectrum(cube_to_time_trace_matrix(corrected), DT)
    freq_t, amp_t = average_amplitude_spectrum(cube_to_time_trace_matrix(target), DT)

    metric_dict["spectrum"] = {
        "narrow_high_freq_ratio_35_80": high_freq_ratio(freq_n, amp_n),
        "network_high_freq_ratio_35_80": high_freq_ratio(freq_p, amp_p),
        "corrected_high_freq_ratio_35_80": high_freq_ratio(freq_c, amp_c),
        "reference_high_freq_ratio_35_80": high_freq_ratio(freq_t, amp_t),
        "network_spectrum_l1": float(np.mean(np.abs(amp_p - amp_t))),
        "corrected_spectrum_l1": float(np.mean(np.abs(amp_c - amp_t))),
        "narrow_spectrum_l1": float(np.mean(np.abs(amp_n - amp_t))),
    }

    np.save(DATA_DIR / f"{out_prefix}_metrics_cutoff_{cutoff:g}.npy", metric_dict)

    mid = corrected.shape[0] // 2
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        (narrow[mid], "Low-pass input"),
        (pred[mid], "Network output"),
        (corrected[mid], "Corrected output"),
        (target[mid], "F3 reference"),
    ]
    for ax, (data, title) in zip(axes, panels):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{out_prefix}_section_compare_cutoff_{cutoff:g}.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_n, amp_n, "b-", lw=2, label="Low-pass input")
    ax.plot(freq_p, amp_p, color="gray", lw=1.8, label="Network output")
    ax.plot(freq_c, amp_c, "r-", lw=2.2, label="Corrected output")
    ax.plot(freq_t, amp_t, "k--", lw=2, label="F3 reference")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{out_prefix}_spectrum_compare_cutoff_{cutoff:g}.png", dpi=300)
    plt.close(fig)

    freq_c_ph, phase_c, valid_c = average_phase_spectrum(cube_to_time_trace_matrix(corrected), DT)
    freq_t_ph, phase_t, valid_t = average_phase_spectrum(cube_to_time_trace_matrix(target), DT)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_c_ph[valid_c], phase_c[valid_c], "r.", ms=3, label="Corrected output")
    ax.plot(freq_t_ph[valid_t], phase_t[valid_t], "k.", ms=3, label="F3 reference")
    ax.set_xlim(0, 100)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{out_prefix}_phase_compare_cutoff_{cutoff:g}.png", dpi=300)
    plt.close(fig)

    return metric_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prefix", default="second_round_f3_full")
    parser.add_argument("--output-prefix", default="scheme1_second_round_f3_full")
    parser.add_argument("--cutoffs", default="25,30,35")
    parser.add_argument("--taper", type=float, default=6.0)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = SOURCE_DIR / f"{args.source_prefix}_output.npy"
    narrow_path = SOURCE_DIR / f"{args.source_prefix}_narrow_input.npy"
    target_path = SOURCE_DIR / f"{args.source_prefix}_wide_reference.npy"
    for path in (pred_path, narrow_path, target_path):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = {}
    for cutoff in [float(x) for x in args.cutoffs.split(",") if x.strip()]:
        print(f"Applying correction with high-pass cutoff {cutoff:g} Hz")
        corrected_path = apply_correction(
            pred_path, narrow_path, target_path, args.output_prefix, cutoff, args.taper
        )
        metrics_dict = evaluate_and_plot(
            corrected_path, pred_path, narrow_path, target_path, args.output_prefix, cutoff
        )
        summary[f"cutoff_{cutoff:g}"] = metrics_dict
        print(metrics_dict["corrected_vs_reference"])

    np.save(DATA_DIR / f"{args.output_prefix}_summary.npy", summary)
    print(f"Saved summary to {DATA_DIR / f'{args.output_prefix}_summary.npy'}")


if __name__ == "__main__":
    main()
