from pathlib import Path

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
F3_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"
LOG_DIR = EXP_ROOT / "logs"

SOURCE_DATA_DIR = F3_ROOT / "data"
BIAS_DATA_DIR = F3_ROOT / "bias_correction_experiment" / "data"

DT = 0.004
FS = 1.0 / DT
HIGH_BAND = (35.0, 80.0)
RESIDUAL_GAIN_BAND = (35.0, 75.0)
RESIDUAL_POST_BAND = (25.0, 35.0, 55.0, 75.0)
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)


def ensure_dirs():
    for path in (DATA_DIR, FIGURE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def average_patch_spectrum(path, dt=DT, max_patches=None):
    patches = np.load(path, mmap_mode="r")
    n_patches, nt, _ = patches.shape
    if max_patches is None or max_patches >= n_patches:
        indices = np.arange(n_patches)
    else:
        indices = np.linspace(0, n_patches - 1, max_patches, dtype=int)
    amp_sum = None
    trace_count = 0
    for idx in indices:
        section = np.asarray(patches[idx], dtype=np.float64)
        section = section - section.mean(axis=0, keepdims=True)
        spec = np.fft.rfft(section, axis=0)
        amp = np.abs(spec).sum(axis=1)
        amp_sum = amp if amp_sum is None else amp_sum + amp
        trace_count += section.shape[1]
    amp = amp_sum / max(trace_count, 1)
    freqs = np.fft.rfftfreq(nt, dt)
    amp = amp / (np.max(amp) + 1e-12)
    return freqs, amp


def average_cube_spectrum(path, dt=DT, max_inlines=None):
    cube = np.load(path, mmap_mode="r")
    n_inline, nt, _ = cube.shape
    if max_inlines is None or max_inlines >= n_inline:
        indices = np.arange(n_inline)
    else:
        indices = np.linspace(0, n_inline - 1, max_inlines, dtype=int)
    amp_sum = None
    trace_count = 0
    for idx in indices:
        section = np.asarray(cube[idx], dtype=np.float64)
        section = section - section.mean(axis=0, keepdims=True)
        spec = np.fft.rfft(section, axis=0)
        amp = np.abs(spec).sum(axis=1)
        amp_sum = amp if amp_sum is None else amp_sum + amp
        trace_count += section.shape[1]
    amp = amp_sum / max(trace_count, 1)
    freqs = np.fft.rfftfreq(nt, dt)
    amp = amp / (np.max(amp) + 1e-12)
    return freqs, amp


def high_freq_ratio(freqs, amp, band=HIGH_BAND):
    mask_high = (freqs >= band[0]) & (freqs <= band[1])
    mask_all = (freqs >= 1.0) & (freqs <= band[1])
    return float(np.sum(amp[mask_high]) / (np.sum(amp[mask_all]) + 1e-12))


def spectrum_l1(freq_a, amp_a, freq_b, amp_b, fmax=100.0):
    common = freq_a[freq_a <= fmax]
    a = amp_a[: common.size]
    b = np.interp(common, freq_b, amp_b)
    return float(np.mean(np.abs(a - b)))


def metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(pred.ravel(), target.ravel())[0, 1])
    peak = max(float(np.max(np.abs(pred))), float(np.max(np.abs(target))), 1e-8)
    psnr = float(20.0 * np.log10(peak / max(rmse, 1e-12)))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "PSNR": psnr, "Correlation": corr}


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
