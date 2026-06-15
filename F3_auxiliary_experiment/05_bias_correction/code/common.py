from pathlib import Path

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
F3_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CODE_DIR = F3_ROOT / "code"
SOURCE_DATA_DIR = F3_ROOT / "data"
SOURCE_PREDICTION_DIR = F3_ROOT / "predictions"

DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"
CHECKPOINT_DIR = EXP_ROOT / "checkpoints"
LOG_DIR = EXP_ROOT / "logs"

DT = 0.004
PATCH_SIZE = 256
PATCH_STRIDE = 64
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)
RESIDUAL_POST_BAND = (25.0, 35.0, 55.0, 75.0)


def ensure_dirs():
    for path in (DATA_DIR, FIGURE_DIR, CHECKPOINT_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


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


def minmax_with_stats(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32), mn, mx
    return (2.0 * (x - mn) / (mx - mn) - 1.0).astype(np.float32), mn, mx


def inverse_minmax(x, mn, mx):
    return ((np.asarray(x, dtype=np.float32) + 1.0) * 0.5 * (mx - mn) + mn).astype(np.float32)


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
    psnr = float(20.0 * np.log10(peak / max(rmse, 1e-12)))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse, "PSNR": psnr, "Correlation": corr}


def high_freq_ratio(freqs, amp, band=(35.0, 80.0)):
    mask_high = (freqs >= band[0]) & (freqs <= band[1])
    mask_all = (freqs >= 1.0) & (freqs <= band[1])
    return float(np.sum(amp[mask_high]) / (np.sum(amp[mask_all]) + 1e-12))


def cube_to_time_trace_matrix(cube):
    return cube.transpose(1, 0, 2).reshape(cube.shape[1], -1)


def patch_starts(n, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    starts = list(range(0, n - patch_size + 1, stride))
    if starts[-1] != n - patch_size:
        starts.append(n - patch_size)
    return starts


def blend_window(patch_size=PATCH_SIZE, edge_weight=0.15):
    center = (patch_size - 1) / 2.0
    dist = np.abs(np.arange(patch_size, dtype=np.float32) - center) / max(center, 1.0)
    one_d = edge_weight + (1.0 - edge_weight) * (1.0 - dist)
    return np.outer(one_d, one_d).astype(np.float32)
