from pathlib import Path

import numpy as np
import torch


# ── Paths ────────────────────────────────────────────────────────────────────
EXP_ROOT = Path(__file__).resolve().parents[1]
F3_ROOT = EXP_ROOT
WORKSPACE_ROOT = EXP_ROOT.parent

DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"
LOG_DIR = EXP_ROOT / "logs"

SOURCE_DATA_DIR = DATA_DIR
SEGY_PATH = WORKSPACE_ROOT / "Rawdata" / "Seismic_data.sgy"

# Wide/narrow amplitude ratio (training uses wide's p99, inference has narrow only)
_metadata_path = SOURCE_DATA_DIR / "synthetic_metadata.npy"
_meta = np.load(_metadata_path, allow_pickle=True).item() if _metadata_path.exists() else {}
R_MEDIAN = _meta.get("r_median", 1.8)

# ── Constants ────────────────────────────────────────────────────────────────
DT = 0.004
FS = 1.0 / DT                 # 250 Hz
PATCH_SIZE = 256
PATCH_STRIDE = 64
SHOTNUM = 651

HIGH_BAND = (35.0, 80.0)
RESIDUAL_BAND = (25.0, 75.0)
RESIDUAL_GAIN_BAND = (35.0, 75.0)
RESIDUAL_POST_BAND = (25.0, 35.0, 55.0, 75.0)
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)

# ── Helpers ──────────────────────────────────────────────────────────────────


def ensure_dirs():
    for path in (DATA_DIR, FIGURE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)


# ── numpy signal utilities ───────────────────────────────────────────────────


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


def bandpass_section(section, dt, band):
    """Zero-phase bandpass filter along time axis (axis=0)."""
    return zero_phase_filter_section(section, dt, band)


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
    a = amp_a[:common.size]
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


# ── Inference gain correction ────────────────────────────────────────────────


def targeted_gain_correction(residual_pred, fs=250, band_low=35, band_high=75,
                              target_ratio=0.35, max_gain=2.0):
    """
    Only apply spectral gain to the 35-75 Hz band.
    Does not touch other frequency bands.

    max_gain limits the maximum amplification factor to avoid over-amplifying noise.
    """
    R = np.fft.rfft(residual_pred, axis=0)
    freqs = np.fft.rfftfreq(residual_pred.shape[0]) * fs

    S = np.abs(R)
    total = S.mean() + 1e-8
    band_mask = (freqs >= band_low) & (freqs <= band_high)
    current_ratio = S[band_mask].mean() / total

    if current_ratio < target_ratio:
        gain = min(target_ratio / (current_ratio + 1e-8), max_gain)
        R[band_mask] *= gain

    return np.fft.irfft(R, n=residual_pred.shape[0], axis=0).astype(np.float32)


# ── PyTorch loss functions ───────────────────────────────────────────────────


def residual_band_loss(pred_residual, label_residual, dt=DT):
    """
    Constrain amplitude and phase matching within the residual band (25-75 Hz).

    By operating on the residual directly (instead of the full wide-band prediction),
    the loss naturally focuses on the infill band without being dominated by
    low-frequency energy.
    """
    nt = pred_residual.shape[-2]
    P = torch.fft.rfft(pred_residual, dim=-2)
    L = torch.fft.rfft(label_residual, dim=-2)

    freqs = torch.fft.rfftfreq(nt, d=dt, device=pred_residual.device)
    band_mask = (freqs >= 25.0) & (freqs <= 75.0)

    P_band = P[:, :, band_mask, :]
    L_band = L[:, :, band_mask, :]

    loss_amp = torch.abs(P_band.abs() - L_band.abs()).mean()

    phase_diff = torch.angle(P_band) - torch.angle(L_band)
    loss_phase = (1.0 - torch.cos(phase_diff)).mean()

    return loss_amp + 0.3 * loss_phase


def residual_energy_ratio_loss(pred_residual, label_residual, dt=DT):
    """
    Constrain the 35-75 Hz energy ratio of predicted residual to match the label.

    Uses an asymmetric (ReLU) loss: only penalizes when predicted high-frequency
    energy is BELOW the label's level. Does not penalize over-shooting.
    """
    def band_energy_ratio(x, freqs, low=35.0, high=75.0):
        S = torch.fft.rfft(x, dim=-2).abs()
        total = S.mean(dim=(2, 3), keepdim=True) + 1e-8
        band_mask = (freqs >= low) & (freqs <= high)
        band = S[:, :, band_mask, :].mean(dim=(2, 3), keepdim=True)
        return band / total

    nt = pred_residual.shape[-2]
    freqs = torch.fft.rfftfreq(nt, d=dt, device=pred_residual.device)

    ratio_pred = band_energy_ratio(pred_residual, freqs)
    ratio_label = band_energy_ratio(label_residual, freqs)

    return torch.relu(ratio_label - ratio_pred).mean()
