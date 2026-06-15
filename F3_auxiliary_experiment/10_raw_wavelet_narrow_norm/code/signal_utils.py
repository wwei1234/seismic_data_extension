import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.signal.windows import tukey


# ── Basic signal utilities ───────────────────────────────────────────────────


def normalize_max_abs(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    m = np.nanmax(np.abs(x))
    if not np.isfinite(m) or m < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / m).astype(np.float32)


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


def average_amplitude_spectrum(section, dt):
    section = np.asarray(section, dtype=np.float64)
    work = section - np.mean(section, axis=0, keepdims=True)
    spec = np.fft.rfft(work, axis=0)
    amp = np.mean(np.abs(spec), axis=1)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    if np.max(amp) > 0:
        amp = amp / np.max(amp)
    return freqs, amp


def convolve_reflectivity(reflectivity_2d, wavelet):
    reflectivity_2d = np.asarray(reflectivity_2d, dtype=np.float32)
    wavelet = np.asarray(wavelet, dtype=np.float32)
    out = np.zeros_like(reflectivity_2d, dtype=np.float32)
    for ix in range(reflectivity_2d.shape[1]):
        out[:, ix] = fftconvolve(reflectivity_2d[:, ix], wavelet, mode="same")
    return out


# ── Q filtering (time-variant attenuation) ───────────────────────────────────


def apply_time_variant_q_filter_trace(trace, dt, q=85.0, strength=0.35, window=96, hop=24):
    trace = np.asarray(trace, dtype=np.float64)
    n = trace.size
    out = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    win = tukey(window, alpha=0.35)
    freqs = np.fft.rfftfreq(window, dt)
    starts = list(range(0, max(1, n - window + 1), hop))
    if starts[-1] != n - window:
        starts.append(n - window)
    for start in starts:
        stop = start + window
        segment = trace[start:stop] * win
        center_time = (start + window / 2.0) * dt
        attenuation = np.exp(-strength * np.pi * freqs * center_time / max(q, 1e-6))
        filtered = np.fft.irfft(np.fft.rfft(segment) * attenuation, n=window)
        out[start:stop] += filtered * win
        weights[start:stop] += win ** 2
    return (out / np.maximum(weights, 1e-8)).astype(np.float32)


def apply_time_variant_q_filter_section(section, dt, q=85.0, strength=0.35, window=96, hop=24):
    section = np.asarray(section, dtype=np.float32)
    out = np.zeros_like(section, dtype=np.float32)
    for ix in range(section.shape[1]):
        out[:, ix] = apply_time_variant_q_filter_trace(
            section[:, ix], dt, q=q, strength=strength, window=window, hop=hop
        )
    return out


# ── Low-frequency background removal ─────────────────────────────────────────


def remove_lowfreq_background(section, dt=0.004, f_cut=8.0, order=4):
    """
    Zero-phase Butterworth highpass filter to remove low-frequency swell
    caused by overly low wavelet low-cut frequency.

    Parameters
    ----------
    section : ndarray (nt, nx)
    dt : float
    f_cut : float
        Cutoff frequency in Hz.
    order : int
        Filter order.
    """
    nyq = 0.5 / dt
    normal_cut = f_cut / nyq
    b, a = butter(order, normal_cut, btype='high')
    result = np.zeros_like(section, dtype=np.float64)
    for ix in range(section.shape[1]):
        result[:, ix] = filtfilt(b, a, section[:, ix].astype(np.float64))
    return result.astype(np.float32)
