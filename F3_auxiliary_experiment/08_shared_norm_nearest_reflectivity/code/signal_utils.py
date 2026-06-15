import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.signal.windows import tukey


def normalize_max_abs(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    m = np.nanmax(np.abs(x))
    if not np.isfinite(m) or m < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / m).astype(np.float32)


def normalize_minmax(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (2.0 * (x - mn) / (mx - mn) - 1.0).astype(np.float32)


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


def zero_phase_filter_trace(trace, dt, band):
    trace = np.asarray(trace, dtype=np.float64)
    spec = np.fft.rfft(trace)
    freqs = np.fft.rfftfreq(trace.size, dt)
    filtered = np.fft.irfft(spec * trapezoid_band(freqs, *band), n=trace.size)
    return filtered.astype(np.float32)


def zero_phase_filter_section(section, dt, band):
    section = np.asarray(section, dtype=np.float64)
    spec = np.fft.rfft(section, axis=0)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    filt = trapezoid_band(freqs, *band)[:, None]
    filtered = np.fft.irfft(spec * filt, n=section.shape[0], axis=0)
    return filtered.astype(np.float32)


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


def estimate_wavelet_frequency_domain(reflectivity, seismic_trace, dt, wavelet_length_ms,
                                      water_level=0.02, smooth_sigma=1.5):
    """
    Estimate a zero-phase-like statistical wavelet from r(t) and s(t).

    This is a regularized frequency-domain deconvolution:
        W(f) = S(f) * conj(R(f)) / (|R(f)|^2 + water_level)
    """
    r = np.asarray(reflectivity, dtype=np.float64)
    s = np.asarray(seismic_trace, dtype=np.float64)
    n = min(r.size, s.size)
    r = r[:n] - np.mean(r[:n])
    s = s[:n] - np.mean(s[:n])
    r *= tukey(n, alpha=0.1)
    s *= tukey(n, alpha=0.1)

    nfft = 1 << int(np.ceil(np.log2(n * 2)))
    r_spec = np.fft.rfft(r, n=nfft)
    s_spec = np.fft.rfft(s, n=nfft)
    denom = np.abs(r_spec) ** 2
    wl = water_level * np.nanmax(denom)
    wavelet_spec = s_spec * np.conj(r_spec) / (denom + wl + 1e-12)

    amp = np.abs(wavelet_spec)
    if smooth_sigma and smooth_sigma > 0:
        amp = gaussian_filter1d(amp, sigma=smooth_sigma)
    phase = np.angle(wavelet_spec)
    wavelet_spec = amp * np.exp(1j * phase)

    full = np.fft.fftshift(np.fft.irfft(wavelet_spec, n=nfft))
    wavelet_len = int(round(wavelet_length_ms / 1000.0 / dt))
    if wavelet_len % 2 == 0:
        wavelet_len += 1
    center = nfft // 2
    half = wavelet_len // 2
    wavelet = full[center - half:center + half + 1].copy()
    wavelet -= np.mean(wavelet)
    wavelet *= tukey(wavelet.size, alpha=0.2)
    return normalize_max_abs(wavelet)


def shape_wavelet_to_band(wavelet, dt, band):
    wavelet = np.asarray(wavelet, dtype=np.float64)
    nfft = 1 << int(np.ceil(np.log2(wavelet.size * 8)))
    spec = np.fft.rfft(wavelet, n=nfft)
    freqs = np.fft.rfftfreq(nfft, dt)
    shaped = np.fft.irfft(spec * trapezoid_band(freqs, *band), n=nfft)
    shaped = np.fft.fftshift(shaped)
    half = wavelet.size // 2
    center = nfft // 2
    out = shaped[center - half:center + half + 1].copy()
    out -= np.mean(out)
    out *= tukey(out.size, alpha=0.2)
    return normalize_max_abs(out)


def shape_wavelet_to_target_spectrum(wavelet, target, dt, smooth_sigma=1.5):
    wavelet = np.asarray(wavelet, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if target.ndim == 1:
        target = target[:, None]
    nfft = 1 << int(np.ceil(np.log2(max(wavelet.size * 8, target.shape[0] * 2))))
    wavelet_spec = np.fft.rfft(wavelet, n=nfft)
    target_work = target - np.mean(target, axis=0, keepdims=True)
    target_spec = np.fft.rfft(target_work, n=nfft, axis=0)
    target_amp = np.mean(np.abs(target_spec), axis=1)
    if smooth_sigma and smooth_sigma > 0:
        target_amp = gaussian_filter1d(target_amp, sigma=smooth_sigma)
    target_amp = target_amp / (np.max(target_amp) + 1e-12)
    shaped_spec = target_amp * np.exp(1j * np.angle(wavelet_spec))
    shaped = np.fft.fftshift(np.fft.irfft(shaped_spec, n=nfft))
    half = wavelet.size // 2
    center = nfft // 2
    out = shaped[center - half:center + half + 1].copy()
    out -= np.mean(out)
    out *= tukey(out.size, alpha=0.2)
    return normalize_max_abs(out)


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
