import numpy as np
from scipy.signal import hilbert


def temporal_filter(data, dt, low_hz, high_hz):
    data = np.asarray(data, dtype=np.float64)
    spectrum = np.fft.rfft(data, axis=-2)
    frequencies = np.fft.rfftfreq(data.shape[-2], d=dt)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    return np.fft.irfft(
        spectrum * mask.reshape((1,) * (data.ndim - 2) + (-1, 1)),
        n=data.shape[-2],
        axis=-2,
    )


def safe_correlation(first, second):
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    first -= first.mean()
    second -= second.mean()
    denominator = np.sqrt(np.sum(first ** 2) * np.sum(second ** 2))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(first * second) / denominator)


def weighted_phase_score(prediction, target, dt, low_hz=25.0, high_hz=80.0):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred_spectrum = np.fft.rfft(prediction, axis=-2)
    target_spectrum = np.fft.rfft(target, axis=-2)
    frequencies = np.fft.rfftfreq(prediction.shape[-2], d=dt)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    pred_band = pred_spectrum[..., mask, :]
    target_band = target_spectrum[..., mask, :]
    weights = np.abs(pred_band) * np.abs(target_band)
    phase_delta = np.angle(pred_band) - np.angle(target_band)
    return float(np.sum(weights * np.cos(phase_delta)) / (np.sum(weights) + 1e-12))


def bandpass_correlation(prediction, target, dt, low_hz=25.0, high_hz=80.0):
    pred_band = temporal_filter(prediction, dt, low_hz, high_hz)
    target_band = temporal_filter(target, dt, low_hz, high_hz)
    return safe_correlation(pred_band, target_band)


def envelope_correlation(prediction, target, dt, low_hz=25.0, high_hz=80.0):
    pred_band = temporal_filter(prediction, dt, low_hz, high_hz)
    target_band = temporal_filter(target, dt, low_hz, high_hz)
    pred_envelope = np.abs(hilbert(pred_band, axis=-2))
    target_envelope = np.abs(hilbert(target_band, axis=-2))
    return safe_correlation(pred_envelope, target_envelope)


def low_frequency_metrics(prediction, narrow_input, dt, high_hz=22.0):
    pred_low = temporal_filter(prediction, dt, 0.0, high_hz)
    input_low = temporal_filter(narrow_input, dt, 0.0, high_hz)
    difference = pred_low - input_low
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    reference_rms = float(np.sqrt(np.mean(input_low ** 2)))
    return {
        "correlation": safe_correlation(pred_low, input_low),
        "nrmse": rmse / max(reference_rms, 1e-12),
    }


def residual_high_frequency_metrics(prediction, target, narrow_input, dt):
    predicted_residual = np.asarray(prediction) - np.asarray(narrow_input)
    target_residual = np.asarray(target) - np.asarray(narrow_input)
    return {
        "bandpass_correlation": bandpass_correlation(
            predicted_residual,
            target_residual,
            dt,
        ),
        "phase_score": weighted_phase_score(
            predicted_residual,
            target_residual,
            dt,
        ),
        "envelope_correlation": envelope_correlation(
            predicted_residual,
            target_residual,
            dt,
        ),
    }
