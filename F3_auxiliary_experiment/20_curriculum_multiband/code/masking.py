from dataclasses import dataclass

import numpy as np

from config import F3_MASK_TASKS


@dataclass(frozen=True)
class MaskedPair:
    input_norm: np.ndarray
    clean_input_norm: np.ndarray
    label_norm: np.ndarray
    target_norm: np.ndarray
    scale: float
    task_name: str
    target_low: float
    target_high: float
    projector: dict


def _cosine_falloff(frequencies, pass_hz, stop_hz):
    mask = np.ones_like(frequencies, dtype=np.float64)
    mask[frequencies >= stop_hz] = 0.0
    transition = (frequencies > pass_hz) & (frequencies < stop_hz)
    position = (frequencies[transition] - pass_hz) / max(stop_hz - pass_hz, 1e-8)
    mask[transition] = 0.5 * (1.0 + np.cos(np.pi * position))
    return mask


def _fft_filter(data, dt, mask):
    spectrum = np.fft.rfft(np.asarray(data, dtype=np.float64), axis=0)
    filtered = np.fft.irfft(
        spectrum * mask[:, None],
        n=data.shape[0],
        axis=0,
    )
    return filtered.astype(np.float32)


def _lowpass(data, dt, pass_hz, stop_hz):
    frequencies = np.fft.rfftfreq(data.shape[0], d=dt)
    return _fft_filter(data, dt, _cosine_falloff(frequencies, pass_hz, stop_hz))


def _bandstop(data, dt, low_hz, high_hz, transition_hz=1.0):
    frequencies = np.fft.rfftfreq(data.shape[0], d=dt)
    low_rise = 1.0 - _cosine_falloff(
        frequencies,
        max(low_hz - transition_hz, 0.0),
        low_hz,
    )
    high_fall = _cosine_falloff(
        frequencies,
        high_hz,
        high_hz + transition_hz,
    )
    removed_band = low_rise * high_fall
    return _fft_filter(data, dt, 1.0 - removed_band)


def _projector(low_hz, high_hz):
    return {
        "low_stop": max(float(low_hz) - 2.0, 0.0),
        "low_pass": float(low_hz),
        "high_pass": float(high_hz),
        "high_stop": min(float(high_hz) + 2.0, 35.0),
    }


def make_masked_pair(
    known_narrow,
    dt,
    task_name,
    rng,
    noise_level=0.0,
):
    if task_name not in F3_MASK_TASKS:
        raise ValueError(f"Unknown F3 masking task: {task_name}")
    known_narrow = np.asarray(known_narrow, dtype=np.float32)
    known_target = _lowpass(known_narrow, dt, pass_hz=34.0, stop_hz=35.0)
    scale = max(float(np.percentile(np.abs(known_target), 99)), 1e-8)
    task = F3_MASK_TASKS[task_name]

    if task["kind"] == "lowpass":
        input_pass = float(rng.uniform(*task["stop"]))
        input_stop = float(rng.uniform(*task["pass"]))
        if input_stop <= input_pass:
            input_stop = input_pass + 2.0
        target_high = float(task["target_high"])
        target = _lowpass(
            known_target,
            dt,
            pass_hz=max(target_high - 1.0, input_stop),
            stop_hz=target_high,
        )
        clean_input = _lowpass(
            target,
            dt,
            pass_hz=input_pass,
            stop_hz=input_stop,
        )
        target_low = input_pass
    else:
        width = float(rng.uniform(*task["width"]))
        center = float(rng.uniform(*task["center"]))
        target_low = max(center - width / 2.0, 4.0)
        target_high = min(center + width / 2.0, 35.0)
        target = known_target
        clean_input = _bandstop(target, dt, target_low, target_high)

    clean_input_norm = (clean_input / scale).astype(np.float32)
    target_norm = (target / scale).astype(np.float32)
    label_norm = (target_norm - clean_input_norm).astype(np.float32)
    input_norm = clean_input_norm.copy()
    if noise_level > 0:
        noise_scale = float(np.std(clean_input_norm)) * float(noise_level)
        input_norm += rng.normal(
            0.0,
            noise_scale,
            size=input_norm.shape,
        ).astype(np.float32)

    return MaskedPair(
        input_norm=input_norm,
        clean_input_norm=clean_input_norm,
        label_norm=label_norm,
        target_norm=target_norm,
        scale=scale,
        task_name=task_name,
        target_low=float(target_low),
        target_high=float(target_high),
        projector=_projector(target_low, target_high),
    )
