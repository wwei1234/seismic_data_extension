from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from config import DT, NARROW_BAND


@dataclass(frozen=True)
class SyntheticSample:
    input_norm: np.ndarray
    clean_narrow_norm: np.ndarray
    label_norm: np.ndarray
    wide_norm: np.ndarray
    scale: float
    scale_source: str


def normalize_wavelet(wavelet):
    wavelet = np.asarray(wavelet, dtype=np.float64)
    wavelet = wavelet - np.mean(wavelet)
    maximum = float(np.max(np.abs(wavelet)))
    if maximum > 0:
        wavelet = wavelet / maximum
    return wavelet.astype(np.float32)


def build_bspline_wavelet(length, band, dt=DT):
    f1, f2, f3, f4 = band
    n_fft = 512
    nyquist = 0.5 / dt
    frequencies = np.fft.rfftfreq(n_fft, dt)
    x = np.asarray([0.0, f1, f2, (f2 + f3) / 2.0, f3, f4, nyquist])
    y = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    amplitude = np.clip(
        make_interp_spline(x, y, k=3, bc_type="clamped")(frequencies),
        0.0,
        1.0,
    )
    full = np.fft.fftshift(np.fft.irfft(amplitude, n=n_fft))
    center = n_fft // 2
    half = length // 2
    return normalize_wavelet(full[center - half:center + half + 1])


def lowpass_section(section, dt=DT, band=NARROW_BAND):
    frequencies = np.fft.rfftfreq(section.shape[0], d=dt)
    f1, f2, f3, f4 = band
    mask = np.zeros_like(frequencies)
    rise = (frequencies >= f1) & (frequencies < f2)
    keep = (frequencies >= f2) & (frequencies <= f3)
    fall = (frequencies > f3) & (frequencies <= f4)
    mask[rise] = (frequencies[rise] - f1) / max(f2 - f1, 1e-8)
    mask[keep] = 1.0
    mask[fall] = (f4 - frequencies[fall]) / max(f4 - f3, 1e-8)
    spectrum = np.fft.rfft(section, axis=0)
    return np.fft.irfft(
        spectrum * mask[:, None],
        n=section.shape[0],
        axis=0,
    ).astype(np.float32)


def make_normalized_sample(clean_wide, noise_level, rng):
    clean_wide = np.asarray(clean_wide, dtype=np.float32)
    clean_narrow = lowpass_section(clean_wide)
    scale = max(float(np.percentile(np.abs(clean_narrow), 99)), 1e-8)
    clean_narrow_norm = (clean_narrow / scale).astype(np.float32)
    wide_norm = (clean_wide / scale).astype(np.float32)
    label_norm = (wide_norm - clean_narrow_norm).astype(np.float32)
    input_norm = clean_narrow_norm.copy()
    if noise_level > 0:
        input_norm += rng.normal(
            0.0,
            float(noise_level) * max(float(np.std(clean_narrow_norm)), 1e-8),
            size=input_norm.shape,
        ).astype(np.float32)
    return SyntheticSample(
        input_norm=input_norm,
        clean_narrow_norm=clean_narrow_norm,
        label_norm=label_norm,
        wide_norm=wide_norm,
        scale=scale,
        scale_source="p99_abs_clean_narrow",
    )


def build_test_sample(noise_level=0.03, seed=5):
    rng = np.random.default_rng(seed)
    time = np.arange(256, dtype=np.float32) * DT
    trace = (
        np.sin(2 * np.pi * 12 * time)
        + 0.6 * np.sin(2 * np.pi * 28 * time)
        + 0.3 * np.sin(2 * np.pi * 58 * time)
    )
    wide = np.repeat(trace[:, None], 32, axis=1)
    return make_normalized_sample(wide, noise_level, rng)


def grouped_section_split(metadata, val_fraction=0.2, seed=42):
    section_ids = sorted({str(row["section_id"]) for row in metadata})
    rng = np.random.default_rng(seed)
    rng.shuffle(section_ids)
    val_count = max(1, int(round(len(section_ids) * val_fraction)))
    val_ids = set(section_ids[:val_count])
    train = []
    val = []
    for row in metadata:
        (val if str(row["section_id"]) in val_ids else train).append(row)
    return train, val


def build_linear_well_section(reflectivities, wells, matches, width=951, crossline_min=300):
    ordered = sorted(wells, key=lambda name: matches[name]["crossline"])
    positions = np.asarray(
        [matches[name]["crossline"] - crossline_min for name in ordered],
        dtype=np.float64,
    )
    positions = np.clip(positions, 0, width - 1)
    n_time = min(len(reflectivities[name]) for name in ordered)
    query = np.arange(width, dtype=np.float64)
    section = np.zeros((n_time, width), dtype=np.float32)
    for time_index in range(n_time):
        values = np.asarray(
            [reflectivities[name][time_index] for name in ordered],
            dtype=np.float64,
        )
        section[time_index] = np.interp(
            query,
            positions,
            values,
            left=values[0],
            right=values[-1],
        )
    return section, ordered, positions


def add_structural_perturbation(section, rng, start_sample=120, full_sample=240):
    n_time, width = section.shape
    x = np.arange(width, dtype=np.float32)
    time = np.arange(n_time, dtype=np.float32)
    center = (width - 1) / 2.0
    weight = np.clip(
        (time - start_sample) / max(full_sample - start_sample, 1),
        0.0,
        1.0,
    )
    shifts = rng.uniform(-0.025, 0.025) * (x - center)
    shifts += rng.uniform(-10.0, 10.0) * ((x - center) / max(center, 1.0)) ** 2
    for _ in range(int(rng.integers(1, 4))):
        shifts += rng.uniform(2.0, 9.0) * np.sin(
            2 * np.pi * x / rng.uniform(260.0, 750.0)
            + rng.uniform(0.0, 2 * np.pi)
        )
    shifts += gaussian_filter1d(rng.normal(0.0, 1.2, width), sigma=10.0)
    shift_2d = weight[:, None] * shifts[None, :]

    if rng.random() < 0.45:
        t0 = int(rng.integers(full_sample, max(full_sample + 1, n_time - 50)))
        x0 = float(rng.uniform(0.2 * width, 0.8 * width))
        dip = float(rng.uniform(-0.4, 0.4))
        throw = float(rng.uniform(3.0, 9.0) * rng.choice([-1, 1]))
        fault_x = x0 + dip * (time - t0)
        active = time >= t0
        side = x[None, :] >= fault_x[:, None]
        taper = np.clip((time - t0) / 60.0, 0.0, 1.0)
        shift_2d += (active[:, None] & side) * taper[:, None] * throw

    output = np.zeros_like(section, dtype=np.float32)
    for trace_index in range(width):
        output[:, trace_index] = np.interp(
            time + shift_2d[:, trace_index],
            time,
            section[:, trace_index],
            left=0.0,
            right=0.0,
        )
    return uniform_filter1d(output, size=5, axis=1).astype(np.float32)
