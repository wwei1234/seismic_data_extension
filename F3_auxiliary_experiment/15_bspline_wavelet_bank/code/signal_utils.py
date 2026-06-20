import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter
from scipy.signal import fftconvolve
from scipy.signal.windows import tukey


# ── Basic utilities ──────────────────────────────────────────────────────────


def normalize_max_abs(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    m = np.nanmax(np.abs(x))
    if not np.isfinite(m) or m < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / m).astype(np.float32)


def trapezoid_band(freqs, f1, f2, f3, f4):
    shape = np.zeros_like(freqs, dtype=np.float64)
    up = (freqs >= f1) & (freqs < f2); keep = (freqs >= f2) & (freqs <= f3)
    down = (freqs > f3) & (freqs <= f4)
    if f2 > f1: shape[up] = (freqs[up] - f1) / (f2 - f1)
    shape[keep] = 1.0
    if f4 > f3: shape[down] = (f4 - freqs[down]) / (f4 - f3)
    return np.clip(shape, 0.0, 1.0)


def zero_phase_filter_section(section, dt, band):
    section = np.asarray(section, dtype=np.float64)
    spec = np.fft.rfft(section, axis=0)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    return np.fft.irfft(spec * trapezoid_band(freqs, *band)[:, None],
                        n=section.shape[0], axis=0).astype(np.float32)


def average_amplitude_spectrum(section, dt):
    section = np.asarray(section, dtype=np.float64)
    work = section - np.mean(section, axis=0, keepdims=True)
    spec = np.fft.rfft(work, axis=0); amp = np.mean(np.abs(spec), axis=1)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    if np.max(amp) > 0: amp = amp / np.max(amp)
    return freqs, amp


def convolve_reflectivity(reflectivity_2d, wavelet):
    reflectivity_2d = np.asarray(reflectivity_2d, dtype=np.float32)
    wavelet = np.asarray(wavelet, dtype=np.float32)
    out = np.zeros_like(reflectivity_2d, dtype=np.float32)
    for ix in range(reflectivity_2d.shape[1]):
        out[:, ix] = fftconvolve(reflectivity_2d[:, ix], wavelet, mode="same")
    return out


# ── Q filter ─────────────────────────────────────────────────────────────────


def apply_time_variant_q_filter_trace(trace, dt, q=85.0, strength=0.35, window=96, hop=24):
    trace = np.asarray(trace, dtype=np.float64); n = trace.size
    out = np.zeros(n, dtype=np.float64); weights = np.zeros(n, dtype=np.float64)
    win = tukey(window, alpha=0.35); freqs = np.fft.rfftfreq(window, dt)
    starts = list(range(0, max(1, n - window + 1), hop))
    if starts[-1] != n - window: starts.append(n - window)
    for start in starts:
        stop = start + window; segment = trace[start:stop] * win
        center_time = (start + window / 2.0) * dt
        attenuation = np.exp(-strength * np.pi * freqs * center_time / max(q, 1e-6))
        filtered = np.fft.irfft(np.fft.rfft(segment) * attenuation, n=window)
        out[start:stop] += filtered * win; weights[start:stop] += win ** 2
    return (out / np.maximum(weights, 1e-8)).astype(np.float32)


def apply_time_variant_q_filter_section(section, dt, q=85.0, strength=0.35, window=96, hop=24):
    section = np.asarray(section, dtype=np.float32)
    out = np.zeros_like(section, dtype=np.float32)
    for ix in range(section.shape[1]):
        out[:, ix] = apply_time_variant_q_filter_trace(
            section[:, ix], dt, q=q, strength=strength, window=window, hop=hop)
    return out


# ── Kriging (multi-well interpolation) ───────────────────────────────────────


def variogram_spherical(h, nugget=0.01, sill=1.0, range_=300):
    """Spherical variogram model — most common in petroleum geostatistics."""
    h = np.asarray(h, dtype=np.float64)
    r = np.where(h <= range_,
                 nugget + (sill - nugget) * (1.5 * (h / range_) - 0.5 * (h / range_) ** 3),
                 sill)
    return r


def kriging_1d(well_positions, well_values, query_positions,
               nugget=0.01, sill=1.0, range_=300):
    """Ordinary kriging: returns estimate and kriging variance."""
    n = len(well_positions)
    h_mat = np.abs(well_positions[:, None] - well_positions[None, :])
    C_mat = sill - variogram_spherical(h_mat, nugget, sill, range_)
    A = np.zeros((n + 1, n + 1)); A[:n, :n] = C_mat
    A[:n, n] = 1.0; A[n, :n] = 1.0
    mean_est = np.zeros(len(query_positions), dtype=np.float64)
    krig_var = np.zeros(len(query_positions), dtype=np.float64)
    for i, x in enumerate(query_positions):
        h_vec = np.abs(well_positions - x)
        c_vec = sill - variogram_spherical(h_vec, nugget, sill, range_)
        b = np.append(c_vec, 1.0)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.ones(n + 1) / n
        mean_est[i] = np.dot(w[:n], well_values)
        krig_var[i] = max(0.0, sill - np.dot(w[:n], c_vec) - w[n])
    return mean_est, krig_var


def sequential_gaussian_simulation(well_positions, well_values, query_positions, rng,
                                    nugget=0.01, sill=1.0, range_=300):
    """Kriging mean + random perturbation → one equiprobable realisation."""
    mean_est, krig_var = kriging_1d(well_positions, well_values, query_positions,
                                    nugget, sill, range_)
    std_est = np.sqrt(np.maximum(krig_var, 0.0))
    return mean_est + rng.normal(0.0, 1.0, len(query_positions)) * std_est


# ── Multi-well reflectivity section ──────────────────────────────────────────


def build_multiwell_section_kriging(reflectivities, well_names_subset, target_width, rng,
                                     nugget=0.01, sill=1.0, range_=300,
                                     return_info=False):
    """Build a 2D reflectivity section by kriging between selected wells."""
    n_wells = len(well_names_subset)
    n_time = min(len(reflectivities[w]) for w in well_names_subset)
    order = rng.permutation(n_wells)
    ordered_wells = [well_names_subset[i] for i in order]
    positions = np.sort(rng.uniform(0, target_width - 1, size=n_wells)).astype(np.float64)
    positions[0] = 0.0; positions[-1] = float(target_width - 1)
    x_query = np.arange(target_width, dtype=np.float64)
    section = np.zeros((n_time, target_width), dtype=np.float32)
    krig_std = np.zeros((n_time, target_width), dtype=np.float32)
    for it in range(n_time):
        rc_values = np.array([reflectivities[w][it] for w in ordered_wells], dtype=np.float64)
        if n_wells == 2:
            section[it, :] = np.interp(x_query, positions, rc_values)
        else:
            mean_est, krig_var = kriging_1d(
                positions, rc_values, x_query,
                nugget=nugget, sill=sill, range_=range_)
            section[it, :] = mean_est
            krig_std[it, :] = np.sqrt(np.maximum(krig_var, 0.0))
    if n_wells > 2:
        random_field = rng.normal(0.0, 1.0, size=section.shape)
        random_field = gaussian_filter(random_field, sigma=(7.0, 35.0), mode="reflect")
        random_field = random_field / (np.std(random_field) + 1e-8)
        section = section + 0.25 * krig_std * random_field
        section = gaussian_filter(section, sigma=(0.4, 1.2), mode="reflect")
    section = section.astype(np.float32)
    if return_info:
        return section, {
            "ordered_wells": ordered_wells,
            "well_positions": [float(x) for x in positions],
        }
    return section


# ── Amplitude envelope ───────────────────────────────────────────────────────


def extract_amplitude_envelope(section, smooth_t=30, smooth_x=100):
    """Extract local RMS amplitude envelope, normalised to mean=1."""
    rms = uniform_filter(section.astype(np.float64) ** 2, size=(smooth_t, smooth_x)) ** 0.5
    envelope = rms / (rms.mean() + 1e-8)
    return envelope.astype(np.float32)


def apply_amplitude_envelope(synthetic_wide, synthetic_narrow, f3_envelope, rng):
    """Modulate synthetic amplitudes with a random patch of the F3 envelope."""
    nt_syn, nx_syn = synthetic_wide.shape
    nt_f3, nx_f3 = f3_envelope.shape
    t_max = max(1, nt_f3 - nt_syn); x_max = max(1, nx_f3 - nx_syn)
    t0 = rng.integers(0, t_max); x0 = rng.integers(0, x_max)
    patch = f3_envelope[t0:t0 + nt_syn, x0:x0 + nx_syn]
    return (synthetic_wide * patch).astype(np.float32), (synthetic_narrow * patch).astype(np.float32)
