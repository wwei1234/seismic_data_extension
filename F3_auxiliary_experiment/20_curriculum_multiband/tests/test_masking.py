import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from masking import make_masked_pair  # noqa: E402


def make_signal(frequencies, dt=0.004, samples=256):
    time = np.arange(samples, dtype=np.float32) * dt
    signal = sum(np.sin(2.0 * np.pi * frequency * time) for frequency in frequencies)
    return np.repeat(signal[:, None], 8, axis=1).astype(np.float32)


def spectral_energy(data, low_hz, high_hz, dt):
    spectrum = np.abs(np.fft.rfft(data, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(data.shape[0], d=dt)
    selected = (frequencies >= low_hz) & (frequencies <= high_hz)
    return float(spectrum[selected].sum() / (spectrum.sum() + 1e-12))


def test_lowpass_task_closes_to_known_target():
    known = make_signal([8, 18, 27])
    pair = make_masked_pair(
        known,
        dt=0.004,
        task_name="C",
        rng=np.random.default_rng(7),
    )
    assert np.allclose(
        pair.input_norm + pair.label_norm,
        pair.target_norm,
        atol=2e-5,
    )
    assert spectral_energy(pair.label_norm, 36, 100, 0.004) < 1e-5


def test_bandstop_label_is_only_removed_known_band():
    known = make_signal([10, 18, 26, 33])
    pair = make_masked_pair(
        known,
        dt=0.004,
        task_name="D",
        rng=np.random.default_rng(11),
    )
    assert np.allclose(
        pair.input_norm + pair.label_norm,
        pair.target_norm,
        atol=2e-5,
    )
    assert pair.target_low < pair.target_high <= 35.0


def test_noise_changes_input_but_not_clean_closure():
    known = make_signal([10, 20, 30])
    clean = make_masked_pair(
        known,
        dt=0.004,
        task_name="B",
        rng=np.random.default_rng(3),
        noise_level=0.0,
    )
    noisy = make_masked_pair(
        known,
        dt=0.004,
        task_name="B",
        rng=np.random.default_rng(3),
        noise_level=0.03,
    )
    assert np.allclose(clean.label_norm, noisy.label_norm)
    assert np.allclose(clean.target_norm, noisy.target_norm)
    assert not np.allclose(clean.input_norm, noisy.input_norm)
    assert np.allclose(
        noisy.clean_input_norm + noisy.label_norm,
        noisy.target_norm,
        atol=2e-5,
    )
