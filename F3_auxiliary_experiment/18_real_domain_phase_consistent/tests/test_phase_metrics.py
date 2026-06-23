import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from phase_metrics import (  # noqa: E402
    low_frequency_metrics,
    weighted_phase_score,
)


DT = 0.004


def sinusoid(frequency_hz, phase=0.0, nt=256, nx=8):
    time = np.arange(nt, dtype=np.float64) * DT
    trace = np.sin(2.0 * np.pi * frequency_hz * time + phase)
    return np.repeat(trace[:, None], nx, axis=1)


def test_weighted_phase_score_is_one_for_identical_signal():
    signal = sinusoid(50.0)
    assert weighted_phase_score(signal, signal, DT) > 0.999


def test_weighted_phase_score_detects_phase_reversal():
    signal = sinusoid(50.0)
    assert weighted_phase_score(signal, -signal, DT) < -0.99


def test_low_frequency_preservation_detects_exact_bypass():
    frequency_step = 1.0 / (256 * DT)
    narrow = sinusoid(16 * frequency_step)
    prediction = narrow + 0.3 * sinusoid(52 * frequency_step)

    result = low_frequency_metrics(prediction, narrow, DT)

    assert result["correlation"] > 0.999
    assert result["nrmse"] < 1e-5
