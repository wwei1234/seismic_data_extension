import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

spec = importlib.util.spec_from_file_location(
    "predict_locked",
    CODE_DIR / "04_predict_locked_f3.py",
)
predict_locked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_locked)


def low_band_nrmse(first, second, dt=0.004, high_hz=22.0):
    frequencies = np.fft.rfftfreq(first.shape[0], dt)
    mask = frequencies <= high_hz
    first_low = np.fft.irfft(
        np.fft.rfft(first, axis=0) * mask[:, None],
        n=first.shape[0],
        axis=0,
    )
    second_low = np.fft.irfft(
        np.fft.rfft(second, axis=0) * mask[:, None],
        n=second.shape[0],
        axis=0,
    )
    return float(
        np.sqrt(np.mean((first_low - second_low) ** 2))
        / (np.sqrt(np.mean(second_low ** 2)) + 1e-8)
    )


def test_prediction_refuses_unlocked_checkpoint(tmp_path):
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"model")
    with pytest.raises((FileNotFoundError, ValueError)):
        predict_locked.validate_before_reference_read(
            checkpoint,
            tmp_path / "model_lock.json",
        )


def test_direct_and_highpass_keep_narrow_body():
    rng = np.random.default_rng(3)
    narrow = rng.normal(size=(256, 16)).astype(np.float32)
    residual = rng.normal(size=(256, 16)).astype(np.float32)
    direct, highpass = predict_locked.recombine(narrow, residual, dt=0.004)
    assert np.allclose(direct, narrow + residual)
    assert low_band_nrmse(highpass, narrow, high_hz=22.0) < 1e-5
