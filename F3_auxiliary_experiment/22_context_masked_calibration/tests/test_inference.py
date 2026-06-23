import importlib.util
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_recombination_is_narrow_plus_predicted_residual():
    module = load_script("predict_heldout", "05_predict_heldout.py")
    narrow = np.linspace(-1.0, 1.0, 256, dtype=np.float32)[:, None]
    residual = np.full((256, 1), 0.25, dtype=np.float32)

    direct, highpass = module.recombine(narrow, residual)

    np.testing.assert_allclose(direct, narrow + residual, atol=1e-6)
    assert direct.shape == narrow.shape
    assert highpass.shape == narrow.shape
