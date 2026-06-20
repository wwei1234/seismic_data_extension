import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
sys.path.insert(0, str(CODE_DIR))


def test_experiment_does_not_duplicate_shared_resources():
    assert (CODE_DIR / "config.py").exists()
    assert (CODE_DIR / "common.py").exists()
    forbidden = {
        "model.py",
        "segy_reader.py",
        "signal_utils.py",
        "01_rematch_wells_estimate_wavelets.py",
    }
    assert not forbidden.intersection(p.name for p in CODE_DIR.glob("*.py"))
    assert not (ROOT / "data" / "井数据").exists()


def test_label_is_clean_wideband_not_residual():
    from wideband_targets import prepare_training_pair

    clean_narrow = np.array([[1.0, -2.0]], dtype=np.float32)
    clean_wide = np.array([[1.5, -3.0]], dtype=np.float32)
    _, label_norm, scale = prepare_training_pair(
        clean_narrow, clean_wide, noise_level=0.0, rng=np.random.default_rng(1)
    )
    expected = np.clip(clean_wide / scale, -1.0, 1.0)
    residual = expected - np.clip(clean_narrow / scale, -1.0, 1.0)
    np.testing.assert_allclose(label_norm, expected)
    assert not np.allclose(label_norm, residual)


def test_input_noise_does_not_enter_label():
    from wideband_targets import prepare_training_pair

    clean_narrow = np.linspace(-1, 1, 16, dtype=np.float32).reshape(4, 4)
    clean_wide = clean_narrow * 1.4
    _, label_a, _ = prepare_training_pair(
        clean_narrow, clean_wide, noise_level=0.01, rng=np.random.default_rng(2)
    )
    _, label_b, _ = prepare_training_pair(
        clean_narrow, clean_wide, noise_level=0.03, rng=np.random.default_rng(3)
    )
    np.testing.assert_allclose(label_a, label_b)


def test_model_output_is_not_forced_to_zero_mean():
    from wideband_training import WidebandModel

    model = WidebandModel(base_c=4)
    model.net = torch.nn.Identity()
    x = torch.full((1, 1, 16, 16), 0.25)
    output = model(x)
    assert torch.allclose(output, x)
    assert output.mean().item() == 0.25


def test_composite_loss_is_zero_for_exact_prediction():
    from wideband_training import WidebandCompositeLoss

    criterion = WidebandCompositeLoss()
    target = torch.randn(2, 1, 32, 32)
    loss, parts = criterion(target, target)
    assert loss.item() < 1e-6
    assert set(parts) == {
        "total",
        "waveform",
        "spectrum",
        "phase",
        "gradient",
        "low_frequency",
    }


def test_prediction_is_directly_denormalized_without_residual_recombination():
    from wideband_inference import denormalize_wide_prediction

    pred_norm = np.array([[0.5, -0.25]], dtype=np.float32)
    result = denormalize_wide_prediction(pred_norm, narrow_scale=20.0)
    np.testing.assert_allclose(result, [[10.0, -5.0]])
