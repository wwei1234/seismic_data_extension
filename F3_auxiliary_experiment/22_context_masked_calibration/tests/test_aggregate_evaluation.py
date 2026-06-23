import importlib.util
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"


def load_evaluation_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_folds", CODE_DIR / "06_evaluate_folds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spectral_only_improvement_does_not_pass_success_gate():
    module = load_evaluation_module()
    aggregate = {
        "inline": {
            "baseline": {"Correlation": 0.70},
            "direct": {
                "Correlation": 0.69,
                "residual_bandpass_correlation_25_80": 0.30,
            },
        },
        "crossline": {
            "baseline": {"Correlation": 0.65},
            "direct": {
                "Correlation": 0.66,
                "residual_bandpass_correlation_25_80": 0.30,
            },
        },
    }

    assert not module.passes_success_gate(aggregate, direct_wins=8)
