import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from curriculum import GatedCheckpointSelector, domain_cycle  # noqa: E402


def test_domain_schedule_matches_three_stages():
    assert domain_cycle(20) == ("f3",)
    assert domain_cycle(100) == ("f3", "f3", "synthetic")
    assert domain_cycle(240) == ("f3", "synthetic")


def test_gate_rejects_good_synthetic_but_bad_f3():
    selector = GatedCheckpointSelector()
    rejected = selector.consider(
        epoch=120,
        f3={"correlation": 0.84, "phase": 0.90, "leakage": 0.01},
        synthetic={"residual_correlation": 0.95},
    )
    assert rejected is False


def test_gate_prefers_synthetic_after_f3_thresholds():
    selector = GatedCheckpointSelector()
    assert selector.consider(
        120,
        {"correlation": 0.87, "phase": 0.83, "leakage": 0.02},
        {"residual_correlation": 0.70},
    )
    assert selector.consider(
        121,
        {"correlation": 0.86, "phase": 0.82, "leakage": 0.02},
        {"residual_correlation": 0.74},
    )
    assert selector.best_epoch == 121


def test_gate_uses_f3_correlation_for_synthetic_ties():
    selector = GatedCheckpointSelector()
    assert selector.consider(
        100,
        {"correlation": 0.86, "phase": 0.82, "leakage": 0.02},
        {"residual_correlation": 0.75},
    )
    assert selector.consider(
        101,
        {"correlation": 0.89, "phase": 0.84, "leakage": 0.01},
        {"residual_correlation": 0.755},
    )
    assert selector.best_epoch == 101
