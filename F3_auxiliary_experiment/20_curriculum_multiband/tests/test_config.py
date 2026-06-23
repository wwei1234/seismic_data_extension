import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from config import (  # noqa: E402
    CURRICULUM_STAGES,
    F3_MASK_TASKS,
    FINAL_PROJECTOR,
    SYNTHETIC_NOISE_LEVELS,
    TOTAL_EPOCHS,
)


def test_curriculum_configuration_is_locked():
    assert TOTAL_EPOCHS == 300
    assert [
        (stage["start"], stage["end"], stage["f3_ratio"], stage["synthetic_ratio"])
        for stage in CURRICULUM_STAGES
    ] == [
        (1, 60, 1, 0),
        (61, 180, 2, 1),
        (181, 300, 1, 1),
    ]
    assert set(F3_MASK_TASKS) == {"A", "B", "C", "D"}
    assert FINAL_PROJECTOR == {
        "low_stop": 32.0,
        "low_pass": 38.0,
        "high_pass": 85.0,
        "high_stop": 100.0,
    }
    assert SYNTHETIC_NOISE_LEVELS == (0.01, 0.03)
