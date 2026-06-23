import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from hybrid_dataset import build_source_schedule  # noqa: E402


def test_source_schedule_uses_seventy_percent_real_samples():
    schedule = build_source_schedule(1000, real_probability=0.7)

    real_fraction = sum(source == "real" for source in schedule) / len(schedule)
    assert 0.69 <= real_fraction <= 0.71
    assert set(schedule) == {"real", "synthetic"}
