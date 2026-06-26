import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from literature_baselines import run_algorithm_experiment  # noqa: E402


if __name__ == "__main__":
    run_algorithm_experiment(
        ROOT,
        "fractional_time_frequency",
        "Fractional Time-Frequency Spectral Extension",
        "Based on fractional time-frequency spectral extension; this run increases the fractional derivative order and high-frequency gain cap.",
    )
