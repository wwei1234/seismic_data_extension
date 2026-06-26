import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from literature_baselines import run_algorithm_experiment  # noqa: E402


if __name__ == "__main__":
    run_algorithm_experiment(
        ROOT,
        "low_frequency_protection",
        "Low-Frequency Protection Spectral Extrapolation",
        "Based on low-frequency protection and spectral extrapolation; this run protects the low-frequency body and injects stronger band-limited high-frequency detail.",
    )
