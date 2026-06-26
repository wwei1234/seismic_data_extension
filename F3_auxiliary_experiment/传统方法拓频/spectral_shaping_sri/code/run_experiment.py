import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from literature_baselines import run_algorithm_experiment  # noqa: E402


if __name__ == "__main__":
    run_algorithm_experiment(
        ROOT,
        "spectral_shaping_sri",
        "Spectral Shaping with Shaping-Regularized Inversion",
        "Based on spectral shaping regularized inversion; this run uses a more aggressive target band and band-limited detail injection.",
    )
