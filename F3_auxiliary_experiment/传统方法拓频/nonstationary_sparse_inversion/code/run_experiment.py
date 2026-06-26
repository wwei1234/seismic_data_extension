import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from literature_baselines import run_algorithm_experiment  # noqa: E402


if __name__ == "__main__":
    run_algorithm_experiment(
        ROOT,
        "nonstationary_sparse_inversion",
        "Nonstationary Sparse Inversion Approximation",
        "Based on nonstationary sparse inversion; this run uses stronger time-varying Q compensation and sparse high-frequency residual retention.",
    )
