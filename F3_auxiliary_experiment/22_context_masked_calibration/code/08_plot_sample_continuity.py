"""Compare representative sample continuity for experiments 18, 21, and 22."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
WORKSPACE_ROOT = ROOT.parent


def adjacent_metrics(section):
    correlations = []
    lags = []
    for index in range(section.shape[1] - 1):
        first = section[:, index] - section[:, index].mean()
        second = section[:, index + 1] - section[:, index + 1].mean()
        correlations.append(float(
            np.dot(first, second)
            / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-12)
        ))
        scores = []
        for lag in range(-6, 7):
            if lag < 0:
                left, right = first[-lag:], second[:lag]
            elif lag > 0:
                left, right = first[:-lag], second[lag:]
            else:
                left, right = first, second
            scores.append(float(
                np.dot(left, right)
                / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12)
            ))
        lags.append(int(np.argmax(scores) - 6))
    return {
        "median_adjacent_correlation": float(np.median(correlations)),
        "minimum_adjacent_correlation": float(np.min(correlations)),
        "nonzero_best_lag_fraction": float(np.mean(np.asarray(lags) != 0)),
    }


def find_unique(root, filename, parent_name=None):
    matches = [
        path for path in root.rglob(filename)
        if parent_name is None or path.parent.name == parent_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename} under {root}.")
    return matches[0]


def main():
    exp18 = WORKSPACE_ROOT / "18_real_domain_phase_consistent" / "data"
    exp21 = WORKSPACE_ROOT / "21_leave_one_well_calibration" / "data"
    exp22 = ROOT / "data"
    datasets = {}
    for name, root, index, parent_name in (
        ("Experiment 18", exp18, None, None),
        ("Experiment 21", exp21, 0, "fold_well1"),
        ("Experiment 22", exp22, 0, "fold_well1"),
    ):
        input_path = find_unique(root, "train_inputs.npy", parent_name)
        inputs = np.load(input_path, mmap_mode="r")
        selected = len(inputs) // 2 if index is None else index
        datasets[name] = np.asarray(inputs[selected])
    rows = [
        {"experiment": name, **adjacent_metrics(section)}
        for name, section in datasets.items()
    ]
    output_dir = ROOT / "figures" / "频谱与连续性"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sample_continuity_stats.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    labels = [row["experiment"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(
        labels, [row["median_adjacent_correlation"] for row in rows]
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Median adjacent-trace correlation")
    axes[1].bar(
        labels, [row["nonzero_best_lag_fraction"] for row in rows]
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Non-zero best-lag fraction")
    for axis in axes:
        axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "sample_continuity_comparison.png", dpi=220)
    plt.close(fig)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
