"""Create the experiment 22 final report and experiment 21 comparison."""

import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from config import EVALUATION_DIR, FIGURE_DIR, WORKSPACE_ROOT  # noqa: E402


def metric_changes(current, previous, axis):
    current_direct = current["aggregate"][axis]["direct"]
    previous_direct = previous["aggregate"][axis]["direct"]
    return {
        "correlation": (
            current_direct["Correlation"] - previous_direct["Correlation"]
        ),
        "residual_correlation": (
            current_direct["residual_bandpass_correlation_25_80"]
            - previous_direct["residual_bandpass_correlation_25_80"]
        ),
        "phase": (
            current_direct["residual_weighted_phase_score_25_80"]
            - previous_direct["residual_weighted_phase_score_25_80"]
        ),
    }


def main():
    current_path = EVALUATION_DIR / "leave_one_well_aggregate_metrics.npy"
    previous_paths = list(
        (WORKSPACE_ROOT / "21_leave_one_well_calibration" / "data").rglob(
            "leave_one_well_aggregate_metrics.npy"
        )
    )
    if len(previous_paths) != 1:
        raise RuntimeError("Expected one experiment 21 aggregate result.")
    current = np.load(current_path, allow_pickle=True).item()
    previous = np.load(previous_paths[0], allow_pickle=True).item()
    comparison = {
        axis: metric_changes(current, previous, axis)
        for axis in ("inline", "crossline")
    }
    current["experiment"] = 22
    current["comparison_vs_21"] = comparison
    output = EVALUATION_DIR / "context_masked_aggregate_metrics.npy"
    np.save(output, current)
    lines = [
        "22号连续上下文局部掩码监督最终评价",
        "================================",
        "",
        "输入为连续256道F3窄频上下文，宽频监督仅限井周32道。",
        "留出井及扩大保护区未参与训练、验证或checkpoint选择。",
        "",
    ]
    for axis in ("inline", "crossline"):
        lines.append(f"{axis} aggregate")
        for name in ("baseline", "direct", "highpass"):
            metrics = current["aggregate"][axis][name]
            lines.append(
                f"  {name}: corr={metrics['Correlation']:.6f}, "
                f"rmse={metrics['RMSE']:.6f}, "
                f"res_corr="
                f"{metrics['residual_bandpass_correlation_25_80']:.6f}, "
                f"phase="
                f"{metrics['residual_weighted_phase_score_25_80']:.6f}, "
                f"spectrum_l1={metrics['spectrum_l1_vs_reference']:.6f}"
            )
        change = comparison[axis]
        lines.append(
            f"  vs21 direct: corr={change['correlation']:+.6f}, "
            f"res_corr={change['residual_correlation']:+.6f}, "
            f"phase={change['phase']:+.6f}"
        )
        lines.append("")
    lines.extend([
        "Direct sections not below baseline: "
        f"{current['direct_sections_not_below_baseline']}/8",
        f"Predeclared success: {current['success']}",
        "",
        "18号属于全区宽频监督上限，不属于相同数据条件。",
    ])
    figure_dirs = [
        path for path in FIGURE_DIR.iterdir()
        if path.is_dir() and list(path.glob("*_sections.png"))
    ]
    if len(figure_dirs) != 1:
        raise RuntimeError("Expected one prediction evaluation figure folder.")
    report = figure_dirs[0] / "context_masked_final_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
