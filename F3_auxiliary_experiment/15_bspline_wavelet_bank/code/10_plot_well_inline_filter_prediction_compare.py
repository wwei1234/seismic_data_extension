"""
Plot prediction/reference filtered section comparisons at well inlines.

Each figure uses the actual well inline and marks the well crossline.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"

INLINE_START = 100
CROSSLINE_START = 300

WELL_POSITIONS = {
    "F02-1": {"inline": 362, "crossline": 336},
    "F03-2": {"inline": 722, "crossline": 848},
    "F03-4": {"inline": 442, "crossline": 1007},
    "F06-1": {"inline": 244, "crossline": 387},
}


def load_cube(name):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def plot_panel(ax, data, title, well_x):
    clip = np.nanpercentile(np.abs(data), 99.0)
    clip = max(float(clip), 1e-8)
    ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
    ax.axvline(well_x, color="yellow", lw=1.4, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Crossline index")
    ax.set_ylabel("Time sample")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="kriging_v1")
    parser.add_argument("--prediction-suffix", default="wide_prediction")
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    prefix = args.prefix
    pred_suffix = args.prediction_suffix
    tag = args.output_tag or pred_suffix
    cubes = {
        "Prediction": load_cube(f"{prefix}_{pred_suffix}.npy"),
        "Reference": load_cube(f"{prefix}_wide_reference.npy"),
        "Pred BP 55-75": load_cube(f"{prefix}_{pred_suffix}_bp55_75.npy"),
        "Ref BP 55-75": load_cube(f"{prefix}_wide_reference_bp55_75.npy"),
        "Pred HP >=55": load_cube(f"{prefix}_{pred_suffix}_hp55.npy"),
        "Ref HP >=55": load_cube(f"{prefix}_wide_reference_hp55.npy"),
        "Ref LP <=55": load_cube(f"{prefix}_wide_reference_lp55.npy"),
    }

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for well_name, pos in WELL_POSITIONS.items():
        il_idx = pos["inline"] - INLINE_START
        xl_idx = pos["crossline"] - CROSSLINE_START

        if il_idx < 0 or il_idx >= cubes["Reference"].shape[0]:
            raise ValueError(f"{well_name} inline index out of range: {il_idx}")
        if xl_idx < 0 or xl_idx >= cubes["Reference"].shape[2]:
            raise ValueError(f"{well_name} crossline index out of range: {xl_idx}")

        panels = [
            ("Prediction", cubes["Prediction"][il_idx]),
            ("Reference", cubes["Reference"][il_idx]),
            ("Pred BP 55-75", cubes["Pred BP 55-75"][il_idx]),
            ("Ref BP 55-75", cubes["Ref BP 55-75"][il_idx]),
            ("Pred HP >=55", cubes["Pred HP >=55"][il_idx]),
            ("Ref HP >=55", cubes["Ref HP >=55"][il_idx]),
            ("Ref LP <=55", cubes["Ref LP <=55"][il_idx]),
        ]

        fig, axes = plt.subplots(2, 4, figsize=(22, 9))
        axes = axes.ravel()
        for ax, (title, data) in zip(axes, panels):
            plot_panel(ax, data, title, xl_idx)
        axes[-1].axis("off")
        fig.suptitle(
            f"{well_name}: inline {pos['inline']} (array index {il_idx}), "
            f"crossline {pos['crossline']} (array index {xl_idx})",
            fontsize=14,
        )
        fig.tight_layout()
        out_path = FIGURE_DIR / f"{prefix}_{tag}_{well_name}_inline_filter_prediction_compare.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {well_name}: {out_path}")


if __name__ == "__main__":
    main()
