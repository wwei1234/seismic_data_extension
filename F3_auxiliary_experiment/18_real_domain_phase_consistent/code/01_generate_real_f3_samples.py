"""Generate leakage-safe real F3 low-pass/residual training pairs."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))
sys.path.append(str(CODE_DIR))

from config import (
    CROSSLINE_GUARD,
    DT,
    INLINE_GUARD,
    NARROW_BAND,
    PATCH_SIZE,
    PATCH_STRIDE,
    RANDOM_SEED,
    REAL_DATA_DIR,
    SEGY_PATH,
    SHOTNUM,
    WELL_CROSSLINES,
    WELL_INLINES,
    ensure_dirs,
)
from real_f3_samples import (
    assign_plane_splits,
    make_real_pair,
    pad_spatial_patch,
    plan_axis_candidates,
)
from segy_reader import read_segy


def balanced_cap(rows, limit, seed):
    rng = np.random.default_rng(seed)
    by_axis = {}
    for row in rows:
        by_axis.setdefault(row["source_axis"], []).append(row)
    selected = []
    axes = sorted(by_axis)
    per_axis = limit // max(len(axes), 1)
    for axis in axes:
        axis_rows = list(by_axis[axis])
        rng.shuffle(axis_rows)
        selected.extend(axis_rows[:per_axis])
    if len(selected) < limit:
        selected_keys = {
            (r["source_axis"], r["section_number"], r["time_start"], r["spatial_start"])
            for r in selected
        }
        remaining = [
            row for row in rows
            if (
                row["source_axis"],
                row["section_number"],
                row["time_start"],
                row["spatial_start"],
            ) not in selected_keys
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[:limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


def extract_wide_patch(cube, candidate):
    t0 = candidate["time_start"]
    x0 = candidate["spatial_start"]
    width = candidate["spatial_size"]
    if candidate["source_axis"] == "inline":
        section = cube[candidate["section_index"]]
    else:
        section = cube[:, :, candidate["section_index"]].T
    patch = section[t0:t0 + PATCH_SIZE, x0:x0 + width]
    return pad_spatial_patch(patch, PATCH_SIZE)


def write_split(cube, rows, split):
    shape = (len(rows), PATCH_SIZE, PATCH_SIZE)
    inputs = np.lib.format.open_memmap(
        REAL_DATA_DIR / f"{split}_inputs.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    labels = np.lib.format.open_memmap(
        REAL_DATA_DIR / f"{split}_labels.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    metadata = []
    for index, candidate in enumerate(rows):
        wide = extract_wide_patch(cube, candidate)
        narrow, residual, scale = make_real_pair(wide, DT, NARROW_BAND)
        inputs[index] = narrow
        labels[index] = residual
        metadata.append({
            **candidate,
            "split": split,
            "normalization": "p99_abs_narrow_unclipped",
            "scale": scale,
            "closure_max_abs": float(np.max(np.abs(narrow + residual - wide / scale))),
        })
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"{split}: {index + 1}/{len(rows)}", flush=True)
    inputs.flush()
    labels.flush()
    np.save(REAL_DATA_DIR / f"{split}_metadata.npy", np.asarray(metadata, dtype=object))
    return inputs, labels, metadata


def plot_example(inputs, labels):
    index = min(len(inputs) // 2, len(inputs) - 1)
    narrow = np.asarray(inputs[index])
    residual = np.asarray(labels[index])
    wide = narrow + residual
    clip = max(float(np.percentile(np.abs(wide), 99)), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title in zip(
        axes,
        (narrow, residual, wide),
        ("Real F3 low-pass input", "Real F3 residual label", "Real F3 wide target"),
    ):
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    output = REAL_DATA_DIR.parents[1] / "figures" / "训练样本" / "real_f3_patch_example.png"
    fig.savefig(output, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=2400)
    parser.add_argument("--max-val", type=int, default=480)
    args = parser.parse_args()

    ensure_dirs()
    cube, geometry = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)
    inlines = np.asarray(geometry["inlines"], dtype=np.int32)
    crosslines = np.asarray(geometry["crosslines"], dtype=np.int32)

    inline_candidates = plan_axis_candidates(
        "inline",
        inlines,
        crosslines,
        WELL_INLINES,
        INLINE_GUARD,
        WELL_CROSSLINES,
        CROSSLINE_GUARD,
        cube.shape[1],
        PATCH_SIZE,
        PATCH_SIZE,
        PATCH_STRIDE,
    )
    crossline_candidates = plan_axis_candidates(
        "crossline",
        crosslines,
        inlines,
        WELL_CROSSLINES,
        CROSSLINE_GUARD,
        WELL_INLINES,
        INLINE_GUARD,
        cube.shape[1],
        PATCH_SIZE,
        PATCH_SIZE // 2,
        PATCH_STRIDE,
    )
    train_rows, val_rows = assign_plane_splits(
        inline_candidates + crossline_candidates,
        val_fraction=0.17,
        seed=RANDOM_SEED,
    )
    train_rows = balanced_cap(train_rows, args.max_train, RANDOM_SEED)
    val_rows = balanced_cap(val_rows, args.max_val, RANDOM_SEED + 1)
    print(
        f"Candidates: inline={len(inline_candidates)}, "
        f"crossline={len(crossline_candidates)}, "
        f"selected train={len(train_rows)}, val={len(val_rows)}",
        flush=True,
    )

    train_inputs, train_labels, train_meta = write_split(cube, train_rows, "train")
    _, _, val_meta = write_split(cube, val_rows, "val")
    plot_example(train_inputs, train_labels)

    summary = {
        "dt": DT,
        "narrow_band": NARROW_BAND,
        "patch_size": PATCH_SIZE,
        "normalization": "p99_abs_narrow_unclipped",
        "well_inlines": WELL_INLINES,
        "well_crosslines": WELL_CROSSLINES,
        "inline_guard": INLINE_GUARD,
        "crossline_guard": CROSSLINE_GUARD,
        "num_train": len(train_meta),
        "num_val": len(val_meta),
        "train_axes": {
            axis: sum(row["source_axis"] == axis for row in train_meta)
            for axis in ("inline", "crossline")
        },
        "val_axes": {
            axis: sum(row["source_axis"] == axis for row in val_meta)
            for axis in ("inline", "crossline")
        },
        "max_closure_error": max(
            row["closure_max_abs"] for row in train_meta + val_meta
        ),
    }
    np.save(REAL_DATA_DIR / "metadata.npy", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
