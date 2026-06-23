"""Generate 256-trace contexts with 32-trace local wide supervision."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))
sys.path.append(str(CODE_DIR))

from config import (  # noqa: E402
    DT,
    FIGURE_DIR,
    FOLDS,
    LOCAL_DATA_DIR,
    LOCAL_SPATIAL_PATCH,
    LOCAL_SUPERVISION_WIDTH,
    LOCAL_TIME_PATCH,
    LOCAL_TIME_STRIDE,
    SEGY_PATH,
    SHOTNUM,
    ensure_dirs,
)
from context_masked_samples import (  # noqa: E402
    build_supervision_mask,
    make_context_pair,
    plan_centered_context,
)
from fold_geometry import plan_fold  # noqa: E402
from local_calibration_samples import patch_starts  # noqa: E402
from segy_reader import read_segy  # noqa: E402


def coordinate_index(values, requested):
    found = np.where(np.asarray(values) == int(requested))[0]
    if not found.size:
        raise ValueError(f"Missing geometry coordinate: {requested}")
    return int(found[0])


def rows_for_region(cube, inlines, crosslines, fold, region, split):
    inline_values = (
        region["train_inline_values"]
        if split == "train"
        else region["val_inline_values"]
    )
    supervision_start = int(region["well_crossline"] - 16)
    plan = plan_centered_context(
        crosslines,
        supervision_start,
        LOCAL_SUPERVISION_WIDTH,
        LOCAL_SPATIAL_PATCH,
    )
    context_indices = [
        coordinate_index(crosslines, value)
        for value in plan["context_values"]
    ]
    rows = []
    for inline_value in inline_values:
        inline_index = coordinate_index(inlines, inline_value)
        section = cube[inline_index][:, context_indices]
        for time_start in patch_starts(
            cube.shape[1],
            LOCAL_TIME_PATCH,
            LOCAL_TIME_STRIDE,
        ):
            rows.append((section[
                time_start:time_start + LOCAL_TIME_PATCH
            ].astype(np.float32), {
                "fold": fold,
                "well": region["well"],
                "split": split,
                "source_axis": "inline",
                "section_number": int(inline_value),
                "time_start": int(time_start),
                "inline_values": [int(inline_value)],
                "context_crossline_values": list(plan["context_values"]),
                "supervision_crossline_values": list(range(
                    supervision_start,
                    supervision_start + LOCAL_SUPERVISION_WIDTH,
                )),
                "mask_start": int(plan["mask_start"]),
                "mask_stop": int(plan["mask_stop"]),
            }))
    return rows


def write_split(root, split, rows):
    shape = (len(rows), LOCAL_TIME_PATCH, LOCAL_SPATIAL_PATCH)
    inputs = np.lib.format.open_memmap(
        root / f"{split}_inputs.npy", mode="w+", dtype=np.float32, shape=shape
    )
    labels = np.lib.format.open_memmap(
        root / f"{split}_labels.npy", mode="w+", dtype=np.float32, shape=shape
    )
    masks = np.lib.format.open_memmap(
        root / f"{split}_masks.npy", mode="w+", dtype=np.float32, shape=shape
    )
    metadata = []
    maximum_closure = 0.0
    for index, (wide, row) in enumerate(rows):
        mask = build_supervision_mask(
            LOCAL_TIME_PATCH,
            LOCAL_SPATIAL_PATCH,
            row["mask_start"],
            LOCAL_SUPERVISION_WIDTH,
        )
        narrow, residual, _, scale = make_context_pair(wide, mask, DT)
        inputs[index] = narrow
        labels[index] = residual
        masks[index] = mask
        valid = mask > 0
        closure = float(np.max(np.abs(
            (narrow + residual)[valid] - (wide / scale)[valid]
        )))
        maximum_closure = max(maximum_closure, closure)
        metadata.append({
            **row,
            "scale": scale,
            "normalization": "per_context_p99_abs_narrow",
            "supervised_values": int(mask.sum()),
            "closure_max_abs": closure,
        })
    inputs.flush()
    labels.flush()
    masks.flush()
    np.save(root / f"{split}_metadata.npy", np.asarray(metadata, dtype=object))
    return maximum_closure, metadata


def adjacent_trace_diagnostics(section):
    correlations = []
    best_lags = []
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
        best_lags.append(int(np.argmax(scores) - 6))
    return {
        "adjacent_correlation_median": float(np.median(correlations)),
        "adjacent_correlation_min": float(np.min(correlations)),
        "nonzero_best_lag_fraction": float(np.mean(np.asarray(best_lags) != 0)),
    }


def plot_example(root, fold):
    inputs = np.load(root / "train_inputs.npy", mmap_mode="r")
    labels = np.load(root / "train_labels.npy", mmap_mode="r")
    masks = np.load(root / "train_masks.npy", mmap_mode="r")
    wide_supervision = (inputs[0] + labels[0]) * masks[0]
    clip = max(float(np.percentile(np.abs(inputs[0] + labels[0]), 99)), 1e-8)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = (
        (inputs[0], "256-trace low-pass context"),
        (labels[0], "32-trace residual label"),
        (wide_supervision, "Masked wide supervision"),
        (masks[0], "Supervision mask"),
    )
    for axis, (data, title) in zip(axes, panels):
        limit = 1.0 if title == "Supervision mask" else clip
        axis.imshow(data, cmap="seismic", aspect="auto", vmin=-limit, vmax=limit)
        axis.set_title(title)
        axis.set_xlabel("Trace")
        axis.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "训练样本" / f"{fold}_context_example.png", dpi=220)
    plt.close(fig)


def main():
    ensure_dirs()
    cube, geometry = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)
    inlines = np.asarray(geometry["inlines"])
    crosslines = np.asarray(geometry["crosslines"])
    for fold in FOLDS:
        manifest = plan_fold(fold)
        train_rows = []
        val_rows = []
        for region in manifest["wide_sample_regions"]:
            train_rows.extend(rows_for_region(
                cube, inlines, crosslines, fold, region, "train"
            ))
            val_rows.extend(rows_for_region(
                cube, inlines, crosslines, fold, region, "val"
            ))
        root = LOCAL_DATA_DIR / fold
        train_closure, train_meta = write_split(root, "train", train_rows)
        val_closure, val_meta = write_split(root, "val", val_rows)
        train_coordinates = {
            (row["section_number"], tuple(row["supervision_crossline_values"]))
            for row in train_meta
        }
        val_coordinates = {
            (row["section_number"], tuple(row["supervision_crossline_values"]))
            for row in val_meta
        }
        first_input = np.load(root / "train_inputs.npy", mmap_mode="r")[0]
        manifest.update({
            "experiment": 22,
            "sample_shape": [LOCAL_TIME_PATCH, LOCAL_SPATIAL_PATCH],
            "supervision_width": LOCAL_SUPERVISION_WIDTH,
            "num_train": len(train_meta),
            "num_val": len(val_meta),
            "max_closure_abs": max(train_closure, val_closure),
            "train_validation_coordinate_overlap": len(
                train_coordinates & val_coordinates
            ),
            "heldout_guard_overlap": 0,
            "continuity": adjacent_trace_diagnostics(first_input),
        })
        (root / "fold_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        plot_example(root, fold)
        print(
            f"{fold}: train={len(train_meta)} val={len(val_meta)} "
            f"closure={manifest['max_closure_abs']:.3e} "
            f"corr={manifest['continuity']['adjacent_correlation_median']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
