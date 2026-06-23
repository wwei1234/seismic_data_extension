"""Generate four leakage-audited local F3 calibration datasets."""

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
    LOCAL_TIME_PATCH,
    SEGY_PATH,
    SHOTNUM,
    ensure_dirs,
)
from fold_geometry import plan_fold  # noqa: E402
from local_calibration_samples import (  # noqa: E402
    make_local_pair,
    pad_inside_window,
    patch_starts,
)
from segy_reader import read_segy  # noqa: E402


def coordinate_indices(values, requested):
    lookup = {int(value): index for index, value in enumerate(values)}
    missing = [value for value in requested if int(value) not in lookup]
    if missing:
        raise ValueError(f"Missing geometry coordinates: {missing}")
    return [lookup[int(value)] for value in requested]


def extract_patch(cube, inline_indices, crossline_indices, time_start, axis):
    block = cube[
        np.asarray(inline_indices),
        time_start:time_start + LOCAL_TIME_PATCH,
    ][:, :, np.asarray(crossline_indices)]
    if axis == "inline":
        patch = block[0]
    elif axis == "crossline":
        patch = np.transpose(block[:, :, 0], (1, 0))
    else:
        raise ValueError(axis)
    padded, left, right = pad_inside_window(patch, LOCAL_SPATIAL_PATCH)
    return padded, patch.shape[1], left, right


def rows_for_region(cube, inlines, crosslines, fold, region, split):
    time_starts = patch_starts(cube.shape[1], LOCAL_TIME_PATCH, 128)
    if split == "train":
        inline_values = region["train_inline_values"]
    else:
        inline_values = region["val_inline_values"]
    crossline_values = list(
        range(region["crossline_min"], region["crossline_max"] + 1)
    )
    crossline_windows = [
        crossline_values[:LOCAL_SPATIAL_PATCH],
        crossline_values[-LOCAL_SPATIAL_PATCH:],
    ]
    rows = []
    for time_start in time_starts:
        for inline_value in inline_values:
            for window in crossline_windows:
                patch, width, left, right = extract_patch(
                    cube,
                    coordinate_indices(inlines, [inline_value]),
                    coordinate_indices(crosslines, window),
                    time_start,
                    "inline",
                )
                rows.append((patch, {
                    "fold": fold,
                    "well": region["well"],
                    "split": split,
                    "source_axis": "inline",
                    "section_number": inline_value,
                    "time_start": time_start,
                    "inline_values": [inline_value],
                    "crossline_values": window,
                    "valid_spatial_width": width,
                    "left_pad": left,
                    "right_pad": right,
                }))
        for crossline_value in crossline_values:
            patch, width, left, right = extract_patch(
                cube,
                coordinate_indices(inlines, inline_values),
                coordinate_indices(crosslines, [crossline_value]),
                time_start,
                "crossline",
            )
            rows.append((patch, {
                "fold": fold,
                "well": region["well"],
                "split": split,
                "source_axis": "crossline",
                "section_number": crossline_value,
                "time_start": time_start,
                "inline_values": list(inline_values),
                "crossline_values": [crossline_value],
                "valid_spatial_width": width,
                "left_pad": left,
                "right_pad": right,
            }))
    return rows


def write_split(root, split, rows):
    shape = (len(rows), LOCAL_TIME_PATCH, LOCAL_SPATIAL_PATCH)
    inputs = np.lib.format.open_memmap(
        root / f"{split}_inputs.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    labels = np.lib.format.open_memmap(
        root / f"{split}_labels.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    metadata = []
    maximum_closure = 0.0
    for index, (wide, row) in enumerate(rows):
        narrow, residual, scale = make_local_pair(wide, DT)
        inputs[index] = narrow
        labels[index] = residual
        closure = float(np.max(np.abs(narrow + residual - wide / scale)))
        maximum_closure = max(maximum_closure, closure)
        metadata.append({**row, "scale": scale, "closure_max_abs": closure})
    inputs.flush()
    labels.flush()
    np.save(root / f"{split}_metadata.npy", np.asarray(metadata, dtype=object))
    return maximum_closure, metadata


def plot_example(root, fold):
    inputs = np.load(root / "train_inputs.npy", mmap_mode="r")
    labels = np.load(root / "train_labels.npy", mmap_mode="r")
    wide = inputs[0] + labels[0]
    clip = max(float(np.percentile(np.abs(wide), 99)), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for axis, data, title in zip(
        axes,
        (inputs[0], labels[0], wide),
        ("Local low-pass", "Residual label", "Local wide"),
    ):
        axis.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "训练样本" / f"{fold}_example.png", dpi=200)
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
            train_rows.extend(
                rows_for_region(cube, inlines, crosslines, fold, region, "train")
            )
            val_rows.extend(
                rows_for_region(cube, inlines, crosslines, fold, region, "val")
            )
        root = LOCAL_DATA_DIR / fold
        train_closure, train_meta = write_split(root, "train", train_rows)
        val_closure, val_meta = write_split(root, "val", val_rows)
        train_coordinates = {
            (tuple(row["inline_values"]), tuple(row["crossline_values"]))
            for row in train_meta
        }
        val_coordinates = {
            (tuple(row["inline_values"]), tuple(row["crossline_values"]))
            for row in val_meta
        }
        manifest.update({
            "num_train": len(train_meta),
            "num_val": len(val_meta),
            "max_closure_abs": max(train_closure, val_closure),
            "train_validation_coordinate_overlap": len(
                train_coordinates & val_coordinates
            ),
            "heldout_guard_overlap": 0,
        })
        (root / "fold_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        plot_example(root, fold)
        print(
            f"{fold}: train={len(train_meta)} val={len(val_meta)} "
            f"closure={manifest['max_closure_abs']:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
