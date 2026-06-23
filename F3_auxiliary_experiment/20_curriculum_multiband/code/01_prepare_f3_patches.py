"""Cache clean observable F3 narrow-band patches for dynamic masking."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))
sys.path.append(str(CODE_DIR))

from config import (  # noqa: E402
    DT,
    F3_PATCH_DIR,
    NARROW_BAND,
    PATCH_SIZE,
    PATCH_STRIDE,
    RANDOM_SEED,
    SEGY_PATH,
    SHOTNUM,
    ensure_dirs,
)
from datasets import grouped_plane_split  # noqa: E402
from leakage_guard import assert_training_paths_are_safe  # noqa: E402
from segy_reader import read_segy  # noqa: E402
from signal_utils import zero_phase_filter_section  # noqa: E402


def patch_starts(length, patch_size, stride):
    if length < patch_size:
        return []
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def patch_has_coverage(patch, minimum=0.95):
    covered = np.max(np.abs(patch), axis=0) > 0
    return float(np.mean(covered)) >= minimum


def collect_candidates(cube, geometry):
    rows = []
    time_starts = patch_starts(cube.shape[1], PATCH_SIZE, PATCH_STRIDE)
    inline_spatial = patch_starts(cube.shape[2], PATCH_SIZE, PATCH_STRIDE)
    for index, number in enumerate(np.asarray(geometry["inlines"])):
        section = cube[index]
        for time_start in time_starts:
            for spatial_start in inline_spatial:
                patch = section[
                    time_start:time_start + PATCH_SIZE,
                    spatial_start:spatial_start + PATCH_SIZE,
                ]
                if patch_has_coverage(patch):
                    rows.append({
                        "source_axis": "inline",
                        "section_index": int(index),
                        "section_number": int(number),
                        "time_start": int(time_start),
                        "spatial_start": int(spatial_start),
                    })

    crossline_spatial = patch_starts(cube.shape[0], PATCH_SIZE, PATCH_STRIDE)
    for index, number in enumerate(np.asarray(geometry["crosslines"])):
        section = cube[:, :, index].T
        for time_start in time_starts:
            for spatial_start in crossline_spatial:
                patch = section[
                    time_start:time_start + PATCH_SIZE,
                    spatial_start:spatial_start + PATCH_SIZE,
                ]
                if patch_has_coverage(patch):
                    rows.append({
                        "source_axis": "crossline",
                        "section_index": int(index),
                        "section_number": int(number),
                        "time_start": int(time_start),
                        "spatial_start": int(spatial_start),
                    })
    return rows


def extract_patch(cube, row):
    if row["source_axis"] == "inline":
        section = cube[row["section_index"]]
    else:
        section = cube[:, :, row["section_index"]].T
    t0 = row["time_start"]
    x0 = row["spatial_start"]
    return section[t0:t0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]


def write_split(cube, rows, split):
    patch_path = F3_PATCH_DIR / f"{split}_clean_narrow.npy"
    metadata_path = F3_PATCH_DIR / f"{split}_metadata.npy"
    assert_training_paths_are_safe([patch_path, metadata_path])
    patches = np.lib.format.open_memmap(
        patch_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), PATCH_SIZE, PATCH_SIZE),
    )
    metadata = []
    for index, row in enumerate(rows):
        raw = extract_patch(cube, row).astype(np.float32)
        patches[index] = zero_phase_filter_section(raw, DT, NARROW_BAND)
        metadata.append({
            **row,
            "clean_id": index,
            "normalization": "dynamic_per_patch_p99_abs_clean_narrow",
            "uses_f3_wide_target": False,
        })
    patches.flush()
    np.save(metadata_path, np.asarray(metadata, dtype=object))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-val", type=int, default=600)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    cube, geometry = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)
    candidates = collect_candidates(cube, geometry)
    train, val = grouped_plane_split(candidates, val_fraction=0.2, seed=RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(train)
    rng.shuffle(val)
    train = train[:args.max_train]
    val = val[:args.max_val]
    write_split(cube, train, "train")
    write_split(cube, val, "val")

    train_planes = sorted({
        (row["source_axis"], row["section_number"]) for row in train
    })
    val_planes = sorted({
        (row["source_axis"], row["section_number"]) for row in val
    })
    manifest = {
        "experiment": 20,
        "num_train": len(train),
        "num_val": len(val),
        "patch_shape": [PATCH_SIZE, PATCH_SIZE],
        "patch_stride": PATCH_STRIDE,
        "narrow_band": list(NARROW_BAND),
        "normalization": "dynamic_per_patch_p99_abs_clean_narrow",
        "uses_f3_narrow": True,
        "uses_f3_wide_target": False,
        "train_planes": train_planes,
        "val_planes": val_planes,
        "planes_disjoint": set(map(tuple, train_planes)).isdisjoint(
            set(map(tuple, val_planes))
        ),
        "smoke": bool(args.smoke),
    }
    (F3_PATCH_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
