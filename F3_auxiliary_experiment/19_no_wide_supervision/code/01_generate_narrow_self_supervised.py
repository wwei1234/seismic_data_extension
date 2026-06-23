"""Create self-supervised pairs using only the observable F3 narrow band."""

import argparse
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))
sys.path.append(str(CODE_DIR))

from config import (
    DT,
    EXTRA_LOW_BAND,
    NARROW_BAND,
    PATCH_SIZE,
    PATCH_STRIDE,
    RANDOM_SEED,
    SEGY_PATH,
    SELF_SUPERVISED_DIR,
    SHOTNUM,
    WELL_CROSSLINES,
    WELL_INLINES,
    ensure_dirs,
)
from leakage_guard import assert_training_paths_are_safe
from sample_utils import patch_starts
from segy_reader import read_segy
from self_supervised_samples import make_narrow_self_supervised_pair
from signal_utils import zero_phase_filter_section


def collect_candidates(cube, geometry):
    candidates = []
    inlines = np.asarray(geometry["inlines"])
    crosslines = np.asarray(geometry["crosslines"])
    t_starts = patch_starts(cube.shape[1], PATCH_SIZE, PATCH_STRIDE)
    x_starts_inline = patch_starts(cube.shape[2], PATCH_SIZE, PATCH_STRIDE)
    for index, number in enumerate(inlines):
        if any(abs(int(number) - value) <= 8 for value in WELL_INLINES):
            continue
        section = cube[index]
        for t0 in t_starts:
            for x0 in x_starts_inline:
                patch = section[t0:t0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
                if np.mean(np.max(np.abs(patch), axis=0) > 0) >= 0.95:
                    candidates.append(("inline", index, int(number), t0, x0))

    x_starts_crossline = patch_starts(cube.shape[0], PATCH_SIZE, PATCH_STRIDE)
    for index, number in enumerate(crosslines):
        if any(abs(int(number) - value) <= 16 for value in WELL_CROSSLINES):
            continue
        section = cube[:, :, index].T
        for t0 in t_starts:
            for x0 in x_starts_crossline:
                patch = section[t0:t0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
                if np.mean(np.max(np.abs(patch), axis=0) > 0) >= 0.95:
                    candidates.append(("crossline", index, int(number), t0, x0))
    return candidates


def extract_patch(cube, candidate):
    axis, index, _, t0, x0 = candidate
    section = cube[index] if axis == "inline" else cube[:, :, index].T
    return section[t0:t0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]


def write_split(cube, rows, split):
    shape = (len(rows), PATCH_SIZE, PATCH_SIZE)
    inputs = np.lib.format.open_memmap(
        SELF_SUPERVISED_DIR / f"{split}_inputs.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    labels = np.lib.format.open_memmap(
        SELF_SUPERVISED_DIR / f"{split}_labels.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    metadata = []
    for idx, row in enumerate(rows):
        raw_patch = extract_patch(cube, row).astype(np.float32)
        known_narrow = zero_phase_filter_section(raw_patch, DT, NARROW_BAND)
        x, label, scale = make_narrow_self_supervised_pair(
            known_narrow,
            DT,
            EXTRA_LOW_BAND,
        )
        inputs[idx] = x
        labels[idx] = label
        metadata.append({
            "source_axis": row[0],
            "section_number": row[2],
            "time_start": row[3],
            "spatial_start": row[4],
            "scale": scale,
            "input_band": EXTRA_LOW_BAND,
            "known_target_band": NARROW_BAND,
            "contains_f3_wide_target": False,
        })
    inputs.flush()
    labels.flush()
    np.save(
        SELF_SUPERVISED_DIR / f"{split}_metadata.npy",
        np.asarray(metadata, dtype=object),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=1200)
    parser.add_argument("--max-val", type=int, default=240)
    args = parser.parse_args()
    ensure_dirs()
    assert_training_paths_are_safe([
        SELF_SUPERVISED_DIR / "train_inputs.npy",
        SELF_SUPERVISED_DIR / "train_labels.npy",
    ])
    cube, geometry = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)
    candidates = collect_candidates(cube, geometry)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(candidates)
    selected = candidates[:args.max_train + args.max_val]
    train = selected[:args.max_train]
    val = selected[args.max_train:]
    write_split(cube, train, "train")
    write_split(cube, val, "val")
    summary = {
        "num_train": len(train),
        "num_val": len(val),
        "uses_f3_narrow": True,
        "uses_f3_wide_target": False,
        "narrow_band": NARROW_BAND,
        "extra_low_band": EXTRA_LOW_BAND,
    }
    np.save(SELF_SUPERVISED_DIR / "metadata.npy", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
