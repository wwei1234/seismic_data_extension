import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from signal_utils import zero_phase_filter_section


def patch_starts(length, patch_size, stride):
    if length < patch_size:
        return []
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def patch_is_guarded(start_value, stop_value, guarded_values, margin):
    for value in guarded_values:
        guard_start = value - margin
        guard_stop = value + margin + 1
        if start_value < guard_stop and stop_value > guard_start:
            return True
    return False


def make_real_pair(wide_patch, dt, narrow_band=(3.0, 6.0, 25.0, 35.0)):
    wide_patch = np.asarray(wide_patch, dtype=np.float32)
    narrow_raw = zero_phase_filter_section(wide_patch, dt, narrow_band)
    scale = max(float(np.percentile(np.abs(narrow_raw), 99)), 1e-8)
    narrow_norm = (narrow_raw / scale).astype(np.float32)
    residual_norm = (wide_patch / scale - narrow_norm).astype(np.float32)
    return narrow_norm, residual_norm, scale


def plan_axis_candidates(
    source_axis,
    axis_values,
    orthogonal_values,
    source_guards,
    source_margin,
    orthogonal_guards,
    orthogonal_margin,
    time_size,
    patch_size,
    spatial_size,
    stride,
):
    axis_values = np.asarray(axis_values)
    orthogonal_values = np.asarray(orthogonal_values)
    candidates = []
    time_starts = patch_starts(time_size, patch_size, stride)
    spatial_starts = patch_starts(
        orthogonal_values.size,
        spatial_size,
        min(stride, spatial_size),
    )
    for section_index, section_number in enumerate(axis_values):
        if any(abs(int(section_number) - int(value)) <= source_margin for value in source_guards):
            continue
        for spatial_start in spatial_starts:
            spatial_stop = spatial_start + spatial_size
            start_value = int(orthogonal_values[spatial_start])
            stop_value = int(orthogonal_values[spatial_stop - 1]) + 1
            if patch_is_guarded(
                start_value,
                stop_value,
                orthogonal_guards,
                orthogonal_margin,
            ):
                continue
            for time_start in time_starts:
                candidates.append({
                    "source_axis": source_axis,
                    "section_index": int(section_index),
                    "section_number": int(section_number),
                    "time_start": int(time_start),
                    "spatial_start": int(spatial_start),
                    "spatial_size": int(spatial_size),
                    "spatial_start_value": start_value,
                    "spatial_stop_value": stop_value,
                })
    return candidates


def assign_plane_splits(candidates, val_fraction=0.2, seed=42):
    plane_keys = sorted({
        (row["source_axis"], int(row["section_number"]))
        for row in candidates
    })
    rng = np.random.default_rng(seed)
    shuffled = list(plane_keys)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_planes = set(shuffled[:val_count])
    train = []
    val = []
    for row in candidates:
        key = (row["source_axis"], int(row["section_number"]))
        (val if key in val_planes else train).append(row)
    return train, val


def pad_spatial_patch(patch, patch_size):
    patch = np.asarray(patch, dtype=np.float32)
    if patch.shape[1] == patch_size:
        return patch
    if patch.shape[1] > patch_size:
        raise ValueError(f"Patch width {patch.shape[1]} exceeds target {patch_size}.")
    total = patch_size - patch.shape[1]
    left = total // 2
    right = total - left
    return np.pad(patch, ((0, 0), (left, right)), mode="reflect").astype(np.float32)
