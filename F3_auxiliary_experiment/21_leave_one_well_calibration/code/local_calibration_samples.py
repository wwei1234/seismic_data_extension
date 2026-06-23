import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from config import (  # noqa: E402
    LOCAL_SPATIAL_PATCH,
    LOCAL_SPATIAL_STRIDE,
    LOCAL_TIME_PATCH,
    LOCAL_TIME_STRIDE,
    NARROW_BAND,
)
from signal_utils import zero_phase_filter_section  # noqa: E402


def patch_starts(length, size, stride):
    if length < size:
        return []
    starts = list(range(0, length - size + 1, stride))
    last = length - size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def make_local_pair(wide_patch, dt):
    wide_patch = np.asarray(wide_patch, dtype=np.float32)
    narrow_raw = zero_phase_filter_section(wide_patch, dt, NARROW_BAND)
    scale = max(float(np.percentile(np.abs(narrow_raw), 99)), 1e-8)
    narrow = (narrow_raw / scale).astype(np.float32)
    residual = (wide_patch / scale - narrow).astype(np.float32)
    return narrow, residual, scale


def pad_inside_window(patch, target_width=LOCAL_SPATIAL_PATCH):
    patch = np.asarray(patch, dtype=np.float32)
    if patch.shape[1] > target_width:
        raise ValueError("Patch width exceeds the target width.")
    total = target_width - patch.shape[1]
    left = total // 2
    right = total - left
    if total == 0:
        return patch, 0, 0
    padded = np.pad(patch, ((0, 0), (left, right)), mode="reflect")
    return padded.astype(np.float32), left, right


def plan_local_patches(window, time_size, split):
    if split == "train":
        inline_groups = [list(window.get(
            "train_inline_values",
            range(window["well_inline"] - 8, window["well_inline"] + 7),
        ))]
    elif split == "val":
        inline_groups = [[value] for value in window.get(
            "val_inline_values",
            [window["well_inline"] + 7, window["well_inline"] + 8],
        )]
    else:
        raise ValueError(f"Unsupported split: {split}")
    crossline_values = list(
        range(window["crossline_min"], window["crossline_max"] + 1)
    )
    rows = []
    for time_start in patch_starts(
        time_size,
        LOCAL_TIME_PATCH,
        LOCAL_TIME_STRIDE,
    ):
        for inline_values in inline_groups:
            rows.append({
                "source_axis": "crossline",
                "time_start": time_start,
                "inline_values": list(inline_values),
                "crossline_values": crossline_values,
            })
        if split == "train":
            for inline_value in inline_groups[0]:
                for spatial_start in patch_starts(
                    len(crossline_values),
                    LOCAL_SPATIAL_PATCH,
                    LOCAL_SPATIAL_STRIDE,
                ):
                    rows.append({
                        "source_axis": "inline",
                        "time_start": time_start,
                        "inline_values": [inline_value],
                        "crossline_values": crossline_values[
                            spatial_start:spatial_start + LOCAL_SPATIAL_PATCH
                        ],
                    })
        else:
            for inline_values in inline_groups:
                rows.append({
                    "source_axis": "inline",
                    "time_start": time_start,
                    "inline_values": inline_values,
                    "crossline_values": crossline_values[:LOCAL_SPATIAL_PATCH],
                })
    return rows
