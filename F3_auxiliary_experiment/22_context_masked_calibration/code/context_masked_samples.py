import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from config import NARROW_BAND  # noqa: E402
from signal_utils import zero_phase_filter_section  # noqa: E402


def plan_centered_context(
    coordinates,
    supervision_start,
    supervision_width,
    context_width,
):
    values = np.asarray(coordinates, dtype=np.int32)
    lookup = {int(value): index for index, value in enumerate(values)}
    supervision = np.arange(
        supervision_start,
        supervision_start + supervision_width,
        dtype=np.int32,
    )
    missing = [int(value) for value in supervision if int(value) not in lookup]
    if missing:
        raise ValueError(f"Missing supervision coordinates: {missing}")
    supervision_first = lookup[int(supervision[0])]
    desired_start = supervision_first - (context_width - supervision_width) // 2
    context_start = min(max(desired_start, 0), len(values) - context_width)
    context_stop = context_start + context_width
    context_values = values[context_start:context_stop]
    if len(context_values) != context_width or np.any(np.diff(context_values) != 1):
        raise ValueError("Context coordinates must be complete and consecutive.")
    mask_start = supervision_first - context_start
    mask_stop = mask_start + supervision_width
    if mask_start < 0 or mask_stop > context_width:
        raise ValueError("Supervision window is outside the context.")
    return {
        "context_values": context_values.tolist(),
        "context_start_index": int(context_start),
        "context_stop_index": int(context_stop),
        "mask_start": int(mask_start),
        "mask_stop": int(mask_stop),
    }


def build_supervision_mask(
    time_size,
    context_width,
    mask_start,
    supervision_width,
):
    mask = np.zeros((time_size, context_width), dtype=np.float32)
    mask[:, mask_start:mask_start + supervision_width] = 1.0
    return mask


def make_context_pair(wide_context, mask, dt):
    wide = np.asarray(wide_context, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if wide.shape != mask.shape:
        raise ValueError("Wide context and supervision mask must match.")
    narrow_raw = zero_phase_filter_section(wide, dt, NARROW_BAND)
    scale = max(float(np.percentile(np.abs(narrow_raw), 99)), 1e-8)
    narrow = (narrow_raw / scale).astype(np.float32)
    wide_norm = (wide / scale).astype(np.float32)
    residual = ((wide_norm - narrow) * mask).astype(np.float32)
    target = (wide_norm * mask).astype(np.float32)
    return narrow, residual, target, scale
