import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))

from signal_utils import zero_phase_filter_section


def make_narrow_self_supervised_pair(
    known_narrow,
    dt,
    extra_low_band=(3.0, 6.0, 18.0, 22.0),
):
    known_narrow = np.asarray(known_narrow, dtype=np.float32)
    extra_low = zero_phase_filter_section(known_narrow, dt, extra_low_band)
    scale = max(float(np.percentile(np.abs(extra_low), 99)), 1e-8)
    input_norm = (extra_low / scale).astype(np.float32)
    label_norm = ((known_narrow - extra_low) / scale).astype(np.float32)
    return input_norm, label_norm, scale
