from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # 15_bspline_wavelet_bank/
WORKSPACE_ROOT = ROOT.parent
PROJECT_ROOT = ROOT
SOURCE_DATA_DIR = ROOT / "data"

DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"

DT = 0.004
PATCH_SIZE = 256
PATCH_STRIDE = 64           # more patches for better training
SYNTHETIC_SECTION_WIDTH = 951   # match F3 crossline headers: 300-1250 inclusive

# ── Kriging ──
KRIGING_NUGGET = 0.01
KRIGING_SILL = 1.0
# range_ estimated from F3 data; fallback 300

# ── Envelope ──
ENVELOPE_SMOOTH_T = 30
ENVELOPE_SMOOTH_X = 100

# ── Narrow band ──
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)

# ── Wide wavelet bank ──
BSPLINE_WAVELET_BANDS = [
    (3.0, 6.0, 55.0, 70.0),
    (3.0, 6.0, 65.0, 80.0),
    (3.0, 6.0, 75.0, 90.0),
]

# ── Noise ──
NOISE_LEVELS = [0.01, 0.03]

# ── Q filter ──
Q_FILTER_Q = 60
Q_FILTER_STRENGTH = 0.35

# ── Low-cut off ──
LOWCUT_FREQ = 0

# ── Combinations ──
# 4 wells → 2-well:6, 3-well:4, 4-well:1 = 11 combos
# Each combo × 3 B-spline wavelets = 33 profiles.
# Noise levels are assigned cyclically from NOISE_LEVELS, not duplicated per section.
PROFILES_PER_COMBO = 3

# ── SEG-Y ──
SEGY_PATH = WORKSPACE_ROOT / "Rawdata" / "Seismic_data.sgy"
SHOTNUM = 651

RANDOM_SEED = 42
