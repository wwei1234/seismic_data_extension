from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # 13_kriging/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA_DIR = PROJECT_ROOT / "01_basic_strategy" / "data"

DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"

DT = 0.004
PATCH_SIZE = 256
PATCH_STRIDE = 64           # more patches for better training
SYNTHETIC_SECTION_WIDTH = 922   # match F3 crossline count (SHOTNUM=651 gives 922)

# ── Kriging ──
KRIGING_NUGGET = 0.01
KRIGING_SILL = 1.0
# range_ estimated from F3 data; fallback 300

# ── Envelope ──
ENVELOPE_SMOOTH_T = 30
ENVELOPE_SMOOTH_X = 100

# ── Narrow band ──
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)

# ── Noise ──
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05]

# ── Q filter ──
Q_FILTER_Q = 60
Q_FILTER_STRENGTH = 0.35

# ── Low-cut off ──
LOWCUT_FREQ = 0

# ── Combinations ──
# 4 wells → 2-well:6, 3-well:4, 4-well:1 = 11 combos
# Each combo × 2 profiles (random seeds) × 4 noise levels = 88 profiles
PROFILES_PER_COMBO = 2

# ── SEG-Y ──
SEGY_PATH = PROJECT_ROOT / "Rawdata" / "Seismic_data.sgy"
SHOTNUM = 651

RANDOM_SEED = 42
