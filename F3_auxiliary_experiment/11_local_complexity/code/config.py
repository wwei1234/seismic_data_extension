from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # 10_raw_wavelet_narrow_norm/
SOURCE_DATA_DIR = ROOT.parent / "01_basic_strategy" / "data"

DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"

DT = 0.004
PATCH_SIZE = 256
PATCH_STRIDE = 128          # ⇐ changed from 64
SYNTHETIC_SECTION_WIDTH = 1024
SYNTHETIC_SECTIONS_PER_WAVELET = 16

# ── Narrow band for low-pass filtering wide → narrow ──
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)

# ── Noise ──
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05]

# ── Q filter ──
Q_FILTER_Q = 60
Q_FILTER_STRENGTH = 0.35

# ── Low-frequency removal (off) ──
LOWCUT_FREQ = 0             # disabled

RANDOM_SEED = 42
