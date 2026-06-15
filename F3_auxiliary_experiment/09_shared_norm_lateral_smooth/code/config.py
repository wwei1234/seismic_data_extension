from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # 09_shared_norm_lateral_smooth/
SOURCE_DATA_DIR = ROOT.parent / "01_basic_strategy" / "data"  # read wavelets

DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"

DT = 0.004
PATCH_SIZE = 256
PATCH_STRIDE = 64
SYNTHETIC_SECTION_WIDTH = 1024
SYNTHETIC_SECTIONS_PER_WAVELET = 16

# ── Modified parameters ──
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05]
Q_FILTER_Q = 60          # lower Q → stronger attenuation with depth
Q_FILTER_STRENGTH = 0.35
LOWCUT_FREQ = 8.0        # highpass cutoff to remove low-freq background

RANDOM_SEED = 42
