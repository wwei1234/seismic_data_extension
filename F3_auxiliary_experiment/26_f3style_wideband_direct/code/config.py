from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
PROJECT_ROOT = ROOT
SOURCE_DATA_DIR = WORKSPACE_ROOT / "shared_data"

DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"

DT = 0.004
PATCH_SIZE = 256
PATCH_STRIDE = 128
SYNTHETIC_SECTION_WIDTH = 951
CROSSLINE_MIN = 300
CROSSLINE_MAX = 1250

# 26 uses the same F3-style synthetic sections as 25, but trains direct wideband labels.
NARROW_BAND = (0.0, 0.0, 30.0, 35.0)
HIGH_BAND = (35.0, 90.0)

NOISE_LEVELS = [0.01, 0.03]
USE_Q_FILTER = False
LOWCUT_FREQ = 0.0

SEGY_PATH = WORKSPACE_ROOT / "Rawdata" / "Seismic_data.sgy"
SHOTNUM = 651
RANDOM_SEED = 42
