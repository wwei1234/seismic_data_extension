from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]   # F3_auxiliary_experiment/
SEGY_PATH = ROOT / "Rawdata" / "Seismic_data.sgy"
WELL_RAW_DIR = ROOT / "Rawdata" / "Well_data" / "All_wells_RawData"

DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
PREDICTION_DIR = ROOT / "predictions"

DT = 0.004
N_TIME = 462
INLINE_START = 100
CROSSLINE_START = 300
SHOTNUM = 651

WELL_POSITIONS = {
    "F02-1": {"inline": 362, "crossline": 336},
    "F03-2": {"inline": 722, "crossline": 848},
    "F03-4": {"inline": 442, "crossline": 1007},
    "F06-1": {"inline": 244, "crossline": 387},
}

# F3 original seismic already has broad content. The auxiliary experiment uses it
# as the wide reference and creates a narrower input by low-pass filtering.
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)
WIDE_BAND = (3.0, 6.0, 55.0, 75.0)

WAVELET_LENGTH_MS = 160.0
PATCH_SIZE = 128
PATCH_STRIDE = 64
RANDOM_SEED = 42
