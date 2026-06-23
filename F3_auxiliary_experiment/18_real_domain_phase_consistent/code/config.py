from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SHARED_DATA_DIR = WORKSPACE_ROOT / "shared_data"
SYNTHETIC_DATA_DIR = WORKSPACE_ROOT / "16_geometry_realistic_samples" / "data"

DATA_DIR = ROOT / "data"
REAL_DATA_DIR = DATA_DIR / "真实样本"
CHECKPOINT_DIR = DATA_DIR / "模型检查点"
PREDICTION_DIR = DATA_DIR / "预测结果"
EVALUATION_DIR = DATA_DIR / "评价结果"
FIGURE_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"

SEGY_PATH = WORKSPACE_ROOT / "Rawdata" / "Seismic_data.sgy"
SHOTNUM = 651
DT = 0.004
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)

PATCH_SIZE = 256
PATCH_STRIDE = 128
RANDOM_SEED = 42

WELL_INLINES = (244, 362, 442, 722)
WELL_CROSSLINES = (336, 387, 848, 1007)
INLINE_GUARD = 8
CROSSLINE_GUARD = 16

PROJECT_LOW_STOP = 22.0
PROJECT_LOW_PASS = 28.0
PROJECT_HIGH_PASS = 85.0
PROJECT_HIGH_STOP = 100.0


def ensure_dirs():
    for path in (
        REAL_DATA_DIR,
        CHECKPOINT_DIR,
        PREDICTION_DIR,
        EVALUATION_DIR,
        FIGURE_DIR / "训练样本",
        FIGURE_DIR / "训练过程",
        FIGURE_DIR / "预测评价",
        LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
