from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SOURCE_DATA_DIR = WORKSPACE_ROOT / "shared_data"
SEGY_PATH = WORKSPACE_ROOT / "Rawdata" / "Seismic_data.sgy"

DATA_DIR = ROOT / "data"
F3_PATCH_DIR = DATA_DIR / "F3多频带自监督"
SYNTHETIC_DIR = DATA_DIR / "测井合成样本"
CHECKPOINT_DIR = DATA_DIR / "模型检查点"
PREDICTION_DIR = DATA_DIR / "预测结果"
EVALUATION_DIR = DATA_DIR / "评价结果"
FIGURE_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"

DT = 0.004
SHOTNUM = 651
PATCH_SIZE = 256
PATCH_STRIDE = 128
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)
NOISE_LEVELS = (0.0, 0.01, 0.03)
TOTAL_EPOCHS = 300
RANDOM_SEED = 42
BATCH_SIZE = 4

WELL_INLINES = (244, 362, 442, 722)
WELL_CROSSLINES = (336, 387, 848, 1007)

F3_MASK_TASKS = {
    "A": {
        "kind": "lowpass",
        "stop": (11.0, 13.0),
        "pass": (15.0, 17.0),
        "target_high": 20.0,
    },
    "B": {
        "kind": "lowpass",
        "stop": (17.0, 19.0),
        "pass": (21.0, 23.0),
        "target_high": 28.0,
    },
    "C": {
        "kind": "lowpass",
        "stop": (21.0, 23.0),
        "pass": (25.0, 27.0),
        "target_high": 35.0,
    },
    "D": {
        "kind": "bandstop",
        "width": (4.0, 10.0),
        "center": (12.0, 30.0),
    },
}

FINAL_PROJECTOR = {
    "low_stop": 32.0,
    "low_pass": 38.0,
    "high_pass": 85.0,
    "high_stop": 100.0,
}

CURRICULUM_STAGES = (
    {
        "name": "f3_foundation",
        "start": 1,
        "end": 60,
        "f3_ratio": 1,
        "synthetic_ratio": 0,
        "lr": 5e-4,
    },
    {
        "name": "f3_priority_joint",
        "start": 61,
        "end": 180,
        "f3_ratio": 2,
        "synthetic_ratio": 1,
        "lr": 3e-4,
    },
    {
        "name": "balanced_joint",
        "start": 181,
        "end": 300,
        "f3_ratio": 1,
        "synthetic_ratio": 1,
        "lr": 1e-4,
    },
)

F3_MIN_CORRELATION = 0.85
F3_MIN_PHASE = 0.80
F3_MAX_LEAKAGE = 0.03
SYNTHETIC_TIE_TOLERANCE = 0.01


def ensure_dirs():
    paths = (
        F3_PATCH_DIR,
        SYNTHETIC_DIR,
        CHECKPOINT_DIR,
        PREDICTION_DIR,
        EVALUATION_DIR,
        FIGURE_DIR / "训练样本",
        FIGURE_DIR / "训练曲线",
        FIGURE_DIR / "频谱分析",
        FIGURE_DIR / "预测评价",
        LOG_DIR,
    )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
