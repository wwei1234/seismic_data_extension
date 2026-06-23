from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
RAW_DATA_DIR = WORKSPACE_ROOT / "Rawdata"
SHARED_DATA_DIR = WORKSPACE_ROOT / "shared_data"
SHARED_CODE_DIR = WORKSPACE_ROOT / "shared_code"
SEGY_PATH = RAW_DATA_DIR / "Seismic_data.sgy"

DATA_DIR = ROOT / "data"
COMMON_DATA_DIR = DATA_DIR / "公共预训练"
LOCAL_DATA_DIR = DATA_DIR / "局部宽频标定"
CHECKPOINT_DIR = DATA_DIR / "模型检查点"
PREDICTION_DIR = DATA_DIR / "预测结果"
EVALUATION_DIR = DATA_DIR / "评价结果"
FIGURE_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"

WELLS = {
    "well1": {"inline": 244, "crossline": 336},
    "well2": {"inline": 362, "crossline": 387},
    "well3": {"inline": 442, "crossline": 848},
    "well4": {"inline": 722, "crossline": 1007},
}
FOLDS = {
    f"fold_{heldout}": {
        "heldout_well": heldout,
        "calibration_wells": tuple(name for name in WELLS if name != heldout),
    }
    for heldout in WELLS
}

DT = 0.004
SHOTNUM = 651
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)
CALIBRATION_INLINE_RADIUS = 8
CALIBRATION_CROSSLINE_RADIUS = 16
HELDOUT_INLINE_GUARD = 16
HELDOUT_CROSSLINE_GUARD = 32
LOCAL_TIME_PATCH = 256
LOCAL_SPATIAL_PATCH = 256
LOCAL_SUPERVISION_WIDTH = 32
LOCAL_TIME_STRIDE = 128
LOCAL_SPATIAL_STRIDE = 8
PATCH_SIZE = 256
PATCH_STRIDE = 128
RANDOM_SEED = 42
BATCH_SIZE = 4
LOCAL_RATIO = 0.6
SYNTHETIC_RATIO = 0.2
F3_RATIO = 0.2
STAGE_A_EPOCHS = 60
STAGE_B_EPOCHS = 40

PROJECTOR = {
    "low_stop": 22.0,
    "low_pass": 28.0,
    "high_pass": 85.0,
    "high_stop": 100.0,
}
NOISE_LEVELS = (0.0, 0.01, 0.03)
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


def ensure_dirs():
    paths = [
        COMMON_DATA_DIR,
        LOCAL_DATA_DIR,
        CHECKPOINT_DIR / "pretrain",
        PREDICTION_DIR,
        EVALUATION_DIR,
        FIGURE_DIR / "标定窗口",
        FIGURE_DIR / "训练样本",
        FIGURE_DIR / "训练曲线",
        FIGURE_DIR / "预测评价",
        LOG_DIR,
    ]
    for fold in FOLDS:
        paths.extend([
            LOCAL_DATA_DIR / fold,
            CHECKPOINT_DIR / fold,
        ])
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
