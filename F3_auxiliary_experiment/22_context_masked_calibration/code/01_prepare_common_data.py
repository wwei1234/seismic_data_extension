"""Audit reusable no-wide-target data for experiment 21."""

import json
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from config import COMMON_DATA_DIR, WORKSPACE_ROOT, ensure_dirs  # noqa: E402
from leakage_guard import sha256_payload  # noqa: E402


def array_summary(path):
    array = np.load(path, mmap_mode="r")
    return {
        "path": str(path.resolve()),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": int(path.stat().st_size),
    }


def build_manifest():
    f3_root = WORKSPACE_ROOT / "20_curriculum_multiband" / "data" / "F3多频带自监督"
    synthetic_root = (
        WORKSPACE_ROOT
        / "20_curriculum_multiband"
        / "data"
        / "测井合成样本"
    )
    manifest = {
        "experiment": 21,
        "stage": "common_pretraining",
        "uses_f3_wide_target": False,
        "f3_source": "narrow_multiband_self_supervision",
        "synthetic_source": "well_constrained_residual",
        "normalization": "per_patch_p99_abs_clean_narrow",
        "arrays": {
            "f3_train": array_summary(f3_root / "train_clean_narrow.npy"),
            "f3_val": array_summary(f3_root / "val_clean_narrow.npy"),
            "synthetic_train_inputs": array_summary(
                synthetic_root / "train_inputs.npy"
            ),
            "synthetic_train_labels": array_summary(
                synthetic_root / "train_labels.npy"
            ),
            "synthetic_val_inputs": array_summary(
                synthetic_root / "val_inputs.npy"
            ),
            "synthetic_val_labels": array_summary(
                synthetic_root / "val_labels.npy"
            ),
        },
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    return manifest


def main():
    ensure_dirs()
    manifest = build_manifest()
    output = COMMON_DATA_DIR / "common_data_manifest.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
