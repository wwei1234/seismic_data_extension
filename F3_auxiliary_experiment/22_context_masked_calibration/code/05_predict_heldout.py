"""Run locked held-out inline and crossline inference for one fold."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_OK", "TRUE")

import numpy as np
import torch


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))
sys.path.append(str(CODE_DIR))

from config import (  # noqa: E402
    CHECKPOINT_DIR,
    DT,
    FOLDS,
    LOCAL_DATA_DIR,
    NARROW_BAND,
    PATCH_SIZE,
    PATCH_STRIDE,
    PREDICTION_DIR,
    SEGY_PATH,
    SHOTNUM,
    ensure_dirs,
)
from leakage_guard import authorize_heldout_reference  # noqa: E402
from phase_model import (  # noqa: E402
    PhaseConsistentResidualModel,
    project_numpy_frequency_band,
)
from segy_reader import read_segy  # noqa: E402
from signal_utils import zero_phase_filter_section  # noqa: E402


def patch_starts(length, size, stride):
    starts = list(range(0, length - size + 1, stride))
    last = length - size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def blend_window(size, edge_weight=0.15):
    values = np.hanning(size).astype(np.float32)
    values = edge_weight + (1.0 - edge_weight) * values
    return np.outer(values, values).astype(np.float32)


def predict_residual(model, section, device):
    output = np.zeros_like(section, dtype=np.float32)
    weight = np.zeros_like(section, dtype=np.float32)
    window = blend_window(PATCH_SIZE)
    model.eval()
    with torch.no_grad():
        for t0 in patch_starts(section.shape[0], PATCH_SIZE, PATCH_STRIDE):
            for x0 in patch_starts(section.shape[1], PATCH_SIZE, PATCH_STRIDE):
                patch = section[t0:t0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
                tensor = torch.from_numpy(patch).float()[None, None].to(device)
                residual = model.net(tensor).cpu().squeeze().numpy()
                output[t0:t0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += residual * window
                weight[t0:t0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += window
    blended = output / np.maximum(weight, 1e-6)
    return project_numpy_frequency_band(blended, dt=DT, **model.projector)


def select_section(cube, geometry, axis, number):
    values = np.asarray(geometry["inlines" if axis == "inline" else "crosslines"])
    found = np.where(values == number)[0]
    if not found.size:
        raise ValueError(f"Missing {axis} {number}")
    index = int(found[0])
    if axis == "inline":
        return cube[index], index
    return np.transpose(cube[:, :, index], (1, 0)), index


def recombine(narrow, residual):
    direct = (narrow + residual).astype(np.float32)
    filtered = project_numpy_frequency_band(
        residual,
        dt=DT,
        low_stop=32.0,
        low_pass=38.0,
        high_pass=85.0,
        high_stop=100.0,
    )
    return direct, (narrow + filtered).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    args = parser.parse_args()
    ensure_dirs()
    fold_root = CHECKPOINT_DIR / args.fold
    checkpoint_path = fold_root / "best_model.pth"
    manifest_path = LOCAL_DATA_DIR / args.fold / "fold_manifest.json"
    lock_path = fold_root / "fold_lock.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    heldout = manifest["heldout_well"]
    requests = (
        ("inline", int(manifest["heldout_inline"])),
        ("crossline", int(manifest["heldout_crossline"])),
    )
    authorizations = {
        axis: authorize_heldout_reference(
            checkpoint_path,
            manifest_path,
            lock_path,
            heldout,
            axis,
            number,
        )
        for axis, number in requests
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseConsistentResidualModel(
        base_c=int(checkpoint["base_c"]),
        projector=checkpoint["projector"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    cube, geometry = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)
    for axis, number in requests:
        reference, selected_index = select_section(
            cube, geometry, axis, number
        )
        reference = reference.astype(np.float32)
        narrow = zero_phase_filter_section(reference, DT, NARROW_BAND)
        scale = max(float(np.percentile(np.abs(narrow), 99)), 1e-8)
        residual = predict_residual(
            model, (narrow / scale).astype(np.float32), device
        ) * scale
        direct, highpass = recombine(narrow, residual)
        prefix = f"{args.fold}_{axis}"
        for name, value in {
            "narrow_input": narrow,
            "residual_prediction": residual,
            "direct_prediction": direct,
            "highpass_prediction": highpass,
            "wide_reference": reference,
        }.items():
            np.save(PREDICTION_DIR / f"{prefix}_{name}.npy", value)
        np.save(PREDICTION_DIR / f"{prefix}_metadata.npy", {
            "fold": args.fold,
            "heldout_well": heldout,
            "section_axis": axis,
            "section_number": number,
            "selected_index": selected_index,
            "normalization": "per_section_p99_abs_narrow",
            "checkpoint_sha256": authorizations[axis][
                "checkpoint_sha256"
            ],
            "reference_read_after_lock": True,
            "uses_heldout_well_wide_target": False,
        })
        print(f"Predicted {args.fold} {axis} {number}", flush=True)


if __name__ == "__main__":
    main()
