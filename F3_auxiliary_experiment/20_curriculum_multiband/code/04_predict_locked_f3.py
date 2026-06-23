"""Run locked experiment 20 inference and save direct/highpass recombinations."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))
sys.path.append(str(CODE_DIR))

from config import (  # noqa: E402
    CHECKPOINT_DIR,
    DT,
    NARROW_BAND,
    PATCH_SIZE,
    PATCH_STRIDE,
    PREDICTION_DIR,
    SEGY_PATH,
    SHOTNUM,
    ensure_dirs,
)
from leakage_guard import verify_model_lock  # noqa: E402
from phase_model import (  # noqa: E402
    PhaseConsistentResidualModel,
    project_numpy_frequency_band,
)
from segy_reader import read_segy  # noqa: E402
from signal_utils import zero_phase_filter_section  # noqa: E402


def validate_before_reference_read(checkpoint_path, lock_path):
    return verify_model_lock(checkpoint_path, lock_path)


def recombine(narrow, residual, dt=DT):
    narrow = np.asarray(narrow, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    direct = (narrow + residual).astype(np.float32)
    highpass_residual = project_numpy_frequency_band(
        residual,
        dt=dt,
        low_stop=32.0,
        low_pass=38.0,
        high_pass=85.0,
        high_stop=100.0,
    )
    highpass = (narrow + highpass_residual).astype(np.float32)
    return direct, highpass


def patch_starts(length, patch_size, stride):
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_window(patch_size, edge_weight=0.15):
    one_dimensional = np.hanning(patch_size).astype(np.float32)
    one_dimensional = edge_weight + (1.0 - edge_weight) * one_dimensional
    return np.outer(one_dimensional, one_dimensional).astype(np.float32)


def predict_residual(model, normalized_section, device, patch_size, stride):
    nt, nx = normalized_section.shape
    output = np.zeros((nt, nx), dtype=np.float32)
    weight = np.zeros((nt, nx), dtype=np.float32)
    window = blend_window(patch_size)
    model.eval()
    with torch.no_grad():
        for time_start in patch_starts(nt, patch_size, stride):
            for spatial_start in patch_starts(nx, patch_size, stride):
                patch = normalized_section[
                    time_start:time_start + patch_size,
                    spatial_start:spatial_start + patch_size,
                ]
                tensor = torch.from_numpy(patch).float()[None, None].to(device)
                raw_residual = model.net(tensor)
                residual = raw_residual.cpu().squeeze().numpy().astype(np.float32)
                output[
                    time_start:time_start + patch_size,
                    spatial_start:spatial_start + patch_size,
                ] += residual * window
                weight[
                    time_start:time_start + patch_size,
                    spatial_start:spatial_start + patch_size,
                ] += window
    blended = output / np.maximum(weight, 1e-6)
    return project_numpy_frequency_band(blended, dt=DT, **model.projector)


def select_sections(cube, geometry, axis, values):
    key = "inlines" if axis == "inline" else "crosslines"
    axis_numbers = np.asarray(geometry[key])
    indices = []
    for value in values:
        found = np.where(axis_numbers == value)[0]
        if not found.size:
            raise ValueError(f"Missing SEG-Y {axis}: {value}")
        indices.append(int(found[0]))
    if axis == "inline":
        sections = cube[indices]
    else:
        sections = np.transpose(cube[:, :, indices], (2, 1, 0))
    return sections, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--lock", default=str(CHECKPOINT_DIR / "model_lock.json"))
    parser.add_argument("--section-axis", choices=("inline", "crossline"), required=True)
    parser.add_argument("--values", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    args = parser.parse_args()
    ensure_dirs()

    lock_metadata = validate_before_reference_read(args.model, args.lock)
    checkpoint = torch.load(args.model, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseConsistentResidualModel(
        base_c=int(checkpoint.get("base_c", 32)),
        dt=DT,
        projector=checkpoint["projector"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    values = [int(value.strip()) for value in args.values.split(",") if value.strip()]
    cube, geometry = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)
    sections, indices = select_sections(cube, geometry, args.section_axis, values)
    shape = sections.shape
    suffixes = (
        "narrow_input",
        "residual_prediction",
        "direct_prediction",
        "highpass_prediction",
        "wide_reference",
    )
    outputs = {
        suffix: np.lib.format.open_memmap(
            PREDICTION_DIR / f"{args.output_prefix}_{suffix}.npy",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        for suffix in suffixes
    }

    for index, reference in enumerate(sections):
        reference = reference.astype(np.float32)
        narrow = zero_phase_filter_section(reference, DT, NARROW_BAND)
        scale = max(float(np.percentile(np.abs(narrow), 99)), 1e-8)
        normalized = (narrow / scale).astype(np.float32)
        residual = predict_residual(
            model,
            normalized,
            device,
            args.patch_size,
            args.stride,
        ) * scale
        direct, highpass = recombine(narrow, residual, DT)
        outputs["narrow_input"][index] = narrow
        outputs["residual_prediction"][index] = residual
        outputs["direct_prediction"][index] = direct
        outputs["highpass_prediction"][index] = highpass
        outputs["wide_reference"][index] = reference
        print(
            f"Predicted {args.section_axis} {values[index]} "
            f"({index + 1}/{len(values)})",
            flush=True,
        )
    for array in outputs.values():
        array.flush()
    np.save(PREDICTION_DIR / f"{args.output_prefix}_metadata.npy", {
        "experiment": 20,
        "section_axis": args.section_axis,
        "section_numbers": values,
        "selected_indices": indices,
        "normalization": "per_section_p99_abs_narrow",
        "lock_sha256": lock_metadata["sha256"],
        "reference_read_after_lock": True,
        "uses_f3_wide_target_in_training": False,
        "shape": shape,
    })


if __name__ == "__main__":
    main()
