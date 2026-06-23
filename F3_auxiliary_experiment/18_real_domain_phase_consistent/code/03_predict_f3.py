"""Predict guarded well inline/crossline sections with experiment 18."""

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

from config import (
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
from phase_model import PhaseConsistentResidualModel, project_numpy_frequency_band
from real_f3_samples import patch_starts
from segy_reader import read_segy
from signal_utils import zero_phase_filter_section


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
        for t0 in patch_starts(nt, patch_size, stride):
            for x0 in patch_starts(nx, patch_size, stride):
                patch = normalized_section[t0:t0 + patch_size, x0:x0 + patch_size]
                tensor = torch.from_numpy(patch).float()[None, None].to(device)
                _, residual = model.forward_with_residual(tensor)
                residual = residual.cpu().squeeze().numpy().astype(np.float32)
                output[t0:t0 + patch_size, x0:x0 + patch_size] += residual * window
                weight[t0:t0 + patch_size, x0:x0 + patch_size] += window
    blended = output / np.maximum(weight, 1e-6)
    return project_numpy_frequency_band(blended, dt=DT)


def select_sections(cube, geometry, axis, values):
    axis_numbers = np.asarray(geometry["inlines" if axis == "inline" else "crosslines"])
    indices = []
    for value in values:
        matches = np.where(axis_numbers == value)[0]
        if not matches.size:
            raise ValueError(f"Missing SEG-Y {axis}: {value}")
        indices.append(int(matches[0]))
    if axis == "inline":
        sections = cube[indices]
    else:
        sections = np.transpose(cube[:, :, indices], (2, 1, 0))
    return sections, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--section-axis", choices=("inline", "crossline"), required=True)
    parser.add_argument("--values", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    args = parser.parse_args()

    ensure_dirs()
    values = [int(value.strip()) for value in args.values.split(",") if value.strip()]
    checkpoint = torch.load(args.model, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseConsistentResidualModel(
        base_c=int(checkpoint.get("base_c", 32)),
        dt=DT,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    cube, geometry = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)
    sections, indices = select_sections(cube, geometry, args.section_axis, values)
    shape = sections.shape
    outputs = {}
    for suffix in ("narrow_input", "residual_prediction", "wide_prediction", "wide_reference"):
        outputs[suffix] = np.lib.format.open_memmap(
            PREDICTION_DIR / f"{args.output_prefix}_{suffix}.npy",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )

    for index, reference in enumerate(sections):
        reference = reference.astype(np.float32)
        narrow_raw = zero_phase_filter_section(reference, DT, NARROW_BAND)
        scale = max(float(np.percentile(np.abs(narrow_raw), 99)), 1e-8)
        narrow_norm = (narrow_raw / scale).astype(np.float32)
        residual_norm = predict_residual(
            model,
            narrow_norm,
            device,
            args.patch_size,
            args.stride,
        )
        residual_raw = residual_norm * scale
        wide_prediction = narrow_raw + residual_raw
        outputs["narrow_input"][index] = narrow_raw
        outputs["residual_prediction"][index] = residual_raw
        outputs["wide_prediction"][index] = wide_prediction
        outputs["wide_reference"][index] = reference
        print(
            f"Predicted {args.section_axis} {values[index]} "
            f"({index + 1}/{len(values)})",
            flush=True,
        )
    for array in outputs.values():
        array.flush()
    np.save(PREDICTION_DIR / f"{args.output_prefix}_metadata.npy", {
        "section_axis": args.section_axis,
        "section_numbers": values,
        "selected_indices": indices,
        "normalization": "p99_abs_narrow_unclipped",
        "prediction_type": "hard_bypass_projected_residual",
        "shape": shape,
    })


if __name__ == "__main__":
    main()
