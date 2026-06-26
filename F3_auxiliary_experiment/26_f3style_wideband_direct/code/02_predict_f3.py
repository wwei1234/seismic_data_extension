"""Run direct wideband prediction on selected F3 inline or crossline sections."""

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

from common import blend_window, patch_starts
from config import DATA_DIR, DT, NARROW_BAND, PATCH_SIZE, PATCH_STRIDE, SEGY_PATH, SHOTNUM
from segy_reader import read_segy
from signal_utils import zero_phase_filter_section
from wideband_inference import denormalize_wide_prediction
from wideband_training import WidebandModel


def predict_section(model, section, device, patch_size, stride):
    nt, nx = section.shape
    output = np.zeros((nt, nx), dtype=np.float32)
    weight = np.zeros((nt, nx), dtype=np.float32)
    window = blend_window(patch_size)
    model.eval()
    with torch.no_grad():
        for t0 in patch_starts(nt, patch_size, stride):
            for x0 in patch_starts(nx, patch_size, stride):
                patch = section[t0:t0 + patch_size, x0:x0 + patch_size]
                tensor = torch.from_numpy(patch).float()[None, None].to(device)
                prediction = model(tensor).cpu().squeeze().numpy().astype(np.float32)
                output[t0:t0 + patch_size, x0:x0 + patch_size] += prediction * window
                weight[t0:t0 + patch_size, x0:x0 + patch_size] += window
    return output / np.maximum(weight, 1e-6)


def parse_values(text):
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def select_sections(cube, geometry, section_axis, requested_values):
    inline_numbers = np.asarray(geometry["inlines"], dtype=np.int32)
    crossline_numbers = np.asarray(geometry["crosslines"], dtype=np.int32)
    axis_numbers = inline_numbers if section_axis == "inline" else crossline_numbers
    selected_indices = []
    for value in requested_values:
        matches = np.where(axis_numbers == value)[0]
        if not matches.size:
            raise ValueError(f"SEG-Y {section_axis} not found: {value}")
        selected_indices.append(int(matches[0]))
    if section_axis == "inline":
        sections = cube[selected_indices]
    else:
        sections = np.transpose(cube[:, :, selected_indices], (2, 1, 0))
    return sections, selected_indices, inline_numbers, crossline_numbers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(DATA_DIR / "checkpoints" / "best_model.pth"),
    )
    parser.add_argument("--output-prefix", default="wide17")
    parser.add_argument("--section-axis", choices=("inline", "crossline"), default="inline")
    parser.add_argument("--inline-values", default="244,362,442,722")
    parser.add_argument("--crossline-values", default="336,387,848,1007")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    if checkpoint.get("target_type") != "wide_band":
        raise ValueError("Checkpoint is not a direct wideband model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WidebandModel(base_c=int(checkpoint.get("base_c", 32))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    cube, geometry = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)
    requested = parse_values(
        args.inline_values if args.section_axis == "inline" else args.crossline_values
    )
    sections, selected_indices, inline_numbers, crossline_numbers = select_sections(
        cube, geometry, args.section_axis, requested
    )

    output_dir = DATA_DIR / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = sections.shape
    narrow_arr = np.lib.format.open_memmap(
        output_dir / f"{args.output_prefix}_narrow_input.npy",
        mode="w+", dtype=np.float32, shape=shape,
    )
    pred_arr = np.lib.format.open_memmap(
        output_dir / f"{args.output_prefix}_wide_prediction.npy",
        mode="w+", dtype=np.float32, shape=shape,
    )
    ref_arr = np.lib.format.open_memmap(
        output_dir / f"{args.output_prefix}_wide_reference.npy",
        mode="w+", dtype=np.float32, shape=shape,
    )

    for idx, reference in enumerate(sections):
        reference = reference.astype(np.float32)
        narrow_raw = zero_phase_filter_section(reference, DT, NARROW_BAND)
        narrow_scale = max(float(np.percentile(np.abs(narrow_raw), 99)), 1e-8)
        narrow_norm = np.clip(narrow_raw / narrow_scale, -1.0, 1.0).astype(np.float32)
        wide_norm = predict_section(
            model, narrow_norm, device, args.patch_size, args.stride
        )
        wide_prediction = denormalize_wide_prediction(wide_norm, narrow_scale)
        narrow_arr[idx] = narrow_raw
        pred_arr[idx] = wide_prediction
        ref_arr[idx] = reference
        print(
            f"Predicted {args.section_axis} {idx + 1}/{shape[0]} "
            f"(SEG-Y {args.section_axis} {requested[idx]})",
            flush=True,
        )

    narrow_arr.flush()
    pred_arr.flush()
    ref_arr.flush()
    metadata = {
        "section_axis": args.section_axis,
        "section_numbers": requested,
        "selected_indices": selected_indices,
        "inline_numbers": inline_numbers,
        "crossline_numbers": crossline_numbers,
        "prediction_type": "direct_wide_band",
        "normalization": "per_section_p99_abs_narrow",
        "shape": shape,
    }
    np.save(output_dir / f"{args.output_prefix}_metadata.npy", metadata)
    print(f"Saved predictions to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
