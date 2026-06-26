"""
Predict on F3 seismic data using the residual-band-constrained model.

Pipeline:
    1. Low-pass filter F3 original → narrow-band input
    2. Min-max normalise narrow input to [-1, 1]
    3. Slide 256×256 window, predict residual
    4. Bandpass-filter residual to 40-50-65-90 Hz (post-processing safety)
    5. Apply targeted spectral gain correction to 50-90 Hz
    6. Save two reconstructions:
       direct   = low-pass input + predicted residual
       highpass = low-pass input + high-pass(predicted residual)
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))
sys.path.append(str(CODE_DIR))

from common import (
    DATA_DIR,
    DT,
    FIGURE_DIR,
    NARROW_BAND,
    PATCH_SIZE,
    PATCH_STRIDE,
    RESIDUAL_GAIN_BAND,
    RESIDUAL_POST_BAND,
    SEGY_PATH,
    SHOTNUM,
    bandpass_section,
    blend_window,
    ensure_dirs,
    inverse_minmax,
    minmax_with_stats,
    patch_starts,
    targeted_gain_correction,
    zero_phase_filter_section,
)
from model import UNetCBAM
from segy_reader import read_segy


def highpass_section(section, dt, f1, f2):
    section = np.asarray(section, dtype=np.float64)
    spec = np.fft.rfft(section, axis=0)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    filt = np.ones_like(freqs, dtype=np.float64)
    filt[freqs < f1] = 0.0
    ramp = (freqs >= f1) & (freqs < f2)
    if f2 > f1:
        filt[ramp] = (freqs[ramp] - f1) / (f2 - f1)
    return np.fft.irfft(spec * filt[:, None], n=section.shape[0], axis=0).astype(np.float32)


class ZeroMeanResidualModel(torch.nn.Module):
    def __init__(self, base_c=32):
        super().__init__()
        self.net = UNetCBAM(base_c=base_c)

    def forward(self, x):
        out = self.net(x)
        return out - out.mean(dim=(-2, -1), keepdim=True)


def predict_residual_section(model, normalized_section, device, patch_size, stride):
    nt, nx = normalized_section.shape
    out = np.zeros((nt, nx), dtype=np.float32)
    weight = np.zeros((nt, nx), dtype=np.float32)
    window = blend_window(patch_size)
    model.eval()
    with torch.no_grad():
        for t0 in patch_starts(nt, patch_size, stride):
            for x0 in patch_starts(nx, patch_size, stride):
                patch = normalized_section[t0:t0 + patch_size, x0:x0 + patch_size]
                tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                residual = model(tensor).cpu().squeeze().numpy().astype(np.float32)
                out[t0:t0 + patch_size, x0:x0 + patch_size] += residual * window
                weight[t0:t0 + patch_size, x0:x0 + patch_size] += window
    return out / np.maximum(weight, 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DATA_DIR / "checkpoints" / "best_model.pth"))
    parser.add_argument("--output-prefix", default="residual_band_f3")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    parser.add_argument("--gain-target-ratio", type=float, default=0.35,
                        help="Target residual high-frequency energy ratio for spectral gain correction.")
    parser.add_argument("--gain-max", type=float, default=2.0,
                        help="Maximum gain factor for spectral correction.")
    parser.add_argument("--hp-cut1", type=float, default=35.0)
    parser.add_argument("--hp-cut2", type=float, default=45.0)
    parser.add_argument(
        "--inline-values",
        default="",
        help="Comma-separated SEG-Y inline numbers to predict, e.g. 244,362,442,722. Empty means all inlines.",
    )
    parser.add_argument(
        "--section-axis",
        choices=("inline", "crossline"),
        default="inline",
        help="Predict inline sections or crossline sections.",
    )
    parser.add_argument(
        "--crossline-values",
        default="",
        help="Comma-separated SEG-Y crossline numbers to predict when --section-axis crossline.",
    )
    args = parser.parse_args()

    ensure_dirs()
    checkpoint = torch.load(args.model, map_location="cpu")
    base_c = checkpoint.get("base_c", 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroMeanResidualModel(base_c=base_c).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    cube_result = read_segy(SEGY_PATH, shotnum=SHOTNUM, return_geometry=True)
    if isinstance(cube_result, tuple):
        cube, geometry = cube_result
        inline_numbers = np.asarray(geometry.get("inlines", np.arange(cube.shape[0])), dtype=np.int32)
    else:
        cube = cube_result
        geometry = {}
        inline_numbers = np.arange(cube.shape[0], dtype=np.int32)

    crossline_numbers = np.asarray(geometry.get("crosslines", np.arange(cube.shape[2])), dtype=np.int32)

    if args.section_axis == "inline":
        value_arg = args.inline_values
        axis_numbers = inline_numbers
        section_source = "inline"
    else:
        value_arg = args.crossline_values
        axis_numbers = crossline_numbers
        section_source = "crossline"

    if value_arg.strip():
        requested = [int(x.strip()) for x in value_arg.split(",") if x.strip()]
        selected_indices = []
        missing = []
        for section_value in requested:
            matches = np.where(axis_numbers == section_value)[0]
            if matches.size == 0:
                missing.append(section_value)
            else:
                selected_indices.append(int(matches[0]))
        if missing:
            raise ValueError(f"Requested {section_source} values not found in SEG-Y geometry: {missing}")
    else:
        selected_indices = list(range(axis_numbers.size))
        requested = [int(x) for x in axis_numbers]

    selected_section_numbers = [int(axis_numbers[idx]) for idx in selected_indices]
    if args.section_axis == "inline":
        cube = cube[selected_indices]
    else:
        cube = np.transpose(cube[:, :, selected_indices], (2, 1, 0))
    n_section, nt, nx = cube.shape

    pred_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_wide_prediction.npy",
        mode="w+", dtype=np.float32, shape=(n_section, nt, nx),
    )
    direct_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_direct_wide_prediction.npy",
        mode="w+", dtype=np.float32, shape=(n_section, nt, nx),
    )
    hp_pred_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_highpass_wide_prediction.npy",
        mode="w+", dtype=np.float32, shape=(n_section, nt, nx),
    )
    residual_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_residual_prediction.npy",
        mode="w+", dtype=np.float32, shape=(n_section, nt, nx),
    )
    residual_hp_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_highpass_residual_prediction.npy",
        mode="w+", dtype=np.float32, shape=(n_section, nt, nx),
    )
    narrow_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_narrow_input.npy",
        mode="w+", dtype=np.float32, shape=(n_section, nt, nx),
    )
    target_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_wide_reference.npy",
        mode="w+", dtype=np.float32, shape=(n_section, nt, nx),
    )

    for section_idx in range(n_section):
        target = cube[section_idx].astype(np.float32)
        narrow_raw = zero_phase_filter_section(target, DT, NARROW_BAND)

        narrow_scale = float(np.percentile(np.abs(narrow_raw), 99))
        narrow_scale = max(narrow_scale, 1e-8)
        narrow_norm = np.clip(narrow_raw / narrow_scale, -1.0, 1.0).astype(np.float32)

        # predict residual in normalised space
        residual_norm = predict_residual_section(
            model, narrow_norm, device, args.patch_size, args.stride
        )

        # post-processing: bandpass the residual (safety)
        residual_norm = bandpass_section(residual_norm, DT, RESIDUAL_POST_BAND)

        wide_norm = narrow_norm + residual_norm
        wide_pred = wide_norm * narrow_scale
        residual_pred = wide_pred - narrow_raw

        # targeted spectral gain correction on the residual
        residual_pred = targeted_gain_correction(
            residual_pred,
            fs=250,
            band_low=RESIDUAL_GAIN_BAND[0], band_high=RESIDUAL_GAIN_BAND[1],
            target_ratio=args.gain_target_ratio,
            max_gain=args.gain_max,
        )
        wide_direct = narrow_raw + residual_pred
        residual_hp = highpass_section(residual_pred, DT, args.hp_cut1, args.hp_cut2)
        wide_highpass = narrow_raw + residual_hp

        pred_arr[section_idx] = wide_direct.astype(np.float32)
        direct_arr[section_idx] = wide_direct.astype(np.float32)
        hp_pred_arr[section_idx] = wide_highpass.astype(np.float32)
        residual_arr[section_idx] = residual_pred.astype(np.float32)
        residual_hp_arr[section_idx] = residual_hp.astype(np.float32)
        narrow_arr[section_idx] = narrow_raw.astype(np.float32)
        target_arr[section_idx] = target
        print(
            f"Predicted {section_source} {section_idx + 1}/{n_section} "
            f"(SEG-Y {section_source} {selected_section_numbers[section_idx]})",
            flush=True,
        )

    pred_arr.flush()
    direct_arr.flush()
    hp_pred_arr.flush()
    residual_arr.flush()
    residual_hp_arr.flush()
    narrow_arr.flush()
    target_arr.flush()
    np.save(DATA_DIR / f"{args.output_prefix}_inline_metadata.npy", {
        "section_axis": args.section_axis,
        "selected_indices": selected_indices,
        "section_numbers": selected_section_numbers,
        "inline_numbers": selected_section_numbers if args.section_axis == "inline" else inline_numbers,
        "crossline_numbers": selected_section_numbers if args.section_axis == "crossline" else crossline_numbers,
        "requested_inline_values": requested if args.section_axis == "inline" else [],
        "requested_crossline_values": requested if args.section_axis == "crossline" else [],
        "crosslines": crossline_numbers,
        "inlines": inline_numbers,
    })

    # ── quick-look figure ──
    mid = n_section // 2
    panels = [
        (narrow_arr[mid], "Low-pass input"),
        (residual_arr[mid], "Residual prediction"),
        (direct_arr[mid], "Direct recombination"),
        (hp_pred_arr[mid], "High-pass residual recombination"),
        (target_arr[mid], "F3 reference"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    for ax, (data, title) in zip(axes, panels):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_section_compare.png", dpi=300)
    plt.close(fig)

    print(f"Saved predictions to {DATA_DIR}")


if __name__ == "__main__":
    main()
