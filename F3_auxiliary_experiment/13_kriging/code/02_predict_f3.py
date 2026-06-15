"""
Predict on F3 seismic data using the residual-band-constrained model.

Pipeline:
    1. Low-pass filter F3 original → narrow-band input
    2. Min-max normalise narrow input to [-1, 1]
    3. Slide 256×256 window, predict residual
    4. Bandpass-filter residual to 25-35-55-75 Hz (post-processing safety)
    5. Add residual to narrow → wide prediction
    6. Apply targeted spectral gain correction to 35-75 Hz
    7. Inverse normalise and save
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    DATA_DIR,
    DT,
    FIGURE_DIR,
    NARROW_BAND,
    PATCH_SIZE,
    PATCH_STRIDE,
    R_MEDIAN,
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
                        help="Target 35-75Hz energy ratio for spectral gain correction.")
    parser.add_argument("--gain-max", type=float, default=2.0,
                        help="Maximum gain factor for spectral correction.")
    args = parser.parse_args()

    ensure_dirs()
    checkpoint = torch.load(args.model, map_location="cpu")
    base_c = checkpoint.get("base_c", 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroMeanResidualModel(base_c=base_c).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    cube = read_segy(SEGY_PATH, shotnum=SHOTNUM)
    n_inline, nt, nx = cube.shape

    pred_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_wide_prediction.npy",
        mode="w+", dtype=np.float32, shape=(n_inline, nt, nx),
    )
    residual_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_residual_prediction.npy",
        mode="w+", dtype=np.float32, shape=(n_inline, nt, nx),
    )
    narrow_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_narrow_input.npy",
        mode="w+", dtype=np.float32, shape=(n_inline, nt, nx),
    )
    target_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_wide_reference.npy",
        mode="w+", dtype=np.float32, shape=(n_inline, nt, nx),
    )

    for il_idx in range(n_inline):
        target = cube[il_idx].astype(np.float32)
        narrow_raw = zero_phase_filter_section(target, DT, NARROW_BAND)

        # Training uses wide's p99; inference only has narrow.
        # Compensate by multiplying back with R_MEDIAN (wide/narrow ratio).
        narrow_scale = float(np.percentile(np.abs(narrow_raw), 99))
        narrow_scale = max(narrow_scale, 1e-8)
        narrow_norm = np.clip(narrow_raw / narrow_scale, -1.0, 1.0).astype(np.float32)
        mn, mx = -narrow_scale, narrow_scale

        # predict residual in normalised space
        residual_norm = predict_residual_section(
            model, narrow_norm, device, args.patch_size, args.stride
        )

        # post-processing: bandpass the residual (safety)
        residual_norm = bandpass_section(residual_norm, DT, RESIDUAL_POST_BAND)

        wide_norm = narrow_norm + residual_norm
        wide_pred = wide_norm * narrow_scale * R_MEDIAN
        residual_pred = wide_pred - narrow_raw

        # targeted spectral gain correction on the residual
        residual_pred = targeted_gain_correction(
            residual_pred,
            fs=250,
            band_low=35, band_high=75,
            target_ratio=args.gain_target_ratio,
            max_gain=args.gain_max,
        )
        wide_pred = narrow_raw + residual_pred

        pred_arr[il_idx] = wide_pred.astype(np.float32)
        residual_arr[il_idx] = residual_pred.astype(np.float32)
        narrow_arr[il_idx] = narrow_raw.astype(np.float32)
        target_arr[il_idx] = target
        print(f"Predicted inline {il_idx + 1}/{n_inline}", flush=True)

    pred_arr.flush()
    residual_arr.flush()
    narrow_arr.flush()
    target_arr.flush()

    # ── quick-look figure ──
    mid = n_inline // 2
    panels = [
        (narrow_arr[mid], "Low-pass input"),
        (residual_arr[mid], "Residual prediction"),
        (pred_arr[mid], "Residual-band prediction"),
        (target_arr[mid], "F3 reference"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
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
