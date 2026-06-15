import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2] / "code"))

from common import (
    CHECKPOINT_DIR,
    DATA_DIR,
    DT,
    FIGURE_DIR,
    NARROW_BAND,
    PATCH_SIZE,
    PATCH_STRIDE,
    blend_window,
    ensure_dirs,
    inverse_minmax,
    minmax_with_stats,
    patch_starts,
    zero_phase_filter_section,
)
from model import UNetCBAM
from segy_reader import read_segy
from config import SEGY_PATH, SHOTNUM


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
    parser.add_argument("--model", default=str(CHECKPOINT_DIR / "best_residual_model.pth"))
    parser.add_argument("--inline-start", type=int, default=0)
    parser.add_argument("--num-inlines", type=int, default=-1)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    parser.add_argument("--output-prefix", default="residual_f3_full")
    args = parser.parse_args()

    ensure_dirs()
    checkpoint = torch.load(args.model, map_location="cpu")
    base_c = checkpoint.get("base_c", 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetCBAM(base_c=base_c).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    cube = read_segy(SEGY_PATH, shotnum=SHOTNUM)
    n_inline = cube.shape[0]
    i0 = max(0, args.inline_start)
    i1 = n_inline if args.num_inlines < 0 else min(n_inline, i0 + args.num_inlines)
    n_out = i1 - i0
    nt, nx = cube.shape[1], cube.shape[2]

    pred_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_wide_prediction.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_out, nt, nx),
    )
    residual_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_residual_prediction.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_out, nt, nx),
    )
    narrow_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_narrow_input.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_out, nt, nx),
    )
    target_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_wide_reference.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_out, nt, nx),
    )
    stats = []

    for out_idx, il_idx in enumerate(range(i0, i1)):
        target = cube[il_idx].astype(np.float32)
        narrow_raw = zero_phase_filter_section(target, DT, NARROW_BAND)
        narrow_norm, mn, mx = minmax_with_stats(narrow_raw)
        residual_norm = predict_residual_section(
            model, narrow_norm, device, args.patch_size, args.stride
        )
        wide_norm = narrow_norm + residual_norm
        wide_pred = inverse_minmax(wide_norm, mn, mx)
        residual_pred = inverse_minmax(wide_norm, mn, mx) - narrow_raw

        pred_arr[out_idx] = wide_pred.astype(np.float32)
        residual_arr[out_idx] = residual_pred.astype(np.float32)
        narrow_arr[out_idx] = narrow_raw.astype(np.float32)
        target_arr[out_idx] = target
        stats.append({"inline_index": int(il_idx), "narrow_min": float(mn), "narrow_max": float(mx)})
        print(f"Predicted inline index {il_idx} ({out_idx + 1}/{n_out})", flush=True)

    pred_arr.flush()
    residual_arr.flush()
    narrow_arr.flush()
    target_arr.flush()
    np.save(DATA_DIR / f"{args.output_prefix}_normalization_stats.npy", stats)

    mid = n_out // 2
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        (narrow_arr[mid], "Low-pass input"),
        (residual_arr[mid], "Predicted residual"),
        (pred_arr[mid], "Residual wide prediction"),
        (target_arr[mid], "F3 reference"),
    ]
    for ax, (data, title) in zip(axes, panels):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.output_prefix}_section_compare.png", dpi=300)
    plt.close(fig)
    print(f"Saved residual predictions to {DATA_DIR}")


if __name__ == "__main__":
    main()
