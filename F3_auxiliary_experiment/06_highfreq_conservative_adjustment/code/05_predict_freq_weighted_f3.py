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
    DATA_DIR,
    DT,
    FIGURE_DIR,
    NARROW_BAND,
    RESIDUAL_POST_BAND,
    ensure_dirs,
    trapezoid_band,
    zero_phase_filter_section,
)
from config import SEGY_PATH, SHOTNUM
from model import UNetCBAM
from segy_reader import read_segy


PATCH_SIZE = 256
PATCH_STRIDE = 64


class ZeroMeanResidualModel(torch.nn.Module):
    def __init__(self, base_c=32):
        super().__init__()
        self.net = UNetCBAM(base_c=base_c)

    def forward(self, x):
        out = self.net(x)
        return out - out.mean(dim=(-2, -1), keepdim=True)


def minmax_with_stats(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32), mn, mx
    return (2.0 * (x - mn) / (mx - mn) - 1.0).astype(np.float32), mn, mx


def inverse_minmax(x, mn, mx):
    return ((np.asarray(x, dtype=np.float32) + 1.0) * 0.5 * (mx - mn) + mn).astype(np.float32)


def patch_starts(n, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    starts = list(range(0, n - patch_size + 1, stride))
    if starts[-1] != n - patch_size:
        starts.append(n - patch_size)
    return starts


def blend_window(patch_size=PATCH_SIZE, edge_weight=0.15):
    center = (patch_size - 1) / 2.0
    dist = np.abs(np.arange(patch_size, dtype=np.float32) - center) / max(center, 1.0)
    one_d = edge_weight + (1.0 - edge_weight) * (1.0 - dist)
    return np.outer(one_d, one_d).astype(np.float32)


def bandpass_section(section, dt, band):
    section = np.asarray(section, dtype=np.float64)
    spec = np.fft.rfft(section, axis=0)
    freqs = np.fft.rfftfreq(section.shape[0], dt)
    filt = trapezoid_band(freqs, *band)[:, None]
    return np.fft.irfft(spec * filt, n=section.shape[0], axis=0).astype(np.float32)


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
    parser.add_argument("--model", default=str(DATA_DIR / "checkpoints" / "best_freq_weighted_model.pth"))
    parser.add_argument("--output-prefix", default="freq_weighted_f3_full")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
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
        mode="w+",
        dtype=np.float32,
        shape=(n_inline, nt, nx),
    )
    residual_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_residual_prediction.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_inline, nt, nx),
    )
    narrow_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_narrow_input.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_inline, nt, nx),
    )
    target_arr = np.lib.format.open_memmap(
        DATA_DIR / f"{args.output_prefix}_wide_reference.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_inline, nt, nx),
    )

    for il_idx in range(n_inline):
        target = cube[il_idx].astype(np.float32)
        narrow_raw = zero_phase_filter_section(target, DT, NARROW_BAND)
        narrow_norm, mn, mx = minmax_with_stats(narrow_raw)
        residual_norm = predict_residual_section(
            model, narrow_norm, device, args.patch_size, args.stride
        )
        residual_norm = bandpass_section(residual_norm, DT, RESIDUAL_POST_BAND)
        wide_norm = narrow_norm + residual_norm
        wide_pred = inverse_minmax(wide_norm, mn, mx)
        residual_pred = wide_pred - narrow_raw

        pred_arr[il_idx] = wide_pred.astype(np.float32)
        residual_arr[il_idx] = residual_pred.astype(np.float32)
        narrow_arr[il_idx] = narrow_raw.astype(np.float32)
        target_arr[il_idx] = target
        print(f"Predicted inline {il_idx + 1}/{n_inline}", flush=True)

    pred_arr.flush()
    residual_arr.flush()
    narrow_arr.flush()
    target_arr.flush()

    mid = n_inline // 2
    panels = [
        (narrow_arr[mid], "Low-pass input"),
        (residual_arr[mid], "Frequency-weighted residual"),
        (pred_arr[mid], "Frequency-weighted prediction"),
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
    fig.savefig(FIGURE_DIR / f"05_{args.output_prefix}_section_compare.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
