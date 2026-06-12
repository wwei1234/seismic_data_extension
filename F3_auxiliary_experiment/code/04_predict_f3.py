import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from config import CHECKPOINT_DIR, DT, FIGURE_DIR, NARROW_BAND, PATCH_SIZE, PATCH_STRIDE, PREDICTION_DIR, SEGY_PATH, SHOTNUM
from model import UNetCBAM
from segy_reader import read_segy
from signal_utils import average_amplitude_spectrum, zero_phase_filter_section


def patch_starts(n, patch_size, stride):
    starts = list(range(0, max(1, n - patch_size + 1), stride))
    if starts[-1] != n - patch_size:
        starts.append(n - patch_size)
    return starts


def minmax_with_stats(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32), mn, mx
    return (2.0 * (x - mn) / (mx - mn) - 1.0).astype(np.float32), mn, mx


def inverse_minmax(x, mn, mx):
    return ((np.asarray(x, dtype=np.float32) + 1.0) * 0.5 * (mx - mn) + mn).astype(np.float32)


def blend_window(patch_size, edge_weight=0.15):
    center = (patch_size - 1) / 2.0
    dist = np.abs(np.arange(patch_size, dtype=np.float32) - center) / max(center, 1.0)
    one_d = edge_weight + (1.0 - edge_weight) * (1.0 - dist)
    return np.outer(one_d, one_d).astype(np.float32)


def predict_section(model, section, device, patch_size=128, stride=64):
    nt, nx = section.shape
    out = np.zeros((nt, nx), dtype=np.float32)
    weight = np.zeros((nt, nx), dtype=np.float32)
    t_starts = patch_starts(nt, patch_size, stride)
    x_starts = patch_starts(nx, patch_size, stride)
    window = blend_window(patch_size)

    model.eval()
    with torch.no_grad():
        for t0 in t_starts:
            for x0 in x_starts:
                patch = section[t0:t0 + patch_size, x0:x0 + patch_size]
                tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                pred = model(tensor).cpu().squeeze().numpy().astype(np.float32)
                out[t0:t0 + patch_size, x0:x0 + patch_size] += pred * window
                weight[t0:t0 + patch_size, x0:x0 + patch_size] += window

    return out / np.maximum(weight, 1e-6)


def save_section_and_spectrum(prefix, name, section):
    fig, ax = plt.subplots(figsize=(9, 5))
    clip = np.nanpercentile(np.abs(section), 99.0)
    clip = max(float(clip), 1e-8)
    ax.imshow(section, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
    ax.set_title(name)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{prefix}_{name.lower().replace(' ', '_')}_section.png", dpi=300)
    plt.close(fig)

    freqs, amp = average_amplitude_spectrum(section, DT)
    spec = np.fft.rfft(section - np.mean(section, axis=0, keepdims=True), axis=0)
    mean_spec = np.mean(spec, axis=1)
    phase = np.angle(mean_spec)
    valid = (freqs >= 1.0) & (freqs <= 100.0) & (amp > 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(freqs, amp, lw=2)
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"{name} amplitude spectrum")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(freqs[valid], phase[valid], ".", ms=3)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].set_title(f"{name} phase spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{prefix}_{name.lower().replace(' ', '_')}_spectrum_phase.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--base-c", type=int, default=None)
    parser.add_argument("--inline-start", type=int, default=0)
    parser.add_argument("--num-inlines", type=int, default=8,
                        help="Use -1 to process all inlines.")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    parser.add_argument("--output-prefix", default="f3_prediction")
    args = parser.parse_args()

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model, map_location="cpu")
    base_c = args.base_c if args.base_c is not None else checkpoint.get("base_c", 32)
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
        PREDICTION_DIR / f"{args.output_prefix}_output.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_out, nt, nx),
    )
    narrow_arr = np.lib.format.open_memmap(
        PREDICTION_DIR / f"{args.output_prefix}_narrow_input.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_out, nt, nx),
    )
    target_arr = np.lib.format.open_memmap(
        PREDICTION_DIR / f"{args.output_prefix}_wide_reference.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_out, nt, nx),
    )
    norm_stats = []

    for out_idx, il_idx in enumerate(range(i0, i1)):
        target = cube[il_idx].astype(np.float32)
        narrow_raw = zero_phase_filter_section(target, DT, NARROW_BAND)
        narrow_norm, narrow_min, narrow_max = minmax_with_stats(narrow_raw)
        pred_norm = predict_section(model, narrow_norm, device, args.patch_size, args.stride)
        pred = inverse_minmax(pred_norm, narrow_min, narrow_max)
        pred_arr[out_idx] = pred
        narrow_arr[out_idx] = narrow_raw.astype(np.float32)
        target_arr[out_idx] = target
        norm_stats.append({
            "inline_index": int(il_idx),
            "narrow_min": float(narrow_min),
            "narrow_max": float(narrow_max),
        })
        print(f"Predicted inline index {il_idx} ({il_idx + 1 - i0}/{i1 - i0})")

    pred_arr.flush()
    narrow_arr.flush()
    target_arr.flush()
    np.save(PREDICTION_DIR / f"{args.output_prefix}_normalization_stats.npy", norm_stats)

    mid = pred_arr.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Low-pass input", "Network output", "F3 reference"]
    for ax, data, title in zip(axes, [narrow_arr[mid], pred_arr[mid], target_arr[mid]], titles):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.output_prefix}_section_compare.png", dpi=300)
    plt.close(fig)

    save_section_and_spectrum(args.output_prefix, "Low-pass input", narrow_arr[mid])
    save_section_and_spectrum(args.output_prefix, "Network output", pred_arr[mid])

    print(f"Saved predictions to {PREDICTION_DIR}")


if __name__ == "__main__":
    main()
