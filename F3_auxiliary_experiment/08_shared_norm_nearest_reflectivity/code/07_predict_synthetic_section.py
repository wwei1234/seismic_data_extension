import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from config import CHECKPOINT_DIR, DATA_DIR, DT, FIGURE_DIR, PATCH_SIZE, PATCH_STRIDE
from model import UNetCBAM
from signal_utils import average_amplitude_spectrum


def patch_starts(n, patch_size, stride):
    starts = list(range(0, n - patch_size + 1, stride))
    if starts[-1] != n - patch_size:
        starts.append(n - patch_size)
    return starts


def blend_window(patch_size, edge_weight=0.15):
    center = (patch_size - 1) / 2.0
    dist = np.abs(np.arange(patch_size, dtype=np.float32) - center) / max(center, 1.0)
    one_d = edge_weight + (1.0 - edge_weight) * (1.0 - dist)
    return np.outer(one_d, one_d).astype(np.float32)


def predict_section(model, section, device, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    nt, nx = section.shape
    out = np.zeros((nt, nx), dtype=np.float32)
    weight = np.zeros((nt, nx), dtype=np.float32)
    window = blend_window(patch_size)
    model.eval()
    with torch.no_grad():
        for t0 in patch_starts(nt, patch_size, stride):
            for x0 in patch_starts(nx, patch_size, stride):
                patch = section[t0:t0 + patch_size, x0:x0 + patch_size]
                tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                pred = model(tensor).cpu().squeeze().numpy().astype(np.float32)
                out[t0:t0 + patch_size, x0:x0 + patch_size] += pred * window
                weight[t0:t0 + patch_size, x0:x0 + patch_size] += window
    return out / np.maximum(weight, 1e-6)


def metrics(pred, target):
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    corr = float(np.corrcoef(pred.ravel(), target.ravel())[0, 1])
    return mae, rmse, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section-id", default=None)
    parser.add_argument("--model", default=str(CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    parser.add_argument("--output-prefix", default="07_train_section")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    inputs = np.load(DATA_DIR / "synthetic_section_inputs.npy", allow_pickle=True).item()
    labels = np.load(DATA_DIR / "synthetic_section_labels.npy", allow_pickle=True).item()
    section_id = args.section_id or sorted(inputs.keys())[0]
    if section_id not in inputs:
        raise KeyError(f"Unknown section id: {section_id}. Available: {sorted(inputs.keys())}")

    x = inputs[section_id].astype(np.float32)
    y = labels[section_id].astype(np.float32)

    checkpoint = torch.load(args.model, map_location="cpu")
    base_c = checkpoint.get("base_c", 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetCBAM(base_c=base_c).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred = predict_section(model, x, device, args.patch_size, args.stride)
    mae, rmse, corr = metrics(pred, y)

    np.save(DATA_DIR / f"{args.output_prefix}_{section_id}_prediction.npy", pred)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    panels = [
        (x, "Synthetic narrow input"),
        (pred, "Network prediction"),
        (y, "Synthetic wide label"),
        (pred - y, "Prediction - label"),
    ]
    for ax, (data, title) in zip(axes.ravel(), panels):
        clip = np.nanpercentile(np.abs(data), 99.0)
        clip = max(float(clip), 1e-8)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.suptitle(f"{section_id}: MAE={mae:.4f}, RMSE={rmse:.4f}, corr={corr:.4f}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.output_prefix}_{section_id}_section_compare.png", dpi=300)
    plt.close(fig)

    freq_x, amp_x = average_amplitude_spectrum(x, DT)
    freq_p, amp_p = average_amplitude_spectrum(pred, DT)
    freq_y, amp_y = average_amplitude_spectrum(y, DT)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_x, amp_x, "b-", lw=2, label="Synthetic narrow input")
    ax.plot(freq_p, amp_p, "r-", lw=2, label="Network prediction")
    ax.plot(freq_y, amp_y, "k--", lw=2, label="Synthetic wide label")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.output_prefix}_{section_id}_spectrum_compare.png", dpi=300)
    plt.close(fig)

    print(f"section_id: {section_id}")
    print(f"shape: {x.shape}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Correlation: {corr:.6f}")
    print(f"Saved section figure: {FIGURE_DIR / f'{args.output_prefix}_{section_id}_section_compare.png'}")
    print(f"Saved spectrum figure: {FIGURE_DIR / f'{args.output_prefix}_{section_id}_spectrum_compare.png'}")


if __name__ == "__main__":
    main()
