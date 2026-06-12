import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))

from config import CHECKPOINT_DIR, DATA_DIR, DT, FIGURE_DIR
from model import UNetCBAM


def average_patch_spectrum(patches, dt):
    patches = np.asarray(patches, dtype=np.float32)
    work = patches - np.mean(patches, axis=1, keepdims=True)
    spec = np.fft.rfft(work, axis=1)
    amp = np.mean(np.abs(spec), axis=(0, 2))
    freqs = np.fft.rfftfreq(patches.shape[1], dt)
    if np.max(amp) > 0:
        amp = amp / np.max(amp)
    return freqs, amp


def metrics(pred, target):
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    corr = float(np.corrcoef(pred.ravel(), target.ravel())[0, 1])
    return mae, rmse, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-show", type=int, default=4)
    parser.add_argument("--output-prefix", default="08_validation")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    val_inputs = np.load(DATA_DIR / "val_inputs.npy").astype(np.float32)
    val_labels = np.load(DATA_DIR / "val_labels.npy").astype(np.float32)

    checkpoint = torch.load(args.model, map_location="cpu")
    base_c = checkpoint.get("base_c", 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetCBAM(base_c=base_c).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(val_inputs).unsqueeze(1),
        torch.from_numpy(val_labels).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for x, _ in loader:
            pred = model(x.to(device)).cpu().squeeze(1).numpy().astype(np.float32)
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)

    mae, rmse, corr = metrics(preds, val_labels)
    np.save(DATA_DIR / f"{args.output_prefix}_predictions.npy", preds)
    np.save(DATA_DIR / f"{args.output_prefix}_metrics.npy", {
        "MAE": mae,
        "RMSE": rmse,
        "Correlation": corr,
        "num_patches": int(val_inputs.shape[0]),
    })

    show_n = min(args.num_show, val_inputs.shape[0])
    fig, axes = plt.subplots(show_n, 4, figsize=(14, 3.2 * show_n))
    if show_n == 1:
        axes = axes[None, :]
    for row in range(show_n):
        panels = [
            (val_inputs[row], "Val narrow input"),
            (preds[row], "Prediction"),
            (val_labels[row], "Val wide label"),
            (preds[row] - val_labels[row], "Prediction - label"),
        ]
        for col, (data, title) in enumerate(panels):
            ax = axes[row, col]
            clip = np.nanpercentile(np.abs(data), 99.0)
            clip = max(float(clip), 1e-8)
            ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
            ax.set_title(title if row == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Validation patches: MAE={mae:.4f}, RMSE={rmse:.4f}, corr={corr:.4f}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.output_prefix}_patch_compare.png", dpi=300)
    plt.close(fig)

    freq_x, amp_x = average_patch_spectrum(val_inputs, DT)
    freq_p, amp_p = average_patch_spectrum(preds, DT)
    freq_y, amp_y = average_patch_spectrum(val_labels, DT)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_x, amp_x, "b-", lw=2, label="Validation narrow input")
    ax.plot(freq_p, amp_p, "r-", lw=2, label="Network prediction")
    ax.plot(freq_y, amp_y, "k--", lw=2, label="Validation wide label")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized average amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.output_prefix}_spectrum_compare.png", dpi=300)
    plt.close(fig)

    print(f"Validation patches: {val_inputs.shape[0]}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Correlation: {corr:.6f}")
    print(f"Saved patch figure: {FIGURE_DIR / f'{args.output_prefix}_patch_compare.png'}")
    print(f"Saved spectrum figure: {FIGURE_DIR / f'{args.output_prefix}_spectrum_compare.png'}")


if __name__ == "__main__":
    main()
