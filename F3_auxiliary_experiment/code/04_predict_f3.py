import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from config import CHECKPOINT_DIR, DATA_DIR, DT, FIGURE_DIR, NARROW_BAND, PREDICTION_DIR, SEGY_PATH, SHOTNUM
from model import UNetCBAM
from segy_reader import read_segy
from signal_utils import normalize_minmax, zero_phase_filter_section


def patch_starts(n, patch_size, stride):
    starts = list(range(0, max(1, n - patch_size + 1), stride))
    if starts[-1] != n - patch_size:
        starts.append(n - patch_size)
    return starts


def predict_section(model, section, device, patch_size=128, stride=64):
    nt, nx = section.shape
    out = np.zeros((nt, nx), dtype=np.float32)
    weight = np.zeros((nt, nx), dtype=np.float32)
    t_starts = patch_starts(nt, patch_size, stride)
    x_starts = patch_starts(nx, patch_size, stride)

    model.eval()
    with torch.no_grad():
        for t0 in t_starts:
            for x0 in x_starts:
                patch = normalize_minmax(section[t0:t0 + patch_size, x0:x0 + patch_size])
                tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                pred = model(tensor).cpu().squeeze().numpy().astype(np.float32)
                out[t0:t0 + patch_size, x0:x0 + patch_size] += pred
                weight[t0:t0 + patch_size, x0:x0 + patch_size] += 1.0

    return out / np.maximum(weight, 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--base-c", type=int, default=None)
    parser.add_argument("--inline-start", type=int, default=0)
    parser.add_argument("--num-inlines", type=int, default=8,
                        help="Use -1 to process all inlines.")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
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

    pred_list = []
    narrow_list = []
    target_list = []

    for il_idx in range(i0, i1):
        target = normalize_minmax(cube[il_idx])
        narrow = normalize_minmax(zero_phase_filter_section(cube[il_idx], DT, NARROW_BAND))
        pred = predict_section(model, narrow, device, args.patch_size, args.stride)
        pred_list.append(pred)
        narrow_list.append(narrow)
        target_list.append(target)
        print(f"Predicted inline index {il_idx} ({il_idx + 1 - i0}/{i1 - i0})")

    pred_arr = np.stack(pred_list).astype(np.float32)
    narrow_arr = np.stack(narrow_list).astype(np.float32)
    target_arr = np.stack(target_list).astype(np.float32)

    np.save(PREDICTION_DIR / f"{args.output_prefix}_output.npy", pred_arr)
    np.save(PREDICTION_DIR / f"{args.output_prefix}_narrow_input.npy", narrow_arr)
    np.save(PREDICTION_DIR / f"{args.output_prefix}_wide_reference.npy", target_arr)

    mid = pred_arr.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Low-pass input", "Network output", "F3 reference"]
    for ax, data, title in zip(axes, [narrow_arr[mid], pred_arr[mid], target_arr[mid]], titles):
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time sample")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{args.output_prefix}_section_compare.png", dpi=300)
    plt.close(fig)

    print(f"Saved predictions to {PREDICTION_DIR}")


if __name__ == "__main__":
    main()
