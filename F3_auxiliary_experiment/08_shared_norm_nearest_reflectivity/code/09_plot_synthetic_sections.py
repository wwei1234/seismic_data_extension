import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_ROOT / "data"
FIGURE_DIR = EXP_ROOT / "figures"


def load_object(path):
    return np.load(path, allow_pickle=True).item()


def plot_page(inputs, labels, sections, page_id, out_path):
    fig, axes = plt.subplots(len(sections), 2, figsize=(13, 13), constrained_layout=True)
    if len(sections) == 1:
        axes = np.asarray([axes])

    for row, section_meta in enumerate(sections):
        key = section_meta["section_id"]
        x = np.asarray(inputs[key], dtype=np.float32)
        y = np.asarray(labels[key], dtype=np.float32)
        clip = np.nanpercentile(np.abs(np.concatenate([x.ravel(), y.ravel()])), 99.0)
        clip = max(float(clip), 1e-8)

        panels = [
            (x, "Narrow input section"),
            (y, "Wide label section"),
        ]
        for col, (data, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
            ax.set_title(f"{title}: {key}", fontsize=9)
            ax.set_xlabel("Trace")
            ax.set_ylabel("Time sample")

    fig.suptitle(f"Synthetic full-section input/label pairs - page {page_id}", fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--per-page", type=int, default=5)
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    inputs = load_object(DATA_DIR / "synthetic_section_inputs.npy")
    labels = load_object(DATA_DIR / "synthetic_section_labels.npy")
    metadata = load_object(DATA_DIR / "synthetic_metadata.npy")
    all_sections = metadata["sections"]

    indices = np.linspace(0, len(all_sections) - 1, args.num, dtype=int)
    selected = [all_sections[idx] for idx in indices]

    for start in range(0, len(selected), args.per_page):
        page = start // args.per_page + 1
        subset = selected[start:start + args.per_page]
        out_path = FIGURE_DIR / f"09_synthetic_section_pairs_page_{page}.png"
        plot_page(inputs, labels, subset, page, out_path)
        print(f"saved: {out_path}")

    print("selected sections:")
    for idx, section_meta in zip(indices, selected):
        print(
            f"  {idx:02d}  {section_meta['section_id']}  "
            f"well={section_meta['well']}  noise={section_meta['noise_level']}"
        )


if __name__ == "__main__":
    main()
