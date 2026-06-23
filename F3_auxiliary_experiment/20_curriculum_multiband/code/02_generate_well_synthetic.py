"""Generate grouped well-constrained synthetic residual samples for experiment 20."""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "shared_code"))
sys.path.append(str(CODE_DIR))

from config import (  # noqa: E402
    F3_PATCH_DIR,
    PATCH_SIZE,
    PATCH_STRIDE,
    RANDOM_SEED,
    SOURCE_DATA_DIR,
    SYNTHETIC_NOISE_LEVELS,
    SYNTHETIC_DIR,
    ensure_dirs,
)
from synthetic_samples import (  # noqa: E402
    add_structural_perturbation,
    build_bspline_wavelet,
    build_linear_well_section,
    make_normalized_sample,
    normalize_wavelet,
)


BSPLINE_BANDS = (
    (3.0, 6.0, 55.0, 70.0),
    (3.0, 6.0, 65.0, 80.0),
    (3.0, 6.0, 75.0, 90.0),
)


def patch_starts(length, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def convolve_section(reflectivity, wavelet):
    output = np.zeros_like(reflectivity, dtype=np.float32)
    for trace_index in range(reflectivity.shape[1]):
        output[:, trace_index] = fftconvolve(
            reflectivity[:, trace_index],
            wavelet,
            mode="same",
        )
    return output


def f3_envelope(section_shape, f3_patch, rng):
    time_profile = np.mean(np.abs(f3_patch), axis=1)
    space_profile = np.mean(np.abs(f3_patch), axis=0)
    time_profile = np.interp(
        np.linspace(0, len(time_profile) - 1, section_shape[0]),
        np.arange(len(time_profile)),
        time_profile,
    )
    space_profile = np.interp(
        np.linspace(0, len(space_profile) - 1, section_shape[1]),
        np.arange(len(space_profile)),
        space_profile,
    )
    envelope = np.sqrt(
        np.maximum(time_profile[:, None], 1e-8)
        * np.maximum(space_profile[None, :], 1e-8)
    )
    envelope /= np.mean(envelope) + 1e-8
    envelope = np.clip(envelope, 0.5, 1.5)
    envelope *= float(rng.uniform(0.9, 1.1))
    return envelope.astype(np.float32)


def build_wavelet_bank(well_names):
    estimated = np.load(SOURCE_DATA_DIR / "well_estimated_wavelets.npy")
    bank = []
    for index, name in enumerate(well_names):
        bank.append({
            "name": f"estimated_{name}",
            "source": "estimated",
            "wavelet": normalize_wavelet(estimated[index]),
        })
    for band in BSPLINE_BANDS:
        bank.append({
            "name": f"bspline_{int(band[2])}_{int(band[3])}",
            "source": "bspline",
            "wavelet": build_bspline_wavelet(estimated.shape[1], band),
        })
    return bank


def build_patch_plan(target_patches, patches_per_section, seed):
    profile_count = int(np.ceil(target_patches / patches_per_section))
    profile_ids = [f"section_{index:04d}" for index in range(profile_count)]
    rng = np.random.default_rng(seed)
    shuffled = profile_ids.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(profile_count * 0.2)))
    val_ids = set(shuffled[:val_count])
    plan = []
    for profile_id in profile_ids:
        for patch_index in range(patches_per_section):
            if len(plan) >= target_patches:
                break
            plan.append({
                "section_id": profile_id,
                "patch_index": patch_index,
                "split": "val" if profile_id in val_ids else "train",
            })
    return plan


def allocate_split(split, count):
    shape = (count, PATCH_SIZE, PATCH_SIZE)
    arrays = {}
    for name in ("inputs", "labels", "clean_narrow", "wide"):
        arrays[name] = np.lib.format.open_memmap(
            SYNTHETIC_DIR / f"{split}_{name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-patches", "--max-patches", type=int, default=2600)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    rng = np.random.default_rng(args.seed)

    reflectivities = np.load(
        SOURCE_DATA_DIR / "well_reflectivities.npy",
        allow_pickle=True,
    ).item()
    matches = np.load(
        SOURCE_DATA_DIR / "well_trace_matches.npy",
        allow_pickle=True,
    ).item()
    well_names = sorted(reflectivities)
    combos = [
        combo
        for count in (2, 3, 4)
        for combo in combinations(well_names, count)
    ]
    wavelet_bank = build_wavelet_bank(well_names)
    f3_patches = np.load(F3_PATCH_DIR / "train_clean_narrow.npy", mmap_mode="r")

    time_starts = patch_starts(min(len(value) for value in reflectivities.values()))
    space_starts = patch_starts(951)
    locations = [(t0, x0) for t0 in time_starts for x0 in space_starts]
    plan = build_patch_plan(args.target_patches, len(locations), args.seed)
    split_counts = {
        split: sum(row["split"] == split for row in plan)
        for split in ("train", "val")
    }
    outputs = {
        split: allocate_split(split, split_counts[split])
        for split in ("train", "val")
    }
    write_indices = {"train": 0, "val": 0}
    metadata = {"train": [], "val": []}

    profile_rows = {}
    for row in plan:
        profile_rows.setdefault(row["section_id"], []).append(row)

    example = None
    for profile_number, (section_id, rows) in enumerate(profile_rows.items()):
        combo = combos[profile_number % len(combos)]
        wavelet_item = wavelet_bank[profile_number % len(wavelet_bank)]
        noise_level = float(
            SYNTHETIC_NOISE_LEVELS[profile_number % len(SYNTHETIC_NOISE_LEVELS)]
        )
        reflectivity, ordered_wells, positions = build_linear_well_section(
            reflectivities,
            combo,
            matches,
        )
        reflectivity = add_structural_perturbation(reflectivity, rng)
        clean_wide = convolve_section(reflectivity, wavelet_item["wavelet"])
        envelope_patch = np.asarray(
            f3_patches[int(rng.integers(0, len(f3_patches)))],
            dtype=np.float32,
        )
        clean_wide *= f3_envelope(clean_wide.shape, envelope_patch, rng)
        sample = make_normalized_sample(clean_wide, noise_level, rng)

        for row in rows:
            t0, x0 = locations[row["patch_index"]]
            split = row["split"]
            output_index = write_indices[split]
            slices = np.s_[t0:t0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            outputs[split]["inputs"][output_index] = sample.input_norm[slices]
            outputs[split]["labels"][output_index] = sample.label_norm[slices]
            outputs[split]["clean_narrow"][output_index] = (
                sample.clean_narrow_norm[slices]
            )
            outputs[split]["wide"][output_index] = sample.wide_norm[slices]
            metadata[split].append({
                "section_id": section_id,
                "patch_index": int(row["patch_index"]),
                "time_start": int(t0),
                "spatial_start": int(x0),
                "wells": list(combo),
                "ordered_wells": ordered_wells,
                "well_positions": [float(value) for value in positions],
                "wavelet_name": wavelet_item["name"],
                "wavelet_source": wavelet_item["source"],
                "noise_level": noise_level,
                "scale": sample.scale,
                "scale_source": sample.scale_source,
                "label_type": "clean_wide_minus_clean_narrow",
                "uses_f3_wide_target": False,
            })
            write_indices[split] += 1
            if example is None:
                example = (
                    sample.clean_narrow_norm[slices].copy(),
                    sample.wide_norm[slices].copy(),
                    sample.label_norm[slices].copy(),
                )

    for split in ("train", "val"):
        for array in outputs[split].values():
            array.flush()
        np.save(
            SYNTHETIC_DIR / f"{split}_metadata.npy",
            np.asarray(metadata[split], dtype=object),
        )

    train_ids = {row["section_id"] for row in metadata["train"]}
    val_ids = {row["section_id"] for row in metadata["val"]}
    all_metadata = metadata["train"] + metadata["val"]
    manifest = {
        "experiment": 20,
        "num_train": split_counts["train"],
        "num_val": split_counts["val"],
        "target_patches": args.target_patches,
        "patch_shape": [PATCH_SIZE, PATCH_SIZE],
        "patch_stride": PATCH_STRIDE,
        "normalization": "per_section_p99_abs_clean_narrow",
        "label_type": "clean_wide_minus_clean_narrow",
        "uses_f3_wide_target": False,
        "section_groups_disjoint": train_ids.isdisjoint(val_ids),
        "wavelet_sources": sorted({
            row["wavelet_source"] for row in all_metadata
        }),
        "noise_levels": sorted({
            row["noise_level"] for row in all_metadata
        }),
        "smoke": bool(args.smoke),
    }
    (SYNTHETIC_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if example is not None:
        clip = max(
            float(np.percentile(np.abs(np.concatenate([x.ravel() for x in example])), 99)),
            1e-8,
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for axis, data, title in zip(
            axes,
            example,
            ("Low-pass input", "Wide label", "Residual label"),
        ):
            axis.imshow(data, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
            axis.set_title(title)
            axis.set_xlabel("Trace")
        axes[0].set_ylabel("Time sample")
        fig.tight_layout()
        fig.savefig(
            SYNTHETIC_DIR.parent.parent / "figures" / "训练样本" / "synthetic_example.png",
            dpi=250,
        )
        plt.close(fig)

    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
