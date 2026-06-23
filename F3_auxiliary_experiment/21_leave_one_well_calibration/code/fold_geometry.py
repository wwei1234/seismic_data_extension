from copy import deepcopy

from config import (
    CALIBRATION_CROSSLINE_RADIUS,
    CALIBRATION_INLINE_RADIUS,
    FOLDS,
    HELDOUT_CROSSLINE_GUARD,
    HELDOUT_INLINE_GUARD,
    WELLS,
)


def _bounds(well, inline_radius, crossline_radius):
    return {
        "inline_min": int(well["inline"] - inline_radius),
        "inline_max": int(well["inline"] + inline_radius),
        "crossline_min": int(well["crossline"] - crossline_radius),
        "crossline_max": int(well["crossline"] + crossline_radius),
    }


def calibration_window(well):
    return _bounds(
        well,
        CALIBRATION_INLINE_RADIUS,
        CALIBRATION_CROSSLINE_RADIUS,
    )


def heldout_guard(well):
    return _bounds(well, HELDOUT_INLINE_GUARD, HELDOUT_CROSSLINE_GUARD)


def regions_overlap(first, second):
    return not (
        first["inline_max"] < second["inline_min"]
        or first["inline_min"] > second["inline_max"]
        or first["crossline_max"] < second["crossline_min"]
        or first["crossline_min"] > second["crossline_max"]
    )


def plan_fold(fold_name):
    fold = deepcopy(FOLDS[fold_name])
    heldout_name = fold["heldout_well"]
    heldout = WELLS[heldout_name]
    regions = []
    for name in fold["calibration_wells"]:
        well = WELLS[name]
        region = {
            **calibration_window(well),
            "well": name,
            "well_inline": well["inline"],
            "well_crossline": well["crossline"],
            "train_inline_values": list(range(well["inline"] - 8, well["inline"] + 7)),
            "val_inline_values": [well["inline"] + 7, well["inline"] + 8],
        }
        regions.append(region)
    manifest = {
        "fold": fold_name,
        "heldout_well": heldout_name,
        "heldout_inline": heldout["inline"],
        "heldout_crossline": heldout["crossline"],
        "calibration_wells": list(fold["calibration_wells"]),
        "heldout_guard": heldout_guard(heldout),
        "wide_sample_regions": regions,
        "uses_heldout_well_wide_target": False,
    }
    validate_fold_manifest(manifest)
    return manifest


def validate_fold_manifest(manifest):
    heldout = manifest["heldout_well"]
    if heldout in manifest["calibration_wells"]:
        raise ValueError("The held-out well cannot be a calibration well.")
    guard = manifest["heldout_guard"]
    for region in manifest["wide_sample_regions"]:
        if regions_overlap(region, guard):
            raise ValueError("A calibration region intersects the held-out guard.")
        if set(region["train_inline_values"]) & set(region["val_inline_values"]):
            raise ValueError("Train and validation planes overlap.")
        expected = calibration_window(WELLS[region["well"]])
        for key, value in expected.items():
            if region[key] != value:
                raise ValueError("Calibration region exceeds its fixed window.")
    if manifest.get("uses_heldout_well_wide_target") is not False:
        raise ValueError("Manifest must forbid held-out wide targets.")
    return manifest
