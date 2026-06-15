from pathlib import Path

import numpy as np

from config import DT, N_TIME, SEGY_PATH, WELL_POSITIONS, WELL_RAW_DIR
from segy_reader import read_trace_by_inline_crossline


NULL_VALUE = -999.25


def read_las_curves(path):
    path = Path(path)
    curves = []
    null_value = NULL_VALUE
    data_lines = []
    in_curve = False
    in_data = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()

            if upper.startswith("NULL"):
                parts = line.replace(".", " .", 1).split()
                for token in parts:
                    try:
                        null_value = float(token)
                        break
                    except ValueError:
                        continue

            if upper.startswith("~CURVE"):
                in_curve = True
                in_data = False
                continue
            if upper.startswith("~A"):
                in_data = True
                in_curve = False
                continue
            if upper.startswith("~"):
                in_curve = False
                continue

            if in_curve:
                mnemonic = line.split(".")[0].strip()
                if mnemonic:
                    curves.append(mnemonic)
            elif in_data:
                data_lines.append(line)

    data = np.array([[float(x) for x in line.split()] for line in data_lines], dtype=np.float64)
    if len(curves) != data.shape[1]:
        curves = ["DEPTH", "RHOB", "DT", "GR", "AI", "AI_rel", "PHIE"][:data.shape[1]]

    out = {name: data[:, idx] for idx, name in enumerate(curves)}
    for key, values in out.items():
        values[np.isclose(values, null_value)] = np.nan
        out[key] = values
    return out


def read_td_curve(path):
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    depth = arr[:, 0]
    time = arr[:, 1]
    order = np.argsort(depth)
    return depth[order], time[order]


def fill_nan_linear(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough valid samples to interpolate.")
    return np.interp(x, x[mask], y[mask])


def compute_ai(curves):
    if "AI" in curves and np.count_nonzero(np.isfinite(curves["AI"])) > 10:
        return fill_nan_linear(curves["DEPTH"], curves["AI"])

    rhob = fill_nan_linear(curves["DEPTH"], curves["RHOB"])
    dt_us_per_m = fill_nan_linear(curves["DEPTH"], curves["DT"])
    vp = 1_000_000.0 / dt_us_per_m
    return rhob * vp


def reflectivity_from_ai(ai):
    ai = np.asarray(ai, dtype=np.float64)
    r = np.zeros_like(ai)
    r[1:] = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1] + 1e-12)
    r[~np.isfinite(r)] = 0.0
    return r


def depth_reflectivity_to_time(depth, reflectivity, td_depth, td_time, dt=DT, n_time=N_TIME):
    valid = np.isfinite(depth) & np.isfinite(reflectivity)
    depth = depth[valid]
    reflectivity = reflectivity[valid]
    time = np.interp(depth, td_depth, td_time, left=np.nan, right=np.nan)

    r_time = np.zeros(n_time, dtype=np.float32)
    counts = np.zeros(n_time, dtype=np.float32)
    sample_index = np.rint(time / dt).astype(np.float64)
    valid = np.isfinite(sample_index) & (sample_index >= 0) & (sample_index < n_time)
    for idx, amp in zip(sample_index[valid].astype(int), reflectivity[valid]):
        r_time[idx] += amp
        counts[idx] += 1.0

    nz = counts > 0
    r_time[nz] /= counts[nz]
    r_time -= np.mean(r_time)
    return r_time


def load_well_reflectivity(well_name, dt=DT, n_time=N_TIME):
    las_path = WELL_RAW_DIR / "Lasfiles" / f"{well_name}_logs.las"
    td_path = WELL_RAW_DIR / "Checkshot" / f"{well_name}_TD.txt"
    curves = read_las_curves(las_path)
    td_depth, td_time = read_td_curve(td_path)
    ai = compute_ai(curves)
    r_depth = reflectivity_from_ai(ai)
    r_time = depth_reflectivity_to_time(
        curves["DEPTH"], r_depth, td_depth, td_time, dt=dt, n_time=n_time
    )
    return r_time, curves, (td_depth, td_time)


def load_all_well_reflectivities():
    out = {}
    for well_name in WELL_POSITIONS:
        out[well_name] = load_well_reflectivity(well_name)[0]
    return out


def load_well_trace(well_name):
    pos = WELL_POSITIONS[well_name]
    trace = read_trace_by_inline_crossline(SEGY_PATH, pos["inline"], pos["crossline"])
    return trace[:N_TIME].astype(np.float32)
