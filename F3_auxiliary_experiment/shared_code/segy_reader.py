import os
import struct

import numpy as np

try:
    import segyio
except ImportError:
    segyio = None


def _read_segy_basic_info(data_dir):
    with open(data_dir, "rb") as f:
        f.seek(3200)
        binary_header = f.read(400)

    dt_us = struct.unpack(">H", binary_header[16:18])[0]
    ns = struct.unpack(">H", binary_header[20:22])[0]
    sample_format = struct.unpack(">H", binary_header[24:26])[0]
    bytes_per_sample = {2: 4, 3: 2, 5: 4, 8: 1}.get(sample_format)
    if bytes_per_sample is None:
        raise ValueError(f"Unsupported SEG-Y sample format: {sample_format}")

    trace_size = 240 + ns * bytes_per_sample
    trace_num = (os.path.getsize(data_dir) - 3600) // trace_size
    return {
        "dt_us": dt_us,
        "ns": ns,
        "sample_format": sample_format,
        "bytes_per_sample": bytes_per_sample,
        "trace_size": trace_size,
        "trace_num": trace_num,
    }


def _sample_dtype(sample_format):
    if sample_format == 2:
        return ">i4"
    if sample_format == 3:
        return ">i2"
    if sample_format == 5:
        return ">f4"
    if sample_format == 8:
        return "i1"
    raise ValueError(f"Unsupported SEG-Y sample format: {sample_format}")


def _header_inline_crossline(header):
    return (
        struct.unpack(">i", header[188:192])[0],
        struct.unpack(">i", header[192:196])[0],
    )


def _scan_trace_geometry(data_dir, info):
    inlines = np.empty(info["trace_num"], dtype=np.int32)
    crosslines = np.empty(info["trace_num"], dtype=np.int32)
    with open(data_dir, "rb") as f:
        for trace_idx in range(info["trace_num"]):
            trace_offset = 3600 + trace_idx * info["trace_size"]
            f.seek(trace_offset)
            header = f.read(240)
            inlines[trace_idx], crosslines[trace_idx] = _header_inline_crossline(header)
    return inlines, crosslines


def _is_valid_geometry(inlines, crosslines):
    return (
        inlines.size > 0
        and np.count_nonzero(inlines) > 0
        and np.count_nonzero(crosslines) > 0
        and np.unique(inlines).size > 1
        and np.unique(crosslines).size > 1
    )


def _read_segy_by_geometry_fallback(data_dir, info, fill_value=0.0, return_geometry=False):
    inlines, crosslines = _scan_trace_geometry(data_dir, info)
    if not _is_valid_geometry(inlines, crosslines):
        return None

    unique_inlines = np.unique(inlines)
    unique_crosslines = np.unique(crosslines)
    inline_to_idx = {int(v): i for i, v in enumerate(unique_inlines)}
    crossline_to_idx = {int(v): i for i, v in enumerate(unique_crosslines)}

    print(
        "start read segy data by inline/crossline headers: "
        f"{unique_inlines.size} inlines, {unique_crosslines.size} crosslines"
    )
    counts = np.array([np.count_nonzero(inlines == il) for il in unique_inlines])
    print(
        "trace count per inline: "
        f"min={int(counts.min())}, max={int(counts.max())}, "
        f"median={float(np.median(counts)):.1f}"
    )

    data = np.full(
        (unique_inlines.size, info["ns"], unique_crosslines.size),
        fill_value,
        dtype=np.float32,
    )
    dtype = np.dtype(_sample_dtype(info["sample_format"]))
    with open(data_dir, "rb") as f:
        for trace_idx, (il, xl) in enumerate(zip(inlines, crosslines)):
            trace_offset = 3600 + trace_idx * info["trace_size"] + 240
            f.seek(trace_offset)
            raw = f.read(info["ns"] * info["bytes_per_sample"])
            data[inline_to_idx[int(il)], :, crossline_to_idx[int(xl)]] = np.frombuffer(
                raw, dtype=dtype
            ).astype(np.float32)

    if return_geometry:
        return data, {
            "inlines": unique_inlines,
            "crosslines": unique_crosslines,
            "trace_counts_per_inline": counts,
        }
    return data


def _read_segy_by_geometry_segyio(data_dir, fill_value=0.0, return_geometry=False):
    with segyio.open(data_dir, "r", ignore_geometry=True) as f:
        inlines = f.attributes(segyio.TraceField.INLINE_3D)[:].astype(np.int32)
        crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:].astype(np.int32)
        if not _is_valid_geometry(inlines, crosslines):
            return None

        unique_inlines = np.unique(inlines)
        unique_crosslines = np.unique(crosslines)
        inline_to_idx = {int(v): i for i, v in enumerate(unique_inlines)}
        crossline_to_idx = {int(v): i for i, v in enumerate(unique_crosslines)}
        time = f.trace[0].shape[0]

        print(
            "start read segy data by inline/crossline headers: "
            f"{unique_inlines.size} inlines, {unique_crosslines.size} crosslines"
        )
        counts = np.array([np.count_nonzero(inlines == il) for il in unique_inlines])
        print(
            "trace count per inline: "
            f"min={int(counts.min())}, max={int(counts.max())}, "
            f"median={float(np.median(counts)):.1f}"
        )

        data = np.full(
            (unique_inlines.size, time, unique_crosslines.size),
            fill_value,
            dtype=np.float32,
        )
        for trace_idx, (il, xl) in enumerate(zip(inlines, crosslines)):
            data[inline_to_idx[int(il)], :, crossline_to_idx[int(xl)]] = np.copy(
                f.trace[trace_idx]
            ).astype(np.float32)

    if return_geometry:
        return data, {
            "inlines": unique_inlines,
            "crosslines": unique_crosslines,
            "trace_counts_per_inline": counts,
        }
    return data


def _read_segy_fixed_chunks(data_dir, shotnum=0):
    info = _read_segy_basic_info(data_dir)
    if not shotnum:
        raise ValueError("shotnum is required for fixed-chunk SEG-Y reading.")

    trace_num = info["trace_num"]
    shot_num = shotnum
    len_shot = trace_num // shot_num
    trailing = trace_num % shot_num
    if trailing:
        print(f"warning: {trailing} trailing traces are ignored by shotnum={shotnum}")

    print("start read segy data by fixed trace chunks")
    data = np.zeros((shot_num, info["ns"], len_shot), dtype=np.float32)
    dtype = np.dtype(_sample_dtype(info["sample_format"]))

    with open(data_dir, "rb") as f:
        for j in range(shot_num):
            for i in range(len_shot):
                trace_idx = j * len_shot + i
                offset = 3600 + trace_idx * info["trace_size"] + 240
                f.seek(offset)
                raw = f.read(info["ns"] * info["bytes_per_sample"])
                data[j, :, i] = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    return data


def read_segy(data_dir, shotnum=0, fill_value=0.0, return_geometry=False, force_fixed=False):
    """
    Read SEG-Y data and organize traces as [inline, time, crossline].

    Default behavior reads inline/crossline trace headers and builds a geometry-aware
    rectangular cube. This avoids mixing traces from adjacent inlines when different
    inlines have different trace counts. Missing inline/crossline cells are filled
    with fill_value.

    Set force_fixed=True only for files without valid inline/crossline headers.
    """
    if force_fixed:
        return _read_segy_fixed_chunks(data_dir, shotnum=shotnum)

    if segyio is not None:
        data = _read_segy_by_geometry_segyio(
            data_dir, fill_value=fill_value, return_geometry=return_geometry
        )
        if data is not None:
            return data

    info = _read_segy_basic_info(data_dir)
    data = _read_segy_by_geometry_fallback(
        data_dir, info, fill_value=fill_value, return_geometry=return_geometry
    )
    if data is not None:
        return data

    return _read_segy_fixed_chunks(data_dir, shotnum=shotnum)


def read_trace_by_inline_crossline(data_dir, inline, crossline):
    """Read one trace by SEG-Y inline/crossline headers."""
    if segyio is not None:
        with segyio.open(data_dir, "r", ignore_geometry=True) as f:
            inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
            xlines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
            matches = np.where((inlines == inline) & (xlines == crossline))[0]
            if len(matches) == 0:
                raise ValueError(f"Trace not found: inline={inline}, crossline={crossline}")
            return np.copy(f.trace[int(matches[0])]).astype(np.float32)

    info = _read_segy_basic_info(data_dir)
    dtype = np.dtype(_sample_dtype(info["sample_format"]))
    with open(data_dir, "rb") as f:
        for trace_idx in range(info["trace_num"]):
            trace_offset = 3600 + trace_idx * info["trace_size"]
            f.seek(trace_offset)
            header = f.read(240)
            il, xl = _header_inline_crossline(header)
            if il == inline and xl == crossline:
                raw = f.read(info["ns"] * info["bytes_per_sample"])
                return np.frombuffer(raw, dtype=dtype).astype(np.float32)

    raise ValueError(f"Trace not found: inline={inline}, crossline={crossline}")
