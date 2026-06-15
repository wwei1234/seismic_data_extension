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


def read_segy(data_dir, shotnum=0):
    """
    Read SEG-Y data and organize traces as [inline, time, crossline].

    For the F3 data, use shotnum=651 because inline ranges from 100 to 750.
    If segyio is unavailable, a pure Python reader is used for common IEEE/int formats.
    """
    if segyio is not None:
        with segyio.open(data_dir, "r", ignore_geometry=True) as f:
            source_x = f.attributes(segyio.TraceField.SourceX)[:]
            trace_num = len(source_x)
            shot_num = shotnum if shotnum else len(set(source_x))
            len_shot = trace_num // shot_num
            time = f.trace[0].shape[0]
            print("start read segy data")
            data = np.zeros((shot_num, time, len_shot), dtype=np.float32)
            for j in range(shot_num):
                beg = j * len_shot
                end = (j + 1) * len_shot
                data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[beg:end]]).T
            return data

    if not shotnum:
        raise ImportError("segyio is not installed; provide shotnum for fallback reading.")

    info = _read_segy_basic_info(data_dir)
    trace_num = info["trace_num"]
    shot_num = shotnum
    len_shot = trace_num // shot_num
    trailing = trace_num % shot_num
    if trailing:
        print(f"warning: {trailing} trailing traces are ignored by shotnum={shotnum}")

    print("start read segy data")
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
            il = struct.unpack(">i", header[188:192])[0]
            xl = struct.unpack(">i", header[192:196])[0]
            if il == inline and xl == crossline:
                raw = f.read(info["ns"] * info["bytes_per_sample"])
                return np.frombuffer(raw, dtype=dtype).astype(np.float32)

    raise ValueError(f"Trace not found: inline={inline}, crossline={crossline}")
