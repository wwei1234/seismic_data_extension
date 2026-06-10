import numpy as np
import os
import struct

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
    return ns, sample_format, bytes_per_sample, trace_size, trace_num


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
    读取SEG-Y格式地震数据，按炮集组织为三维数组。

    参数:
        data_dir: SEG-Y文件路径
        shotnum: 炮数。为0时自动从道头中的SourceX字段推断炮数。

    返回:
        data: 形状为 (shot_num, time_samples, traces_per_shot) 的numpy数组
    """
    if segyio is None:
        if not shotnum:
            raise ImportError("segyio is not installed; please provide shotnum for fallback reading.")

        time, sample_format, bytes_per_sample, trace_size, trace_num = _read_segy_basic_info(data_dir)
        shot_num = shotnum
        len_shot = trace_num // shot_num
        if trace_num % shot_num:
            print(f"warning: {trace_num % shot_num} trailing traces are ignored by shotnum={shotnum}")

        print("start read segy data")
        data = np.zeros((shot_num, time, len_shot), dtype=np.float32)
        dtype = np.dtype(_sample_dtype(sample_format))

        with open(data_dir, "rb") as f:
            for j in range(shot_num):
                for i in range(len_shot):
                    trace_idx = j * len_shot + i
                    f.seek(3600 + trace_idx * trace_size + 240)
                    raw = f.read(time * bytes_per_sample)
                    data[j, :, i] = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        return data

    with segyio.open(data_dir, 'r', ignore_geometry=True) as f:
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)  # 总道数
        if shotnum:
            shot_num = shotnum
        else:
            shot_num = len(set(sourceX))  # 从SourceX去重推断炮数
        len_shot = trace_num // shot_num   # 每炮道数
        time = f.trace[0].shape[0]
        print('start read segy data')
        data = np.zeros((shot_num, time, len_shot))
        for j in range(0, shot_num):
            data[j, :, :] = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
        return data
