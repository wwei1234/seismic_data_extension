"""
地震剖面绘制脚本
================
绘制二维地震剖面图及独立的 colorbar。
支持 .npy 和 .sgy/.segy 格式输入。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ==================== 参数设置 ====================

# 输入数据路径（.npy / .sgy / .segy）
DATA_PATH = r"F3_Demo_2023\Rawdata\Seismic_data.sgy"
# DATA_PATH = r"prediction_results\network_output_normalized.npy"

# SEG-Y 选项
PRESTACK = True              # 是否按叠前炮集方式读取
SHOTNUM = 651                   # 炮数（prestack 模式，0 = 自动推断）
SELECT_SHOT = 0               # 叠前数据选取第几炮用于绘图

# 数据裁剪: 时间范围 (start, end)，None 表示不裁剪
TIME_RANGE = None             # 例如 (500, 1500)，设为 None 使用全部
TRACE_RANGE = None            # 例如 (100, 500)，设为 None 使用全部

# 采样间隔 (s) —— 仅用于 extent 坐标轴标注
DT = 0.004

# 起始时间 (s)
TIME_OFFSET = 262

# 输出目录
OUTPUT_DIR = "F3_section"

# 输出文件名前缀
OUTPUT_BASENAME = "seismic_section_inline_262"

# 绘图参数
CMAP = "seismic"
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 4

# Colorbar 参数
COLORBAR_WIDTH = 1.0
COLORBAR_HEIGHT = 4.0
COLORBAR_LABEL = "Amplitude"


# ==================== 数据加载 ====================

def load_section(path, prestack=False, shotnum=0):
    """
    自动识别格式并加载二维地震剖面。

    参数:
        path: 文件路径（.npy / .sgy / .segy）
        prestack: 是否按叠前炮集方式读取
        shotnum: 炮数（prestack 模式，0 = 自动推断）

    返回:
        data: 二维数组 (time_samples, traces)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        data = np.load(path)
    elif ext in (".sgy", ".segy"):
        if prestack:
            from segy_reader import read_segy
            data_3d = read_segy(path, shotnum=shotnum)
            print(f"  叠前数据: 共 {data_3d.shape[0]} 炮")
            data = data_3d[SELECT_SHOT, :, :]
            print(f"  选取第 {SELECT_SHOT} 炮用于绘图")
        else:
            import segyio
            with segyio.open(path, "r", ignore_geometry=True) as f:
                data = np.asarray([np.copy(trace) for trace in f.trace]).T
    else:
        raise ValueError(f"不支持的文件格式: {ext}。仅支持 .npy / .sgy / .segy")

    if data.ndim != 2:
        raise ValueError(f"期望二维剖面数据，实际维度: {data.ndim}")

    return data.astype(np.float64)


def crop_section(data, time_range=None, trace_range=None):
    """裁剪剖面数据。"""
    if time_range is not None:
        t0, t1 = time_range
        data = data[t0:t1, :]
        print(f"  时间裁剪: [{t0}, {t1}) → 形状 {data.shape}")
    if trace_range is not None:
        x0, x1 = trace_range
        data = data[:, x0:x1]
        print(f"  道数裁剪: [{x0}, {x1}) → 形状 {data.shape}")
    return data


# ==================== 主程序 ====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 加载数据 ---
    print(f"加载数据: {DATA_PATH}")
    data = load_section(DATA_PATH, prestack=PRESTACK, shotnum=SHOTNUM)
    print(f"  原始形状: {data.shape}")
    print(f"  值范围: [{data.min():.4f}, {data.max():.4f}]")

    # --- 裁剪 ---
    data = crop_section(data, TIME_RANGE, TRACE_RANGE)

    n_samples, n_traces = data.shape
    total_time = n_samples * DT
    time_extent = [0, n_traces, total_time + TIME_OFFSET, TIME_OFFSET]

    # --- 绘制剖面 ---
    plt.rcParams["font.family"] = "Times New Roman"

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    plt.imshow(data, cmap=CMAP, aspect="auto", extent=time_extent)
    plt.xlabel("Trace", fontsize=24)
    plt.ylabel("Time (s)", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    for ext in ["png", "eps"]:
        out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.{ext}")
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"  已保存: {out_path}")

    plt.close(fig)

    # --- 单独绘制 colorbar ---
    vmin, vmax = np.nanmin(data), np.nanmax(data)

    fig_cb = plt.figure(figsize=(COLORBAR_WIDTH, COLORBAR_HEIGHT))
    ax_cb = fig_cb.add_axes([0.4, 0.05, 0.3, 0.90])

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])

    cbar = fig_cb.colorbar(sm, cax=ax_cb, orientation="vertical")
    cbar.set_label(COLORBAR_LABEL, fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    for ext in ["png", "eps"]:
        out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_colorbar.{ext}")
        fig_cb.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"  已保存: {out_path}")

    plt.close(fig_cb)
    print("✓ 完成")


if __name__ == "__main__":
    main()
