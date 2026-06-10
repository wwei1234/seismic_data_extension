"""
地震合成记录生成脚本
====================
利用反射系数与子波进行褶积，生成窄频带和宽频带合成地震记录。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey


# ==================== 参数设置 ====================

# 输入文件路径
REFLECTIVITY_PATH = r"data\reflectivity.npy"
NARROW_WAVELET_PATH = r"wavelets\narrow_band_wavelet.npy"
WIDE_WAVELET_PATH = r"wavelets\wide_band_wavelet.npy"

# 输出目录
OUTPUT_DIR = "seismic_records"

# 道处理参数
TUKEY_ALPHA = 0.05             # Tukey 窗 alpha 值（边缘平滑强度）

# 绘图参数
PLOT_CLIP_RATIO = 0.8          # 显示裁剪比例（相对于最大绝对值）


# ==================== 褶积函数 ====================

def convolve_with_wavelet(reflectivity, wavelet, tukey_alpha=0.05):
    """
    对二维反射系数数组进行逐道褶积，并进行去直流和加窗处理。

    参数:
        reflectivity: 二维反射系数 (nz, nx)
        wavelet: 一维子波
        tukey_alpha: Tukey 窗 alpha 值

    返回:
        seismic: 合成地震记录 (nz, nx)
    """
    nz, nx = reflectivity.shape
    seismic = np.zeros_like(reflectivity)

    # Tukey 窗用于消除边缘截断引起的低频能量
    win = tukey(nz, alpha=tukey_alpha)

    for i in range(nx):
        # 提取单道反射系数，去直流
        trace_r = reflectivity[:, i]
        trace_r = trace_r - np.mean(trace_r)

        # 褶积
        trace_s = np.convolve(trace_r, wavelet, mode="same")

        # 去直流 + 加窗
        trace_s = trace_s - np.mean(trace_s)
        trace_s = trace_s * win

        # 道归一化
        if np.max(np.abs(trace_s)) > 0:
            seismic[:, i] = trace_s / np.max(np.abs(trace_s))
        else:
            seismic[:, i] = trace_s

    return seismic


# ==================== 绘图函数 ====================

def plot_seismic(data, title, filename, output_dir, clip_ratio=0.8):
    """绘制并保存地震剖面图。"""
    os.makedirs(output_dir, exist_ok=True)
    v_lim = np.max(np.abs(data)) * clip_ratio

    plt.figure(figsize=(10, 8))
    im = plt.imshow(data, cmap="RdGy", aspect="auto", interpolation="bilinear",
                    vmin=-v_lim, vmax=v_lim)
    plt.title(title, fontsize=22, fontweight="bold", pad=15)
    plt.xlabel("Trace Number", fontsize=20, fontweight="bold")
    plt.ylabel("Time Sample", fontsize=20, fontweight="bold")
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Normalized Amplitude", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200, bbox_inches="tight")


# ==================== 主程序 ====================

def main():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16

    # --- 加载数据 ---
    print("加载数据...")
    reflectivity = np.load(REFLECTIVITY_PATH)
    narrow_wavelet = np.load(NARROW_WAVELET_PATH)
    wide_wavelet = np.load(WIDE_WAVELET_PATH)
    print(f"  反射系数: {reflectivity.shape}")
    print(f"  窄频子波: {narrow_wavelet.shape}")
    print(f"  宽频子波: {wide_wavelet.shape}")

    # --- 褶积 ---
    print("计算窄频带合成记录...")
    seismic_narrow = convolve_with_wavelet(reflectivity, narrow_wavelet, TUKEY_ALPHA)
    print("计算宽频带合成记录...")
    seismic_wide = convolve_with_wavelet(reflectivity, wide_wavelet, TUKEY_ALPHA)

    print(f"Narrow band shape: {seismic_narrow.shape}")
    print(f"Wide band shape: {seismic_wide.shape}")

    # --- 保存 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "seismic_narrow_band.npy"), seismic_narrow)
    np.save(os.path.join(OUTPUT_DIR, "seismic_wide_band.npy"), seismic_wide)
    print(f"✓ 合成记录已保存至 '{OUTPUT_DIR}'")

    # --- 频谱检查 ---
    sample_trace = seismic_wide[:, seismic_wide.shape[1] // 2]
    spectrum = np.abs(np.fft.fft(sample_trace))
    print(f"\n频谱检查 (中间道):")
    print(f"0 Hz 振幅值: {spectrum[0]:.2e} (应趋于 0)")

    # --- 绘图 ---
    print("绘制地震剖面...")
    plot_seismic(seismic_narrow, "Narrow Band Seismic Record (5 Hz)",
                 "seismic_narrow_band.png", OUTPUT_DIR, PLOT_CLIP_RATIO)
    plot_seismic(seismic_wide, "Wide Band Seismic Record (5+15 Hz)",
                 "seismic_wide_band.png", OUTPUT_DIR, PLOT_CLIP_RATIO)
    plt.show()
    print("✓ 完成")


if __name__ == "__main__":
    main()
