"""
地震子波生成脚本
================
生成窄频带 Ricker 子波和宽频带复合 Ricker 子波，并绘制时域/频域对比图。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal.windows import tukey


# ==================== 参数设置 ====================

# 输出目录
OUTPUT_DIR = "wavelets"

# 子波参数
WAVELET_LENGTH = 401          # 子波采样点数
DT = 0.001                    # 采样间隔 (s)

# 窄频子波参数
NARROW_FREQ = 5               # 主频 (Hz)

# 宽频复合子波参数
WIDE_LOW_FREQ = 5             # 低频分量主频 (Hz)
WIDE_HIGH_FREQ = 15           # 高频分量主频 (Hz)
WIDE_LOW_WEIGHT = 0.6         # 低频分量权重
WIDE_HIGH_WEIGHT = 2.0        # 高频分量权重

# 绘图参数
FREQ_DISPLAY_MAX = 80         # 频域图频率显示上限 (Hz)


# ==================== 子波生成函数 ====================

def generate_ricker_wavelet(frequency, length, dt=0.004):
    """
    生成 Ricker 子波并强制去直流。

    参数:
        frequency: 主频 (Hz)
        length: 采样点数
        dt: 采样间隔 (s)

    返回:
        wavelet: 归一化后的 Ricker 子波
    """
    t = np.linspace(-length * dt / 2, length * dt / 2, length)
    pi_sq = np.pi ** 2
    wavelet = (1 - 2 * pi_sq * frequency ** 2 * t ** 2) * np.exp(-pi_sq * frequency ** 2 * t ** 2)

    # 去直流 + 归一化
    wavelet = wavelet - np.mean(wavelet)
    return wavelet / np.max(np.abs(wavelet))


def generate_composite_wavelet(low_freq, high_freq, low_weight, high_weight, length, dt=0.004):
    """
    生成复合子波（低频 + 高频 Ricker 加权叠加），去直流 + 加窗平滑。

    参数:
        low_freq: 低频分量主频 (Hz)
        high_freq: 高频分量主频 (Hz)
        low_weight: 低频分量权重
        high_weight: 高频分量权重
        length: 采样点数
        dt: 采样间隔 (s)

    返回:
        composite: 归一化后的复合子波
    """
    low_wavelet = generate_ricker_wavelet(low_freq, length, dt)
    high_wavelet = generate_ricker_wavelet(high_freq, length, dt)

    # 加权叠加
    composite = low_weight * low_wavelet + high_weight * high_wavelet

    # 去直流
    composite = composite - np.mean(composite)

    # Tukey 窗防止截断效应
    composite = composite * tukey(length, alpha=0.1)

    # 重新去直流 + 归一化
    composite = composite - np.mean(composite)
    return composite / np.max(np.abs(composite))


# ==================== 主程序 ====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 生成子波 ---
    narrow_wavelet = generate_ricker_wavelet(frequency=NARROW_FREQ, length=WAVELET_LENGTH, dt=DT)
    wide_wavelet = generate_composite_wavelet(
        low_freq=WIDE_LOW_FREQ, high_freq=WIDE_HIGH_FREQ,
        low_weight=WIDE_LOW_WEIGHT, high_weight=WIDE_HIGH_WEIGHT,
        length=WAVELET_LENGTH, dt=DT,
    )

    # --- DC 检查 ---
    print(f"--- DC Offset 检查 (越接近 0 越好) ---")
    print(f"Narrow Band Mean: {np.mean(narrow_wavelet):.2e}")
    print(f"Wide Band Mean:   {np.mean(wide_wavelet):.2e}")

    # --- 保存 ---
    np.save(os.path.join(OUTPUT_DIR, "narrow_band_wavelet.npy"), narrow_wavelet)
    np.save(os.path.join(OUTPUT_DIR, "wide_band_wavelet.npy"), wide_wavelet)

    # --- 频谱 ---
    time_axis = np.linspace(-WAVELET_LENGTH * DT / 2, WAVELET_LENGTH * DT / 2, WAVELET_LENGTH)
    freq_axis = fftfreq(WAVELET_LENGTH, DT)[:WAVELET_LENGTH // 2]
    narrow_spectrum = np.abs(fft(narrow_wavelet))[:WAVELET_LENGTH // 2]
    wide_spectrum = np.abs(fft(wide_wavelet))[:WAVELET_LENGTH // 2]
    narrow_spectrum_norm = narrow_spectrum / np.max(narrow_spectrum)
    wide_spectrum_norm = wide_spectrum / np.max(wide_spectrum)

    # --- 绘图 ---
    plt.rcParams["font.family"] = "Times New Roman"

    # 时域图
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(time_axis * 1000, narrow_wavelet, "b-", linewidth=2, label="Narrow Band")
    ax1.plot(time_axis * 1000, wide_wavelet, "r-", linewidth=2, label="Wide Band")
    ax1.set_xlabel("Time (ms)", fontsize=20)
    ax1.set_ylabel("Amplitude", fontsize=20)
    ax1.tick_params(axis="x", labelsize=16)
    ax1.tick_params(axis="y", labelsize=16)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ["png", "eps"]:
        plt.savefig(os.path.join(OUTPUT_DIR, f"wavelet_time_domain.{ext}"), dpi=600, bbox_inches="tight")

    # 频域图
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(freq_axis, narrow_spectrum_norm, "b-", linewidth=2.5, label="Narrow Band")
    ax2.plot(freq_axis, wide_spectrum_norm, "r-", linewidth=2.5, label="Wide Band")
    ax2.fill_between(freq_axis, 0, wide_spectrum_norm, alpha=0.1)
    ax2.set_xlabel("Frequency (Hz)", fontsize=20)
    ax2.set_ylabel("Normalized Amplitude", fontsize=20)
    ax2.set_xlim([0, FREQ_DISPLAY_MAX])
    ax2.set_ylim([0, 1.05])
    ax2.tick_params(axis="x", labelsize=16)
    ax2.tick_params(axis="y", labelsize=16)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ["png", "eps"]:
        plt.savefig(os.path.join(OUTPUT_DIR, f"wavelet_frequency_domain.{ext}"), dpi=600, bbox_inches="tight")

    plt.show()
    print(f"\n✓ 子波已保存至 '{OUTPUT_DIR}'")
    print(f"✓ 0Hz 能量已通过双重去直流操作压制。")


if __name__ == "__main__":
    main()
