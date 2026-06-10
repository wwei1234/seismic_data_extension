"""
波阻抗计算脚本
==============
从 Marmousi 弹性模型的密度和 P 波速度 SEG-Y 文件计算波阻抗。
波阻抗 = 密度 × 速度
"""

import numpy as np
import matplotlib.pyplot as plt
from segy_reader import read_segy


# ==================== 参数设置 ====================

# Marmousi 弹性模型 SEG-Y 文件路径
DENSITY_SEGY_PATH = r"elastic-marmousi-model\model\MODEL_DENSITY_1.25m.segy"
VELOCITY_SEGY_PATH = r"elastic-marmousi-model\model\MODEL_P-WAVE_VELOCITY_1.25m.segy"

# 炮数（Marmousi 模型为单炮，设为 1）
SHOT_NUM = 1

# 输出目录和文件名
OUTPUT_DIR = "data"
IMPEDANCE_FILENAME = "impe"
DENSITY_FILENAME = "density"
VELOCITY_FILENAME = "velocity"


# ==================== 主程序 ====================

def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 加载 ---
    print("读取密度模型...")
    density = read_segy(DENSITY_SEGY_PATH, shotnum=SHOT_NUM)[0]
    print(f"  密度形状: {density.shape}, 范围: [{density.min():.1f}, {density.max():.1f}]")

    print("读取速度模型...")
    vel = read_segy(VELOCITY_SEGY_PATH, shotnum=SHOT_NUM)[0]
    print(f"  速度形状: {vel.shape}, 范围: [{vel.min():.1f}, {vel.max():.1f}]")

    # --- 计算波阻抗 ---
    print("计算波阻抗 (密度 × 速度)...")
    impe = density * vel
    print(f"  波阻抗形状: {impe.shape}, 范围: [{impe.min():.1f}, {impe.max():.1f}]")

    # --- 保存 ---
    np.save(os.path.join(OUTPUT_DIR, IMPEDANCE_FILENAME), impe)
    np.save(os.path.join(OUTPUT_DIR, DENSITY_FILENAME), density)
    np.save(os.path.join(OUTPUT_DIR, VELOCITY_FILENAME), vel)
    print(f"✓ 数据已保存至 '{OUTPUT_DIR}/'")

    # --- 绘图 ---
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(10, 8))
    plt.imshow(impe, "seismic", aspect="auto")
    plt.colorbar(label="Impedance")
    plt.title("Acoustic Impedance", fontsize=18, fontweight="bold")
    plt.xlabel("Trace Number", fontsize=14)
    plt.ylabel("Depth Sample", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
