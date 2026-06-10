"""
反射系数计算脚本
================
从二维波阻抗数组计算反射系数。
反射系数公式: R[i] = (Z[i+1] - Z[i]) / (Z[i+1] + Z[i])
"""

import numpy as np
import matplotlib.pyplot as plt


# ==================== 参数设置 ====================

# 输入波阻抗文件路径 (.npy)
IMPEDANCE_PATH = r"data\impe.npy"

# 输出文件路径（设为 None 则不保存）
OUTPUT_PATH = r"data\reflectivity.npy"


# ==================== 反射系数计算函数 ====================

def compute_reflectivity(impedance):
    """
    计算二维波阻抗数组的反射系数。

    参数:
        impedance: 形状为 (nz, nx) 的二维波阻抗数组

    返回:
        reflectivity: 形状相同的反射系数数组
    """
    if not isinstance(impedance, np.ndarray) or impedance.ndim != 2:
        raise ValueError("输入必须是二维 numpy 数组")

    reflectivity = np.zeros_like(impedance, dtype=float)

    # 沿深度方向 (axis=0) 计算相邻层间反射系数
    numerator = impedance[1:] - impedance[:-1]
    denominator = impedance[1:] + impedance[:-1] + 1e-10
    r_interface = numerator / denominator

    # 将计算结果放在下层位置（从索引 1 开始）
    reflectivity[1:] = r_interface

    return reflectivity


# ==================== 主程序 ====================

def main():
    print(f"加载波阻抗: {IMPEDANCE_PATH}")
    Z = np.load(IMPEDANCE_PATH)
    print(f"  形状: {Z.shape}")
    print(f"  值范围: [{Z.min():.2f}, {Z.max():.2f}]")

    print("计算反射系数...")
    R = compute_reflectivity(Z)

    if OUTPUT_PATH is not None:
        np.save(OUTPUT_PATH, R)
        print(f"✓ 反射系数已保存至: {OUTPUT_PATH}")
    else:
        print("  (未保存，OUTPUT_PATH 为 None)")

    print(f"反射系数值范围: [{R.min():.6f}, {R.max():.6f}]")

    # 绘图
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(10, 6))
    plt.imshow(R, "seismic", aspect="auto")
    plt.colorbar(label="Reflection Coefficient")
    plt.title("Reflectivity Section", fontsize=18, fontweight="bold")
    plt.xlabel("Trace Number", fontsize=14)
    plt.ylabel("Depth Sample", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
