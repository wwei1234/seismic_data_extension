"""
单道曲线对比绘制脚本
====================
提取地震剖面中的单道数据，对比原始、网络输入、网络输出和标签的波形。
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ==================== 参数设置 ====================

# 数据目录
DATA_DIR = r"prediction_results"

# 输入文件
ORIGINAL_FILE = "noisy_normalized_input.npy"    # 网络输入（带噪声窄频带）
NETWORK_OUTPUT_FILE = "network_output_normalized.npy"  # 网络预测输出
TARGET_FILE = "target_normalized.npy"           # 真实标签（宽频带）

# 道号（取第几道进行对比）
TRACE_INDEX = 150

# 时间轴参数
TOTAL_TIME = 0.5            # 总时长 (s)
TIME_OFFSET = 1.5           # 起始时间 (s)

# 输出目录
OUTPUT_DIR = "prediction_results"

# 曲线颜色与标签
CURVE_CONFIG = [
    ("b-", "Original Input", 3),
    ("r-", "Target (Wide Band)", 3),
    ("g-", "Network Prediction", 3),
]


# ==================== 主程序 ====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 加载数据 ---
    original = np.load(os.path.join(DATA_DIR, ORIGINAL_FILE))[:, TRACE_INDEX]
    net_pred = np.load(os.path.join(DATA_DIR, NETWORK_OUTPUT_FILE))[:, TRACE_INDEX]
    target = np.load(os.path.join(DATA_DIR, TARGET_FILE))[:, TRACE_INDEX]

    n_samples = len(original)
    dt = TOTAL_TIME / (n_samples - 1)
    time = np.linspace(TIME_OFFSET, TOTAL_TIME + TIME_OFFSET, n_samples, endpoint=True)

    print(f"单道对比: 道号 {TRACE_INDEX}, 采样点数 {n_samples}")
    print(f"  输入范围: [{original.min():.4f}, {original.max():.4f}]")
    print(f"  输出范围: [{net_pred.min():.4f}, {net_pred.max():.4f}]")
    print(f"  标签范围: [{target.min():.4f}, {target.max():.4f}]")

    # --- 绘图 ---
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 20

    fig, ax = plt.subplots(figsize=(14, 8))

    data_list = [original, target, net_pred]
    for (style, label, width), data in zip(CURVE_CONFIG, data_list):
        ax.plot(time, data, style, linewidth=width, label=label, alpha=0.8)

    ax.set_xlabel("Time (s)", fontsize=24)
    ax.set_ylabel("Amplitude", fontsize=24)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=1)
    ax.legend(loc="upper right", frameon=True, shadow=True, fancybox=True)

    plt.tight_layout()
    for ext in ["png", "eps"]:
        out_path = os.path.join(OUTPUT_DIR, f"single_trace_comparison.{ext}")
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"  已保存: {out_path}")

    plt.show()
    print("✓ 完成")


if __name__ == "__main__":
    main()
