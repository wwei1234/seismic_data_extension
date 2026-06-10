"""
数据划分标注脚本
================
在地震剖面上用彩色矩形框标注训练集、验证集和测试集的划分区域。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ==================== 参数设置 ====================

# 输入数据路径
DATA_PATH = r"seismic_records\seismic_narrow_band.npy"

# 采样间隔 (s)
DT = 0.001

# 输出目录
OUTPUT_DIR = "seismic_records"

# 区域划分: (起始采样点, 结束采样点, 标签, 颜色)
REGIONS = [
    (350,  990,  "Training Set",   "#FF3131"),
    (990,  1500, "Validation Set", "#39FF14"),
    (1500, 2000, "Test Set",       "#1F51FF"),
    (2000, None, "Training Set",   "#FF3131"),  # None 表示到末尾
]

# 标注样式
REGION_LINEWIDTH = 3             # 矩形框线宽
REGION_GAP_SAMPLES = 10          # 框之间间隙 (采样点)
REGION_X_PADDING = 2             # 水平方向缩进 (道数)

# 图例位置 (相对于右上角的偏移)
LEGEND_X_POS = 1.0
LEGEND_Y_POS = 1.04


# ==================== 主程序 ====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 加载数据 ---
    print(f"加载数据: {DATA_PATH}")
    data = np.load(DATA_PATH)
    num_rows, num_cols = data.shape
    total_time = num_rows * DT
    print(f"  形状: {data.shape}, 总时长: {total_time:.2f} s")

    # --- 绘图 ---
    plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(data, cmap="gray", aspect="auto",
              extent=[0, num_cols, total_time, 0])

    # 添加区域标注
    for start_row, end_row, label, color in REGIONS:
        end_row = end_row if end_row is not None else num_rows

        adj_start = (start_row + REGION_GAP_SAMPLES) * DT
        adj_end = (end_row - REGION_GAP_SAMPLES) * DT
        height = max(0, adj_end - adj_start)
        width = num_cols - (2 * REGION_X_PADDING)

        rect = patches.Rectangle(
            (REGION_X_PADDING, adj_start), width, height,
            linewidth=REGION_LINEWIDTH, edgecolor=color,
            facecolor="none", label=label,
        )
        ax.add_patch(rect)

    # 图例去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="upper right",
        bbox_to_anchor=(LEGEND_X_POS, LEGEND_Y_POS),
        fontsize=16, framealpha=0.9, edgecolor="black",
        borderaxespad=0.,
    )

    ax.set_xlabel("Trace", fontsize=24)
    ax.set_ylabel("Time (s)", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    plt.tight_layout()

    # 保存
    for ext in ["png", "eps"]:
        out_path = os.path.join(OUTPUT_DIR, f"seismic_rows_separated.{ext}")
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"  已保存: {out_path}")

    plt.show()
    plt.close(fig)
    print("✓ 完成")


if __name__ == "__main__":
    main()
