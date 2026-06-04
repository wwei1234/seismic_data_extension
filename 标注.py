import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# 1. 加载数据
data = np.load(r"seismic_records\seismic_narrow_band.npy")

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'

# 2. 创建主图
fig, ax = plt.subplots(figsize=(12, 8))

num_rows = data.shape[0]
num_cols = data.shape[1]
dt = 0.001 
total_time = num_rows * dt

# 绘制地震剖面
ax.imshow(data, cmap="gray", aspect='auto', 
          extent=[0, num_cols, total_time, 0])

# --- 3. 添加区域标注 (带间隙偏移) ---
# 定义区域：(起始行, 结束行, 标签名称, 颜色)
regions = [
    (350, 990, 'Training Set', '#FF3131'), 
    (990, 1500, 'Validation Set', '#39FF14'), 
    (1500, 2000, 'Test Set', '#1F51FF'), 
    (2000, num_rows, 'Training Set', '#FF3131')
]

# 设置偏移参数 (单位：行)
# gap_rows 决定了框与框之间空出的行数
gap_rows = 10 

for start_row, end_row, label, color in regions:
    # 对起始和结束行进行微调，使其不重叠
    # start 往后移一点，end 往前缩一点
    adj_start = (start_row + gap_rows) * dt
    adj_end = (end_row - gap_rows) * dt
    
    # 确保高度不为负数
    height = max(0, adj_end - adj_start)
    
    # 为了美观，水平方向也可以稍微往里缩一点 (可选)
    x_padding = 2 
    width = num_cols - (2 * x_padding)

    rect = patches.Rectangle((x_padding, adj_start), width, height, 
                             linewidth=3, edgecolor=color, facecolor='none', 
                             label=label)
    ax.add_patch(rect)

# 4. 图例处理 (去重)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# 坐标微调控制：
# x 越小越靠左，越大越靠右
# y 越小越靠下，越大越靠上
x_pos = 1  # 0.98 表示靠近右边缘
y_pos = 1.04 # 0.98 表示靠近顶边缘

ax.legend(
    by_label.values(), 
    by_label.keys(), 
    loc='upper right',          # 以图例的右上角作为对齐基准
    bbox_to_anchor=(x_pos, y_pos), # 将该对齐基准放置在坐标系中的位置
    fontsize=16, 
    framealpha=0.9, 
    edgecolor='black',
    borderaxespad=0.           # 移除图例与锚点之间的默认留白，方便精准控制
)

# 5. 设置坐标轴
ax.set_xlabel('Trace', fontsize=24)
ax.set_ylabel('Time (s)', fontsize=24)
ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()

# 6. 保存主图
plt.savefig(r"seismic_records\seismic_rows_separated.png", dpi=600, bbox_inches='tight')
plt.savefig(r"seismic_records\seismic_rows_separated.eps", dpi=600, bbox_inches='tight', format="eps")
plt.show()
# 释放内存
plt.close(fig)
