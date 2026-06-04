import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

data = np.load(r"prediction_results\network_output_normalized.npy")

plt.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(10, 4))
plt.imshow(data, cmap = "gray", aspect='auto', extent=[0, data.shape[1], data.shape[0]*0.001 + 1.5, 1.5])
plt.xlabel('Trace', fontsize=24)
plt.ylabel('Time (s)', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

plt.savefig(
    r"prediction_results\network_output_normalized.png",
    dpi=600, bbox_inches='tight'
)
plt.savefig(
    r"prediction_results\network_output_normalized.eps",
    dpi=600, bbox_inches='tight', format="eps"
)
# plt.show()


vmin, vmax = np.nanmin(data), np.nanmax(data)   # 或你自己指定的范围

# ------------------ 单独绘制 colorbar ------------------
fig_cb = plt.figure(figsize=(1.0, 4.0))          # 窄而高的比例，高度可调
ax_cb = fig_cb.add_axes([0.4, 0.05, 0.3, 0.90])  # [left, bottom, width, height]

norm = Normalize(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap="gray", norm=norm)
sm.set_array([])   # 必须有这一行

cbar = fig_cb.colorbar(sm, cax=ax_cb, orientation='vertical')
cbar.set_label('Amplitude', fontsize=20)
cbar.ax.tick_params(labelsize=18)

plt.rcParams['font.family'] = 'Times New Roman'

fig_cb.savefig(
    r"prediction_results\network_output_normalized_colorbar.png",
    dpi=600, bbox_inches='tight'
)
fig_cb.savefig(
    r"prediction_results\network_output_normalized_colorbar.eps",
    dpi=600, bbox_inches='tight', format="eps"  
)

# 可选：显示看看效果
# plt.show()
plt.close(fig_cb)   # 建议关闭，防止内存占用