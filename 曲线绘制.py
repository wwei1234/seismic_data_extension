import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = 'prediction_results'

original = np.load(r"prediction_results\noisy_normalized_input.npy")[:, 150]

net_pred = np.load(r"prediction_results\network_output_normalized.npy")[:, 150]

target = np.load(r"prediction_results\target_normalized.npy")[:, 150]


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
# 绘制频谱对比图
fig, ax = plt.subplots(figsize=(14, 8))

n_samples = len(original)           # 例如 2000
total_time = 0.5                    # 假設總共 2 秒
dt = total_time / (n_samples - 1)   # 或者直接給採樣間隔 dt = 0.001
time = np.linspace(1.5, total_time + 1.5, n_samples, endpoint=True)

# 然後把 plt.plot 改成用 time 當 x
ax.plot(time, original, 'b-', linewidth=3, label='Original', alpha=0.8)
ax.plot(time, target,   'r-', linewidth=3, label='Target',   alpha=0.8)
ax.plot(time, net_pred, 'g-', linewidth=3, label='Network Prediction', alpha=0.8)


ax.set_xlabel('Times(s)', fontsize=24)
ax.set_ylabel('Amplitude', fontsize=24)


# 添加网格和图例
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'single_trace_comparison.png'), 
            dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'single_trace_comparison.eps'), 
            dpi=600, bbox_inches='tight', format="eps")
plt.show()


