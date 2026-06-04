import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import os

def normalize_data(data):
    """归一化数据到[-1, 1]"""
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min < 1e-10:
        return np.zeros_like(data), data_min, data_max
    normalized = 2 * (data - data_min) / (data_max - data_min) - 1
    return normalized, data_min, data_max

def compute_amplitude_spectrum(seismic_data, dt=0.004):
    """
    计算地震记录的平均振幅谱。
    - seismic_data: 2D地震记录 (时间 x 道数)
    - dt: 采样间隔（秒）
    返回：频率轴和平均振幅谱
    """
    nz, nx = seismic_data.shape
    
    # 对所有道计算FFT并求平均
    amplitude_spectrum = np.zeros(nz)
    
    for i in range(nx):
        trace = seismic_data[:, i]
        # 计算FFT
        fft_trace = fft.fft(trace)
        # 计算振幅谱
        amplitude_spectrum += np.abs(fft_trace)
    
    # 平均
    amplitude_spectrum /= nx
    
    # 频率轴
    freqs = fft.fftfreq(nz, dt)
    
    # 只取正频率部分
    positive_freqs = freqs[:nz//2]
    positive_amplitude = amplitude_spectrum[:nz//2]
    
    return positive_freqs, positive_amplitude


# 加载地震记录
seismic_narrow = np.load(r"seismic_records\seismic_narrow_band.npy")
seismic_wide = np.load(r"seismic_records\seismic_wide_band.npy")

# seismic_narrow = normalize_data(seismic_narrow)[0]
# 采样间隔
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import os

def normalize_data(data):
    """归一化数据到[-1, 1]"""
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min < 1e-10:
        return np.zeros_like(data), data_min, data_max
    normalized = 2 * (data - data_min) / (data_max - data_min) - 1
    return normalized, data_min, data_max

def compute_amplitude_spectrum(seismic_data, dt=0.004):
    """
    计算地震记录的平均振幅谱。
    - seismic_data: 2D地震记录 (时间 x 道数)
    - dt: 采样间隔（秒）
    返回：频率轴和平均振幅谱
    """
    nz, nx = seismic_data.shape
    
    # 对所有道计算FFT并求平均
    amplitude_spectrum = np.zeros(nz)
    
    for i in range(nx):
        trace = seismic_data[:, i]
        # 计算FFT
        fft_trace = fft.fft(trace)
        # 计算振幅谱
        amplitude_spectrum += np.abs(fft_trace)
    
    # 平均
    amplitude_spectrum /= nx
    
    # 频率轴
    freqs = fft.fftfreq(nz, dt)
    
    # 只取正频率部分
    positive_freqs = freqs[:nz//2]
    positive_amplitude = amplitude_spectrum[:nz//2]
    
    return positive_freqs, positive_amplitude

original = np.load(r"prediction_results\noisy_normalized_input.npy")

net_pred = np.load(r"prediction_results\network_output_normalized.npy")

target = np.load(r"prediction_results\target_normalized.npy")


dt = 0.001  # 4ms

# 计算振幅谱
freqs_original, amp_original = compute_amplitude_spectrum(original, dt)
freqs_target, amp_target = compute_amplitude_spectrum(target, dt)
freqs_net, amp_net = compute_amplitude_spectrum(net_pred, dt)


# 归一化
amp_original_norm = amp_original / np.max(amp_original)
amp_target_norm = amp_target / np.max(amp_target)
amp_net_norm = amp_net / np.max(amp_net)

# 创建输出文件夹
output_dir = 'prediction_results'
os.makedirs(output_dir, exist_ok=True)

# 设置绘图参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

# 绘制频谱对比图
fig, ax = plt.subplots(figsize=(14, 8))

# 绘制频谱曲线
ax.plot(freqs_original, amp_original_norm, 'b-', linewidth=3, 
        label='Original', alpha=0.8)
ax.plot(freqs_target, amp_target_norm, 'r-', linewidth=3, 
        label='Target', alpha=0.8)
ax.plot(freqs_net, amp_net_norm, 'g-', linewidth=3, 
        label='Network Prediction', alpha=0.8)

# 设置坐标轴
ax.set_xlabel('Frequency (Hz)', fontsize=24)
ax.set_ylabel('Normalized Amplitude', fontsize=24)

# 限制频率显示范围以便更清楚地看到差异
ax.set_xlim([0, 200])
ax.set_ylim([0, 1.05])

# 添加网格和图例
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'frequency_spectrum_comparison.png'), 
            dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'frequency_spectrum_comparison.eps'), 
            dpi=600, bbox_inches='tight', format="eps")
plt.show()

