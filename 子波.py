import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq
from scipy.signal.windows import tukey

def generate_ricker_wavelet(frequency, length, dt=0.004):
    """
    生成 Ricker 子波并强制去直流。
    """
    t = np.linspace(-length*dt/2, length*dt/2, length)
    pi_sq = np.pi**2
    wavelet = (1 - 2 * pi_sq * frequency**2 * t**2) * np.exp(-pi_sq * frequency**2 * t**2)
    
    # 关键点 1：归一化前去直流（消除浮点数计算微小偏差）
    wavelet = wavelet - np.mean(wavelet)
    return wavelet / np.max(np.abs(wavelet))

def generate_composite_wavelet(low_freq, high_freq, low_weight, high_weight, length, dt=0.004):
    """
    生成复合子波并强制去直流 + 加窗平滑。
    """
    low_wavelet = generate_ricker_wavelet(low_freq, length, dt)
    high_wavelet = generate_ricker_wavelet(high_freq, length, dt)
    
    # 1. 加权叠加
    composite = low_weight * low_wavelet + high_weight * high_wavelet
    
    # 2. 关键点 2：叠加后再次去直流，确保 0Hz 处严格为 0
    composite = composite - np.mean(composite)
    
    # 3. 关键点 3：加 Tukey 窗（alpha=0.1 表示两端各 5% 渐变），防止截断效应
    composite = composite * tukey(length, alpha=0.1)
    
    # 4. 重新去直流（加窗可能微弱改变均值）并最终归一化
    composite = composite - np.mean(composite)
    return composite / np.max(np.abs(composite))

# --- 设置参数 ---
output_dir = 'wavelets'
os.makedirs(output_dir, exist_ok=True)
wavelet_length = 401
dt = 0.001

# --- 生成子波 ---
narrow_wavelet = generate_ricker_wavelet(frequency=5, length=wavelet_length, dt=dt)
wide_wavelet = generate_composite_wavelet(
    low_freq=5, high_freq=15, 
    low_weight=0.6, high_weight=2, 
    length=wavelet_length, dt=dt
)

# --- 验证均值（DC 分量） ---
print(f"--- DC Offset 检查 (越接近 0 越好) ---")
print(f"Narrow Band Mean: {np.mean(narrow_wavelet):.2e}")
print(f"Wide Band Mean:   {np.mean(wide_wavelet):.2e}")

# 保存数据
# np.save(os.path.join(output_dir, 'narrow_band_wavelet.npy'), narrow_wavelet)
# np.save(os.path.join(output_dir, 'wide_band_wavelet.npy'), wide_wavelet)

# --- 绘图准备 ---
time_axis = np.linspace(-wavelet_length*dt/2, wavelet_length*dt/2, wavelet_length)
freq_axis = fftfreq(wavelet_length, dt)[:wavelet_length//2]
narrow_spectrum = np.abs(fft(narrow_wavelet))[:wavelet_length//2]
wide_spectrum = np.abs(fft(wide_wavelet))[:wavelet_length//2]

# 归一化频谱用于观察
narrow_spectrum_norm = narrow_spectrum / np.max(narrow_spectrum)
wide_spectrum_norm = wide_spectrum / np.max(wide_spectrum)

# --- 绘制频谱对比图 ---
plt.rcParams['font.family'] = 'Times New Roman'

# --- 时域图 ---
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(time_axis * 1000, narrow_wavelet, 'b-', linewidth=2, label='Narrow Band')
ax1.plot(time_axis * 1000, wide_wavelet, 'r-', linewidth=2, label='Wide Band')
ax1.set_xlabel('Time (ms)', fontsize=20)
ax1.set_ylabel('Amplitude', fontsize=20)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, 'wavelet_time_domain.png'),
    dpi=600, bbox_inches='tight'
)
plt.savefig(
    os.path.join(output_dir, 'wavelet_time_domain.eps'),
    dpi=600, bbox_inches='tight', format="eps"
)

# --- 频域图 ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(freq_axis, narrow_spectrum_norm, 'b-', linewidth=2.5, label='Narrow Band')
ax2.plot(freq_axis, wide_spectrum_norm, 'r-', linewidth=2.5, label='Wide Band')
ax2.fill_between(freq_axis, 0, wide_spectrum_norm, alpha=0.1)
ax2.set_xlabel('Frequency (Hz)', fontsize=20)
ax2.set_ylabel('Normalized Amplitude', fontsize=20)
ax2.set_xlim([0, 80])
ax2.set_ylim([0, 1.05])
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, 'wavelet_frequency_domain.png'),
    dpi=600, bbox_inches='tight'
)
plt.savefig(
    os.path.join(output_dir, 'wavelet_frequency_domain.eps'),
    dpi=600, bbox_inches='tight', format="eps"
)
plt.show()


print(f"\n✓ 子波已保存至 '{output_dir}'")
print(f"✓ 0Hz 能量已通过双重去直流操作压制。")