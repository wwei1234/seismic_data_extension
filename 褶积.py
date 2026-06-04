import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal.windows import tukey

def convolve_with_wavelet(reflectivity, wavelet):
    """
    对二维反射系数数组进行卷积，并进行道处理优化。
    """
    nz, nx = reflectivity.shape
    seismic = np.zeros_like(reflectivity)
    
    # 获取窗函数，用于消除边缘截断引起的低频能量
    # alpha=0.05 表示每端各 2.5% 的长度进行平滑
    win = tukey(nz, alpha=0.05)
    
    for i in range(nx):
        # 1. 提取单道反射系数
        trace_r = reflectivity[:, i]
        
        # 2. 预处理反射系数：去直流（防止地层背景趋势干扰）
        trace_r = trace_r - np.mean(trace_r)
        
        # 3. 执行卷积
        trace_s = np.convolve(trace_r, wavelet, mode='same')
        
        # 4. 后处理地震记录：再次去直流（消除卷积数值残差）
        trace_s = trace_s - np.mean(trace_s)
        
        # 5. 加窗平滑：消除边缘效应
        trace_s = trace_s * win
        
        # 6. 最终道归一化（防止道间能量差异过大，也可根据需要选择全记录归一化）
        if np.max(np.abs(trace_s)) > 0:
            seismic[:, i] = trace_s / np.max(np.abs(trace_s))
        else:
            seismic[:, i] = trace_s
            
    return seismic

# --- 1. 加载数据 ---
# 请确保路径正确
reflectivity = np.load(r"data\reflectivity.npy")
narrow_wavelet = np.load(r"wavelets\narrow_band_wavelet.npy")
wide_wavelet = np.load(r"wavelets\wide_band_wavelet.npy")

# --- 2. 计算合成地震记录 ---
# 此时生成的记录已经经过了去直流和边缘平滑处理
seismic_narrow = convolve_with_wavelet(reflectivity, narrow_wavelet)
seismic_wide = convolve_with_wavelet(reflectivity, wide_wavelet)

print(f"Narrow band shape: {seismic_narrow.shape}")
print(f"Wide band shape: {seismic_wide.shape}")

# --- 3. 保存结果 ---
output_dir = 'seismic_records'
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, 'seismic_narrow_band.npy'), seismic_narrow)
np.save(os.path.join(output_dir, 'seismic_wide_band.npy'), seismic_wide)

# --- 4. 设置绘图参数 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

def plot_seismic(data, title, filename):
    plt.figure(figsize=(10, 8))
    # 使用 vmin/vmax 对称显示，有助于观察直流偏移（如果有偏移，颜色会整体偏灰或偏白）
    v_lim = np.max(np.abs(data)) * 0.8 
    im = plt.imshow(data, cmap='RdGy', aspect='auto', interpolation='bilinear', 
                    vmin=-v_lim, vmax=v_lim)
    
    plt.title(title, fontsize=22, fontweight='bold', pad=15)
    plt.xlabel('Trace Number', fontsize=20, fontweight='bold')
    plt.ylabel('Time Sample', fontsize=20, fontweight='bold')
    
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label('Normalized Amplitude', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200, bbox_inches='tight')
    plt.show()

# --- 5. 绘图展示 ---
plot_seismic(seismic_narrow, 'Narrow Band Seismic Record (15 Hz)', 'seismic_narrow_band.png')
plot_seismic(seismic_wide, 'Wide Band Seismic Record (15+50 Hz)', 'seismic_wide_band.png')

# --- 6. 频谱检查 (验证 0 Hz 是否被压制) ---
sample_trace = seismic_wide[:, seismic_wide.shape[1]//2] # 取中间一道
spectrum = np.abs(np.fft.fft(sample_trace))
print(f"\n频谱检查 (中间道):")
print(f"0 Hz 振幅值: {spectrum[0]:.2e} (应趋于 0)")