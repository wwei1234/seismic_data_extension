import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy import fft


class SeismicBandwidthDataset(Dataset):
    """
    地震数据频带拓宽数据集
    """
    def __init__(self, 
                 narrow_band_path, 
                 wide_band_path, 
                 patch_size=128,
                 noise_levels=None,  # 新增：噪声强度列表
                 use_flip=True,
                 stride=64,
                 time_range=None):  # 新增：时间范围参数
        """
        参数:
            narrow_band_path: 窄频带地震记录npy文件路径
            wide_band_path: 宽频带地震记录npy文件路径
            patch_size: 样本大小 (默认128x128)
            noise_levels: 噪声强度列表，例如 [0.1, 0.15, 0.2]
                         每个值表示噪声标准差 = 数据标准差 * noise_level
                         如果为None，默认使用 [0.15]
            use_flip: 是否使用翻转进行数据增强
            stride: 切片时的步长，小于patch_size可以增加样本数量
            time_range: 时间范围，tuple (start, end)，如 (1500, 2000)
                       如果为None，使用全部数据
        """
        self.patch_size = patch_size
        self.use_flip = use_flip
        
        # 设置噪声强度列表
        if noise_levels is None:
            self.noise_levels = [0.15]
        else:
            self.noise_levels = noise_levels
        
        # 加载数据
        narrow_full = np.load(narrow_band_path)
        wide_full = np.load(wide_band_path)
        
        # 根据时间范围裁剪
        if time_range is not None:
            start, end = time_range
            self.narrow_band = narrow_full[start:end, :]
            self.wide_band = wide_full[start:end, :]
        else:
            self.narrow_band = narrow_full[350:, :]
            self.wide_band = wide_full[350:, :]
        
        # 检查数据形状是否一致
        assert self.narrow_band.shape == self.wide_band.shape, \
            "Narrow band and wide band data must have the same shape!"
        
        # 提取patch的位置列表
        self.patch_positions = self._extract_patch_positions(stride)
        
        # 计算总样本数
        num_patches = len(self.patch_positions)
        flip_multiplier = 2 if use_flip else 1
        num_noise_variants = len(self.noise_levels)
        self.total_samples = num_patches * flip_multiplier * num_noise_variants
        
        print(f"Dataset initialized:")
        print(f"  Original data shape: {self.narrow_band.shape}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Number of patches: {num_patches}")
        print(f"  Flip augmentation: {use_flip}")
        print(f"  Noise levels: {self.noise_levels}")
        print(f"  Number of noise variants: {num_noise_variants}")
        print(f"  Total samples: {self.total_samples}")
     
    def _extract_patch_positions(self, stride):
        """
        提取所有可能的patch位置
        """
        h, w = self.narrow_band.shape
        positions = []
        
        for i in range(0, h - self.patch_size + 1, stride):
            for j in range(0, w - self.patch_size + 1, stride):
                positions.append((i, j))
        
        return positions
    
    def _normalize(self, data):
        """
        归一化数据到[-1, 1]
        """
        data_min = data.min()
        data_max = data.max()
        
        if data_max - data_min < 1e-10:  # 避免除零
            return np.zeros_like(data)
        
        normalized = 2 * (data - data_min) / (data_max - data_min) - 1
        return normalized
    
    def _add_gaussian_noise(self, data, noise_coeff):
        """
        添加高斯随机噪声
        噪声标准差 = 数据标准差 * noise_coeff
        """
        data_std = np.std(data)
        noise_std = data_std * noise_coeff
        noise = np.random.normal(0, noise_std, data.shape)
        
        return data + noise
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        获取一个样本
        """
        # 计算对应的patch索引、翻转状态和噪声变体索引
        num_patches = len(self.patch_positions)
        flip_multiplier = 2 if self.use_flip else 1
        num_noise_variants = len(self.noise_levels)
        
        # 确定噪声变体索引（使用哪个噪声强度）
        noise_idx = idx % num_noise_variants
        idx = idx // num_noise_variants
        
        # 确定是否翻转
        if self.use_flip:
            flip = idx % 2 == 1
            idx = idx // 2
        else:
            flip = False
        
        # 确定patch位置
        patch_idx = idx % num_patches
        i, j = self.patch_positions[patch_idx]
        
        # 提取patch
        narrow_patch = self.narrow_band[i:i+self.patch_size, j:j+self.patch_size].copy()
        wide_patch = self.wide_band[i:i+self.patch_size, j:j+self.patch_size].copy()
        
        # 翻转（如果需要）
        if flip:
            # 随机选择翻转方向
            flip_direction = (noise_idx + patch_idx) % 3
            if flip_direction == 0:  # 水平翻转
                narrow_patch = np.fliplr(narrow_patch)
                wide_patch = np.fliplr(wide_patch)
            elif flip_direction == 1:  # 垂直翻转
                narrow_patch = np.flipud(narrow_patch)
                wide_patch = np.flipud(wide_patch)
            else:  # 水平+垂直翻转
                narrow_patch = np.fliplr(np.flipud(narrow_patch))
                wide_patch = np.fliplr(np.flipud(wide_patch))
        
        # 获取当前样本使用的噪声强度
        current_noise_level = self.noise_levels[noise_idx]
        
        # 先添加噪声，再归一化
        narrow_noisy = self._add_gaussian_noise(narrow_patch, current_noise_level)
        narrow_normalized = self._normalize(narrow_noisy)
        wide_normalized = self._normalize(wide_patch)
        
        # 转换为Tensor，并添加通道维度 [C, H, W]
        sample = torch.from_numpy(narrow_normalized).float().unsqueeze(0)
        label = torch.from_numpy(wide_normalized).float().unsqueeze(0)
        
        return sample, label
    
    def set_noise_levels(self, noise_levels):
        """
        动态设置噪声强度列表
        """
        self.noise_levels = noise_levels
        # 重新计算总样本数
        num_patches = len(self.patch_positions)
        flip_multiplier = 2 if self.use_flip else 1
        num_noise_variants = len(self.noise_levels)
        self.total_samples = num_patches * flip_multiplier * num_noise_variants
        print(f"Noise levels updated to: {noise_levels}")
        print(f"Total samples updated to: {self.total_samples}")


def compute_spectrum(data):
    """
    计算2D数据的平均频谱
    """
    # 对每一列（trace）计算FFT并平均
    spectrum = np.zeros(data.shape[0])
    
    for i in range(data.shape[1]):
        trace = data[:, i]
        fft_trace = fft.fft(trace)
        spectrum += np.abs(fft_trace)
    
    spectrum /= data.shape[1]
    
    # 只取正频率部分
    spectrum = spectrum[:data.shape[0]//2]
    
    return spectrum


# # ======================== 测试代码 ========================
# if __name__ == "__main__":
#     # 创建数据集 - 指定多个噪声强度
#     dataset = SeismicBandwidthDataset(
#         narrow_band_path=r'seismic_records\seismic_narrow_band.npy',
#         wide_band_path=r'seismic_records\seismic_wide_band.npy',
#         patch_size=128,
#         noise_levels=[0.1, 0.15, 0.2],  # 指定3个不同的噪声强度
#         use_flip=True,
#         stride=64
#     )
    
#     print(f"\nTotal samples in dataset: {len(dataset)}")
    
#     # 获取一个样本
#     sample, label = dataset[5000]
    
#     print(f"\nSample shape: {sample.shape}")
#     print(f"Label shape: {label.shape}")
#     print(f"Sample value range: [{sample.min():.3f}, {sample.max():.3f}]")
#     print(f"Label value range: [{label.min():.3f}, {label.max():.3f}]")
    
#     # 转换为numpy用于绘图
#     sample_np = sample.squeeze().numpy()
#     label_np = label.squeeze().numpy()
    
#     # 计算频谱
#     dt = 0.004  # 采样间隔
#     freqs = fft.fftfreq(sample_np.shape[0], dt)
#     freqs = freqs[:sample_np.shape[0]//2]
    
#     spectrum_sample = compute_spectrum(sample_np)
#     spectrum_label = compute_spectrum(label_np)
    
#     # 归一化频谱
#     spectrum_sample_norm = spectrum_sample / np.max(spectrum_sample)
#     spectrum_label_norm = spectrum_label / np.max(spectrum_label)
    
#     # ==================== 绘图：只绘制网络输入、输出及频谱 ====================
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['font.size'] = 14
    
#     fig = plt.figure(figsize=(16, 8))
    
#     # 子图1: 网络输入（样本 - 带噪声的窄频带）
#     ax1 = plt.subplot(2, 2, 1)
#     im1 = ax1.imshow(sample_np, cmap='seismic', aspect='auto', 
#                      vmin=-1, vmax=1, interpolation='bilinear')
#     ax1.set_title('Network Input (Narrow Band + Noise)', fontsize=20, fontweight='bold', pad=15)
#     ax1.set_xlabel('Trace Number', fontsize=18, fontweight='bold')
#     ax1.set_ylabel('Time Sample', fontsize=18, fontweight='bold')
#     cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
#     cbar1.set_label('Amplitude', fontsize=16, fontweight='bold')
#     cbar1.ax.tick_params(labelsize=14)
    
#     # 子图2: 网络输出（标签 - 宽频带）
#     ax2 = plt.subplot(2, 2, 2)
#     im2 = ax2.imshow(label_np, cmap='seismic', aspect='auto', 
#                      vmin=-1, vmax=1, interpolation='bilinear')
#     ax2.set_title('Network Target (Wide Band)', fontsize=20, fontweight='bold', pad=15)
#     ax2.set_xlabel('Trace Number', fontsize=18, fontweight='bold')
#     ax2.set_ylabel('Time Sample', fontsize=18, fontweight='bold')
#     cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
#     cbar2.set_label('Amplitude', fontsize=16, fontweight='bold')
#     cbar2.ax.tick_params(labelsize=14)
    
#     # 子图3: 输入频谱
#     ax3 = plt.subplot(2, 2, 3)
#     ax3.plot(freqs, spectrum_sample_norm, 'b-', linewidth=3, alpha=0.8)
#     ax3.fill_between(freqs, spectrum_sample_norm, alpha=0.3, color='blue')
#     ax3.set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
#     ax3.set_ylabel('Normalized Amplitude', fontsize=18, fontweight='bold')
#     ax3.set_title('Input Spectrum (Narrow Band)', fontsize=20, fontweight='bold', pad=15)
#     ax3.set_xlim([0, 100])
#     ax3.set_ylim([0, 1.05])
#     ax3.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
#     ax3.tick_params(labelsize=14)
    
#     # 标注主频
#     sample_peak_idx = np.argmax(spectrum_sample_norm)
#     sample_peak_freq = freqs[sample_peak_idx]
#     ax3.axvline(x=sample_peak_freq, color='darkblue', linestyle=':', linewidth=2.5, alpha=0.7)
#     ax3.text(sample_peak_freq, 0.95, f'{sample_peak_freq:.1f} Hz', 
#             color='darkblue', fontsize=16, fontweight='bold', ha='center',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
#     # 子图4: 输出频谱
#     ax4 = plt.subplot(2, 2, 4)
#     ax4.plot(freqs, spectrum_label_norm, 'r-', linewidth=3, alpha=0.8)
#     ax4.fill_between(freqs, spectrum_label_norm, alpha=0.3, color='red')
#     ax4.set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
#     ax4.set_ylabel('Normalized Amplitude', fontsize=18, fontweight='bold')
#     ax4.set_title('Target Spectrum (Wide Band)', fontsize=20, fontweight='bold', pad=15)
#     ax4.set_xlim([0, 100])
#     ax4.set_ylim([0, 1.05])
#     ax4.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
#     ax4.tick_params(labelsize=14)
    
#     # 标注主频
#     label_peak_idx = np.argmax(spectrum_label_norm)
#     label_peak_freq = freqs[label_peak_idx]
#     ax4.axvline(x=label_peak_freq, color='darkred', linestyle=':', linewidth=2.5, alpha=0.7)
#     ax4.text(label_peak_freq, 0.95, f'{label_peak_freq:.1f} Hz', 
#             color='darkred', fontsize=16, fontweight='bold', ha='center',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
#     plt.tight_layout()
#     # plt.savefig('network_input_output_visualization.png', dpi=200, bbox_inches='tight')
#     plt.show()
    

    