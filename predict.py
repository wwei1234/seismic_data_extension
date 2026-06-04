import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import os
from U_Net_CBAM import UNet


def add_gaussian_noise(data, noise_coeff=0.15):
    """添加高斯随机噪声"""
    data_std = np.std(data)
    noise_std = data_std * noise_coeff
    noise = np.random.normal(0, noise_std, data.shape)
    return data + noise


def normalize_data(data):
    """归一化数据到[-1, 1]"""
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min < 1e-10:
        return np.zeros_like(data), data_min, data_max
    normalized = 2 * (data - data_min) / (data_max - data_min) - 1
    return normalized, data_min, data_max


def compute_spectrum(data, dt=0.004):
    """计算2D数据的平均频谱"""
    spectrum = np.zeros(data.shape[0])
    for i in range(data.shape[1]):
        trace = data[:, i]
        fft_trace = fft.fft(trace)
        spectrum += np.abs(fft_trace)
    spectrum /= data.shape[1]
    freqs = fft.fftfreq(data.shape[0], dt)
    positive_freqs = freqs[:data.shape[0]//2]
    positive_spectrum = spectrum[:data.shape[0]//2]
    return positive_freqs, positive_spectrum


def load_model(model_path, device):
    """加载训练好的模型"""
    model = UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载成功: {model_path}")
    if 'epoch' in checkpoint:
        print(f"  训练轮数: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  验证损失: {checkpoint['val_loss']:.4f}")
    return model


def predict_seismic(model, input_data, device):
    """使用模型进行预测"""
    input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_data = output_tensor.cpu().squeeze().numpy()
    return output_data


def visualize_prediction_results(original_norm, noisy_norm, output_norm, target_norm, save_dir='prediction_results'):
    """
    可视化预测结果：
    - 4个独立的地震剖面图窗口（同时显示）
    - 1个频谱对比图窗口
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    dt = 0.004
    
    # ========== 1. 创建4个独立的地震剖面图 ==========
    sections = [
        (original_norm, 'Original Sample (Normalized)', 'original_normalized', 'Figure 1'),
        (noisy_norm, 'Noisy Sample (Network Input)', 'noisy_input', 'Figure 2'),
        (output_norm, 'Network Output (Predicted)', 'network_output', 'Figure 3'),
        (target_norm, 'Ground Truth (Wide Band)', 'ground_truth', 'Figure 4')
    ]
    
    fig_list = []
    for data, title, filename, fig_name in sections:
        fig = plt.figure(num=fig_name, figsize=(10, 8))
        fig_list.append(fig)
        
        ax = fig.add_subplot(111)
        im = ax.imshow(data, cmap='seismic', aspect='auto', 
                      vmin=-1, vmax=1, interpolation='bilinear')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('Trace Number', fontsize=16, fontweight='bold')
        ax.set_ylabel('Time Sample', fontsize=16, fontweight='bold')
        ax.tick_params(labelsize=14)
        
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Amplitude', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=200, bbox_inches='tight')
    
    # ========== 2. 创建频谱对比图（单独窗口）==========
    # 计算频谱
    f_orig, s_orig = compute_spectrum(original_norm, dt)
    f_noisy, s_noisy = compute_spectrum(noisy_norm, dt)
    f_out, s_out = compute_spectrum(output_norm, dt)
    f_tgt, s_tgt = compute_spectrum(target_norm, dt)
    
    # 归一化频谱
    s_orig_norm = s_orig / np.max(s_orig)
    s_noisy_norm = s_noisy / np.max(s_noisy)
    s_out_norm = s_out / np.max(s_out)
    s_tgt_norm = s_tgt / np.max(s_tgt)
    
    # 创建频谱对比图
    fig_spec = plt.figure(num='Frequency Spectrum Comparison', figsize=(14, 8))
    fig_list.append(fig_spec)
    
    ax_spec = fig_spec.add_subplot(111)
    
    # 绘制四条频谱曲线
    ax_spec.plot(f_orig, s_orig_norm, 'b-', linewidth=2.5, 
                alpha=0.8, label='Original (Normalized)')
    ax_spec.plot(f_noisy, s_noisy_norm, 'orange', linewidth=2.5, 
                alpha=0.8, label='Noisy Input')
    ax_spec.plot(f_out, s_out_norm, 'r-', linewidth=2.5, 
                alpha=0.8, label='Network Output')
    ax_spec.plot(f_tgt, s_tgt_norm, 'g--', linewidth=2.5, 
                alpha=0.8, label='Ground Truth')
    
    # 标注主频
    colors = ['blue', 'orange', 'red', 'green']
    spectra = [s_orig_norm, s_noisy_norm, s_out_norm, s_tgt_norm]
    labels = ['Original', 'Noisy', 'Output', 'Target']
    
    for spectrum, color, label in zip(spectra, colors, labels):
        peak_idx = np.argmax(spectrum)
        peak_freq = f_orig[peak_idx]
        ax_spec.axvline(x=peak_freq, color=color, linestyle=':', 
                       linewidth=1.5, alpha=0.4)
    
    ax_spec.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
    ax_spec.set_ylabel('Normalized Amplitude', fontsize=16, fontweight='bold')
    ax_spec.set_title('Frequency Spectrum Comparison', fontsize=18, fontweight='bold', pad=15)
    ax_spec.set_xlim([0, 100])
    ax_spec.set_ylim([0, 1.05])
    ax_spec.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax_spec.legend(fontsize=14, loc='upper right', frameon=True, shadow=True)
    ax_spec.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectrum_comparison.png'), 
                dpi=200, bbox_inches='tight')
    
    print(f"\n所有可视化结果已保存到: {save_dir}/")
    print(f"  - 4张地震剖面图")
    print(f"  - 1张频谱对比图")
    
    return fig_list


def calculate_metrics(output, target):
    """计算预测指标"""
    mae = np.mean(np.abs(output - target))
    mse = np.mean((output - target) ** 2)
    rmse = np.sqrt(mse)
    
    max_val = max(np.max(np.abs(target)), np.max(np.abs(output)))
    if mse > 0:
        psnr = 20 * np.log10(max_val / rmse)
    else:
        psnr = float('inf')
    
    correlation = np.corrcoef(output.flatten(), target.flatten())[0, 1]
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'PSNR': psnr,
        'Correlation': correlation
    }
    
    return metrics


# ==================== 主预测脚本 ====================
if __name__ == "__main__":
    print("="*70)
    print("地震频带拓宽网络 - 预测脚本")
    print("="*70)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 数据路径
    narrow_path = r'seismic_records\seismic_narrow_band.npy'
    wide_path = r'seismic_records\seismic_wide_band.npy'
    model_path = r'checkpoints\best_model.pth'
    
    # 加载数据
    print("\n加载数据...")
    try:
        narrow_full = np.load(narrow_path)
        wide_full = np.load(wide_path)
        print(f"数据加载成功！")
        print(f"  窄频带数据形状: {narrow_full.shape}")
        print(f"  宽频带数据形状: {wide_full.shape}")
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 - {e}")
        print("请检查数据文件路径是否正确")
        exit(1)
    
    # 提取测试集部分 (1500-2000)
    test_start, test_end = 1500, 2000
    narrow_test_original = narrow_full[test_start:test_end, :]
    wide_test = wide_full[test_start:test_end, :]
    
    print(f"\n测试集范围: [{test_start}, {test_end})")
    print(f"测试集数据形状: {narrow_test_original.shape}")
    
    # ========== 处理流程 ==========
    # 1. 先归一化原始样本（用于对比）
    print("\n数据处理流程:")
    print("  步骤1: 归一化原始窄频带数据...")
    original_normalized, orig_min, orig_max = normalize_data(narrow_test_original)
    
    # 2. 添加随机噪声到原始数据
    print("  步骤2: 添加高斯随机噪声...")
    noise_coeff = 0.2
    narrow_test_noisy = add_gaussian_noise(narrow_test_original, noise_coeff)
    print(f"    噪声系数: {noise_coeff}")
    print(f"    原始数据标准差: {np.std(narrow_test_original):.6f}")
    print(f"    噪声标准差: {np.std(narrow_test_original) * noise_coeff:.6f}")
    
    # 3. 归一化带噪声的数据（网络输入）
    print("  步骤3: 归一化带噪声数据（网络输入）...")
    noisy_normalized, noisy_min, noisy_max = normalize_data(narrow_test_noisy)
    
    # 4. 归一化目标数据
    print("  步骤4: 归一化目标数据...")
    target_normalized, target_min, target_max = normalize_data(wide_test)
    
    # 保存处理后的数据
    results_dir = 'prediction_results2'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n保存中间数据到: {results_dir}/")
    np.save(os.path.join(results_dir, 'original_sample.npy'), narrow_test_original)
    np.save(os.path.join(results_dir, 'original_normalized.npy'), original_normalized)
    np.save(os.path.join(results_dir, 'noisy_sample.npy'), narrow_test_noisy)
    np.save(os.path.join(results_dir, 'noisy_normalized_input.npy'), noisy_normalized)
    np.save(os.path.join(results_dir, 'target_sample.npy'), wide_test)
    np.save(os.path.join(results_dir, 'target_normalized.npy'), target_normalized)
    print("  ✓ 所有中间数据已保存")
    
    # 加载模型
    print("\n加载训练好的模型...")
    try:
        model = load_model(model_path, device)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 - {model_path}")
        print("请确保模型已训练并保存在正确位置")
        exit(1)
    
    # 预测
    print("\n开始预测...")
    output_normalized = predict_seismic(model, noisy_normalized, device)
    print("预测完成！")
    
    # 保存预测结果
    np.save(os.path.join(results_dir, 'network_output_normalized.npy'), output_normalized)
    print(f"预测结果已保存: {results_dir}/network_output_normalized.npy")
    
    # 计算评估指标
    print("\n计算评估指标...")
    metrics = calculate_metrics(output_normalized, target_normalized)
    
    print("\n" + "="*70)
    print("预测性能指标:")
    print("="*70)
    for key, value in metrics.items():
        print(f"  {key:15s}: {value:.6f}")
    print("="*70)
    
    # 保存指标
    np.save(os.path.join(results_dir, 'metrics.npy'), metrics)
    
    # 可视化结果（同时显示所有窗口）
    print("\n生成可视化结果...")
    print("正在创建5个独立窗口:")
    print("  1. 原始样本（归一化）")
    print("  2. 带噪声样本（网络输入）")
    print("  3. 网络输出（预测结果）")
    print("  4. 真实标签（宽频带）")
    print("  5. 频谱对比图")
    
    fig_list = visualize_prediction_results(
        original_normalized,
        noisy_normalized,
        output_normalized,
        target_normalized,
        save_dir=results_dir
    )
    
    print("\n" + "="*70)
    print("预测任务完成！")
    print("="*70)
    print(f"\n所有结果保存在: {results_dir}/")
    print("\n数据文件:")
    print("  - original_sample.npy: 原始窄频带数据")
    print("  - original_normalized.npy: 归一化的原始数据")
    print("  - noisy_sample.npy: 添加噪声后的数据")
    print("  - noisy_normalized_input.npy: 网络输入（归一化+噪声）")
    print("  - network_output_normalized.npy: 网络预测输出")
    print("  - target_sample.npy: 真实宽频带数据")
    print("  - target_normalized.npy: 归一化的真实标签")
    print("  - metrics.npy: 性能评估指标")
    
    print("\n图片文件:")
    print("  - original_normalized.png: 原始样本图")
    print("  - noisy_input.png: 带噪声输入图")
    print("  - network_output.png: 网络输出图")
    print("  - ground_truth.png: 真实标签图")
    print("  - spectrum_comparison.png: 频谱对比图")
    
    print("\n所有窗口将同时显示，关闭窗口以结束程序...")
    print("="*70)
    
    # 同时显示所有窗口
    plt.show()