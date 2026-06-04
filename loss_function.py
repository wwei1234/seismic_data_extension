import torch
import torch.nn as nn
import torch.fft


class SeismicBandwidthLoss(nn.Module):
    """
    地震频带拓宽复合损失函数
    
    数学公式:
    L_total = λ₁ · L_MAE + λ₂ · L_freq
    
    其中:
    1. L_MAE = (1/N) ∑|y_pred - y_true|
       - y_pred: 网络输出 (预测的宽频带地震记录)
       - y_true: 真实标签 (真实的宽频带地震记录)
       - N: 像素总数
       - 物理含义: 像素级振幅差异，确保输出在空间域与真实值接近
    
    2. L_freq = (1/K) ∑ₖ |F(y_pred)ₖ - F(y_true)ₖ|
       - F(): 傅里叶变换
       - k: 频率索引
       - K: 频率分量总数
       - 物理含义: 频域振幅谱差异，确保输出在各个频段都与真实值一致
    
    3. λ₁, λ₂: 权重系数，用于平衡两项损失的贡献
    """
    
    def __init__(self, lambda_mae=1.0, lambda_freq=0.5):
        """
        参数:
            lambda_mae: MAE损失的权重系数 (λ₁)
            lambda_freq: 频域损失的权重系数 (λ₂)
        """
        super(SeismicBandwidthLoss, self).__init__()
        self.lambda_mae = lambda_mae
        self.lambda_freq = lambda_freq
        self.mae_loss = nn.L1Loss()  # MAE损失函数
        
    def compute_frequency_spectrum(self, x):
        """
        计算频域振幅谱
        
        参数:
            x: 输入张量 [B, C, H, W]
               - B: batch size
               - C: 通道数 (对地震数据通常为1)
               - H: 时间采样点数
               - W: 道数
        
        返回:
            频域振幅谱 [B, C, H, W]
        """
        # 对时间维度 (H) 进行FFT
        fft_result = torch.fft.rfft(x, dim=2, norm='ortho')
        # 计算振幅谱
        amplitude_spectrum = torch.abs(fft_result)
        
        return amplitude_spectrum
    
    def forward(self, pred, target):
        """
        计算总损失
        
        参数:
            pred: 网络预测输出 [B, C, H, W]
            target: 真实标签 [B, C, H, W]
        
        返回:
            total_loss: 总损失
            loss_dict: 包含各项损失的字典，用于监控
        """
        # 1. 端对端MAE损失 (空间域)
        # L_MAE = (1/N) ∑|y_pred - y_true|
        loss_mae = self.mae_loss(pred, target)
        
        # 2. 频域约束损失
        # 计算预测和目标的频谱
        pred_spectrum = self.compute_frequency_spectrum(pred)
        target_spectrum = self.compute_frequency_spectrum(target)
        
        # L_freq = (1/K) ∑|F(y_pred) - F(y_true)|
        loss_freq = torch.mean(torch.abs(pred_spectrum - target_spectrum))
        
        # 总损失
        # L_total = λ₁ · L_MAE + λ₂ · L_freq
        total_loss = self.lambda_mae * loss_mae + self.lambda_freq * loss_freq
        
        # 返回损失字典用于监控
        loss_dict = {
            'total': total_loss.item(),
            'mae': loss_mae.item(),
            'freq': loss_freq.item()
        }
        
        return total_loss, loss_dict