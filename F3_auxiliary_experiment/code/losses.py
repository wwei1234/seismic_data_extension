import torch
import torch.nn as nn


class BandwidthLoss(nn.Module):
    def __init__(self, lambda_mae=1.0, lambda_freq=0.5):
        super().__init__()
        self.lambda_mae = lambda_mae
        self.lambda_freq = lambda_freq
        self.mae = nn.L1Loss()

    @staticmethod
    def amplitude_spectrum(x):
        return torch.abs(torch.fft.rfft(x, dim=2, norm="ortho"))

    def forward(self, pred, target):
        loss_mae = self.mae(pred, target)
        pred_spec = self.amplitude_spectrum(pred)
        target_spec = self.amplitude_spectrum(target)
        loss_freq = torch.mean(torch.abs(pred_spec - target_spec))
        total = self.lambda_mae * loss_mae + self.lambda_freq * loss_freq
        return total, {
            "total": float(total.detach().cpu()),
            "mae": float(loss_mae.detach().cpu()),
            "freq": float(loss_freq.detach().cpu()),
        }
