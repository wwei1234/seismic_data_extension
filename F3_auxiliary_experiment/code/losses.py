import torch
import torch.nn as nn


class BandwidthLoss(nn.Module):
    def __init__(self, lambda_mae=1.0, lambda_freq=0.5, lambda_phase=0.3):
        super().__init__()
        self.lambda_mae = lambda_mae
        self.lambda_freq = lambda_freq
        self.lambda_phase = lambda_phase
        self.mae = nn.L1Loss()

    @staticmethod
    def spectrum(x):
        return torch.fft.rfft(x, dim=2, norm="ortho")

    def forward(self, pred, target):
        loss_mae = self.mae(pred, target)
        pred_spec = self.spectrum(pred)
        target_spec = self.spectrum(target)
        loss_freq = torch.mean(torch.abs(torch.abs(pred_spec) - torch.abs(target_spec)))
        phase_delta = torch.angle(pred_spec) - torch.angle(target_spec)
        loss_phase = torch.mean(1.0 - torch.cos(phase_delta))
        total = (
            self.lambda_mae * loss_mae
            + self.lambda_freq * loss_freq
            + self.lambda_phase * loss_phase
        )
        return total, {
            "total": float(total.detach().cpu()),
            "mae": float(loss_mae.detach().cpu()),
            "freq": float(loss_freq.detach().cpu()),
            "phase": float(loss_phase.detach().cpu()),
        }
