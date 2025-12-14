import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralLoss(nn.Module):
    def __init__(self, n_fft=32, hop_length=16, win_length=32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def stft(self, x):
        """
        x: [Batch, 1, Length]
        Returns magnitude spectrogram.
        """
        # Squeeze channel dim for stft: [Batch, Length]
        x = x.squeeze(1)
        
        # Move window to same device
        if self.window.device != x.device:
            self.window = self.window.to(x.device)

        # Compute STFT
        # return_complex=True is required in newer PyTorch
        x_stft = torch.stft(x, self.n_fft, self.hop_length, self.win_length, 
                            window=self.window, return_complex=True)
        
        # Compute Magnitude: sqrt(real^2 + imag^2)
        # Clamp to avoid log(0)
        mag = torch.abs(x_stft).clamp(min=1e-7)
        return mag

    def forward(self, pred, target):
        mag_pred = self.stft(pred)
        mag_target = self.stft(target)

        # Log Magnitude Loss (L1 distance of logs) - helps with dynamic range
        log_loss = F.l1_loss(torch.log(mag_pred), torch.log(mag_target))
        
        # Magnitude Loss (L1 distance)
        mag_loss = F.l1_loss(mag_pred, mag_target)

        return log_loss + mag_loss

class DiffusionLoss(nn.Module):
    def __init__(self, loss_type="mse", lambda_spectral=0.001):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_spectral = lambda_spectral
        self.mse = nn.MSELoss()
        self.spectral = SpectralLoss()

    def forward(self, pred_x0, true_x0):
        """
        Args:
            pred_x0: Model prediction of clean signal
            true_x0: Ground truth clean signal
        """
        # 1. Standard Time-Domain Loss
        loss_mse = self.mse(pred_x0, true_x0)

        if self.loss_type == "mse":
            return loss_mse
        
        elif self.loss_type == "spectral":
            # 2. Add Frequency-Domain Loss
            loss_spec = self.spectral(pred_x0, true_x0)
            return loss_mse + (self.lambda_spectral * loss_spec)
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")