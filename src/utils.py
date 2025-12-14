import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_snr(clean, denoised):
    """
    Computes SNR in dB.
    Args:
        clean: (np.array) Ground truth
        denoised: (np.array) Prediction
    """
    noise = clean - denoised
    power_signal = np.sum(clean ** 2)
    power_noise = np.sum(noise ** 2)
    if power_noise == 0:
        return 100.0 # Perfect reconstruction
    return 10 * np.log10(power_signal / power_noise)

def calculate_rmse(clean, denoised):
    return np.sqrt(np.mean((clean - denoised) ** 2))

def calculate_lsd(clean_tensor, denoised_tensor):
    """
    Log-Spectral Distance using PyTorch STFT.
    """
    # Define STFT params similar to loss function
    n_fft = 32
    win_length = 32
    hop_length = 16
    window = torch.hann_window(win_length).to(clean_tensor.device)

    clean_stft = torch.stft(clean_tensor.squeeze(1), n_fft, hop_length, win_length, window=window, return_complex=True)
    denoised_stft = torch.stft(denoised_tensor.squeeze(1), n_fft, hop_length, win_length, window=window, return_complex=True)

    clean_log = torch.log(torch.abs(clean_stft).clamp(min=1e-7))
    denoised_log = torch.log(torch.abs(denoised_stft).clamp(min=1e-7))

    # LSD formula: Mean square error of log spectra
    lsd = torch.sqrt(torch.mean((clean_log - denoised_log) ** 2))
    return lsd.item()

def save_plot_comparison(clean, noisy, baseline, spectral, save_path, sample_idx=0):
    """
    Plots Time-domain waveforms and Frequency-domain PSD.
    """
    clean = clean.squeeze()
    noisy = noisy.squeeze()
    baseline = baseline.squeeze()
    spectral = spectral.squeeze()
    
    fs = 125 # Assumption based on MIT-BIH
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # 1. Time Domain
    ax[0].plot(clean, label='Ground Truth', color='black', alpha=0.6, linewidth=1.5)
    ax[0].plot(noisy, label='Noisy Input', color='gray', alpha=0.3)
    ax[0].plot(baseline, label='Baseline (MSE)', color='blue', linestyle='--')
    ax[0].plot(spectral, label='Proposed (Spectral)', color='red', linestyle='-.')
    ax[0].set_title(f'Waveform Reconstruction (Sample {sample_idx})')
    ax[0].legend(loc='upper right')
    ax[0].grid(True, alpha=0.3)
    
    # 2. Frequency Domain (PSD)
    f_c, P_c = signal.welch(clean, fs=fs, nperseg=64)
    f_b, P_b = signal.welch(baseline, fs=fs, nperseg=64)
    f_s, P_s = signal.welch(spectral, fs=fs, nperseg=64)
    
    ax[1].semilogy(f_c, P_c, label='Ground Truth', color='black')
    ax[1].semilogy(f_b, P_b, label='Baseline (MSE)', color='blue', linestyle='--')
    ax[1].semilogy(f_s, P_s, label='Proposed (Spectral)', color='red', linestyle='-.')
    ax[1].set_title('Power Spectral Density (Log Scale)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('PSD (V**2/Hz)')
    ax[1].legend()
    ax[1].grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()