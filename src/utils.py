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

def apply_lowpass_filter(waveform, fs=125, cutoff=40):
    """
    Áp dụng bộ lọc Butterworth Low-pass để loại bỏ nhiễu tần số cao.
    Args:
        waveform (np.array): Tín hiệu đầu vào (1D array)
        fs (int): Tần số lấy mẫu (Dataset Kaggle này thường là 125Hz)
        cutoff (int): Tần số cắt (40Hz là chuẩn cho ECG để giữ QRS nhưng bỏ nhiễu)
    """
    # 1. Thiết kế bộ lọc 
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    
    # 2. Lọc 2 chiều (Zero-phase filter) để không bị lệch pha tín hiệu
    # filtfilt chạy xuôi rồi chạy ngược, giúp giữ nguyên vị trí đỉnh QRS
    filtered = signal.filtfilt(b, a, waveform)
    
    return filtered.astype(np.float32)

def save_plot_comparison(clean, noisy, baseline, spectral, save_path, sample_idx=0, ensemble=None):
    """
    So sánh Waveform và PSD. Hỗ trợ thêm đường Ensemble .
    """
    clean = clean.squeeze()
    noisy = noisy.squeeze()
    baseline = baseline.squeeze()
    spectral = spectral.squeeze()
    if ensemble is not None:
        ensemble = ensemble.squeeze()
    
    fs = 125
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 10)) # Tăng kích thước ảnh
    
    # 1. Time Domain
    ax[0].plot(clean, label='Ground Truth', color='black', alpha=0.6, linewidth=2)
    ax[0].plot(noisy, label='Noisy Input', color='lightgray', alpha=0.5)
    ax[0].plot(baseline, label='Baseline (MSE)', color='blue', linestyle='--', linewidth=1.5)
    ax[0].plot(spectral, label='Spectral', color='red', linestyle='-.', linewidth=1.5, alpha=0.8)
    
    if ensemble is not None:
        # Vẽ Ensemble màu Xanh Lá Cây
        ax[0].plot(ensemble, label='Ensemble (Best)', color='green', linestyle='-', linewidth=2)
        
    ax[0].set_title(f'Waveform Reconstruction (Sample {sample_idx})')
    ax[0].legend(loc='upper right')
    ax[0].grid(True, alpha=0.3)
    
    # 2. Frequency Domain (PSD)
    f_c, P_c = signal.welch(clean, fs=fs, nperseg=64)
    f_b, P_b = signal.welch(baseline, fs=fs, nperseg=64)
    f_s, P_s = signal.welch(spectral, fs=fs, nperseg=64)
    
    ax[1].semilogy(f_c, P_c, label='Ground Truth', color='black', linewidth=2)
    ax[1].semilogy(f_b, P_b, label='Baseline', color='blue', linestyle='--')
    ax[1].semilogy(f_s, P_s, label='Spectral', color='red', linestyle='-.')
    
    if ensemble is not None:
        f_e, P_e = signal.welch(ensemble, fs=fs, nperseg=64)
        ax[1].semilogy(f_e, P_e, label='Ensemble', color='green', linestyle='-', linewidth=2)

    ax[1].set_title('Power Spectral Density (Log Scale)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('PSD (V**2/Hz)')
    ax[1].legend()
    ax[1].grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    

def plot_spectrograms(clean, noisy, spectral, save_path, fs=125):
    """
    Vẽ so sánh Spectrogram để thấy sự tái tạo tần số theo thời gian.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)
    
    # Định nghĩa hàm vẽ con để đỡ lặp code
    def draw_spec(ax, signal_data, title):
        # nperseg nhỏ vì tín hiệu ngắn (187 điểm)
        f, t, Sxx = signal.spectrogram(signal_data, fs=fs, nperseg=32, noverlap=16)
        # Chuyển sang dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10) 
        im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno')
        ax.set_title(title)
        ax.set_ylabel('Frequency [Hz]')
        return im

    draw_spec(axs[0], clean, "Ground Truth Spectrogram")
    draw_spec(axs[1], noisy, "Noisy Input Spectrogram")
    im = draw_spec(axs[2], spectral, "Proposed (Spectral) Reconstruction")
    
    axs[2].set_xlabel('Time [sec]')
    
    # Thêm colorbar chung
    fig.colorbar(im, ax=axs.ravel().tolist(), label='Power [dB]')
    plt.savefig(save_path)
    plt.close()

def plot_residual_error(clean, baseline, spectral, save_path, ensemble=None):
    """
    Vẽ sai số tuyệt đối. Thêm sai số của Ensemble.
    """
    clean = clean.squeeze()
    baseline = baseline.squeeze()
    spectral = spectral.squeeze()
    
    diff_base = np.abs(clean - baseline)
    diff_spec = np.abs(clean - spectral)
    
    if ensemble is not None:
        ensemble = ensemble.squeeze()
        diff_ens = np.abs(clean - ensemble)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Chỉ vẽ phần sai số (Residual) cho gọn
    ax.plot(diff_base, 'b', label='Baseline Error', linewidth=1, alpha=0.6)
    ax.plot(diff_spec, 'r', label='Spectral Error', linewidth=1, alpha=0.6)
    
    if ensemble is not None:
        # Đường sai số của Ensemble
        ax.plot(diff_ens, 'g', label='Ensemble Error', linewidth=2)
        ax.fill_between(np.arange(len(diff_ens)), diff_ens, color='green', alpha=0.1)
    
    ax.set_title("Absolute Residual Error (|True - Pred|)")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Absolute Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metric_distributions(df, save_dir):
    """
    Vẽ Boxplot so sánh phân phối chỉ số giữa 2 model.
    """
    metrics = ['SNR', 'RMSE', 'LSD']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    models = df['Model'].unique()
    colors = ['blue', 'red'] # Blue cho Baseline, Red cho Spectral
    
    for i, metric in enumerate(metrics):
        # Chuẩn bị dữ liệu cho boxplot
        data_to_plot = [df[df['Model'] == m][metric].values for m in models]
        
        bplot = axs[i].boxplot(data_to_plot, patch_artist=True, labels=models)
        axs[i].set_title(f'{metric} Distribution')
        axs[i].grid(True, linestyle='--', alpha=0.3)
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_boxplot.png"))
    plt.close()