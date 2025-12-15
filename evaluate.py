import torch
import numpy as np
import pandas as pd
import os
import yaml
import argparse
from torch.utils.data import DataLoader

from src.dataset import ECGDataset
from src.model import ConditionalUNet1D
from src.diffusion import GaussianDiffusion
from src.utils import (set_seed, 
                       calculate_snr, 
                       calculate_rmse, 
                       calculate_lsd, 
                       save_plot_comparison, 
                       apply_lowpass_filter, 
                       plot_spectrograms,          
                       plot_metric_distributions,  
                       plot_residual_error)        

def load_model(config_path, checkpoint_path, device):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = ConditionalUNet1D(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Initialize Diffusion wrapper
    diffusion = GaussianDiffusion(
        model, 
        timesteps=config.get('timesteps', 1000), 
        device=device
    )
    return diffusion

def evaluate():
    # Hardcoded paths based on directory structure requirement
    baseline_cfg = "configs/baseline.yaml"
    spectral_cfg = "configs/spectral.yaml"
    
    baseline_ckpt = "experiments/baseline_run/checkpoint.pth"
    spectral_ckpt = "experiments/spectral_run/checkpoint.pth"
    
    output_dir = "results"
    plot_dir = os.path.join(output_dir, "comparison_plots")
    
    spec_dir = os.path.join(plot_dir, "spectrograms")
    res_dir = os.path.join(plot_dir, "residuals")
    
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    print(">>> Loading Models...")
    # Load Baseline
    diff_baseline = load_model(baseline_cfg, baseline_ckpt, device)
    
    # Load Spectral
    diff_spectral = load_model(spectral_cfg, spectral_ckpt, device)
    
    # Load Data
    dataset = ECGDataset("data/raw/mitbih_train.csv", train=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(">>> Starting Inference (This may take time due to sampling)...")
    
    metrics = {
        'Model': [],
        'SNR': [],
        'RMSE': [],
        'LSD': []
    }
    
    # Run only 1 batch for evaluation
    batch = next(iter(dataloader))
    
    clean = batch['clean'].to(device)
    noisy = batch['noisy'].to(device)
    
    # 1. Baseline Inference
    print("Running Baseline Inference...")
    pred_baseline = diff_baseline.sample(noisy)
    
    # 2. Spectral Inference
    print("Running Spectral Inference...")
    pred_spectral = diff_spectral.sample(noisy)
    
    print(">>> Applying Post-processing Filter to Spectral Output...")
    
    # Chuyển sang Numpy để lọc
    spectral_np = pred_spectral.cpu().numpy()
    spectral_filtered_np = np.zeros_like(spectral_np)
    
    # Lặp qua từng mẫu trong batch để lọc
    for idx in range(spectral_np.shape[0]):
        raw_sig = spectral_np[idx].squeeze()
        # Lọc bỏ tần số > 40Hz
        filt_sig = apply_lowpass_filter(raw_sig, fs=125, cutoff=40)
        spectral_filtered_np[idx, 0, :] = filt_sig
        
    # Cập nhật lại biến numpy dùng để tính SNR/RMSE và vẽ hình
    spectral_np = spectral_filtered_np
    
    # Cập nhật lại Tensor dùng để tính LSD
    pred_spectral = torch.from_numpy(spectral_filtered_np).to(device)

    # 3. Ensemble Model (Blending)
    print(">>> Creating Ensemble Model (Blending)...")
    
    # Tỷ lệ pha trộn: 50% Baseline + 50% Spectral
    alpha = 0.7
    beta = 1 - alpha 
    pred_ensemble = (alpha * pred_baseline) + ((beta) * pred_spectral)
    ensemble_np = pred_ensemble.cpu().numpy()
    # =======================================================
    
    # 3. Compute Metrics and Save Plots
    print(">>> Computing Metrics and Generating Advanced Plots...")
    
    clean_np = clean.cpu().numpy()
    baseline_np = pred_baseline.cpu().numpy()
    
    noisy_np_cpu = noisy.cpu().numpy() # Lấy noisy dạng numpy để vẽ
    
    # Loop through batch
    for i in range(clean.shape[0]):
        # A. Baseline Metrics
        b_snr = calculate_snr(clean_np[i], baseline_np[i])
        b_rmse = calculate_rmse(clean_np[i], baseline_np[i])
        b_lsd = calculate_lsd(clean[i:i+1], pred_baseline[i:i+1])
        
        metrics['Model'].append('Baseline')
        metrics['SNR'].append(b_snr)
        metrics['RMSE'].append(b_rmse)
        metrics['LSD'].append(b_lsd)
        
        # B. Spectral Metrics
        s_snr = calculate_snr(clean_np[i], spectral_np[i])
        s_rmse = calculate_rmse(clean_np[i], spectral_np[i])
        s_lsd = calculate_lsd(clean[i:i+1], pred_spectral[i:i+1])
        
        metrics['Model'].append('Spectral')
        metrics['SNR'].append(s_snr)
        metrics['RMSE'].append(s_rmse)
        metrics['LSD'].append(s_lsd)

        # C. Ensemble Metrics 
        e_snr = calculate_snr(clean_np[i], ensemble_np[i])
        e_rmse = calculate_rmse(clean_np[i], ensemble_np[i])
        e_lsd = calculate_lsd(clean[i:i+1], pred_ensemble[i:i+1])
        
        metrics['Model'].append('Ensemble')
        metrics['SNR'].append(e_snr)
        metrics['RMSE'].append(e_rmse)
        metrics['LSD'].append(e_lsd)
        
        if i < 5:

            # 1. Waveform & PSD 
            save_path = os.path.join(plot_dir, f"sample_{i}.png")
            save_plot_comparison(
                clean_np[i], 
                noisy_np_cpu[i], 
                baseline_np[i], 
                spectral_np[i], 
                save_path, 
                sample_idx=i,
                ensemble=ensemble_np[i]
            )
            
            # 2. Spectrogram (Cần squeeze về 1D)
            plot_spectrograms(
                clean_np[i].squeeze(), 
                noisy_np_cpu[i].squeeze(), 
                spectral_np[i].squeeze(), 
                os.path.join(spec_dir, f"spec_{i}.png")
            )
            
            # 3. Residual Error 
            plot_residual_error(
                clean_np[i], 
                baseline_np[i], 
                spectral_np[i],
                os.path.join(res_dir, f"residual_{i}.png"),
                ensemble=ensemble_np[i]
            )
            # -------------------------------------------------

    # Save to CSV
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(output_dir, "final_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    # --- BOXPLOT TỔNG HỢP ---
    print(">>> Generating Boxplots...")
    plot_metric_distributions(df, output_dir)
    # ---------------------------------
    
    # Print Average Results
    print("\n>>> Final Average Results:")
    summary = df.groupby('Model').mean()
    print(summary)
    print(f"\nResults saved to {output_dir}/")

if __name__ == "__main__":
    evaluate()