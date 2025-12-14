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
from src.utils import set_seed, calculate_snr, calculate_rmse, calculate_lsd, save_plot_comparison

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
    os.makedirs(plot_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    print(">>> Loading Models...")
    # Load Baseline
    diff_baseline = load_model(baseline_cfg, baseline_ckpt, device)
    
    # Load Spectral
    diff_spectral = load_model(spectral_cfg, spectral_ckpt, device)
    
    # Load Data (Use train=True to generate noise on the fly, but fixed seed ensures consistency)
    # Ideally, we would have a separate test csv, but instruction says use 'mitbih_train.csv'
    dataset = ECGDataset("data/raw/mitbih_train.csv", train=True)
    # Take a small batch for evaluation to save time
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(">>> Starting Inference (This may take time due to sampling)...")
    
    # Metrics containers
    metrics = {
        'Model': [],
        'SNR': [],
        'RMSE': [],
        'LSD': []
    }
    
    # Run only 1 batch for demonstration/study purposes
    batch = next(iter(dataloader))
    
    clean = batch['clean'].to(device)
    noisy = batch['noisy'].to(device)
    
    # 1. Baseline Inference
    print("Running Baseline Inference...")
    pred_baseline = diff_baseline.sample(noisy)
    
    # 2. Spectral Inference
    print("Running Spectral Inference...")
    pred_spectral = diff_spectral.sample(noisy)
    
    # 3. Compute Metrics and Save Plots
    print(">>> Computing Metrics...")
    
    # Convert to numpy for numeric metrics (SNR, RMSE)
    clean_np = clean.cpu().numpy()
    baseline_np = pred_baseline.cpu().numpy()
    spectral_np = pred_spectral.cpu().numpy()
    
    # Loop through batch
    for i in range(clean.shape[0]):
        # Baseline Metrics
        b_snr = calculate_snr(clean_np[i], baseline_np[i])
        b_rmse = calculate_rmse(clean_np[i], baseline_np[i])
        b_lsd = calculate_lsd(clean[i:i+1], pred_baseline[i:i+1])
        
        metrics['Model'].append('Baseline')
        metrics['SNR'].append(b_snr)
        metrics['RMSE'].append(b_rmse)
        metrics['LSD'].append(b_lsd)
        
        # Spectral Metrics
        s_snr = calculate_snr(clean_np[i], spectral_np[i])
        s_rmse = calculate_rmse(clean_np[i], spectral_np[i])
        s_lsd = calculate_lsd(clean[i:i+1], pred_spectral[i:i+1])
        
        metrics['Model'].append('Spectral')
        metrics['SNR'].append(s_snr)
        metrics['RMSE'].append(s_rmse)
        metrics['LSD'].append(s_lsd)
        
        # Save Plots (Save first 3 samples only to avoid clutter)
        if i < 3:
            save_path = os.path.join(plot_dir, f"sample_{i}.png")
            save_plot_comparison(
                clean_np[i], 
                noisy.cpu().numpy()[i], 
                baseline_np[i], 
                spectral_np[i], 
                save_path, 
                sample_idx=i
            )

    # Save to CSV
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(output_dir, "final_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    # Print Average Results
    print("\n>>> Final Average Results:")
    summary = df.groupby('Model').mean()
    print(summary)
    print(f"\nResults saved to {output_dir}/")

if __name__ == "__main__":
    evaluate()