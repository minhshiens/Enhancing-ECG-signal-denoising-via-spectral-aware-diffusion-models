import argparse
import yaml
import os
import torch
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from src.dataset import ECGDataset
from src.model import ConditionalUNet1D
from src.losses import DiffusionLoss
from src.utils import set_seed # Assumed you implement this simple helper

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def train(config, config_path):

    print(f"Python đang chạy: {sys.executable}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # ÉP BẮT BUỘC DÙNG CUDA (Nếu không có sẽ báo lỗi đỏ lòm ngay lập tức)
    device = torch.device("cuda") 
    print(f"✅ ĐANG BẮT BUỘC SỬ DỤNG: {torch.cuda.get_device_name(0)}")
    
    # 1. Setup Dirs
    exp_dir = f"experiments/{config['exp_name']}"
    os.makedirs(exp_dir, exist_ok=True)
    # Save config for reproducibility
    with open(os.path.join(exp_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Training on {device} | Experiment: {config['exp_name']}")

    # 2. Data
    dataset = ECGDataset(config['data_path'], train=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # 3. Model
    model = ConditionalUNet1D(in_channels=2, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(config['lr']))

    # 4. Loss
    criterion = DiffusionLoss(loss_type=config['loss_type'], 
                              lambda_spectral=config.get('lambda', 0.1)).to(device)

    # 5. Diffusion Params
    T = config['timesteps']
    betas = linear_beta_schedule(T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # 6. Loop
    for epoch in range(config['epochs']):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        epoch_loss = 0

        for batch in pbar:
            clean = batch['clean'].to(device)   # x_0
            condition = batch['noisy'].to(device) # Condition (Artifacts)

            # Sample t
            t = torch.randint(0, T, (clean.shape[0],), device=device).long()

            # Forward Diffusion: Add Gaussian noise to Clean signal to get x_t
            noise = torch.randn_like(clean)
            x_t = (
                sqrt_alphas_cumprod[t, None, None] * clean +
                sqrt_one_minus_alphas_cumprod[t, None, None] * noise
            )

            # Model Prediction
            # We predict x_0 directly to facilitate Spectral Loss calculation
            # (Alternatively, predict epsilon and convert to x_0 inside loss)
            pred_x0 = model(x_t, t, condition)

            # Calculate Loss
            loss = criterion(pred_x0, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pth"))
        
        # Simple logging
        with open(os.path.join(exp_dir, "train.log"), "a") as f:
            f.write(f"Epoch {epoch+1}, Avg Loss: {epoch_loss / len(dataloader)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config, args.config)