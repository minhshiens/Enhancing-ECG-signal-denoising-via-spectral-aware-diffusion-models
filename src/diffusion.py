import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class GaussianDiffusion:
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Define Beta Schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for Posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Coefficients for posterior mean
        # mean = coeff1 * x_0 + coeff2 * x_t
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )

    def p_sample(self, x_t, t, condition):
        """
        Single step denoising: x_t -> x_{t-1}
        """
        # 1. Predict x_0 using the model
        # Note: condition is the Artifact Noisy Signal
        pred_x0 = self.model(x_t, t, condition)
        
        # Clip pred_x0 to [-1, 1] for stability (standard in diffusion for signals)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # 2. Compute Posterior Mean
        model_mean = (
            self.posterior_mean_coef1[t][:, None, None] * pred_x0 +
            self.posterior_mean_coef2[t][:, None, None] * x_t
        )

        # 3. Compute Posterior Variance
        # Use log variance for numerical stability
        posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance[t], min=1e-20)
        )
        
        # 4. Sample x_{t-1}
        noise = torch.randn_like(x_t)
        # No noise when t=0
        nonzero_mask = (t > 0).float().unsqueeze(1).unsqueeze(2)
        
        x_prev = model_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance[:, None, None]) * noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample(self, condition):
        """
        Full reverse process: Noise -> Clean Signal
        Args:
            condition: The Artifact Noisy Input [Batch, 1, Length]
        """
        batch_size = condition.shape[0]
        length = condition.shape[2]
        
        # Start from pure Gaussian noise
        x_t = torch.randn(batch_size, 1, length).to(self.device)
        
        # Loop backwards from T-1 to 0
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps, leave=False):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x_t, _ = self.p_sample(x_t, t, condition)
            
        return x_t