import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from src.noise_augment import add_noise

class ECGDataset(Dataset):
    def __init__(self, csv_path, train=True):
        """
        Args:
            csv_path (str): Path to mitbih_train.csv
            train (bool): If True, applies random noise augmentation.
        """
        # Load data, drop last column (label)
        self.data = pd.read_csv(csv_path, header=None).iloc[:, :-1].values.astype(np.float32)
        self.train = train

    def __len__(self):
        return len(self.data)

    def normalize(self, signal):
        """Min-Max normalization to [-1, 1] for Diffusion stability"""
        min_val = signal.min()
        max_val = signal.max()
        if max_val - min_val == 0:
            return signal
        return 2 * (signal - min_val) / (max_val - min_val) - 1

    def __getitem__(self, idx):
        clean_raw = self.data[idx]
        
        # Apply normalization to clean signal
        clean_norm = self.normalize(clean_raw)
        
        if self.train:
            # Generate noise based on the normalized clean signal context
            noisy_norm = add_noise(clean_norm)
        else:
            noisy_norm = add_noise(clean_norm)

        # Convert to Tensor [Channels, Length] -> [1, 187]
        clean_tensor = torch.from_numpy(clean_norm).unsqueeze(0).float()
        noisy_tensor = torch.from_numpy(noisy_norm).unsqueeze(0).float()

        return {'clean': clean_tensor, 'noisy': noisy_tensor}
