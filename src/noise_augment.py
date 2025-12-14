import numpy as np
import torch

def get_baseline_wander(length, fs=125):
    """Generates low-frequency baseline wander using sinusoids."""
    t = np.linspace(0, length/fs, length)
    # Random amplitude and frequency (0.05Hz - 1Hz typical for respiration)
    amp = np.random.uniform(0.3, 1.0)
    freq = np.random.uniform(0.1, 0.4)
    phase = np.random.uniform(0, 2*np.pi)
    return amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)

def get_muscle_artifact(length):
    """Generates high-frequency muscle artifact noise."""
    # Random noise filtered or just high variance randoms
    noise = np.random.normal(0, 1, length).astype(np.float32)
    # Scale it down to be an artifact, not overwhelming
    scale = np.random.uniform(0.2, 0.5)
    return noise * scale

def add_noise(clean_signal):
    """
    Args:
        clean_signal (np.array): Shape (L,)
    Returns:
        noisy_signal (np.array): Shape (L,) with BW, MA, and Gaussian noise.
    """
    length = clean_signal.shape[0]
    
    # 1. Baseline Wander
    bw = get_baseline_wander(length)
    
    # 2. Muscle Artifact
    ma = get_muscle_artifact(length)
    
    # 3. Gaussian Noise (Sensor noise)
    gaussian = np.random.normal(0, 0.15, length).astype(np.float32)
    
    # Combine
    noise_total = bw + ma + gaussian
    noisy_signal = clean_signal + noise_total
    
    # Clip to reasonable ECG range to prevent explosion
    return np.clip(noisy_signal, -2.0, 2.0)