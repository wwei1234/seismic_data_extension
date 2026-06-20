import numpy as np


def prepare_training_pair(clean_narrow, clean_wide, noise_level, rng):
    scale = max(float(np.percentile(np.abs(clean_narrow), 99)), 1e-8)
    clean_input_norm = np.clip(clean_narrow / scale, -1.0, 1.0).astype(np.float32)
    label_norm = np.clip(clean_wide / scale, -1.0, 1.0).astype(np.float32)
    input_norm = clean_input_norm.copy()
    if noise_level > 0:
        noise = rng.normal(0.0, noise_level, input_norm.shape).astype(np.float32)
        input_norm = np.clip(input_norm + noise, -1.0, 1.0)
    return input_norm.astype(np.float32), label_norm, scale
