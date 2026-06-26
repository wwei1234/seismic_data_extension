import numpy as np


def denormalize_wide_prediction(pred_norm, narrow_scale):
    return np.asarray(pred_norm, dtype=np.float32) * float(narrow_scale)
