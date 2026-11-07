import os
import random
import numpy as np


def set_seed(seed: int = 42):
    """Seed python, numpy and torch for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def moving_average(values, window: int = 20):
    """Compute rolling mean smoothing curve."""
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")
