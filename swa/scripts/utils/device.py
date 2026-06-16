"""
Device detection utility.

Unified detection of available compute device, priority: CUDA > MPS > CPU.
"""

import torch


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name() -> str:
    """Get device name string."""
    return str(get_device())


def is_cuda() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps() -> bool:
    """Check if Apple MPS is available."""
    return torch.backends.mps.is_available()


def get_xgboost_device() -> str:
    """XGBoost device string (MPS not supported, falls back to CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_lightgbm_device() -> str:
    """LightGBM device string (MPS not supported, falls back to CPU)."""
    return "gpu" if torch.cuda.is_available() else "cpu"
