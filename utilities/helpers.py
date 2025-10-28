"""
Helper functions for B2SC pipeline.

This module contains utility functions that are used across different components
of the B2SC pipeline, including device management, random seed control,
data validation, and common transformations.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Union
from termcolor import colored


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across numpy, torch, and other libraries.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(device_str: str = "auto") -> torch.device:
    """
    Setup and return the appropriate device for computation.

    Args:
        device_str: Device specification ("auto", "cuda", "cpu", "mps")

    Returns:
        torch.device: Configured device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using {torch.cuda.device_count()} GPU(s)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(device_str)
        print(f"Using specified device: {device}")

    return device


def validate_file_exists(
    file_path: Union[str, Path], description: str = "File"
) -> Path:
    """
    Validate that a file exists and return Path object.

    Args:
        file_path: Path to file
        description: Description for error messages

    Returns:
        Path: Validated path object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def ensure_directory(
    dir_path: Union[str, Path], description: str = "Directory"
) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        dir_path: Path to directory
        description: Description for logging

    Returns:
        Path: Directory path object
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print a summary of model architecture and parameters.

    Args:
        model: PyTorch model
        model_name: Name for display
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(colored(f"\n{model_name} Summary:", "cyan"))
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024**2)
    print(f"  Model size: {size_mb:.2f} MB")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
