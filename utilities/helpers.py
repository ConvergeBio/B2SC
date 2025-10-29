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
import scanpy as sc
import pandas as pd
import anndata


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


def make_pseudo_bulk_adata(sc_h5ad_path: str, groupby: str, out_h5ad: str = None):
    """
    Reads an .h5ad, sums counts over the `groupby` obs column,
    and returns a new AnnData (groups × genes).

    Parameters
    ----------
    sc_h5ad_path
        Path to your single-cell AnnData (.h5ad).
    groupby
        Column in adata.obs to aggregate by (e.g. 'cell_type' or 'sample').
    out_h5ad
        If provided, will write the result to this path.

    Returns
    -------
    AnnData
        .X  : numpy array of shape (n_groups, n_genes)
        .obs: DataFrame with index = unique group labels
        .var: DataFrame with index = gene names
    """
    # 1) load
    adata = sc.read_h5ad(sc_h5ad_path)

    # 2) ensure dense matrix
    X = adata.X
    if not isinstance(X, (np.ndarray,)):
        X = X.toarray()

    # 3) build a cell×gene DataFrame
    df = pd.DataFrame(data=X, index=adata.obs_names, columns=adata.var_names)

    # 4) attach the grouping vector
    if groupby not in adata.obs:
        raise KeyError(f"'{groupby}' not in adata.obs")
    df[groupby] = adata.obs[groupby].values

    # 5) sum counts per group → DataFrame (n_groups × n_genes)
    bulk_counts = df.groupby(groupby).sum()

    # 6) construct the AnnData
    bulk_adata = anndata.AnnData(
        X=bulk_counts.values,
        obs=pd.DataFrame(index=bulk_counts.index),
        var=pd.DataFrame(index=bulk_counts.columns),
    )

    # optional: write to disk
    if out_h5ad:
        bulk_adata.write(out_h5ad)

    return bulk_adata


def calculate_pseudobulk(adata: anndata.AnnData, groupby_col: str) -> anndata.AnnData:
    """
    Calculate pseudobulk data by summing expression values for each group.

    This is a more flexible version that works directly with AnnData objects
    without requiring file I/O.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell AnnData object with cells as observations
    groupby_col : str
        Column in adata.obs to group by (e.g., 'donor_id', 'cell_type', 'sample')

    Returns
    -------
    anndata.AnnData
        Pseudobulk AnnData object with groups as observations
        .X : expression matrix (n_groups × n_genes)
        .obs : metadata for each group
        .var : gene metadata (copied from input)
    """
    # Ensure dense matrix
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # Check if groupby column exists
    if groupby_col not in adata.obs.columns:
        raise KeyError(f"Column '{groupby_col}' not found in adata.obs")

    # Create DataFrame for easier grouping
    df = pd.DataFrame(data=X, index=adata.obs_names, columns=adata.var_names)

    # Add grouping column
    df[groupby_col] = adata.obs[groupby_col].values

    # Sum expression values by group
    pseudobulk_df = df.groupby(groupby_col).sum()

    # Create new AnnData object
    pseudobulk_adata = anndata.AnnData(
        X=pseudobulk_df.values,
        obs=pd.DataFrame(index=pseudobulk_df.index),
        var=adata.var.copy(),
    )

    # Add group information to obs
    pseudobulk_adata.obs[groupby_col] = pseudobulk_adata.obs.index

    return pseudobulk_adata
