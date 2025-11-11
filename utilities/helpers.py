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


def get_cell_counts_per_bulk_sample(
    sc_data: anndata.AnnData, bulk_donor_ids: list, groupby_col: str
) -> Dict[str, int]:
    """
    Calculate the number of cells for each bulk sample from corresponding SC data.

    Args:
        sc_data: Single-cell AnnData object
        bulk_donor_ids: List of donor IDs from bulk samples
        groupby_col: Column name used for grouping (e.g., 'donor_id')

    Returns:
        Dict mapping donor_id to cell count
    """
    cell_counts = {}

    for donor_id in bulk_donor_ids:
        # Count cells for this donor
        donor_mask = sc_data.obs[groupby_col] == donor_id
        cell_count = donor_mask.sum()
        cell_counts[donor_id] = cell_count

    return cell_counts


def fit_pca_and_transform(
    train_data: np.ndarray,
    val_data: np.ndarray = None,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple:
    """
    Fit PCA on training data and transform both training and validation data.

    Args:
        train_data: Training data array of shape (n_samples, n_features)
        val_data: Optional validation data array of shape (n_samples, n_features)
        n_components: Number of PCA components
        random_state: Random state for reproducibility

    Returns:
        Tuple of (train_transformed, val_transformed, pca_model)
        val_transformed is None if val_data is None
    """
    from sklearn.decomposition import PCA
    import numpy as np

    # Check for NaN values in training data
    if np.isnan(train_data).any():
        print("⚠️  NaN values detected in training data for PCA!")
        print(f"   Training data shape: {train_data.shape}")
        nan_count = np.isnan(train_data).sum()
        total_elements = train_data.size
        print(
            f"   NaN count: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%)"
        )

        # Replace NaN values with robust imputation
        print("   Replacing NaN values with robust imputation...")
        train_data_clean = train_data.copy()

        # Calculate robust statistics for each column
        for col_idx in range(train_data.shape[1]):
            col_data = train_data_clean[:, col_idx]
            mask = np.isnan(col_data)

            if mask.any():
                # Get valid (non-NaN) values for this column
                valid_values = col_data[~mask]

                if len(valid_values) > 0:
                    # Use median for robustness (less affected by outliers than mean)
                    col_median = np.median(valid_values)

                    # Check if median is finite
                    if np.isfinite(col_median):
                        fill_value = col_median
                    else:
                        fill_value = 0.0
                else:
                    # All values in column are NaN
                    fill_value = 0.0

                train_data_clean[mask, col_idx] = fill_value

        # Final check for any remaining non-finite values
        inf_mask = ~np.isfinite(train_data_clean)
        if inf_mask.any():
            print(
                f"   Warning: Found {inf_mask.sum()} infinite values, replacing with zeros"
            )
            train_data_clean[inf_mask] = 0.0

        train_data = train_data_clean
        print("   ✅ NaN/Inf values replaced successfully")

    # Fit PCA on training data
    pca = PCA(n_components=n_components, random_state=random_state)
    train_transformed = pca.fit_transform(train_data)

    # Transform validation data if provided
    val_transformed = None
    if val_data is not None:
        # Check for NaN values in validation data
        if np.isnan(val_data).any():
            print("⚠️  NaN values detected in validation data for PCA!")
            print(f"   Validation data shape: {val_data.shape}")
            nan_count = np.isnan(val_data).sum()
            total_elements = val_data.size
            print(
                f"   NaN count: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%)"
            )

            # Replace NaN values with robust imputation using training data statistics
            print("   Replacing NaN values with training data statistics...")
            val_data_clean = val_data.copy()

            # Use robust statistics from training data
            for col_idx in range(val_data.shape[1]):
                mask = np.isnan(val_data_clean[:, col_idx])
                if mask.any():
                    # Use training data median for this column
                    train_col_data = train_data[:, col_idx]
                    train_valid_values = train_col_data[~np.isnan(train_col_data)]

                    if len(train_valid_values) > 0:
                        train_col_median = np.median(train_valid_values)
                        if np.isfinite(train_col_median):
                            fill_value = train_col_median
                        else:
                            fill_value = 0.0
                    else:
                        fill_value = 0.0

                    val_data_clean[mask, col_idx] = fill_value

            # Final check for any remaining non-finite values in validation data
            inf_mask = ~np.isfinite(val_data_clean)
            if inf_mask.any():
                print(
                    f"   Warning: Found {inf_mask.sum()} infinite values in validation data, replacing with zeros"
                )
                val_data_clean[inf_mask] = 0.0

            val_data = val_data_clean
            print("   ✅ NaN/Inf values in validation data replaced successfully")

        val_transformed = pca.transform(val_data)

    return train_transformed, val_transformed, pca


def plot_deconvolution_comparison_3d(
    train_deconv: np.ndarray,
    train_real: np.ndarray,
    val_deconv: np.ndarray = None,
    val_real: np.ndarray = None,
    train_labels: list = None,
    val_labels: list = None,
    train_class_labels: list = None,
    val_class_labels: list = None,
    pca_model=None,
    save_path: str = None,
    figsize: tuple = (10, 8),
    title: str = "Deconvoluted vs Real SC Samples Comparison",
) -> None:
    """
    Create interactive 2D scatter plots comparing deconvoluted and real single-cell samples.
    Shows two plots: one colored by train/val split, another by class labels.

    Args:
        train_deconv: Training deconvoluted samples (n_samples, 2)
        train_real: Training real SC samples (n_samples, 2)
        val_deconv: Validation deconvoluted samples (n_samples, 2)
        val_real: Validation real SC samples (n_samples, 2)
        train_labels: Labels for training samples
        val_labels: Labels for validation samples
        train_class_labels: Class labels for training samples
        val_class_labels: Class labels for validation samples
        pca_model: Fitted PCA model for variance explanation
        save_path: Optional path to save the plot
        figsize: Figure size
        title: Plot title
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio

        use_plotly = True
    except ImportError:
        print("   Warning: plotly not available, falling back to matplotlib")
        use_plotly = False

    if use_plotly:
        # Create two side-by-side 2D plots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Colored by Train/Val Split", "Colored by Class Labels"),
            horizontal_spacing=0.1,
        )

        # Prepare axis labels based on the reduction method
        if pca_model is not None:
            # Check if it's a PCA model (has explained_variance_ratio_) or UMAP
            if hasattr(pca_model, "explained_variance_ratio_"):
                # PCA model
                pc1_label = f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})"
                pc2_label = f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})"
                total_variance = pca_model.explained_variance_ratio_[:2].sum()
            else:
                # UMAP model
                pc1_label = "UMAP1"
                pc2_label = "UMAP2"
                total_variance = None
        else:
            pc1_label, pc2_label = "Dim1", "Dim2"
            total_variance = None

        # Define colors and markers
        colors = {
            "train_deconv": "blue",
            "train_real": "darkblue",
            "val_deconv": "red",
            "val_real": "darkred",
        }

        markers = {"deconv": "circle", "real": "diamond"}

        sizes = {"deconv": 8, "real": 10}

        # Plot 1: Colored by Train/Val Split
        # Training data
        fig.add_trace(
            go.Scatter(
                x=train_deconv[:, 0],
                y=train_deconv[:, 1],
                mode="markers",
                marker=dict(
                    size=sizes["deconv"],
                    color=colors["train_deconv"],
                    symbol=markers["deconv"],
                    opacity=0.7,
                    line=dict(width=1, color="black"),
                ),
                name="Train - Deconvoluted",
                text=[f"Donor: {label}" for label in train_labels],
                hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                legendgroup="split",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=train_real[:, 0],
                y=train_real[:, 1],
                mode="markers",
                marker=dict(
                    size=sizes["real"],
                    color=colors["train_real"],
                    symbol=markers["real"],
                    opacity=0.9,
                    line=dict(width=1, color="black"),
                ),
                name="Train - Real SC",
                text=[f"Donor: {label}" for label in train_labels],
                hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                legendgroup="split",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Add validation data to plot 1 if available
        if val_deconv is not None and val_real is not None:
            fig.add_trace(
                go.Scatter(
                    x=val_deconv[:, 0],
                    y=val_deconv[:, 1],
                    mode="markers",
                    marker=dict(
                        size=sizes["deconv"],
                        color=colors["val_deconv"],
                        symbol=markers["deconv"],
                        opacity=0.7,
                        line=dict(width=1, color="black"),
                    ),
                    name="Val - Deconvoluted",
                    text=[f"Donor: {label}" for label in val_labels],
                    hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                    legendgroup="split",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=val_real[:, 0],
                    y=val_real[:, 1],
                    mode="markers",
                    marker=dict(
                        size=sizes["real"],
                        color=colors["val_real"],
                        symbol=markers["real"],
                        opacity=0.9,
                        line=dict(width=1, color="black"),
                    ),
                    name="Val - Real SC",
                    text=[f"Donor: {label}" for label in val_labels],
                    hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                    legendgroup="split",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Plot 2: Colored by Class Labels
        if train_class_labels is not None:
            # Define class colors
            class_colors = {0: "green", 1: "orange"}  # 0=Control, 1=Disease
            class_names = {0: "Control", 1: "Disease"}

            # Group training data by class
            for class_label in set(train_class_labels):
                if class_label is not None:
                    # Get indices for this class
                    class_indices = [
                        i
                        for i, cl in enumerate(train_class_labels)
                        if cl == class_label
                    ]

                    if class_indices:
                        # Deconvoluted samples for this class
                        fig.add_trace(
                            go.Scatter(
                                x=train_deconv[class_indices, 0],
                                y=train_deconv[class_indices, 1],
                                mode="markers",
                                marker=dict(
                                    size=sizes["deconv"],
                                    color=class_colors.get(class_label, "gray"),
                                    symbol=markers["deconv"],
                                    opacity=0.7,
                                    line=dict(width=1, color="black"),
                                ),
                                name=f"{class_names.get(class_label, f'Class {class_label}')} - Deconv",
                                text=[
                                    f"Donor: {train_labels[i]}, Class: {class_names.get(class_label, class_label)}"
                                    for i in class_indices
                                ],
                                hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                                legendgroup="class",
                                showlegend=True,
                            ),
                            row=1,
                            col=2,
                        )

                        # Real samples for this class
                        fig.add_trace(
                            go.Scatter(
                                x=train_real[class_indices, 0],
                                y=train_real[class_indices, 1],
                                mode="markers",
                                marker=dict(
                                    size=sizes["real"],
                                    color=class_colors.get(class_label, "gray"),
                                    symbol=markers["real"],
                                    opacity=0.9,
                                    line=dict(width=1, color="black"),
                                ),
                                name=f"{class_names.get(class_label, f'Class {class_label}')} - Real",
                                text=[
                                    f"Donor: {train_labels[i]}, Class: {class_names.get(class_label, class_label)}"
                                    for i in class_indices
                                ],
                                hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                                legendgroup="class",
                                showlegend=True,
                            ),
                            row=1,
                            col=2,
                        )

            # Add validation data to plot 2 if available
            if (
                val_deconv is not None
                and val_real is not None
                and val_class_labels is not None
            ):
                for class_label in set(val_class_labels):
                    if class_label is not None:
                        # Get indices for this class
                        class_indices = [
                            i
                            for i, cl in enumerate(val_class_labels)
                            if cl == class_label
                        ]

                        if class_indices:
                            # Deconvoluted samples for this class
                            fig.add_trace(
                                go.Scatter(
                                    x=val_deconv[class_indices, 0],
                                    y=val_deconv[class_indices, 1],
                                    mode="markers",
                                    marker=dict(
                                        size=sizes["deconv"],
                                        color=class_colors.get(class_label, "gray"),
                                        symbol=markers["deconv"],
                                        opacity=0.5,  # Slightly more transparent for val
                                        line=dict(width=1, color="black"),
                                    ),
                                    name=f"{class_names.get(class_label, f'Class {class_label}')} - Deconv",
                                    text=[
                                        f"Donor: {val_labels[i]}, Class: {class_names.get(class_label, class_label)} (Val)"
                                        for i in class_indices
                                    ],
                                    hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                                    legendgroup="class",
                                    showlegend=False,  # Don't duplicate legend
                                ),
                                row=1,
                                col=2,
                            )

                            # Real samples for this class
                            fig.add_trace(
                                go.Scatter(
                                    x=val_real[class_indices, 0],
                                    y=val_real[class_indices, 1],
                                    mode="markers",
                                    marker=dict(
                                        size=sizes["real"],
                                        color=class_colors.get(class_label, "gray"),
                                        symbol=markers["real"],
                                        opacity=0.7,  # Slightly more transparent for val
                                        line=dict(width=1, color="black"),
                                    ),
                                    name=f"{class_names.get(class_label, f'Class {class_label}')} - Real",
                                    text=[
                                        f"Donor: {val_labels[i]}, Class: {class_names.get(class_label, class_label)} (Val)"
                                        for i in class_indices
                                    ],
                                    hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                                    legendgroup="class",
                                    showlegend=False,  # Don't duplicate legend
                                ),
                                row=1,
                                col=2,
                            )

        # Update layout
        fig.update_layout(
            title=title
            + (f" (Total variance: {total_variance:.1%})" if total_variance else ""),
            height=600,
            width=1600,  # Wider for two plots
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        )

        # Update axes for both subplots
        fig.update_xaxes(
            title_text=pc1_label, showgrid=True, gridwidth=1, gridcolor="lightgray"
        )
        fig.update_yaxes(
            title_text=pc2_label, showgrid=True, gridwidth=1, gridcolor="lightgray"
        )

        # Save and show
        if save_path:
            # Save as HTML for interactivity
            html_path = (
                save_path.replace(".png", ".html")
                if save_path.endswith(".png")
                else save_path + ".html"
            )
            fig.write_html(html_path)
            print(f"   Interactive plot saved to: {html_path}")

            # Also save static image
            try:
                fig.write_image(save_path, width=800, height=600, scale=2)
                print(f"   Static plot saved to: {save_path}")
            except Exception as e:
                print(f"   Warning: Could not save static image: {e}")

        # Suppress output by not returning the figure
        fig.show()
        return None

    else:
        # Fallback to matplotlib with two 2D plots
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Prepare axis labels based on the reduction method
        if pca_model is not None:
            # Check if it's a PCA model (has explained_variance_ratio_) or UMAP
            if hasattr(pca_model, "explained_variance_ratio_"):
                # PCA model
                pc1_label = f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})"
                pc2_label = f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})"
                total_variance = pca_model.explained_variance_ratio_[:2].sum()
            else:
                # UMAP model
                pc1_label = "UMAP1"
                pc2_label = "UMAP2"
                total_variance = None
        else:
            pc1_label, pc2_label = "Dim1", "Dim2"
            total_variance = None

        # Define colors and markers
        colors = {"train": "blue", "val": "red"}
        markers = {"deconv": "o", "real": "^"}
        alphas = {"deconv": 0.7, "real": 0.9}
        sizes = {"deconv": 50, "real": 80}

        # Plot 1: Training data colored by split
        ax1.scatter(
            train_deconv[:, 0],
            train_deconv[:, 1],
            c=colors["train"],
            marker=markers["deconv"],
            alpha=alphas["deconv"],
            s=sizes["deconv"],
            label="Train - Deconvoluted",
            edgecolors="black",
            linewidth=0.5,
        )
        ax1.scatter(
            train_real[:, 0],
            train_real[:, 1],
            c=colors["train"],
            marker=markers["real"],
            alpha=alphas["real"],
            s=sizes["real"],
            label="Train - Real SC",
            edgecolors="darkblue",
            linewidth=0.5,
        )

        # Plot validation data if available
        if val_deconv is not None and val_real is not None:
            ax1.scatter(
                val_deconv[:, 0],
                val_deconv[:, 1],
                c=colors["val"],
                marker=markers["deconv"],
                alpha=alphas["deconv"],
                s=sizes["deconv"],
                label="Val - Deconvoluted",
                edgecolors="black",
                linewidth=0.5,
            )
            ax1.scatter(
                val_real[:, 0],
                val_real[:, 1],
                c=colors["val"],
                marker=markers["real"],
                alpha=alphas["real"],
                s=sizes["real"],
                label="Val - Real SC",
                edgecolors="darkred",
                linewidth=0.5,
            )

        ax1.set_xlabel(pc1_label)
        ax1.set_ylabel(pc2_label)
        ax1.set_title("Colored by Train/Val Split")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Colored by Class Labels
        if train_class_labels is not None:
            class_colors = {0: "green", 1: "orange"}  # 0=Control, 1=Disease
            class_names = {0: "Control", 1: "Disease"}

            # Group training data by class
            for class_label in set(train_class_labels):
                if class_label is not None:
                    class_indices = [
                        i
                        for i, cl in enumerate(train_class_labels)
                        if cl == class_label
                    ]

                    if class_indices:
                        ax2.scatter(
                            train_deconv[class_indices, 0],
                            train_deconv[class_indices, 1],
                            c=class_colors.get(class_label, "gray"),
                            marker=markers["deconv"],
                            alpha=alphas["deconv"],
                            s=sizes["deconv"],
                            label=f"{class_names.get(class_label, f'Class {class_label}')} - Deconv",
                            edgecolors="black",
                            linewidth=0.5,
                        )

                        ax2.scatter(
                            train_real[class_indices, 0],
                            train_real[class_indices, 1],
                            c=class_colors.get(class_label, "gray"),
                            marker=markers["real"],
                            alpha=alphas["real"],
                            s=sizes["real"],
                            label=f"{class_names.get(class_label, f'Class {class_label}')} - Real",
                            edgecolors="black",
                            linewidth=0.5,
                        )

            # Add validation data to plot 2 if available
            if (
                val_deconv is not None
                and val_real is not None
                and val_class_labels is not None
            ):
                for class_label in set(val_class_labels):
                    if class_label is not None:
                        class_indices = [
                            i
                            for i, cl in enumerate(val_class_labels)
                            if cl == class_label
                        ]

                        if class_indices:
                            ax2.scatter(
                                val_deconv[class_indices, 0],
                                val_deconv[class_indices, 1],
                                c=class_colors.get(class_label, "gray"),
                                marker=markers["deconv"],
                                alpha=0.5,  # More transparent for validation
                                s=sizes["deconv"],
                                edgecolors="black",
                                linewidth=0.5,
                            )

                            ax2.scatter(
                                val_real[class_indices, 0],
                                val_real[class_indices, 1],
                                c=class_colors.get(class_label, "gray"),
                                marker=markers["real"],
                                alpha=0.7,  # More transparent for validation
                                s=sizes["real"],
                                edgecolors="black",
                                linewidth=0.5,
                            )

        ax2.set_xlabel(pc1_label)
        ax2.set_ylabel(pc2_label)
        ax2.set_title("Colored by Class Labels")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.suptitle(
            title
            + (f" (Total variance: {total_variance:.1%})" if total_variance else "")
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"   Plot saved to: {save_path}")

        plt.show()


# ============================================================================
# Deconvolution Comparison Visualization Helper Functions
# ============================================================================


def validate_models_for_visualization(pipeline):
    """Validate that required models are trained before visualization."""
    if not pipeline._gmvae_trained:
        raise ValueError(
            "GMVAE must be trained before visualization. Call train_gmvae() first."
        )
    if not pipeline._bulk_encoder_trained:
        raise ValueError(
            "BulkEncoder must be trained before visualization. Call train_bulk_encoder() first."
        )
    if not pipeline._data_loaded:
        pipeline._load_training_data()


def setup_visualization_environment(pipeline, random_seed: int):
    """Setup environment for visualization including random seeds and model loading."""
    import torch
    import numpy as np
    from termcolor import colored

    print(colored("🔬 Generating deconvolution comparison visualization...", "cyan"))

    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Load GMVAE model once and reuse it
    print("   Loading GMVAE model...")
    gmvae = pipeline._load_trained_gmvae()
    device = pipeline.gmvae_trainer.device

    return gmvae, device


def convert_cells_to_latent(real_cells, gmvae, device):
    """Convert real single cells to latent space representation."""
    import torch

    real_cells_tensor = torch.tensor(real_cells.X.toarray(), dtype=torch.float32)
    if real_cells_tensor.dim() == 1:
        real_cells_tensor = real_cells_tensor.unsqueeze(0)

    with torch.no_grad():
        mus, _, pis = gmvae.encode(real_cells_tensor.to(device))
        # mus shape: [num_cells, n_components, latent_dim]
        # pis shape: [num_cells, n_components]

        # Weight the mus by the mixture probabilities and sum over components
        # to get a single latent representation per cell
        pis_normalized = torch.softmax(pis, dim=1)  # [num_cells, n_components]
        weighted_mus = (mus * pis_normalized.unsqueeze(-1)).sum(
            dim=1
        )  # [num_cells, latent_dim]

    # Average real cells to single vector
    return weighted_mus.mean(dim=0).cpu().numpy()


def process_bulk_sample_pair(
    pipeline,
    bulk_sample,
    donor_id,
    cell_counts,
    gmvae,
    device,
    random_seed_offset,
    sc_data,
    split_name,
    verbose=False,
):
    """Process a single bulk sample and its corresponding real cells."""
    if cell_counts[donor_id] == 0:
        if verbose:
            print(f"   Skipping {split_name.lower()} donor {donor_id}: no cells")
        return None

    if verbose:
        print(
            f"   Processing {split_name.lower()} donor {donor_id} ({cell_counts[donor_id]} cells)"
        )

    try:
        # Generate deconvoluted cells matching the real cell count
        num_cells = cell_counts[donor_id]
        deconv_latent, cell_types = pipeline.bulk_to_latent(
            bulk_sample,
            num_samples=num_cells,
            random_seed=random_seed_offset,
        )

        # Check for NaN and Inf values in deconvoluted latent vectors
        has_nan = torch.isnan(deconv_latent).any()
        has_inf = torch.isinf(deconv_latent).any()

        if has_nan or has_inf:
            nan_count = torch.isnan(deconv_latent).sum().item()
            inf_count = torch.isinf(deconv_latent).sum().item()
            total_elements = deconv_latent.numel()

            if has_nan and has_inf:
                print(
                    colored(
                        f"⚠️  NaN and Inf values detected in deconvoluted latent vectors for donor {donor_id}!",
                        "red",
                    )
                )
                print(
                    f"   Shape: {deconv_latent.shape}, NaN count: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%), Inf count: {inf_count}/{total_elements} ({100*inf_count/total_elements:.2f}%)"
                )
            elif has_nan:
                print(
                    colored(
                        f"⚠️  NaN values detected in deconvoluted latent vectors for donor {donor_id}!",
                        "red",
                    )
                )
                print(
                    f"   Shape: {deconv_latent.shape}, NaN count: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%)"
                )
            else:  # has_inf
                print(
                    colored(
                        f"⚠️  Inf values detected in deconvoluted latent vectors for donor {donor_id}!",
                        "red",
                    )
                )
                print(
                    f"   Shape: {deconv_latent.shape}, Inf count: {inf_count}/{total_elements} ({100*inf_count/total_elements:.2f}%)"
                )

            print(
                "   This indicates potential issues in GMVAE training or bulk encoder."
            )
        assert deconv_latent.shape[0] == num_cells
        assert deconv_latent.shape[1] == pipeline.config.latent_dim
        # Average deconvoluted cells to single vector
        deconv_vector = deconv_latent.mean(dim=0).cpu().numpy()

        # Get real single-cell data for this donor
        donor_mask = sc_data.obs[pipeline.config.pseudobulk_groupby] == donor_id
        real_cells = sc_data[donor_mask]

        # ASSERTION: Verify that this donor has exactly one unique label
        unique_donors = real_cells.obs[pipeline.config.pseudobulk_groupby].unique()
        assert len(unique_donors) == 1, (
            f"Expected exactly 1 unique donor ID for donor {donor_id}, "
            f"but found {len(unique_donors)}: {unique_donors}. "
            f"This indicates a data consistency issue."
        )
        assert unique_donors[0] == donor_id, (
            f"Expected donor ID {donor_id}, but found {unique_donors[0]}. "
            f"This indicates a data filtering issue."
        )

        # ASSERTION: Verify we're not confusing cell types with donor labels
        if "cell_type" in real_cells.obs.columns:
            unique_cell_types = real_cells.obs["cell_type"].unique()
            # Each donor should potentially have multiple cell types (this is expected for bulk samples)
            # print(
            #     f"   Donor {donor_id} contains {len(unique_cell_types)} cell types: {list(unique_cell_types)}"
            # )

            # Ensure donor_id is not accidentally a cell type
            assert donor_id not in unique_cell_types, (
                f"Donor ID '{donor_id}' appears to be a cell type, not a sample identifier. "
                f"Found cell types: {unique_cell_types}. "
                f"This indicates confusion between donor labels and cell type labels."
            )

        # Extract class label information if available
        class_label = None
        if "sample_label" in real_cells.obs.columns:
            unique_class_labels = real_cells.obs["sample_label"].unique()
            if len(unique_class_labels) == 1:
                class_label = unique_class_labels[0]
                print(f"   Donor {donor_id} class label: {class_label}")
            else:
                print(
                    f"   Warning: Donor {donor_id} has multiple class labels: {unique_class_labels}"
                )
                class_label = unique_class_labels[0]  # Use the first one

        # Convert to latent space using GMVAE encoder
        real_vector = convert_cells_to_latent(real_cells, gmvae, device)

        if verbose:
            print(f"   ✓ Successfully processed {split_name.lower()} donor {donor_id}")

        return {
            "deconv_vector": deconv_vector,
            "cell_types": cell_types,
            "real_vector": real_vector,
            "donor_id": donor_id,
            "class_label": class_label,
        }

    except Exception as e:
        print(
            f"   Warning: Could not process {split_name.lower()} donor {donor_id}: {e}"
        )
        return None


def process_dataloader_samples(
    pipeline,
    dataloader,
    sc_data,
    gmvae,
    device,
    random_seed,
    max_samples,
    split_name,
    verbose=False,
):
    """
    Process bulk samples from a dataloader and generate corresponding deconvoluted/real cell pairs.

    This function iterates through a PyTorch dataloader containing bulk RNA-seq samples,
    and for each bulk sample:
    1. Validates the donor ID format and uniqueness
    2. Deconvolutes the bulk sample into virtual single cells using the trained models
    3. Retrieves the corresponding real single-cell data for the same donor
    4. Converts both deconvoluted and real cells to latent space representations
    5. Collects sample metadata (donor IDs and class labels)

    The function includes extensive validation to ensure data integrity, checking for:
    - Proper donor ID formatting (should be sample identifiers, not cell types)
    - No duplicate donor IDs within batches
    - Sufficient cell counts for each donor

    Args:
        pipeline: B2SC pipeline object containing trained models and configuration
        dataloader: PyTorch DataLoader yielding (bulk_data, donor_ids) batches
        sc_data: AnnData object containing single-cell reference data
        gmvae: Trained GMVAE model for latent space conversion
        device: PyTorch device for model inference
        random_seed: Base random seed for reproducible sampling
        max_samples: Maximum number of samples to process from this dataloader
        split_name: String identifier for the data split ("Training" or "Validation")

    Returns:
        tuple: (deconv_vectors, real_vectors, labels, class_labels)
            - deconv_vectors: List of deconvoluted latent vectors
            - real_vectors: List of corresponding real SC latent vectors
            - labels: List of donor IDs for each sample pair
            - class_labels: List of class labels (e.g., disease status) for each sample

    Raises:
        AssertionError: If donor ID validation fails or data integrity checks fail
    """
    deconv_vectors = []
    real_vectors = []
    labels = []
    class_labels = []

    count = 0

    for batch_idx, (bulk_data, donor_ids) in enumerate(dataloader):
        if count >= max_samples:
            break

        # ASSERTION: Verify each bulk sample has exactly one unique donor ID
        for donor_id in donor_ids:
            assert isinstance(donor_id, (str, int, float)), (
                f"Donor ID should be a simple identifier (str/int/float), "
                f"but got {type(donor_id)}: {donor_id}"
            )

        # ASSERTION: Verify no duplicate donor IDs in this batch
        unique_donor_ids = list(set(donor_ids))
        assert len(unique_donor_ids) == len(donor_ids), (
            f"Found duplicate donor IDs in batch {batch_idx}: {donor_ids}. "
            f"Each bulk sample should have a unique identifier."
        )

        # Get cell counts for this batch
        cell_counts = get_cell_counts_per_bulk_sample(
            sc_data,
            donor_ids,
            pipeline.config.pseudobulk_groupby,
        )

        for i, donor_id in enumerate(donor_ids):
            if count >= max_samples:
                break

            bulk_sample = bulk_data[i].numpy()
            random_seed_offset = (
                random_seed
                + (1000 if split_name == "Validation" else 0)
                + batch_idx * 100
                + i
            )

            result = process_bulk_sample_pair(
                pipeline,
                bulk_sample,
                donor_id,
                cell_counts,
                gmvae,
                device,
                random_seed_offset,
                sc_data,
                split_name,
                verbose,
            )

            if result is not None:
                deconv_vectors.append(result["deconv_vector"])
                real_vectors.append(result["real_vector"])
                labels.append(result["donor_id"])
                class_labels.append(result["class_label"])
                count += 1

    # FINAL ASSERTION: Verify all collected labels are unique sample identifiers
    if verbose:
        print(
            f"   Collected {len(labels)} unique sample labels for {split_name}: {labels}"
        )

    return deconv_vectors, real_vectors, labels, class_labels


def prepare_data_for_pca(
    train_deconv_vectors,
    train_real_vectors,
    val_deconv_vectors,
    val_real_vectors,
    verbose=False,
):
    """
    Prepare and validate data arrays for PCA processing.

    This function converts lists of latent vectors into numpy arrays and performs
    validation checks to ensure the data is suitable for PCA analysis. It:

    1. Converts input vector lists to numpy arrays
    2. Validates that training data exists (required)
    3. Handles optional validation data (can be None/empty)
    4. Checks dimensional consistency between all datasets
    5. Detects and reports NaN values that could indicate training issues
    6. Provides diagnostic information about data quality

    Args:
        train_deconv_vectors: List of training deconvoluted latent vectors
        train_real_vectors: List of training real SC latent vectors
        val_deconv_vectors: List of validation deconvoluted latent vectors (optional)
        val_real_vectors: List of validation real SC latent vectors (optional)

    Returns:
        tuple: (train_deconv, train_real, val_deconv, val_real) as numpy arrays
               val_deconv and val_real are None if no validation data provided

    Raises:
        ValueError: If no training samples are available
        AssertionError: If latent dimensions don't match between datasets
    """
    import numpy as np

    if not train_deconv_vectors:
        raise ValueError("No training samples could be processed for visualization")

    # Convert to numpy arrays
    train_deconv = np.array(train_deconv_vectors)
    train_real = np.array(train_real_vectors)

    val_deconv = np.array(val_deconv_vectors) if val_deconv_vectors else None
    val_real = np.array(val_real_vectors) if val_real_vectors else None

    if verbose:
        print(f"   Collected {len(train_deconv)} training pairs")
        if val_deconv is not None:
            print(f"   Collected {len(val_deconv)} validation pairs")

    assert (
        train_deconv.shape[1]
        == train_real.shape[1]
        == val_deconv.shape[1]
        == val_real.shape[1]
    ), "Training and validation data latent dimensions must match"

    # Check for NaN values in the collected data
    def check_nan_in_array(arr, name):
        if arr is not None and np.isnan(arr).any():
            nan_count = np.isnan(arr).sum()
            total_elements = arr.size
            print(f"⚠️  NaN values detected in {name}!")
            print(
                f"   Shape: {arr.shape}, NaN count: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%)"
            )
            return True
        return False

    has_nan = False
    has_nan |= check_nan_in_array(train_deconv, "training deconvoluted vectors")
    has_nan |= check_nan_in_array(train_real, "training real vectors")
    has_nan |= check_nan_in_array(val_deconv, "validation deconvoluted vectors")
    has_nan |= check_nan_in_array(val_real, "validation real vectors")

    if has_nan:
        print(
            "🚨 NaN values detected in latent vectors! This indicates issues in GMVAE training."
        )
        print(
            "   The visualization will attempt to handle these NaN values, but you should:"
        )
        print("   1. Check GMVAE training logs for numerical instability")
        print("   2. Consider reducing learning rates or loss weights")
        print("   3. Ensure proper gradient clipping is enabled")

    return train_deconv, train_real, val_deconv, val_real


def fit_umap_and_transform(
    data: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> tuple:
    """
    Fit UMAP on data and transform it.

    Args:
        data: Input data array of shape (n_samples, n_features)
        n_components: Number of UMAP components
        random_state: Random state for reproducibility
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        metric: Distance metric for UMAP

    Returns:
        Tuple of (transformed_data, umap_model)
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "UMAP is required for UMAP dimensionality reduction. "
            "Install it with: pip install umap-learn"
        )

    import numpy as np

    # Check for NaN values in data
    if np.isnan(data).any():
        print("⚠️  NaN values detected in data for UMAP!")
        print(f"   Data shape: {data.shape}")
        nan_count = np.isnan(data).sum()
        total_elements = data.size
        print(
            f"   NaN count: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%)"
        )

        # Replace NaN values with column medians (robust imputation)
        data_clean = data.copy()
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            mask = np.isnan(col_data)

            if mask.any():
                # Get valid (non-NaN) values for this column
                valid_values = col_data[~mask]

                if len(valid_values) > 0:
                    # Use median for robustness (less affected by outliers than mean)
                    col_median = np.median(valid_values)

                    # Check if median is finite
                    if np.isfinite(col_median):
                        fill_value = col_median
                    else:
                        fill_value = 0.0
                else:
                    # All values in column are NaN
                    fill_value = 0.0

                data_clean[mask, col_idx] = fill_value

        # Final check for any remaining non-finite values
        inf_mask = ~np.isfinite(data_clean)
        if inf_mask.any():
            print(
                f"   Warning: Found {inf_mask.sum()} infinite values, replacing with zeros"
            )
            data_clean[inf_mask] = 0.0

        data = data_clean
        print("   ✅ NaN/Inf values replaced successfully")

    # Fit UMAP
    print(f"   Fitting UMAP with {n_components} components...")
    umap_model = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        verbose=False,
    )

    transformed_data = umap_model.fit_transform(data)

    print(f"   ✅ UMAP completed: {data.shape} -> {transformed_data.shape}")

    return transformed_data, umap_model


def apply_pca_transformation_combined(
    train_deconv, train_real, val_deconv, val_real, random_seed, method="PCA"
):
    """Apply 2D dimensionality reduction to the data using combined train+val fitting.

    Args:
        train_deconv: Training deconvoluted data
        train_real: Training real data
        val_deconv: Validation deconvoluted data (optional)
        val_real: Validation real data (optional)
        random_seed: Random seed for reproducibility
        method: Dimensionality reduction method ("PCA" or "UMAP")

    Returns:
        Tuple of (train_deconv_reduced, train_real_reduced, val_deconv_reduced, val_real_reduced, reduction_model)
    """
    import numpy as np

    # Combine ALL data (training + validation) for fitting
    all_data_parts = [train_deconv, train_real]
    if val_deconv is not None and val_real is not None:
        all_data_parts.extend([val_deconv, val_real])

    all_data = np.vstack(all_data_parts)

    # Apply 2D dimensionality reduction: fit on combined train+val data
    if method == "PCA":
        print(
            "   Applying 2D PCA on combined train+val data for dimensionality reduction..."
        )
        all_reduced, _, reduction_model = fit_pca_and_transform(
            all_data, None, n_components=2, random_state=random_seed
        )
    elif method == "UMAP":
        print(
            "   Applying 2D UMAP on combined train+val data for dimensionality reduction..."
        )
        all_reduced, reduction_model = fit_umap_and_transform(
            all_data, n_components=2, random_state=random_seed
        )
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'PCA' or 'UMAP'.")

    # Split the transformed data back to original groups
    current_idx = 0

    # Training deconvoluted
    n_train_deconv = len(train_deconv)
    train_deconv_reduced = all_reduced[current_idx : current_idx + n_train_deconv]
    current_idx += n_train_deconv

    # Training real
    n_train_real = len(train_real)
    train_real_reduced = all_reduced[current_idx : current_idx + n_train_real]
    current_idx += n_train_real

    # Validation data (if available)
    if val_deconv is not None and val_real is not None:
        # Validation deconvoluted
        n_val_deconv = len(val_deconv)
        val_deconv_reduced = all_reduced[current_idx : current_idx + n_val_deconv]
        current_idx += n_val_deconv

        # Validation real
        n_val_real = len(val_real)
        val_real_reduced = all_reduced[current_idx : current_idx + n_val_real]
    else:
        val_deconv_reduced = None
        val_real_reduced = None

    return (
        train_deconv_reduced,
        train_real_reduced,
        val_deconv_reduced,
        val_real_reduced,
        reduction_model,
    )


def create_deconvolution_visualization_plot(
    train_deconv_reduced,
    train_real_reduced,
    val_deconv_reduced,
    val_real_reduced,
    train_labels,
    val_labels,
    train_class_labels,
    val_class_labels,
    reduction_model,
    save_path,
    figsize,
):
    """Create the final visualization plot."""
    print("   Creating 2D comparison plot...")
    plot_deconvolution_comparison_3d(
        train_deconv=train_deconv_reduced,
        train_real=train_real_reduced,
        val_deconv=val_deconv_reduced,
        val_real=val_real_reduced,
        train_labels=train_labels,
        val_labels=val_labels,
        train_class_labels=train_class_labels,
        val_class_labels=val_class_labels,
        pca_model=reduction_model,
        save_path=save_path,
        figsize=figsize,
    )
