"""
Data management for B2SC pipeline.
"""

import scanpy as sc
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import anndata
from termcolor import colored
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from config import B2SCConfig


class DonorGroupedCellsDataset(Dataset):
    """
    Dataset that groups cells by donor and returns the full cell matrix.

    For GMVAE training: Returns all cells from a donor as a matrix (NOT summed).
    This preserves individual cell information for cell-level processing.

    Returns:
        data: Tensor of shape (num_cells_in_donor, n_genes)
        cell_type_labels: Tensor of shape (num_cells_in_donor,) with encoded cell types
        donor_id: String identifier for the donor
    """

    def __init__(self, adata, donor_col: str = "donor_id"):
        self.adata = adata
        self.donor_col = donor_col

        # Unique donors
        self.donor_ids = self.adata.obs[donor_col].unique().tolist()

        # Group indices by donor
        self.grouped_indices = {
            donor: np.where(self.adata.obs[donor_col] == donor)[0]
            for donor in self.donor_ids
        }

        # Validate that cell_type_encoded exists
        if "cell_type_encoded" not in self.adata.obs.columns:
            raise ValueError(
                "cell_type_encoded column not found in adata.obs. "
                "Please encode cell types before creating dataset."
            )

    def __len__(self):
        return len(self.donor_ids)

    def __getitem__(self, idx):
        donor = self.donor_ids[idx]
        indices = self.grouped_indices[donor]

        # Get cell matrix for this donor (NOT summed)
        donor_cell_matrix = self.adata.X[indices]

        # If sparse, convert to dense
        if not isinstance(donor_cell_matrix, np.ndarray):
            donor_cell_matrix = donor_cell_matrix.toarray()

        # Get cell type labels for these cells
        cell_type_labels = self.adata.obs.iloc[indices]["cell_type_encoded"].values

        # Validate dimensions
        assert donor_cell_matrix.shape[0] == len(cell_type_labels), (
            f"Mismatch: cell matrix has {donor_cell_matrix.shape[0]} cells "
            f"but got {len(cell_type_labels)} labels for donor {donor}"
        )

        # Return tensors
        return (
            torch.tensor(
                donor_cell_matrix, dtype=torch.float32
            ),  # (num_cells, n_genes)
            torch.tensor(cell_type_labels, dtype=torch.long),  # (num_cells,)
            donor,  # donor_id as string
        )


class DonorGroupedDataset(Dataset):
    """Dataset that groups cells by donor and returns summed expression per donor (for BulkEncoder)."""

    def __init__(self, adata, donor_col: str = "donor_id"):
        self.adata = adata
        self.donor_col = donor_col
        # Unique donors
        self.donor_ids = self.adata.obs[donor_col].unique().tolist()
        # Group indices by donor
        self.grouped_indices = {
            donor: np.where(self.adata.obs[donor_col] == donor)[0]
            for donor in self.donor_ids
        }

    def __len__(self):
        return len(self.donor_ids)

    def __getitem__(self, idx):
        donor = self.donor_ids[idx]
        indices = self.grouped_indices[donor]
        # Sum across all rows for this donor
        donor_matrix = self.adata.X[indices]
        # If sparse, convert to dense
        if not isinstance(donor_matrix, np.ndarray):
            donor_matrix = donor_matrix.toarray()
        summed = donor_matrix.sum(axis=0)
        # Return torch tensor
        return torch.tensor(summed, dtype=torch.float32), donor


def _gmvae_flatten_collate_fn(batch):
    """
    Custom collate function for GMVAE dataloader.

    Flattens cells from multiple donors into a single batch.
    This allows batch_size to represent number of donors, while processing
    all their cells together.

    Args:
        batch: List of tuples (data, cell_type_labels, donor_id)
               where data.shape = (num_cells_i, n_genes) for donor i

    Returns:
        data: Tensor of shape (total_cells, n_genes) - flattened across donors
        cell_type_labels: Tensor of shape (total_cells,) - flattened
        num_cells_per_donor: Tensor of shape (batch_size,) - cells per donor
        donor_ids: List of donor_id strings
    """
    # Unpack batch
    data_list, labels_list, donor_ids = zip(*batch)

    # Get dimensions
    batch_size = len(data_list)  # Number of donors
    num_cells_per_donor = [d.shape[0] for d in data_list]
    total_cells = sum(num_cells_per_donor)
    n_genes = data_list[0].shape[1]

    # Flatten all cells from all donors
    flattened_data = torch.cat(data_list, dim=0)  # (total_cells, n_genes)
    flattened_labels = torch.cat(labels_list, dim=0)  # (total_cells,)

    # Validation assertions
    assert flattened_data.shape == (total_cells, n_genes), (
        f"Expected flattened data shape ({total_cells}, {n_genes}), "
        f"got {flattened_data.shape}"
    )
    assert flattened_labels.shape == (total_cells,), (
        f"Expected flattened labels shape ({total_cells},), "
        f"got {flattened_labels.shape}"
    )
    assert flattened_data.shape[0] == flattened_labels.shape[0], (
        f"Mismatch: flattened_data has {flattened_data.shape[0]} cells "
        f"but flattened_labels has {flattened_labels.shape[0]} labels"
    )

    num_cells_tensor = torch.tensor(num_cells_per_donor, dtype=torch.long)

    return flattened_data, flattened_labels, num_cells_tensor, list(donor_ids)


class DataManager:
    """Handles all data loading, preprocessing, and validation."""

    def __init__(self, config: B2SCConfig):
        self.config = config

        # Data storage (works for both approaches)
        self.sc_train_data = None
        self.sc_val_data = None
        self.bulk_train_data = None
        self.bulk_val_data = None

        # Common metadata
        self.gene_list = None
        self.cell_type_mapping = None
        self.color_map = None
        self.cell_type_fractions = None

    def load_single_cell_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and preprocess single-cell data for GMVAE and BulkEncoder training."""
        return self._load_sc_based_data()

    def _load_sc_based_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load data using the new approach (sc_training_path + sc_validation_path)."""
        print(colored("Loading single-cell data...", "yellow"))

        # Load training data
        print(f"Loading training data from: {self.config.sc_training_path}")
        self.sc_train_data = sc.read_h5ad(self.config.sc_training_path)

        # Load validation data if provided
        if self.config.sc_validation_path:
            print(f"Loading validation data from: {self.config.sc_validation_path}")
            self.sc_val_data = sc.read_h5ad(self.config.sc_validation_path)

        # Compute pseudobulk data with caching
        self.bulk_train_data = self._compute_pseudobulk(
            self.sc_train_data, dataset_type="train"
        )

        if self.sc_val_data is not None:
            self.bulk_val_data = self._compute_pseudobulk(
                self.sc_val_data, dataset_type="validation"
            )

        # Extract metadata from training data
        self._extract_metadata_from_sc_data(self.sc_train_data)

        # Apply the same cell type encoding to validation data if it exists
        if self.sc_val_data is not None:
            self._apply_cell_type_encoding(self.sc_val_data)

        # Validate that cell_type_encoded exists
        if "cell_type_encoded" not in self.sc_train_data.obs.columns:
            raise ValueError(
                "cell_type_encoded column not found in adata.obs. "
                "Cell types must be encoded before creating dataset."
            )

        # Create datasets
        # GMVAE: Groups cells by donor, then flattens them in collate function
        # batch_size = number of donors to sample per batch
        gmvae_train_dataset = DonorGroupedCellsDataset(
            self.sc_train_data, self.config.pseudobulk_groupby
        )

        # BulkEncoder: Groups cells by donor, returns summed expression
        be_train_dataset = DonorGroupedDataset(
            self.sc_train_data, self.config.pseudobulk_groupby
        )

        # Create dataloaders
        # GMVAE dataloader: batch_size = number of donors per batch
        # The collate function will flatten all cells from those donors
        gmvae_dataloader = DataLoader(
            gmvae_train_dataset,
            batch_size=self.config.gmvae_batch_size,
            shuffle=True,
            collate_fn=_gmvae_flatten_collate_fn,
        )

        be_dataloader = DataLoader(
            be_train_dataset,
            batch_size=self.config.bulk_encoder_batch_size,
            shuffle=True,
        )

        print(colored("✅ Data loading completed", "green"))

        # Log dimensions for verification
        sample_batch = next(iter(gmvae_dataloader))
        data, cell_type_labels, num_cells_per_donor, donor_ids = sample_batch
        print(colored(f"📊 GMVAE Data Dimensions:", "cyan"))
        print(f"   Number of donors in batch: {len(donor_ids)}")
        print(f"   Cells per donor: {num_cells_per_donor.tolist()}")
        print(f"   Total cells in batch: {data.shape[0]}")
        print(f"   Data shape: {data.shape}")  # Should be (total_cells, n_genes)
        print(
            f"   cell_type_labels shape: {cell_type_labels.shape}"
        )  # Should be (total_cells,)

        # Validate dimensions
        total_cells = data.shape[0]
        n_genes = data.shape[1]
        num_donors = len(donor_ids)
        assert (
            data.ndim == 2
        ), f"Expected 2D data (total_cells, n_genes), got {data.ndim}D with shape {data.shape}"
        assert (
            cell_type_labels.ndim == 1
        ), f"Expected 1D labels (total_cells,), got {cell_type_labels.ndim}D with shape {cell_type_labels.shape}"
        assert total_cells == cell_type_labels.shape[0], (
            f"Cell count mismatch: data has {total_cells} cells "
            f"but cell_type_labels has {cell_type_labels.shape[0]} labels"
        )
        assert (
            num_donors == self.config.gmvae_batch_size
            or num_donors <= self.config.gmvae_batch_size
        ), (
            f"Batch size issue: expected {self.config.gmvae_batch_size} donors, "
            f"got {num_donors} (last batch may be smaller)"
        )
        assert total_cells == sum(num_cells_per_donor.tolist()), (
            f"Cell count mismatch: total_cells={total_cells} but "
            f"sum(num_cells_per_donor)={sum(num_cells_per_donor.tolist())}"
        )
        print(colored(f"✅ Dimension validation passed!", "green"))

        return gmvae_dataloader, be_dataloader

    def _compute_pseudobulk(
        self, sc_data: anndata.AnnData, dataset_type: str
    ) -> anndata.AnnData:
        """Compute pseudobulk data from single-cell data with caching."""
        from utilities.helpers import calculate_pseudobulk

        # Generate cache filename
        cache_dir = Path(self.config.base_dir) / "cache"
        cache_dir.mkdir(exist_ok=True)

        # Create a hash of the input data for cache validation
        import hashlib

        data_hash = hashlib.md5(
            f"{self.config.sc_training_path}_{self.config.sc_validation_path}_{self.config.pseudobulk_groupby}_{dataset_type}".encode()
        ).hexdigest()[:8]

        cache_file = cache_dir / f"pseudobulk_{dataset_type}_{data_hash}.h5ad"

        # Try to load from cache if enabled
        if self.config.cache_pseudobulk and cache_file.exists():
            print(f"Loading cached pseudobulk data for {dataset_type}: {cache_file}")
            return sc.read_h5ad(cache_file)

        # Compute pseudobulk
        print(
            f"Computing pseudobulk data for {dataset_type} (groupby: {self.config.pseudobulk_groupby})..."
        )
        pseudobulk_data = calculate_pseudobulk(sc_data, self.config.pseudobulk_groupby)

        # Cache if enabled
        if self.config.cache_pseudobulk:
            print(f"Caching pseudobulk data: {cache_file}")
            pseudobulk_data.write_h5ad(cache_file)

        return pseudobulk_data

    def _extract_metadata_from_sc_data(self, sc_data: anndata.AnnData):
        """Extract metadata from single-cell data."""
        # Get gene list
        self.gene_list = sc_data.var_names.tolist()

        # Create cell type mapping if not exists
        if "cell_type" in sc_data.obs.columns:
            unique_cell_types = sc_data.obs["cell_type"].unique()
            self.cell_type_mapping = {
                ct: i for i, ct in enumerate(sorted(unique_cell_types))
            }

            # Add encoded cell types if not present
            if "cell_type_encoded" not in sc_data.obs.columns:
                sc_data.obs["cell_type_encoded"] = sc_data.obs["cell_type"].map(
                    self.cell_type_mapping
                )

        # Calculate cell type fractions
        if "cell_type_encoded" in sc_data.obs.columns:
            cell_type_counts = (
                sc_data.obs["cell_type_encoded"].value_counts().sort_index()
            )
            total_cells = len(sc_data.obs)
            self.cell_type_fractions = torch.FloatTensor(
                cell_type_counts.values / total_cells
            )

        # Set input dimension
        if self.config.input_dim is None:
            self.config.input_dim = len(self.gene_list)

        # Set color map
        self.color_map = self._get_colormap()

        print(
            f"Extracted metadata: {len(self.gene_list)} genes, {len(self.cell_type_mapping)} cell types"
        )

    def _apply_cell_type_encoding(self, sc_data: anndata.AnnData):
        """Apply cell type encoding to data using the existing cell type mapping."""
        if "cell_type" in sc_data.obs.columns and self.cell_type_mapping is not None:
            # Add encoded cell types using the mapping from training data
            if "cell_type_encoded" not in sc_data.obs.columns:
                sc_data.obs["cell_type_encoded"] = sc_data.obs["cell_type"].map(
                    self.cell_type_mapping
                )
                print(f"Applied cell type encoding to {len(sc_data.obs)} cells")

            # Check for unknown cell types
            unknown_mask = sc_data.obs["cell_type_encoded"].isna()
            if unknown_mask.any():
                unknown_types = sc_data.obs.loc[unknown_mask, "cell_type"].unique()
                print(
                    f"Warning: Found unknown cell types in validation data: {unknown_types}"
                )
                # For now, we'll drop cells with unknown cell types
                sc_data._inplace_subset_obs(~unknown_mask)
                print(f"Removed {unknown_mask.sum()} cells with unknown cell types")

    def get_validation_dataloaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Get validation dataloaders if validation data is available."""
        # Check if validation data is available
        if (
            not hasattr(self.config, "sc_validation_path")
            or not self.config.sc_validation_path
            or self.sc_val_data is None
        ):
            return None, None

        # Create validation datasets
        # GMVAE: Groups cells by donor, then flattens them in collate function
        gmvae_val_dataset = DonorGroupedCellsDataset(
            self.sc_val_data, self.config.pseudobulk_groupby
        )

        # BulkEncoder: Groups cells by donor, returns summed expression
        be_val_dataset = DonorGroupedDataset(
            self.sc_val_data, self.config.pseudobulk_groupby
        )

        # Create validation dataloaders
        # GMVAE: batch_size = number of donors per batch
        gmvae_val_dataloader = DataLoader(
            gmvae_val_dataset,
            batch_size=self.config.gmvae_batch_size,
            shuffle=False,  # No shuffling for validation
            collate_fn=_gmvae_flatten_collate_fn,
        )

        be_val_dataloader = DataLoader(
            be_val_dataset,
            batch_size=self.config.bulk_encoder_batch_size,
            shuffle=False,
        )

        return gmvae_val_dataloader, be_val_dataloader

    def load_bulk_data(
        self, bulk_data_path: str = None
    ) -> Tuple[DataLoader, List[str]]:
        """Load bulk RNA-seq data for generation."""
        if bulk_data_path is None:
            bulk_data_path = self.config.bulk_data_for_deconv_path

        if bulk_data_path is None:
            raise ValueError("No bulk data path provided")

        print(colored(f"Loading bulk data from {bulk_data_path}...", "yellow"))

        bulk_loader, sample_ids = self._load_bulk_data_h5ad(
            bulk_h5ad_path=bulk_data_path,
            gene_list=self.gene_list,
            batch_size=1,
            include_sample_id=None,  # Load all samples by default
        )

        print(f"Loaded {len(sample_ids)} bulk samples")
        return bulk_loader, sample_ids

    def get_data_statistics(self) -> Dict[str, Any]:
        """Return comprehensive data statistics."""
        if self.gene_list is None:
            return {"error": "No data loaded"}

        stats = {
            "num_genes": len(self.gene_list),
            "num_cell_types": (
                len(self.cell_type_mapping) if self.cell_type_mapping else 0
            ),
            "cell_type_fractions": (
                self.cell_type_fractions.tolist()
                if self.cell_type_fractions is not None
                else []
            ),
            "cell_types": (
                list(self.cell_type_mapping.keys()) if self.cell_type_mapping else []
            ),
        }
        return stats

    def validate_data_compatibility(self) -> bool:
        """Ensure SC and bulk data are compatible."""
        if self.gene_list is None:
            print(colored("Warning: No single-cell data loaded", "yellow"))
            return False

        # Additional validation can be added here
        return True

    def _load_bulk_data_h5ad(
        self,
        bulk_h5ad_path: str,
        gene_list: List[str],
        normalize_data: bool = False,
        batch_size: int = None,
        include_sample_id: List[str] = None,
    ) -> Tuple[DataLoader, List[str]]:
        """Load bulk data from h5ad file."""
        # 1) Load the AnnData
        adata_bulk = sc.read_h5ad(bulk_h5ad_path)
        if include_sample_id is not None:
            assert isinstance(include_sample_id, list)
            adata_bulk = adata_bulk[adata_bulk.obs.index.isin(include_sample_id)]
            assert len(adata_bulk) == len(include_sample_id)
            print(f"Loaded {len(adata_bulk)} samples from {bulk_h5ad_path}")

        if normalize_data:
            print("Normalizing sample bulk data ...")
            sc.pp.normalize_total(adata_bulk, target_sum=1e4, inplace=True)
            sc.pp.log1p(adata_bulk)
        else:
            print("Not normalizing sample bulk data")

        # print(f"Sum expression per sample: {adata_bulk.X.sum(axis=1)}")

        # 2) Extract counts matrix (dense)
        X = adata_bulk.X
        if not isinstance(X, (np.ndarray,)):
            X = X.toarray()

        # 3) Build a DataFrame [samples × genes]
        df = pd.DataFrame(
            data=X, index=adata_bulk.obs_names, columns=adata_bulk.var_names
        )

        # 4) Reindex columns to match your reference gene_list (missing → 0)
        df = df.reindex(columns=gene_list).fillna(0)

        # 5) Convert to Tensor [samples × genes]
        data_tensor = torch.Tensor(df.values)
        # dummy labels (not used downstream)
        labels = torch.zeros(data_tensor.size(0), dtype=torch.long)

        # 6) Wrap in a DataLoader
        ds = TensorDataset(data_tensor, labels)
        bs = batch_size or data_tensor.size(0)
        return DataLoader(ds, batch_size=bs, shuffle=False), df.index.tolist()

    def _get_colormap(self) -> Dict[str, str]:
        """Get colormap for cell types."""
        # Default liver colormap
        colormap_dict = {
            "B cell": "red",
            "Kupffer cell": "brown",
            "T cell": "blue",
            "endothelial cell": "darkturquoise",
            "endothelial cell of hepatic sinusoid": "lightseagreen",
            "hepatic stellate cell": "olive",
            "hepatocyte": "gold",
            "mature NK T cell": "magenta",
            "myeloid leukocyte": "darkorange",
            "natural killer cell": "cyan",
        }

        # Extended colormap for more cell types
        extended_colormap = {
            "Naive B cells": "red",
            "Non-classical monocytes": "black",
            "Classical Monocytes": "orange",
            "Natural killer  cells": "cyan",
            "CD8+ NKT-like cells": "pink",
            "Memory CD4+ T cells": "magenta",
            "Naive CD8+ T cells": "blue",
            "Platelets": "yellow",
            "Pre-B cells": "cornflowerblue",
            "Plasmacytoid Dendritic cells": "lime",
            "Effector CD4+ T cells": "grey",
            "Macrophages": "tan",
            "Myeloid Dendritic cells": "green",
            "Effector CD8+ T cells": "brown",
            "Plasma B cells": "purple",
            "Memory B cells": "darkred",
            "Naive CD4+ T cells": "darkblue",
            "Progenitor cells": "darkgreen",
            "γδ-T cells": "darkcyan",
            "Eosinophils": "darkolivegreen",
            "Neutrophils": "darkorchid",
            "Basophils": "darkred",
            "Mast cells": "darkseagreen",
            "Intermediate monocytes": "darkslateblue",
            "Megakaryocyte": "darkslategrey",
            "Endothelial": "darkturquoise",
            "Erythroid-like and erythroid precursor cells": "darkviolet",
            "HSC/MPP cells": "deeppink",
            "Granulocytes": "deepskyblue",
            "ISG expressing immune cells": "dimgray",
            "Cancer cells": "dodgerblue",
            "Memory CD8+ T cells": "darkkhaki",
            "Pro-B cells": "darkorange",
            "Immature B cells": "darkgoldenrod",
        }

        # Use extended colormap if available, otherwise use default
        return extended_colormap
