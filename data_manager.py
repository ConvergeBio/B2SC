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
from sklearn.cluster import KMeans
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from config import B2SCConfig


class DonorGroupedDataset(Dataset):
    """Dataset that groups cells by donor and returns summed expression per donor."""

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


class DataManager:
    """Handles all data loading, preprocessing, and validation."""

    def __init__(self, config: B2SCConfig):
        self.config = config
        self.sc_data = None
        self.bulk_data = None
        self.gene_list = None
        self.cell_type_mapping = None
        self.color_map = None
        self.cell_type_fractions = None

    def load_single_cell_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and preprocess single-cell data for GMVAE and BulkEncoder training."""
        print(colored("Loading single-cell data...", "yellow"))

        # Load data
        (
            gmvae_dataset,
            be_dataset,
            X_train,
            y_train,
            cell_type_fractions,
            mapping_dict,
            gene_list,
        ) = self._load_data(
            data_dir=self.config.sc_training_data_path,
            barcode_path=self.config.barcode_path,
            generate_pseudo_cells=self.config.generate_pseudo_cells,
            test_samples=self.config.test_samples,
            normalize_data=self.config.normalize_data,
        )

        # Store for later use
        self.gene_list = gene_list
        self.cell_type_mapping = mapping_dict
        self.cell_type_fractions = torch.FloatTensor(cell_type_fractions)
        self.color_map = self._get_colormap()

        # Auto-detect dimensions if not set
        if self.config.input_dim is None:
            self.config.input_dim = len(gene_list)

        # Auto-calculate batch size if not set
        if self.config.batch_size is None:
            self.config.batch_size = len(gmvae_dataset) // 20

        print(f"Loaded {len(gmvae_dataset)} cells with {len(gene_list)} genes")
        print(f"Found {len(mapping_dict)} cell types")
        print(f"Batch size: {self.config.batch_size}")

        # Create data loaders
        gmvae_dataloader = DataLoader(
            gmvae_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        be_dataloader = DataLoader(
            be_dataset, batch_size=3, shuffle=True, drop_last=False
        )

        return gmvae_dataloader, be_dataloader

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

    def _read_adata(self, data_dir: str) -> sc.AnnData:
        """Load .h5ad or 10x directory into AnnData, ensure barcode index."""
        if data_dir.endswith(".h5ad"):
            adata = sc.read_h5ad(data_dir)
        else:
            adata = sc.read_10x_mtx(data_dir, var_names="gene_symbols", cache=True)
        if adata.obs.index.name is None:
            adata.obs.index.name = "barcodes"
        return adata

    def _label_from_csv(
        self, adata: sc.AnnData, barcode_path: str, drop_labels: set = {"Unknown"}
    ) -> sc.AnnData:
        """Merge in real cell-type labels, dropping unwanted ones."""
        df = pd.read_csv(barcode_path, header=None, names=["barcodes", "labels"]).iloc[
            1:
        ]  # skip header row
        df = df[~df["labels"].isin(drop_labels)]
        adata = adata[adata.obs.index.isin(df["barcodes"])]

        adata.obs = adata.obs.reset_index(drop=False, names="barcodes").merge(
            df, on="barcodes", how="left"
        )
        return adata

    def _label_by_clustering(
        self, adata: sc.AnnData, n_clusters: int = 5
    ) -> sc.AnnData:
        """Compute PCA → KMeans clusters → store as string labels."""
        sc.tl.pca(adata, svd_solver="arpack")
        X_pca = adata.obsm["X_pca"]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(X_pca).astype(str)
        adata.obs["labels"] = clusters
        return adata

    def _encode_labels(self, labels: pd.Series) -> Tuple[torch.LongTensor, dict]:
        """Map each unique label (string or digit) to a unique integer, return tensor + reverse mapping."""
        unique = (
            list(labels.cat.categories)
            if hasattr(labels, "cat")
            else sorted(labels.unique())
        )
        label2int = {}
        next_int = 0
        for lab in unique:
            label2int[lab] = int(lab) if lab.isdigit() else next_int
            next_int = max(next_int, label2int[lab] + 1)
        int_series = labels.map(label2int).astype(int)
        return torch.LongTensor(int_series.values), label2int

    def _split_and_package(
        self,
        X: np.ndarray,
        labels: torch.LongTensor,
        train_frac: float = 1.0,
        seed: int = 0,
    ):
        """Random train-only split, return shuffled TensorDataset + raw arrays."""
        torch.manual_seed(seed)
        n = X.shape[0]
        idx = torch.randperm(n)
        cut = int(train_frac * n)
        train_idx = idx[:cut]
        X_train = torch.Tensor(X[train_idx.numpy()])
        y_train = labels[train_idx]
        ds = TensorDataset(X_train, y_train)
        return ds, X_train, y_train

    def _load_data(
        self,
        data_dir: str,
        barcode_path: str = None,
        generate_pseudo_cells: bool = False,
        n_clusters: int = 5,
        train_frac: float = 1.0,
        test_samples: List[str] = [],
        normalize_data: bool = True,
    ):
        """Main data loading function."""
        # Set random seeds for reproducibility
        np.random.seed(4)
        torch.manual_seed(4)

        # 1) I/O
        adata = self._read_adata(data_dir)
        orig_num_obs = len(adata)
        if test_samples:
            # Exclude test samples from training data
            adata = adata[~adata.obs.donor_id.isin(test_samples)]
            new_num_obs = len(adata)
            print(
                colored(
                    f"Excluded {orig_num_obs - new_num_obs} samples from training data, "
                    f"which are {100 * (orig_num_obs - new_num_obs) / orig_num_obs}% of the original data",
                    "yellow",
                )
            )

        if normalize_data:
            print("Normalizing data...")
            sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
            sc.pp.log1p(adata)

        print(f"Sum expression per cell: {adata.X.sum(axis=1)}")
        print(f"Max sum expression per cell: {adata.X.sum(axis=1).max()}")

        # 2) Labeling
        if generate_pseudo_cells:
            adata = self._label_by_clustering(adata, n_clusters=n_clusters)
        else:
            if "cell_type" not in adata.obs.columns and barcode_path is None:
                raise ValueError(
                    "barcode_path is required unless generate_pseudo_cells=True"
                )
            elif "cell_type" in adata.obs.columns:
                adata.obs["labels"] = adata.obs[
                    "cell_type"
                ].cat.remove_unused_categories()
            else:
                adata = self._label_from_csv(adata, barcode_path)

        # Force categorical dtype (keeps ordering stable)
        adata.obs["labels"] = adata.obs["labels"].astype("category")
        assert len(adata.obs.labels.cat.categories) == len(
            np.unique(adata.obs.labels)
        )  # ensure that the number of categories is the same as the number of unique labels
        normalized_cell_type_counts = adata.obs.labels.value_counts(normalize=True)[
            adata.obs.labels.value_counts() > 0
        ]
        print("Cell types: ", normalized_cell_type_counts)

        # 3) Encoding
        labels_tensor, mapping_dict = self._encode_labels(adata.obs["labels"])
        # turn sparse → dense
        X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
        gene_list = adata.var_names.tolist()

        # 4) Split & package
        gmvae_dataset, X_train, y_train = self._split_and_package(
            X, labels_tensor, train_frac
        )
        assert len(normalized_cell_type_counts) == len(np.unique(labels_tensor))

        # compute cell-type (or pseudo-type) fractions on full data
        counts = adata.obs["labels"].value_counts().sort_index()
        cell_type_fractions = counts.values / counts.values.sum()

        be_dataset = DonorGroupedDataset(adata)

        return (
            gmvae_dataset,
            be_dataset,
            X_train,
            y_train,
            cell_type_fractions,
            mapping_dict,
            gene_list,
        )

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
