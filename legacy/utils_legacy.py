import scanpy as sc
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import argparse
import anndata
from termcolor import colored
from sklearn.cluster import KMeans

# Set random seed.
np.random.seed(4)
# Set torch seed.
torch.manual_seed(4)

class DonorGroupedDataset(Dataset):
    def __init__(self, adata, donor_col: str = 'donor_id'):
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


def get_colormap_liver():
    colormap_dict = {
    'B cell': 'red',  # close to 'Naive B cells' / 'Memory B cells'
    'Kupffer cell': 'brown',  # related to macrophages, fits earthy tones
    'T cell': 'blue',  # general color from CD8+/CD4+ T cells
    'endothelial cell': 'darkturquoise',  # from original list
    'endothelial cell of hepatic sinusoid': 'lightseagreen',  # variant of endothelial
    'hepatic stellate cell': 'olive',  # liver-resident cell, earthy tone
    'hepatocyte': 'gold',  # distinctive and bright
    'mature NK T cell': 'magenta',  # hybrid NK-T character, blends NK and T cell colors
    'myeloid leukocyte': 'darkorange',  # aligned with myeloid/monocyte lineage
    'natural killer cell': 'cyan'  # matching 'Natural killer cells'
    }
    return colormap_dict

def get_celltype2int_dict():
    mapping_dict = {
        'Naive B cells': 0, 'Non-classical monocytes': 1, 'Classical Monocytes': 2, 'Natural killer  cells': 3,
        'CD8+ NKT-like cells': 4, 'Memory CD4+ T cells': 5, 'Naive CD8+ T cells': 6, 'Platelets': 7, 'Pre-B cells':8,
        'Plasmacytoid Dendritic cells':9, 'Effector CD4+ T cells':10, 'Macrophages':11, 'Myeloid Dendritic cells':12,
        'Effector CD8+ T cells':13, 'Plasma B cells': 14, 'Memory B cells': 15, "Naive CD4+ T cells": 16,
        'Progenitor cells':17, 'γδ-T cells':18, 'Eosinophils': 19, 'Neutrophils': 20, 'Basophils': 21, 'Mast cells': 22,
        'Intermediate monocytes': 23, 'Megakaryocyte': 24, 'Endothelial': 25, 'Erythroid-like and erythroid precursor cells': 26,
        'HSC/MPP cells': 27, 'Granulocytes': 28, 'ISG expressing immune cells': 29, 'Cancer cells': 30, "Memory CD8+ T cells": 31,
        "Pro-B cells": 32, "Immature B cells": 33
    }
    return mapping_dict


def get_celltype2strint_dict():
    mapping_dict = {
        'Naive B cells': '0', 'Non-classical monocytes': '1', 'Classical Monocytes': '2', 'Natural killer  cells': '3',
        'CD8+ NKT-like cells': '4', 'Memory CD4+ T cells': '5', 'Naive CD8+ T cells': '6', 'Platelets': '7', 'Pre-B cells': '8',
        'Plasmacytoid Dendritic cells': '9', 'Effector CD4+ T cells': '10', 'Macrophages': '11', 'Myeloid Dendritic cells': '12',
        'Effector CD8+ T cells': '13', 'Plasma B cells': '14', 'Memory B cells': '15', "Naive CD4+ T cells": "16",
        'Progenitor cells':'17', 'γδ-T cells':'18', 'Eosinophils': '19', 'Neutrophils': '20', 'Basophils': '21', 'Mast cells': '22',
        'Intermediate monocytes': '23', 'Megakaryocyte': '24', 'Endothelial': '25', 'Erythroid-like and erythroid precursor cells': '26',
        'HSC/MPP cells': '27', 'Granulocytes': '28', 'ISG expressing immune cells': '29', 'Cancer cells': '30', "Memory CD8+ T cells": "31",
        "Pro-B cells": "32", "Immature B cells": "33"
        }
    return mapping_dict


def get_colormap():
    color_map = {
        'Naive B cells': 'red', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'orange', 'Natural killer  cells': 'cyan',
        'CD8+ NKT-like cells': 'pink', 'Memory CD4+ T cells': 'magenta', 'Naive CD8+ T cells': 'blue', 'Platelets': 'yellow', 'Pre-B cells':'cornflowerblue',
        'Plasmacytoid Dendritic cells':'lime', 'Effector CD4+ T cells':'grey', 'Macrophages':'tan', 'Myeloid Dendritic cells':'green',
        'Effector CD8+ T cells':'brown', 'Plasma B cells': 'purple', "Memory B cells": "darkred", "Naive CD4+ T cells": "darkblue",
        'Progenitor cells':'darkgreen', 'γδ-T cells':'darkcyan', 'Eosinophils': 'darkolivegreen', 'Neutrophils': 'darkorchid', 'Basophils': 'darkred',
        'Mast cells': 'darkseagreen', 'Intermediate monocytes': 'darkslateblue', 'Megakaryocyte': 'darkslategrey', 'Endothelial': 'darkturquoise',
        'Erythroid-like and erythroid precursor cells': 'darkviolet', 'HSC/MPP cells': 'deeppink', 'Granulocytes': 'deepskyblue',
        'ISG expressing immune cells': 'dimgray', 'Cancer cells': 'dodgerblue', 'Memory CD8+ T cells': 'darkkhaki', 'Pro-B cells': 'darkorange',
        'Immature B cells': 'darkgoldenrod'
        # 'CD4+ NKT-like cells': 'darkmagenta',
    }
    return color_map


import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import TensorDataset

def read_adata(data_dir: str) -> sc.AnnData:
    """Load .h5ad or 10x directory into AnnData, ensure barcode index."""
    if data_dir.endswith(".h5ad"):
        adata = sc.read_h5ad(data_dir)
    else:
        adata = sc.read_10x_mtx(data_dir, var_names="gene_symbols", cache=True)
    if adata.obs.index.name is None:
        adata.obs.index.name = "barcodes"
    return adata

def label_from_csv(adata: sc.AnnData, barcode_path: str,
                   drop_labels: set = {"Unknown"}) -> sc.AnnData:
    """Merge in real cell-type labels, dropping unwanted ones."""
    df = (
        pd.read_csv(barcode_path, header=None, names=["barcodes","labels"])
          .iloc[1:]  # skip header row
    )
    df = df[~df["labels"].isin(drop_labels)]
    adata = adata[adata.obs.index.isin(df["barcodes"])]

    adata.obs = (
        adata.obs.reset_index(drop=False, names="barcodes")
                 .merge(df, on="barcodes", how="left")
    )
    return adata

def label_by_clustering(adata: sc.AnnData, n_clusters: int=5) -> sc.AnnData:
    """Compute PCA → KMeans clusters → store as string labels."""
    sc.tl.pca(adata, svd_solver="arpack")
    X_pca = adata.obsm["X_pca"]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(X_pca).astype(str)
    adata.obs["labels"] = clusters
    return adata

def encode_labels(labels: pd.Series) -> (torch.LongTensor, dict):
    """
    Map each unique label (string or digit) to a unique integer,
    return tensor + reverse mapping.
    """
    unique = list(labels.cat.categories) if hasattr(labels, "cat") else sorted(labels.unique())
    label2int = {}
    next_int = 0
    for lab in unique:
        label2int[lab] = int(lab) if lab.isdigit() else next_int
        next_int = max(next_int, label2int[lab] + 1)
    int_series = labels.map(label2int).astype(int)
    return torch.LongTensor(int_series.values), label2int

def split_and_package(X: np.ndarray, labels: torch.LongTensor,
                      train_frac: float=1.0, seed: int=0):
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

def load_data(
    data_dir: str,
    barcode_path: str = None,
    generate_pseudo_cells: bool = False,
    n_clusters: int = 5,
    train_frac: float = 1.0, 
    test_samples: list[str] = [],
    normalize_data : bool = True
):
    # 1) I/O
    adata = read_adata(data_dir)
    orig_num_obs = len(adata)
    if test_samples:
        # Exclude test samples from training data
        adata = adata[~adata.obs.donor_id.isin(test_samples)]
        new_num_obs = len(adata)
        print(colored(f"Excluded {orig_num_obs - new_num_obs} samples from training data, which are {100 * (orig_num_obs - new_num_obs) / orig_num_obs}% of the original data", "yellow"))

    if normalize_data:
        print("Normalizing data...")
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata)

    print(f"Sum expression per cell: {adata.X.sum(axis=1)}")
    print(f"Sum expression per cell: {adata.X.sum(axis=1).max()}")
    # 2) Labeling
    if generate_pseudo_cells:
        adata = label_by_clustering(adata, n_clusters=n_clusters)
    else:
        if 'cell_type' not in adata.obs.columns and barcode_path is None:
            raise ValueError("barcode_path is required unless generate_pseudo_cells=True")
        elif 'cell_type' in adata.obs.columns:
            adata.obs["labels"] = adata.obs["cell_type"].cat.remove_unused_categories()
        else:
            adata = label_from_csv(adata, barcode_path)

    # Force categorical dtype (keeps ordering stable)

    adata.obs["labels"] = adata.obs["labels"].astype("category")
    assert(len(adata.obs.labels.cat.categories) == len(np.unique(adata.obs.labels))) # ensure that the number of categories is the same as the number of unique labels
    normalized_cell_type_counts = adata.obs.labels.value_counts(normalize=True)[adata.obs.labels.value_counts() > 0]
    print("Cell types: ", normalized_cell_type_counts)
    # 3) Encoding
    labels_tensor, mapping_dict = encode_labels(adata.obs["labels"])
    # turn sparse → dense
    X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    gene_list = adata.var_names.tolist()

    # 4) Split & package
    gmvae_dataset, X_train, y_train = split_and_package(X, labels_tensor, train_frac)
    assert(len(normalized_cell_type_counts) == len(np.unique(labels_tensor)))
    # compute cell-type (or pseudo-type) fractions on full data
    counts = adata.obs["labels"].value_counts().sort_index()
    cell_type_fractions = counts.values / counts.values.sum()
    # cell_type_fractions_non_zero = cell_type_fractions[cell_type_fractions > 0]
    # print(f"Cell type fractions: {cell_type_fractions_non_zero}")
    be_dataset = DonorGroupedDataset(adata)


    return gmvae_dataset, be_dataset, X_train, y_train, cell_type_fractions, mapping_dict, gene_list



def get_saved_GMM_params(mus_path, vars_path):
    gmm_mus_celltypes = torch.load(mus_path).squeeze().T
    gmm_vars_celltypes = torch.load(vars_path).squeeze().T
    return gmm_mus_celltypes, gmm_vars_celltypes


def configure(data_dir, barcode_path, generate_pseudo_cells=False, test_samples=None):
    gmvae_dataset, be_dataset, X_tensor, cell_types_tensor, cell_type_fractions, mapping_dict, gene_list = load_data(
                                                                                        data_dir=data_dir,
                                                                                        barcode_path=barcode_path,
                                                                                        generate_pseudo_cells=generate_pseudo_cells,
                                                                                        test_samples=test_samples
                                                                                        )
    assert(len(gmvae_dataset) > 0)
    assert(len(gmvae_dataset) == X_tensor.shape[0])
    assert(len(np.unique(cell_types_tensor)) > 0)

    unique_cell_type_ids = np.unique(cell_types_tensor)
    print(f"Data contains {len(np.unique(unique_cell_type_ids))} cell types")
    undetected_cell_tyeps = set(unique_cell_type_ids) - set(mapping_dict.values())
    if undetected_cell_tyeps:
        print(colored(f"Warning: Data contains cell types that are not in the mapping_dict: {undetected_cell_tyeps}", "yellow"))
        

    num_cells = X_tensor.shape[0]
    num_genes = X_tensor.shape[1]

    parser = argparse.ArgumentParser(description='Process neural network parameters.')
    
    args = parser.parse_args()
    args.num_cells = num_cells
    args.learning_rate = 5e-4
    args.hidden_dim = 2048
    args.latent_dim = 1024
    args.train_GMVAE_epochs = 30
    args.bulk_encoder_epochs = 100
    # args.dropout = 0.05
    args.batch_size = num_cells//20
    args.input_dim = num_genes
    print(f"Batch size: {args.batch_size}")
    
    gmvae_dataloader = DataLoader(gmvae_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    be_dataloader = DataLoader(be_dataset, batch_size=3, shuffle=True, drop_last=False)

    args.gmvae_dataloader = gmvae_dataloader
    args.be_dataloader = be_dataloader
    args.cell_types_tensor = cell_types_tensor

    args.mapping_dict = mapping_dict

    args.unique_cell_types = np.unique(cell_types_tensor)
    relevant_cell_types = [k for k, v in args.mapping_dict.items() if v in args.unique_cell_types]
    
    cell_type_fractions_ = []

    # Create a dictionary mapping from cell type to its fraction
    cell_type_to_fraction = {cell_type: cell_type_fractions[cell_type] for cell_type in args.unique_cell_types}

    args.color_map = get_colormap_liver()
    assert(len(args.color_map) == len(args.unique_cell_types))
    # args.K = 34 # number of cell types

    
    for i in range(max(cell_type_to_fraction.keys())+1):
        # Append the fraction if the cell type is present, else append 0
        cell_type_fractions_.append(cell_type_to_fraction.get(i, 0))
    
    args.cell_type_fractions = torch.FloatTensor(np.array(cell_type_fractions_))
    print(args.cell_type_fractions)
    print("@@")
    
    args.X_tensor = X_tensor
    label_map = {str(v): k for k, v in mapping_dict.items()}
    args.label_map = label_map
    args.gene_list = gene_list

    print('Configuration is complete.')
    return args

# def load_bulk_data(bulk_csv_path, gene_list, batch_size=None):
#     """
#     Reads a bulk counts CSV (genes × samples), aligns to gene_list,
#     returns a DataLoader of shape [n_samples, n_genes].
#     """

#     # 1) Read, index by gene
#     df = pd.read_csv(bulk_csv_path, index_col=0)
#     # 2) Re-order (and fill missing genes with zero)
#     df = df.reindex(gene_list).fillna(0)
#     # 3) Make Tensor [samples × genes]
#     data = torch.Tensor(df.values.T)
#     # 4) Dummy labels (not used in generation)
#     labels = torch.zeros(data.size(0), dtype=torch.long)
#     ds = TensorDataset(data, labels)
#     bs = batch_size or data.size(0)
#     return DataLoader(ds, batch_size=bs, shuffle=False)

def make_pseudo_bulk_adata(
    sc_h5ad_path: str,
    groupby: str,
    out_h5ad: str = None
    ):
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
    if not isinstance(X, (np.ndarray, )):
        X = X.toarray()

    # 3) build a cell×gene DataFrame
    df = pd.DataFrame(
        data    = X,
        index   = adata.obs_names,
        columns = adata.var_names
    )

    # 4) attach the grouping vector
    if groupby not in adata.obs:
        raise KeyError(f"'{groupby}' not in adata.obs")
    df[groupby] = adata.obs[groupby].values

    # 5) sum counts per group → DataFrame (n_groups × n_genes)
    bulk_counts = df.groupby(groupby).sum()

    # 6) construct the AnnData
    bulk_adata = anndata.AnnData(
        X   = bulk_counts.values,
        obs = pd.DataFrame(index=bulk_counts.index),
        var = pd.DataFrame(index=bulk_counts.columns)
    )

    # optional: write to disk
    if out_h5ad:
        bulk_adata.write(out_h5ad)

    return bulk_adata

def load_bulk_data_h5ad(bulk_h5ad_path, 
                        gene_list,
                        normalize_data = False,
                        batch_size=None,
                        include_sample_id : list[str]= None):
    """
    Reads a bulk AnnData (.h5ad), aligns to gene_list,
    and returns a DataLoader of shape [n_samples, n_genes].
    
    bulk_h5ad_path: path to your .h5ad (obs = samples, var = genes)
    gene_list:      list of gene names from your sc reference
    batch_size:     int, samples per batch (defaults to all samples)
    """
    # 1) Load the AnnData
    adata_bulk = sc.read_h5ad(bulk_h5ad_path)
    if include_sample_id is not None:
        assert(isinstance(include_sample_id, list))
        adata_bulk = adata_bulk[adata_bulk.obs.index.isin(include_sample_id)]
        assert(len(adata_bulk) == len(include_sample_id))
        print(f"Loaded {len(adata_bulk)} samples from {bulk_h5ad_path}")
    if normalize_data:
        print("Normalizing sample bulk data ...")
        sc.pp.normalize_total(adata_bulk, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata_bulk)
    
    else:
        print("Not normalizing sample bulk data")

    print(f"Sum expression per sample: {adata_bulk.X.sum(axis=1)}")
    # 2) Extract counts matrix (dense)
    X = adata_bulk.X
    if not isinstance(X, (np.ndarray, )):
        X = X.toarray()
    
    # 3) Build a DataFrame [samples × genes]
    df = pd.DataFrame(
        data    = X,
        index   = adata_bulk.obs_names,
        columns = adata_bulk.var_names
    )
    
    # 4) Reindex columns to match your reference gene_list (missing → 0)
    df = df.reindex(columns=gene_list).fillna(0)
    
    # 5) Convert to Tensor [samples × genes]
    data_tensor = torch.Tensor(df.values)
    # dummy labels (not used downstream)
    labels      = torch.zeros(data_tensor.size(0), dtype=torch.long)
    
    # 6) Wrap in a DataLoader
    ds = TensorDataset(data_tensor, labels)
    bs = batch_size or data_tensor.size(0)
    return DataLoader(ds, batch_size=bs, shuffle=False), df.index.tolist()