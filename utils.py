import scanpy as sc
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import anndata
from termcolor import colored

# Set random seed.
np.random.seed(4)
# Set torch seed.
torch.manual_seed(4)

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


def load_data(data_dir, barcode_path):
    if data_dir.endswith(".h5ad"):
        adata = sc.read_h5ad(data_dir)
    else:
        adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

    # Read barcodes_with_labels tsv file.
    barcodes_with_labels = pd.read_csv(barcode_path, sep=',', header=None).iloc[1:]
    barcodes_with_labels.columns = ['barcodes', 'labels']

    # Remove rows with 'unknown' or NaN labels
    barcodes_with_labels = barcodes_with_labels[(barcodes_with_labels['labels'] != 'Unknown')]

    # Cleaned labels after filtering
    labels = barcodes_with_labels['labels'].values

    filtered_barcodes = barcodes_with_labels['barcodes'].values
    adata = adata[adata.obs.index.isin(filtered_barcodes)]

    adata.obs['barcodes'] = adata.obs.index

    adata.obs = adata.obs.reset_index(drop=True)
    adata.obs = adata.obs.merge(barcodes_with_labels, on='barcodes', how='left')
    adata.X = adata.X.toarray()
    adata.obs.index = adata.obs.index.astype(str)

    mapping_dict = get_celltype2strint_dict()
    

    adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict)
    adata.obs['labels'] = adata.obs['labels'].astype('category')
    

    # Assuming adata is your DataFrame and labels is adata.obs['labels']
    labels = adata.obs['labels']
    # import pdb; pdb.set_trace()

    # Create a custom mapping from unique strings to integers
    label_to_int = {}
    max_int_label = -1

    for label in labels.unique():
        if label.isdigit():  # Check if the label is numeric
            int_val = int(label)
            label_to_int[label] = int_val
            max_int_label = max(max_int_label, int_val)
        else:
            # For non-numeric labels, assign a unique integer
            # starting from the next integer after the highest numeric label
            if label not in label_to_int:
                max_int_label += 1
                label_to_int[label] = max_int_label

    # Map the labels in your DataFrame to integers
    int_labels = labels.map(label_to_int)

    # Convert the Series of integers to a LongTensor
    labels = torch.LongTensor(int_labels.values)

    # import pdb; pdb.set_trace()
    # Verify the tensor
    print(labels)
    
    # Plot UMAP with get_colormap().
    # color_map = get_colormap()
    # import umap
    # import matplotlib.pyplot as plt
    # mapping_dict = get_celltype2int_dict()
    # label_map = {v: k for k, v in mapping_dict.items()}

    # reducer = umap.UMAP()
    # embedding = reducer.fit_transform(adata.X)
    # plt.figure(figsize=(12, 10))
    # # import pdb; pdb.set_trace()
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=[color_map[label_map[label.item()]] for label in labels], s=5)
    # # Remove ticks
    # plt.xticks([])
    # plt.yticks([])
    # # Name the axes.
    # plt.xlabel('UMAP1')
    # plt.ylabel('UMAP2')
    # plt.title('UMAP of Input Data')
    # plt.savefig('umap_input.png')
    # plt.close()
    # import pdb; pdb.set_trace()


    X_tensor = torch.Tensor(adata.X)
    # randomly sample 80% of the data.
    rand_index = np.random.choice(X_tensor.shape[0], int(0.8*X_tensor.shape[0]), replace=False)
    # Select those not in the random index to be the test set.
    X_tensor = X_tensor[rand_index]
    # expose gene names for later alignment
    gene_list = adata.var_names.tolist()
    labels = labels[rand_index]
    

    # mask = np.isin(np.arange(X_tensor.shape[0]), rand_index, invert=True)
    # X_tensor = X_tensor[mask]
    # labels = labels[mask]
    print(f"Shape of X_tensor: {X_tensor.shape}")
    print(f"Shape of labels: {labels.shape}")

    dataset = TensorDataset(X_tensor, labels)

    # Cell type fractions
    cell_type_fractions = np.unique(adata.obs['labels'].values, return_counts=True)[1]/len(adata.obs['labels'].values)
    
    return dataset, X_tensor, labels, cell_type_fractions, mapping_dict, gene_list


def get_saved_GMM_params(mus_path, vars_path):
    gmm_mus_celltypes = torch.load(mus_path).squeeze().T
    gmm_vars_celltypes = torch.load(vars_path).squeeze().T
    return gmm_mus_celltypes, gmm_vars_celltypes


def configure(data_dir, barcode_path):
    dataset, X_tensor, cell_types_tensor, cell_type_fractions, mapping_dict, gene_list = load_data(
                                                                                        data_dir=data_dir,
                                                                                        barcode_path=barcode_path,
                                                                                        )
    assert(len(dataset) > 0)
    assert(len(dataset) == X_tensor.shape[0])
    assert(len(np.unique(cell_types_tensor)) > 0)

    unique_cell_type_ids = np.unique(cell_types_tensor)
    print(f"Data contains {len(np.unique(unique_cell_type_ids))} cell types")
    undetected_cell_tyeps = set(unique_cell_type_ids) - set(mapping_dict.keys())
    if undetected_cell_tyeps:
        print(colored(f"Warning: Data contains cell types that are not in the mapping_dict: {undetected_cell_tyeps}", "yellow"))
        

    num_cells = X_tensor.shape[0]
    num_genes = X_tensor.shape[1]

    parser = argparse.ArgumentParser(description='Process neural network parameters.')
    
    args = parser.parse_args()
    args.num_cells = num_cells
    # args.learning_rate = 1e-3
    args.hidden_dim = 600
    args.latent_dim = 300
    args.train_GMVAE_epochs = 3
    args.bulk_encoder_epochs = 3
    # args.dropout = 0.05
    args.batch_size = num_cells
    args.input_dim = num_genes
    
    dataloader = DataLoader(dataset, batch_size=num_cells//5, shuffle=True, drop_last=True)

    args.dataloader = dataloader
    args.cell_types_tensor = cell_types_tensor

    args.mapping_dict = mapping_dict
    args.color_map = get_colormap()
    args.K = 34 # number of cell types
    unique_cell_types = np.unique(cell_types_tensor)
    cell_type_fractions_ = []

    # Create a dictionary mapping from cell type to its fraction
    cell_type_to_fraction = {cell_type: fraction for cell_type, fraction in zip(unique_cell_types, cell_type_fractions)}
    

    for i in range(args.K):
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

def load_bulk_data_h5ad(bulk_h5ad_path, gene_list, batch_size=None):
    """
    Reads a bulk AnnData (.h5ad), aligns to gene_list,
    and returns a DataLoader of shape [n_samples, n_genes].
    
    bulk_h5ad_path: path to your .h5ad (obs = samples, var = genes)
    gene_list:      list of gene names from your sc reference
    batch_size:     int, samples per batch (defaults to all samples)
    """
    # 1) Load the AnnData
    adata_bulk = sc.read_h5ad(bulk_h5ad_path)
    
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
    return DataLoader(ds, batch_size=bs, shuffle=False)