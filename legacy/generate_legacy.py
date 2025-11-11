import torch
import umap
import matplotlib.pyplot as plt
import numpy as np
import pickle
from termcolor import colored

def split_generated_tensors(generated_aggregate_tensor: torch.Tensor, sampled_celltypes: torch.Tensor, num_samples: int, sample_ids: list[str]) -> dict:
    """
    Splits an interleaved tensor of shape (N * M, K), where M = num_samples,
    into a dictionary mapping each sample index (0 to M-1) to its own tensor of shape (N, K).

    Args:
        generated_aggregate_tensor (torch.Tensor): Tensor of shape (N * M, K),
            where rows are interleaved as:
            [sample_0_row0,
             sample_1_row0,
             ...,
             sample_{M-1}_row0,
             sample_0_row1,
             ...,
             sample_{M-1}_row1,
             ...]
        num_samples (int): Number of samples (M).

    Returns:
        dict: A dictionary with integer keys 0..M-1. Each value is a tensor of shape (N, K)
            containing all rows for that sample in order.

    Raises:
        AssertionError: if input tensor is not 2D, or if the number of rows is not divisible by num_samples,
                        or if any per‐sample slice does not have the same number of rows.
    """
    # Ensure input is a 2D tensor
    assert isinstance(generated_aggregate_tensor, torch.Tensor), \
        f"Expected a torch.Tensor, but got {type(generated_aggregate_tensor)}"
    assert generated_aggregate_tensor.dim() == 2, \
        f"Input tensor must be 2D, but has shape {generated_aggregate_tensor.shape}"

    total_rows, num_cols = generated_aggregate_tensor.shape

    # Check divisibility to infer N
    assert total_rows % num_samples == 0, (
        f"Number of rows ({total_rows}) must be divisible by num_samples ({num_samples})"
    )
    rows_per_sample = total_rows // num_samples  # N

    # Prepare output dict
    generated_tensor_dict = {}
    cell_type_dict = {}
    for sample_idx, sample_name in enumerate(sample_ids):
        # Slice out rows for this sample: start at `sample_idx`, step by `num_samples`
        sample_tensor = generated_aggregate_tensor[sample_idx : total_rows : num_samples, :]
        cell_type_tensor = sampled_celltypes[sample_idx : total_rows : num_samples]
        # Verify that we got exactly N rows for this sample
        assert sample_tensor.shape == (rows_per_sample, num_cols), (
            f"Sample {sample_idx}: expected shape ({rows_per_sample}, {num_cols}), "
            f"but got {sample_tensor.shape}"
        )

        generated_tensor_dict[sample_name] = sample_tensor
        cell_type_dict[sample_name] = cell_type_tensor
    # Final sanity check: dictionary should have exactly M entries
    assert len(generated_tensor_dict) == num_samples, (
        f"Expected {num_samples} entries in the output dict, but got {len(generated_tensor_dict)}"
    )
    assert len(cell_type_dict) == num_samples, (
        f"Expected {num_samples} entries in the output dict, but got {len(cell_type_dict)}"
    )

    return generated_tensor_dict, cell_type_dict

def generate_(encoder, GMVAE_model, dataloader, device='cuda'):
    generated_list = []
    labels_list = []
    # Generate one cell per batch.
    for i, (data, labels) in enumerate(dataloader):
        # print(f"Sample # {i+1}")
        data = data.to(device)

        # Create a vector of summed gene expressions per batch
        bulk_data = data.sum(dim=0)
        bulk_data = bulk_data.unsqueeze(0) 

        # Forward pass
        mus, logvars, pis = encoder(bulk_data)
        mus = mus.squeeze()
        logvars = logvars.squeeze()
        pis = pis.squeeze()

        generated, k = GMVAE_model.module.decode_bulk(mus, logvars, pis)
        # print("Generated cell type: ", k.item())
        generated_list.append(generated)
        labels_list.append(k.item())
    
    generated_tensor = torch.stack(generated_list)
    
    return generated_tensor, labels_list


def generate(encoder, GMVAE_model, dataloader, sample_ids, num_cells, mapping_dict, color_map, device='cuda', base_dir=''):
    encoder.eval()
    GMVAE_model.eval()
    encoder = encoder.to(device)
    GMVAE_model = GMVAE_model.to(device)
        
    generated_aggregate = []
    sampled_celltypes = []
    
    print(f"Generating {num_cells} cells...")

    for i in range(num_cells):
        if (i + 1) % 100 == 0:
            print(f"Generating {i + 1}th cell...")
        
        gt, label = generate_(encoder, GMVAE_model, dataloader, device=device)
        
        # Append gt to generated_aggregate without changing its type
        generated_aggregate.append(gt)
        for l in label:
            sampled_celltypes.append(l)
        
        if ((i + 1) % 500 == 0) or (i == num_cells - 1):
            # Process and save the data every 200 iterations

            # Convert the list of tensors to a single tensor
            generated_aggregate_tensor = torch.stack(generated_aggregate)
            generated_aggregate_tensor = generated_aggregate_tensor.squeeze()
            generated_aggregate_tensor = generated_aggregate_tensor.cpu()
            
            # Convert sampled_celltypes to a tensor
            sampled_celltypes_tensor = torch.LongTensor(sampled_celltypes)
            
            # Reshape generated_aggregate_tensor
            input_dim = generated_aggregate_tensor.shape[-1]
            generated_aggregate_tensor = generated_aggregate_tensor.reshape(-1, input_dim)
            
            # Save tensors
            torch.save(generated_aggregate_tensor, f"{base_dir}/generated_aggregate_tensor.pt")
            torch.save(sampled_celltypes_tensor, f"{base_dir}/sampled_celltypes.pt")
            
            print(f"{i + 1}th generated_aggregate_tensor saved.")


    generated_aggregate_tensor = torch.load(f"{base_dir}/generated_aggregate_tensor.pt")
    sampled_celltypes = torch.load(f"{base_dir}/sampled_celltypes.pt")
    sampled_celltypes = torch.LongTensor(sampled_celltypes)
    
    print("Generated cell type proportion:")
    print(np.unique(sampled_celltypes, return_counts=True)[0])
    print(np.unique(sampled_celltypes, return_counts=True)[1]/len(sampled_celltypes))
    
    # save mapping_dict as pickle:
    with open(f"{base_dir}/mapping_dict.pkl", "wb") as f:
        pickle.dump(mapping_dict, f)
    print(colored(f"mapping_dict saved to {base_dir}/mapping_dict.pkl", "green"))
    
    generated_tensor_dict, cell_type_dict = split_generated_tensors(generated_aggregate_tensor, sampled_celltypes, num_samples=len(dataloader), sample_ids=sample_ids)
    
    # save generated_tensor_dict as pickle:
    with open(f"{base_dir}/generated_tensor_dict.pkl", "wb") as f:
        pickle.dump(generated_tensor_dict, f)
    print(colored(f"generated_tensor_dict saved to {base_dir}/generated_tensor_dict.pkl", "green"))
    
    # save cell_type_dict as pickle:
    with open(f"{base_dir}/cell_type_dict.pkl", "wb") as f:
        pickle.dump(cell_type_dict, f)
    print(colored(f"cell_type_dict saved to {base_dir}/cell_type_dict.pkl", "green"))
    
    return generated_tensor_dict, cell_type_dict


