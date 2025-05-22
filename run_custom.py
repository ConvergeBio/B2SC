import os
import torch
from utils import configure, load_bulk_data_h5ad
from main import train_model_GMVAE, train_model_BulkEncoder
from generate import generate  # or generate_bulk if you adapt it
from pathlib import Path

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    BASE_DIR = Path("/home/shared-ssh-key/")
    assert BASE_DIR.exists()

    # ── 1) Configure using your sc reference
    sc_dir     = (BASE_DIR / "convergeSC-Internal/bulk2single/data/adamson_small_sc.h5ad").as_posix()
    barcode_fp = (BASE_DIR / "convergeSC-Internal/bulk2single/data/adamson_small_barcode_2_celltype.csv").as_posix()
    args       = configure(sc_dir, barcode_fp)

    # ── 2) Train GMVAE (on sc data)
    train_model_GMVAE(
        max_epochs        = args.train_GMVAE_epochs,
        dataloader        = args.dataloader,
        proportion_tensor = args.cell_type_fractions,
        mapping_dict      = args.mapping_dict,
        color_map         = args.color_map,
        model_param_tuple = (args.input_dim, args.hidden_dim, args.latent_dim, args.K),
        device            = device
    )

    # ── 3) Train BulkEncoder (on sc data)
    train_model_BulkEncoder(
        max_epochs        = args.bulk_encoder_epochs,
        dataloader        = args.dataloader,
        model_param_tuple = (args.input_dim, args.hidden_dim, args.latent_dim, args.K),
        device            = device,
        train_more        = False
    )

    # ── 4) Load your bulk counts & generate pseudo-single-cells
    #     
    bulk_loader = load_bulk_data_h5ad(
        bulk_h5ad_path="/home/shared-ssh-key/convergeSC-Internal/bulk2single/data/adamson_small_pseudobulk.h5ad",
        gene_list     = args.gene_list,
        batch_size    = None    # or set to e.g. 1 or N
    )

    # The existing `generate` in generate.py expects (bulkEncoder_model, GMVAE_model, loader, ...)
    # If it only takes args.dataloader, you’ll need to adjust it so:
    #    generate(bulk_loader, bulkEncoder_model, GMVAE_model, ...)
    # Here’s how you’d call it if it already supports a bulk loader:
    from models import GaussianMixtureVAE, bulkEncoder
    # reload the two trained models
    input_dim, hidden_dim, latent_dim, K = args.input_dim, args.hidden_dim, args.latent_dim, args.K
    gmvae = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, K)
    be   = bulkEncoder(input_dim, hidden_dim, latent_dim, K)

    gmvae = torch.nn.DataParallel(gmvae).to(device)
    be    = be.to(device)

    gmvae.load_state_dict(torch.load("/home/shared-ssh-key/B2SC/saved_files/adamson_small/GMVAE_model.pt"), strict=True)
    be.load_state_dict(torch.load("/home/shared-ssh-key/B2SC/saved_files/adamson_small/bulkEncoder_model.pt"), strict=True)

    # Finally generate:
    generate(encoder=be, GMVAE_model=gmvae, dataloader=bulk_loader, num_cells=1000, mapping_dict=args.mapping_dict, color_map=args.color_map, device=device)
