import os
import torch
from utils import configure, load_bulk_data_h5ad
from main import train_model_GMVAE, train_model_BulkEncoder
from generate import generate  
from pathlib import Path

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    os.chdir('/home/shared-ssh-key/B2SC')
    barcode_fp = None
    BASE_DIR = Path("../convergeSC-Internal/bulk2single/data/")
    assert BASE_DIR.exists()

    # ── 1) Configure using your sc reference
    # sc_dir     = (BASE_DIR / "convergeSC-Internal/bulk2single/data/SRX20558685.h5ad").as_posix()
    # sc_dir = (BASE_DIR / "convergeSC-Internal/bulk2single/data/mus_musculus_healthy_liver_SRX13549197.h5ad").as_posix()
    sc_dir = (BASE_DIR / "mouse_liver_sc_healthy_5K_highly_variable.h5ad").as_posix()
    # sc_dir = (BASE_DIR / "convergeSC-Internal/bulk2single/data/adamson_small_sc.h5ad").as_posix()
    assert(os.path.exists(sc_dir))
    # barcode_fp = "/home/shared-ssh-key/convergeSC-Internal/bulk2single/data/adamson_small_barcode_2_celltype.csv"
    if barcode_fp is not None:
        assert(os.path.exists(barcode_fp))

    generate_pseudo_cells = False
    test_samples = []#['3_11_M', '3-M-8/9']
    args = configure(sc_dir, barcode_fp, generate_pseudo_cells=generate_pseudo_cells, test_samples=test_samples)

    # ── 2) Train GMVAE (on sc data)
    train_model_GMVAE(
        max_epochs        = args.train_GMVAE_epochs,
        dataloader        = args.gmvae_dataloader,
        proportion_tensor = args.cell_type_fractions,
        mapping_dict      = args.mapping_dict,
        color_map         = args.color_map,
        model_param_tuple = (args.input_dim, args.hidden_dim, args.latent_dim, len(args.cell_type_fractions)),
        gammas            = args.gammas,
        device            = device,
        learning_rate     = args.learning_rate,
        load_pretrained   = False,
        base_dir          = BASE_DIR
    )

    # ── 3) Train BulkEncoder (on sc data)
    train_model_BulkEncoder(
        max_epochs        = args.bulk_encoder_epochs,
        dataloader        = args.be_dataloader,
        model_param_tuple = (args.input_dim, args.hidden_dim, args.latent_dim, len(args.cell_type_fractions)),
        device            = device,
        train_more        = False,
        load_pretrained   = False,
        base_dir          = BASE_DIR
    )

    # ── 4) Load your bulk counts & generate pseudo-single-cells
    #     
    bulk_loader, sample_ids = load_bulk_data_h5ad(
        bulk_h5ad_path="/home/shared-ssh-key/convergeSC-Internal/bulk2single/data/mouse_liver_sc_healthy_5K_highly_variable_pseudobulk.h5ad",
        gene_list     = args.gene_list,
        batch_size    = 1,   # or set to e.g. 1 or N,
        include_sample_id = ['3-M-8/9']#test_samples
    )

    # The existing `generate` in generate.py expects (bulkEncoder_model, GMVAE_model, loader, ...)
    # If it only takes args.dataloader, you’ll need to adjust it so:
    #    generate(bulk_loader, bulkEncoder_model, GMVAE_model, ...)
    # Here’s how you’d call it if it already supports a bulk loader:
    from models import GaussianMixtureVAE, bulkEncoder
    # reload the two trained models
    input_dim, hidden_dim, latent_dim, K = args.input_dim, args.hidden_dim, args.latent_dim, len(args.cell_type_fractions)
    gmvae = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim,  K)
    be   = bulkEncoder(input_dim, hidden_dim, latent_dim, K)

    gmvae = torch.nn.DataParallel(gmvae).to(device)
    be    = be.to(device)

    gmvae.load_state_dict(torch.load(BASE_DIR.as_posix() + "/GMVAE_model.pt"), strict=True)
    be.load_state_dict(torch.load(BASE_DIR.as_posix() + "/bulkEncoder_model.pt"), strict=True)

    # Finally generate:
    generated_tensor_dict, sampled_celltypes = generate(encoder=be, GMVAE_model=gmvae, dataloader=bulk_loader, sample_ids=sample_ids, num_cells=500, mapping_dict=args.mapping_dict, color_map=args.color_map, device=device, base_dir=BASE_DIR)
