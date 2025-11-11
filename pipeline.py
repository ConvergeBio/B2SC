"""
Main B2SC pipeline orchestrating the entire workflow.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from termcolor import colored

from config import B2SCConfig
from data_manager import DataManager
from gmvae_trainer import GMVAETrainer
from bulk_encoder_trainer import BulkEncoderTrainer
from results import GenerationResults, split_generated_tensors
from metrics import MetricsTracker
from models import GaussianMixtureVAE, BulkEncoder
from utilities import set_random_seeds


class B2SCPipeline:
    """Main pipeline orchestrating the entire B2SC workflow including generation."""

    def __init__(self, config: B2SCConfig):
        """
        Initialize the B2SC pipeline.

        Args:
            config: B2SC configuration object
        """
        self.config = config

        # Set random seeds for reproducibility
        set_random_seeds(42)

        self.data_manager = DataManager(config)
        self.gmvae_trainer = GMVAETrainer(config, self.data_manager)
        self.bulk_encoder_trainer = BulkEncoderTrainer(
            config, self.gmvae_trainer, self.data_manager
        )
        self.logger = self._setup_logging()

        # Pipeline state
        self._gmvae_trained = False
        self._bulk_encoder_trained = False
        self._data_loaded = False

        print(colored("B2SC Pipeline initialized", "green"))

    @property
    def gmvae_trained(self) -> bool:
        """Check if GMVAE model has been trained."""
        return self._gmvae_trained

    @property
    def bulk_encoder_trained(self) -> bool:
        """Check if BulkEncoder model has been trained."""
        return self._bulk_encoder_trained

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logger = logging.getLogger("B2SCPipeline")
        logger.setLevel(logging.INFO)

        # Create console handler if not already exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def train_gmvae(self) -> MetricsTracker:
        """Train the GMVAE model."""
        self.logger.info("Starting GMVAE training...")

        # Load data if not already loaded
        if not self._data_loaded:
            self._load_training_data()

        # Initialize GMVAE model
        n_components = len(self.data_manager.cell_type_fractions)
        self.gmvae_trainer.initialize_model(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.gmvae_hidden_dim,
            latent_dim=self.config.latent_dim,
            n_components=n_components,
        )

        # Get validation dataloaders if available
        gmvae_val_dataloader, _ = self.data_manager.get_validation_dataloaders()

        # Train the model
        metrics_tracker = self.gmvae_trainer.train(
            dataloader=self.gmvae_dataloader,
            proportion_tensor=self.data_manager.cell_type_fractions,
            mapping_dict=self.data_manager.cell_type_mapping,
            color_map=self.data_manager.color_map,
            val_dataloader=gmvae_val_dataloader,
        )

        self._gmvae_trained = True
        self.logger.info("GMVAE training completed successfully")
        return metrics_tracker

    def train_bulk_encoder(self) -> MetricsTracker:
        """Train the BulkEncoder model."""
        if not self._gmvae_trained:
            raise ValueError(
                "GMVAE must be trained before training BulkEncoder. Call train_gmvae() first."
            )

        self.logger.info("Starting BulkEncoder training...")

        # Load data if not already loaded
        if not self._data_loaded:
            self._load_training_data()

        # Initialize BulkEncoder model
        n_components = len(self.data_manager.cell_type_fractions)
        self.bulk_encoder_trainer.initialize_model(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.bulk_hidden_dim,
            latent_dim=self.config.latent_dim,
            n_components=n_components,
        )

        # Get validation dataloaders if available
        _, be_val_dataloader = self.data_manager.get_validation_dataloaders()

        # Train the model
        metrics_tracker = self.bulk_encoder_trainer.train(
            dataloader=self.be_dataloader, val_dataloader=be_val_dataloader
        )

        self._bulk_encoder_trained = True
        self.logger.info("BulkEncoder training completed successfully")
        return metrics_tracker

    def bulk_to_latent(
        self,
        bulk_sample,
        num_samples: int = 100,
        random_seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert bulk sample to latent space embeddings (Z matrix) with cell type information.

        Args:
            bulk_sample: Bulk RNA-seq data as numpy array, tensor, or list [n_genes] or [1, n_genes]
            num_samples: Number of virtual cells to sample from the bulk
            random_seed: Random seed for reproducible sampling

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Z matrix of shape [num_samples, latent_dim]
                - Cell type indices of shape [num_samples]

        Raises:
            ValueError: If models are not trained or input format is invalid
        """
        # Validate that both models are trained
        if not self._gmvae_trained:
            raise ValueError(
                "GMVAE must be trained before using bulk_to_latent(). Call train_gmvae() first."
            )

        if not self._bulk_encoder_trained:
            raise ValueError(
                "BulkEncoder must be trained before using bulk_to_latent(). Call train_bulk_encoder() first."
            )

        # Set random seed if provided
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # Convert input to tensor and ensure correct shape
        if isinstance(bulk_sample, (list, np.ndarray)):
            bulk_tensor = torch.FloatTensor(bulk_sample)
        elif isinstance(bulk_sample, torch.Tensor):
            bulk_tensor = bulk_sample.float()
        else:
            raise ValueError(
                f"bulk_sample must be list, numpy array, or torch tensor, got {type(bulk_sample)}"
            )

        # Ensure 2D tensor [1, n_genes]
        if bulk_tensor.dim() == 1:
            bulk_tensor = bulk_tensor.unsqueeze(0)
        elif bulk_tensor.dim() != 2 or bulk_tensor.shape[0] != 1:
            raise ValueError(
                f"bulk_sample must have shape [n_genes] or [1, n_genes], got {bulk_tensor.shape}"
            )

        # Validate input dimension
        if bulk_tensor.shape[1] != self.config.input_dim:
            raise ValueError(
                f"bulk_sample has {bulk_tensor.shape[1]} genes, expected {self.config.input_dim}"
            )

        # Move to device
        device = next(self.bulk_encoder_trainer.model.parameters()).device
        bulk_tensor = bulk_tensor.to(device)

        # Step 1: Get mixture parameters from BulkEncoder
        self.bulk_encoder_trainer.model.eval()
        with torch.no_grad():
            mus, logvars, pis = self.bulk_encoder_trainer.model(bulk_tensor)
            # Remove batch dimension: [1, n_components, latent_dim] -> [n_components, latent_dim]
            mus = mus.squeeze(0)
            logvars = logvars.squeeze(0)
            pis = pis.squeeze(0)

            # Check for NaN/Inf in BulkEncoder outputs
            if torch.isnan(mus).any() or torch.isinf(mus).any():
                print(
                    f"⚠️  NaN/Inf detected in BulkEncoder mus: {torch.isnan(mus).sum().item()}/{mus.numel()} NaN, {torch.isinf(mus).sum().item()}/{mus.numel()} Inf"
                )
                print(f"   mus range: [{mus.min().item():.6f}, {mus.max().item():.6f}]")

            if torch.isnan(logvars).any() or torch.isinf(logvars).any():
                print(
                    f"⚠️  NaN/Inf detected in BulkEncoder logvars: {torch.isnan(logvars).sum().item()}/{logvars.numel()} NaN, {torch.isinf(logvars).sum().item()}/{logvars.numel()} Inf"
                )
                print(
                    f"   logvars range: [{logvars.min().item():.6f}, {logvars.max().item():.6f}]"
                )

            if torch.isnan(pis).any() or torch.isinf(pis).any():
                print(
                    f"⚠️  NaN/Inf detected in BulkEncoder pis: {torch.isnan(pis).sum().item()}/{pis.numel()} NaN, {torch.isinf(pis).sum().item()}/{pis.numel()} Inf"
                )
                print(
                    f"   pis range: [{pis.min().item():.6f}, {pis.max().item():.6f}], sum: {pis.sum().item():.6f}"
                )

            # Clamp extreme values to prevent numerical instability
            mus = torch.clamp(mus, min=-50, max=50)
            logvars = torch.clamp(
                logvars, min=-20, max=10
            )  # Prevent exp(logvars) from exploding
            pis = torch.clamp(pis, min=1e-8, max=1.0)
            pis = pis / pis.sum()  # Renormalize to ensure sum=1

        # Step 2: Sample Z vectors from GMVAE using mixture parameters
        self.gmvae_trainer.model.eval()
        Z_list = []
        cell_type_list = []

        with torch.no_grad():
            for i in range(num_samples):
                # Sample one Z vector and get the selected cell type
                z, k = self.gmvae_trainer.model.reparameterize_with_proportion(
                    mus, logvars, pis
                )

                # Check for NaN/Inf in sampled latent vector
                if torch.isnan(z).any() or torch.isinf(z).any():
                    print(
                        f"⚠️  NaN/Inf detected in sampled latent vector {i}: {torch.isnan(z).sum().item()}/{z.numel()} NaN, {torch.isinf(z).sum().item()}/{z.numel()} Inf"
                    )
                    print(f"   Selected component k: {k.item()}")
                    print(
                        f"   Selected mus: {mus[k].min().item():.6f} to {mus[k].max().item():.6f}"
                    )
                    print(
                        f"   Selected logvars: {logvars[k].min().item():.6f} to {logvars[k].max().item():.6f}"
                    )

                    # Replace NaN/Inf values with zeros as fallback
                    z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))

                Z_list.append(z)
                cell_type_list.append(k)

        # Stack into matrices
        Z_matrix = torch.stack(Z_list)  # Shape: [num_samples, latent_dim]
        cell_types = torch.stack(cell_type_list)  # Shape: [num_samples]

        # Final check for NaN/Inf in the complete Z matrix
        if torch.isnan(Z_matrix).any() or torch.isinf(Z_matrix).any():
            nan_count = torch.isnan(Z_matrix).sum().item()
            inf_count = torch.isinf(Z_matrix).sum().item()
            total_elements = Z_matrix.numel()
            print(
                f"🚨 Final Z_matrix contains {nan_count}/{total_elements} NaN and {inf_count}/{total_elements} Inf values!"
            )
            print(f"   Z_matrix shape: {Z_matrix.shape}")
            print(
                f"   Z_matrix range: [{Z_matrix[torch.isfinite(Z_matrix)].min().item():.6f}, {Z_matrix[torch.isfinite(Z_matrix)].max().item():.6f}]"
            )

        # Print cell type distribution as proportions
        unique_types, counts = torch.unique(cell_types, return_counts=True)
        proportions = counts.float() / len(cell_types)
        print(
            f"Cell type distribution (proportions): {dict(zip(unique_types.cpu().numpy(), proportions.cpu().numpy()))}"
        )

        return Z_matrix, cell_types

    def generate_cells(
        self,
        bulk_data_path: str = None,
        sample_ids: List[str] = None,
        num_cells: int = 500,
    ) -> GenerationResults:
        """
        Generate single cells from bulk samples.

        Args:
            bulk_data_path: Path to bulk data (optional, uses config if None)
            sample_ids: List of sample identifiers to generate from (optional, uses all if None)
            num_cells: Number of cells to generate per sample (default: 500)

        Returns:
            GenerationResults: Complete generation results
        """
        if not self._gmvae_trained or not self._bulk_encoder_trained:
            raise ValueError(
                "Both GMVAE and BulkEncoder must be trained before generation. Call train_gmvae() and train_bulk_encoder() first."
            )

        self.logger.info("Starting cell generation...")

        # Load bulk data
        bulk_loader, all_sample_ids = self.data_manager.load_bulk_data(bulk_data_path)

        # Filter sample IDs if specified
        if sample_ids is not None:
            # Validate that requested sample IDs exist
            missing_samples = set(sample_ids) - set(all_sample_ids)
            if missing_samples:
                raise ValueError(
                    f"Requested sample IDs not found in bulk data: {missing_samples}"
                )
            # Filter the dataloader (this is a simplification - in practice might need to recreate dataloader)
            self.logger.info(
                f"Generating cells for {len(sample_ids)} specified samples"
            )
        else:
            sample_ids = all_sample_ids
            self.logger.info(f"Generating cells for all {len(sample_ids)} samples")

        # Load trained models
        gmvae_model = self._load_trained_gmvae()
        bulk_encoder_model = self._load_trained_bulk_encoder()

        # Generate cells
        generated_tensor_dict, cell_type_dict = self._generate_cells_internal(
            encoder=bulk_encoder_model,
            gmvae_model=gmvae_model,
            dataloader=bulk_loader,
            sample_ids=sample_ids,
            num_cells=num_cells,
        )

        # Create generation stats
        generation_stats = {
            "num_cells_per_sample": num_cells,
            "total_samples": len(sample_ids),
            "total_cells_generated": sum(
                tensor.shape[0] for tensor in generated_tensor_dict.values()
            ),
            "bulk_data_path": bulk_data_path or self.config.bulk_data_for_deconv_path,
        }

        # Create results object
        results = GenerationResults(
            generated_cells=generated_tensor_dict,
            cell_types=cell_type_dict,
            generation_stats=generation_stats,
            mapping_dict=self.data_manager.cell_type_mapping,
            color_map=self.data_manager.color_map,
        )

        self.logger.info(
            f"Cell generation completed. Generated {generation_stats['total_cells_generated']} cells for {len(sample_ids)} samples"
        )
        return results

    def validate_generation_quality(
        self, results: GenerationResults
    ) -> Dict[str, float]:
        """Validate quality of generated cells."""
        self.logger.info("Validating generation quality...")

        quality_metrics = results.validate_quality()

        # Log key quality metrics
        self.logger.info(f"Quality validation results:")
        self.logger.info(
            f"  Total cells generated: {quality_metrics['total_cells_generated']}"
        )
        self.logger.info(
            f"  Unique cell types: {quality_metrics['num_unique_cell_types']}"
        )
        self.logger.info(
            f"  Samples with negative values: {quality_metrics['samples_with_negative_values']}"
        )
        self.logger.info(
            f"  Mean expression per cell: {quality_metrics['mean_expression_per_cell']:.2f}"
        )

        return quality_metrics

    def run_full_pipeline(
        self, bulk_data_path: str = None
    ) -> Tuple[MetricsTracker, MetricsTracker, GenerationResults]:
        """Run the complete B2SC pipeline: train GMVAE, train BulkEncoder, generate cells."""
        self.logger.info("Starting full B2SC pipeline...")

        # Step 1: Train GMVAE
        gmvae_metrics = self.train_gmvae()

        # Step 2: Train BulkEncoder
        bulk_encoder_metrics = self.train_bulk_encoder()

        # Step 3: Generate cells
        results = self.generate_cells(bulk_data_path=bulk_data_path)

        # Step 4: Validate results
        quality_metrics = self.validate_generation_quality(results)

        # Step 5: Save everything
        self._save_pipeline_results(gmvae_metrics, bulk_encoder_metrics, results)

        self.logger.info("Full B2SC pipeline completed successfully!")
        return gmvae_metrics, bulk_encoder_metrics, results

    def _load_training_data(self):
        """Load training data for both GMVAE and BulkEncoder."""
        self.logger.info("Loading training data...")
        self.gmvae_dataloader, self.be_dataloader = (
            self.data_manager.load_single_cell_data()
        )
        self._data_loaded = True
        self.logger.info("Training data loaded successfully")

    def _infer_gmvae_architecture(self, state_dict: dict) -> tuple:
        """
        Infer GMVAE architecture from a state dict.

        Returns:
            tuple: (hidden_dim, latent_dim, use_weight_norm)
        """
        # Check if weight norm was used
        use_weight_norm = "fc0.parametrizations.weight.original0" in state_dict

        # Infer dimensions from decoder (fc3 is authoritative for latent_dim)
        # fc3: Linear(latent_dim, hidden_dim//2)
        if use_weight_norm:
            inferred_latent_dim = state_dict[
                "fc3.parametrizations.weight.original1"
            ].shape[1]
            inferred_hidden_dim = (
                state_dict["fc3.parametrizations.weight.original1"].shape[0] * 2
            )
        else:
            inferred_latent_dim = state_dict["fc3.weight"].shape[1]
            inferred_hidden_dim = state_dict["fc3.weight"].shape[0] * 2

        return inferred_hidden_dim, inferred_latent_dim, use_weight_norm

    def load_pretrained_gmvae(self, pretrained_dir: Path):
        """
        Load a pre-trained GMVAE from an external checkpoint directory.

        This method:
        1. Infers the GMVAE architecture from the checkpoint
        2. Initializes the GMVAE model with the correct architecture
        3. Loads the pre-trained weights
        4. Sets the _gmvae_trained flag to True

        Args:
            pretrained_dir: Path to directory containing GMVAE checkpoint files
                           (GMVAE_model.pt, GMVAE_mus.pt, GMVAE_logvars.pt, GMVAE_pis.pt)

        Raises:
            FileNotFoundError: If required checkpoint files are missing
        """
        import shutil

        pretrained_dir = Path(pretrained_dir)

        # Check for required files
        required_files = [
            "GMVAE_model.pt",
            "GMVAE_mus.pt",
            "GMVAE_logvars.pt",
            "GMVAE_pis.pt",
        ]
        missing_files = [f for f in required_files if not (pretrained_dir / f).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing GMVAE files in {pretrained_dir}:\n"
                + "\n".join(f"  - {f}" for f in missing_files)
            )

        # Load data first if not already loaded
        if not self._data_loaded:
            self._load_training_data()

        # Load checkpoint and infer architecture
        gmvae_state_dict = torch.load(
            pretrained_dir / "GMVAE_model.pt", weights_only=True
        )
        hidden_dim, latent_dim, use_weight_norm = self._infer_gmvae_architecture(
            gmvae_state_dict
        )

        self.logger.info(
            f"Detected GMVAE architecture: hidden_dim={hidden_dim}, "
            f"latent_dim={latent_dim}, use_weight_norm={use_weight_norm}"
        )

        # Initialize GMVAE with detected architecture
        n_components = len(self.data_manager.cell_type_fractions)
        self.gmvae_trainer.model = GaussianMixtureVAE(
            input_dim=self.config.input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_components=n_components,
            use_weight_norm=use_weight_norm,
        ).to(self.gmvae_trainer.device)

        # Load weights
        self.gmvae_trainer.model.load_state_dict(gmvae_state_dict)

        # Copy parameter files to base_dir
        self.config.base_dir.mkdir(parents=True, exist_ok=True)
        for param_file in ["GMVAE_mus.pt", "GMVAE_logvars.pt", "GMVAE_pis.pt"]:
            shutil.copy(pretrained_dir / param_file, self.config.base_dir / param_file)

        # Mark GMVAE as trained
        self._gmvae_trained = True

        self.logger.info(f"✅ Loaded pre-trained GMVAE from: {pretrained_dir}")

    def _load_trained_gmvae(self) -> GaussianMixtureVAE:
        """
        Load trained GMVAE model from base_dir.

        Used for generation/inference. Infers architecture from checkpoint.
        """
        model_path = self.config.base_dir / "GMVAE_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained GMVAE model not found at {model_path}")

        # Load state dict and infer architecture
        state_dict = torch.load(
            model_path, map_location=self.gmvae_trainer.device, weights_only=True
        )
        hidden_dim, latent_dim, use_weight_norm = self._infer_gmvae_architecture(
            state_dict
        )

        # Initialize model with inferred architecture
        n_components = len(self.data_manager.cell_type_fractions)
        gmvae = GaussianMixtureVAE(
            input_dim=self.config.input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_components=n_components,
            use_weight_norm=use_weight_norm,
        )

        # Load weights
        gmvae.load_state_dict(state_dict)

        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            gmvae = nn.DataParallel(gmvae)

        gmvae = gmvae.to(self.gmvae_trainer.device)
        gmvae.eval()

        return gmvae

    def _load_trained_bulk_encoder(self) -> BulkEncoder:
        """Load trained BulkEncoder model."""
        n_components = len(self.data_manager.cell_type_fractions)
        bulk_encoder = BulkEncoder(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.bulk_hidden_dim,
            latent_dim=self.config.latent_dim,
            n_components=n_components,
        )

        # Load model weights
        model_path = self.config.base_dir / "bulkEncoder_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained BulkEncoder model not found at {model_path}"
            )

        bulk_encoder.load_state_dict(
            torch.load(model_path, map_location=self.bulk_encoder_trainer.device)
        )
        bulk_encoder = bulk_encoder.to(self.bulk_encoder_trainer.device)
        bulk_encoder.eval()

        return bulk_encoder

    def _generate_cells_internal(
        self,
        encoder: BulkEncoder,
        gmvae_model: GaussianMixtureVAE,
        dataloader,
        sample_ids: List[str],
        num_cells: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Internal cell generation logic."""
        device = self.gmvae_trainer.device

        encoder.eval()
        gmvae_model.eval()
        encoder = encoder.to(device)
        gmvae_model = gmvae_model.to(device)

        generated_aggregate = []
        sampled_celltypes = []

        print(f"Generating {num_cells} cells...")

        for i in range(num_cells):
            if (i + 1) % 100 == 0:
                print(f"Generating {i + 1}th cell...")

            gt, label = self._generate_single_batch(
                encoder, gmvae_model, dataloader, device=device
            )

            # Append gt to generated_aggregate without changing its type
            generated_aggregate.append(gt)
            for l in label:
                sampled_celltypes.append(l)

            if ((i + 1) % 500 == 0) or (i == num_cells - 1):
                # Process and save the data every 500 iterations

                # Convert the list of tensors to a single tensor
                generated_aggregate_tensor = torch.stack(generated_aggregate)
                generated_aggregate_tensor = generated_aggregate_tensor.squeeze()
                generated_aggregate_tensor = generated_aggregate_tensor.cpu()

                # Convert sampled_celltypes to a tensor
                sampled_celltypes_tensor = torch.LongTensor(sampled_celltypes)

                # Reshape generated_aggregate_tensor
                input_dim = generated_aggregate_tensor.shape[-1]
                generated_aggregate_tensor = generated_aggregate_tensor.reshape(
                    -1, input_dim
                )

                # Save tensors to base directory
                torch.save(
                    generated_aggregate_tensor,
                    self.config.base_dir / "generated_aggregate_tensor.pt",
                )
                torch.save(
                    sampled_celltypes_tensor,
                    self.config.base_dir / "sampled_celltypes.pt",
                )

                print(f"{i + 1}th generated_aggregate_tensor saved.")

        # Load final results
        generated_aggregate_tensor = torch.load(
            self.config.base_dir / "generated_aggregate_tensor.pt"
        )
        sampled_celltypes = torch.load(self.config.base_dir / "sampled_celltypes.pt")
        sampled_celltypes = torch.LongTensor(sampled_celltypes)

        print("Generated cell type proportion:")
        unique_types, counts = np.unique(sampled_celltypes, return_counts=True)
        print(unique_types)
        print(counts / len(sampled_celltypes))

        # Split tensors by sample
        generated_tensor_dict, cell_type_dict = split_generated_tensors(
            generated_aggregate_tensor,
            sampled_celltypes,
            num_samples=len(dataloader),
            sample_ids=sample_ids,
        )

        return generated_tensor_dict, cell_type_dict

    def _generate_single_batch(self, encoder, gmvae_model, dataloader, device="cuda"):
        """Generate cells for a single batch."""
        generated_list = []
        labels_list = []

        # Generate one cell per batch.
        for i, (data, labels) in enumerate(dataloader):
            data = data.to(device)

            # Create a vector of summed gene expressions per batch
            bulk_data = data.sum(dim=0)
            bulk_data = bulk_data.unsqueeze(0)

            # Forward pass
            mus, logvars, pis = encoder(bulk_data)
            mus = mus.squeeze()
            logvars = logvars.squeeze()
            pis = pis.squeeze()

            # Generate from GMVAE
            if hasattr(gmvae_model, "module"):  # DataParallel case
                generated, k = gmvae_model.module.decode_bulk(mus, logvars, pis)
            else:
                generated, k = gmvae_model.decode_bulk(mus, logvars, pis)

            generated_list.append(generated)
            labels_list.append(k.item())

        generated_tensor = torch.stack(generated_list)
        return generated_tensor, labels_list

    def _save_pipeline_results(
        self,
        gmvae_metrics: MetricsTracker,
        bulk_encoder_metrics: MetricsTracker,
        results: GenerationResults,
    ):
        """Save all pipeline results."""
        self.logger.info("Saving pipeline results...")

        # Save generation results
        results.save_results(self.config.base_dir / "generation_results")

        # Save training metrics plots
        gmvae_metrics.plot_detailed_loss_grid(
            self.config.base_dir / "gmvae_detailed_loss_grid.png", config=self.config
        )
        bulk_encoder_metrics.plot_detailed_loss_grid(
            self.config.base_dir / "bulk_encoder_detailed_loss_grid.png",
            config=self.config,
        )

        # Save configuration
        config_path = self.config.base_dir / "pipeline_config.txt"
        with open(config_path, "w") as f:
            f.write("B2SC Pipeline Configuration\n")
            f.write("=" * 30 + "\n\n")
            for key, value in self.config.to_dict().items():
                f.write(f"{key}: {value}\n")

        self.logger.info(f"All results saved to {self.config.base_dir}")

    def visualize_deconvolution_comparison(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 12),
        random_seed: int = 42,
        max_samples_per_split: int = 20,
        dim_reduction_method: str = "PCA",
        verbose: bool = False,
    ):
        """
        Compare deconvoluted bulk samples with their corresponding real single-cell samples.

        For each bulk sample:
        1. Get the corresponding single-cell samples that were used to create it
        2. Generate deconvoluted virtual cells matching the number of real cells
        3. Average both to single vectors per sample
        4. Apply dimensionality reduction (PCA or UMAP)
        5. Plot in 2D: deconvoluted vs real, training vs validation

        Args:
            save_path: Optional path to save the plot
            figsize: Figure size for the plot
            random_seed: Random seed for reproducible sampling
            max_samples_per_split: Maximum number of samples per train/val split
            dim_reduction_method: Dimensionality reduction method ("PCA" or "UMAP")
            verbose: Whether to print progress messages (default: False)
        """
        from utilities.helpers import (
            validate_models_for_visualization,
            setup_visualization_environment,
            process_dataloader_samples,
            prepare_data_for_pca,
            apply_pca_transformation_combined,
            create_deconvolution_visualization_plot,
        )

        # Validate dimensionality reduction method
        if dim_reduction_method.upper() not in ["PCA", "UMAP"]:
            raise ValueError(
                f"dim_reduction_method must be 'PCA' or 'UMAP', got '{dim_reduction_method}'"
            )

        dim_reduction_method = dim_reduction_method.upper()
        print(f"   Using {dim_reduction_method} for dimensionality reduction...")

        # Step 1: Validate models and setup environment
        validate_models_for_visualization(self)
        gmvae, device = setup_visualization_environment(self, random_seed)

        # Step 2: Process training data
        if verbose:
            print("   Processing training bulk samples...")
        train_deconv_vectors, train_real_vectors, train_labels, train_class_labels = (
            process_dataloader_samples(
                self,
                self.be_dataloader,
                self.data_manager.sc_train_data,
                gmvae,
                device,
                random_seed,
                max_samples_per_split,
                "Training",
                verbose,
            )
        )

        # Step 3: Process validation data if available
        val_deconv_vectors, val_real_vectors, val_labels, val_class_labels = (
            [],
            [],
            [],
            [],
        )
        _, val_dataloader = self.data_manager.get_validation_dataloaders()
        if val_dataloader is not None:
            if verbose:
                print("   Processing validation bulk samples...")
            val_deconv_vectors, val_real_vectors, val_labels, val_class_labels = (
                process_dataloader_samples(
                    self,
                    val_dataloader,
                    self.data_manager.sc_val_data,
                    gmvae,
                    device,
                    random_seed,
                    max_samples_per_split,
                    "Validation",
                    verbose,
                )
            )

        # Step 4: Prepare data for dimensionality reduction
        train_deconv, train_real, val_deconv, val_real = prepare_data_for_pca(
            train_deconv_vectors,
            train_real_vectors,
            val_deconv_vectors,
            val_real_vectors,
            verbose,
        )

        # Step 5: Apply dimensionality reduction (combined train+val fitting)
        (
            train_deconv_reduced,
            train_real_reduced,
            val_deconv_reduced,
            val_real_reduced,
            reduction_model,
        ) = apply_pca_transformation_combined(
            train_deconv,
            train_real,
            val_deconv,
            val_real,
            random_seed,
            dim_reduction_method,
        )

        # Step 6: Create visualization
        create_deconvolution_visualization_plot(
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
        )

        # Print summary statistics
        print(f"\n📊 Deconvolution Comparison Summary:")
        print(f"   Training sample pairs: {len(train_deconv)}")
        if val_deconv is not None:
            print(f"   Validation sample pairs: {len(val_deconv)}")

        # Print method-specific information
        if hasattr(reduction_model, "explained_variance_ratio_"):
            # PCA model
            total_variance = reduction_model.explained_variance_ratio_[:2].sum()
            print(f"   PCA variance explained (2D): {total_variance:.1%}")
        else:
            # UMAP model
            print(f"   UMAP dimensionality reduction completed")

        return {
            "train_deconv_reduced": train_deconv_reduced,
            "train_real_reduced": train_real_reduced,
            "val_deconv_reduced": val_deconv_reduced,
            "val_real_reduced": val_real_reduced,
            "reduction_model": reduction_model,
            "dim_reduction_method": dim_reduction_method,
            "train_labels": train_labels,
            "val_labels": val_labels,
            "train_class_labels": train_class_labels,
            "val_class_labels": val_class_labels,
        }

    def get_pipeline_status(self) -> Dict[str, bool]:
        """Get current pipeline training status."""
        return {
            "data_loaded": self._data_loaded,
            "gmvae_trained": self._gmvae_trained,
            "bulk_encoder_trained": self._bulk_encoder_trained,
            "ready_for_generation": self._gmvae_trained and self._bulk_encoder_trained,
        }

    def reset_pipeline(self):
        """Reset pipeline state (useful for retraining)."""
        self.logger.info("Resetting pipeline state...")
        self._gmvae_trained = False
        self._bulk_encoder_trained = False
        self._data_loaded = False

        # Reset trainers
        self.gmvae_trainer = GMVAETrainer(self.config, self.data_manager)
        self.bulk_encoder_trainer = BulkEncoderTrainer(
            self.config, self.gmvae_trainer, self.data_manager
        )

        self.logger.info("Pipeline state reset successfully")
