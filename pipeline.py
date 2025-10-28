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
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            n_components=n_components,
        )

        # Train the model
        metrics_tracker = self.gmvae_trainer.train(
            dataloader=self.gmvae_dataloader,
            proportion_tensor=self.data_manager.cell_type_fractions,
            mapping_dict=self.data_manager.cell_type_mapping,
            color_map=self.data_manager.color_map,
            plot_umap=True,
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
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            n_components=n_components,
        )

        # Train the model
        metrics_tracker = self.bulk_encoder_trainer.train(dataloader=self.be_dataloader)

        self._bulk_encoder_trained = True
        self.logger.info("BulkEncoder training completed successfully")
        return metrics_tracker

    def generate_cells(
        self,
        bulk_data_path: str = None,
        sample_ids: List[str] = None,
        num_cells: int = None,
    ) -> GenerationResults:
        """
        Generate single cells from bulk samples.

        Args:
            bulk_data_path: Path to bulk data (optional, uses config if None)
            sample_ids: List of sample identifiers to generate from (optional, uses all if None)
            num_cells: Number of cells to generate per sample (uses config if None)

        Returns:
            GenerationResults: Complete generation results
        """
        if not self._gmvae_trained or not self._bulk_encoder_trained:
            raise ValueError(
                "Both GMVAE and BulkEncoder must be trained before generation. Call train_gmvae() and train_bulk_encoder() first."
            )

        self.logger.info("Starting cell generation...")

        # Use config defaults if not provided
        if num_cells is None:
            num_cells = self.config.num_cells_to_generate

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
            "bulk_data_path": bulk_data_path or self.config.bulk_data_path,
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

    def _load_trained_gmvae(self) -> GaussianMixtureVAE:
        """Load trained GMVAE model."""
        n_components = len(self.data_manager.cell_type_fractions)
        gmvae = GaussianMixtureVAE(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            n_components=n_components,
        )

        # Load model weights
        model_path = self.config.base_dir / "GMVAE_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained GMVAE model not found at {model_path}")

        gmvae.load_state_dict(
            torch.load(model_path, map_location=self.gmvae_trainer.device)
        )

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
            hidden_dim=self.config.hidden_dim,
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
        gmvae_metrics.plot_training_curves(
            self.config.base_dir / "gmvae_training_curves.png"
        )
        bulk_encoder_metrics.plot_training_curves(
            self.config.base_dir / "bulk_encoder_training_curves.png"
        )

        # Save configuration
        config_path = self.config.base_dir / "pipeline_config.txt"
        with open(config_path, "w") as f:
            f.write("B2SC Pipeline Configuration\n")
            f.write("=" * 30 + "\n\n")
            for key, value in self.config.to_dict().items():
                f.write(f"{key}: {value}\n")

        self.logger.info(f"All results saved to {self.config.base_dir}")

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
