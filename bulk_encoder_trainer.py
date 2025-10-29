"""
Bulk encoder trainer for B2SC pipeline.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from termcolor import colored

from config import B2SCConfig
from models import BulkEncoder, GaussianMixtureVAE
from metrics import MetricsTracker, TrainingMetrics
from utilities import setup_device, print_model_summary


class BulkEncoderTrainer:
    """Handles BulkEncoder model training and management."""

    def __init__(self, config: B2SCConfig, gmvae_trainer=None, data_manager=None):
        """
        Initialize BulkEncoder trainer.

        Args:
            config: B2SC configuration object
            gmvae_trainer: GMVAE trainer instance (for accessing trained parameters)
            data_manager: Data manager instance (optional for now)
        """
        self.config = config
        self.gmvae_trainer = gmvae_trainer
        self.data_manager = data_manager
        self.model = None
        self.optimizer = None
        self.metrics_tracker = MetricsTracker()
        self.device = setup_device(self.config.device)

        # GMVAE parameters (loaded from trained model)
        self.scMus = None
        self.scLogVars = None
        self.scPis = None

    def load_gmvae_parameters(self) -> bool:
        """
        Load GMVAE parameters from saved files.

        Returns:
            bool: True if parameters loaded successfully, False otherwise
        """
        try:
            mus_path = self.config.base_dir / "GMVAE_mus.pt"
            logvars_path = self.config.base_dir / "GMVAE_logvars.pt"
            pis_path = self.config.base_dir / "GMVAE_pis.pt"

            print(f"Loading scMus, scLogVars, and scPis from {self.config.base_dir}")

            self.scMus = (
                torch.load(mus_path).to(self.device).detach().requires_grad_(False)
            )
            assert torch.any(self.scMus != 0), "scMus contains only zeros"

            self.scLogVars = (
                torch.load(logvars_path).to(self.device).detach().requires_grad_(False)
            )
            assert torch.any(self.scLogVars != 0), "scLogVars contains only zeros"

            self.scPis = (
                torch.load(pis_path).to(self.device).detach().requires_grad_(False)
            )

            # Validate scPis
            sorted_x, _ = torch.sort(self.scPis)
            max_val = sorted_x[-1]
            second_max = sorted_x[-2]  # 2nd largest value

            if max_val >= 1000 * second_max:
                print(
                    colored(
                        "Warning: Max value is ≥1000x larger than the second-largest value.",
                        "red",
                    )
                )

            assert self.scPis.max() > 0, "scPis contains only zeros"
            assert self.scPis.min() >= 0, "scPis contains negative values"

            print(f"Loaded GMVAE parameters:")
            print(f"  scMus shape: {self.scMus.shape}")
            print(f"  scLogVars shape: {self.scLogVars.shape}")
            print(f"  scPis shape: {self.scPis.shape}")

            return True

        except Exception as e:
            print(colored(f"Failed to load GMVAE parameters: {e}", "red"))
            return False

    def initialize_model(
        self, input_dim: int, hidden_dim: int, latent_dim: int, n_components: int
    ) -> BulkEncoder:
        """
        Initialize BulkEncoder model with proper architecture.

        Args:
            input_dim: Number of input features (genes)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            n_components: Number of mixture components

        Returns:
            Initialized BulkEncoder model
        """
        print(colored("Initializing bulkEncoder model...", "yellow"))
        self.model = BulkEncoder(input_dim, hidden_dim, latent_dim, n_components)

        # Check for existing checkpoint
        checkpoint_path = self.config.base_dir / "bulkEncoder_model.pt"
        if checkpoint_path.exists():
            try:
                encoder_state_dict = torch.load(checkpoint_path)
                self.model.load_state_dict(encoder_state_dict, strict=True)
                print("Loaded existing bulkEncoder_model.pt")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(
                        colored(
                            f"Warning: Architecture mismatch in saved checkpoint. Reinitializing model.",
                            "yellow",
                        )
                    )
                    print(
                        colored(
                            f"Current architecture: hidden_dim={hidden_dim}, latent_dim={latent_dim}",
                            "yellow",
                        )
                    )
                    # Initialize with new architecture instead
                    for m in self.model.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.xavier_normal_(m.weight)
                            nn.init.zeros_(m.bias)
                    print(
                        "Initialized BulkEncoder with Xavier weights (new architecture)"
                    )
                else:
                    raise e
        else:
            # Initialize weights
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            print("Initialized BulkEncoder with Xavier weights")

        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.bulk_encoder_lr
        )

        # Print model summary
        print_model_summary(self.model, "BulkEncoder")

        print("BulkEncoder model initialized")
        return self.model

    def train(self, dataloader) -> MetricsTracker:
        """
        Train the BulkEncoder model.

        Args:
            dataloader: DataLoader with single-cell data

        Returns:
            MetricsTracker: Complete training history and metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        if self.scMus is None or self.scLogVars is None or self.scPis is None:
            if not self.load_gmvae_parameters():
                raise ValueError("Failed to load GMVAE parameters. Train GMVAE first.")

        self.model.train()

        for epoch in range(self.config.bulk_encoder_epochs):
            print(
                f"Training BulkEncoder: Epoch {epoch} out of {self.config.bulk_encoder_epochs}"
            )

            # Train one epoch
            epoch_metrics = self._train_epoch(epoch, dataloader)

            # Add metrics to tracker
            self.metrics_tracker.add_epoch_metrics(epoch_metrics)

            # Save intermediate checkpoints
            if ((epoch + 1) % 20 == 0) and (
                epoch != self.config.bulk_encoder_epochs - 1
            ):
                self._save_intermediate_checkpoint(epoch)

            # Save final checkpoint
            elif epoch == self.config.bulk_encoder_epochs - 1:
                self._save_final_checkpoint(epoch)

        return self.metrics_tracker

    def _train_epoch(self, epoch: int, dataloader) -> TrainingMetrics:
        """Train one epoch and return metrics."""
        total_loss = 0
        epoch_mus_loss = 0
        epoch_logvars_loss = 0
        epoch_pis_loss = 0
        epoch_uniform_penalty = 0

        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(self.device)

            # Create bulk data by summing single cells
            bulk_data = data.sum(dim=0)
            # print(f"Sum expression per sample: {bulk_data.sum()}")
            bulk_data = bulk_data.unsqueeze(0)

            # Forward pass
            mus, logvars, pis = self.model(bulk_data)

            mus = mus.squeeze()
            logvars = logvars.squeeze()
            pis = pis.squeeze()

            # Calculate losses
            assert (
                mus.shape == self.scMus.shape
            ), f"mus shape {mus.shape} != scMus shape {self.scMus.shape}"
            mus_loss = F.mse_loss(mus, self.scMus)

            assert (
                logvars.shape == self.scLogVars.shape
            ), f"logvars shape {logvars.shape} != scLogVars shape {self.scLogVars.shape}"
            logvars_loss = F.mse_loss(logvars, self.scLogVars)

            assert (
                pis.shape == self.scPis.shape
            ), f"pis shape {pis.shape} != scPis shape {self.scPis.shape}"
            pis_loss = F.mse_loss(pis, self.scPis)

            # Uniform penalty (encourages diversity in proportions)
            C = pis.shape[0]
            uniform = torch.full_like(pis, 1.0 / C)
            uniform_penalty = F.cosine_similarity(
                pis.unsqueeze(0), uniform.unsqueeze(0)
            ).squeeze()

            # Total loss with configurable weights
            loss = (
                mus_loss * self.config.bulk_mus_loss_weight
                + logvars_loss * self.config.bulk_logvars_loss_weight
                + pis_loss * self.config.bulk_pis_loss_weight
                + uniform_penalty * self.config.bulk_uniform_penalty_weight
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print(colored(f"Loss: {loss.item():.4f}", "magenta"))

            # Accumulate losses
            total_loss += loss.item()
            epoch_mus_loss += mus_loss.item()
            epoch_logvars_loss += logvars_loss.item()
            epoch_pis_loss += pis_loss.item()
            epoch_uniform_penalty += uniform_penalty.item()

        # Print epoch summary
        # print(colored(f"Uniform penalty: {uniform_penalty}", "cyan"))
        # print(colored(f"pis_loss: {pis_loss}", "cyan"))
        # print(colored(f"mus_loss: {mus_loss}", "cyan"))
        # print(colored(f"logvars_loss: {logvars_loss}", "cyan"))

        if (epoch + 1) % 1 == 0:
            print(
                "Epoch[{}/{}]: mus_loss:{:.3f}, vars_loss:{:.3f}, pis_loss:{:.3f}".format(
                    epoch + 1,
                    self.config.bulk_encoder_epochs,
                    mus_loss.item(),
                    logvars_loss.item(),
                    pis_loss.item(),
                )
            )

        # Create metrics object (using accumulated losses like GMVAE)
        metrics = TrainingMetrics(
            epoch=epoch,
            total_loss=total_loss,
            reconstruction_loss=0.0,  # Not applicable for BulkEncoder
            kl_loss=0.0,  # Not applicable for BulkEncoder
            # BulkEncoder specific losses
            mus_loss=epoch_mus_loss,
            logvars_loss=epoch_logvars_loss,
            pis_loss=epoch_pis_loss,
            uniform_penalty=epoch_uniform_penalty,
        )

        return metrics

    def _save_intermediate_checkpoint(self, epoch: int):
        """Save intermediate training checkpoint."""
        print(
            colored(
                f"Saving an intermediate checkpoint of bulkEncoder model to {self.config.base_dir}...",
                "yellow",
            )
        )
        torch.save(
            self.model.state_dict(), self.config.base_dir / "bulkEncoder_model.pt"
        )

    def _save_final_checkpoint(self, epoch: int):
        """Save final training checkpoint."""
        print(colored("Saving last checkpoint of bulkEncoder model...", "yellow"))
        torch.save(
            self.model.state_dict(), self.config.base_dir / "bulkEncoder_model.pt"
        )

        # Save metrics
        self.metrics_tracker.save_metrics(
            self.config.base_dir / "bulk_encoder_metrics.pkl"
        )
        self.metrics_tracker.plot_training_curves(
            self.config.base_dir / "bulk_encoder_training_curves.png"
        )

        # Generate stacked loss contribution plot
        self.metrics_tracker.plot_stacked_loss_contributions(
            self.config.base_dir / "bulk_encoder_loss_contributions.png",
            model_type="BulkEncoder",
        )

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint and training state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": [m.to_dict() for m in self.metrics_tracker.history],
            "config": self.config.to_dict(),
        }

        checkpoint_path = (
            self.config.base_dir / f"bulk_encoder_checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.config.base_dir / "bulk_encoder_best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load model from checkpoint."""
        if not checkpoint_path.exists():
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore metrics history
        metrics_data = checkpoint.get("metrics_history", [])
        self.metrics_tracker.history = [
            TrainingMetrics.from_dict(data) for data in metrics_data
        ]

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return True

    def validate_model(self, validation_loader) -> TrainingMetrics:
        """Run validation and return metrics."""
        if self.model is None:
            raise ValueError("Model not initialized")

        self.model.eval()
        total_loss = 0
        total_mus_loss = 0
        total_logvars_loss = 0
        total_pis_loss = 0

        with torch.no_grad():
            for data, _ in validation_loader:
                data = data.to(self.device)

                # Create bulk data
                bulk_data = data.sum(dim=0).unsqueeze(0)

                # Forward pass
                mus, logvars, pis = self.model(bulk_data)
                mus = mus.squeeze()
                logvars = logvars.squeeze()
                pis = pis.squeeze()

                # Calculate losses
                mus_loss = F.mse_loss(mus, self.scMus)
                logvars_loss = F.mse_loss(logvars, self.scLogVars)
                pis_loss = F.mse_loss(pis, self.scPis)

                C = pis.shape[0]
                uniform = torch.full_like(pis, 1.0 / C)
                uniform_penalty = F.cosine_similarity(
                    pis.unsqueeze(0), uniform.unsqueeze(0)
                ).squeeze()

                loss = (
                    mus_loss + logvars_loss + (pis_loss * 50) + (uniform_penalty * 100)
                )

                total_loss += loss.item()
                total_mus_loss += mus_loss.item()
                total_logvars_loss += logvars_loss.item()
                total_pis_loss += pis_loss.item()

        return TrainingMetrics(
            epoch=-1,  # Validation epoch
            total_loss=total_loss,
            reconstruction_loss=total_mus_loss,
            kl_loss=total_logvars_loss,
            fraction_loss=total_pis_loss,
        )

    def generate_visualizations(self, save_dir: Path):
        """Generate training curve plots and other visualizations."""
        if save_dir != self.config.base_dir:
            self.metrics_tracker.plot_training_curves(
                save_dir / "bulk_encoder_training_curves.png"
            )
