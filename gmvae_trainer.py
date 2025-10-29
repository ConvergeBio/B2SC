"""
GMVAE trainer for B2SC pipeline.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from termcolor import colored

from config import B2SCConfig
from models import GaussianMixtureVAE
from metrics import MetricsTracker, TrainingMetrics
from utilities import setup_device, print_model_summary


def zinb_loss(y_true, y_pred, pi, r, eps=1e-10):
    """
    Zero-Inflated Negative Binomial loss function.

    Args:
        y_true: True gene expression values
        y_pred: Predicted gene expression values
        pi: Zero-inflation parameter
        r: Overdispersion parameter
        eps: Small epsilon for numerical stability

    Returns:
        ZINB loss value
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Negative binomial part
    nb_case = (
        -torch.lgamma(r + eps)
        + torch.lgamma(y_true + r + eps)
        - torch.lgamma(y_true + 1.0)
        + r * torch.log(pi + eps)
        + y_true * torch.log(1.0 - (pi + eps))
    )

    # Zero-inflated part
    zero_nb = torch.pow(pi, r)
    zero_case = torch.where(
        y_true < eps,
        -torch.log(
            zero_nb + (1.0 - zero_nb) * torch.exp(-r * torch.log(1.0 - pi + eps))
        ),
        torch.tensor(0.0, device=y_true.device),
    )

    return -torch.mean(zero_case + nb_case)


class GMVAETrainer:
    """Handles GMVAE model training and management."""

    def __init__(self, config: B2SCConfig, data_manager=None):
        """
        Initialize GMVAE trainer.

        Args:
            config: B2SC configuration object
            data_manager: Data manager instance (optional for now)
        """
        self.config = config
        self.data_manager = data_manager
        self.model = None
        self.optimizer = None
        self.metrics_tracker = MetricsTracker()
        self.device = setup_device(self.config.device)

    def initialize_model(
        self, input_dim: int, hidden_dim: int, latent_dim: int, n_components: int
    ) -> GaussianMixtureVAE:
        """
        Initialize GMVAE model with proper architecture.

        Args:
            input_dim: Number of input features (genes)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            n_components: Number of mixture components

        Returns:
            Initialized GMVAE model
        """
        self.model = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, n_components)

        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        # Initialize weights
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.gmvae_lr
        )

        # Print model summary
        print_model_summary(self.model, "GMVAE")

        print("Initialized GMVAE model")
        return self.model

    def train(
        self,
        dataloader,
        proportion_tensor: torch.Tensor,
        mapping_dict: Dict,
        color_map: Dict,
        plot_umap: bool = False,
    ) -> MetricsTracker:
        """
        Train the GMVAE model.

        Args:
            dataloader: DataLoader with single-cell data
            proportion_tensor: Cell type proportions
            mapping_dict: Mapping from indices to cell type names
            color_map: Color mapping for visualizations
            plot_umap: Whether to generate UMAP plots

        Returns:
            MetricsTracker: Complete training history and metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        self.model.train()

        # KL weight annealing
        kl_weight = 0.0
        kl_weight_max = 1.0
        kl_weight_increment = kl_weight_max / int((self.config.gmvae_epochs * 1.2))

        for epoch in range(self.config.gmvae_epochs):
            print(f"Epoch {epoch} out of {self.config.gmvae_epochs}")

            # Update KL weight
            if kl_weight < kl_weight_max:
                kl_weight += kl_weight_increment
                kl_weight = min(kl_weight, kl_weight_max)

            # Train one epoch
            epoch_metrics = self._train_epoch(
                epoch,
                dataloader,
                proportion_tensor,
                kl_weight,
                mapping_dict,
                color_map,
                plot_umap,
            )

            # Add metrics to tracker
            self.metrics_tracker.add_epoch_metrics(epoch_metrics)

            # Save intermediate results
            if ((epoch + 1) % 10 == 0) and (epoch != self.config.gmvae_epochs - 1):
                self._save_intermediate_results(epoch)
                if plot_umap:
                    self._generate_umap_plots(
                        epoch, dataloader, mapping_dict, color_map
                    )

            # Save final results
            elif epoch == self.config.gmvae_epochs - 1:
                self._save_final_results(epoch)

        return self.metrics_tracker

    def _train_epoch(
        self,
        epoch: int,
        dataloader,
        proportion_tensor: torch.Tensor,
        kl_weight: float,
        mapping_dict: Dict,
        color_map: Dict,
        plot_umap: bool,
    ) -> TrainingMetrics:
        """Train one epoch and return metrics."""
        total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_fraction_loss = 0
        epoch_zinb_loss = 0

        for idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            reconstructed, mus, logvars, pis, zs = self.model(data, labels)

            if epoch == 0 and idx == 0:
                print(
                    "************************************* First batch of data: *************************************************"
                )
                print("reconstructed.shape:", reconstructed.shape)
                print("data.shape:", data.shape)
                print("mus.shape:", mus.shape)
                print("logvars.shape:", logvars.shape)
                print("pis.shape:", pis.shape)
                print("zs.shape:", zs.shape)

            assert reconstructed.shape == data.shape

            # Calculate losses
            proportion_tensor_reshaped = proportion_tensor.to(pis.device)
            fraction_loss = F.mse_loss(pis.mean(0), proportion_tensor_reshaped)
            loss_recon = F.mse_loss(reconstructed, data)

            # Get ZINB parameters from model
            if hasattr(self.model, "module"):
                prob_extra_zero = self.model.module.prob_extra_zero
                over_disp = self.model.module.over_disp
            else:
                prob_extra_zero = self.model.prob_extra_zero
                over_disp = self.model.over_disp

            zinb_loss_val = zinb_loss(data, reconstructed, prob_extra_zero, over_disp)

            loss_kl = 0.5 * torch.sum(-1 - logvars + mus.pow(2) + logvars.exp()) / 1e9
            loss_kl_weighted = loss_kl * kl_weight
            loss_kl_weighted = 1 if loss_kl_weighted > 1 else loss_kl_weighted

            # Total loss with configurable weights
            loss = (
                loss_recon * self.config.gmvae_recon_loss_weight
                + zinb_loss_val * self.config.gmvae_zinb_loss_weight
                + fraction_loss * self.config.gmvae_fraction_loss_weight
                + loss_kl_weighted * self.config.gmvae_kl_loss_weight
            )
            loss.backward()
            self.optimizer.step()

            # print(colored(f"Loss: {loss.item():.4f}", "magenta"))

            # Accumulate losses
            total_loss += loss.item()
            epoch_recon_loss += loss_recon.item()
            epoch_kl_loss += loss_kl.item()
            epoch_fraction_loss += fraction_loss.item()
            epoch_zinb_loss += zinb_loss_val.item()

        # Print epoch summary
        # print(colored(f"fraction_loss: {fraction_loss:.4f}", "cyan"))
        # print(colored(f"loss_kl unweighted: {loss_kl:.4f}", "blue"))
        # print(colored(f"loss_kl weighted: {loss_kl_weighted:.4f}", "blue"))
        # print(colored(f"kl_weight: {kl_weight:.4f}", "blue"))
        # print(colored(f"loss_recon: {loss_recon:.4f}", "blue"))
        print("Epoch finished")

        # Create metrics object
        metrics = TrainingMetrics(
            epoch=epoch,
            total_loss=total_loss,
            reconstruction_loss=epoch_recon_loss,
            kl_loss=epoch_kl_loss,
            fraction_loss=epoch_fraction_loss,
            zinb_loss=epoch_zinb_loss,
            kl_weight=kl_weight,
        )

        return metrics

    def _save_intermediate_results(self, epoch: int):
        """Save intermediate training results."""
        print("Saving intermediate results to folder:", self.config.base_dir)

        # Extract parameters (handle DataParallel)
        if hasattr(self.model, "module"):
            model_for_extraction = self.model.module
        else:
            model_for_extraction = self.model

        # Save model state
        torch.save(self.model.state_dict(), self.config.base_dir / "GMVAE_model.pt")
        print("GMVAE Model saved.")

    def _save_final_results(self, epoch: int):
        """Save final training results."""
        print(
            colored(f"Saving final results to folder: {self.config.base_dir}", "green")
        )

        # Save model state
        torch.save(self.model.state_dict(), self.config.base_dir / "GMVAE_model.pt")
        print("GMVAE Model saved.")

        # Extract and save GMVAE parameters for BulkEncoder
        self._save_gmvae_parameters()

        # Save metrics
        self.metrics_tracker.save_metrics(self.config.base_dir / "gmvae_metrics.pkl")
        self.metrics_tracker.plot_training_curves(
            self.config.base_dir / "gmvae_training_curves.png"
        )

        # Generate stacked loss contribution plot
        self.metrics_tracker.plot_stacked_loss_contributions(
            self.config.base_dir / "gmvae_loss_contributions.png", model_type="GMVAE"
        )

    def _generate_umap_plots(
        self, epoch: int, dataloader, mapping_dict: Dict, color_map: Dict
    ):
        """Generate UMAP visualizations."""
        print("Plotting UMAP...")
        self.model.eval()

        # Collect data for UMAP
        all_z = []
        all_reconstructed = []
        all_labels = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                reconstructed, mus, logvars, pis, zs = self.model(data, labels)

                all_z.append(zs.cpu().numpy())
                all_reconstructed.append(reconstructed.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Concatenate all data
        z = np.concatenate(all_z, axis=0)
        reconstructed = np.concatenate(all_reconstructed, axis=0)

        # Convert labels to colors
        label_map = {str(v): k for k, v in mapping_dict.items()}
        z_colors = [color_map[label_map[str(label)]] for label in all_labels]

        # UMAP transformation
        reducer = umap.UMAP()
        embedding_z = reducer.fit_transform(z)
        embedding_recon = reducer.fit_transform(reconstructed)

        # Plot latent space
        plt.figure(figsize=(12, 10))
        plt.scatter(embedding_z[:, 0], embedding_z[:, 1], c=z_colors, s=5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title("UMAP of reparameterized z")
        plt.grid(True, edgecolor="black")
        plt.savefig(self.config.base_dir / "umap_latent.png")
        plt.close()

        # Plot reconstructed space
        plt.figure(figsize=(12, 10))
        plt.scatter(embedding_recon[:, 0], embedding_recon[:, 1], c=z_colors, s=5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title("UMAP of Reconstructed Data")
        plt.grid(True, edgecolor="black")
        plt.savefig(self.config.base_dir / "umap_recon.png")
        plt.close()

        self.model.train()

    def _save_gmvae_parameters(self):
        """Extract and save GMVAE parameters (mus, logvars, pis) for BulkEncoder training."""
        print("Extracting and saving GMVAE parameters...")

        self.model.eval()

        # Collect parameters from all training data
        all_mus = []
        all_logvars = []
        all_pis = []

        # Get the training dataloader from data manager
        gmvae_dataloader, _ = self.data_manager.load_single_cell_data()

        with torch.no_grad():
            for data, labels in gmvae_dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                # Forward pass to get parameters
                reconstructed, mus, logvars, pis, zs = self.model(data, labels)

                all_mus.append(mus.cpu())
                all_logvars.append(logvars.cpu())
                all_pis.append(pis.cpu())

        # Concatenate and compute means
        mus = torch.cat(all_mus, dim=0).mean(0)
        logvars = torch.cat(all_logvars, dim=0).mean(0)
        pis = torch.cat(all_pis, dim=0).mean(0)

        # Save the parameters
        print(f"Saving mus, logvars, and pis to folder: {self.config.base_dir}")
        torch.save(mus, self.config.base_dir / "GMVAE_mus.pt")
        torch.save(logvars, self.config.base_dir / "GMVAE_logvars.pt")
        torch.save(pis, self.config.base_dir / "GMVAE_pis.pt")
        print("GMVAE mus, logvars, and pis saved.")

        self.model.train()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint and training state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": [m.to_dict() for m in self.metrics_tracker.history],
            "config": self.config.to_dict(),
        }

        checkpoint_path = self.config.base_dir / f"gmvae_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.config.base_dir / "gmvae_best_model.pt"
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

    def get_latent_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract trained mu, logvar, and pi parameters from saved files.

        These parameters are saved during training and used by the BulkEncoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (mus, logvars, pis)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Load parameters from saved files (these are saved during training)
        try:
            mus_path = self.config.base_dir / "GMVAE_mus.pt"
            logvars_path = self.config.base_dir / "GMVAE_logvars.pt"
            pis_path = self.config.base_dir / "GMVAE_pis.pt"

            if not all(p.exists() for p in [mus_path, logvars_path, pis_path]):
                raise FileNotFoundError(
                    "GMVAE parameter files not found. Train the model first."
                )

            mus = torch.load(mus_path, map_location=self.device)
            logvars = torch.load(logvars_path, map_location=self.device)
            pis = torch.load(pis_path, map_location=self.device)

            return mus, logvars, pis

        except Exception as e:
            raise RuntimeError(f"Failed to load GMVAE parameters: {e}")

    def validate_model(self, validation_loader) -> TrainingMetrics:
        """Run validation and return metrics."""
        if self.model is None:
            raise ValueError("Model not initialized")

        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        with torch.no_grad():
            for data, labels in validation_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                reconstructed, mus, logvars, pis, zs = self.model(data, labels)

                recon_loss = F.mse_loss(reconstructed, data)
                kl_loss = (
                    0.5 * torch.sum(-1 - logvars + mus.pow(2) + logvars.exp()) / 1e9
                )
                loss = recon_loss + kl_loss

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

        return TrainingMetrics(
            epoch=-1,  # Validation epoch
            total_loss=total_loss,
            reconstruction_loss=total_recon_loss,
            kl_loss=total_kl_loss,
        )

    def generate_visualizations(self, save_dir: Path):
        """Generate UMAP plots and other visualizations."""
        if save_dir != self.config.base_dir:
            self.metrics_tracker.plot_training_curves(
                save_dir / "gmvae_training_curves.png"
            )
