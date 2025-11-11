"""
GMVAE trainer for B2SC pipeline.
"""

import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from termcolor import colored
from tqdm import tqdm
from scipy import stats

from config import B2SCConfig
from models import GaussianMixtureVAE
from metrics import MetricsTracker, GMVAEMetrics
from utilities import setup_device, print_model_summary
from loss import GMVAELossCalculator


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

        # Initialize loss calculator (will be updated after model initialization)
        self.loss_calculator = None

    def _reconstruction_warmup_weight(self, epoch: int) -> float:
        """
        Calculate reconstruction loss weight using sigmoid warm-up.

        Args:
            epoch: Current training epoch

        Returns:
            Weight multiplier for reconstruction loss (0.0 to 1.0)
        """
        steepness = self.config.gmvae_hyperparameters.get("recon_warmup_steepness", 2.0)
        midpoint = self.config.gmvae_hyperparameters.get("recon_warmup_midpoint", 3.5)

        # Sigmoid function: 1 / (1 + exp(-steepness * (epoch - midpoint)))
        weight = 1.0 / (1.0 + np.exp(-steepness * (epoch - midpoint)))
        return float(weight)

    def _kl_annealing_weight(self, epoch: int) -> float:
        """
        Calculate KL annealing weight for current epoch.

        Gradually increases KL weight from start to max over specified epochs
        to prevent posterior collapse. After annealing completes, maintains max weight.

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            Current KL weight (between kl_anneal_start and kl_anneal_max)
        """
        start_weight = self.config.gmvae_hyperparameters.get("kl_anneal_start", 0.0)
        max_weight = self.config.gmvae_hyperparameters.get("kl_anneal_max", 1.0)
        anneal_epochs = self.config.gmvae_hyperparameters.get("kl_anneal_epochs")

        # If kl_anneal_epochs is None, use the total number of epochs
        if anneal_epochs is None:
            anneal_epochs = self.config.gmvae_epochs

        # Linear annealing over epochs
        if epoch >= anneal_epochs:
            return max_weight

        progress = epoch / anneal_epochs
        current_weight = start_weight + (max_weight - start_weight) * progress

        return float(current_weight)

    def _cleanup_reconstruction_plots(self):
        """
        Clean up old reconstruction distribution plots from previous training runs.

        This prevents old plots from being included in the summary figure
        when running multiple consecutive training sessions.
        """
        recon_dir = Path(self.config.base_dir) / "reconstruction_distributions"

        if recon_dir.exists():
            # Remove all files in the directory
            shutil.rmtree(recon_dir)
            print(f"  🧹 Cleaned up old reconstruction distribution plots")

    def _plot_reconstruction_distributions(
        self,
        data: torch.Tensor,
        reconstructed: torch.Tensor,
        epoch: int,
        batch_idx: int,
        label: str,
    ):
        """
        Plot KDE distributions comparing input and reconstructed data.

        Args:
            data: Original input data [n_cells, n_genes]
            reconstructed: Reconstructed data [n_cells, n_genes]
            epoch: Current epoch number
            batch_idx: Current batch index
            label: Label for the plot (e.g., "first_batch", "last_batch")
        """
        # Flatten tensors to 1D for KDE
        data_flat = data.detach().cpu().numpy().flatten()
        recon_flat = reconstructed.detach().cpu().numpy().flatten()

        # Sample if too many points (for faster KDE computation)
        max_points = 50000
        if len(data_flat) > max_points:
            indices = np.random.choice(len(data_flat), max_points, replace=False)
            data_flat = data_flat[indices]
            recon_flat = recon_flat[indices]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot KDE for original data
        try:
            kde_data = stats.gaussian_kde(data_flat)
            x_range = np.linspace(
                min(data_flat.min(), recon_flat.min()),
                max(data_flat.max(), recon_flat.max()),
                200,
            )
            ax.plot(
                x_range,
                kde_data(x_range),
                label="Original Data",
                linewidth=2,
                alpha=0.8,
            )

            # Plot KDE for reconstructed data
            kde_recon = stats.gaussian_kde(recon_flat)
            ax.plot(
                x_range,
                kde_recon(x_range),
                label="Reconstructed",
                linewidth=2,
                alpha=0.8,
            )

        except Exception as e:
            # Fallback to histograms if KDE fails
            ax.hist(data_flat, bins=50, alpha=0.5, density=True, label="Original Data")
            ax.hist(recon_flat, bins=50, alpha=0.5, density=True, label="Reconstructed")

        # Calculate and display statistics
        mse = np.mean((data_flat - recon_flat) ** 2)
        correlation = np.corrcoef(data_flat, recon_flat)[0, 1]

        ax.set_xlabel("Gene Expression Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            f"Reconstruction Distribution - Epoch {epoch} ({label})\n"
            f"MSE: {mse:.4f}, Correlation: {correlation:.4f}",
            fontsize=14,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save figure
        save_dir = Path(self.config.base_dir) / "reconstruction_distributions"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"epoch_{epoch:03d}_{label}_batch_{batch_idx}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  📊 Saved reconstruction distribution plot: {save_path.name}")

    def _create_reconstruction_summary_figure(self):
        """
        Create a summary figure showing reconstruction distribution progression.

        Combines all saved reconstruction distribution plots into one grid figure
        showing how distributions converge over training.
        """
        recon_dir = Path(self.config.base_dir) / "reconstruction_distributions"

        if not recon_dir.exists():
            print("  ⚠️  No reconstruction distribution plots found, skipping summary.")
            return

        # Find all saved plots
        plot_files = sorted(recon_dir.glob("epoch_*.png"))

        if len(plot_files) == 0:
            print("  ⚠️  No reconstruction distribution plots found, skipping summary.")
            return

        # Create summary figure with all plots
        n_plots = len(plot_files)
        n_cols = min(2, n_plots)  # Max 2 columns
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

        # Figure size: 20% smaller area than original (12x6 -> 10.7x5.4 per subplot)
        fig = plt.figure(figsize=(10.7 * n_cols, 5.4 * n_rows))

        for idx, plot_file in enumerate(plot_files):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)

            # Read and display the saved plot
            img = plt.imread(str(plot_file))
            ax.imshow(img)
            ax.axis("off")

            # Add subplot title
            plot_name = plot_file.stem
            ax.set_title(plot_name.replace("_", " ").title(), fontsize=14, pad=10)

        plt.suptitle(
            "Reconstruction Distribution Progression",
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout()

        # Save summary figure
        summary_path = self.config.base_dir / "reconstruction_distribution_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")

        # Display in notebook if running interactively
        plt.show()

        plt.close(fig)

        print(f"  📊 Saved reconstruction summary: {summary_path.name}")

    def initialize_model(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_components: int,
        load_from_existing: bool = False,
    ) -> GaussianMixtureVAE:
        """
        Initialize GMVAE model with proper architecture.

        Args:
            input_dim: Number of input features (genes)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            n_components: Number of mixture components
            load_from_existing: Whether to load from existing checkpoint (default: False)

        Returns:
            Initialized GMVAE model
        """
        print(colored("🧬 Initializing GMVAE model...", "yellow"))
        use_weight_norm = self.config.gmvae_hyperparameters.get(
            "use_weight_norm", False
        )
        self.model = GaussianMixtureVAE(
            input_dim,
            hidden_dim,
            latent_dim,
            n_components,
            use_weight_norm=use_weight_norm,
        )

        # Check if we should load from existing checkpoint
        if load_from_existing:
            checkpoint_path = self.config.base_dir / "GMVAE_model.pt"
            if checkpoint_path.exists():
                try:
                    gmvae_state_dict = torch.load(checkpoint_path, weights_only=True)
                    self.model.load_state_dict(gmvae_state_dict, strict=True)
                    print(
                        colored(
                            "✅ Loaded existing GMVAE checkpoint from GMVAE_model.pt",
                            "green",
                        )
                    )
                except Exception as e:
                    print(colored(f"⚠️  Failed to load GMVAE checkpoint: {e}", "red"))
                    print(
                        colored(
                            "🔄 Initializing GMVAE with fresh weights instead", "yellow"
                        )
                    )
                    self._initialize_fresh_weights()
            else:
                print(
                    colored(
                        "⚠️  No existing GMVAE checkpoint found - initializing with fresh weights",
                        "yellow",
                    )
                )
                self._initialize_fresh_weights()
        else:
            print(
                colored(
                    "🆕 Initializing GMVAE with fresh weights (default behavior)",
                    "cyan",
                )
            )
            self._initialize_fresh_weights()

        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.gmvae_lr
        )

        # Initialize loss calculator now that model is available
        self.loss_calculator = GMVAELossCalculator(
            config=self.config,
            model=self.model,
            reconstruction_warmup_fn=self._reconstruction_warmup_weight,
        )

        return self.model

    def _initialize_fresh_weights(self):
        """Initialize model weights from scratch."""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        print(colored("🎲 Initialized GMVAE with Xavier normal weights", "cyan"))

    def train(
        self,
        dataloader,
        proportion_tensor: torch.Tensor,
        mapping_dict: Dict,
        color_map: Dict,
        val_dataloader=None,
    ) -> MetricsTracker:
        """
        Train the GMVAE model.

        Args:
            dataloader: DataLoader with single-cell data
            proportion_tensor: Cell type proportions
            mapping_dict: Mapping from indices to cell type names
            color_map: Color mapping for visualizations
            val_dataloader: Optional validation DataLoader

        Returns:
            MetricsTracker: Complete training history and metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        self.model.train()

        # Clean up old reconstruction distribution plots from previous runs
        self._cleanup_reconstruction_plots()

        # Create progress bar for epochs
        epoch_pbar = tqdm(
            range(self.config.gmvae_epochs),
            desc="GMVAE Training",
            unit="epoch",
            position=0,  # Position 0 for main progress bar
        )

        for epoch in epoch_pbar:

            # Check model parameters at start of epoch
            model_has_nan_start = any(
                torch.isnan(p).any() for p in self.model.parameters()
            )
            if model_has_nan_start:
                print(f"🚨 Model has NaN parameters at start of epoch {epoch}!")
                print(
                    f"   Current learning rate: {self.optimizer.param_groups[0]['lr']}"
                )
                print(
                    f"   This indicates the learning rate is too high or model initialization issue."
                )
                print(f"   💡 Try reducing learning rate to 1e-5 or 1e-6")
                break

            # Calculate KL weight for this epoch using annealing schedule
            kl_weight = self._kl_annealing_weight(epoch)
            print(
                f"KL Weight: {kl_weight:.6f} (epoch {epoch}/{self.config.gmvae_epochs})"
            )

            # Train one epoch
            epoch_metrics = self._train_epoch(
                epoch,
                dataloader,
                proportion_tensor,
                kl_weight,
                mapping_dict,
                color_map,
            )

            # Run validation if provided
            if val_dataloader is not None:
                val_metrics = self._validate_epoch(
                    epoch, val_dataloader, proportion_tensor, kl_weight
                )
                # Add validation metrics to the epoch metrics
                epoch_metrics.val_total_loss = val_metrics.total_loss
                epoch_metrics.val_reconstruction_loss = val_metrics.reconstruction_loss
                epoch_metrics.val_kl_loss = val_metrics.kl_loss
                epoch_metrics.val_fraction_loss = val_metrics.fraction_loss
                epoch_metrics.val_zinb_loss = val_metrics.zinb_loss
                epoch_metrics.val_reconstruction_correlation = (
                    val_metrics.val_reconstruction_correlation
                )

            # Add metrics to tracker
            self.metrics_tracker.add_epoch_metrics(epoch_metrics)

            # Update progress bar with weighted loss information
            progress_info = {
                "Train Loss": f"{epoch_metrics.total_loss:.3e}",
                "Recon": f"{epoch_metrics.weighted_reconstruction_loss:.3e}",
                "KL": f"{epoch_metrics.weighted_kl_loss:.3e}",
            }
            if epoch_metrics.val_total_loss is not None:
                progress_info["Val Loss"] = f"{epoch_metrics.val_total_loss:.3e}"

            epoch_pbar.set_postfix(progress_info)

            # Save intermediate results
            if ((epoch + 1) % 10 == 0) and (epoch != self.config.gmvae_epochs - 1):
                self._save_intermediate_results(epoch)

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
    ) -> GMVAEMetrics:
        """Train one epoch and return metrics."""
        total_loss = 0
        # Raw (unweighted) loss accumulators
        epoch_raw_recon_loss = 0
        epoch_raw_kl_loss = 0
        epoch_raw_fraction_loss = 0
        epoch_raw_zinb_loss = 0
        # Weighted loss accumulators (for verification)
        epoch_weighted_recon_loss = 0
        epoch_weighted_kl_loss = 0
        epoch_weighted_fraction_loss = 0
        epoch_weighted_zinb_loss = 0
        # Correlation accumulator (for model collapse detection)
        epoch_correlation_sum = 0.0
        epoch_correlation_count = 0

        # Get total number of batches for identifying last batch
        num_batches = len(dataloader)

        # Create batch progress bar (nested within epoch progress bar)
        batch_iterator = tqdm(
            dataloader,
            total=num_batches,
            desc=f"  Epoch {epoch+1} Batches",
            leave=False,
            unit="batch",
            position=1,  # Position 1 to nest under epoch progress bar (position 0)
            disable=False,  # Ensure it's not disabled
        )

        for idx, batch in enumerate(batch_iterator):
            # Unpack batch from donor-grouped collate function
            data, cell_type_labels, num_cells_per_donor, donor_ids = batch
            data = data.to(self.device)
            cell_type_labels = cell_type_labels.to(self.device)
            # Note: cell_type_labels are cell type encodings (0=T_cell, 1=B_cell, 2=NK_cell, etc.)
            # Note: data contains all cells from batch_size donors, flattened

            # Validate input dimensions
            assert (
                data.ndim == 2
            ), f"Expected data to be 2D (total_cells, n_genes), got {data.ndim}D with shape {data.shape}"
            assert (
                cell_type_labels.ndim == 1
            ), f"Expected cell_type_labels to be 1D (total_cells,), got {cell_type_labels.ndim}D with shape {cell_type_labels.shape}"
            assert (
                data.shape[0] == cell_type_labels.shape[0]
            ), f"Cell count mismatch: data has {data.shape[0]} cells but cell_type_labels has {cell_type_labels.shape[0]} labels"

            self.optimizer.zero_grad()
            reconstructed, mus, logvars, pis, zs = self.model(data, cell_type_labels)

            assert reconstructed.shape == data.shape

            # Plot reconstruction distributions for first and last batches
            is_first_batch = epoch == 0 and idx == 0
            is_last_batch = idx == num_batches - 1

            if is_first_batch:
                self._plot_reconstruction_distributions(
                    data=data,
                    reconstructed=reconstructed,
                    epoch=epoch,
                    batch_idx=idx,
                    label="first_batch",
                )
            elif is_last_batch:
                self._plot_reconstruction_distributions(
                    data=data,
                    reconstructed=reconstructed,
                    epoch=epoch,
                    batch_idx=idx,
                    label="last_batch",
                )

            # Calculate losses using shared loss calculator
            loss_components = self.loss_calculator.calculate_losses(
                data=data,
                reconstructed=reconstructed,
                mus=mus,
                logvars=logvars,
                pis=pis,
                proportion_tensor=proportion_tensor,
                epoch=epoch,
                kl_weight=kl_weight,
            )

            # Extract individual components
            loss_recon_raw = loss_components.reconstruction_loss_raw
            loss_kl_raw = loss_components.kl_loss_raw
            fraction_loss_raw = loss_components.fraction_loss_raw
            zinb_loss_raw = loss_components.zinb_loss_raw

            loss_recon_weighted = loss_components.reconstruction_loss_weighted
            loss_kl_weighted = loss_components.kl_loss_weighted
            fraction_loss_weighted = loss_components.fraction_loss_weighted
            zinb_loss_weighted = loss_components.zinb_loss_weighted

            recon_warmup_weight = loss_components.recon_warmup_weight

            # Total loss from shared calculator
            loss = loss_components.total_loss
            # Check for NaN in loss before backward pass
            if torch.isnan(loss):
                print(f"  ⚠️  NaN loss detected! Skipping backward pass for batch {idx}")
                # Check if model parameters have NaN
                model_has_nan = any(
                    torch.isnan(p).any() for p in self.model.parameters()
                )
                if model_has_nan:
                    print(f"  🚨 Model parameters have NaN! This is a critical issue.")
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(f"     Current learning rate: {current_lr}")
                    print(
                        f"     💡 The learning rate {current_lr} is too high for this model."
                    )
                    print(
                        f"     💡 Please restart training with learning rate 1e-5 or 1e-6"
                    )
                    return None  # Exit training
                continue

            # Backward pass
            loss.backward()

            # Check for NaN in gradients before optimizer step
            grad_has_nan = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in self.model.parameters()
            )
            if grad_has_nan:
                print(
                    f"  ⚠️  NaN gradients detected! Skipping optimizer step for batch {idx}"
                )
                self.optimizer.zero_grad()
                continue

            self.optimizer.step()

            # Check if model parameters became NaN after optimizer step
            model_has_nan_after = any(
                torch.isnan(p).any() for p in self.model.parameters()
            )
            if model_has_nan_after:
                print(f"  🚨 Model parameters became NaN after optimizer step!")
                print(
                    f"     This indicates the learning rate is too high or gradients exploded."
                )
                return None  # Exit training

            # print(colored(f"Loss: {loss.item():.4f}", "magenta"))

            # Calculate correlation between original and reconstructed data (for model collapse detection)
            with torch.no_grad():
                # Flatten to (batch_size, n_features)
                data_flat = data.view(data.size(0), -1)
                recon_flat = reconstructed.view(reconstructed.size(0), -1)

                # Calculate per-sample correlation
                data_mean = data_flat.mean(dim=1, keepdim=True)
                recon_mean = recon_flat.mean(dim=1, keepdim=True)

                data_centered = data_flat - data_mean
                recon_centered = recon_flat - recon_mean

                numerator = (data_centered * recon_centered).sum(dim=1)
                data_std = torch.sqrt((data_centered**2).sum(dim=1))
                recon_std = torch.sqrt((recon_centered**2).sum(dim=1))
                denominator = data_std * recon_std

                # Avoid division by zero
                correlation = numerator / (denominator + 1e-8)
                batch_correlation = correlation.mean().item()

                epoch_correlation_sum += batch_correlation
                epoch_correlation_count += 1

            # Accumulate losses
            total_loss += loss.item()

            # Raw (unweighted) losses
            epoch_raw_recon_loss += loss_recon_raw.item()
            epoch_raw_kl_loss += loss_kl_raw.item()
            epoch_raw_fraction_loss += fraction_loss_raw.item()
            epoch_raw_zinb_loss += zinb_loss_raw.item()

            # Weighted losses (for verification)
            epoch_weighted_recon_loss += loss_recon_weighted.item()
            epoch_weighted_kl_loss += loss_kl_weighted.item()
            epoch_weighted_fraction_loss += fraction_loss_weighted.item()
            epoch_weighted_zinb_loss += zinb_loss_weighted.item()

            # Update batch progress bar with current loss
            batch_iterator.set_postfix(
                {
                    "Loss": f"{loss.item():.3e}",
                    "Recon": f"{loss_recon_raw.item():.2e}",
                    "KL": f"{loss_kl_raw.item():.2e}",
                }
            )

        # Close batch progress bar
        batch_iterator.close()

        print("Epoch finished")

        # Average losses over batches (to match validation loss calculation)
        num_batches = len(dataloader)
        avg_total_loss = total_loss / num_batches

        # Raw (unweighted) averages
        avg_raw_recon_loss = epoch_raw_recon_loss / num_batches
        avg_raw_kl_loss = epoch_raw_kl_loss / num_batches
        avg_raw_fraction_loss = epoch_raw_fraction_loss / num_batches
        avg_raw_zinb_loss = epoch_raw_zinb_loss / num_batches

        # Weighted averages (for verification)
        avg_weighted_recon_loss = epoch_weighted_recon_loss / num_batches
        avg_weighted_kl_loss = epoch_weighted_kl_loss / num_batches
        avg_weighted_fraction_loss = epoch_weighted_fraction_loss / num_batches
        avg_weighted_zinb_loss = epoch_weighted_zinb_loss / num_batches

        # Average correlation
        avg_correlation = (
            epoch_correlation_sum / epoch_correlation_count
            if epoch_correlation_count > 0
            else 0.0
        )

        # Create metrics object
        recon_warmup_weight = self._reconstruction_warmup_weight(epoch)
        metrics = GMVAEMetrics(
            epoch=epoch,
            total_loss=avg_total_loss,
            reconstruction_loss=avg_raw_recon_loss,  # Store RAW reconstruction loss
            kl_loss=avg_raw_kl_loss,  # Store RAW KL loss
            fraction_loss=avg_raw_fraction_loss,  # Store RAW fraction loss
            zinb_loss=avg_raw_zinb_loss,  # Store RAW ZINB loss
            kl_weight=kl_weight,
            recon_warmup_weight=recon_warmup_weight,
            reconstruction_correlation=avg_correlation,  # Store correlation for model collapse detection
            # Store weighted losses for verification
            weighted_reconstruction_loss=avg_weighted_recon_loss,
            weighted_kl_loss=avg_weighted_kl_loss,
            weighted_fraction_loss=avg_weighted_fraction_loss,
            weighted_zinb_loss=avg_weighted_zinb_loss,
        )

        return metrics

    def _validate_epoch(
        self,
        epoch: int,
        val_dataloader,
        proportion_tensor: torch.Tensor,
        kl_weight: float,
    ) -> GMVAEMetrics:
        """Run validation for one epoch."""
        print(f"  Validation dataloader has {len(val_dataloader)} batches")
        self.model.eval()

        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_kl_loss = 0.0
        total_fraction_loss = 0.0
        total_zinb_loss = 0.0
        total_correlation = 0.0
        correlation_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                # Unpack batch from donor-grouped collate function
                data, cell_type_labels, num_cells_per_donor, donor_ids = batch
                data = data.to(self.device)
                cell_type_labels = cell_type_labels.to(self.device)

                # Validate input dimensions
                assert (
                    data.ndim == 2
                ), f"Expected data to be 2D (total_cells, n_genes), got {data.ndim}D with shape {data.shape}"
                assert (
                    cell_type_labels.ndim == 1
                ), f"Expected cell_type_labels to be 1D (total_cells,), got {cell_type_labels.ndim}D with shape {cell_type_labels.shape}"
                assert (
                    data.shape[0] == cell_type_labels.shape[0]
                ), f"Cell count mismatch: data has {data.shape[0]} cells but cell_type_labels has {cell_type_labels.shape[0]} labels"

                # Forward pass
                reconstructed, mus, logvars, pis, zs = self.model(
                    data, cell_type_labels
                )

                # Calculate losses using shared loss calculator (same as training)
                loss_components = self.loss_calculator.calculate_losses(
                    data=data,
                    reconstructed=reconstructed,
                    mus=mus,
                    logvars=logvars,
                    pis=pis,
                    proportion_tensor=proportion_tensor,
                    epoch=epoch,
                    kl_weight=kl_weight,
                )

                # Extract RAW (unweighted) components
                reconstruction_loss_raw = loss_components.reconstruction_loss_raw
                kl_loss_raw = loss_components.kl_loss_raw
                fraction_loss_raw = loss_components.fraction_loss_raw
                zinb_loss_raw = loss_components.zinb_loss_raw
                loss = loss_components.total_loss

                # Accumulate losses (simple sum, same as training)
                total_loss += loss.item()
                total_reconstruction_loss += reconstruction_loss_raw.item()
                total_kl_loss += kl_loss_raw.item()
                total_fraction_loss += fraction_loss_raw.item()
                total_zinb_loss += zinb_loss_raw.item()

                # Calculate correlation between original and reconstructed data
                data_flat = data.view(data.size(0), -1)
                recon_flat = reconstructed.view(reconstructed.size(0), -1)

                # Calculate per-sample correlation
                data_mean = data_flat.mean(dim=1, keepdim=True)
                recon_mean = recon_flat.mean(dim=1, keepdim=True)

                data_centered = data_flat - data_mean
                recon_centered = recon_flat - recon_mean

                numerator = (data_centered * recon_centered).sum(dim=1)
                data_std = torch.sqrt((data_centered**2).sum(dim=1))
                recon_std = torch.sqrt((recon_centered**2).sum(dim=1))
                denominator = data_std * recon_std

                # Avoid division by zero
                correlation = numerator / (denominator + 1e-8)
                batch_correlation = correlation.mean().item()

                total_correlation += batch_correlation
                correlation_count += 1

        # Average losses over batches (same as training)
        num_batches = len(val_dataloader)
        avg_loss = total_loss / num_batches
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_fraction_loss = total_fraction_loss / num_batches
        avg_zinb_loss = total_zinb_loss / num_batches
        avg_correlation = (
            total_correlation / correlation_count if correlation_count > 0 else 0.0
        )

        # Print final averaged validation losses
        print(f"  Validation Averages (over {num_batches} batches):")
        print(f"    Total Loss: {avg_loss:.4f}")
        print(f"    Reconstruction Loss: {avg_reconstruction_loss:.4f}")
        print(f"    KL Loss: {avg_kl_loss:.4f}")
        print(f"    Fraction Loss: {avg_fraction_loss:.4f}")
        print(f"    ZINB Loss: {avg_zinb_loss:.4f}")
        print(f"    Reconstruction Correlation: {avg_correlation:.4f}")

        # Set model back to training mode
        self.model.train()

        return GMVAEMetrics(
            epoch=epoch,
            total_loss=avg_loss,
            reconstruction_loss=avg_reconstruction_loss,
            kl_loss=avg_kl_loss,
            fraction_loss=avg_fraction_loss,
            zinb_loss=avg_zinb_loss,
            val_reconstruction_correlation=avg_correlation,
        )

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

        # Generate detailed loss grid visualization
        self.metrics_tracker.plot_detailed_loss_grid(
            self.config.base_dir / "gmvae_detailed_loss_grid.png", config=self.config
        )

        # Generate stacked loss contribution plot
        self.metrics_tracker.plot_stacked_loss_contributions(
            self.config.base_dir / "gmvae_loss_contributions.png", model_type="GMVAE"
        )

        # Generate reconstruction distribution summary
        self._create_reconstruction_summary_figure()

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
            for batch in gmvae_dataloader:
                # Unpack batch from donor-grouped collate function
                data, cell_type_labels, num_cells_per_donor, donor_ids = batch
                data = data.to(self.device)
                cell_type_labels = cell_type_labels.to(self.device)

                # Forward pass to get parameters
                reconstructed, mus, logvars, pis, zs = self.model(
                    data, cell_type_labels
                )

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
            GMVAEMetrics.from_dict(data) for data in metrics_data
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

    def validate_model(self, validation_loader) -> GMVAEMetrics:
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

        return GMVAEMetrics(
            epoch=-1,  # Validation epoch
            total_loss=total_loss,
            reconstruction_loss=total_recon_loss,
            kl_loss=total_kl_loss,
        )

    def generate_visualizations(self, save_dir: Path):
        """Generate visualization plots."""
        if save_dir != self.config.base_dir:
            self.metrics_tracker.plot_detailed_loss_grid(
                save_dir / "gmvae_detailed_loss_grid.png", config=self.config
            )
