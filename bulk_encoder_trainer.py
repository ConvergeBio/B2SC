"""
Bulk encoder trainer for B2SC pipeline.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from termcolor import colored
from tqdm import tqdm

from config import B2SCConfig
from models import BulkEncoder, GaussianMixtureVAE
from metrics import MetricsTracker, BulkEncoderMetrics
from utilities import setup_device, print_model_summary
from loss import BulkEncoderLossCalculator, prepare_bulk_encoder_targets


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

        # Initialize loss calculator
        self.loss_calculator = BulkEncoderLossCalculator(config=self.config)

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
                torch.load(mus_path, weights_only=True)
                .to(self.device)
                .detach()
                .requires_grad_(False)
            )
            assert torch.any(self.scMus != 0), "scMus contains only zeros"

            self.scLogVars = (
                torch.load(logvars_path, weights_only=True)
                .to(self.device)
                .detach()
                .requires_grad_(False)
            )
            assert torch.any(self.scLogVars != 0), "scLogVars contains only zeros"

            self.scPis = (
                torch.load(pis_path, weights_only=True)
                .to(self.device)
                .detach()
                .requires_grad_(False)
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

    def freeze_gmvae(self):
        """
        Freeze GMVAE model to prevent gradient updates during BulkEncoder training.

        This ensures the GMVAE stays frozen when generating dynamic targets.
        """
        if self.gmvae_trainer is not None and self.gmvae_trainer.model is not None:
            self.gmvae_trainer.model.eval()
            for param in self.gmvae_trainer.model.parameters():
                param.requires_grad = False
            print("✅ GMVAE frozen (eval mode, no gradients)")
        else:
            print("⚠️  GMVAE trainer/model not available for freezing")

    def unfreeze_gmvae(self):
        """
        Unfreeze GMVAE model after BulkEncoder training is complete.

        Restores the GMVAE to trainable state (though typically not trained further).
        This is good practice for memory management and future use.
        """
        if self.gmvae_trainer is not None and self.gmvae_trainer.model is not None:
            self.gmvae_trainer.model.train()
            for param in self.gmvae_trainer.model.parameters():
                param.requires_grad = True
            print("✅ GMVAE unfrozen (train mode, gradients enabled)")
        else:
            print("⚠️  GMVAE trainer/model not available for unfreezing")

    def _get_dynamic_targets_from_gmvae(self, donor_ids, use_validation_data=False):
        """
        Generate sample-specific targets by passing constituent cells through frozen GMVAE.

        For each bulk sample (donor), this method:
        1. Retrieves the single cells that comprise the bulk
        2. Passes them through the frozen GMVAE encoder
        3. Averages the encoded representations to get donor-level targets

        Args:
            donor_ids: List of donor identifiers for the current batch
            use_validation_data: If True, use validation single-cell data

        Returns:
            tuple: (target_mus, target_logvars, target_pis)
                - target_mus: (batch_size, n_components, latent_dim)
                - target_logvars: (batch_size, n_components, latent_dim)
                - target_pis: (batch_size, n_components)
        """
        if self.gmvae_trainer is None or self.gmvae_trainer.model is None:
            raise ValueError(
                "GMVAE trainer/model not available. Cannot generate dynamic targets."
            )

        # Ensure GMVAE is in eval mode
        self.gmvae_trainer.model.eval()

        # Select appropriate dataset
        if use_validation_data and self.data_manager.sc_val_data is not None:
            sc_data = self.data_manager.sc_val_data
        else:
            sc_data = self.data_manager.sc_train_data

        batch_mus = []
        batch_logvars = []
        batch_pis = []

        with torch.no_grad():
            for donor_id in donor_ids:
                # Get cells for this donor
                donor_mask = sc_data.obs[self.config.pseudobulk_groupby] == donor_id
                donor_cells_obs = sc_data.obs[donor_mask]

                if len(donor_cells_obs) == 0:
                    raise ValueError(
                        f"Donor '{donor_id}' has no cells in the dataset. "
                        f"Check data consistency."
                    )

                # Get cell data and labels
                donor_indices = np.where(donor_mask)[0]
                donor_cell_matrix = sc_data.X[donor_indices]

                # Convert to dense if sparse
                if not isinstance(donor_cell_matrix, np.ndarray):
                    donor_cell_matrix = donor_cell_matrix.toarray()

                # Get cell type labels
                cell_type_labels = donor_cells_obs["cell_type_encoded"].values

                # Convert to tensors
                cells_tensor = torch.tensor(donor_cell_matrix, dtype=torch.float32).to(
                    self.device
                )
                labels_tensor = torch.tensor(cell_type_labels, dtype=torch.long).to(
                    self.device
                )

                # Pass through GMVAE encoder
                mus, logvars, pis = self.gmvae_trainer.model.encode(cells_tensor)
                # mus shape: (n_cells, n_components, latent_dim)
                # logvars shape: (n_cells, n_components, latent_dim)
                # pis shape: (n_cells, n_components)

                # Average across cells to get donor-level representation
                donor_mus = mus.mean(dim=0)  # (n_components, latent_dim)
                donor_logvars = logvars.mean(dim=0)  # (n_components, latent_dim)
                donor_pis = pis.mean(dim=0)  # (n_components,)

                batch_mus.append(donor_mus)
                batch_logvars.append(donor_logvars)
                batch_pis.append(donor_pis)

        # Stack to create batch tensors
        target_mus = torch.stack(batch_mus)  # (batch_size, n_components, latent_dim)
        target_logvars = torch.stack(
            batch_logvars
        )  # (batch_size, n_components, latent_dim)
        target_pis = torch.stack(batch_pis)  # (batch_size, n_components)

        return target_mus, target_logvars, target_pis

    def _train_epoch(self, epoch: int, dataloader) -> BulkEncoderMetrics:
        """
        Dispatcher for training one epoch - calls static or dynamic version.

        Uses dynamic targets if use_dynamic_targets=True in config, otherwise uses static.
        """
        if self.config.use_dynamic_targets:
            return self._train_epoch_dynamic(epoch, dataloader)
        else:
            return self._train_epoch_static(epoch, dataloader)

    def _validate_epoch(self, epoch: int, val_dataloader) -> BulkEncoderMetrics:
        """
        Dispatcher for validation one epoch - calls static or dynamic version.

        Uses dynamic targets if use_dynamic_targets=True in config, otherwise uses static.
        """
        if self.config.use_dynamic_targets:
            return self._validate_epoch_dynamic(epoch, val_dataloader)
        else:
            return self._validate_epoch_static(epoch, val_dataloader)

    def initialize_model(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_components: int,
        load_from_existing: bool = False,
    ) -> BulkEncoder:
        """
        Initialize BulkEncoder model with proper architecture.

        Args:
            input_dim: Number of input features (genes)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            n_components: Number of mixture components
            load_from_existing: Whether to load from existing checkpoint (default: False)

        Returns:
            Initialized BulkEncoder model
        """
        print(colored("🔬 Initializing BulkEncoder model...", "yellow"))
        self.model = BulkEncoder(input_dim, hidden_dim, latent_dim, n_components)

        # Check if we should load from existing checkpoint
        if load_from_existing:
            checkpoint_path = self.config.base_dir / "bulkEncoder_model.pt"
            if checkpoint_path.exists():
                try:
                    encoder_state_dict = torch.load(checkpoint_path, weights_only=True)
                    self.model.load_state_dict(encoder_state_dict, strict=True)
                    print(
                        colored(
                            "✅ Loaded existing BulkEncoder checkpoint from bulkEncoder_model.pt",
                            "green",
                        )
                    )
                except Exception as e:
                    print(
                        colored(f"⚠️  Failed to load BulkEncoder checkpoint: {e}", "red")
                    )
                    print(
                        colored(
                            "🔄 Initializing BulkEncoder with fresh weights instead",
                            "yellow",
                        )
                    )
                    self._initialize_fresh_weights()
            else:
                print(
                    colored(
                        "⚠️  No existing BulkEncoder checkpoint found - initializing with fresh weights",
                        "yellow",
                    )
                )
                self._initialize_fresh_weights()
        else:
            print(
                colored(
                    "🆕 Initializing BulkEncoder with fresh weights (default behavior)",
                    "cyan",
                )
            )
            self._initialize_fresh_weights()

        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.bulk_encoder_lr
        )

        # Print model summary
        print_model_summary(self.model, "BulkEncoder")

        print("BulkEncoder model initialized")
        return self.model

    def _initialize_fresh_weights(self):
        """Initialize model weights from scratch."""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        print(colored("🎲 Initialized BulkEncoder with Xavier normal weights", "cyan"))

    def train(self, dataloader, val_dataloader=None) -> MetricsTracker:
        """
        Train the BulkEncoder model.

        Args:
            dataloader: DataLoader with single-cell data
            val_dataloader: Optional validation DataLoader

        Returns:
            MetricsTracker: Complete training history and metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        if self.scMus is None or self.scLogVars is None or self.scPis is None:
            if not self.load_gmvae_parameters():
                raise ValueError("Failed to load GMVAE parameters. Train GMVAE first.")

        self.model.train()

        # Freeze GMVAE if using dynamic targets
        if self.config.use_dynamic_targets:
            print("🔒 Freezing GMVAE for dynamic target generation...")
            print("Using dynamic targets for BulkEncoder training...")
            self.freeze_gmvae()

        # Create progress bar for epochs
        epoch_pbar = tqdm(
            range(self.config.bulk_encoder_epochs),
            desc="BulkEncoder Training",
            unit="epoch",
        )

        for epoch in epoch_pbar:

            # Train one epoch
            epoch_metrics = self._train_epoch(epoch, dataloader)

            # Run validation if provided
            if val_dataloader is not None:
                val_metrics = self._validate_epoch(epoch, val_dataloader)
                # Add validation metrics to the epoch metrics
                epoch_metrics.val_total_loss = val_metrics.total_loss
                epoch_metrics.val_mus_loss = val_metrics.mus_loss
                epoch_metrics.val_logvars_loss = val_metrics.logvars_loss
                epoch_metrics.val_pis_loss = val_metrics.pis_loss
                epoch_metrics.val_uniform_penalty = val_metrics.uniform_penalty
            else:
                print(f"⚠️  No validation dataloader provided for epoch {epoch}")

            # Add metrics to tracker
            self.metrics_tracker.add_epoch_metrics(epoch_metrics)

            # Update progress bar with weighted loss information
            progress_info = {
                "Train Loss": f"{epoch_metrics.total_loss:.3e}",
                "Mus": f"{epoch_metrics.weighted_mus_loss:.3e}",
                "LogVars": f"{epoch_metrics.weighted_logvars_loss:.3e}",
                "Pis": f"{epoch_metrics.weighted_pis_loss:.3e}",
                "Uniform": f"{epoch_metrics.weighted_uniform_penalty:.3e}",
            }
            if epoch_metrics.val_total_loss is not None:
                progress_info["Val Loss"] = f"{epoch_metrics.val_total_loss:.3e}"

            epoch_pbar.set_postfix(progress_info)

            # Save intermediate checkpoints
            if ((epoch + 1) % 20 == 0) and (
                epoch != self.config.bulk_encoder_epochs - 1
            ):
                self._save_intermediate_checkpoint(epoch)

            # Save final checkpoint
            elif epoch == self.config.bulk_encoder_epochs - 1:
                self._save_final_checkpoint(epoch)

        # Unfreeze GMVAE if it was frozen
        if self.config.use_dynamic_targets:
            print("🔓 Unfreezing GMVAE after training...")
            self.unfreeze_gmvae()

        return self.metrics_tracker

    def _train_epoch_static(self, epoch: int, dataloader) -> BulkEncoderMetrics:
        """
        Train one epoch using static global targets (original approach).

        Uses pre-computed global average scMus and scLogVars for all samples.
        """
        total_loss = 0
        epoch_mus_loss = 0
        epoch_logvars_loss = 0
        epoch_pis_loss = 0
        epoch_uniform_penalty = 0

        for batch_idx, (data, donor_ids) in enumerate(dataloader):
            data = data.to(self.device)

            # Note: data is already bulk data from DonorGroupedDataset
            # No need to sum - each row is already a summed donor sample
            bulk_data = data

            # Forward pass
            mus, logvars, pis = self.model(bulk_data)

            # Get per-sample targets for this batch (training data)
            batch_target_pis = self._get_batch_target_proportions(
                donor_ids, use_validation_data=False
            )
            batch_target_pis = batch_target_pis.to(self.device)

            # Prepare targets using shared helper function
            batch_size = mus.shape[0]
            target_mus, target_logvars = prepare_bulk_encoder_targets(
                batch_size, self.scMus, self.scLogVars
            )

            # Calculate losses using shared loss calculator
            loss_components = self.loss_calculator.calculate_losses(
                pred_mus=mus,
                pred_logvars=logvars,
                pred_pis=pis,
                target_mus=target_mus,
                target_logvars=target_logvars,
                target_pis=batch_target_pis,
            )

            # Extract individual components for backward compatibility
            mus_loss = loss_components.mus_loss
            logvars_loss = loss_components.logvars_loss
            pis_loss = loss_components.pis_loss
            uniform_penalty = loss_components.uniform_penalty
            loss = loss_components.total_loss

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

        # Calculate weighted losses for verification
        weighted_mus = self.config.bulk_mus_loss_weight * epoch_mus_loss
        weighted_logvars = self.config.bulk_logvars_loss_weight * epoch_logvars_loss
        weighted_pis = self.config.bulk_pis_loss_weight * epoch_pis_loss
        weighted_uniform = (
            self.config.bulk_uniform_penalty_weight * epoch_uniform_penalty
        )

        # Create metrics object
        metrics = BulkEncoderMetrics(
            epoch=epoch,
            total_loss=total_loss,
            mus_loss=epoch_mus_loss,
            logvars_loss=epoch_logvars_loss,
            pis_loss=epoch_pis_loss,
            uniform_penalty=epoch_uniform_penalty,
            # Store weighted losses for verification and plotting
            weighted_mus_loss=weighted_mus,
            weighted_logvars_loss=weighted_logvars,
            weighted_pis_loss=weighted_pis,
            weighted_uniform_penalty=weighted_uniform,
        )

        return metrics

    def _train_epoch_dynamic(self, epoch: int, dataloader) -> BulkEncoderMetrics:
        """
        Train one epoch using dynamic sample-specific targets from GMVAE.

        For each bulk sample, passes its constituent single cells through the frozen
        GMVAE encoder to generate sample-specific mus, logvars, and pis as targets.
        """
        total_loss = 0
        epoch_mus_loss = 0
        epoch_logvars_loss = 0
        epoch_pis_loss = 0
        epoch_uniform_penalty = 0

        for batch_idx, (data, donor_ids) in enumerate(dataloader):
            data = data.to(self.device)

            # Note: data is already bulk data from DonorGroupedDataset
            bulk_data = data

            # Forward pass
            pred_mus, pred_logvars, pred_pis = self.model(bulk_data)

            # Get dynamic targets from GMVAE for this batch (training data)
            target_mus, target_logvars, target_pis = (
                self._get_dynamic_targets_from_gmvae(
                    donor_ids, use_validation_data=False
                )
            )

            # Move targets to device
            target_mus = target_mus.to(self.device)
            target_logvars = target_logvars.to(self.device)
            target_pis = target_pis.to(self.device)

            # Calculate losses using shared loss calculator
            loss_components = self.loss_calculator.calculate_losses(
                pred_mus=pred_mus,
                pred_logvars=pred_logvars,
                pred_pis=pred_pis,
                target_mus=target_mus,
                target_logvars=target_logvars,
                target_pis=target_pis,
            )

            # Extract individual components
            mus_loss = loss_components.mus_loss
            logvars_loss = loss_components.logvars_loss
            pis_loss = loss_components.pis_loss
            uniform_penalty = loss_components.uniform_penalty
            loss = loss_components.total_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            epoch_mus_loss += mus_loss.item()
            epoch_logvars_loss += logvars_loss.item()
            epoch_pis_loss += pis_loss.item()
            epoch_uniform_penalty += uniform_penalty.item()

        # Print epoch summary
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

        # Calculate weighted losses for verification
        weighted_mus = self.config.bulk_mus_loss_weight * epoch_mus_loss
        weighted_logvars = self.config.bulk_logvars_loss_weight * epoch_logvars_loss
        weighted_pis = self.config.bulk_pis_loss_weight * epoch_pis_loss
        weighted_uniform = (
            self.config.bulk_uniform_penalty_weight * epoch_uniform_penalty
        )

        # Create metrics object
        metrics = BulkEncoderMetrics(
            epoch=epoch,
            total_loss=total_loss,
            mus_loss=epoch_mus_loss,
            logvars_loss=epoch_logvars_loss,
            pis_loss=epoch_pis_loss,
            uniform_penalty=epoch_uniform_penalty,
            weighted_mus_loss=weighted_mus,
            weighted_logvars_loss=weighted_logvars,
            weighted_pis_loss=weighted_pis,
            weighted_uniform_penalty=weighted_uniform,
        )

        return metrics

    def _validate_epoch_static(self, epoch: int, val_dataloader) -> BulkEncoderMetrics:
        """
        Run validation for one epoch using static global targets (original approach).

        Uses pre-computed global average scMus and scLogVars for all samples.
        """
        self.model.eval()

        total_loss = 0.0
        total_mus_loss = 0.0
        total_logvars_loss = 0.0
        total_pis_loss = 0.0
        total_uniform_penalty = 0.0
        num_batches = 0
        total_samples = 0  # Track total samples for proper averaging

        with torch.no_grad():
            for batch_idx, (bulk_data, donor_ids) in enumerate(val_dataloader):
                bulk_data = bulk_data.to(self.device)

                # Forward pass
                pred_mus, pred_logvars, pred_pis = self.model(bulk_data)

                # Get per-sample targets for this batch (validation data)
                batch_target_pis = self._get_batch_target_proportions(
                    donor_ids, use_validation_data=True
                )
                batch_target_pis = batch_target_pis.to(self.device)

                # Prepare targets using shared helper function
                batch_size = pred_mus.shape[0]
                target_mus, target_logvars = prepare_bulk_encoder_targets(
                    batch_size, self.scMus, self.scLogVars
                )

                # Calculate losses using shared loss calculator (same as training)
                loss_components = self.loss_calculator.calculate_losses(
                    pred_mus=pred_mus,
                    pred_logvars=pred_logvars,
                    pred_pis=pred_pis,
                    target_mus=target_mus,
                    target_logvars=target_logvars,
                    target_pis=batch_target_pis,
                )

                # Extract individual components for backward compatibility
                mus_loss = loss_components.mus_loss
                logvars_loss = loss_components.logvars_loss
                pis_loss = loss_components.pis_loss
                uniform_penalty = loss_components.uniform_penalty
                loss = loss_components.total_loss

                # Accumulate losses (multiply by batch size for proper averaging)
                batch_size = bulk_data.shape[0]
                total_loss += loss.item() * batch_size
                total_mus_loss += mus_loss.item() * batch_size
                total_logvars_loss += logvars_loss.item() * batch_size
                total_pis_loss += pis_loss.item() * batch_size
                total_uniform_penalty += uniform_penalty.item() * batch_size
                num_batches += 1
                total_samples += batch_size

        # Average losses (divide by total samples, not batches)
        avg_loss = total_loss / total_samples
        avg_mus_loss = total_mus_loss / total_samples
        avg_logvars_loss = total_logvars_loss / total_samples
        avg_pis_loss = total_pis_loss / total_samples
        avg_uniform_penalty = total_uniform_penalty / total_samples

        # Set model back to training mode
        self.model.train()

        return BulkEncoderMetrics(
            epoch=epoch,
            total_loss=avg_loss,
            mus_loss=avg_mus_loss,
            logvars_loss=avg_logvars_loss,
            pis_loss=avg_pis_loss,
            uniform_penalty=avg_uniform_penalty,
        )

    def _validate_epoch_dynamic(self, epoch: int, val_dataloader) -> BulkEncoderMetrics:
        """
        Run validation for one epoch using dynamic sample-specific targets from GMVAE.

        For each bulk sample, passes its constituent single cells through the frozen
        GMVAE encoder to generate sample-specific mus, logvars, and pis as targets.
        """
        self.model.eval()

        total_loss = 0.0
        total_mus_loss = 0.0
        total_logvars_loss = 0.0
        total_pis_loss = 0.0
        total_uniform_penalty = 0.0
        num_batches = 0
        total_samples = 0  # Track total samples for proper averaging

        with torch.no_grad():
            for batch_idx, (bulk_data, donor_ids) in enumerate(val_dataloader):
                bulk_data = bulk_data.to(self.device)

                # Forward pass
                pred_mus, pred_logvars, pred_pis = self.model(bulk_data)

                # Get dynamic targets from GMVAE for this batch (validation data)
                target_mus, target_logvars, target_pis = (
                    self._get_dynamic_targets_from_gmvae(
                        donor_ids, use_validation_data=True
                    )
                )

                # Move targets to device
                target_mus = target_mus.to(self.device)
                target_logvars = target_logvars.to(self.device)
                target_pis = target_pis.to(self.device)

                # Calculate losses using shared loss calculator (same as training)
                loss_components = self.loss_calculator.calculate_losses(
                    pred_mus=pred_mus,
                    pred_logvars=pred_logvars,
                    pred_pis=pred_pis,
                    target_mus=target_mus,
                    target_logvars=target_logvars,
                    target_pis=target_pis,
                )

                # Extract individual components
                mus_loss = loss_components.mus_loss
                logvars_loss = loss_components.logvars_loss
                pis_loss = loss_components.pis_loss
                uniform_penalty = loss_components.uniform_penalty
                loss = loss_components.total_loss

                # Accumulate losses (multiply by batch size for proper averaging)
                batch_size = bulk_data.shape[0]
                total_loss += loss.item() * batch_size
                total_mus_loss += mus_loss.item() * batch_size
                total_logvars_loss += logvars_loss.item() * batch_size
                total_pis_loss += pis_loss.item() * batch_size
                total_uniform_penalty += uniform_penalty.item() * batch_size
                num_batches += 1
                total_samples += batch_size

        # Average losses (divide by total samples, not batches)
        avg_loss = total_loss / total_samples
        avg_mus_loss = total_mus_loss / total_samples
        avg_logvars_loss = total_logvars_loss / total_samples
        avg_pis_loss = total_pis_loss / total_samples
        avg_uniform_penalty = total_uniform_penalty / total_samples

        # Set model back to training mode
        self.model.train()

        return BulkEncoderMetrics(
            epoch=epoch,
            total_loss=avg_loss,
            mus_loss=avg_mus_loss,
            logvars_loss=avg_logvars_loss,
            pis_loss=avg_pis_loss,
            uniform_penalty=avg_uniform_penalty,
        )

    def _get_batch_target_proportions(self, donor_ids, use_validation_data=False):
        """
        Get target cell type proportions for a batch of donors.

        This function calculates the true cell type proportions for each donor in the batch
        by analyzing the single-cell data. These proportions serve as targets for training
        the BulkEncoder to predict mixture probabilities (pis).

        Args:
            donor_ids: List or iterable of donor identifiers corresponding to the current batch.
                      These should match the values in the pseudobulk_groupby column of the
                      single-cell data.
            use_validation_data (bool): If True and validation data is available, use
                                      sc_val_data instead of sc_train_data. Defaults to False.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, n_cell_types) containing the
                         cell type proportions for each donor. Each row sums to 1.0 and
                         represents the true mixture of cell types for that donor.
                         The order of cell types matches self.data_manager.cell_type_mapping.

        Raises:
            ValueError: If a donor_id has no corresponding cells in the dataset, which
                       indicates a data consistency issue between bulk and single-cell data.
        """
        batch_proportions = []

        # Use appropriate dataset based on context
        if use_validation_data and self.data_manager.sc_val_data is not None:
            sc_data = self.data_manager.sc_val_data
        else:
            sc_data = self.data_manager.sc_train_data

        for donor_id in donor_ids:
            # Get cells for this donor
            donor_mask = sc_data.obs[self.config.pseudobulk_groupby] == donor_id
            donor_cells = sc_data.obs[donor_mask]

            # Calculate cell type proportions for this donor
            total_cells = len(donor_cells)

            # Handle edge case: donor with no cells
            if total_cells == 0:
                raise ValueError(
                    f"Donor '{donor_id}' has no cells in the dataset. "
                    f"This should not happen as bulk data is created by summing donor cells. "
                    f"Check data consistency between single-cell data and bulk dataloader."
                )

            cell_type_counts = donor_cells["cell_type"].value_counts()

            # Create proportion vector matching the order of cell types in mapping
            proportions = torch.zeros(len(self.data_manager.cell_type_mapping))
            for cell_type, count in cell_type_counts.items():
                if cell_type in self.data_manager.cell_type_mapping:
                    cell_type_idx = self.data_manager.cell_type_mapping[cell_type]
                    proportions[cell_type_idx] = count / total_cells

            batch_proportions.append(proportions)

        result = torch.stack(batch_proportions)

        # Assert that each row sums to approximately 1.0
        row_sums = result.sum(dim=1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-6
        ), f"Cell type proportions do not sum to 1.0. Row sums: {row_sums}"

        return result

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

        # Generate detailed loss grid visualization
        self.metrics_tracker.plot_detailed_loss_grid(
            self.config.base_dir / "bulk_encoder_detailed_loss_grid.png",
            config=self.config,
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
            BulkEncoderMetrics.from_dict(data) for data in metrics_data
        ]

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return True

    def validate_model(self, validation_loader) -> BulkEncoderMetrics:
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

                # Calculate losses (same reduction as other methods)
                mus_loss = F.mse_loss(mus, self.scMus, reduction="mean")
                logvars_loss = F.mse_loss(logvars, self.scLogVars, reduction="mean")
                pis_loss = F.mse_loss(pis, self.scPis, reduction="mean")

                # C = pis.shape[0]
                # uniform = torch.full_like(pis, 1.0 / C)
                # uniform_penalty = F.cosine_similarity(
                #     pis.unsqueeze(0), uniform.unsqueeze(0)
                # ).squeeze()
                uniform_penalty = torch.tensor(0.0)  # Placeholder

                loss = (
                    mus_loss
                    + logvars_loss
                    + (pis_loss * 50)  # + (uniform_penalty * 100)  # COMMENTED OUT
                )

                total_loss += loss.item()
                total_mus_loss += mus_loss.item()
                total_logvars_loss += logvars_loss.item()
                total_pis_loss += pis_loss.item()

        return BulkEncoderMetrics(
            epoch=-1,  # Validation epoch
            total_loss=total_loss,
            mus_loss=total_mus_loss,
            logvars_loss=total_logvars_loss,
            pis_loss=total_pis_loss,
            uniform_penalty=0.0,  # Not calculated in validation
        )

    def generate_visualizations(self, save_dir: Path):
        """Generate detailed loss grid and other visualizations."""
        if save_dir != self.config.base_dir:
            self.metrics_tracker.plot_detailed_loss_grid(
                save_dir / "bulk_encoder_detailed_loss_grid.png", config=self.config
            )
