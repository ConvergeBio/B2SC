"""
Training metrics tracking and management for B2SC pipeline.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import pickle


@dataclass
class TrainingMetrics:
    """Base container for training metrics."""

    epoch: int
    total_loss: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/saving."""
        result = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is not None:
                result[field] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingMetrics":
        """Create from dictionary."""
        # Filter data to only include fields that exist in this class
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class GMVAEMetrics(TrainingMetrics):
    """Training metrics for GMVAE model."""

    reconstruction_loss: float  # Raw (unweighted) reconstruction loss
    kl_loss: float  # Raw (unweighted) KL loss
    fraction_loss: Optional[float] = None  # Raw (unweighted) fraction loss
    zinb_loss: Optional[float] = None  # Raw (unweighted) ZINB loss
    kl_weight: Optional[float] = None
    recon_warmup_weight: Optional[float] = None  # Dynamic warmup weight

    # Weighted losses (for verification and debugging)
    weighted_reconstruction_loss: Optional[float] = None
    weighted_kl_loss: Optional[float] = None
    weighted_fraction_loss: Optional[float] = None
    weighted_zinb_loss: Optional[float] = None

    # Correlation metrics (for model collapse detection)
    reconstruction_correlation: Optional[float] = (
        None  # Train reconstruction correlation
    )
    val_reconstruction_correlation: Optional[float] = (
        None  # Validation reconstruction correlation
    )

    # Validation metrics (optional)
    val_total_loss: Optional[float] = None
    val_reconstruction_loss: Optional[float] = None
    val_kl_loss: Optional[float] = None
    val_fraction_loss: Optional[float] = None
    val_zinb_loss: Optional[float] = None


@dataclass
class BulkEncoderMetrics(TrainingMetrics):
    """Training metrics for BulkEncoder model."""

    mus_loss: float  # Raw (unweighted) mus loss
    logvars_loss: float  # Raw (unweighted) logvars loss
    pis_loss: float  # Raw (unweighted) pis loss
    uniform_penalty: float  # Raw (unweighted) uniform penalty

    # Weighted losses (for verification and debugging)
    weighted_mus_loss: Optional[float] = None
    weighted_logvars_loss: Optional[float] = None
    weighted_pis_loss: Optional[float] = None
    weighted_uniform_penalty: Optional[float] = None

    # Validation metrics (optional)
    val_total_loss: Optional[float] = None
    val_mus_loss: Optional[float] = None
    val_logvars_loss: Optional[float] = None
    val_pis_loss: Optional[float] = None
    val_uniform_penalty: Optional[float] = None


def calculate_total_losses_from_components(
    raw_losses: List[List[float]],
    weighted_losses: List[List[float]],
    loss_names: List[str],
) -> tuple:
    """
    Helper function to calculate total losses from individual components.

    Args:
        raw_losses: List of raw loss lists for each component
        weighted_losses: List of weighted loss lists for each component
        loss_names: Names of loss components for debugging

    Returns:
        tuple: (calculated_total_unweighted, calculated_total_weighted)
    """
    if not raw_losses or not weighted_losses:
        return [], []

    num_epochs = len(raw_losses[0])
    calculated_total_unweighted = []
    calculated_total_weighted = []

    for epoch_idx in range(num_epochs):
        # Sum all raw losses for this epoch
        total_unweighted = sum(losses[epoch_idx] for losses in raw_losses)
        calculated_total_unweighted.append(total_unweighted)

        # Sum all weighted losses for this epoch (handle None values)
        epoch_weighted_values = []
        for losses in weighted_losses:
            value = losses[epoch_idx]
            if value is not None:
                epoch_weighted_values.append(value)
        total_weighted = sum(epoch_weighted_values) if epoch_weighted_values else 0.0
        calculated_total_weighted.append(total_weighted)

    return calculated_total_unweighted, calculated_total_weighted


def verify_loss_calculations(
    stored_totals: List[float],
    calculated_weighted: List[float],
    epochs: List[int],
    model_type: str = "Model",
) -> None:
    """
    Helper function to verify that stored total losses match calculated weighted totals.

    Args:
        stored_totals: Total losses as stored in metrics
        calculated_weighted: Total losses calculated from weighted components
        epochs: Epoch numbers for display
        model_type: Model type name for display (e.g., "GMVAE", "BulkEncoder")
    """
    print(f"🔍 Verifying {model_type} loss calculations...")
    for i, epoch in enumerate(epochs):
        stored_total = stored_totals[i]
        calc_weighted = calculated_weighted[i]

        diff_abs = abs(stored_total - calc_weighted)
        diff_rel = diff_abs / max(abs(stored_total), 1e-8)

        if diff_rel > 0.05:  # 5% tolerance
            print(
                f"⚠️  Epoch {epoch}: Stored total ({stored_total:.2e}) != Sum of weighted ({calc_weighted:.2e})"
            )
            print(f"   Absolute difference: {diff_abs:.2e}")
            print(f"   Relative difference: {diff_rel:.1%}")
            print(
                f"   Note: Large discrepancy may indicate fallback loss calculation was used"
            )
        else:
            print(
                f"✅ Epoch {epoch}: Total loss matches sum of weighted components (diff: {diff_rel:.1%})"
            )


class MetricsTracker:
    """Tracks and manages training metrics over time."""

    def __init__(self):
        self.history: List[Union[TrainingMetrics, GMVAEMetrics, BulkEncoderMetrics]] = (
            []
        )

    def add_epoch_metrics(
        self, metrics: Union[TrainingMetrics, GMVAEMetrics, BulkEncoderMetrics]
    ):
        """Add metrics for an epoch."""
        self.history.append(metrics)

    def get_best_epoch(self, metric: str = "total_loss") -> int:
        """Get epoch with best performance."""
        if not self.history:
            return -1

        if metric == "total_loss":
            # Lower is better for loss
            best_idx = min(
                range(len(self.history)), key=lambda i: self.history[i].total_loss
            )
        else:
            # Assume lower is better for other metrics too
            best_idx = min(
                range(len(self.history)),
                key=lambda i: getattr(self.history[i], metric, float("inf")),
            )

        return self.history[best_idx].epoch

    def plot_detailed_loss_grid(self, save_path: Path, config=None):
        """
        Generate detailed grid visualization for loss components.

        Grid layout:
        - Each row represents a loss type (total, reconstruction, KL, etc.)
        - Each column shows: absolute loss, log-scale loss, weighted loss, weight value
        - X-axis always shows epochs

        Args:
            save_path: Path to save the plot
            config: Configuration object to extract loss weights
        """
        if not self.history:
            return

        # Detect model type based on available losses
        is_bulk_encoder = any(
            getattr(m, "mus_loss", None) is not None for m in self.history
        )

        if is_bulk_encoder:
            self._plot_bulk_encoder_grid(save_path, config)
        else:
            self._plot_gmvae_grid(save_path, config)

    def _plot_gmvae_grid(self, save_path: Path, config=None):
        """Generate GMVAE detailed grid (5 rows x 4 columns)."""
        import numpy as np

        # Convert epochs to 1-based indexing for display
        epochs = [m.epoch + 1 for m in self.history]

        # Extract loss data
        total_losses = [m.total_loss for m in self.history]
        recon_losses = [m.reconstruction_loss for m in self.history]
        kl_losses = [m.kl_loss for m in self.history]
        fraction_losses = [getattr(m, "fraction_loss", 0) or 0 for m in self.history]
        zinb_losses = [getattr(m, "zinb_loss", 0) or 0 for m in self.history]

        # Extract dynamic weights from metrics
        warmup_weights = [
            getattr(m, "recon_warmup_weight", 1.0) or 1.0 for m in self.history
        ]
        # Extract dynamic KL weights (changes due to annealing)
        dynamic_kl_weights = [getattr(m, "kl_weight", None) for m in self.history]

        if config:
            static_recon_weight = config.gmvae_recon_loss_weight
            static_kl_weight = config.gmvae_kl_loss_weight  # Fallback only
            fraction_weight = config.gmvae_fraction_loss_weight
            zinb_weight = config.gmvae_zinb_loss_weight
        else:
            # Default weights if config not available
            static_recon_weight = 1.0
            static_kl_weight = 1.0  # Fallback only
            fraction_weight = 1.0
            zinb_weight = 1.0

        # Use dynamic KL weights if available, otherwise fall back to static
        # IMPORTANT: effective_kl_weights = annealing_weight × configured_weight
        if any(w is not None for w in dynamic_kl_weights):
            effective_kl_weights = [
                (w * static_kl_weight) if w is not None else static_kl_weight
                for w in dynamic_kl_weights
            ]
        else:
            effective_kl_weights = [static_kl_weight] * len(epochs)

        # Use stored weighted losses if available, otherwise calculate them
        weighted_recon = [
            getattr(m, "weighted_reconstruction_loss", 0) or 0 for m in self.history
        ]
        weighted_fraction = [
            getattr(m, "weighted_fraction_loss", 0) or 0 for m in self.history
        ]
        weighted_zinb = [getattr(m, "weighted_zinb_loss", 0) or 0 for m in self.history]

        # ALWAYS recalculate KL weighted loss using the annealed weights for consistency
        # This ensures the "KL Loss - Weighted" column matches the "KL Loss - Weight" column
        weighted_kl = [k * kl_w for k, kl_w in zip(kl_losses, effective_kl_weights)]

        # If other weighted losses are not stored, calculate them (fallback)
        if all(w == 0 for w in weighted_recon):
            weighted_recon = [
                r * static_recon_weight * w
                for r, w in zip(recon_losses, warmup_weights)
            ]
        if all(w == 0 for w in weighted_fraction):
            weighted_fraction = [f * fraction_weight for f in fraction_losses]
        if all(w == 0 for w in weighted_zinb):
            weighted_zinb = [z * zinb_weight for z in zinb_losses]

        # Effective weights (for reconstruction, this varies with warmup)
        effective_recon_weights = [static_recon_weight * w for w in warmup_weights]

        # Extract correlation data
        train_correlations = [
            getattr(m, "reconstruction_correlation", None) for m in self.history
        ]
        val_correlations = [
            getattr(m, "val_reconstruction_correlation", None) for m in self.history
        ]

        # Create 6x4 grid (added row for correlation)
        fig, axes = plt.subplots(6, 4, figsize=(25, 18))
        fig.suptitle(
            "GMVAE Training Metrics - Detailed Loss Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # Calculate total losses using helper function
        raw_loss_components = [recon_losses, kl_losses, fraction_losses, zinb_losses]
        weighted_loss_components = [
            weighted_recon,
            weighted_kl,
            weighted_fraction,
            weighted_zinb,
        ]
        loss_names = ["Reconstruction", "KL", "Fraction", "ZINB"]

        calculated_total_unweighted, calculated_total_weighted = (
            calculate_total_losses_from_components(
                raw_loss_components, weighted_loss_components, loss_names
            )
        )

        # Verify loss calculations using helper function
        verify_loss_calculations(
            total_losses, calculated_total_weighted, epochs, "GMVAE"
        )

        # Define loss data and labels
        # IMPORTANT: For "Total Loss - Absolute", we should show sum of RAW losses, not stored total
        loss_data = [
            (
                "Total Loss",
                calculated_total_unweighted,  # Sum of RAW losses (should be much higher)
                [1.0] * len(epochs),
                calculated_total_weighted,  # Sum of WEIGHTED losses (matches stored total)
            ),
            (
                "Reconstruction Loss",
                recon_losses,
                effective_recon_weights,
                weighted_recon,
            ),
            ("KL Loss", kl_losses, effective_kl_weights, weighted_kl),
            (
                "Fraction Loss",
                fraction_losses,
                [fraction_weight] * len(epochs),
                weighted_fraction,
            ),
            ("ZINB Loss", zinb_losses, [zinb_weight] * len(epochs), weighted_zinb),
        ]

        colors = ["blue", "red", "green", "orange", "purple"]

        for row, (loss_name, absolute_loss, weight, weighted_loss) in enumerate(
            loss_data
        ):
            color = colors[row]

            # Column 1: Absolute Loss
            axes[row, 0].plot(epochs, absolute_loss, color=color, linewidth=2)
            axes[row, 0].set_title(f"{loss_name} - Absolute")
            axes[row, 0].set_xlabel("Epoch")
            axes[row, 0].set_ylabel("Loss Value")
            axes[row, 0].grid(True, alpha=0.3)

            # Column 2: Log Scale Loss
            # Handle zeros and negative values for log scale
            log_loss = [max(l, 1e-10) for l in absolute_loss]  # Avoid log(0)
            axes[row, 1].semilogy(epochs, log_loss, color=color, linewidth=2)
            axes[row, 1].set_title(f"{loss_name} - Log Scale")
            axes[row, 1].set_xlabel("Epoch")
            axes[row, 1].set_ylabel("Loss Value (log)")
            axes[row, 1].grid(True, alpha=0.3)

            # Column 3: Weighted Loss
            axes[row, 2].plot(
                epochs, weighted_loss, color=color, linewidth=2, label="Training"
            )
            axes[row, 2].set_title(f"{loss_name} - Weighted")
            axes[row, 2].set_xlabel("Epoch")
            axes[row, 2].set_ylabel("Weighted Loss")
            axes[row, 2].grid(True, alpha=0.3)

            # Add validation losses to first 3 columns if available
            val_attr_map = {
                "Total Loss": "val_total_loss",
                "Reconstruction Loss": "val_reconstruction_loss",
                "KL Loss": "val_kl_loss",
                "Fraction Loss": "val_fraction_loss",
                "ZINB Loss": "val_zinb_loss",
            }

            if loss_name in val_attr_map:
                val_losses = [
                    getattr(m, val_attr_map[loss_name], None) for m in self.history
                ]
                if any(v is not None for v in val_losses):
                    val_epochs = [
                        epochs[i] for i, v in enumerate(val_losses) if v is not None
                    ]
                    val_values = [v for v in val_losses if v is not None]

                    # For "Total Loss - Absolute", calculate sum of RAW validation losses
                    # (val_total_loss is the weighted sum, not the raw sum)
                    if loss_name == "Total Loss":
                        val_absolute_values = []
                        for m in self.history:
                            if m.val_total_loss is not None:
                                # Sum of RAW validation component losses
                                raw_sum = (
                                    (m.val_reconstruction_loss or 0)
                                    + (m.val_kl_loss or 0)
                                    + (m.val_fraction_loss or 0)
                                    + (m.val_zinb_loss or 0)
                                )
                                val_absolute_values.append(raw_sum)
                        val_absolute_display = (
                            val_absolute_values if val_absolute_values else val_values
                        )
                    else:
                        # For component losses, use the raw values directly
                        val_absolute_display = val_values

                    # Add validation to absolute plot (Column 1)
                    axes[row, 0].plot(
                        val_epochs,
                        val_absolute_display,
                        color=color,
                        linewidth=2,
                        linestyle="--",
                        alpha=0.8,
                        label="Validation",
                    )

                    # Add validation to log plot (Column 2)
                    val_log_loss = [max(l, 1e-10) for l in val_absolute_display]
                    axes[row, 1].semilogy(
                        val_epochs,
                        val_log_loss,
                        color=color,
                        linewidth=2,
                        linestyle="--",
                        alpha=0.8,
                        label="Validation",
                    )

                    # Add validation to weighted plot (Column 3) - approximate weighting
                    if loss_name == "Total Loss":
                        val_weighted = val_values  # Total is already weighted
                    else:
                        # For component losses, apply approximate weighting
                        # Note: This is approximate since validation doesn't store weighted losses
                        if isinstance(weight, list):
                            avg_weight = sum(weight) / len(weight)
                        else:
                            avg_weight = weight
                        val_weighted = [v * avg_weight for v in val_values]

                    axes[row, 2].plot(
                        val_epochs,
                        val_weighted,
                        color=color,
                        linewidth=2,
                        linestyle="--",
                        alpha=0.8,
                        label="Validation",
                    )

            # Add legends to first 3 columns if validation data exists
            if loss_name in val_attr_map and any(
                getattr(m, val_attr_map[loss_name], None) is not None
                for m in self.history
            ):
                axes[row, 0].legend(fontsize=8)
                axes[row, 1].legend(fontsize=8)
                axes[row, 2].legend(fontsize=8)

            # Column 4: Weight Value (skip for Total Loss as it's redundant)
            if loss_name == "Total Loss":
                # For total loss, show a message instead of redundant weight=1.0
                axes[row, 3].text(
                    0.5,
                    0.5,
                    "Total Loss\n(sum of weighted\ncomponents)",
                    ha="center",
                    va="center",
                    transform=axes[row, 3].transAxes,
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5
                    ),
                )
                axes[row, 3].set_title(f"{loss_name} - Weight")
                axes[row, 3].set_xlabel("Epoch")
                axes[row, 3].set_ylabel("Weight Value")
                axes[row, 3].grid(True, alpha=0.3)
                axes[row, 3].set_xlim([min(epochs), max(epochs)])
                axes[row, 3].set_ylim([0, 1.2])
            else:
                # For component losses, show the actual weights
                if isinstance(weight, list):
                    weight_values = weight
                    max_weight = max(weight) if weight else 1.0
                else:
                    weight_values = [weight] * len(epochs)
                    max_weight = weight

                axes[row, 3].plot(
                    epochs, weight_values, color=color, linewidth=3, linestyle="--"
                )
                axes[row, 3].set_title(f"{loss_name} - Weight")
                axes[row, 3].set_xlabel("Epoch")
                axes[row, 3].set_ylabel("Weight Value")
                axes[row, 3].grid(True, alpha=0.3)
                axes[row, 3].set_ylim(
                    [0, max(max_weight * 1.1, 0.1)]
                )  # Set reasonable y-limits

        # Row 5: Reconstruction Correlation (Model Collapse Detection)
        row = 5
        color = "cyan"

        # Filter out None values for plotting
        train_corr_vals = [c if c is not None else 0.0 for c in train_correlations]
        val_corr_vals = [c if c is not None else 0.0 for c in val_correlations]

        has_train_corr = any(c is not None for c in train_correlations)
        has_val_corr = any(c is not None for c in val_correlations)

        # Column 1: Training Correlation
        if has_train_corr:
            axes[row, 0].plot(
                epochs, train_corr_vals, color=color, linewidth=2, label="Training"
            )
        axes[row, 0].set_title("Reconstruction Correlation - Training")
        axes[row, 0].set_xlabel("Epoch")
        axes[row, 0].set_ylabel("Pearson r")
        # Dynamic y-axis scaling to show subtle changes
        if has_train_corr:
            axes[row, 0].margins(y=0.1)  # Add 10% margin above/below data
        axes[row, 0].grid(True, alpha=0.3)
        if has_train_corr:
            axes[row, 0].legend(fontsize=8)

        # Column 2: Validation Correlation
        if has_val_corr:
            val_epochs = [
                epochs[i] for i, c in enumerate(val_correlations) if c is not None
            ]
            val_vals = [c for c in val_correlations if c is not None]
            if val_vals:
                axes[row, 1].plot(
                    val_epochs,
                    val_vals,
                    color=color,
                    linewidth=2,
                    linestyle="--",
                    label="Validation",
                )
        axes[row, 1].set_title("Reconstruction Correlation - Validation")
        axes[row, 1].set_xlabel("Epoch")
        axes[row, 1].set_ylabel("Pearson r")
        # Dynamic y-axis scaling to show subtle changes
        if has_val_corr and val_vals:
            axes[row, 1].margins(y=0.1)  # Add 10% margin above/below data
        axes[row, 1].grid(True, alpha=0.3)
        if has_val_corr and val_vals:
            axes[row, 1].legend(fontsize=8)

        # Column 3: Both correlations together
        if has_train_corr:
            axes[row, 2].plot(
                epochs, train_corr_vals, color=color, linewidth=2, label="Training"
            )
        if has_val_corr and val_vals:
            axes[row, 2].plot(
                val_epochs,
                val_vals,
                color=color,
                linewidth=2,
                linestyle="--",
                alpha=0.8,
                label="Validation",
            )
        axes[row, 2].set_title("Reconstruction Correlation - Combined")
        axes[row, 2].set_xlabel("Epoch")
        axes[row, 2].set_ylabel("Pearson r")
        # Dynamic y-axis scaling to show subtle changes
        if has_train_corr or (has_val_corr and val_vals):
            axes[row, 2].margins(y=0.1)  # Add 10% margin above/below data
        axes[row, 2].grid(True, alpha=0.3)
        if has_train_corr or (has_val_corr and val_vals):
            axes[row, 2].legend(fontsize=8)

        # Column 4: Correlation vs MSE (Model Collapse Detector)
        # Plot reconstruction loss vs correlation to detect model collapse
        # Good: correlation increases as MSE decreases
        # Bad: correlation flat/decreasing while MSE decreases (model collapse)
        ax_twin = axes[row, 3].twinx()
        if has_train_corr:
            axes[row, 3].plot(
                epochs, train_corr_vals, color=color, linewidth=2, label="Correlation"
            )
        # Plot MSE on the RIGHT y-axis (ax_twin)
        ax_twin.plot(
            epochs,
            recon_losses,
            color="red",
            linewidth=2,
            alpha=0.5,
            linestyle=":",
            label="MSE",
        )
        axes[row, 3].set_title("Collapse Detection: Corr vs MSE")
        axes[row, 3].set_xlabel("Epoch")
        axes[row, 3].set_ylabel("Correlation", color=color)
        axes[row, 3].tick_params(axis="y", labelcolor=color)
        ax_twin.set_ylabel("MSE Loss", color="red")
        ax_twin.tick_params(axis="y", labelcolor="red")
        # Dynamic y-axis scaling to show subtle changes
        if has_train_corr:
            axes[row, 3].margins(y=0.1)  # Add 10% margin above/below data
        axes[row, 3].grid(True, alpha=0.3)
        axes[row, 3].legend(loc="upper left", fontsize=8)
        # Only create MSE legend if we have data
        if len(recon_losses) > 0:
            ax_twin.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()  # Display in notebooks
        plt.close()

    def _plot_bulk_encoder_grid(self, save_path: Path, config=None):
        """Generate BulkEncoder detailed grid (4 rows x 4 columns)."""
        import numpy as np

        # Convert epochs to 1-based indexing for display
        epochs = [m.epoch + 1 for m in self.history]

        # Extract loss data
        total_losses = [m.total_loss for m in self.history]
        mus_losses = [m.mus_loss for m in self.history]
        logvars_losses = [m.logvars_loss for m in self.history]
        pis_losses = [m.pis_loss for m in self.history]
        uniform_penalties = [m.uniform_penalty for m in self.history]

        # Extract weights from config if available
        if config:
            mus_weight = config.bulk_mus_loss_weight
            logvars_weight = config.bulk_logvars_loss_weight
            pis_weight = config.bulk_pis_loss_weight
            uniform_weight = config.bulk_uniform_penalty_weight
        else:
            # Default weights if config not available
            mus_weight = 1.0
            logvars_weight = 1.0
            pis_weight = 1.0
            uniform_weight = 1.0

        # Calculate weighted losses (with fallback to stored values if available)
        weighted_mus = []
        weighted_logvars = []
        weighted_pis = []
        weighted_uniform = []

        for m in self.history:
            # Use stored weighted values if available, otherwise calculate
            weighted_mus.append(
                getattr(m, "weighted_mus_loss", m.mus_loss * mus_weight)
            )
            weighted_logvars.append(
                getattr(m, "weighted_logvars_loss", m.logvars_loss * logvars_weight)
            )
            weighted_pis.append(
                getattr(m, "weighted_pis_loss", m.pis_loss * pis_weight)
            )
            weighted_uniform.append(
                getattr(
                    m, "weighted_uniform_penalty", m.uniform_penalty * uniform_weight
                )
            )

        # Create 5x4 grid (added uniform penalty row)
        fig, axes = plt.subplots(5, 4, figsize=(25, 15))
        fig.suptitle(
            "BulkEncoder Training Metrics - Detailed Loss Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # Calculate total losses using helper function
        raw_loss_components = [
            mus_losses,
            logvars_losses,
            pis_losses,
            uniform_penalties,
        ]
        weighted_loss_components = [
            weighted_mus,
            weighted_logvars,
            weighted_pis,
            weighted_uniform,
        ]
        loss_names = ["Mus", "LogVars", "Pis", "Uniform"]

        calculated_total_unweighted, calculated_total_weighted = (
            calculate_total_losses_from_components(
                raw_loss_components, weighted_loss_components, loss_names
            )
        )

        # Verify loss calculations using helper function
        verify_loss_calculations(
            total_losses, calculated_total_weighted, epochs, "BulkEncoder"
        )

        # Define loss data and labels
        loss_data = [
            (
                "Total Loss",
                calculated_total_unweighted,  # Use calculated unweighted for absolute
                1.0,
                calculated_total_weighted,  # Use calculated weighted for weighted
            ),
            ("Mus Loss", mus_losses, mus_weight, weighted_mus),
            ("LogVars Loss", logvars_losses, logvars_weight, weighted_logvars),
            ("Pis Loss", pis_losses, pis_weight, weighted_pis),
            ("Uniform Penalty", uniform_penalties, uniform_weight, weighted_uniform),
        ]

        colors = ["blue", "red", "green", "orange", "purple"]

        for row, (loss_name, absolute_loss, weight, weighted_loss) in enumerate(
            loss_data
        ):
            color = colors[row]

            # Column 1: Absolute Loss
            axes[row, 0].plot(epochs, absolute_loss, color=color, linewidth=2)
            axes[row, 0].set_title(f"{loss_name} - Absolute")
            axes[row, 0].set_xlabel("Epoch")
            axes[row, 0].set_ylabel("Loss Value")
            axes[row, 0].grid(True, alpha=0.3)

            # Column 2: Log Scale Loss
            # Handle zeros and negative values for log scale
            log_loss = [max(l, 1e-10) for l in absolute_loss]  # Avoid log(0)
            axes[row, 1].semilogy(epochs, log_loss, color=color, linewidth=2)
            axes[row, 1].set_title(f"{loss_name} - Log Scale")
            axes[row, 1].set_xlabel("Epoch")
            axes[row, 1].set_ylabel("Loss Value (log)")
            axes[row, 1].grid(True, alpha=0.3)

            # Column 3: Weighted Loss
            axes[row, 2].plot(
                epochs, weighted_loss, color=color, linewidth=2, label="Training"
            )
            axes[row, 2].set_title(f"{loss_name} - Weighted")
            axes[row, 2].set_xlabel("Epoch")
            axes[row, 2].set_ylabel("Weighted Loss")
            axes[row, 2].grid(True, alpha=0.3)

            # Add validation losses to first 3 columns if available
            val_attr_map = {
                "Total Loss": "val_total_loss",
                "Mus Loss": "val_mus_loss",
                "LogVars Loss": "val_logvars_loss",
                "Pis Loss": "val_pis_loss",
                "Uniform Penalty": "val_uniform_penalty",
            }

            if loss_name in val_attr_map:
                val_losses = [
                    getattr(m, val_attr_map[loss_name], None) for m in self.history
                ]
                if any(v is not None for v in val_losses):
                    val_epochs = [
                        epochs[i] for i, v in enumerate(val_losses) if v is not None
                    ]
                    val_values = [v for v in val_losses if v is not None]

                    # Add validation to absolute plot (Column 1)
                    axes[row, 0].plot(
                        val_epochs,
                        val_values,
                        color=color,
                        linewidth=2,
                        linestyle="--",
                        alpha=0.8,
                        label="Validation",
                    )

                    # Add validation to log plot (Column 2)
                    val_log_loss = [max(l, 1e-10) for l in val_values]
                    axes[row, 1].semilogy(
                        val_epochs,
                        val_log_loss,
                        color=color,
                        linewidth=2,
                        linestyle="--",
                        alpha=0.8,
                        label="Validation",
                    )

                    # Add validation to weighted plot (Column 3) - apply same weight
                    if loss_name == "Total Loss":
                        val_weighted = val_values  # Total is already weighted
                    else:
                        val_weighted = [v * weight for v in val_values]

                    axes[row, 2].plot(
                        val_epochs,
                        val_weighted,
                        color=color,
                        linewidth=2,
                        linestyle="--",
                        alpha=0.8,
                        label="Validation",
                    )

            # Add legends to first 3 columns if validation data exists
            if loss_name in val_attr_map and any(
                getattr(m, val_attr_map[loss_name], None) is not None
                for m in self.history
            ):
                axes[row, 0].legend(fontsize=8)
                axes[row, 1].legend(fontsize=8)
                axes[row, 2].legend(fontsize=8)

            # Column 4: Weight Value (constant line)
            weight_values = [weight] * len(epochs)
            axes[row, 3].plot(
                epochs, weight_values, color=color, linewidth=3, linestyle="--"
            )
            axes[row, 3].set_title(f"{loss_name} - Weight")
            axes[row, 3].set_xlabel("Epoch")
            axes[row, 3].set_ylabel("Weight Value")
            axes[row, 3].grid(True, alpha=0.3)
            axes[row, 3].set_ylim(
                [0, max(weight * 1.1, 0.1)]
            )  # Set reasonable y-limits

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()  # Display in notebooks
        plt.close()

    def plot_stacked_loss_contributions(
        self, save_path: Path, model_type: str = "GMVAE"
    ):
        """Generate stacked loss contribution plots showing relative importance of each loss component."""
        if not self.history:
            return

        epochs = [m.epoch for m in self.history]

        if model_type.upper() == "GMVAE":
            # GMVAE losses - handle None values
            recon_losses = [
                getattr(m, "reconstruction_loss", 0) or 0 for m in self.history
            ]
            kl_losses = [getattr(m, "kl_loss", 0) or 0 for m in self.history]
            fraction_losses = [
                getattr(m, "fraction_loss", 0) or 0 for m in self.history
            ]
            zinb_losses = [getattr(m, "zinb_loss", 0) or 0 for m in self.history]

            # Only include non-zero loss components
            loss_components = {}
            colors = []
            color_map = {
                "Reconstruction Loss": "#FF6B6B",
                "KL Loss": "#4ECDC4",
                "Fraction Loss": "#45B7D1",
                "ZINB Loss": "#96CEB4",
            }

            if any(loss > 0 for loss in recon_losses):
                loss_components["Reconstruction Loss"] = recon_losses
                colors.append(color_map["Reconstruction Loss"])
            if any(loss > 0 for loss in kl_losses):
                loss_components["KL Loss"] = kl_losses
                colors.append(color_map["KL Loss"])
            if any(loss > 0 for loss in fraction_losses):
                loss_components["Fraction Loss"] = fraction_losses
                colors.append(color_map["Fraction Loss"])
            if any(loss > 0 for loss in zinb_losses):
                loss_components["ZINB Loss"] = zinb_losses
                colors.append(color_map["ZINB Loss"])

            # Extract weighted losses for GMVAE
            weighted_recon = [
                getattr(m, "weighted_reconstruction_loss", 0) or 0 for m in self.history
            ]
            weighted_kl = [getattr(m, "weighted_kl_loss", 0) or 0 for m in self.history]
            weighted_fraction = [
                getattr(m, "weighted_fraction_loss", 0) or 0 for m in self.history
            ]
            weighted_zinb = [
                getattr(m, "weighted_zinb_loss", 0) or 0 for m in self.history
            ]

            # If weighted losses are not stored, calculate them using dynamic weights
            if all(w == 0 for w in weighted_recon):
                # Fallback calculation - use stored weights if available
                warmup_weights = [
                    getattr(m, "recon_warmup_weight", 1.0) or 1.0 for m in self.history
                ]
                config_recon_weight = 1.0  # Default, should be passed from config
                weighted_recon = [
                    r * config_recon_weight * w
                    for r, w in zip(recon_losses, warmup_weights)
                ]

            if all(w == 0 for w in weighted_kl):
                # Use dynamic KL weights
                kl_weights = [getattr(m, "kl_weight", 1.0) or 1.0 for m in self.history]
                weighted_kl = [k * w for k, w in zip(kl_losses, kl_weights)]

            if all(w == 0 for w in weighted_fraction):
                weighted_fraction = [f * 1.0 for f in fraction_losses]  # Default weight

            if all(w == 0 for w in weighted_zinb):
                weighted_zinb = [z * 1.0 for z in zinb_losses]  # Default weight

            # Create weighted loss components dictionary
            weighted_loss_components = {}
            if any(loss > 0 for loss in weighted_recon):
                weighted_loss_components["Reconstruction Loss"] = weighted_recon
            if any(loss > 0 for loss in weighted_kl):
                weighted_loss_components["KL Loss"] = weighted_kl
            if any(loss > 0 for loss in weighted_fraction):
                weighted_loss_components["Fraction Loss"] = weighted_fraction
            if any(loss > 0 for loss in weighted_zinb):
                weighted_loss_components["ZINB Loss"] = weighted_zinb

        else:  # BulkEncoder
            # BulkEncoder losses - handle None values
            mus_losses = [getattr(m, "mus_loss", 0) or 0 for m in self.history]
            logvars_losses = [getattr(m, "logvars_loss", 0) or 0 for m in self.history]
            pis_losses = [getattr(m, "pis_loss", 0) or 0 for m in self.history]
            uniform_penalties = [
                getattr(m, "uniform_penalty", 0) or 0 for m in self.history
            ]

            # Only include non-zero loss components
            loss_components = {}
            colors = []
            color_map = {
                "Mus Loss": "#FF9F43",
                "LogVars Loss": "#10AC84",
                "Pis Loss": "#EE5A24",
                "Uniform Penalty": "#5F27CD",
            }

            if any(loss > 0 for loss in mus_losses):
                loss_components["Mus Loss"] = mus_losses
                colors.append(color_map["Mus Loss"])
            if any(loss > 0 for loss in logvars_losses):
                loss_components["LogVars Loss"] = logvars_losses
                colors.append(color_map["LogVars Loss"])
            if any(loss > 0 for loss in pis_losses):
                loss_components["Pis Loss"] = pis_losses
                colors.append(color_map["Pis Loss"])
            if any(loss > 0 for loss in uniform_penalties):
                loss_components["Uniform Penalty"] = uniform_penalties
                colors.append(color_map["Uniform Penalty"])

            # Extract weighted losses for BulkEncoder
            weighted_mus = [
                getattr(m, "weighted_mus_loss", 0) or 0 for m in self.history
            ]
            weighted_logvars = [
                getattr(m, "weighted_logvars_loss", 0) or 0 for m in self.history
            ]
            weighted_pis = [
                getattr(m, "weighted_pis_loss", 0) or 0 for m in self.history
            ]
            weighted_uniform = [
                getattr(m, "weighted_uniform_penalty", 0) or 0 for m in self.history
            ]

            # If weighted losses are not stored, calculate them using config weights
            if all(w == 0 for w in weighted_mus):
                weighted_mus = [m * 1.0 for m in mus_losses]  # Default weight
            if all(w == 0 for w in weighted_logvars):
                weighted_logvars = [l * 1.0 for l in logvars_losses]  # Default weight
            if all(w == 0 for w in weighted_pis):
                weighted_pis = [p * 1.0 for p in pis_losses]  # Default weight
            if all(w == 0 for w in weighted_uniform):
                weighted_uniform = [
                    u * 1.0 for u in uniform_penalties
                ]  # Default weight

            # Create weighted loss components dictionary for BulkEncoder
            weighted_loss_components = {}
            if any(loss > 0 for loss in weighted_mus):
                weighted_loss_components["Mus Loss"] = weighted_mus
            if any(loss > 0 for loss in weighted_logvars):
                weighted_loss_components["LogVars Loss"] = weighted_logvars
            if any(loss > 0 for loss in weighted_pis):
                weighted_loss_components["Pis Loss"] = weighted_pis
            if any(loss > 0 for loss in weighted_uniform):
                weighted_loss_components["Uniform Penalty"] = weighted_uniform

        # Check if we have any loss components to plot
        if not loss_components:
            print(f"Warning: No non-zero loss components found for {model_type}")
            return

        # Create figure with subplots - now 2x3 to include weighted versions
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(
            f"{model_type} Loss Component Analysis - Raw vs Weighted",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Raw Absolute values stacked area plot
        ax1 = axes[0, 0]
        loss_arrays = [loss_components[name] for name in loss_components.keys()]
        ax1.stackplot(
            epochs,
            *loss_arrays,
            labels=list(loss_components.keys()),
            colors=colors,
            alpha=0.8,
        )
        ax1.set_title("Raw Loss Values (Stacked)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss Value")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Raw Percentage contribution stacked area plot
        ax2 = axes[0, 1]
        # Calculate total loss for each epoch
        total_losses = []
        for i in range(len(epochs)):
            total = sum(loss_arrays[j][i] for j in range(len(loss_arrays)))
            total_losses.append(total if total > 0 else 1e-8)  # Avoid division by zero

        # Calculate percentages
        percentage_arrays = []
        for loss_array in loss_arrays:
            percentages = [
                (loss_array[i] / total_losses[i]) * 100 for i in range(len(epochs))
            ]
            percentage_arrays.append(percentages)

        ax2.stackplot(
            epochs,
            *percentage_arrays,
            labels=list(loss_components.keys()),
            colors=colors,
            alpha=0.8,
        )
        ax2.set_title("Raw Loss Contribution % (Stacked)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Weighted Absolute values stacked area plot
        ax3 = axes[0, 2]
        if weighted_loss_components:
            weighted_arrays = [
                weighted_loss_components[name]
                for name in weighted_loss_components.keys()
            ]
            weighted_colors = [
                color_map[name] for name in weighted_loss_components.keys()
            ]
            ax3.stackplot(
                epochs,
                *weighted_arrays,
                labels=list(weighted_loss_components.keys()),
                colors=weighted_colors,
                alpha=0.8,
            )
            ax3.set_title("Weighted Loss Values (Stacked)")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Weighted Loss Value")
            ax3.legend(loc="upper right", fontsize=8)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(
                0.5,
                0.5,
                "Weighted losses\nnot available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("Weighted Loss Values (Stacked)")

        # 4. Weighted Percentage contribution stacked area plot
        ax4 = axes[1, 0]
        if weighted_loss_components:
            # Calculate total weighted loss for each epoch
            total_weighted_losses = []
            for i in range(len(epochs)):
                total = sum(weighted_arrays[j][i] for j in range(len(weighted_arrays)))
                total_weighted_losses.append(total if total > 0 else 1e-8)

            # Calculate weighted percentages
            weighted_percentage_arrays = []
            for weighted_array in weighted_arrays:
                percentages = [
                    (weighted_array[i] / total_weighted_losses[i]) * 100
                    for i in range(len(epochs))
                ]
                weighted_percentage_arrays.append(percentages)

            ax4.stackplot(
                epochs,
                *weighted_percentage_arrays,
                labels=list(weighted_loss_components.keys()),
                colors=weighted_colors,
                alpha=0.8,
            )
            ax4.set_title("Weighted Loss Contribution % (Stacked)")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Percentage (%)")
            ax4.set_ylim(0, 100)
            ax4.legend(loc="upper right", fontsize=8)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "Weighted percentages\nnot available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Weighted Loss Contribution % (Stacked)")

        # 5. Log-scale individual losses
        ax5 = axes[1, 1]
        for i, (name, losses) in enumerate(loss_components.items()):
            # Add small epsilon to avoid log(0)
            safe_losses = [max(loss, 1e-8) for loss in losses]
            ax5.plot(
                epochs,
                safe_losses,
                label=name,
                color=colors[i],
                linewidth=2,
                marker="o",
                markersize=3,
            )
        ax5.set_title("Individual Loss Components (Log Scale)")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Loss Value (Log Scale)")
        ax5.set_yscale("log")
        ax5.legend(loc="upper right", fontsize=8)
        ax5.grid(True, alpha=0.3)

        # 6. Normalized losses (relative to first epoch)
        ax6 = axes[1, 2]
        for i, (name, losses) in enumerate(loss_components.items()):
            if not losses or len(losses) == 0:
                continue

            # Normalize relative to the first epoch value
            first_epoch_loss = losses[0] if losses[0] > 0 else 1e-8
            normalized_losses = [loss / first_epoch_loss for loss in losses]

            ax6.plot(
                epochs,
                normalized_losses,
                label=name,
                color=colors[i],
                linewidth=2,
                marker="o",
                markersize=3,
            )
        ax6.set_title("Loss Components (Relative to First Epoch)")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Relative Loss Value")
        ax6.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="First Epoch")
        ax6.legend(loc="upper right", fontsize=8)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()  # Show plot in notebooks
        plt.close()

    def save_metrics(self, path: Path):
        """Save metrics history to file."""
        metrics_data = [m.to_dict() for m in self.history]
        with open(path, "wb") as f:
            pickle.dump(metrics_data, f)

    def load_metrics(self, path: Path):
        """Load metrics history from file."""
        with open(path, "rb") as f:
            metrics_data = pickle.load(f)
        self.history = [TrainingMetrics.from_dict(data) for data in metrics_data]
