"""
Training metrics tracking and management for B2SC pipeline.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import pickle


@dataclass
class TrainingMetrics:
    """Container for training metrics and history."""

    epoch: int
    total_loss: float
    reconstruction_loss: float
    kl_loss: float
    fraction_loss: Optional[float] = None
    zinb_loss: Optional[float] = None
    kl_weight: Optional[float] = None

    # BulkEncoder specific losses
    mus_loss: Optional[float] = None
    logvars_loss: Optional[float] = None
    pis_loss: Optional[float] = None
    uniform_penalty: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging/saving."""
        return {
            "epoch": self.epoch,
            "total_loss": self.total_loss,
            "reconstruction_loss": self.reconstruction_loss,
            "kl_loss": self.kl_loss,
            "fraction_loss": self.fraction_loss,
            "zinb_loss": self.zinb_loss,
            "kl_weight": self.kl_weight,
            "mus_loss": self.mus_loss,
            "logvars_loss": self.logvars_loss,
            "pis_loss": self.pis_loss,
            "uniform_penalty": self.uniform_penalty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "TrainingMetrics":
        """Create from dictionary."""
        return cls(**data)


class MetricsTracker:
    """Tracks and manages training metrics over time."""

    def __init__(self):
        self.history: List[TrainingMetrics] = []

    def add_epoch_metrics(self, metrics: TrainingMetrics):
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

    def plot_training_curves(self, save_path: Path):
        """Generate training curve plots."""
        if not self.history:
            return

        epochs = [m.epoch for m in self.history]
        total_losses = [m.total_loss for m in self.history]
        recon_losses = [m.reconstruction_loss for m in self.history]
        kl_losses = [m.kl_loss for m in self.history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Training Metrics")

        # Total loss
        axes[0, 0].plot(epochs, total_losses)
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)

        # Reconstruction loss
        axes[0, 1].plot(epochs, recon_losses)
        axes[0, 1].set_title("Reconstruction Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True)

        # KL loss
        axes[1, 0].plot(epochs, kl_losses)
        axes[1, 0].set_title("KL Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True)

        # Fraction loss (if available)
        fraction_losses = [
            m.fraction_loss for m in self.history if m.fraction_loss is not None
        ]
        if fraction_losses:
            fraction_epochs = [
                m.epoch for m in self.history if m.fraction_loss is not None
            ]
            axes[1, 1].plot(fraction_epochs, fraction_losses)
            axes[1, 1].set_title("Fraction Loss")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No Fraction Loss Data",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()  # Show plot in notebooks
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

        # Check if we have any loss components to plot
        if not loss_components:
            print(f"Warning: No non-zero loss components found for {model_type}")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"{model_type} Loss Component Analysis", fontsize=16, fontweight="bold"
        )

        # 1. Absolute values stacked area plot
        ax1 = axes[0, 0]
        loss_arrays = [loss_components[name] for name in loss_components.keys()]
        ax1.stackplot(
            epochs,
            *loss_arrays,
            labels=list(loss_components.keys()),
            colors=colors,
            alpha=0.8,
        )
        ax1.set_title("Absolute Loss Values (Stacked)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss Value")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Percentage contribution stacked area plot
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
        ax2.set_title("Loss Contribution Percentages (Stacked)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Log-scale individual losses
        ax3 = axes[1, 0]
        for i, (name, losses) in enumerate(loss_components.items()):
            # Add small epsilon to avoid log(0)
            safe_losses = [max(loss, 1e-8) for loss in losses]
            ax3.plot(
                epochs,
                safe_losses,
                label=name,
                color=colors[i],
                linewidth=2,
                marker="o",
                markersize=3,
            )
        ax3.set_title("Individual Loss Components (Log Scale)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss Value (Log Scale)")
        ax3.set_yscale("log")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Normalized losses (0-1 scale for each component)
        ax4 = axes[1, 1]
        for i, (name, losses) in enumerate(loss_components.items()):
            # Normalize each loss component to 0-1 range
            min_loss = min(losses) if losses else 0
            max_loss = max(losses) if losses else 1
            range_loss = max_loss - min_loss if max_loss != min_loss else 1
            normalized_losses = [(loss - min_loss) / range_loss for loss in losses]
            ax4.plot(
                epochs,
                normalized_losses,
                label=name,
                color=colors[i],
                linewidth=2,
                marker="o",
                markersize=3,
            )
        ax4.set_title("Normalized Loss Components (0-1 Scale)")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Normalized Loss Value")
        ax4.set_ylim(0, 1)
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, alpha=0.3)

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
