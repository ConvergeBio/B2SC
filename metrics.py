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
