"""
B2SC: Bulk-to-Single-Cell generation pipeline.

A modern, object-oriented framework for generating single-cell RNA-seq data
from bulk RNA-seq samples using deep learning models.
"""

from .config import B2SCConfig
from .pipeline import B2SCPipeline
from .data_manager import DataManager
from .results import GenerationResults
from .metrics import TrainingMetrics, MetricsTracker

# Import models for advanced usage
from .models import GaussianMixtureVAE, BulkEncoder

# Import trainers for advanced usage
from .gmvae_trainer import GMVAETrainer
from .bulk_encoder_trainer import BulkEncoderTrainer

__version__ = "2.0.0"
__author__ = "B2SC Development Team"

__all__ = [
    # Main classes for typical usage
    "B2SCConfig",
    "B2SCPipeline",
    # Data and results
    "DataManager",
    "GenerationResults",
    # Metrics
    "TrainingMetrics",
    "MetricsTracker",
    # Models (for advanced usage)
    "GaussianMixtureVAE",
    "BulkEncoder",
    # Trainers (for advanced usage)
    "GMVAETrainer",
    "BulkEncoderTrainer",
]
