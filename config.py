"""
Configuration management for B2SC pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class B2SCConfig:
    """Configuration class for B2SC pipeline."""

    # Data configuration
    sc_training_path: str  # Path to single-cell training data (.h5ad format)
    sc_validation_path: Optional[str] = (
        None  # Path to single-cell validation data (.h5ad format)
    )
    bulk_data_for_deconv_path: Optional[str] = None
    barcode_path: Optional[str] = None

    # Pseudobulk computation settings
    pseudobulk_groupby: str = (
        "donor_id"  # Column to group by for pseudobulk computation
    )
    cache_pseudobulk: bool = True  # Cache computed pseudobulk data

    # Shared model architecture
    input_dim: Optional[int] = None  # Auto-detected from data
    latent_dim: int = 1024  # Shared latent space dimension for both models

    # GMVAE hyperparameters
    gmvae_hyperparameters: Dict[str, Any] = field(
        default_factory=lambda: {
            # Architecture
            "hidden_dim": 2048,
            "use_weight_norm": False,  # Apply weight normalization to hidden layers
            # Training
            "epochs": 30,
            "learning_rate": 5e-4,
            "batch_size": 32,  # Number of donors to sample per batch (cells are flattened)
            # Loss weights (reduced for numerical stability)
            "recon_loss_weight": 1.0,
            "zinb_loss_weight": 0.5,
            "fraction_loss_weight": 2.0,
            "kl_loss_weight": 0.0,  # Set to 0 by default, skip calculation when 0
            # KL divergence configuration
            "kl_scale_divisor": 1,  # Scale factor for KL loss (adjusted for mean reduction)
            "kl_clamp_max": 1000.0,  # Maximum KL value (prevent extreme positives)
            # KL annealing (gradual increase to prevent posterior collapse)
            "kl_anneal_start": 0.0,  # Starting KL weight (typically 0)
            "kl_anneal_max": 1.0,  # Maximum KL weight to reach
            "kl_anneal_epochs": None,  # Number of epochs to reach max KL weight (None = use total epochs)
            # Reconstruction loss warm-up (sigmoid)
            "recon_warmup_steepness": 2.0,  # Controls how steep the S-curve is
            "recon_warmup_midpoint": 3.5,  # Epoch where warmup reaches ~50%
        }
    )

    # BulkEncoder hyperparameters
    bulk_encoder_hyperparameters: Dict[str, Any] = field(
        default_factory=lambda: {
            # Architecture
            "hidden_dim": 2048,
            # Training
            "epochs": 100,
            "learning_rate": 1e-3,
            "batch_size": 3,
            # Loss weights
            "mus_loss_weight": 1.0,
            "logvars_loss_weight": 1.0,
            "pis_loss_weight": 50.0,
            "uniform_penalty_weight": 50.0,  # Restored uniform penalty
            # Target generation strategy
            "use_dynamic_targets": True,  # Use sample-specific GMVAE targets (recommended)
        }
    )

    # I/O configuration
    base_dir: Path = Path("./results")
    device: str = "auto"  # auto, cuda, cpu, mps

    # Advanced options
    normalize_data: bool = True

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure base_dir is a Path object
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)

        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Validate required paths (only if they are expected to exist)
        # Note: During tutorial setup, files may not exist yet
        if (
            self.sc_training_path
            and not str(self.sc_training_path).endswith(
                ("train_data.h5ad", "test_data.h5ad", "val_data.h5ad")
            )
            and not Path(self.sc_training_path).exists()
        ):
            raise FileNotFoundError(
                f"Single-cell training data path not found: {self.sc_training_path}"
            )

    def set_hyperparameters(self, updates: Dict[str, Dict[str, Any]]):
        """Set hyperparameters using a standardized hierarchical format.

        Args:
            updates: Dictionary with model names as keys and parameter dictionaries as values

        Example:
            config.set_hyperparameters({
                "gmvae": {
                    "learning_rate": 1e-3,
                    "batch_size": 64,
                    "fraction_loss_weight": 100.0
                },
                "bulk_encoder": {
                    "learning_rate": 1e-4,
                    "epochs": 50
                }
            })
        """
        for model_name, params in updates.items():
            if model_name == "gmvae":
                for param_name, value in params.items():
                    if param_name in self.gmvae_hyperparameters:
                        self.gmvae_hyperparameters[param_name] = value
                    else:
                        raise KeyError(
                            f"GMVAE hyperparameter '{param_name}' not found. "
                            f"Available parameters: {list(self.gmvae_hyperparameters.keys())}"
                        )
            elif model_name == "bulk_encoder":
                for param_name, value in params.items():
                    if param_name in self.bulk_encoder_hyperparameters:
                        self.bulk_encoder_hyperparameters[param_name] = value
                    else:
                        raise KeyError(
                            f"BulkEncoder hyperparameter '{param_name}' not found. "
                            f"Available parameters: {list(self.bulk_encoder_hyperparameters.keys())}"
                        )
            else:
                raise ValueError(
                    f"Invalid model name '{model_name}'. Use 'gmvae' or 'bulk_encoder'."
                )

    def get_available_parameters(self) -> Dict[str, List[str]]:
        """Get list of available parameters for each model.

        Returns:
            Dictionary with model names as keys and parameter lists as values
        """
        return {
            "gmvae": list(self.gmvae_hyperparameters.keys()),
            "bulk_encoder": list(self.bulk_encoder_hyperparameters.keys()),
        }

    # Backward compatibility properties
    @property
    def hidden_dim(self) -> int:
        """Backward compatibility: returns GMVAE hidden_dim."""
        return self.gmvae_hyperparameters["hidden_dim"]

    @property
    def batch_size(self) -> int:
        """Backward compatibility: returns GMVAE batch_size."""
        return self.gmvae_hyperparameters["batch_size"]

    # GMVAE backward compatibility properties
    @property
    def gmvae_hidden_dim(self) -> int:
        return self.gmvae_hyperparameters["hidden_dim"]

    @property
    def gmvae_epochs(self) -> int:
        return self.gmvae_hyperparameters["epochs"]

    @property
    def gmvae_lr(self) -> float:
        return self.gmvae_hyperparameters["learning_rate"]

    @property
    def gmvae_batch_size(self) -> int:
        return self.gmvae_hyperparameters["batch_size"]

    @property
    def gmvae_recon_loss_weight(self) -> float:
        return self.gmvae_hyperparameters["recon_loss_weight"]

    @property
    def gmvae_zinb_loss_weight(self) -> float:
        return self.gmvae_hyperparameters["zinb_loss_weight"]

    @property
    def gmvae_fraction_loss_weight(self) -> float:
        return self.gmvae_hyperparameters["fraction_loss_weight"]

    @property
    def gmvae_kl_loss_weight(self) -> float:
        return self.gmvae_hyperparameters["kl_loss_weight"]

    # BulkEncoder backward compatibility properties
    @property
    def bulk_hidden_dim(self) -> int:
        return self.bulk_encoder_hyperparameters["hidden_dim"]

    @property
    def bulk_encoder_epochs(self) -> int:
        return self.bulk_encoder_hyperparameters["epochs"]

    @property
    def bulk_encoder_lr(self) -> float:
        return self.bulk_encoder_hyperparameters["learning_rate"]

    @property
    def bulk_encoder_batch_size(self) -> int:
        return self.bulk_encoder_hyperparameters["batch_size"]

    @property
    def bulk_mus_loss_weight(self) -> float:
        return self.bulk_encoder_hyperparameters["mus_loss_weight"]

    @property
    def bulk_logvars_loss_weight(self) -> float:
        return self.bulk_encoder_hyperparameters["logvars_loss_weight"]

    @property
    def bulk_pis_loss_weight(self) -> float:
        return self.bulk_encoder_hyperparameters["pis_loss_weight"]

    @property
    def bulk_uniform_penalty_weight(self) -> float:
        return self.bulk_encoder_hyperparameters["uniform_penalty_weight"]

    @property
    def use_dynamic_targets(self) -> bool:
        return self.bulk_encoder_hyperparameters["use_dynamic_targets"]

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        result = {}

        # Add all fields
        for field_name, field_info in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if isinstance(value, Path):
                value = str(value)
            result[field_name] = value

        return result

    @classmethod
    def from_dict(cls, config_dict: dict) -> "B2SCConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
