"""
Configuration management for B2SC pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class B2SCConfig:
    """Configuration class for B2SC pipeline."""

    # Data configuration
    sc_data_path: str
    bulk_data_path: Optional[str] = None
    barcode_path: Optional[str] = None

    # Model architecture
    input_dim: Optional[int] = None  # Auto-detected from data
    hidden_dim: int = 2048
    latent_dim: int = 1024

    # Training configuration
    gmvae_epochs: int = 30
    bulk_encoder_epochs: int = 100
    learning_rate: float = 5e-4
    batch_size: Optional[int] = None  # Auto-calculated

    # Generation configuration
    num_cells_to_generate: int = 500

    # I/O configuration
    base_dir: Path = Path("./results")
    device: str = "auto"  # auto, cuda, cpu, mps

    # Advanced options
    normalize_data: bool = True
    generate_pseudo_cells: bool = False
    test_samples: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure base_dir is a Path object
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)

        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Validate required paths
        if not Path(self.sc_data_path).exists():
            raise FileNotFoundError(
                f"Single-cell data path not found: {self.sc_data_path}"
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "B2SCConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
