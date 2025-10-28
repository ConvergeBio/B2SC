# B2SC Codebase Reorganization: Architectural Plan

## Overview

This document outlines a comprehensive plan to reorganize the B2SC (Bulk-to-Single-Cell) codebase into a more efficient, object-oriented architecture. The current implementation has scattered functionality across multiple files with procedural code. The new architecture will provide a clean, modular, and maintainable structure.

## Current State Analysis

### Existing Components
1. **Models** (`models.py`):
   - `GaussianMixtureVAE`: Encoder-decoder architecture with mixture components
   - `bulkEncoder`: Encodes bulk RNA-seq to latent space parameters

2. **Training Scripts**:
   - `train_GMVAE.py`: Training logic for GMVAE
   - `train_bulkEncoder.py`: Training logic for bulk encoder
   - `main.py`: Orchestration functions for training

3. **Generation** (`generate.py`):
   - Cell generation from bulk samples
   - Result processing and saving

4. **Utilities** (`utils.py`):
   - Data loading and preprocessing
   - Configuration management
   - Helper functions

5. **Execution** (`run_custom.py`):
   - Main execution script with hardcoded parameters

### Current Issues
- Scattered functionality across multiple files
- Procedural approach with global state management
- Hardcoded parameters and paths
- Difficult to extend or modify
- No clear separation of concerns
- Inconsistent error handling and logging
- Manual checkpoint management

## Proposed Architecture

### Core Design Principles
1. **Single Responsibility**: Each class has one clear purpose
2. **Encapsulation**: Internal state and methods are properly encapsulated
3. **Composition over Inheritance**: Use composition to build complex functionality
4. **Configuration-Driven**: All parameters configurable through config objects
5. **Extensibility**: Easy to add new models or training strategies
6. **Testability**: Clear interfaces for unit testing

### Class Hierarchy

```
B2SCPipeline
├── DataManager
├── GMVAETrainer
│   ├── GaussianMixtureVAE (model)
│   └── TrainingMetrics
├── BulkEncoderTrainer  
│   ├── BulkEncoder (model)
│   └── TrainingMetrics
└── GenerationResults (embedded)
```

## Detailed Component Design

### 1. Configuration Management

**File**: `config.py`

```python
@dataclass
class B2SCConfig:
    # Data configuration
    sc_data_path: str
    bulk_data_path: str = None
    barcode_path: str = None
    
    # Model architecture
    input_dim: int = None  # Auto-detected from data
    hidden_dim: int = 2048
    latent_dim: int = 1024
    
    # Training configuration
    gmvae_epochs: int = 30
    bulk_encoder_epochs: int = 100
    learning_rate: float = 5e-4
    batch_size: int = None  # Auto-calculated
    
    # Generation configuration
    num_cells_to_generate: int = 500
    
    # I/O configuration
    base_dir: Path = Path("./results")
    device: str = "auto"  # auto, cuda, cpu
    
    # Advanced options
    normalize_data: bool = True
    generate_pseudo_cells: bool = False
    test_samples: List[str] = field(default_factory=list)
```

### 2. Data Management

**File**: `data_manager.py`

```python
class DataManager:
    """Handles all data loading, preprocessing, and validation."""
    
    def __init__(self, config: B2SCConfig):
        self.config = config
        self.sc_data = None
        self.bulk_data = None
        self.gene_list = None
        self.cell_type_mapping = None
        
    def load_single_cell_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and preprocess single-cell data for GMVAE and BulkEncoder training."""
        
    def load_bulk_data(self) -> Tuple[DataLoader, List[str]]:
        """Load bulk RNA-seq data for generation."""
        
    def get_data_statistics(self) -> Dict[str, Any]:
        """Return comprehensive data statistics."""
        
    def validate_data_compatibility(self) -> bool:
        """Ensure SC and bulk data are compatible."""
```

### 3. Training Metrics

**File**: `metrics.py`

```python
@dataclass
class TrainingMetrics:
    """Container for training metrics and history."""
    epoch: int
    total_loss: float
    reconstruction_loss: float
    kl_loss: float
    fraction_loss: float = None
    zinb_loss: float = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging/saving."""
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'TrainingMetrics':
        """Create from dictionary."""

class MetricsTracker:
    """Tracks and manages training metrics over time."""
    
    def __init__(self):
        self.history: List[TrainingMetrics] = []
        
    def add_epoch_metrics(self, metrics: TrainingMetrics):
        """Add metrics for an epoch."""
        
    def get_best_epoch(self, metric: str = 'total_loss') -> int:
        """Get epoch with best performance."""
        
    def plot_training_curves(self, save_path: Path):
        """Generate training curve plots."""
        
    def save_metrics(self, path: Path):
        """Save metrics history to file."""
```

### 4. GMVAE Trainer

**File**: `gmvae_trainer.py`

```python
class GMVAETrainer:
    """Handles GMVAE model training and management."""
    
    def __init__(self, config: B2SCConfig, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self.model = None
        self.optimizer = None
        self.metrics_tracker = MetricsTracker()
        self.device = self._setup_device()
        
    def initialize_model(self) -> GaussianMixtureVAE:
        """Initialize GMVAE model with proper architecture."""
        
    def train(self, dataloader: DataLoader, 
              proportion_tensor: torch.Tensor) -> MetricsTracker:
        """
        Train the GMVAE model.
        
        Returns:
            MetricsTracker: Complete training history and metrics
        """
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint and training state."""
        
    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load model from checkpoint."""
        
    def get_latent_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract trained mu, logvar, and pi parameters."""
        
    def validate_model(self, validation_loader: DataLoader) -> TrainingMetrics:
        """Run validation and return metrics."""
        
    def generate_visualizations(self, save_dir: Path):
        """Generate UMAP plots and other visualizations."""
```

### 5. Bulk Encoder Trainer

**File**: `bulk_encoder_trainer.py`

```python
class BulkEncoderTrainer:
    """Handles BulkEncoder model training and management."""
    
    def __init__(self, config: B2SCConfig, 
                 gmvae_trainer: GMVAETrainer,
                 data_manager: DataManager):
        self.config = config
        self.gmvae_trainer = gmvae_trainer
        self.data_manager = data_manager
        self.model = None
        self.optimizer = None
        self.metrics_tracker = MetricsTracker()
        
    def initialize_model(self) -> BulkEncoder:
        """Initialize BulkEncoder model."""
        
    def train(self, dataloader: DataLoader) -> MetricsTracker:
        """
        Train the BulkEncoder model using GMVAE parameters as targets.
        
        Returns:
            MetricsTracker: Complete training history and metrics
        """
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint and training state."""
        
    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load model from checkpoint."""
```

### 6. Generation Results

**File**: `results.py`

```python
@dataclass
class GenerationResults:
    """Container for generation results."""
    generated_cells: Dict[str, torch.Tensor]  # sample_id -> generated_cells
    cell_types: Dict[str, torch.Tensor]       # sample_id -> cell_type_labels
    generation_stats: Dict[str, Any]
    
    def save_results(self, save_dir: Path):
        """Save all results to directory."""
        
    @classmethod
    def load_results(cls, save_dir: Path) -> 'GenerationResults':
        """Load results from directory."""
```

### 7. Main Pipeline

**File**: `pipeline.py`

```python
class B2SCPipeline:
    """Main pipeline orchestrating the entire B2SC workflow including generation."""
    
    def __init__(self, config: B2SCConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.gmvae_trainer = GMVAETrainer(config, self.data_manager)
        self.bulk_encoder_trainer = BulkEncoderTrainer(config, self.gmvae_trainer, self.data_manager)
        self.logger = self._setup_logging()
        
    def train_gmvae(self) -> MetricsTracker:
        """Train the GMVAE model."""
        
    def train_bulk_encoder(self) -> MetricsTracker:
        """Train the BulkEncoder model."""
        
    def generate_cells(self, bulk_data_path: str = None, 
                      sample_ids: List[str] = None,
                      num_cells: int = None) -> GenerationResults:
        """
        Generate single cells from bulk samples.
        
        Args:
            bulk_data_path: Path to bulk data (optional, uses config if None)
            sample_ids: List of sample identifiers to generate from
            num_cells: Number of cells to generate per sample (uses config if None)
            
        Returns:
            GenerationResults: Complete generation results
        """
        
    def validate_generation_quality(self, results: GenerationResults) -> Dict[str, float]:
        """Validate quality of generated cells."""
        
    def run_full_pipeline(self, bulk_data_path: str = None) -> Tuple[MetricsTracker, MetricsTracker, GenerationResults]:
        """Run the complete B2SC pipeline: train GMVAE, train BulkEncoder, generate cells."""
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. Create configuration system (`config.py`)
2. Implement metrics tracking (`metrics.py`)
3. Set up logging and error handling utilities
4. Create base directory structure

### Phase 2: Data Management (Week 1-2)
1. Implement `DataManager` class
2. Migrate and refactor data loading functions from `utils.py`
3. Add comprehensive data validation
4. Create unit tests for data loading

### Phase 3: Model Training Classes (Week 2-3)
1. Implement `GMVAETrainer` class
2. Implement `BulkEncoderTrainer` class
3. Migrate training logic from existing files
4. Add checkpoint management and resuming
5. Implement training visualizations

### Phase 4: Generation System (Week 3)
1. Integrate generation methods into `B2SCPipeline` class
2. Migrate generation logic from `generate.py`
3. Add result validation and quality metrics
4. Implement result saving/loading

### Phase 5: Main Pipeline (Week 4)
1. Complete `B2SCPipeline` class with integrated generation
2. Create new main execution script
3. Add comprehensive error handling
4. Create configuration templates

### Phase 6: Testing and Documentation (Week 4-5)
1. Create comprehensive unit tests
2. Add integration tests
3. Write detailed documentation
4. Create example notebooks
5. Performance benchmarking

## Migration Strategy

### Backward Compatibility
- Keep existing files during transition
- Create adapter functions for old interfaces
- Gradual migration of functionality

### File Mapping
```
Current → New
├── models.py → models/ (split into separate model files)
├── train_GMVAE.py → gmvae_trainer.py (refactored)
├── train_bulkEncoder.py → bulk_encoder_trainer.py (refactored)
├── generate.py → pipeline.py (integrated into B2SCPipeline)
├── utils.py → data_manager.py + utilities/ (split)
├── main.py → pipeline.py (refactored)
└── run_custom.py → examples/run_pipeline.py (simplified)
```

## Benefits of New Architecture

### For Developers
- **Modularity**: Easy to modify individual components
- **Testability**: Clear interfaces for unit testing
- **Extensibility**: Simple to add new models or features
- **Maintainability**: Cleaner code organization

### For Users
- **Simplicity**: Single pipeline class for complete workflow
- **Flexibility**: Easy configuration without code changes
- **Reproducibility**: Complete state saving/loading
- **Robustness**: Better error handling and validation

### For Research
- **Experimentation**: Easy to swap components or try variations
- **Metrics**: Comprehensive tracking of training progress
- **Visualization**: Built-in plotting and analysis tools
- **Comparison**: Easy to compare different configurations

## Example Usage

### Simple Pipeline Execution
```python
from b2sc import B2SCConfig, B2SCPipeline

# Configure pipeline
config = B2SCConfig(
    sc_data_path="data/single_cell.h5ad",
    bulk_data_path="data/bulk_samples.h5ad",
    base_dir=Path("./results"),
    gmvae_epochs=50,
    bulk_encoder_epochs=100
)

# Run complete pipeline
pipeline = B2SCPipeline(config)
gmvae_metrics, be_metrics, results = pipeline.run_full_pipeline()

# Access results
print(f"Generated {len(results.generated_cells)} samples")
results.save_results(config.base_dir / "generation_results")
```

### Advanced Usage
```python
# Custom training with validation
pipeline = B2SCPipeline(config)

# Train GMVAE with custom monitoring
gmvae_metrics = pipeline.train_gmvae()
gmvae_metrics.plot_training_curves(config.base_dir / "gmvae_curves.png")

# Train BulkEncoder
be_metrics = pipeline.train_bulk_encoder()

# Generate with custom parameters
results = pipeline.generate_cells(num_cells=1000)
quality_metrics = pipeline.validate_generation_quality(results)
```

## Conclusion

This architectural plan provides a comprehensive roadmap for transforming the B2SC codebase into a modern, maintainable, and extensible system. The new object-oriented design will significantly improve code quality, user experience, and research productivity while maintaining all existing functionality.

The phased implementation approach ensures minimal disruption during the transition, and the backward compatibility strategy allows for gradual migration of existing workflows.
