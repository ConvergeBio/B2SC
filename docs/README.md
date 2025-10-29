# B2SC: Bulk-to-Single-Cell Generation Pipeline

## Overview

B2SC (Bulk-to-Single-Cell) is a deep learning framework that generates realistic single-cell RNA-seq data from bulk RNA-seq samples. The pipeline uses two main neural network models:

1. **GMVAE (Gaussian Mixture Variational Autoencoder)**: Learns the distribution of single-cell gene expression
2. **BulkEncoder**: Predicts single-cell distribution parameters from bulk RNA-seq data

Together, these models enable the generation of synthetic single-cell data that captures the cellular heterogeneity present in bulk samples.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Pipeline Usage](#pipeline-usage)
- [Individual Model Usage](#individual-model-usage)
- [Configuration Parameters](#configuration-parameters)
- [Data Requirements](#data-requirements)
- [Advanced Usage](#advanced-usage)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Simple Pipeline Execution

```python
from B2SC import B2SCConfig, B2SCPipeline

# Configure the pipeline
config = B2SCConfig(
    sc_training_data_path="data/single_cell.h5ad",
    bulk_data_for_deconv_path="data/bulk_samples.h5ad",
    base_dir="./results"
)

# Run the complete pipeline
pipeline = B2SCPipeline(config)
gmvae_metrics, be_metrics, results = pipeline.run_full_pipeline()

# Access results
print(f"Generated cells for {len(results.generated_cells)} samples")
results.save_results(config.base_dir / "generation_results")
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Required Packages

```bash
pip install torch torchvision scanpy pandas numpy matplotlib umap-learn scikit-learn termcolor anndata
```

### Setup

1. Clone or download the B2SC repository
2. Navigate to the B2SC directory
3. Install dependencies
4. Import and use the pipeline

## Pipeline Usage

### Complete Workflow

The B2SC pipeline consists of three main steps:

1. **Train GMVAE**: Learn single-cell expression patterns
2. **Train BulkEncoder**: Learn to predict single-cell parameters from bulk data
3. **Generate Cells**: Create synthetic single-cell data from bulk samples

```python
from B2SC import B2SCConfig, B2SCPipeline

# Step 1: Configure
config = B2SCConfig(
    sc_training_data_path="path/to/single_cell_data.h5ad",
    bulk_data_for_deconv_path="path/to/bulk_data.h5ad",
    base_dir="./my_results",
    gmvae_epochs=50,
    bulk_encoder_epochs=100,
    num_cells_to_generate=1000
)

# Step 2: Initialize pipeline
pipeline = B2SCPipeline(config)

# Step 3: Run complete workflow
gmvae_metrics, be_metrics, results = pipeline.run_full_pipeline()
```

### Step-by-Step Execution

For more control over the training process:

```python
# Initialize pipeline
pipeline = B2SCPipeline(config)

# Step 1: Train GMVAE
print("Training GMVAE...")
gmvae_metrics = pipeline.train_gmvae()

# Analyze GMVAE training
best_epoch = gmvae_metrics.get_best_epoch()
print(f"Best GMVAE epoch: {best_epoch}")
gmvae_metrics.plot_training_curves(config.base_dir / "gmvae_curves.png")

# Step 2: Train BulkEncoder
print("Training BulkEncoder...")
be_metrics = pipeline.train_bulk_encoder()

# Step 3: Generate cells
print("Generating cells...")
results = pipeline.generate_cells(
    sample_ids=["sample1", "sample2"],  # Specific samples
    num_cells=500  # Custom number of cells
)

# Step 4: Validate results
quality_metrics = pipeline.validate_generation_quality(results)
print("Quality metrics:", quality_metrics)
```

## Individual Model Usage

### Using GMVAE Independently

```python
from B2SC import B2SCConfig, GMVAETrainer, DataManager

# Setup
config = B2SCConfig(sc_training_data_path="data/single_cell.h5ad")
data_manager = DataManager(config)
gmvae_trainer = GMVAETrainer(config, data_manager)

# Load data
gmvae_loader, _ = data_manager.load_single_cell_data()

# Initialize and train model
gmvae_trainer.initialize_model(
    input_dim=config.input_dim,
    hidden_dim=config.hidden_dim,
    latent_dim=config.latent_dim,
    n_components=len(data_manager.cell_type_fractions)
)

# Train
metrics = gmvae_trainer.train(
    dataloader=gmvae_loader,
    proportion_tensor=data_manager.cell_type_fractions,
    mapping_dict=data_manager.cell_type_mapping,
    color_map=data_manager.color_map
)
```

### Using BulkEncoder Independently

```python
from B2SC import B2SCConfig, BulkEncoderTrainer, GMVAETrainer, DataManager

# Setup (requires trained GMVAE)
config = B2SCConfig(sc_training_data_path="data/single_cell.h5ad")
data_manager = DataManager(config)
gmvae_trainer = GMVAETrainer(config, data_manager)  # Must be trained first
be_trainer = BulkEncoderTrainer(config, gmvae_trainer, data_manager)

# Load data
_, be_loader = data_manager.load_single_cell_data()

# Initialize and train
be_trainer.initialize_model(
    input_dim=config.input_dim,
    hidden_dim=config.hidden_dim,
    latent_dim=config.latent_dim,
    n_components=len(data_manager.cell_type_fractions)
)

metrics = be_trainer.train(dataloader=be_loader)
```

## Configuration Parameters

### B2SCConfig Parameters

#### Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sc_training_data_path` | str | **Required** | Path to single-cell RNA-seq training data (.h5ad format) |
| `bulk_data_for_deconv_path` | str | None | Path to bulk RNA-seq data for deconvolution (.h5ad format) |
| `barcode_path` | str | None | Path to cell type barcode file (.csv format) |

#### Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | Auto-detected | Number of genes (features) |
| `latent_dim` | int | 1024 | **Shared** latent space dimension for both models |

##### Model-Specific Hidden Dimensions
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gmvae_hidden_dim` | int | 2048 | GMVAE hidden layer dimension |
| `bulk_hidden_dim` | int | 2048 | BulkEncoder hidden layer dimension |

**Important:** The `latent_dim` must be the same for both models because:
- GMVAE learns to encode cells into this latent space
- BulkEncoder predicts mixture parameters for the **same** latent space
- During generation, both models must work in the same latent space

**Note:** For backward compatibility, `hidden_dim` still works and refers to `gmvae_hidden_dim`.

#### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gmvae_epochs` | int | 30 | Number of training epochs for GMVAE |
| `bulk_encoder_epochs` | int | 100 | Number of training epochs for BulkEncoder |
| `gmvae_lr` | float | 5e-4 | Learning rate for GMVAE model |
| `bulk_encoder_lr` | float | 1e-3 | Learning rate for BulkEncoder model |
| `batch_size` | int | Auto-calculated | Batch size (typically num_cells/20) |

#### GMVAE Loss Weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gmvae_recon_loss_weight` | float | 5.0 | Weight for reconstruction loss |
| `gmvae_zinb_loss_weight` | float | 1.0 | Weight for ZINB loss |
| `gmvae_fraction_loss_weight` | float | 10.0 | Weight for fraction loss |
| `gmvae_kl_loss_weight` | float | 1.0 | Weight for KL divergence loss |

#### BulkEncoder Loss Weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bulk_mus_loss_weight` | float | 1.0 | Weight for means (mus) reconstruction loss |
| `bulk_logvars_loss_weight` | float | 1.0 | Weight for log-variances reconstruction loss |
| `bulk_pis_loss_weight` | float | 50.0 | Weight for mixture proportions (pis) loss |
| `bulk_uniform_penalty_weight` | float | 100.0 | Weight for uniform penalty (encourages diversity) |

#### Generation Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_cells_to_generate` | int | 500 | Number of cells to generate per bulk sample |

#### I/O Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_dir` | Path | "./results" | Directory for saving outputs |
| `device` | str | "auto" | Computing device ("auto", "cuda", "cpu", "mps") |

#### Advanced Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize_data` | bool | True | Whether to normalize single-cell data |
| `generate_pseudo_cells` | bool | False | Use clustering instead of cell type labels |
| `test_samples` | List[str] | [] | Sample IDs to exclude from training |

### Parameter Selection Guidelines

#### Model Architecture
- **`hidden_dim`**: Larger values (2048-4096) for complex datasets, smaller (512-1024) for simple datasets
- **`latent_dim`**: Should be smaller than hidden_dim, typically 256-1024
- **`input_dim`**: Automatically detected from data, represents number of genes

#### Training Parameters
- **`gmvae_epochs`**: 30-100 epochs usually sufficient, monitor training curves
- **`bulk_encoder_epochs`**: 50-200 epochs, depends on data complexity
- **`gmvae_lr`**: Start with 5e-4, reduce if GMVAE training is unstable
- **`bulk_encoder_lr`**: Start with 1e-3, reduce if BulkEncoder training is unstable

#### Loss Weight Tuning

If certain losses aren't decreasing properly (e.g., fraction loss), you can adjust the weights:

```python
# Example: Tune loss weights for better training
config = B2SCConfig(
    # ... other parameters ...
    
    # GMVAE loss weights
    gmvae_fraction_loss_weight=20.0,  # Default: 10.0 - increase if fraction loss not decreasing
    gmvae_recon_loss_weight=3.0,      # Default: 5.0 - reduce if reconstruction dominates
    gmvae_kl_loss_weight=2.0,         # Default: 1.0 - increase for better latent space
    
    # BulkEncoder loss weights  
    bulk_pis_loss_weight=100.0,       # Default: 50.0 - increase if fraction loss not decreasing
    bulk_uniform_penalty_weight=50.0, # Default: 100.0 - reduce if dominating
    bulk_mus_loss_weight=2.0,         # Default: 1.0 - fine-tune reconstruction
    bulk_logvars_loss_weight=2.0,     # Default: 1.0 - fine-tune reconstruction
)
```
- **`batch_size`**: Auto-calculated as num_cells/20, can be manually set

#### Generation Parameters
- **`num_cells_to_generate`**: 100-5000 cells per sample, depends on analysis needs

## Data Requirements

### Single-Cell Data Format

**File**: `.h5ad` (AnnData format)

**Required structure**:
```
adata.X: Gene expression matrix (cells × genes)
adata.obs: Cell metadata including:
  - 'cell_type': Cell type annotations (if available)
  - 'donor_id': Donor/sample identifiers (for BulkEncoder training)
adata.var: Gene metadata
```

**Example**:
```python
import scanpy as sc
adata = sc.read_h5ad("single_cell_data.h5ad")
print(adata.obs.columns)  # Should include 'cell_type', 'donor_id'
print(adata.shape)        # (n_cells, n_genes)
```

### Bulk Data Format

**File**: `.h5ad` (AnnData format)

**Required structure**:
```
adata.X: Gene expression matrix (samples × genes)
adata.obs: Sample metadata
adata.var: Gene metadata (should match single-cell data)
```

**Example**:
```python
bulk_adata = sc.read_h5ad("bulk_data.h5ad")
print(bulk_adata.shape)  # (n_samples, n_genes)
```

### Data Preprocessing

The pipeline automatically handles:
- **Normalization**: Log-normalization with target sum 1e4
- **Gene alignment**: Matches genes between single-cell and bulk data
- **Cell type encoding**: Converts cell type labels to integers

## Advanced Usage

### Custom Training Monitoring

```python
# Train with custom monitoring
pipeline = B2SCPipeline(config)

# GMVAE training with validation
gmvae_metrics = pipeline.train_gmvae()

# Check training progress
for epoch_metrics in gmvae_metrics.history:
    print(f"Epoch {epoch_metrics.epoch}: Loss = {epoch_metrics.total_loss:.4f}")

# Get best performing epoch
best_epoch = gmvae_metrics.get_best_epoch()
print(f"Best epoch: {best_epoch}")
```

### Custom Generation Parameters

```python
# Generate with specific parameters
results = pipeline.generate_cells(
    bulk_data_for_deconv_path="custom_bulk_data.h5ad",
    sample_ids=["sample_A", "sample_B", "sample_C"],
    num_cells=1000
)

# Access generated data
for sample_id, cells in results.generated_cells.items():
    print(f"Sample {sample_id}: {cells.shape[0]} cells, {cells.shape[1]} genes")
    
# Get cell type predictions
for sample_id, cell_types in results.cell_types.items():
    unique_types, counts = torch.unique(cell_types, return_counts=True)
    print(f"Sample {sample_id} cell types: {dict(zip(unique_types.tolist(), counts.tolist()))}")
```

### Model Checkpointing

```python
# Save model checkpoints during training
gmvae_trainer.save_checkpoint(epoch=25, is_best=True)

# Load from checkpoint
gmvae_trainer.load_checkpoint("path/to/checkpoint.pt")

# Resume training from checkpoint
metrics = gmvae_trainer.train(dataloader, start_epoch=25)
```

### Quality Assessment

```python
# Validate generation quality
quality_metrics = pipeline.validate_generation_quality(results)

print("Quality Assessment:")
for metric, value in quality_metrics.items():
    print(f"  {metric}: {value}")

# Key quality metrics:
# - total_cells_generated: Number of cells created
# - num_unique_cell_types: Diversity of cell types
# - mean_expression_per_cell: Average expression level
# - samples_with_negative_values: Data quality check
```

## Output Files

### Directory Structure

```
results/
├── GMVAE_model.pt              # Trained GMVAE model
├── GMVAE_mus.pt               # GMVAE latent parameters
├── GMVAE_logvars.pt           # GMVAE latent parameters  
├── GMVAE_pis.pt               # GMVAE mixture weights
├── bulkEncoder_model.pt        # Trained BulkEncoder model
├── gmvae_training_curves.png   # GMVAE training plots
├── bulk_encoder_training_curves.png  # BulkEncoder training plots
├── gmvae_metrics.pkl          # GMVAE training history
├── bulk_encoder_metrics.pkl   # BulkEncoder training history
├── generation_results/         # Generated cell data
│   ├── generated_cells.pkl    # Generated expression matrices
│   ├── cell_types.pkl         # Predicted cell types
│   ├── generation_stats.pkl   # Generation statistics
│   ├── summary_statistics.txt # Human-readable summary
│   ├── cell_type_distribution.png  # Cell type plots
│   └── per_sample_distribution.png # Per-sample plots
└── pipeline_config.txt        # Configuration used
```

### Loading Results

```python
from B2SC import GenerationResults

# Load saved results
results = GenerationResults.load_results("results/generation_results")

# Access data
generated_cells = results.generated_cells  # Dict[sample_id, tensor]
cell_types = results.cell_types            # Dict[sample_id, tensor]
stats = results.generation_stats           # Dict with statistics

# Get summary
summary = results.get_summary_statistics()
print(f"Generated {summary['total_cells_generated']} cells")
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues
```python
# Reduce batch size
config = B2SCConfig(
    sc_training_data_path="data.h5ad",
    batch_size=32,  # Smaller batch size
    hidden_dim=1024  # Smaller model
)
```

#### 2. Training Instability
```python
# Reduce learning rate
config = B2SCConfig(
    sc_training_data_path="data.h5ad",
    gmvae_lr=5e-4,
    bulk_encoder_lr=1e-3,
    gmvae_epochs=50      # More epochs
)
```

#### 3. Poor Generation Quality
```python
# Increase model capacity and training time
config = B2SCConfig(
    sc_training_data_path="data.h5ad",
    hidden_dim=4096,        # Larger model
    gmvae_epochs=100,       # More training
    bulk_encoder_epochs=200
)
```

#### 4. Device Issues
```python
# Specify device explicitly
config = B2SCConfig(
    sc_training_data_path="data.h5ad",
    device="cpu"  # or "cuda" or "mps"
)
```

### Error Messages

| Error | Solution |
|-------|----------|
| `FileNotFoundError: Single-cell data path not found` | Check that `sc_data_path` points to existing .h5ad file |
| `CUDA out of memory` | Reduce `batch_size` or `hidden_dim`, or use `device="cpu"` |
| `GMVAE must be trained before training BulkEncoder` | Train GMVAE first or use `pipeline.run_full_pipeline()` |
| `GMVAE parameter files not found` | Ensure GMVAE training completed successfully |

### Performance Tips

1. **Use GPU**: Set `device="cuda"` for faster training
2. **Optimize batch size**: Larger batches (if memory allows) train faster
3. **Monitor training**: Use training curves to detect convergence
4. **Save checkpoints**: Regular checkpointing prevents loss of progress

### Data Tips

1. **Gene alignment**: Ensure bulk and single-cell data have matching genes
2. **Cell type quality**: Clean cell type annotations improve results
3. **Sample size**: More diverse single-cell data improves generalization
4. **Normalization**: Use consistent normalization between datasets

## Example Workflows

### Workflow 1: Basic Analysis

```python
from B2SC import B2SCConfig, B2SCPipeline

# Simple configuration
config = B2SCConfig(
    sc_training_data_path="my_single_cell_data.h5ad",
    bulk_data_for_deconv_path="my_bulk_data.h5ad",
    base_dir="./analysis_results"
)

# Run pipeline
pipeline = B2SCPipeline(config)
gmvae_metrics, be_metrics, results = pipeline.run_full_pipeline()

# Save and analyze results
results.save_results(config.base_dir / "final_results")
print(f"Analysis complete! Generated cells for {len(results.generated_cells)} samples")
```

### Workflow 2: Research Analysis

```python
# Research-grade configuration
config = B2SCConfig(
    sc_training_data_path="reference_atlas.h5ad",
    bulk_data_for_deconv_path="patient_samples.h5ad",
    base_dir="./research_output",
    hidden_dim=4096,           # Large model
    latent_dim=2048,
    gmvae_epochs=100,          # Extensive training
    bulk_encoder_epochs=200,
    num_cells_to_generate=2000, # Many cells per sample
    gmvae_lr=1e-4,             # Conservative learning rate for GMVAE
    bulk_encoder_lr=1e-3       # Standard learning rate for BulkEncoder
)

# Step-by-step execution with monitoring
pipeline = B2SCPipeline(config)

# Train GMVAE with monitoring
gmvae_metrics = pipeline.train_gmvae()
gmvae_metrics.plot_training_curves(config.base_dir / "gmvae_analysis.png")

# Train BulkEncoder
be_metrics = pipeline.train_bulk_encoder()

# Generate cells for specific samples
results = pipeline.generate_cells(
    sample_ids=["patient_001", "patient_002", "control_001"],
    num_cells=5000
)

# Comprehensive quality assessment
quality = pipeline.validate_generation_quality(results)
print("Quality Assessment:", quality)

# Save everything
results.save_results(config.base_dir / "generation_output")
```

This comprehensive guide should help users understand how to use B2SC for their own analyses, from basic usage to advanced research applications.

