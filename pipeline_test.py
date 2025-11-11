#!/usr/bin/env python3
"""
Pipeline Test Script - Following Tutorial Flow

This script replicates the tutorial.ipynb flow for debugging purposes.
It generates mock data, splits it, configures the pipeline, and trains both
GMVAE and BulkEncoder models.

Usage:
    python pipeline_test.py
"""

# Standard library imports
import sys
import traceback
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Add the mock_data module to path
mock_data_path = (
    "/home/converge-bio/virtual-cell/virtual_cell/patient/src/explainability"
)
if mock_data_path not in sys.path:
    sys.path.append(mock_data_path)

# External module imports
from mock_data import mock_dataset_with_cell_types, get_cell_type_info

# Add B2SC to path and import components
b2sc_path = "/home/converge-bio/B2SC"
if b2sc_path not in sys.path:
    sys.path.append(b2sc_path)

# Local B2SC imports
from config import B2SCConfig
from pipeline import B2SCPipeline
from data_manager import DataManager
from gmvae_trainer import GMVAETrainer
from bulk_encoder_trainer import BulkEncoderTrainer
from utilities import calculate_pseudobulk

# Set up plotting
plt.style.use("default")
sns.set_palette("husl")


def main():
    """Main pipeline test function."""
    print("🚀 Starting B2SC Pipeline Test")
    print("=" * 50)

    # ============================================================================
    # 1. SETUP AND DATA GENERATION
    # ============================================================================
    print("\n📊 1. GENERATING MOCK DATA")
    print("-" * 30)

    # Parameters for mock data generation
    n_samples = 50
    n_cells = 500
    n_genes = 1000
    random_seed = 42

    # Define cell type proportions
    cell_type_proportions = {
        0: {"T_cell": 0.3, "B_cell": 0.5, "NK_cell": 0.2},
        1: {"T_cell": 0.6, "B_cell": 0.3, "NK_cell": 0.1},
    }

    # Define differential gene expression patterns for each class
    # Class 0: Control samples, Class 1: Disease samples
    differential_genes_per_class = [
        {  # Class 0 (Control)
            "T_cell": {10: 0.5, 25: 0.7, 50: 0.6},  # Lower expression
            "B_cell": {100: 0.4, 150: 0.8, 200: 0.5},
            "NK_cell": {300: 0.6, 350: 0.7, 400: 0.4},
        },
        {  # Class 1 (Disease)
            "T_cell": {10: 2.0, 25: 1.8, 50: 2.2},  # Higher expression
            "B_cell": {100: 2.5, 150: 1.5, 200: 2.0},
            "NK_cell": {300: 1.8, 350: 1.6, 400: 2.1},
        },
    ]

    print("Generating mock single-cell data with ZINB distribution...")
    X, y, cell_types = mock_dataset_with_cell_types(
        n_samples=n_samples,
        n_cells=n_cells,
        n_genes=n_genes,
        cell_type_proportions=cell_type_proportions,
        differential_genes_per_class=differential_genes_per_class,
        random_seed=random_seed,
        num_classes=2,
        # ZINB parameters for realistic scRNA-seq data
        nb_n=10.0,  # Dispersion parameter
        nb_p=0.3,  # Success probability
        zero_inflation=0.6,  # 60% dropout rate (realistic for scRNA-seq)
    )

    print(f"✅ Generated data shape: {X.shape}")
    print(f"✅ Labels shape: {y.shape}")
    print(f"✅ Cell types shape: {cell_types.shape}")
    print(f"✅ Class distribution: {np.bincount(y)}")

    # Show ZINB characteristics
    total_zeros = np.sum(X == 0)
    total_values = X.size
    zero_fraction = total_zeros / total_values
    print(f"✅ ZINB characteristics:")
    print(f"   - Zero fraction: {zero_fraction:.3f} ({total_zeros:,}/{total_values:,})")
    print(f"   - Mean expression: {np.mean(X):.3f}")
    print(f"   - Expression range: [{np.min(X):.1f}, {np.max(X):.1f}]")
    print(f"   - Standard deviation: {np.std(X):.3f}")

    # ============================================================================
    # 2. CONVERT TO ANNDATA FORMAT
    # ============================================================================
    print("\n🔄 2. CONVERTING TO ANNDATA FORMAT")
    print("-" * 35)

    # Reshape data: (n_samples, n_cells, n_genes) -> (n_samples * n_cells, n_genes)
    X_reshaped = X.reshape(-1, n_genes)
    print(f"Reshaped data: {X_reshaped.shape}")

    # Create cell and gene names
    cell_names = [
        f"sample_{i}_cell_{j}" for i in range(n_samples) for j in range(n_cells)
    ]
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # Create donor IDs and sample labels for each cell
    donor_ids = [f"donor_{i}" for i in range(n_samples) for _ in range(n_cells)]
    sample_labels = [y[i] for i in range(n_samples) for _ in range(n_cells)]

    # Flatten cell types
    cell_types_flat = cell_types.flatten()

    # Map cell type indices to names
    if isinstance(list(cell_type_proportions.values())[0], dict):
        # Class-specific proportions: extract cell type names from first class
        cell_type_names = sorted(list(cell_type_proportions[0].keys()))
    else:
        # Global proportions: use keys directly
        cell_type_names = list(cell_type_proportions.keys())

    cell_type_labels = [cell_type_names[ct] for ct in cell_types_flat]

    # Create AnnData object
    adata = anndata.AnnData(
        X=X_reshaped,
        obs=pd.DataFrame(
            {
                "donor_id": donor_ids,
                "cell_type": cell_type_labels,
                "sample_label": sample_labels,
            },
            index=cell_names,
        ),
        var=pd.DataFrame(index=gene_names),
    )

    print(f"✅ AnnData object created: {adata.shape}")
    print(f"✅ Observations: {list(adata.obs.columns)}")
    print(f"✅ Cell type distribution:")
    print(adata.obs["cell_type"].value_counts())

    # ============================================================================
    # 3. DATA SPLITTING
    # ============================================================================
    print("\n✂️  3. SPLITTING DATA")
    print("-" * 20)

    # Get unique donors and their labels
    donor_info = adata.obs.groupby("donor_id")["sample_label"].first().reset_index()
    unique_donors = donor_info["donor_id"].values
    donor_labels = donor_info["sample_label"].values

    print(f"Total donors: {len(unique_donors)}")
    print(f"Donor label distribution: {np.bincount(donor_labels)}")

    # Split donors into train/val/test (80/10/10)
    # First split: 80% train, 20% temp
    train_donors, temp_donors, train_labels, temp_labels = train_test_split(
        unique_donors,
        donor_labels,
        test_size=0.2,
        random_state=random_seed,
        stratify=donor_labels,
    )

    # Second split: 10% val, 10% test from the 20% temp
    val_donors, test_donors, val_labels, test_labels = train_test_split(
        temp_donors,
        temp_labels,
        test_size=0.5,
        random_state=random_seed,
        stratify=temp_labels,
    )

    print(f"\nSplit sizes:")
    print(
        f"Train donors: {len(train_donors)} ({len(train_donors)/len(unique_donors)*100:.1f}%)"
    )
    print(
        f"Val donors: {len(val_donors)} ({len(val_donors)/len(unique_donors)*100:.1f}%)"
    )
    print(
        f"Test donors: {len(test_donors)} ({len(test_donors)/len(unique_donors)*100:.1f}%)"
    )

    # Create boolean masks for each split
    train_mask = adata.obs["donor_id"].isin(train_donors)
    val_mask = adata.obs["donor_id"].isin(val_donors)
    test_mask = adata.obs["donor_id"].isin(test_donors)

    # Create split datasets
    adata_train = adata[train_mask].copy()
    adata_val = adata[val_mask].copy()
    adata_test = adata[test_mask].copy()

    print(f"\nDataset sizes:")
    print(
        f"Train: {adata_train.shape[0]:,} cells from {adata_train.obs['donor_id'].nunique()} donors"
    )
    print(
        f"Val: {adata_val.shape[0]:,} cells from {adata_val.obs['donor_id'].nunique()} donors"
    )
    print(
        f"Test: {adata_test.shape[0]:,} cells from {adata_test.obs['donor_id'].nunique()} donors"
    )

    # ============================================================================
    # 4. SAVE DATA FILES
    # ============================================================================
    print("\n💾 4. SAVING DATA FILES")
    print("-" * 25)

    output_dir = Path("/home/converge-bio/B2SC/examples/pipeline_test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving training data...")
    adata_train.write_h5ad(output_dir / "train_data.h5ad")
    print(f"✅ Training data saved to: {output_dir / 'train_data.h5ad'}")

    # Also save validation and test data for completeness
    adata_val.write_h5ad(output_dir / "val_data.h5ad")
    adata_test.write_h5ad(output_dir / "test_data.h5ad")
    print(f"✅ Validation data saved to: {output_dir / 'val_data.h5ad'}")
    print(f"✅ Test data saved to: {output_dir / 'test_data.h5ad'}")

    # ============================================================================
    # 5. CONFIGURE B2SC PIPELINE
    # ============================================================================
    print("\n⚙️  5. CONFIGURING B2SC PIPELINE")
    print("-" * 35)

    # Configure B2SC pipeline
    config = B2SCConfig(
        sc_training_path=str(output_dir / "train_data.h5ad"),
        sc_validation_path=str(output_dir / "val_data.h5ad"),  # Use validation data
        pseudobulk_groupby="donor_id",  # Group by donor for pseudobulk computation
        cache_pseudobulk=False,  # Disable caching for debugging
        base_dir=str(output_dir / "results"),
        latent_dim=128,
        device="auto",
    )

    # Set model-specific hyperparameters using the new hierarchical structure
    config.gmvae_hyperparameters.update(
        {
            "hidden_dim": 256,
            "epochs": 3,  # Fewer epochs for debugging
        }
    )

    config.bulk_encoder_hyperparameters.update(
        {
            "hidden_dim": 1024,
            "epochs": 30,
            "learning_rate": 1e-5,
        }
    )

    # Example of setting hyperparameters using the standardized format
    config.set_hyperparameters(
        {"gmvae": {"fraction_loss_weight": 50.0, "learning_rate": 1e-6}}
    )

    print(f"Configuration:")
    print(f"  Training data: {config.sc_training_path}")
    print(f"  Validation data: {config.sc_validation_path}")
    print(f"  Pseudobulk groupby: {config.pseudobulk_groupby}")
    print(f"  Cache enabled: {config.cache_pseudobulk}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  GMVAE epochs: {config.gmvae_epochs}")
    print(f"  BulkEncoder epochs: {config.bulk_encoder_epochs}")
    print(f"  Device: {config.device}")
    print(f"\nGMVAE hyperparameters: {config.gmvae_hyperparameters}")
    print(f"BulkEncoder hyperparameters: {config.bulk_encoder_hyperparameters}")

    # ============================================================================
    # 6. INITIALIZE PIPELINE
    # ============================================================================
    print("\n🏗️  6. INITIALIZING PIPELINE")
    print("-" * 30)

    # Initialize pipeline
    print("Initializing B2SC pipeline...")
    pipeline = B2SCPipeline(config)

    print(f"✅ Pipeline initialized")
    print(f"Training data file exists: {Path(config.sc_training_path).exists()}")
    print(
        f"Validation data file exists: {Path(config.sc_validation_path).exists() if config.sc_validation_path else 'No validation path'}"
    )

    # ============================================================================
    # 7. TRAIN GMVAE
    # ============================================================================
    print("\n🧠 7. TRAINING GMVAE")
    print("-" * 25)
    print("=" * 50)
    print("TRAINING GMVAE")
    print("=" * 50)

    try:
        gmvae_metrics = pipeline.train_gmvae()
        print(f"\n✅ GMVAE training completed!")
        print(f"Total epochs: {len(gmvae_metrics.history)}")
        if gmvae_metrics.history:
            final_metrics = gmvae_metrics.history[-1]
            print(f"Final training loss: {final_metrics.total_loss:.4f}")
            if (
                hasattr(final_metrics, "val_total_loss")
                and final_metrics.val_total_loss is not None
            ):
                print(f"Final validation loss: {final_metrics.val_total_loss:.4f}")
                print(
                    f"Training/Validation ratio: {final_metrics.total_loss / final_metrics.val_total_loss:.2f}x"
                )

            # Print loss breakdown
            print(f"\nLoss breakdown (final epoch):")
            print(f"  Reconstruction: {final_metrics.reconstruction_loss:.4f}")
            print(f"  KL: {final_metrics.kl_loss:.4f}")
            print(f"  Fraction: {final_metrics.fraction_loss:.4f}")
            print(f"  ZINB: {final_metrics.zinb_loss:.4f}")

            if (
                hasattr(final_metrics, "val_reconstruction_loss")
                and final_metrics.val_reconstruction_loss is not None
            ):
                print(f"\nValidation loss breakdown (final epoch):")
                print(f"  Reconstruction: {final_metrics.val_reconstruction_loss:.4f}")
                print(f"  KL: {final_metrics.val_kl_loss:.4f}")
                print(f"  Fraction: {final_metrics.val_fraction_loss:.4f}")
                print(f"  ZINB: {final_metrics.val_zinb_loss:.4f}")

    except Exception as e:
        print(f"❌ GMVAE training failed: {e}")
        print("Stopping pipeline test.")
        return False

    # ============================================================================
    # 8. TRAIN BULK ENCODER
    # ============================================================================
    print("\n🔄 8. TRAINING BULK ENCODER")
    print("-" * 30)
    print("=" * 50)
    print("TRAINING BULK ENCODER")
    print("=" * 50)

    try:
        be_metrics = pipeline.train_bulk_encoder()
        print(f"\n✅ BulkEncoder training completed!")
        print(f"Total epochs: {len(be_metrics.history)}")
        if be_metrics.history:
            final_metrics = be_metrics.history[-1]
            print(f"Final training loss: {final_metrics.total_loss:.4f}")
            if (
                hasattr(final_metrics, "val_total_loss")
                and final_metrics.val_total_loss is not None
            ):
                print(f"Final validation loss: {final_metrics.val_total_loss:.4f}")
                print(
                    f"Training/Validation ratio: {final_metrics.total_loss / final_metrics.val_total_loss:.2f}x"
                )

            # Print loss breakdown
            print(f"\nLoss breakdown (final epoch):")
            print(f"  Mus: {final_metrics.mus_loss:.4f}")
            print(f"  LogVars: {final_metrics.logvars_loss:.4f}")
            print(f"  Pis: {final_metrics.pis_loss:.4f}")
            print(f"  Uniform Penalty: {final_metrics.uniform_penalty:.4f}")

    except Exception as e:
        print(f"❌ BulkEncoder training failed: {e}")
        print("Stopping pipeline test.")
        return False

    # ============================================================================
    # 9. TEST DECONVOLUTION COMPARISON VISUALIZATION
    # ============================================================================
    print("\n🎨 9. TESTING DECONVOLUTION COMPARISON VISUALIZATION")
    print("-" * 55)
    print("=" * 50)
    print("TESTING VISUALIZATION")
    print("=" * 50)

    try:
        print("Running deconvolution comparison visualization...")

        # Test with a small number of samples for debugging
        viz_result = pipeline.visualize_deconvolution_comparison(
            max_samples_per_split=10,  # Small number for debugging
            save_path=str(
                output_dir / "results" / "deconvolution_comparison_debug.png"
            ),
            random_seed=42,
        )

        print(f"\n✅ Deconvolution comparison visualization completed!")

        if viz_result:
            print(f"Visualization result keys: {list(viz_result.keys())}")

            # Print sample information
            if viz_result.get("train_deconv_pca") is not None:
                print(
                    f"Training samples processed: {len(viz_result['train_deconv_pca'])}"
                )

            if viz_result.get("val_deconv_pca") is not None:
                print(
                    f"Validation samples processed: {len(viz_result['val_deconv_pca'])}"
                )

            # Print PCA variance information
            if viz_result.get("pca_model") is not None:
                pca_model = viz_result["pca_model"]
                total_variance = pca_model.explained_variance_ratio_[:2].sum()
                print(f"PCA variance explained: {total_variance:.1%}")
        print(f"PC1: {pca_model.explained_variance_ratio_[0]:.1%}")
        print(f"PC2: {pca_model.explained_variance_ratio_[1]:.1%}")

    except Exception as e:
        print(f"❌ Deconvolution comparison visualization failed: {e}")
        traceback.print_exc()
        print("Continuing with pipeline test...")

    # ============================================================================
    # 10. SUMMARY
    # ============================================================================
    print("\n🎉 10. PIPELINE TEST SUMMARY")
    print("-" * 30)

    print("✅ Pipeline test completed successfully!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"📋 Generated files:")
    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            print(f"  - {file_path.name}")

    # Print final metrics comparison
    if gmvae_metrics.history and be_metrics.history:
        gmvae_final = gmvae_metrics.history[-1]
        be_final = be_metrics.history[-1]

        print(f"\n📊 Final Training Metrics:")
        print(f"  GMVAE Total Loss: {gmvae_final.total_loss:.4f}")
        print(f"  BulkEncoder Total Loss: {be_final.total_loss:.4f}")

        if (
            hasattr(gmvae_final, "val_total_loss")
            and gmvae_final.val_total_loss is not None
        ):
            print(f"  GMVAE Validation Loss: {gmvae_final.val_total_loss:.4f}")
        if hasattr(be_final, "val_total_loss") and be_final.val_total_loss is not None:
            print(f"  BulkEncoder Validation Loss: {be_final.val_total_loss:.4f}")

    return True


if __name__ == "__main__":
    print("✅ All imports successful!")
    success = main()
    if success:
        print("\n🎯 Pipeline test completed successfully!")
    else:
        print("\n❌ Pipeline test failed!")
        sys.exit(1)
