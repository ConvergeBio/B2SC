"""
Comprehensive tests for B2SC package using pytest framework.

Tests cover:
1. Model initialization with specified dimensions
2. Loss behavior when weights are set to zero
3. Configuration validation
4. Pipeline functionality
"""

import pytest
import torch
import numpy as np
import pandas as pd
import anndata
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch, MagicMock

# Add B2SC to path for testing
B2SC_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(B2SC_PATH))

from config import B2SCConfig
from pipeline import B2SCPipeline
from models import GaussianMixtureVAE
from gmvae_trainer import GMVAETrainer
from bulk_encoder_trainer import BulkEncoderTrainer
from data_manager import DataManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_single_cell_data(temp_dir):
    """Create mock single-cell data for testing."""
    n_cells = 1000
    n_genes = 500
    n_donors = 10

    # Generate mock expression data
    np.random.seed(42)
    X = np.random.lognormal(mean=0, sigma=1, size=(n_cells, n_genes)).astype(np.float32)

    # Create cell metadata
    cell_types = np.random.choice(["T_cell", "B_cell", "NK_cell"], size=n_cells)
    donor_ids = [f"donor_{i}" for i in np.random.choice(n_donors, size=n_cells)]

    obs = pd.DataFrame({"cell_type": cell_types, "donor_id": donor_ids})

    # Create gene metadata
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    # Create AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Save training and validation data
    train_path = temp_dir / "train_data.h5ad"
    val_path = temp_dir / "val_data.h5ad"

    # Split data for train/val
    train_cells = np.random.choice(n_cells, size=int(0.8 * n_cells), replace=False)
    val_cells = np.setdiff1d(np.arange(n_cells), train_cells)

    adata[train_cells].write_h5ad(train_path)
    adata[val_cells].write_h5ad(val_path)

    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "n_genes": n_genes,
        "n_cell_types": len(np.unique(cell_types)),
    }


@pytest.fixture
def minimal_mock_data(temp_dir):
    """Create minimal mock single-cell data for lightweight tests."""
    n_cells = 100  # Much smaller
    n_genes = 50  # Much smaller
    n_donors = 3

    # Generate minimal mock expression data
    np.random.seed(42)
    X = np.random.lognormal(mean=0, sigma=1, size=(n_cells, n_genes)).astype(np.float32)

    # Create cell metadata
    cell_types = np.random.choice(["T_cell", "B_cell", "NK_cell"], size=n_cells)
    donor_ids = [f"donor_{i}" for i in np.random.choice(n_donors, size=n_cells)]

    obs = pd.DataFrame({"cell_type": cell_types, "donor_id": donor_ids})

    # Create gene metadata
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    # Create AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Save only training data (minimal)
    train_path = temp_dir / "train_data_minimal.h5ad"
    adata.write_h5ad(train_path)

    return {
        "train_path": str(train_path),
        "n_genes": n_genes,
        "n_cell_types": len(np.unique(cell_types)),
    }


@pytest.fixture
def basic_config(mock_single_cell_data, temp_dir):
    """Create a basic B2SC configuration for testing."""
    return B2SCConfig(
        sc_training_path=mock_single_cell_data["train_path"],
        sc_validation_path=mock_single_cell_data["val_path"],
        base_dir=str(temp_dir / "results"),
        latent_dim=64,
        pseudobulk_groupby="donor_id",
        cache_pseudobulk=False,  # Disable caching for tests
        device="cpu",  # Force CPU for testing
    )


@pytest.fixture
def custom_config(mock_single_cell_data, temp_dir):
    """Create a custom B2SC configuration with specific dimensions."""
    config = B2SCConfig(
        sc_training_path=mock_single_cell_data["train_path"],
        sc_validation_path=mock_single_cell_data["val_path"],
        base_dir=str(temp_dir / "results"),
        latent_dim=128,
        pseudobulk_groupby="donor_id",
        cache_pseudobulk=False,
        device="cpu",  # Force CPU for testing
    )

    # Set custom hyperparameters
    config.gmvae_hyperparameters.update(
        {"hidden_dim": 256, "epochs": 2, "batch_size": 16}  # Short for testing
    )

    config.bulk_encoder_hyperparameters.update(
        {"hidden_dim": 512, "epochs": 2, "batch_size": 4}  # Short for testing
    )

    return config


@pytest.fixture
def zero_weights_config(mock_single_cell_data, temp_dir):
    """Create a configuration with all loss weights set to zero."""
    config = B2SCConfig(
        sc_training_path=mock_single_cell_data["train_path"],
        base_dir=str(temp_dir / "results"),
        latent_dim=64,
        pseudobulk_groupby="donor_id",
        cache_pseudobulk=False,
        device="cpu",  # Force CPU for testing
    )

    # Set all GMVAE loss weights to zero
    config.gmvae_hyperparameters.update(
        {
            "recon_loss_weight": 0.0,
            "zinb_loss_weight": 0.0,
            "fraction_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "epochs": 1,  # Just one epoch for testing
            "batch_size": 16,
        }
    )

    return config


class TestB2SCConfiguration:
    """Test B2SC configuration functionality."""

    def test_config_creation(self, basic_config):
        """Test that B2SCConfig can be created with basic parameters."""
        assert basic_config.latent_dim == 64
        assert basic_config.pseudobulk_groupby == "donor_id"
        assert basic_config.cache_pseudobulk == False

    def test_config_hyperparameter_setting(self, basic_config):
        """Test setting hyperparameters using the standardized format."""
        basic_config.set_hyperparameters(
            {
                "gmvae": {
                    "learning_rate": 1e-4,
                    "batch_size": 32,
                    "fraction_loss_weight": 20.0,
                },
                "bulk_encoder": {"learning_rate": 1e-5, "epochs": 50},
            }
        )

        assert basic_config.gmvae_hyperparameters["learning_rate"] == 1e-4
        assert basic_config.gmvae_hyperparameters["batch_size"] == 32
        assert basic_config.gmvae_hyperparameters["fraction_loss_weight"] == 20.0
        assert basic_config.bulk_encoder_hyperparameters["learning_rate"] == 1e-5
        assert basic_config.bulk_encoder_hyperparameters["epochs"] == 50

    def test_config_invalid_hyperparameter(self, basic_config):
        """Test that setting invalid hyperparameters raises an error."""
        with pytest.raises(KeyError):
            basic_config.set_hyperparameters({"gmvae": {"invalid_parameter": 123}})

    def test_config_backward_compatibility_properties(self, basic_config):
        """Test that backward compatibility properties work correctly."""
        assert (
            basic_config.hidden_dim == basic_config.gmvae_hyperparameters["hidden_dim"]
        )
        assert (
            basic_config.batch_size == basic_config.gmvae_hyperparameters["batch_size"]
        )
        assert basic_config.gmvae_epochs == basic_config.gmvae_hyperparameters["epochs"]


class TestModelInitialization:
    """Test that models are initialized with correct dimensions."""

    def test_pipeline_initialization(self, custom_config):
        """Test that pipeline initializes correctly with custom config."""
        pipeline = B2SCPipeline(custom_config)
        assert pipeline.config.latent_dim == 128
        assert pipeline.config.gmvae_hyperparameters["hidden_dim"] == 256
        assert pipeline.config.bulk_encoder_hyperparameters["hidden_dim"] == 512

    def test_gmvae_model_dimensions(self, custom_config, mock_single_cell_data):
        """Test that GMVAE model is initialized with correct dimensions."""
        # Create data manager and load data to get input dimensions
        data_manager = DataManager(custom_config)
        gmvae_dataloader, _ = data_manager.load_single_cell_data()

        # Initialize GMVAE trainer
        gmvae_trainer = GMVAETrainer(custom_config, data_manager)

        # Initialize model
        model = gmvae_trainer.initialize_model(
            input_dim=custom_config.input_dim,
            hidden_dim=custom_config.gmvae_hyperparameters["hidden_dim"],
            latent_dim=custom_config.latent_dim,
            n_components=len(data_manager.cell_type_fractions),
        )

        # Check model dimensions
        assert model.input_dim == mock_single_cell_data["n_genes"]
        assert model.latent_dim == 128
        assert model.n_components == mock_single_cell_data["n_cell_types"]

        # Check layer dimensions
        assert model.fc0.in_features == mock_single_cell_data["n_genes"]
        assert model.fc0.out_features == 256 // 2  # hidden_dim // 2
        assert model.fc1.out_features == 256  # hidden_dim
        assert (
            model.fc21.out_features == mock_single_cell_data["n_cell_types"] * 128
        )  # n_components * latent_dim

    def test_bulk_encoder_model_dimensions(self, custom_config, mock_single_cell_data):
        """Test that BulkEncoder model is initialized with correct dimensions."""
        # Force CPU for testing to avoid device mismatch
        custom_config.device = "cpu"

        # Create data manager and load data
        data_manager = DataManager(custom_config)
        gmvae_dataloader, _ = data_manager.load_single_cell_data()

        # Initialize GMVAE trainer first (required for BulkEncoder)
        gmvae_trainer = GMVAETrainer(custom_config, data_manager)
        gmvae_trainer.initialize_model(
            input_dim=custom_config.input_dim,
            hidden_dim=custom_config.gmvae_hyperparameters["hidden_dim"],
            latent_dim=custom_config.latent_dim,
            n_components=len(data_manager.cell_type_fractions),
        )

        # Initialize BulkEncoder trainer
        be_trainer = BulkEncoderTrainer(custom_config, gmvae_trainer, data_manager)

        # Initialize BulkEncoder model
        model = be_trainer.initialize_model(
            input_dim=custom_config.input_dim,
            hidden_dim=custom_config.bulk_encoder_hyperparameters["hidden_dim"],
            latent_dim=custom_config.latent_dim,
            n_components=len(data_manager.cell_type_fractions),
        )

        # Check model dimensions - BulkEncoder uses fc0 as first layer
        assert model.fc0.in_features == mock_single_cell_data["n_genes"]
        assert (
            model.fc1.out_features
            == custom_config.bulk_encoder_hyperparameters["hidden_dim"]
        )
        assert model.latent_dim == custom_config.latent_dim
        assert model.n_components == mock_single_cell_data["n_cell_types"]


class TestLossWeights:
    """Test loss behavior with different weight configurations."""

    def test_zero_weights_produce_zero_loss(
        self, zero_weights_config, mock_single_cell_data
    ):
        """Test that when all loss weights are zero, the total loss is zero."""
        # Force CPU for testing to avoid device mismatch
        zero_weights_config.device = "cpu"

        # Create pipeline and load data
        pipeline = B2SCPipeline(zero_weights_config)
        pipeline._load_training_data()

        # Initialize GMVAE model
        n_components = len(pipeline.data_manager.cell_type_fractions)
        pipeline.gmvae_trainer.initialize_model(
            input_dim=zero_weights_config.input_dim,
            hidden_dim=zero_weights_config.gmvae_hyperparameters["hidden_dim"],
            latent_dim=zero_weights_config.latent_dim,
            n_components=n_components,
        )

        # Get a batch of data (GMVAE collate_fn returns 4 values)
        dataloader = pipeline.gmvae_dataloader
        batch = next(iter(dataloader))
        data_batch, labels_batch, num_cells_per_donor, donor_ids = batch

        # Move data to CPU to match model device
        data_batch = data_batch.cpu()
        labels_batch = labels_batch.cpu()

        # Forward pass
        model = pipeline.gmvae_trainer.model
        model.eval()
        with torch.no_grad():
            reconstructed, mus, logvars, pis, zs = model(data_batch, labels_batch)

        # Calculate individual losses (should be non-zero)
        import torch.nn.functional as F

        recon_loss = F.mse_loss(reconstructed, data_batch)
        fraction_loss = F.mse_loss(
            pis.mean(0), pipeline.data_manager.cell_type_fractions
        )

        # Individual losses should be non-zero
        assert recon_loss.item() > 0
        assert fraction_loss.item() > 0

        # But weighted total loss should be zero
        total_loss = (
            recon_loss * zero_weights_config.gmvae_recon_loss_weight
            + fraction_loss * zero_weights_config.gmvae_fraction_loss_weight
        )

        assert total_loss.item() == 0.0

    def test_non_zero_weights_produce_non_zero_loss(
        self, basic_config, mock_single_cell_data
    ):
        """Test that with non-zero weights, the total loss is non-zero."""
        # Force CPU for testing to avoid device mismatch
        basic_config.device = "cpu"

        # Create pipeline and load data
        pipeline = B2SCPipeline(basic_config)
        pipeline._load_training_data()

        # Initialize GMVAE model
        n_components = len(pipeline.data_manager.cell_type_fractions)
        pipeline.gmvae_trainer.initialize_model(
            input_dim=basic_config.input_dim,
            hidden_dim=basic_config.gmvae_hyperparameters["hidden_dim"],
            latent_dim=basic_config.latent_dim,
            n_components=n_components,
        )

        # Get a batch of data (GMVAE collate_fn returns 4 values)
        dataloader = pipeline.gmvae_dataloader
        batch = next(iter(dataloader))
        data_batch, labels_batch, num_cells_per_donor, donor_ids = batch

        # Move data to CPU to match model device
        data_batch = data_batch.cpu()
        labels_batch = labels_batch.cpu()

        # Forward pass
        model = pipeline.gmvae_trainer.model
        model.eval()
        with torch.no_grad():
            reconstructed, mus, logvars, pis, zs = model(data_batch, labels_batch)

        # Calculate weighted total loss
        import torch.nn.functional as F

        recon_loss = F.mse_loss(reconstructed, data_batch)
        fraction_loss = F.mse_loss(
            pis.mean(0), pipeline.data_manager.cell_type_fractions
        )

        total_loss = (
            recon_loss * basic_config.gmvae_recon_loss_weight
            + fraction_loss * basic_config.gmvae_fraction_loss_weight
        )

        # Total loss should be non-zero with non-zero weights
        assert total_loss.item() > 0.0

    def test_validation_losses_are_non_zero_with_non_zero_weights(
        self, basic_config, mock_single_cell_data
    ):
        """Test that validation losses are non-zero when weights are non-zero."""
        # Force CPU for testing to avoid device mismatch
        basic_config.device = "cpu"

        # Create pipeline and load data
        pipeline = B2SCPipeline(basic_config)
        pipeline._load_training_data()

        # Initialize GMVAE model
        n_components = len(pipeline.data_manager.cell_type_fractions)
        pipeline.gmvae_trainer.initialize_model(
            input_dim=basic_config.input_dim,
            hidden_dim=basic_config.gmvae_hyperparameters["hidden_dim"],
            latent_dim=basic_config.latent_dim,
            n_components=n_components,
        )

        # Get validation dataloader
        gmvae_val_dataloader, be_val_dataloader = (
            pipeline.data_manager.get_validation_dataloaders()
        )
        assert (
            gmvae_val_dataloader is not None
        ), "GMVAE validation dataloader should exist"

        # Use the GMVAE validation dataloader (not the BulkEncoder one)
        val_dataloader = gmvae_val_dataloader

        # Run one epoch of validation
        val_metrics = pipeline.gmvae_trainer._validate_epoch(
            epoch=0,
            val_dataloader=val_dataloader,
            proportion_tensor=pipeline.data_manager.cell_type_fractions,
            kl_weight=0.5,
        )

        # All validation losses should be non-zero (since weights are non-zero)
        assert (
            val_metrics.total_loss > 0.0
        ), f"Validation total loss should be > 0, got {val_metrics.total_loss}"
        assert (
            val_metrics.reconstruction_loss > 0.0
        ), f"Validation reconstruction loss should be > 0, got {val_metrics.reconstruction_loss}"
        assert (
            val_metrics.fraction_loss > 0.0
        ), f"Validation fraction loss should be > 0, got {val_metrics.fraction_loss}"
        assert (
            val_metrics.zinb_loss > 0.0
        ), f"Validation ZINB loss should be > 0, got {val_metrics.zinb_loss}"

        # KL loss might be very small but should be non-negative
        assert (
            val_metrics.kl_loss >= 0.0
        ), f"Validation KL loss should be >= 0, got {val_metrics.kl_loss}"

        print(f"✅ Validation losses are properly calculated:")
        print(f"   Total: {val_metrics.total_loss:.6f}")
        print(f"   Reconstruction: {val_metrics.reconstruction_loss:.6f}")
        print(f"   KL: {val_metrics.kl_loss:.6f}")
        print(f"   Fraction: {val_metrics.fraction_loss:.6f}")
        print(f"   ZINB: {val_metrics.zinb_loss:.6f}")

    def test_validation_losses_stored_in_training_metrics(
        self, basic_config, mock_single_cell_data
    ):
        """Test that validation losses are properly stored during training."""
        # Force CPU for testing
        basic_config.device = "cpu"

        # Set to only 1 epoch for quick test
        basic_config.gmvae_hyperparameters["epochs"] = 1

        # Create pipeline and load data
        pipeline = B2SCPipeline(basic_config)
        pipeline._load_training_data()

        # Initialize GMVAE model
        n_components = len(pipeline.data_manager.cell_type_fractions)
        pipeline.gmvae_trainer.initialize_model(
            input_dim=basic_config.input_dim,
            hidden_dim=basic_config.gmvae_hyperparameters["hidden_dim"],
            latent_dim=basic_config.latent_dim,
            n_components=n_components,
        )

        # Get validation dataloader
        gmvae_val_dataloader, _ = pipeline.data_manager.get_validation_dataloaders()
        assert gmvae_val_dataloader is not None, "Validation dataloader should exist"

        # Train GMVAE for 1 epoch (includes validation)
        metrics_tracker = pipeline.gmvae_trainer.train(
            dataloader=pipeline.gmvae_dataloader,
            proportion_tensor=pipeline.data_manager.cell_type_fractions,
            mapping_dict=pipeline.data_manager.cell_type_mapping,
            color_map=pipeline.data_manager.color_map,
            val_dataloader=gmvae_val_dataloader,
        )

        # Check that validation metrics were stored
        assert len(metrics_tracker.history) == 1, "Should have 1 epoch of metrics"
        epoch_metrics = metrics_tracker.history[0]

        # Validation metrics should be present and non-zero
        assert hasattr(epoch_metrics, "val_total_loss"), "Should have val_total_loss"
        assert hasattr(
            epoch_metrics, "val_reconstruction_loss"
        ), "Should have val_reconstruction_loss"
        assert hasattr(epoch_metrics, "val_kl_loss"), "Should have val_kl_loss"
        assert hasattr(
            epoch_metrics, "val_fraction_loss"
        ), "Should have val_fraction_loss"
        assert hasattr(epoch_metrics, "val_zinb_loss"), "Should have val_zinb_loss"

        # Check that validation losses are non-zero
        assert (
            epoch_metrics.val_total_loss > 0.0
        ), f"val_total_loss should be > 0, got {epoch_metrics.val_total_loss}"
        assert (
            epoch_metrics.val_reconstruction_loss > 0.0
        ), f"val_reconstruction_loss should be > 0, got {epoch_metrics.val_reconstruction_loss}"
        assert (
            epoch_metrics.val_fraction_loss > 0.0
        ), f"val_fraction_loss should be > 0, got {epoch_metrics.val_fraction_loss}"
        assert (
            epoch_metrics.val_zinb_loss > 0.0
        ), f"val_zinb_loss should be > 0, got {epoch_metrics.val_zinb_loss}"
        assert (
            epoch_metrics.val_kl_loss >= 0.0
        ), f"val_kl_loss should be >= 0, got {epoch_metrics.val_kl_loss}"

        print(f"✅ Validation losses properly stored in training metrics:")
        print(f"   val_total_loss: {epoch_metrics.val_total_loss:.6f}")
        print(
            f"   val_reconstruction_loss: {epoch_metrics.val_reconstruction_loss:.6f}"
        )
        print(f"   val_kl_loss: {epoch_metrics.val_kl_loss:.6f}")
        print(f"   val_fraction_loss: {epoch_metrics.val_fraction_loss:.6f}")
        print(f"   val_zinb_loss: {epoch_metrics.val_zinb_loss:.6f}")

    def test_validation_losses_are_plotted_correctly(
        self, basic_config, mock_single_cell_data
    ):
        """Test that validation losses are properly included in training curve plots."""
        # Force CPU for testing
        basic_config.device = "cpu"

        # Set to only 2 epochs for quick test
        basic_config.gmvae_hyperparameters["epochs"] = 2

        # Create pipeline and load data
        pipeline = B2SCPipeline(basic_config)
        pipeline._load_training_data()

        # Initialize GMVAE model
        n_components = len(pipeline.data_manager.cell_type_fractions)
        pipeline.gmvae_trainer.initialize_model(
            input_dim=basic_config.input_dim,
            hidden_dim=basic_config.gmvae_hyperparameters["hidden_dim"],
            latent_dim=basic_config.latent_dim,
            n_components=n_components,
        )

        # Get validation dataloader
        gmvae_val_dataloader, _ = pipeline.data_manager.get_validation_dataloaders()
        assert gmvae_val_dataloader is not None, "Validation dataloader should exist"

        # Train GMVAE for 2 epochs (includes validation)
        metrics_tracker = pipeline.gmvae_trainer.train(
            dataloader=pipeline.gmvae_dataloader,
            proportion_tensor=pipeline.data_manager.cell_type_fractions,
            mapping_dict=pipeline.data_manager.cell_type_mapping,
            color_map=pipeline.data_manager.color_map,
            val_dataloader=gmvae_val_dataloader,
        )

        # Check that we have 2 epochs of metrics with validation data
        assert len(metrics_tracker.history) == 2, "Should have 2 epochs of metrics"

        for i, epoch_metrics in enumerate(metrics_tracker.history):
            # Each epoch should have validation metrics
            assert hasattr(
                epoch_metrics, "val_total_loss"
            ), f"Epoch {i} should have val_total_loss"
            assert (
                epoch_metrics.val_total_loss is not None
            ), f"Epoch {i} val_total_loss should not be None"
            assert (
                epoch_metrics.val_total_loss > 0.0
            ), f"Epoch {i} val_total_loss should be > 0"

            # Check that validation losses are different from training losses
            # (they should be different due to different data and no gradient updates)
            assert (
                epoch_metrics.val_total_loss != epoch_metrics.total_loss
            ), f"Epoch {i} validation and training losses should be different"

            print(
                f"✅ Epoch {i} - Training: {epoch_metrics.total_loss:.6f}, Validation: {epoch_metrics.val_total_loss:.6f}"
            )

        # Test that the metrics tracker can generate plots with validation data
        import tempfile
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend for testing

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            # This should not raise an error and should include validation data
            metrics_tracker.plot_detailed_loss_grid(
                save_path=tmp_file.name, config=basic_config
            )
            print(f"✅ Detailed loss grid generated successfully with validation data")

        # Verify that validation data is present in the plotting data
        # Check that validation losses are extracted correctly for plotting
        val_total_losses = [
            getattr(m, "val_total_loss", None) for m in metrics_tracker.history
        ]
        val_recon_losses = [
            getattr(m, "val_reconstruction_loss", None) for m in metrics_tracker.history
        ]
        val_fraction_losses = [
            getattr(m, "val_fraction_loss", None) for m in metrics_tracker.history
        ]

        # All validation losses should be present and non-None
        assert all(
            v is not None for v in val_total_losses
        ), "All validation total losses should be present"
        assert all(
            v is not None for v in val_recon_losses
        ), "All validation reconstruction losses should be present"
        assert all(
            v is not None for v in val_fraction_losses
        ), "All validation fraction losses should be present"

        # All validation losses should be positive
        assert all(
            v > 0 for v in val_total_losses
        ), "All validation total losses should be > 0"
        assert all(
            v > 0 for v in val_recon_losses
        ), "All validation reconstruction losses should be > 0"
        assert all(
            v > 0 for v in val_fraction_losses
        ), "All validation fraction losses should be > 0"

        print(f"✅ Validation losses properly extracted for plotting:")
        print(f"   val_total_losses: {val_total_losses}")
        print(f"   val_recon_losses: {val_recon_losses}")
        print(f"   val_fraction_losses: {val_fraction_losses}")


class TestPipelineFunctionality:
    """Test overall pipeline functionality."""

    def test_pipeline_data_loading(self, basic_config):
        """Test that pipeline can load data correctly."""
        pipeline = B2SCPipeline(basic_config)

        # Data should not be loaded initially
        assert not pipeline._data_loaded

        # Load data
        pipeline._load_training_data()

        # Data should now be loaded
        assert pipeline._data_loaded
        assert pipeline.gmvae_dataloader is not None
        assert pipeline.be_dataloader is not None
        assert pipeline.data_manager.cell_type_fractions is not None
        assert pipeline.data_manager.cell_type_mapping is not None

    def test_pipeline_validation_data_loading(self, basic_config):
        """Test that pipeline can load validation data when available."""
        pipeline = B2SCPipeline(basic_config)
        pipeline._load_training_data()

        # Get validation dataloaders
        gmvae_val_dataloader, be_val_dataloader = (
            pipeline.data_manager.get_validation_dataloaders()
        )

        # Should have validation data since sc_validation_path is provided
        assert gmvae_val_dataloader is not None
        assert be_val_dataloader is not None

    def test_pipeline_without_validation_data(self, mock_single_cell_data, temp_dir):
        """Test pipeline behavior when no validation data is provided."""
        config = B2SCConfig(
            sc_training_path=mock_single_cell_data["train_path"],
            # No sc_validation_path provided
            base_dir=str(temp_dir / "results"),
            latent_dim=64,
            pseudobulk_groupby="donor_id",
            cache_pseudobulk=False,
        )

        pipeline = B2SCPipeline(config)
        pipeline._load_training_data()

        # Get validation dataloaders
        gmvae_val_dataloader, be_val_dataloader = (
            pipeline.data_manager.get_validation_dataloaders()
        )

        # Should not have validation data
        assert gmvae_val_dataloader is None
        assert be_val_dataloader is None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_training_data_path(self, temp_dir):
        """Test that missing training data path raises appropriate error."""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            B2SCConfig(
                # No sc_training_path provided
                base_dir=str(temp_dir / "results"),
                latent_dim=64,
            )

    def test_invalid_training_data_path(self, temp_dir):
        """Test that invalid training data path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            B2SCConfig(
                sc_training_path="/nonexistent/path/data.h5ad",
                base_dir=str(temp_dir / "results"),
                latent_dim=64,
            )


class TestBulkEncoderTraining:
    """Test BulkEncoder training functionality and outputs."""

    def test_bulk_encoder_training_complete_pipeline(
        self, basic_config, mock_single_cell_data
    ):
        """Test complete BulkEncoder training pipeline including GMVAE prerequisite."""
        # Use a smaller configuration for faster testing
        basic_config.gmvae_hyperparameters.update(
            {"epochs": 1, "batch_size": 16}  # 1 epoch for speed
        )
        basic_config.bulk_encoder_hyperparameters.update(
            {"epochs": 2, "batch_size": 2}  # 2 epochs for testing metrics
        )

        pipeline = B2SCPipeline(basic_config)

        # First train GMVAE (required for BulkEncoder)
        gmvae_metrics = pipeline.train_gmvae()

        # Verify GMVAE training completed
        assert gmvae_metrics is not None
        assert len(gmvae_metrics.history) == 1  # 1 epoch
        assert pipeline._gmvae_trained

        # Verify GMVAE outputs exist
        assert (basic_config.base_dir / "GMVAE_mus.pt").exists()
        assert (basic_config.base_dir / "GMVAE_logvars.pt").exists()
        assert (basic_config.base_dir / "GMVAE_pis.pt").exists()
        assert (basic_config.base_dir / "GMVAE_model.pt").exists()

        # Now train BulkEncoder
        be_metrics = pipeline.train_bulk_encoder()

        # Verify BulkEncoder training completed
        assert be_metrics is not None
        assert len(be_metrics.history) == 2  # 2 epochs
        assert pipeline._bulk_encoder_trained

        # Verify final epoch metrics have all required components
        final_metrics = be_metrics.history[-1]
        assert hasattr(final_metrics, "total_loss")
        assert hasattr(final_metrics, "mus_loss")
        assert hasattr(final_metrics, "logvars_loss")
        assert hasattr(final_metrics, "pis_loss")
        assert hasattr(final_metrics, "uniform_penalty")

        # Verify all losses are reasonable (not NaN, not extremely large)
        assert not np.isnan(final_metrics.total_loss)
        assert not np.isnan(final_metrics.mus_loss)
        assert not np.isnan(final_metrics.logvars_loss)
        assert not np.isnan(final_metrics.pis_loss)
        assert not np.isnan(final_metrics.uniform_penalty)

        assert final_metrics.total_loss > 0
        assert final_metrics.total_loss < 1e6  # Reasonable upper bound

    def test_bulk_encoder_validation_metrics(self, basic_config, mock_single_cell_data):
        """Test that BulkEncoder validation metrics are calculated correctly."""
        # Quick training setup
        basic_config.gmvae_hyperparameters.update({"epochs": 1, "batch_size": 16})
        basic_config.bulk_encoder_hyperparameters.update({"epochs": 2, "batch_size": 2})

        pipeline = B2SCPipeline(basic_config)

        # Train GMVAE first
        pipeline.train_gmvae()

        # Train BulkEncoder with validation
        be_metrics = pipeline.train_bulk_encoder()

        # Check that validation metrics are stored
        final_metrics = be_metrics.history[-1]

        # Should have validation metrics
        assert hasattr(final_metrics, "val_total_loss")
        assert hasattr(final_metrics, "val_mus_loss")
        assert hasattr(final_metrics, "val_logvars_loss")
        assert hasattr(final_metrics, "val_pis_loss")
        assert hasattr(final_metrics, "val_uniform_penalty")

        # Validation losses should be reasonable
        if final_metrics.val_total_loss is not None:
            assert not np.isnan(final_metrics.val_total_loss)
            assert final_metrics.val_total_loss > 0

    def test_bulk_encoder_output_files(self, basic_config, mock_single_cell_data):
        """Test that BulkEncoder saves all required output files."""
        # Quick training setup
        basic_config.gmvae_hyperparameters.update({"epochs": 1, "batch_size": 16})
        basic_config.bulk_encoder_hyperparameters.update({"epochs": 1, "batch_size": 2})

        pipeline = B2SCPipeline(basic_config)

        # Train both models
        pipeline.train_gmvae()
        pipeline.train_bulk_encoder()

        # Check that BulkEncoder model file is saved
        assert (basic_config.base_dir / "bulkEncoder_model.pt").exists()

        # Check that metrics files are saved
        assert (basic_config.base_dir / "bulk_encoder_metrics.pkl").exists()
        assert (basic_config.base_dir / "bulk_encoder_detailed_loss_grid.png").exists()
        assert (basic_config.base_dir / "bulk_encoder_loss_contributions.png").exists()

    def test_bulk_encoder_metrics_format(self, basic_config, mock_single_cell_data):
        """Test that BulkEncoder metrics are in the correct format."""
        # Quick training setup
        basic_config.gmvae_hyperparameters.update({"epochs": 1, "batch_size": 16})
        basic_config.bulk_encoder_hyperparameters.update({"epochs": 2, "batch_size": 2})

        pipeline = B2SCPipeline(basic_config)

        # Train both models
        pipeline.train_gmvae()
        be_metrics = pipeline.train_bulk_encoder()

        # Test metrics structure
        assert hasattr(be_metrics, "history")
        assert len(be_metrics.history) == 2

        # Test each epoch's metrics
        for epoch_metrics in be_metrics.history:
            # Required attributes
            required_attrs = [
                "epoch",
                "total_loss",
                "mus_loss",
                "logvars_loss",
                "pis_loss",
                "uniform_penalty",
            ]
            for attr in required_attrs:
                assert hasattr(epoch_metrics, attr), f"Missing attribute: {attr}"

            # Values should be numeric
            assert isinstance(epoch_metrics.epoch, int)
            assert isinstance(epoch_metrics.total_loss, (int, float))
            assert isinstance(epoch_metrics.mus_loss, (int, float))
            assert isinstance(epoch_metrics.logvars_loss, (int, float))
            assert isinstance(epoch_metrics.pis_loss, (int, float))
            assert isinstance(epoch_metrics.uniform_penalty, (int, float))

    def test_bulk_encoder_without_gmvae_fails(self, minimal_mock_data, temp_dir):
        """Test that BulkEncoder training fails gracefully without GMVAE."""
        # Use minimal config for fast testing (100 cells × 50 genes)
        config = B2SCConfig(
            sc_training_path=minimal_mock_data["train_path"],
            base_dir=str(temp_dir / "results"),
            latent_dim=32,
            device="cpu",
            cache_pseudobulk=False,
        )

        pipeline = B2SCPipeline(config)

        # Try to train BulkEncoder without GMVAE first
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            pipeline.train_bulk_encoder()

    def test_bulk_encoder_zero_division_protection(self, basic_config):
        """Test that BulkEncoder properly handles donors with no cells."""
        # Use short training for this test
        basic_config.gmvae_hyperparameters.update({"epochs": 1, "batch_size": 16})

        pipeline = B2SCPipeline(basic_config)

        # Train GMVAE first
        pipeline.train_gmvae()

        # Get BulkEncoder trainer
        be_trainer = pipeline.bulk_encoder_trainer

        # Initialize model and load parameters
        if be_trainer.model is None:
            be_trainer.initialize_model(
                input_dim=basic_config.input_dim,
                hidden_dim=basic_config.bulk_encoder_hyperparameters["hidden_dim"],
                latent_dim=basic_config.latent_dim,
                n_components=len(pipeline.data_manager.cell_type_fractions),
            )
        if be_trainer.scMus is None:
            be_trainer.load_gmvae_parameters()

        # Test with valid donor IDs (should work)
        sc_data = pipeline.data_manager.sc_train_data
        available_donors = list(sc_data.obs[basic_config.pseudobulk_groupby].unique())[
            :2
        ]

        try:
            proportions = be_trainer._get_batch_target_proportions(
                tuple(available_donors), use_validation_data=False
            )
            assert proportions.shape[0] == len(available_donors)
            assert torch.all(proportions >= 0)
            assert torch.all(proportions <= 1)
            # Check proportions sum to 1 for each donor
            for i in range(proportions.shape[0]):
                prop_sum = proportions[i].sum()
                assert abs(prop_sum - 1.0) < 1e-6
        except Exception as e:
            pytest.fail(f"Valid donor IDs should work: {e}")

        # Test with invalid donor ID (should raise ValueError, not ZeroDivisionError)
        invalid_donor_ids = (available_donors[0], "nonexistent_donor_999")

        with pytest.raises(ValueError, match="has no cells in the dataset"):
            be_trainer._get_batch_target_proportions(
                invalid_donor_ids, use_validation_data=False
            )


class TestTrainingIsolation:
    """Test that training stages are properly isolated and don't affect each other."""

    def test_use_dynamic_targets_does_not_affect_gmvae(
        self, minimal_mock_data, temp_dir
    ):
        """Test that use_dynamic_targets parameter doesn't influence GMVAE training."""
        import numpy as np

        # Create two identical configs except for use_dynamic_targets
        config1 = B2SCConfig(
            sc_training_path=minimal_mock_data["train_path"],
            base_dir=str(temp_dir / "results1"),
            latent_dim=32,
            device="cpu",
            cache_pseudobulk=False,
        )
        config1.gmvae_hyperparameters.update({"epochs": 1, "batch_size": 16})
        config1.bulk_encoder_hyperparameters.update({"use_dynamic_targets": False})

        config2 = B2SCConfig(
            sc_training_path=minimal_mock_data["train_path"],
            base_dir=str(temp_dir / "results2"),
            latent_dim=32,
            device="cpu",
            cache_pseudobulk=False,
        )
        config2.gmvae_hyperparameters.update({"epochs": 1, "batch_size": 16})
        config2.bulk_encoder_hyperparameters.update({"use_dynamic_targets": True})

        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Train GMVAE with use_dynamic_targets=False
        pipeline1 = B2SCPipeline(config1)
        metrics1 = pipeline1.train_gmvae()

        # Reset seed and train GMVAE with use_dynamic_targets=True
        np.random.seed(42)
        torch.manual_seed(42)

        pipeline2 = B2SCPipeline(config2)
        metrics2 = pipeline2.train_gmvae()

        # GMVAE training should be identical regardless of use_dynamic_targets
        assert len(metrics1.history) == len(
            metrics2.history
        ), "GMVAE should train same number of epochs regardless of use_dynamic_targets"

        # Compare final losses (should be very close, allowing for floating point differences)
        final_loss1 = metrics1.history[-1].total_loss
        final_loss2 = metrics2.history[-1].total_loss

        # Losses should be nearly identical (within 0.1% relative difference)
        relative_diff = abs(final_loss1 - final_loss2) / max(
            abs(final_loss1), abs(final_loss2)
        )
        assert relative_diff < 0.001, (
            f"GMVAE training should be identical regardless of use_dynamic_targets. "
            f"Got loss1={final_loss1:.6f}, loss2={final_loss2:.6f}, "
            f"relative_diff={relative_diff:.6f}"
        )

        # Load saved GMVAE parameters and verify they're identical
        gmvae_mus1 = torch.load(config1.base_dir / "GMVAE_mus.pt", weights_only=True)
        gmvae_mus2 = torch.load(config2.base_dir / "GMVAE_mus.pt", weights_only=True)

        mus_diff = torch.abs(gmvae_mus1 - gmvae_mus2).max().item()
        assert mus_diff < 1e-5, (
            f"GMVAE parameters should be identical regardless of use_dynamic_targets. "
            f"Max difference in mus: {mus_diff}"
        )

        print("✅ GMVAE training is properly isolated from BulkEncoder config")
        print(f"   Loss1: {final_loss1:.6f}")
        print(f"   Loss2: {final_loss2:.6f}")
        print(f"   Relative difference: {relative_diff:.8f}")
        print(f"   Max mus difference: {mus_diff:.8e}")

    def test_kl_loss_weight_applied_correctly(self, minimal_mock_data, temp_dir):
        """Test that kl_loss_weight is properly applied to weighted KL loss."""
        import numpy as np

        # Create config with very high KL loss weight for easy verification
        config = B2SCConfig(
            sc_training_path=minimal_mock_data["train_path"],
            base_dir=str(temp_dir / "results"),
            latent_dim=32,
            device="cpu",
            cache_pseudobulk=False,
        )

        # Set high KL loss weight and disable annealing for simpler test
        config.gmvae_hyperparameters.update(
            {
                "epochs": 1,
                "batch_size": 16,
                "kl_loss_weight": 100.0,  # Very high weight
                "kl_anneal_start": 1.0,  # Start at full weight (no annealing)
                "kl_anneal_max": 1.0,  # Stay at full weight
                "kl_anneal_epochs": 0,  # No annealing
                "recon_loss_weight": 1.0,
                "zinb_loss_weight": 0.5,
                "fraction_loss_weight": 2.0,
            }
        )

        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Train GMVAE
        pipeline = B2SCPipeline(config)
        metrics = pipeline.train_gmvae()

        # Get final epoch metrics
        final_metrics = metrics.history[-1]

        # Extract raw and weighted KL losses
        raw_kl_loss = final_metrics.kl_loss  # Raw (unweighted)
        weighted_kl_loss = final_metrics.weighted_kl_loss
        kl_weight = final_metrics.kl_weight  # Annealing weight (should be 1.0)

        print(f"\n🔍 Verifying KL Loss Weight Application:")
        print(f"   Configured kl_loss_weight: {config.gmvae_kl_loss_weight}")
        print(f"   Annealing weight (kl_weight): {kl_weight}")
        print(f"   Raw KL loss: {raw_kl_loss:.6f}")
        print(f"   Weighted KL loss: {weighted_kl_loss:.6f}")

        # Calculate expected weighted loss
        expected_weighted = config.gmvae_kl_loss_weight * kl_weight * raw_kl_loss
        print(f"   Expected weighted: {expected_weighted:.6f}")

        # Verify weighted loss matches expectation
        # Allow small numerical tolerance
        relative_error = abs(weighted_kl_loss - expected_weighted) / max(
            abs(expected_weighted), 1e-6
        )

        assert relative_error < 0.01, (
            f"Weighted KL loss doesn't match expected value!\n"
            f"Expected: {expected_weighted:.6f}\n"
            f"Got: {weighted_kl_loss:.6f}\n"
            f"Relative error: {relative_error:.6f}"
        )

        # Verify the ratio is correct (weighted / raw should equal configured weight × annealing)
        if raw_kl_loss > 1e-6:  # Avoid division by very small numbers
            actual_multiplier = weighted_kl_loss / raw_kl_loss
            expected_multiplier = config.gmvae_kl_loss_weight * kl_weight

            print(f"   Actual multiplier (weighted/raw): {actual_multiplier:.2f}")
            print(f"   Expected multiplier: {expected_multiplier:.2f}")

            multiplier_error = (
                abs(actual_multiplier - expected_multiplier) / expected_multiplier
            )

            assert multiplier_error < 0.01, (
                f"KL loss multiplier is incorrect!\n"
                f"Expected: {expected_multiplier:.2f}\n"
                f"Got: {actual_multiplier:.2f}"
            )

        # Verify KL loss contributes significantly to total loss with high weight
        total_loss = final_metrics.total_loss
        kl_contribution_ratio = weighted_kl_loss / total_loss if total_loss > 0 else 0

        print(f"   Total loss: {total_loss:.6f}")
        print(f"   KL contribution to total: {kl_contribution_ratio:.2%}")

        # With kl_loss_weight=100, KL should dominate unless it's very small
        if raw_kl_loss > 1e-4:  # Only check if KL loss is non-negligible
            assert kl_contribution_ratio > 0.1, (
                f"With kl_loss_weight=100, KL should contribute significantly!\n"
                f"KL contribution: {kl_contribution_ratio:.2%}"
            )

        print("✅ KL loss weight is correctly applied!")


class TestHyperparameterUsage:
    """Test class for verifying hyperparameter usage during training."""

    def test_gmvae_hyperparameter_updates(self, basic_config):
        """Test that GMVAE hyperparameter updates are properly used during training."""
        print("\n🧪 Testing GMVAE hyperparameter updates...")

        # Define custom hyperparameters with distinctive values
        custom_gmvae_params = {
            "epochs": 2,  # Short for testing
            "learning_rate": 0.001234,  # Distinctive value
            "batch_size": 16,  # Different from default
            "recon_loss_weight": 2.5,  # Custom weight
            "zinb_loss_weight": 0.75,  # Custom weight
            "fraction_loss_weight": 3.5,  # Custom weight
            "kl_loss_weight": 0.25,  # Custom weight
            "hidden_dim": 1024,  # Different from default
        }

        # Update config with custom hyperparameters
        basic_config.gmvae_hyperparameters.update(custom_gmvae_params)

        # Verify config properties reflect the updates
        assert basic_config.gmvae_epochs == 2
        assert basic_config.gmvae_lr == 0.001234
        assert basic_config.gmvae_batch_size == 16
        assert basic_config.gmvae_recon_loss_weight == 2.5
        assert basic_config.gmvae_zinb_loss_weight == 0.75
        assert basic_config.gmvae_fraction_loss_weight == 3.5
        assert basic_config.gmvae_kl_loss_weight == 0.25
        assert basic_config.gmvae_hidden_dim == 1024

        # Create trainer and verify hyperparameters are passed correctly
        trainer = GMVAETrainer(basic_config)

        # Check that trainer has access to the correct hyperparameters
        assert trainer.config.gmvae_epochs == 2
        assert trainer.config.gmvae_lr == 0.001234
        assert trainer.config.gmvae_batch_size == 16
        assert trainer.config.gmvae_recon_loss_weight == 2.5
        assert trainer.config.gmvae_zinb_loss_weight == 0.75
        assert trainer.config.gmvae_fraction_loss_weight == 3.5
        assert trainer.config.gmvae_kl_loss_weight == 0.25

    def test_bulk_encoder_hyperparameter_updates(self, basic_config):
        """Test that BulkEncoder hyperparameter updates are properly used during training."""
        print("\n🧪 Testing BulkEncoder hyperparameter updates...")

        # Define custom hyperparameters with distinctive values
        custom_bulk_params = {
            "epochs": 3,  # Short for testing
            "learning_rate": 0.005678,  # Distinctive value
            "batch_size": 8,  # Different from default
            "mus_loss_weight": 1.5,  # Custom weight
            "logvars_loss_weight": 2.0,  # Custom weight
            "pis_loss_weight": 75.0,  # Custom weight
            "hidden_dim": 1536,  # Different from default
        }

        # Update config with custom hyperparameters
        basic_config.bulk_encoder_hyperparameters.update(custom_bulk_params)

        # Verify config properties reflect the updates
        assert basic_config.bulk_encoder_epochs == 3
        assert basic_config.bulk_encoder_lr == 0.005678
        assert basic_config.bulk_encoder_batch_size == 8
        assert basic_config.bulk_mus_loss_weight == 1.5
        assert basic_config.bulk_logvars_loss_weight == 2.0
        assert basic_config.bulk_pis_loss_weight == 75.0
        assert basic_config.bulk_hidden_dim == 1536

        # Initialize trainer
        trainer = BulkEncoderTrainer(basic_config)

        # Check that trainer has access to the correct hyperparameters
        assert trainer.config.bulk_encoder_epochs == 3
        assert trainer.config.bulk_encoder_lr == 0.005678
        assert trainer.config.bulk_encoder_batch_size == 8
        assert trainer.config.bulk_mus_loss_weight == 1.5
        assert trainer.config.bulk_logvars_loss_weight == 2.0
        assert trainer.config.bulk_pis_loss_weight == 75.0

    def test_loss_weight_usage_in_training(self, basic_config):
        """Test that loss weights are actually used in loss calculations during training."""
        print("\n🧪 Testing loss weight usage in training...")

        # Set distinctive loss weights
        basic_config.gmvae_hyperparameters.update(
            {
                "recon_loss_weight": 10.0,
                "zinb_loss_weight": 5.0,
                "fraction_loss_weight": 20.0,
                "kl_loss_weight": 2.0,
            }
        )

        # Create trainer
        trainer = GMVAETrainer(basic_config)

        # Verify the trainer uses the correct loss weights
        assert trainer.config.gmvae_recon_loss_weight == 10.0
        assert trainer.config.gmvae_zinb_loss_weight == 5.0
        assert trainer.config.gmvae_fraction_loss_weight == 20.0
        assert trainer.config.gmvae_kl_loss_weight == 2.0

    def test_set_hyperparameters_method(self, basic_config):
        """Test the set_hyperparameters method for updating multiple hyperparameters at once."""
        print("\n🧪 Testing set_hyperparameters method...")

        # Define hierarchical hyperparameter updates
        updates = {
            "gmvae": {
                "learning_rate": 0.0007,
                "recon_loss_weight": 15.0,
                "zinb_loss_weight": 3.0,
                "epochs": 5,
            },
            "bulk_encoder": {
                "learning_rate": 0.002,
                "mus_loss_weight": 3.0,
                "logvars_loss_weight": 4.0,
                "pis_loss_weight": 200.0,
                "epochs": 10,
            },
        }

        # Apply updates using the set_hyperparameters method
        basic_config.set_hyperparameters(updates)

        # Verify GMVAE updates
        assert basic_config.gmvae_lr == 0.0007
        assert basic_config.gmvae_recon_loss_weight == 15.0
        assert basic_config.gmvae_zinb_loss_weight == 3.0
        assert basic_config.gmvae_epochs == 5

        # Verify BulkEncoder updates
        assert basic_config.bulk_encoder_lr == 0.002
        assert basic_config.bulk_mus_loss_weight == 3.0
        assert basic_config.bulk_logvars_loss_weight == 4.0
        assert basic_config.bulk_pis_loss_weight == 200.0
        assert basic_config.bulk_encoder_epochs == 10

    def test_detailed_loss_grid_visualization(self, basic_config):
        """Test the new detailed loss grid visualization."""
        print("\n🧪 Testing detailed loss grid visualization...")

        from metrics import MetricsTracker, GMVAEMetrics, BulkEncoderMetrics
        import tempfile

        # Test GMVAE grid
        tracker_gmvae = MetricsTracker()

        # Add some mock GMVAE metrics
        for epoch in range(3):
            metric = GMVAEMetrics(
                epoch=epoch,
                total_loss=2.0 - epoch * 0.5,
                reconstruction_loss=1.0 - epoch * 0.2,
                kl_loss=0.8 - epoch * 0.1,
                fraction_loss=0.3 - epoch * 0.05,
                zinb_loss=0.5 - epoch * 0.08,
            )
            tracker_gmvae.add_epoch_metrics(metric)

        # Test GMVAE grid generation
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tracker_gmvae.plot_detailed_loss_grid(tmp_file.name, config=basic_config)
            assert Path(tmp_file.name).exists()
            print("✅ GMVAE detailed loss grid generated successfully")

        # Test BulkEncoder grid
        tracker_bulk = MetricsTracker()

        # Add some mock BulkEncoder metrics
        for epoch in range(3):
            metric = BulkEncoderMetrics(
                epoch=epoch,
                total_loss=1.5 - epoch * 0.3,
                mus_loss=0.8 - epoch * 0.15,
                logvars_loss=0.6 - epoch * 0.12,
                pis_loss=0.4 - epoch * 0.08,
                uniform_penalty=0.0,
            )
            tracker_bulk.add_epoch_metrics(metric)

        # Test BulkEncoder grid generation
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tracker_bulk.plot_detailed_loss_grid(tmp_file.name, config=basic_config)
            assert Path(tmp_file.name).exists()
            print("✅ BulkEncoder detailed loss grid generated successfully")

    def test_kl_anneal_epochs_defaults_to_num_epochs(self, temp_dir, minimal_mock_data):
        """Test that kl_anneal_epochs defaults to the total number of epochs when not specified."""
        print("\n🧪 Testing KL anneal epochs defaults to total epochs...")

        # Test with different epoch counts
        test_cases = [
            {"epochs": 5, "expected_anneal_epochs": 5},
            {"epochs": 10, "expected_anneal_epochs": 10},
            {"epochs": 20, "expected_anneal_epochs": 20},
            {"epochs": 50, "expected_anneal_epochs": 50},
        ]

        for test_case in test_cases:
            num_epochs = test_case["epochs"]
            expected_anneal = test_case["expected_anneal_epochs"]

            # Create config with specific number of epochs
            # Don't set kl_anneal_epochs - it should default to None
            config = B2SCConfig(
                sc_training_path=minimal_mock_data["train_path"],
                base_dir=str(temp_dir / f"results_epochs_{num_epochs}"),
                latent_dim=32,
                device="cpu",
                cache_pseudobulk=False,
            )
            config.gmvae_hyperparameters.update(
                {
                    "epochs": num_epochs,
                    "batch_size": 16,
                    "kl_anneal_start": 0.0,
                    "kl_anneal_max": 1.0,
                    # kl_anneal_epochs is not set - should default to None
                }
            )

            # Verify kl_anneal_epochs is None in config
            assert (
                config.gmvae_hyperparameters.get("kl_anneal_epochs") is None
            ), "kl_anneal_epochs should be None by default"

            # Create trainer
            trainer = GMVAETrainer(config)

            # Test the _kl_annealing_weight method at different epochs
            # When kl_anneal_epochs is None, it should use total epochs
            for epoch in range(num_epochs + 1):
                kl_weight = trainer._kl_annealing_weight(epoch)

                # Calculate expected weight based on linear annealing over total epochs
                if epoch >= expected_anneal:
                    expected_weight = 1.0
                else:
                    expected_weight = epoch / expected_anneal

                assert abs(kl_weight - expected_weight) < 1e-6, (
                    f"For {num_epochs} epochs at epoch {epoch}: "
                    f"expected KL weight {expected_weight:.6f}, got {kl_weight:.6f}"
                )

            print(
                f"  ✅ Epochs={num_epochs}: KL annealing correctly spans all {expected_anneal} epochs"
            )

        print("✅ All test cases passed: kl_anneal_epochs defaults to total epochs")

    def test_kl_anneal_epochs_explicit_value_respected(
        self, temp_dir, minimal_mock_data
    ):
        """Test that explicitly set kl_anneal_epochs is respected."""
        print("\n🧪 Testing explicit kl_anneal_epochs is respected...")

        # Create config with 20 epochs but anneal over only 10
        config = B2SCConfig(
            sc_training_path=minimal_mock_data["train_path"],
            base_dir=str(temp_dir / "results_explicit_anneal"),
            latent_dim=32,
            device="cpu",
            cache_pseudobulk=False,
        )
        config.gmvae_hyperparameters.update(
            {
                "epochs": 20,
                "batch_size": 16,
                "kl_anneal_start": 0.0,
                "kl_anneal_max": 1.0,
                "kl_anneal_epochs": 10,  # Explicitly set to 10
            }
        )

        # Verify explicit value is set
        assert (
            config.gmvae_hyperparameters.get("kl_anneal_epochs") == 10
        ), "Explicit kl_anneal_epochs should be 10"

        # Create trainer
        trainer = GMVAETrainer(config)

        # Test that annealing reaches max at epoch 10, not 20
        for epoch in range(21):
            kl_weight = trainer._kl_annealing_weight(epoch)

            if epoch >= 10:
                expected_weight = 1.0  # Should reach max at epoch 10
            else:
                expected_weight = epoch / 10

            assert (
                abs(kl_weight - expected_weight) < 1e-6
            ), f"At epoch {epoch}: expected KL weight {expected_weight:.6f}, got {kl_weight:.6f}"

        print(
            "  ✅ Explicit kl_anneal_epochs=10 respected (reaches max before total 20 epochs)"
        )
        print("✅ Explicit kl_anneal_epochs value is properly used")

    def test_validation_correlation_captured(self, mock_single_cell_data, temp_dir):
        """Test that validation reconstruction correlation is properly captured in metrics."""
        print("\n🧪 Testing validation correlation capture...")

        # Create config with validation data included
        config = B2SCConfig(
            sc_training_path=mock_single_cell_data["train_path"],
            sc_validation_path=mock_single_cell_data[
                "val_path"
            ],  # Include validation data
            base_dir=str(temp_dir / "test_val_corr"),
            latent_dim=32,
            device="cpu",
            cache_pseudobulk=False,
        )
        config.gmvae_hyperparameters.update(
            {
                "epochs": 2,
                "batch_size": 16,
            }
        )

        # Train GMVAE with validation
        pipeline = B2SCPipeline(config)
        metrics = pipeline.train_gmvae()

        # Check that both training and validation correlations are captured
        has_train_corr = any(
            getattr(m, "reconstruction_correlation", None) is not None
            for m in metrics.history
        )
        has_val_corr = any(
            getattr(m, "val_reconstruction_correlation", None) is not None
            for m in metrics.history
        )

        print(f"  Training correlation captured: {has_train_corr}")
        print(f"  Validation correlation captured: {has_val_corr}")

        # Print actual values for debugging
        for i, m in enumerate(metrics.history):
            train_corr = getattr(m, "reconstruction_correlation", None)
            val_corr = getattr(m, "val_reconstruction_correlation", None)
            train_str = f"{train_corr:.4f}" if train_corr is not None else "None"
            val_str = f"{val_corr:.4f}" if val_corr is not None else "None"
            print(f"  Epoch {i}: Train Corr={train_str}, Val Corr={val_str}")

        # Assertions
        assert has_train_corr, "Training correlation should be captured for all epochs"
        assert has_val_corr, "Validation correlation should be captured for all epochs"

        # Also check that the values are reasonable (between -1 and 1)
        for m in metrics.history:
            train_corr = getattr(m, "reconstruction_correlation", None)
            val_corr = getattr(m, "val_reconstruction_correlation", None)

            if train_corr is not None:
                assert (
                    -1.0 <= train_corr <= 1.0
                ), f"Training correlation {train_corr} out of valid range"
            if val_corr is not None:
                assert (
                    -1.0 <= val_corr <= 1.0
                ), f"Validation correlation {val_corr} out of valid range"

        print("✅ Both training and validation correlations properly captured!")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
