"""
Example script demonstrating how to use the B2SC pipeline.

This script shows both simple and advanced usage patterns for the new OOP B2SC implementation.
"""

import os
import sys
from pathlib import Path

# Add B2SC to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Clean import style as shown in architectural plan
try:
    from B2SC import B2SCConfig, B2SCPipeline
except ImportError:
    # Fallback for development
    from config import B2SCConfig
    from pipeline import B2SCPipeline


def simple_pipeline_example():
    """Simple example: run the complete pipeline with default settings."""
    print("=" * 60)
    print("SIMPLE PIPELINE EXAMPLE")
    print("=" * 60)

    # Configure pipeline
    config = B2SCConfig(
        sc_training_data_path="/home/shared-ssh-key/convergeSC-Internal/bulk2single/data/mouse_liver_sc_healthy_5K_highly_variable.h5ad",
        bulk_data_for_deconv_path="/home/shared-ssh-key/convergeSC-Internal/bulk2single/data/mouse_liver_sc_healthy_5K_highly_variable_pseudobulk.h5ad",
        base_dir=Path("./results_simple"),
        gmvae_epochs=30,
        bulk_encoder_epochs=100,
    )

    # Run complete pipeline
    pipeline = B2SCPipeline(config)
    gmvae_metrics, be_metrics, results = pipeline.run_full_pipeline()

    # Access results
    print(f"Generated {len(results.generated_cells)} samples")
    print(f"Total cells generated: {results.generation_stats['total_cells_generated']}")

    # Save results
    results.save_results(config.base_dir / "generation_results")

    print("Simple pipeline completed successfully!")


def advanced_pipeline_example():
    """Advanced example: step-by-step training with custom monitoring."""
    print("=" * 60)
    print("ADVANCED PIPELINE EXAMPLE")
    print("=" * 60)

    # Configure pipeline with custom settings
    config = B2SCConfig(
        sc_training_data_path="/home/shared-ssh-key/convergeSC-Internal/bulk2single/data/mouse_liver_sc_healthy_5K_highly_variable.h5ad",
        bulk_data_for_deconv_path="/home/shared-ssh-key/convergeSC-Internal/bulk2single/data/mouse_liver_sc_healthy_5K_highly_variable_pseudobulk.h5ad",
        base_dir=Path("./results_advanced"),
        gmvae_epochs=50,  # More epochs for better training
        bulk_encoder_epochs=150,
        gmvae_lr=5e-4,
        bulk_encoder_lr=1e-3,
        hidden_dim=1024,  # Smaller hidden dimension
        latent_dim=512,
    )

    # Initialize pipeline
    pipeline = B2SCPipeline(config)

    # Check pipeline status
    status = pipeline.get_pipeline_status()
    print(f"Initial pipeline status: {status}")

    # Step 1: Train GMVAE with custom monitoring
    print("\n--- Training GMVAE ---")
    gmvae_metrics = pipeline.train_gmvae()

    # Analyze GMVAE training
    best_epoch = gmvae_metrics.get_best_epoch()
    print(f"Best GMVAE epoch: {best_epoch}")

    # Generate detailed training visualizations
    gmvae_metrics.plot_detailed_loss_grid(
        config.base_dir / "gmvae_detailed_loss_grid.png", config=config
    )

    # Step 2: Train BulkEncoder
    print("\n--- Training BulkEncoder ---")
    be_metrics = pipeline.train_bulk_encoder()

    # Analyze BulkEncoder training
    best_epoch = be_metrics.get_best_epoch()
    print(f"Best BulkEncoder epoch: {best_epoch}")

    # Step 3: Generate cells with custom parameters
    print("\n--- Generating Cells ---")
    results = pipeline.generate_cells(
        sample_ids=["3-M-8/9"], num_cells=1000  # Generate only for specific samples
    )

    # Step 4: Validate generation quality
    print("\n--- Validating Results ---")
    quality_metrics = pipeline.validate_generation_quality(results)

    # Print detailed quality metrics
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value}")

    # Step 5: Save results with custom analysis
    results.save_results(config.base_dir / "generation_results")

    # Get detailed statistics
    summary_stats = results.get_summary_statistics()
    print(f"\nGeneration Summary:")
    print(f"  Samples: {summary_stats['num_samples']}")
    print(f"  Total cells: {summary_stats['total_cells_generated']}")
    print(f"  Cell types found: {len(summary_stats['overall_cell_type_counts'])}")

    print("Advanced pipeline completed successfully!")


def configuration_examples():
    """Show different configuration options."""
    print("=" * 60)
    print("CONFIGURATION EXAMPLES")
    print("=" * 60)

    # Example 1: Minimal configuration
    print("1. Minimal Configuration:")
    minimal_config = B2SCConfig(sc_training_data_path="path/to/single_cell_data.h5ad")
    print(f"   Input dim: {minimal_config.input_dim} (auto-detected)")
    print(f"   Batch size: {minimal_config.batch_size} (auto-calculated)")
    print(f"   Device: {minimal_config.device}")

    # Example 2: GPU configuration
    print("\n2. GPU Configuration:")
    gpu_config = B2SCConfig(
        sc_training_data_path="path/to/single_cell_data.h5ad",
        device="cuda",
        batch_size=128,
    )
    print(f"   Device: {gpu_config.device}")
    print(f"   Batch size: {gpu_config.batch_size}")

    # Example 3: Research configuration
    print("\n3. Research Configuration:")
    research_config = B2SCConfig(
        sc_training_data_path="path/to/single_cell_data.h5ad",
        bulk_data_for_deconv_path="path/to/bulk_data.h5ad",
        base_dir=Path("./research_results"),
        gmvae_epochs=100,
        bulk_encoder_epochs=200,
        gmvae_lr=1e-4,
        bulk_encoder_lr=1e-3,
        hidden_dim=4096,
        latent_dim=2048,
        normalize_data=True,
    )
    print(
        f"   Architecture: {research_config.input_dim} -> {research_config.hidden_dim} -> {research_config.latent_dim}"
    )
    print(
        f"   Training: GMVAE={research_config.gmvae_epochs}, BE={research_config.bulk_encoder_epochs}"
    )
    print(f"   Generation: 500 cells per sample (default)")

    print("\nConfiguration examples completed!")


def main():
    """Main function to run examples."""
    print("B2SC Pipeline Examples")
    print("=" * 60)

    # Change to appropriate directory
    os.chdir("/home/converge-bio/B2SC")

    # Show configuration examples
    configuration_examples()

    # Note: Uncomment the examples you want to run
    # WARNING: These examples require actual data files and will take time to run

    print("\n" + "=" * 60)
    print("NOTE: To run the actual pipeline examples, uncomment the lines below")
    print("and ensure you have the required data files available.")
    print("=" * 60)

    # Uncomment to run simple example
    # simple_pipeline_example()

    # Uncomment to run advanced example
    # advanced_pipeline_example()


if __name__ == "__main__":
    main()
