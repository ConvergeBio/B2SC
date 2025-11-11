#!/usr/bin/env python3
"""
BulkEncoder Hyperparameter Tuning with W&B Sweeps (Stage 2)

Tunes BulkEncoder hyperparameters using a pre-trained GMVAE from Stage 1.

The pre-trained GMVAE is loaded (NOT trained), then only the BulkEncoder
is trained with different hyperparameter configurations.

Usage:
  1. Run Stage 1 (gmvae_sweep.py) to get best GMVAE
  2. Update bulk_encoder_sweep_config.yaml with:
     - pretrained_gmvae_dir: path to best GMVAE run directory
     - gmvae_*: hyperparameters from best GMVAE run
  3. Run this script

Evaluation Metric: val_pis_loss (raw, unweighted MSE)
- Measures cell type proportion prediction accuracy
- Independent of tunable loss weights (unbiased)
"""

import argparse
import yaml
import wandb
from pathlib import Path
import os
from dotenv import load_dotenv

# Add B2SC to path
import sys

b2sc_path = Path(__file__).parent.parent
if str(b2sc_path) not in sys.path:
    sys.path.insert(0, str(b2sc_path))

from config import B2SCConfig
from pipeline import B2SCPipeline


# Load environment variables
load_dotenv()


def train_bulk_encoder():
    """Training function called by wandb agent."""
    wandb.init()
    config = wandb.config

    # Extract data configuration
    data_config = {
        "sc_training_path": config.sc_training_path,
        "sc_validation_path": config.sc_validation_path,
        "base_dir": config.base_dir,
        "pretrained_gmvae_dir": config.pretrained_gmvae_dir,  # Path to Stage 1 best model
    }

    # Create B2SC configuration
    run_name = wandb.run.name or f"run_{wandb.run.id}"
    base_dir = Path(data_config["base_dir"]) / "bulk_encoder_sweep" / run_name

    b2sc_config = B2SCConfig(
        sc_training_path=data_config["sc_training_path"],
        sc_validation_path=data_config["sc_validation_path"],
        base_dir=base_dir,
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        # BulkEncoder hyperparameters (TUNED in this sweep)
        bulk_encoder_hyperparameters={
            "hidden_dim": config.bulk_encoder_hidden_dim,
            "epochs": config.bulk_encoder_epochs,
            "learning_rate": config.bulk_encoder_learning_rate,
            "batch_size": config.bulk_encoder_batch_size,
            "mus_loss_weight": config.bulk_encoder_mus_loss_weight,
            "logvars_loss_weight": config.bulk_encoder_logvars_loss_weight,
            "pis_loss_weight": config.bulk_encoder_pis_loss_weight,
            "uniform_penalty_weight": config.bulk_encoder_uniform_penalty_weight,
        },
        device="auto",
    )

    # Initialize pipeline
    pipeline = B2SCPipeline(b2sc_config)

    # Load pre-trained GMVAE from Stage 1 (DO NOT TRAIN)
    print("\n" + "=" * 80)
    print("Loading Pre-trained GMVAE from Stage 1")
    print("=" * 80)

    pretrained_dir = Path(data_config["pretrained_gmvae_dir"])

    if not pretrained_dir.exists():
        raise FileNotFoundError(
            f"Pre-trained GMVAE directory not found: {pretrained_dir}\n"
            f"Please run Stage 1 (gmvae_sweep.py) first and specify the best run's directory."
        )

    # Use pipeline's built-in method to load pretrained GMVAE
    # This handles: architecture inference, model initialization, weight loading,
    # copying parameter files, and setting the _gmvae_trained flag
    pipeline.load_pretrained_gmvae(pretrained_dir)

    print("=" * 80)

    # Train BulkEncoder with tuned hyperparameters
    print("\n" + "=" * 80)
    print("Training BulkEncoder (Hyperparameter Tuning)")
    print("=" * 80)
    bulk_encoder_metrics = pipeline.train_bulk_encoder()

    # Log BulkEncoder metrics to W&B
    for metrics in bulk_encoder_metrics.history:
        log_dict = {
            "epoch": metrics.epoch,
            # Training metrics (raw, unweighted)
            "train_total_loss": metrics.total_loss,
            "train_mus_loss": metrics.mus_loss,
            "train_logvars_loss": metrics.logvars_loss,
            "train_pis_loss": metrics.pis_loss,
            "train_uniform_penalty": metrics.uniform_penalty,
        }

        # Validation metrics (raw, unweighted)
        if hasattr(metrics, "val_total_loss") and metrics.val_total_loss is not None:
            log_dict.update(
                {
                    "val_total_loss": metrics.val_total_loss,
                    "val_mus_loss": getattr(metrics, "val_mus_loss", 0) or 0,
                    "val_logvars_loss": getattr(metrics, "val_logvars_loss", 0) or 0,
                    "val_pis_loss": getattr(metrics, "val_pis_loss", 0) or 0,
                    "val_uniform_penalty": getattr(metrics, "val_uniform_penalty", 0)
                    or 0,
                }
            )

        wandb.log(log_dict)

    # Calculate best validation pis loss (our primary metric)
    val_pis_losses = [
        m.val_pis_loss
        for m in bulk_encoder_metrics.history
        if hasattr(m, "val_pis_loss") and m.val_pis_loss is not None
    ]

    if val_pis_losses:
        best_val_pis = min(val_pis_losses)
        print(
            f"\n📊 Best validation pis loss (fraction prediction): {best_val_pis:.6f}"
        )
    else:
        # Fallback to training loss if no validation
        best_val_pis = bulk_encoder_metrics.history[-1].pis_loss
        print(
            f"\n⚠️  No validation data - using final training pis loss: {best_val_pis:.6f}"
        )

    # Log the primary optimization metric
    wandb.log({"val_metric": best_val_pis})

    # Also log final epoch metrics for reference
    final_metrics = bulk_encoder_metrics.history[-1]
    wandb.log(
        {
            "final_train_pis": final_metrics.pis_loss,
            "final_train_mus": final_metrics.mus_loss,
            "final_train_logvars": final_metrics.logvars_loss,
            "final_train_total": final_metrics.total_loss,
        }
    )

    print("\n" + "=" * 80)
    print("BULK ENCODER TRAINING COMPLETED")
    print(f"Primary Metric (val_pis_loss): {best_val_pis:.6f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run W&B sweep for BulkEncoder hyperparameter tuning (Stage 2)"
    )
    parser.add_argument("sweep_config", help="Path to BulkEncoder sweep config YAML")
    parser.add_argument(
        "--count", type=int, default=10, help="Number of runs (default: 10)"
    )
    parser.add_argument(
        "--project", default=None, help="W&B project name (overrides config file)"
    )

    args = parser.parse_args()

    # Load sweep configuration
    with open(args.sweep_config, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Use project from command line if provided, otherwise use config file
    project = (
        args.project
        if args.project
        else sweep_config.get("project", "b2sc-bulk-encoder-hpt")
    )

    # Create and run sweep
    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"\n🚀 Starting BulkEncoder sweep: {sweep_id}")
    print(f"   Project: {project}")
    print(f"   Sweep name: {sweep_config.get('name', 'auto-generated')}")
    print(f"   Runs: {args.count}")
    print(f"   Optimizing: {sweep_config['metric']['name']}")

    wandb.agent(
        sweep_id, function=train_bulk_encoder, count=args.count, project=project
    )

    print(f"\n✅ Sweep completed! View results at: https://wandb.ai/{project}")


if __name__ == "__main__":
    main()
