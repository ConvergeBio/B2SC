#!/usr/bin/env python3
"""
GMVAE Hyperparameter Tuning with W&B Sweeps (Stage 1)

This script performs hyperparameter optimization for the GMVAE model only.
The best configuration from this sweep should be used for Stage 2 (BulkEncoder tuning).

Evaluation Metric: val_reconstruction_loss (raw, unweighted)
- Independent of tunable loss weights
- Directly measures core objective: gene expression reconstruction quality
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


def train_gmvae():
    """Training function called by wandb agent."""
    wandb.init()
    config = wandb.config

    # Extract data configuration
    data_config = {
        "sc_training_path": config.sc_training_path,
        "sc_validation_path": config.sc_validation_path,
        "base_dir": config.base_dir,
    }

    # Create B2SC configuration
    run_name = wandb.run.name or f"run_{wandb.run.id}"
    base_dir = Path(data_config["base_dir"]) / "gmvae_sweep" / run_name

    b2sc_config = B2SCConfig(
        sc_training_path=data_config["sc_training_path"],
        sc_validation_path=data_config["sc_validation_path"],
        base_dir=base_dir,
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        gmvae_hyperparameters={
            "hidden_dim": config.gmvae_hidden_dim,
            "epochs": config.gmvae_epochs,
            "learning_rate": config.gmvae_learning_rate,
            "batch_size": config.gmvae_batch_size,
            "recon_loss_weight": config.gmvae_recon_loss_weight,
            "zinb_loss_weight": config.gmvae_zinb_loss_weight,
            "fraction_loss_weight": config.gmvae_fraction_loss_weight,
            "kl_loss_weight": config.gmvae_kl_loss_weight,
            "recon_warmup_steepness": config.gmvae_recon_warmup_steepness,
            "recon_warmup_midpoint": config.gmvae_recon_warmup_midpoint,
            "use_weight_norm": config.gmvae_use_weight_norm,
        },
        device="auto",
    )

    # Initialize pipeline
    pipeline = B2SCPipeline(b2sc_config)

    # Train GMVAE only
    print("\n" + "=" * 80)
    print("TRAINING GMVAE (Stage 1)")
    print("=" * 80)
    gmvae_metrics = pipeline.train_gmvae()

    # Log all GMVAE metrics to W&B
    for metrics in gmvae_metrics.history:
        log_dict = {
            "epoch": metrics.epoch,
            # Training metrics (raw, unweighted)
            "train_total_loss": metrics.total_loss,
            "train_recon_loss": metrics.reconstruction_loss,
            "train_zinb_loss": getattr(metrics, "zinb_loss", 0) or 0,
            "train_fraction_loss": getattr(metrics, "fraction_loss", 0) or 0,
            "train_kl_loss": metrics.kl_loss,
        }
        
        # Validation metrics (raw, unweighted)
        if hasattr(metrics, "val_total_loss") and metrics.val_total_loss is not None:
            log_dict.update({
                "val_total_loss": metrics.val_total_loss,
                "val_recon_loss": getattr(metrics, "val_reconstruction_loss", 0) or 0,
                "val_zinb_loss": getattr(metrics, "val_zinb_loss", 0) or 0,
                "val_fraction_loss": getattr(metrics, "val_fraction_loss", 0) or 0,
                "val_kl_loss": getattr(metrics, "val_kl_loss", 0) or 0,
            })
        
        wandb.log(log_dict)

    # Calculate best validation reconstruction loss (our primary metric)
    val_recon_losses = [
        m.val_reconstruction_loss
        for m in gmvae_metrics.history
        if hasattr(m, "val_reconstruction_loss") and m.val_reconstruction_loss is not None
    ]
    
    if val_recon_losses:
        best_val_recon = min(val_recon_losses)
        print(f"\n📊 Best validation reconstruction loss: {best_val_recon:.2f}")
    else:
        # Fallback to training loss if no validation
        best_val_recon = gmvae_metrics.history[-1].reconstruction_loss
        print(f"\n⚠️  No validation data - using final training reconstruction loss: {best_val_recon:.2f}")
    
    # Log the primary optimization metric
    wandb.log({"val_metric": best_val_recon})
    
    # Also log final epoch metrics for reference
    final_metrics = gmvae_metrics.history[-1]
    wandb.log({
        "final_train_recon": final_metrics.reconstruction_loss,
        "final_train_total": final_metrics.total_loss,
    })

    print("\n" + "=" * 80)
    print("GMVAE TRAINING COMPLETED")
    print(f"Primary Metric (val_reconstruction_loss): {best_val_recon:.2f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run W&B sweep for GMVAE hyperparameter tuning (Stage 1)"
    )
    parser.add_argument("sweep_config", help="Path to GMVAE sweep config YAML")
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
    project = args.project if args.project else sweep_config.get('project', 'b2sc-gmvae-hpt')
    
    # Create and run sweep
    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"\n🚀 Starting GMVAE sweep: {sweep_id}")
    print(f"   Project: {project}")
    print(f"   Sweep name: {sweep_config.get('name', 'auto-generated')}")
    print(f"   Runs: {args.count}")
    print(f"   Optimizing: {sweep_config['metric']['name']}")
    
    wandb.agent(sweep_id, function=train_gmvae, count=args.count, project=project)
    
    print(f"\n✅ Sweep completed! View results at: https://wandb.ai/{args.project}")


if __name__ == "__main__":
    main()

