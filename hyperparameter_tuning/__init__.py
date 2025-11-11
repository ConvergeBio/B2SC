"""
B2SC Hyperparameter Tuning Package

This package provides tools for hyperparameter optimization of the B2SC pipeline
using Weights & Biases (wandb) sweeps.

Main components:
- run_sweep.py: Main script for running hyperparameter sweeps
- analyze_results.py: Script for analyzing and visualizing sweep results
- sweep_config.yaml: W&B sweep configuration defining the search space
- data_config.yaml: Data paths configuration

For usage instructions, see README.md
"""

__version__ = "1.0.0"
__author__ = "B2SC Development Team"

from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

