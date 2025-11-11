"""
Generation results container for B2SC pipeline.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import anndata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional
from termcolor import colored


@dataclass
class GenerationResults:
    """Container for generation results."""

    generated_cells: Dict[str, torch.Tensor]  # sample_id -> generated_cells
    cell_types: Dict[str, torch.Tensor]  # sample_id -> cell_type_labels
    generation_stats: Dict[str, Any]
    mapping_dict: Optional[Dict[str, int]] = None
    color_map: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.generated_cells:
            raise ValueError("generated_cells cannot be empty")

        if not self.cell_types:
            raise ValueError("cell_types cannot be empty")

        # Validate that sample IDs match
        gen_samples = set(self.generated_cells.keys())
        ct_samples = set(self.cell_types.keys())
        if gen_samples != ct_samples:
            raise ValueError(
                f"Sample ID mismatch: generated_cells has {gen_samples}, cell_types has {ct_samples}"
            )

    def save_results(self, save_dir: Path):
        """Save all results to directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(colored(f"Saving generation results to {save_dir}...", "yellow"))

        # Save generated cells
        generated_cells_path = save_dir / "generated_cells.pkl"
        with open(generated_cells_path, "wb") as f:
            pickle.dump(self.generated_cells, f)
        print(colored(f"Generated cells saved to {generated_cells_path}", "green"))

        # Save cell types
        cell_types_path = save_dir / "cell_types.pkl"
        with open(cell_types_path, "wb") as f:
            pickle.dump(self.cell_types, f)
        print(colored(f"Cell types saved to {cell_types_path}", "green"))

        # Save generation stats
        stats_path = save_dir / "generation_stats.pkl"
        with open(stats_path, "wb") as f:
            pickle.dump(self.generation_stats, f)
        print(colored(f"Generation stats saved to {stats_path}", "green"))

        # Save mapping dict if available
        if self.mapping_dict is not None:
            mapping_path = save_dir / "mapping_dict.pkl"
            with open(mapping_path, "wb") as f:
                pickle.dump(self.mapping_dict, f)
            print(colored(f"Mapping dict saved to {mapping_path}", "green"))

        # Save color map if available
        if self.color_map is not None:
            colormap_path = save_dir / "color_map.pkl"
            with open(colormap_path, "wb") as f:
                pickle.dump(self.color_map, f)
            print(colored(f"Color map saved to {colormap_path}", "green"))

        # Save summary statistics
        self._save_summary_stats(save_dir)

        # Generate visualizations (save to files)
        self._generate_visualizations(save_dir)

        # Also display visualizations inline in notebooks
        print(colored("Displaying visualizations inline...", "yellow"))
        self.plot_visualizations(show_plots=True)

    @classmethod
    def load_results(cls, save_dir: Path) -> "GenerationResults":
        """Load results from directory."""
        save_dir = Path(save_dir)

        # Load required files
        with open(save_dir / "generated_cells.pkl", "rb") as f:
            generated_cells = pickle.load(f)

        with open(save_dir / "cell_types.pkl", "rb") as f:
            cell_types = pickle.load(f)

        with open(save_dir / "generation_stats.pkl", "rb") as f:
            generation_stats = pickle.load(f)

        # Load optional files
        mapping_dict = None
        if (save_dir / "mapping_dict.pkl").exists():
            with open(save_dir / "mapping_dict.pkl", "rb") as f:
                mapping_dict = pickle.load(f)

        color_map = None
        if (save_dir / "color_map.pkl").exists():
            with open(save_dir / "color_map.pkl", "rb") as f:
                color_map = pickle.load(f)

        return cls(
            generated_cells=generated_cells,
            cell_types=cell_types,
            generation_stats=generation_stats,
            mapping_dict=mapping_dict,
            color_map=color_map,
        )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        summary = {
            "num_samples": len(self.generated_cells),
            "sample_ids": list(self.generated_cells.keys()),
            "total_cells_generated": sum(
                tensor.shape[0] for tensor in self.generated_cells.values()
            ),
        }

        # Per-sample statistics
        sample_stats = {}
        for sample_id, cells in self.generated_cells.items():
            cell_types_tensor = self.cell_types[sample_id]
            unique_types, counts = torch.unique(cell_types_tensor, return_counts=True)

            sample_stats[sample_id] = {
                "num_cells": cells.shape[0],
                "num_genes": cells.shape[1],
                "cell_type_counts": {
                    int(ct): int(count) for ct, count in zip(unique_types, counts)
                },
                "cell_type_proportions": {
                    int(ct): float(count) / len(cell_types_tensor)
                    for ct, count in zip(unique_types, counts)
                },
            }

        summary["per_sample_stats"] = sample_stats

        # Overall cell type statistics
        all_cell_types = torch.cat(list(self.cell_types.values()))
        unique_types, counts = torch.unique(all_cell_types, return_counts=True)
        summary["overall_cell_type_counts"] = {
            int(ct): int(count) for ct, count in zip(unique_types, counts)
        }
        summary["overall_cell_type_proportions"] = {
            int(ct): float(count) / len(all_cell_types)
            for ct, count in zip(unique_types, counts)
        }

        # Add generation stats
        summary.update(self.generation_stats)

        return summary

    def to_anndata(self) -> anndata.AnnData:
        """Convert generation results to AnnData object."""
        # Concatenate all generated cells
        all_cells = []
        all_cell_types = []
        all_sample_ids = []

        for sample_id, cells_tensor in self.generated_cells.items():
            cell_types_tensor = self.cell_types[sample_id]

            # Convert tensors to numpy
            cells_np = cells_tensor.detach().cpu().numpy()
            cell_types_np = cell_types_tensor.detach().cpu().numpy()

            all_cells.append(cells_np)
            all_cell_types.extend(cell_types_np)
            all_sample_ids.extend([sample_id] * cells_np.shape[0])

        # Concatenate all cells
        X = np.vstack(all_cells)

        # Create observation metadata
        obs = pd.DataFrame({"sample_id": all_sample_ids, "cell_type": all_cell_types})

        # Create variable metadata (genes)
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(X.shape[1])])

        # Create AnnData object
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        return adata

    def print_summary(self):
        """Print a formatted summary of generation results."""
        summary = self.get_summary_statistics()

        print(f"📊 Generation Summary")
        print(f"{'='*25}")
        print(f"Total samples: {summary['num_samples']}")
        print(f"Total cells: {summary['total_cells_generated']}")
        print(f"Samples: {', '.join(summary['sample_ids'])}")

        print(f"\n📋 Per-sample breakdown:")
        for sample_id, stats in summary["per_sample_stats"].items():
            print(f"  {sample_id}: {stats['num_cells']} cells")

    @property
    def sample_ids(self) -> List[str]:
        """Get list of sample IDs."""
        return list(self.generated_cells.keys())

    def save(self, filepath: Path):
        """Save generation results to file."""
        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "GenerationResults":
        """Load generation results from file."""
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def plot_visualizations(self, show_plots: bool = True):
        """Generate and optionally display visualization plots inline."""
        try:
            # Cell type distribution plot
            self._plot_cell_type_distribution_inline(show_plots)

            # Per-sample cell type distribution
            self._plot_per_sample_distribution_inline(show_plots)

        except Exception as e:
            print(colored(f"Warning: Failed to generate visualizations: {e}", "yellow"))

    def _plot_cell_type_distribution_inline(self, show_plot: bool = True):
        """Plot overall cell type distribution inline."""
        all_cell_types = torch.cat(list(self.cell_types.values()))
        unique_types, counts = torch.unique(all_cell_types, return_counts=True)

        # Convert to numpy for plotting
        types_np = unique_types.cpu().numpy()
        counts_np = counts.cpu().numpy()

        # Create labels using mapping dict if available
        if self.mapping_dict:
            reverse_mapping = {v: k for k, v in self.mapping_dict.items()}
            labels = [reverse_mapping.get(int(t), f"Type_{int(t)}") for t in types_np]
        else:
            labels = [f"Type_{int(t)}" for t in types_np]

        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, counts_np, alpha=0.7)
        plt.title("Overall Cell Type Distribution")
        plt.xlabel("Cell Type")
        plt.ylabel("Number of Cells")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, counts_np):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(counts_np),
                str(int(count)),
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        if show_plot:
            plt.show()
        plt.close()

    def _plot_per_sample_distribution_inline(self, show_plot: bool = True):
        """Plot cell type distribution per sample inline."""
        n_samples = len(self.generated_cells)
        cols = min(3, n_samples)
        rows = (n_samples + cols - 1) // cols

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_samples == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if n_samples == 1 else axes
        else:
            axes = axes.flatten()

        for idx, (sample_id, cell_types_tensor) in enumerate(self.cell_types.items()):
            ax = axes[idx] if n_samples > 1 else axes[0]

            unique_types, counts = torch.unique(cell_types_tensor, return_counts=True)
            types_np = unique_types.cpu().numpy()
            counts_np = counts.cpu().numpy()

            # Create labels
            if self.mapping_dict:
                reverse_mapping = {v: k for k, v in self.mapping_dict.items()}
                labels = [
                    reverse_mapping.get(int(t), f"Type_{int(t)}") for t in types_np
                ]
            else:
                labels = [f"Type_{int(t)}" for t in types_np]

            # Create bar plot
            bars = ax.bar(labels, counts_np, alpha=0.7)
            ax.set_title(f"Sample: {sample_id}")
            ax.set_xlabel("Cell Type")
            ax.set_ylabel("Number of Cells")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, count in zip(bars, counts_np):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(counts_np),
                    str(int(count)),
                    ha="center",
                    va="bottom",
                )

        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        if show_plot:
            plt.show()
        plt.close()

    def _save_summary_stats(self, save_dir: Path):
        """Save summary statistics to text file."""
        summary = self.get_summary_statistics()

        summary_path = save_dir / "summary_statistics.txt"
        with open(summary_path, "w") as f:
            f.write("B2SC Generation Results Summary\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Number of samples: {summary['num_samples']}\n")
            f.write(f"Total cells generated: {summary['total_cells_generated']}\n\n")

            f.write("Sample IDs:\n")
            for sample_id in summary["sample_ids"]:
                f.write(f"  - {sample_id}\n")
            f.write("\n")

            f.write("Overall Cell Type Distribution:\n")
            for ct, count in summary["overall_cell_type_counts"].items():
                prop = summary["overall_cell_type_proportions"][ct]
                # Try to get cell type name from mapping
                ct_name = ct
                if self.mapping_dict:
                    reverse_mapping = {v: k for k, v in self.mapping_dict.items()}
                    ct_name = reverse_mapping.get(ct, str(ct))
                f.write(f"  Cell Type {ct} ({ct_name}): {count} cells ({prop:.3f})\n")
            f.write("\n")

            f.write("Per-Sample Statistics:\n")
            for sample_id, stats in summary["per_sample_stats"].items():
                f.write(f"\n  Sample: {sample_id}\n")
                f.write(f"    Cells: {stats['num_cells']}\n")
                f.write(f"    Genes: {stats['num_genes']}\n")
                f.write(f"    Cell Type Distribution:\n")
                for ct, count in stats["cell_type_counts"].items():
                    prop = stats["cell_type_proportions"][ct]
                    ct_name = ct
                    if self.mapping_dict:
                        reverse_mapping = {v: k for k, v in self.mapping_dict.items()}
                        ct_name = reverse_mapping.get(ct, str(ct))
                    f.write(f"      Type {ct} ({ct_name}): {count} ({prop:.3f})\n")

        print(colored(f"Summary statistics saved to {summary_path}", "green"))

    def _generate_visualizations(self, save_dir: Path):
        """Generate visualization plots."""
        try:
            # Cell type distribution plot
            self._plot_cell_type_distribution(save_dir)

            # Per-sample cell type distribution
            self._plot_per_sample_distribution(save_dir)

        except Exception as e:
            print(colored(f"Warning: Failed to generate visualizations: {e}", "yellow"))

    def _plot_cell_type_distribution(self, save_dir: Path):
        """Plot overall cell type distribution."""
        all_cell_types = torch.cat(list(self.cell_types.values()))
        unique_types, counts = torch.unique(all_cell_types, return_counts=True)

        # Convert to numpy for plotting
        types_np = unique_types.numpy()
        counts_np = counts.numpy()

        # Get cell type names if mapping is available
        labels = []
        for ct in types_np:
            if self.mapping_dict:
                reverse_mapping = {v: k for k, v in self.mapping_dict.items()}
                labels.append(reverse_mapping.get(int(ct), str(ct)))
            else:
                labels.append(str(ct))

        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(types_np)), counts_np)
        plt.xlabel("Cell Type")
        plt.ylabel("Number of Cells")
        plt.title("Overall Cell Type Distribution")
        plt.xticks(range(len(types_np)), labels, rotation=45, ha="right")
        plt.grid(True)
        plt.tight_layout()

        plot_path = save_dir / "cell_type_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(colored(f"Cell type distribution plot saved to {plot_path}", "green"))

    def _plot_per_sample_distribution(self, save_dir: Path):
        """Plot cell type distribution per sample."""
        num_samples = len(self.generated_cells)
        if num_samples == 0:
            return

        # Create subplots
        cols = min(3, num_samples)
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if num_samples == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()

        for idx, (sample_id, cell_types_tensor) in enumerate(self.cell_types.items()):
            if idx >= len(axes):
                break

            unique_types, counts = torch.unique(cell_types_tensor, return_counts=True)
            types_np = unique_types.numpy()
            counts_np = counts.numpy()

            # Get cell type names
            labels = []
            for ct in types_np:
                if self.mapping_dict:
                    reverse_mapping = {v: k for k, v in self.mapping_dict.items()}
                    labels.append(reverse_mapping.get(int(ct), str(ct)))
                else:
                    labels.append(str(ct))

            ax = axes[idx]
            bars = ax.bar(range(len(types_np)), counts_np)
            ax.set_xlabel("Cell Type")
            ax.set_ylabel("Number of Cells")
            ax.set_title(f"Sample: {sample_id}")
            ax.set_xticks(range(len(types_np)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(True)

        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        plot_path = save_dir / "per_sample_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(colored(f"Per-sample distribution plot saved to {plot_path}", "green"))

    def validate_quality(self) -> Dict[str, float]:
        """Validate quality of generated cells."""
        quality_metrics = {}

        # Basic validation metrics
        total_cells = sum(tensor.shape[0] for tensor in self.generated_cells.values())
        quality_metrics["total_cells_generated"] = float(total_cells)

        # Check for negative values (shouldn't happen with proper generation)
        negative_counts = []
        for sample_id, cells in self.generated_cells.items():
            neg_count = (cells < 0).sum().item()
            negative_counts.append(neg_count)
        quality_metrics["total_negative_values"] = float(sum(negative_counts))
        quality_metrics["samples_with_negative_values"] = float(
            sum(1 for count in negative_counts if count > 0)
        )

        # Cell type diversity
        all_cell_types = torch.cat(list(self.cell_types.values()))
        unique_types = torch.unique(all_cell_types)
        quality_metrics["num_unique_cell_types"] = float(len(unique_types))

        # Expression statistics
        all_cells = torch.cat(list(self.generated_cells.values()))
        quality_metrics["mean_expression_per_cell"] = float(all_cells.sum(dim=1).mean())
        quality_metrics["std_expression_per_cell"] = float(all_cells.sum(dim=1).std())
        quality_metrics["mean_expression_per_gene"] = float(all_cells.sum(dim=0).mean())
        quality_metrics["std_expression_per_gene"] = float(all_cells.sum(dim=0).std())

        return quality_metrics


def split_generated_tensors(
    generated_aggregate_tensor: torch.Tensor,
    sampled_celltypes: torch.Tensor,
    num_samples: int,
    sample_ids: List[str],
) -> tuple:
    """
    Splits an interleaved tensor of shape (N * M, K), where M = num_samples,
    into a dictionary mapping each sample index (0 to M-1) to its own tensor of shape (N, K).

    Args:
        generated_aggregate_tensor (torch.Tensor): Tensor of shape (N * M, K),
            where rows are interleaved as:
            [sample_0_row0,
             sample_1_row0,
             ...,
             sample_{M-1}_row0,
             sample_0_row1,
             ...,
             sample_{M-1}_row1,
             ...]
        sampled_celltypes (torch.Tensor): Cell type labels corresponding to each row
        num_samples (int): Number of samples (M).
        sample_ids (List[str]): List of sample identifiers

    Returns:
        tuple: (generated_tensor_dict, cell_type_dict)
            - generated_tensor_dict: Dictionary with sample names as keys, tensors as values
            - cell_type_dict: Dictionary with sample names as keys, cell type tensors as values

    Raises:
        AssertionError: if input tensor is not 2D, or if the number of rows is not divisible by num_samples,
                        or if any per‐sample slice does not have the same number of rows.
    """
    # Ensure input is a 2D tensor
    assert isinstance(
        generated_aggregate_tensor, torch.Tensor
    ), f"Expected a torch.Tensor, but got {type(generated_aggregate_tensor)}"
    assert (
        generated_aggregate_tensor.dim() == 2
    ), f"Input tensor must be 2D, but has shape {generated_aggregate_tensor.shape}"

    total_rows, num_cols = generated_aggregate_tensor.shape

    # Check divisibility to infer N
    assert (
        total_rows % num_samples == 0
    ), f"Number of rows ({total_rows}) must be divisible by num_samples ({num_samples})"
    rows_per_sample = total_rows // num_samples  # N

    # Prepare output dict
    generated_tensor_dict = {}
    cell_type_dict = {}
    for sample_idx, sample_name in enumerate(sample_ids):
        # Slice out rows for this sample: start at `sample_idx`, step by `num_samples`
        sample_tensor = generated_aggregate_tensor[sample_idx:total_rows:num_samples, :]
        cell_type_tensor = sampled_celltypes[sample_idx:total_rows:num_samples]
        # Verify that we got exactly N rows for this sample
        assert sample_tensor.shape == (rows_per_sample, num_cols), (
            f"Sample {sample_idx}: expected shape ({rows_per_sample}, {num_cols}), "
            f"but got {sample_tensor.shape}"
        )

        generated_tensor_dict[sample_name] = sample_tensor
        cell_type_dict[sample_name] = cell_type_tensor

    # Final sanity check: dictionary should have exactly M entries
    assert (
        len(generated_tensor_dict) == num_samples
    ), f"Expected {num_samples} entries in the output dict, but got {len(generated_tensor_dict)}"
    assert (
        len(cell_type_dict) == num_samples
    ), f"Expected {num_samples} entries in the output dict, but got {len(cell_type_dict)}"

    return generated_tensor_dict, cell_type_dict
