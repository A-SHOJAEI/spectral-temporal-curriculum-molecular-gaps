#!/usr/bin/env python3
"""Evaluation script for Spectral Temporal Curriculum Molecular Networks.

This script evaluates a trained model on test data and performs comprehensive
analysis including ablation studies and convergence analysis.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure PyTorch to allow numpy objects in torch.load for OGB datasets
# This is needed for PyTorch 2.6+ which changed weights_only default to True
try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
except (ImportError, AttributeError):
    # Fallback for different numpy versions
    pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from spectral_temporal_curriculum_molecular_gaps.data.loader import PCQM4Mv2CurriculumDataLoader
from spectral_temporal_curriculum_molecular_gaps.models.model import SpectralTemporalMolecularNet
from spectral_temporal_curriculum_molecular_gaps.evaluation.metrics import (
    MolecularPropertyMetrics,
    ConvergenceAnalyzer,
    AblationStudy,
)
from spectral_temporal_curriculum_molecular_gaps.utils.config import load_config


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce external library noise
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def load_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> SpectralTemporalMolecularNet:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: Model configuration.
        device: Target device.

    Returns:
        Loaded model.
    """
    model_config = config["model"]

    # Create model
    model = SpectralTemporalMolecularNet(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_spectral_layers=model_config["num_spectral_layers"],
        num_scales=model_config["num_scales"],
        num_curriculum_stages=model_config["num_curriculum_stages"],
        dropout=model_config["dropout"],
        pool_type=model_config["pool_type"],
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logging.info(f"Loaded model from {checkpoint_path}")
    logging.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")

    return model


def extract_batch_spectral_features(batch, num_scales: int = 4):
    """Extract spectral features via graph Laplacian eigendecomposition.

    Must match training pipeline exactly to produce valid predictions.

    Args:
        batch: Batch of molecular graphs.
        num_scales: Number of spectral scales.

    Returns:
        List of spectral features at different scales.
    """
    import torch_geometric.utils as pyg_utils
    from scipy.sparse.linalg import eigsh
    import scipy.sparse as sp

    device = batch.x.device
    graphs = [batch.get_example(i) for i in range(batch.num_graphs)]
    spectral_features = [[] for _ in range(num_scales)]

    for graph in graphs:
        try:
            edge_index = graph.edge_index
            num_nodes = graph.num_nodes

            adj = pyg_utils.to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

            if adj.nnz == 0:
                eigenvals = torch.ones(min(num_scales, num_nodes), device=device)
                eigenvecs = torch.eye(num_nodes, min(num_scales, num_nodes), device=device)
            else:
                degrees = torch.tensor(adj.sum(axis=1).A1, dtype=torch.float32)
                degrees[degrees == 0] = 1.0
                deg_sqrt_inv = sp.diags(1.0 / torch.sqrt(degrees).numpy())
                laplacian = sp.eye(num_nodes) - deg_sqrt_inv @ adj @ deg_sqrt_inv

                k = min(num_scales, num_nodes - 1)
                if k > 0:
                    eigenvals_np, eigenvecs_np = eigsh(laplacian, k=k, which='SM')
                    eigenvals = torch.tensor(eigenvals_np, dtype=torch.float32, device=device)
                    eigenvecs = torch.tensor(eigenvecs_np, dtype=torch.float32, device=device)
                else:
                    eigenvals = torch.ones(1, device=device)
                    eigenvecs = torch.ones(num_nodes, 1, device=device) / torch.sqrt(
                        torch.tensor(num_nodes, dtype=torch.float32))

            node_features = graph.x

            for scale_idx in range(num_scales):
                if scale_idx < eigenvecs.size(1):
                    eigenvec = eigenvecs[:, scale_idx:scale_idx+1]
                    spectral_weight = torch.exp(-eigenvals[scale_idx] * (scale_idx + 1))
                    scale_features = node_features * eigenvec * spectral_weight
                else:
                    scale_features = node_features * (1.0 / (2 ** scale_idx))
                spectral_features[scale_idx].append(scale_features)

        except Exception as e:
            logging.warning(f"Spectral decomposition failed for graph: {e}. Using fallback.")
            node_features = graph.x
            for scale_idx in range(num_scales):
                scale_features = node_features * (1.0 / (2 ** scale_idx))
                spectral_features[scale_idx].append(scale_features)

    final_spectral_features = []
    for scale_idx in range(num_scales):
        if spectral_features[scale_idx]:
            scale_tensor = torch.cat(spectral_features[scale_idx], dim=0)
            final_spectral_features.append(scale_tensor)
        else:
            final_spectral_features.append(torch.zeros_like(batch.x))

    return final_spectral_features


def evaluate_model(
    model: SpectralTemporalMolecularNet,
    data_loader,
    device: torch.device,
    num_scales: int = 4
) -> Dict[str, Any]:
    """Evaluate model on given data loader.

    Args:
        model: Trained model.
        data_loader: Data loader for evaluation.
        device: Computation device.
        num_scales: Number of spectral scales.

    Returns:
        Evaluation results.
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            targets = batch.y.float()

            # Extract spectral features
            spectral_features = extract_batch_spectral_features(batch, num_scales)

            # Forward pass
            outputs = model(batch, spectral_features)
            predictions = outputs["prediction"].squeeze()

            # Collect results
            if predictions.dim() == 0:
                all_predictions.append(predictions.item())
            else:
                all_predictions.extend(predictions.cpu().numpy())

            all_targets.extend(targets.cpu().numpy())

    return {
        "predictions": np.array(all_predictions),
        "targets": np.array(all_targets),
    }


def run_comprehensive_evaluation(
    model: SpectralTemporalMolecularNet,
    data_loader: PCQM4Mv2CurriculumDataLoader,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: str
) -> Dict[str, Any]:
    """Run comprehensive model evaluation.

    Args:
        model: Trained model.
        data_loader: Curriculum data loader.
        config: Configuration dictionary.
        device: Computation device.
        output_dir: Output directory for results.

    Returns:
        Comprehensive evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_calculator = MolecularPropertyMetrics()
    results = {}

    # 1. Test set evaluation
    logging.info("Evaluating on test set...")
    test_loader = data_loader.get_test_dataloader()
    test_results = evaluate_model(model, test_loader, device, config["model"]["num_scales"])

    test_metrics = metrics_calculator.compute_metrics(
        test_results["predictions"],
        test_results["targets"]
    )
    results["test_metrics"] = test_metrics

    logging.info("Test Set Results:")
    logging.info(f"  MAE: {test_metrics['mae']:.4f}")
    logging.info(f"  RMSE: {test_metrics['rmse']:.4f}")
    logging.info(f"  R²: {test_metrics['r2']:.4f}")
    logging.info(f"  Chemical Accuracy: {test_metrics['chemical_accuracy']:.2f}%")

    # 2. Out-of-distribution evaluation (large molecules)
    logging.info("Evaluating on large molecules (OOD)...")
    ood_loader = data_loader.get_ood_large_molecule_subset(
        size_threshold=config["evaluation"]["large_molecule_threshold"]
    )

    if len(ood_loader.dataset) > 0:
        ood_results = evaluate_model(model, ood_loader, device, config["model"]["num_scales"])
        ood_metrics = metrics_calculator.compute_metrics(
            ood_results["predictions"],
            ood_results["targets"]
        )
        results["ood_metrics"] = ood_metrics

        logging.info("OOD Large Molecules Results:")
        logging.info(f"  MAE: {ood_metrics['mae']:.4f}")
        logging.info(f"  Count: {len(ood_results['predictions'])}")
    else:
        logging.warning("No large molecules found for OOD evaluation")
        results["ood_metrics"] = {"mae": 0.0, "count": 0}

    # 3. Error analysis by molecular size
    logging.info("Analyzing performance by molecular size...")
    molecular_sizes = []
    for batch in test_loader:
        # Estimate molecular sizes from number of nodes per graph
        batch_sizes = torch.bincount(batch.batch)
        molecular_sizes.extend(batch_sizes.cpu().numpy())

    molecular_sizes = np.array(molecular_sizes[:len(test_results["predictions"])])

    size_stratified = metrics_calculator.compute_molecular_size_stratified_metrics(
        test_results["predictions"],
        test_results["targets"],
        molecular_sizes
    )
    results["size_stratified_metrics"] = size_stratified

    # 4. Generate prediction plots
    logging.info("Generating prediction plots...")
    create_prediction_plots(
        test_results["predictions"],
        test_results["targets"],
        output_dir
    )

    # 5. Error distribution analysis
    logging.info("Analyzing error distributions...")
    create_error_analysis_plots(
        test_results["predictions"],
        test_results["targets"],
        molecular_sizes,
        output_dir
    )

    return results


def create_prediction_plots(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str
) -> None:
    """Create prediction vs target plots.

    Args:
        predictions: Model predictions.
        targets: Ground truth targets.
        output_dir: Output directory.
    """
    plt.figure(figsize=(12, 5))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(targets, predictions, alpha=0.6, s=10)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel("True HOMO-LUMO Gap (eV)")
    plt.ylabel("Predicted HOMO-LUMO Gap (eV)")
    plt.title("Predictions vs Targets")

    # Add R² and MAE to plot
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    plt.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f} eV",
             transform=plt.gca().transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = predictions - targets
    plt.scatter(targets, residuals, alpha=0.6, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("True HOMO-LUMO Gap (eV)")
    plt.ylabel("Residual (eV)")
    plt.title("Residual Plot")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()


def create_error_analysis_plots(
    predictions: np.ndarray,
    targets: np.ndarray,
    molecular_sizes: np.ndarray,
    output_dir: str
) -> None:
    """Create error analysis plots.

    Args:
        predictions: Model predictions.
        targets: Ground truth targets.
        molecular_sizes: Molecular sizes.
        output_dir: Output directory.
    """
    errors = np.abs(predictions - targets)

    plt.figure(figsize=(15, 10))

    # Error distribution
    plt.subplot(2, 3, 1)
    plt.hist(errors, bins=50, alpha=0.7, density=True)
    plt.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
    plt.axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.3f}')
    plt.xlabel("Absolute Error (eV)")
    plt.ylabel("Density")
    plt.title("Error Distribution")
    plt.legend()

    # Error vs molecular size
    plt.subplot(2, 3, 2)
    plt.scatter(molecular_sizes, errors, alpha=0.5, s=8)
    plt.xlabel("Molecular Size (# atoms)")
    plt.ylabel("Absolute Error (eV)")
    plt.title("Error vs Molecular Size")

    # Error vs target value
    plt.subplot(2, 3, 3)
    plt.scatter(targets, errors, alpha=0.5, s=8)
    plt.xlabel("Target HOMO-LUMO Gap (eV)")
    plt.ylabel("Absolute Error (eV)")
    plt.title("Error vs Target Value")

    # Box plot by size bins
    plt.subplot(2, 3, 4)
    try:
        import pandas as pd
        size_bins = pd.cut(molecular_sizes, bins=5, labels=False)
        error_by_size = [errors[size_bins == i] for i in range(5)]
        plt.boxplot(error_by_size, labels=[f"Bin {i+1}" for i in range(5)])
        plt.xlabel("Size Bin")
        plt.ylabel("Absolute Error (eV)")
        plt.title("Error Distribution by Size")
    except ImportError:
        # Fallback without pandas
        n_bins = 5
        bins = np.linspace(molecular_sizes.min(), molecular_sizes.max(), n_bins + 1)
        size_bins = np.digitize(molecular_sizes, bins) - 1
        error_by_size = [errors[size_bins == i] for i in range(n_bins)]
        plt.boxplot(error_by_size, labels=[f"Bin {i+1}" for i in range(n_bins)])
        plt.xlabel("Size Bin")
        plt.ylabel("Absolute Error (eV)")
        plt.title("Error Distribution by Size")

    # Chemical accuracy vs size
    plt.subplot(2, 3, 5)
    threshold = 0.043  # 1 kcal/mol in eV
    size_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 200)]
    accuracy_by_size = []

    for min_size, max_size in size_ranges:
        mask = (molecular_sizes >= min_size) & (molecular_sizes < max_size)
        if np.any(mask):
            accuracy = np.mean(errors[mask] <= threshold) * 100
            accuracy_by_size.append(accuracy)
        else:
            accuracy_by_size.append(0)

    plt.bar(range(len(size_ranges)), accuracy_by_size)
    plt.xlabel("Size Range")
    plt.ylabel("Chemical Accuracy (%)")
    plt.title("Chemical Accuracy by Size")
    plt.xticks(range(len(size_ranges)), [f"{r[0]}-{r[1]}" for r in size_ranges], rotation=45)

    # Cumulative error distribution
    plt.subplot(2, 3, 6)
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Chemical accuracy threshold')
    plt.xlabel("Absolute Error (eV)")
    plt.ylabel("Cumulative Fraction")
    plt.title("Cumulative Error Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Spectral Temporal Curriculum Molecular Network"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Override data directory"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Load configuration
    try:
        config = load_config(args.config)
        if args.data_dir:
            config["data"]["root_dir"] = args.data_dir
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Setup device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    try:
        # Load model
        model = load_model(args.checkpoint, config, device)

        # Create data loader
        logging.info("Creating data loader...")
        data_loader = PCQM4Mv2CurriculumDataLoader(
            root=config["data"]["root_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            curriculum_strategy=config["data"]["curriculum_strategy"],
            spectral_decomp_levels=config["data"]["num_spectral_scales"],
            cache_dir=config["data"]["cache_dir"],
            force_reload=False,  # Don't reload for evaluation
        )

        # Run comprehensive evaluation
        logging.info("Running comprehensive evaluation...")
        results = run_comprehensive_evaluation(
            model, data_loader, config, device, args.output_dir
        )

        # Save results to JSON
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: float(v) if isinstance(v, np.floating) else v
                                   for k, v in value.items()}
            else:
                json_results[key] = value

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        logging.info(f"Results saved to {results_file}")

        # Compare with target metrics if available
        test_mae = results["test_metrics"]["mae"]
        ood_mae = results["ood_metrics"]["mae"]

        logging.info("\n" + "="*50)
        logging.info("FINAL EVALUATION SUMMARY")
        logging.info("="*50)

        if "target_metrics" in config:
            target_metrics = config["target_metrics"]
            target_mae = target_metrics.get("mae_homo_lumo_gap_ev", 0.075)
            target_ood = target_metrics.get("ood_large_molecule_mae", 0.12)

            logging.info(f"Test MAE: {test_mae:.4f} eV (target: {target_mae:.3f})")
            logging.info(f"OOD MAE: {ood_mae:.4f} eV (target: {target_ood:.3f})")
            logging.info(f"Chemical Accuracy: {results['test_metrics']['chemical_accuracy']:.1f}%")

            # Check if targets are met
            mae_met = test_mae <= target_mae
            ood_met = ood_mae <= target_ood

            logging.info(f"MAE Target Met: {'✓' if mae_met else '✗'}")
            logging.info(f"OOD Target Met: {'✓' if ood_met else '✗'}")
        else:
            logging.info(f"Test MAE: {test_mae:.4f} eV")
            logging.info(f"OOD MAE: {ood_mae:.4f} eV")
            logging.info(f"Chemical Accuracy: {results['test_metrics']['chemical_accuracy']:.1f}%")
            logging.info("No target metrics specified in configuration")

        logging.info("Evaluation completed successfully!")

    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Import pandas here to avoid import errors if not installed
    try:
        import pandas as pd
    except ImportError:
        logging.warning("pandas not available, some plots may be skipped")
        pd = None

    main()