#!/usr/bin/env python3
"""Training script for Spectral Temporal Curriculum Molecular Networks.

This script trains a molecular property predictor using spectral graph wavelets
and curriculum learning on the PCQM4Mv2 dataset.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Set torch load defaults BEFORE importing torch to handle OGB dataset loading
# PyTorch 2.6+ changed weights_only default to True, which breaks OGB dataset loading
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

import torch
import numpy as np

# Configure PyTorch to allow numpy objects in torch.load for OGB datasets
# This is needed for PyTorch 2.6+ which changed weights_only default to True
try:
    # Add safe globals for numpy types that may appear in pickled OGB data
    import numpy.core.multiarray

    # Build list of safe numpy globals
    safe_globals_list = [
        numpy.core.multiarray._reconstruct,
        numpy.ndarray,
        numpy.dtype,
    ]

    # Add numpy dtypes for various numeric types
    try:
        safe_globals_list.extend([
            numpy.float64,
            numpy.float32,
            numpy.int64,
            numpy.int32,
            numpy.uint8,
        ])
    except AttributeError:
        pass

    # Filter out None values and add to safe globals
    safe_globals_list = [g for g in safe_globals_list if g is not None]
    torch.serialization.add_safe_globals(safe_globals_list)

except (ImportError, AttributeError) as e:
    # Fallback for different numpy/torch versions
    # This is safe to ignore as older versions don't have the restriction
    pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from spectral_temporal_curriculum_molecular_gaps.data.loader import PCQM4Mv2CurriculumDataLoader
from spectral_temporal_curriculum_molecular_gaps.models.model import SpectralTemporalMolecularNet
from spectral_temporal_curriculum_molecular_gaps.training.trainer import CurriculumTrainer
from spectral_temporal_curriculum_molecular_gaps.utils.config import load_config, validate_config


def setup_logging(log_dir: str, log_level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        log_dir: Directory for log files.
        log_level: Logging level.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Setup file handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"train_{int(time.time())}.log")
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(file_formatter)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Reduce noise from external libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def set_random_seeds(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: Whether to enable deterministic operations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for additional determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


def create_model(config: dict, device: torch.device) -> SpectralTemporalMolecularNet:
    """Create and initialize model.

    Args:
        config: Model configuration.
        device: Target device.

    Returns:
        Initialized model.
    """
    model_config = config["model"]

    model = SpectralTemporalMolecularNet(
        input_dim=model_config.get("input_dim", 9),
        hidden_dim=model_config.get("hidden_dim", 256),
        num_spectral_layers=model_config.get("num_spectral_layers", 4),
        num_scales=model_config.get("num_scales", 4),
        num_curriculum_stages=model_config.get("num_curriculum_stages", 4),
        dropout=model_config.get("dropout", 0.1),
        pool_type=model_config.get("pool_type", "attention"),
    )

    # Move to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"Model created with {total_params:,} total parameters")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    return model


def create_data_loader(config: dict) -> PCQM4Mv2CurriculumDataLoader:
    """Create curriculum data loader.

    Args:
        config: Data configuration.

    Returns:
        Curriculum data loader.
    """
    data_config = config["data"]
    training_config = config["training"]

    # Create data loader
    data_loader = PCQM4Mv2CurriculumDataLoader(
        root=data_config.get("root_dir", "./data"),
        batch_size=training_config.get("batch_size", 32),
        num_workers=training_config.get("num_workers", 4),
        curriculum_strategy=data_config.get("curriculum_strategy", "complexity_based"),
        spectral_decomp_levels=data_config.get("num_spectral_scales", 4),
        cache_dir=data_config.get("cache_dir"),
        force_reload=data_config.get("force_reload", False),
    )

    # Analyze complexity distribution
    complexity_stats = data_loader.analyze_complexity_distribution()
    logging.info("Molecular complexity distribution:")
    for key, value in complexity_stats.items():
        logging.info(f"  {key}: {value:.4f}")

    return data_loader


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Spectral Temporal Curriculum Molecular Network"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Override data directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID (None for auto-select)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without training"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        validate_config(config)
        logging.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.data_dir:
        config["data"]["root_dir"] = args.data_dir
    if args.checkpoint_dir:
        config["experiment"]["checkpoint_dir"] = args.checkpoint_dir

    # Debug mode adjustments
    if args.debug:
        config["training"]["num_epochs"] = 2
        config["training"]["batch_size"] = 4
        config["model"]["hidden_dim"] = 64
        config["training"]["log_frequency"] = 1
        logging.info("Debug mode enabled - using reduced parameters")

    # Setup logging
    log_dir = config["experiment"]["log_dir"]
    setup_logging(log_dir)

    # Set random seeds for reproducibility
    repro_config = config["reproducibility"]
    set_random_seeds(repro_config["seed"], repro_config["deterministic"])

    # Setup device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    try:
        # Create data loader
        logging.info("Creating curriculum data loader...")
        data_loader = create_data_loader(config)
        logging.info("Data loader created successfully")

        # Create model
        logging.info("Creating model...")
        model = create_model(config, device)
        logging.info("Model created successfully")

        # Create trainer
        logging.info("Creating trainer...")
        trainer = CurriculumTrainer(
            model=model,
            data_loader=data_loader,
            config=config,
            device=device,
            checkpoint_dir=config["experiment"]["checkpoint_dir"],
            log_dir=log_dir,
        )

        # Resume from checkpoint if specified
        if args.resume:
            if os.path.exists(args.resume):
                logging.info(f"Resuming from checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                trainer.current_epoch = checkpoint["epoch"] + 1
                trainer.best_val_loss = checkpoint["best_val_loss"]
                if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
                    trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logging.info(f"Resumed from epoch {trainer.current_epoch}")
            else:
                logging.error(f"Checkpoint file not found: {args.resume}")
                sys.exit(1)

        if args.dry_run:
            logging.info("Dry run completed successfully")
            return

        # Start training
        logging.info("Starting training...")
        start_time = time.time()

        final_metrics = trainer.train()

        end_time = time.time()
        training_time = end_time - start_time

        # Log final results
        logging.info("Training completed!")
        logging.info(f"Training time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        logging.info("Final metrics:")
        for metric, value in final_metrics.items():
            logging.info(f"  {metric}: {value:.4f}")

        # Save final results
        results_dir = config["experiment"]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)

        results_file = os.path.join(results_dir, f"final_results_{int(time.time())}.txt")
        with open(results_file, "w") as f:
            f.write("Spectral Temporal Curriculum Molecular Networks - Final Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Training time: {training_time:.2f} seconds\n")
            f.write(f"Total epochs: {trainer.current_epoch}\n")
            f.write(f"Best validation loss: {trainer.best_val_loss:.4f}\n")
            f.write("\nFinal Test Metrics:\n")
            for metric, value in final_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")

        logging.info(f"Results saved to {results_file}")

        # Compare with target metrics if available
        if "target_metrics" in config:
            target_metrics = config["target_metrics"]
            logging.info("\nComparison with target metrics:")

            test_mae = final_metrics.get("mae", float("inf"))
            target_mae = target_metrics.get("mae_homo_lumo_gap_ev", 0.075)
            mae_status = "✓" if test_mae <= target_mae else "✗"
            logging.info(f"  MAE: {test_mae:.4f} (target: {target_mae:.3f}) {mae_status}")

            ood_mae = final_metrics.get("ood_large_molecule_mae", float("inf"))
            target_ood = target_metrics.get("ood_large_molecule_mae", 0.12)
            ood_status = "✓" if ood_mae <= target_ood else "✗"
            logging.info(f"  OOD MAE: {ood_mae:.4f} (target: {target_ood:.3f}) {ood_status}")
        else:
            logging.info("No target metrics specified in configuration")

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()