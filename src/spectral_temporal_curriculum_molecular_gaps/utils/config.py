"""Configuration utilities for managing hyperparameters and settings."""

import logging
import os
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save configuration file.

    Raises:
        OSError: If unable to write to file.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)

        logger.info(f"Saved configuration to {config_path}")

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.

    Args:
        base_config: Base configuration dictionary.
        override_config: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If configuration is invalid.
    """
    required_keys = [
        'model', 'training', 'data', 'experiment', 'reproducibility'
    ]

    # Optional but commonly used sections
    optional_keys = ['evaluation', 'curriculum', 'target_metrics']

    # Check required top-level keys
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")

    # Validate model configuration
    model_config = config['model']
    model_required = ['hidden_dim', 'num_spectral_layers', 'num_scales']
    for key in model_required:
        if key not in model_config:
            raise ValueError(f"Missing required model parameter: {key}")

    # Validate training configuration
    training_config = config['training']
    training_required = ['num_epochs', 'learning_rate', 'batch_size']
    for key in training_required:
        if key not in training_config:
            raise ValueError(f"Missing required training parameter: {key}")

    # Validate data configuration
    data_config = config['data']
    data_required = ['root_dir']
    for key in data_required:
        if key not in data_config:
            raise ValueError(f"Missing required data parameter: {key}")

    # Validate value ranges
    if training_config['learning_rate'] <= 0 or training_config['learning_rate'] > 1:
        raise ValueError("Learning rate must be in (0, 1]")

    if training_config['batch_size'] <= 0:
        raise ValueError("Batch size must be positive")

    if training_config['num_epochs'] <= 0:
        raise ValueError("Number of epochs must be positive")

    if model_config['hidden_dim'] <= 0:
        raise ValueError("Hidden dimension must be positive")

    if model_config['num_spectral_layers'] <= 0:
        raise ValueError("Number of spectral layers must be positive")

    if model_config['num_scales'] <= 0:
        raise ValueError("Number of scales must be positive")

    # Validate evaluation configuration if present
    if 'evaluation' in config:
        evaluation_config = config['evaluation']
        if 'large_molecule_threshold' in evaluation_config:
            if evaluation_config['large_molecule_threshold'] <= 0:
                raise ValueError("Large molecule threshold must be positive")
        if 'chemical_accuracy_threshold' in evaluation_config:
            if evaluation_config['chemical_accuracy_threshold'] <= 0:
                raise ValueError("Chemical accuracy threshold must be positive")
        if 'bootstrap_samples' in evaluation_config:
            if evaluation_config['bootstrap_samples'] <= 0:
                raise ValueError("Bootstrap samples must be positive")
        if 'confidence_level' in evaluation_config:
            cl = evaluation_config['confidence_level']
            if cl <= 0 or cl >= 1:
                raise ValueError("Confidence level must be in (0, 1)")

    # Validate experiment configuration
    if 'experiment' in config:
        exp_config = config['experiment']
        exp_required = ['checkpoint_dir', 'log_dir']
        for key in exp_required:
            if key not in exp_config:
                raise ValueError(f"Missing required experiment parameter: {key}")

    # Validate reproducibility configuration
    if 'reproducibility' in config:
        repro_config = config['reproducibility']
        repro_required = ['seed']
        for key in repro_required:
            if key not in repro_config:
                raise ValueError(f"Missing required reproducibility parameter: {key}")

    logger.info("Configuration validation passed")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration parameters.

    Returns:
        Default configuration dictionary.
    """
    return {
        'model': {
            'input_dim': 9,
            'hidden_dim': 256,
            'num_spectral_layers': 4,
            'num_scales': 4,
            'num_curriculum_stages': 4,
            'dropout': 0.1,
            'pool_type': 'attention',
        },
        'training': {
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': 32,
            'num_workers': 4,
            'optimizer': 'adamw',
            'lr_scheduler': 'reduce_on_plateau',
            'lr_factor': 0.5,
            'lr_patience': 10,
            'min_lr': 0.00001,
            'grad_clip': 1.0,
            'early_stopping_patience': 20,
            'save_frequency': 10,
            'log_frequency': 100,
            'stage_loss_weight': 0.1,
        },
        'curriculum': {
            'strategy': 'linear',
            'warmup_epochs': 10,
        },
        'data': {
            'root_dir': './data',
            'num_spectral_scales': 4,
            'cache_dir': None,
            'force_reload': False,
            'curriculum_strategy': 'complexity_based',
        },
        'experiment': {
            'name': 'spectral_temporal_curriculum',
            'run_name': None,
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'results_dir': 'results',
        },
        'reproducibility': {
            'seed': 42,
            'deterministic': True,
        },
    }