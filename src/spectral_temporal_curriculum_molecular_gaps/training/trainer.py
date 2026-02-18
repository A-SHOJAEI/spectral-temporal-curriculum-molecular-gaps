"""Training pipeline with curriculum learning and MLflow integration."""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np

# MLflow imports with error handling
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ..data.loader import PCQM4Mv2CurriculumDataLoader
from ..data.preprocessing import SpectralFeatureExtractor, CurriculumScheduler
from ..models.model import SpectralTemporalMolecularNet
from ..evaluation.metrics import MolecularPropertyMetrics

logger = logging.getLogger(__name__)


class CurriculumTrainer:
    """Trainer for spectral temporal molecular networks with curriculum learning."""

    def __init__(
        self,
        model: SpectralTemporalMolecularNet,
        data_loader: PCQM4Mv2CurriculumDataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ) -> None:
        """Initialize curriculum trainer.

        Args:
            model: Spectral temporal molecular network.
            data_loader: Curriculum data loader.
            config: Training configuration dictionary.
            device: Training device (CPU/GPU).
            checkpoint_dir: Directory for saving checkpoints.
            log_dir: Directory for logging.
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_curriculum()
        self._setup_metrics()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'val_mae': [],
            'curriculum_fraction': [], 'learning_rate': []
        }

        # Initialize MLflow if available
        self._setup_mlflow()

    def _setup_optimizer(self) -> None:
        """Setup optimizer based on configuration."""
        training_config = self.config.get('training', {})
        optimizer_name = training_config.get('optimizer', 'adamw')
        lr = training_config.get('learning_rate', 0.001)
        weight_decay = training_config.get('weight_decay', 0.01)

        if optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            momentum = training_config.get('momentum', 0.9)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        logger.info(f"Initialized {optimizer_name} optimizer with lr={lr}")

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        training_config = self.config.get('training', {})
        scheduler_type = training_config.get('lr_scheduler', 'reduce_on_plateau')

        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=training_config.get('lr_factor', 0.5),
                patience=training_config.get('lr_patience', 10),
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('num_epochs', 100),
                eta_min=training_config.get('min_lr', 0.00001)
            )
        else:
            self.scheduler = None

    def _setup_curriculum(self) -> None:
        """Setup curriculum learning components."""
        curriculum_config = self.config.get('curriculum', {})
        training_config = self.config.get('training', {})

        self.curriculum_scheduler = CurriculumScheduler(
            strategy=curriculum_config.get('strategy', 'linear'),
            warmup_epochs=curriculum_config.get('warmup_epochs', 10),
            total_epochs=training_config.get('num_epochs', 100)
        )

        data_config = self.config.get('data', {})
        self.spectral_extractor = SpectralFeatureExtractor(
            num_levels=data_config.get('num_spectral_scales', 4)
        )

    def _setup_metrics(self) -> None:
        """Setup evaluation metrics."""
        self.metrics = MolecularPropertyMetrics()

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, skipping tracking setup")
            return

        try:
            # Set experiment
            experiment_config = self.config.get('experiment', {})
            experiment_name = experiment_config.get('name', 'spectral_temporal_curriculum')
            mlflow.set_experiment(experiment_name)

            # Start run
            run_name = experiment_config.get('run_name', f"run_{int(time.time())}")
            mlflow.start_run(run_name=run_name)

            # Log configuration
            mlflow.log_params(self.config)

            logger.info(f"MLflow tracking initialized for experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    def train(self) -> Dict[str, float]:
        """Main training loop with curriculum learning.

        Returns:
            Final training metrics.
        """
        logger.info("Starting curriculum training...")

        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 100)
        early_stopping_patience = training_config.get('early_stopping_patience', 20)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Get curriculum fraction for current epoch
            curriculum_fraction = self.curriculum_scheduler.get_curriculum_fraction(epoch)
            logger.info(f"Epoch {epoch}: curriculum fraction = {curriculum_fraction:.3f}")

            # Train one epoch
            train_metrics = self._train_epoch(curriculum_fraction)

            # Validate
            val_metrics = self._validate_epoch()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self._log_epoch_metrics(epoch, train_metrics, val_metrics,
                                  curriculum_fraction, current_lr)

            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            # Save regular checkpoint
            training_config = self.config.get('training', {})
            if epoch % training_config.get('save_frequency', 10) == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        logger.info("Training completed!")

        # Final evaluation
        final_metrics = self._final_evaluation()

        # Cleanup MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except Exception:
                pass

        return final_metrics

    def _train_epoch(self, curriculum_fraction: float) -> Dict[str, float]:
        """Train one epoch with curriculum learning.

        Args:
            curriculum_fraction: Fraction of curriculum data to use.

        Returns:
            Training metrics for the epoch.
        """
        self.model.train()
        epoch_losses = []
        epoch_maes = []

        # Get curriculum dataloader
        train_loader = self.data_loader.get_curriculum_dataloader(
            curriculum_fraction=curriculum_fraction,
            shuffle=True
        )

        # Determine curriculum stage based on fraction
        if curriculum_fraction <= 0.25:
            curriculum_stage = 0
        elif curriculum_fraction <= 0.5:
            curriculum_stage = 1
        elif curriculum_fraction <= 0.75:
            curriculum_stage = 2
        else:
            curriculum_stage = 3
        self._current_curriculum_stage = curriculum_stage

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            targets = batch.y.float()

            # Extract spectral features for the batch
            spectral_features = self._extract_batch_spectral_features(batch)

            # Forward pass
            outputs = self.model(
                batch,
                spectral_features,
                curriculum_stage=curriculum_stage
            )

            # Compute losses
            training_config = self.config.get('training', {})
            losses = self.model.compute_loss(
                outputs,
                targets,
                curriculum_stage=curriculum_stage,
                stage_weight=training_config.get('stage_loss_weight', 0.1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                training_config.get('grad_clip', 1.0)
            )

            self.optimizer.step()

            # Track metrics
            epoch_losses.append(losses['total'].item())
            mae = torch.abs(outputs['prediction'].squeeze() - targets).mean().item()
            epoch_maes.append(mae)

            # Log batch metrics periodically
            if batch_idx % training_config.get('log_frequency', 100) == 0:
                logger.info(
                    f"Batch {batch_idx}: loss={losses['total'].item():.4f}, "
                    f"mae={mae:.4f}, stage={curriculum_stage}"
                )

        return {
            'loss': np.mean(epoch_losses),
            'mae': np.mean(epoch_maes),
            'curriculum_stage': curriculum_stage
        }

    def _validate_epoch(self) -> Dict[str, float]:
        """Validate model performance.

        Returns:
            Validation metrics.
        """
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []

        val_loader = self.data_loader.get_validation_dataloader()

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                targets = batch.y.float()

                # Extract spectral features
                spectral_features = self._extract_batch_spectral_features(batch)

                # Forward pass with current curriculum stage
                curriculum_stage = getattr(self, '_current_curriculum_stage', None)
                outputs = self.model(batch, spectral_features, curriculum_stage=curriculum_stage)

                # Compute loss
                losses = self.model.compute_loss(outputs, targets)
                val_losses.append(losses['total'].item())

                # Collect predictions and targets
                predictions = outputs['prediction'].squeeze().cpu().numpy()
                all_predictions.extend(predictions if predictions.ndim > 0 else [predictions])
                all_targets.extend(targets.cpu().numpy())

        # Compute comprehensive metrics
        val_metrics = self.metrics.compute_metrics(
            np.array(all_predictions),
            np.array(all_targets)
        )
        val_metrics['loss'] = np.mean(val_losses)

        return val_metrics

    def _extract_batch_spectral_features(self, batch) -> List[torch.Tensor]:
        """Extract spectral features for a batch of graphs using graph Laplacian eigendecomposition.

        Args:
            batch: Batch of molecular graphs.

        Returns:
            List of spectral features at different scales.

        Raises:
            RuntimeError: If eigendecomposition fails for all graphs in batch.
        """
        import torch_geometric.utils as pyg_utils
        from scipy.sparse.linalg import eigsh
        import scipy.sparse as sp

        data_config = self.config.get('data', {})
        num_scales = data_config.get('num_spectral_scales', 4)
        feature_dim = batch.x.size(1)
        device = batch.x.device

        # Split batch into individual graphs
        graphs = [batch.get_example(i) for i in range(batch.num_graphs)]
        spectral_features = [[] for _ in range(num_scales)]

        for graph in graphs:
            try:
                # Convert edge_index to adjacency matrix
                edge_index = graph.edge_index
                num_nodes = graph.num_nodes

                # Create normalized Laplacian matrix
                adj = pyg_utils.to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

                # Handle isolated nodes
                if adj.nnz == 0:
                    # For isolated nodes, use identity-based features
                    eigenvals = torch.ones(min(num_scales, num_nodes), device=device)
                    eigenvecs = torch.eye(num_nodes, min(num_scales, num_nodes), device=device)
                else:
                    # Compute degree matrix
                    degrees = torch.tensor(adj.sum(axis=1).A1, dtype=torch.float32)
                    degrees[degrees == 0] = 1.0  # Avoid division by zero

                    # Create normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
                    deg_sqrt_inv = sp.diags(1.0 / torch.sqrt(degrees).numpy())
                    laplacian = sp.eye(num_nodes) - deg_sqrt_inv @ adj @ deg_sqrt_inv

                    # Compute smallest eigenvalues and eigenvectors
                    k = min(num_scales, num_nodes - 1)
                    if k > 0:
                        eigenvals_np, eigenvecs_np = eigsh(laplacian, k=k, which='SM')
                        eigenvals = torch.tensor(eigenvals_np, dtype=torch.float32, device=device)
                        eigenvecs = torch.tensor(eigenvecs_np, dtype=torch.float32, device=device)
                    else:
                        eigenvals = torch.ones(1, device=device)
                        eigenvecs = torch.ones(num_nodes, 1, device=device) / torch.sqrt(torch.tensor(num_nodes, dtype=torch.float32))

                # Extract spectral features at different scales
                node_features = graph.x  # [num_nodes, feature_dim]

                for scale_idx in range(num_scales):
                    if scale_idx < eigenvecs.size(1):
                        # Use actual eigenvector for this scale
                        eigenvec = eigenvecs[:, scale_idx:scale_idx+1]  # [num_nodes, 1]
                        spectral_weight = torch.exp(-eigenvals[scale_idx] * (scale_idx + 1))

                        # Project node features onto eigenvector space
                        scale_features = node_features * eigenvec * spectral_weight
                    else:
                        # Fallback for scales beyond available eigenvectors
                        scale_features = node_features * (1.0 / (2 ** scale_idx))

                    spectral_features[scale_idx].append(scale_features)

            except Exception as e:
                logger.warning(f"Spectral decomposition failed for graph: {e}. Using fallback.")
                # Fallback to simple scaling
                node_features = graph.x
                for scale_idx in range(num_scales):
                    scale_factor = 1.0 / (2 ** scale_idx)
                    scale_features = node_features * scale_factor
                    spectral_features[scale_idx].append(scale_features)

        # Concatenate features for each scale across all graphs in batch
        final_spectral_features = []
        for scale_idx in range(num_scales):
            if spectral_features[scale_idx]:
                scale_tensor = torch.cat(spectral_features[scale_idx], dim=0)
                final_spectral_features.append(scale_tensor)
            else:
                # Emergency fallback
                fallback_tensor = torch.zeros_like(batch.x)
                final_spectral_features.append(fallback_tensor)

        return final_spectral_features

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        curriculum_fraction: float,
        learning_rate: float
    ) -> None:
        """Log metrics for current epoch.

        Args:
            epoch: Current epoch number.
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
            curriculum_fraction: Current curriculum fraction.
            learning_rate: Current learning rate.
        """
        # Update training history
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['val_mae'].append(val_metrics['mae'])
        self.training_history['curriculum_fraction'].append(curriculum_fraction)
        self.training_history['learning_rate'].append(learning_rate)

        # Log to console
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_mae={val_metrics['mae']:.4f}, "
            f"lr={learning_rate:.6f}"
        )

        # Log to MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_mae': val_metrics['mae'],
                    'curriculum_fraction': curriculum_fraction,
                    'learning_rate': learning_rate,
                }, step=epoch)
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best checkpoint so far.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved at epoch {epoch}")

            # Log model to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.pytorch.log_model(self.model, "best_model")
                except Exception as e:
                    logger.warning(f"MLflow model logging failed: {e}")

    def _final_evaluation(self) -> Dict[str, float]:
        """Perform final evaluation on test set.

        Returns:
            Final evaluation metrics.
        """
        logger.info("Performing final evaluation...")

        # Load best model
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded best model for final evaluation")

        self.model.eval()
        test_loader = self.data_loader.get_test_dataloader()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                targets = batch.y.float()

                spectral_features = self._extract_batch_spectral_features(batch)
                outputs = self.model(batch, spectral_features)

                predictions = outputs['prediction'].squeeze().cpu().numpy()
                all_predictions.extend(predictions if predictions.ndim > 0 else [predictions])
                all_targets.extend(targets.cpu().numpy())

        # Compute final metrics
        final_metrics = self.metrics.compute_metrics(
            np.array(all_predictions),
            np.array(all_targets)
        )

        # Evaluate on large molecules (OOD)
        ood_metrics = self._evaluate_ood_large_molecules()
        final_metrics['ood_large_molecule_mae'] = ood_metrics['mae']

        logger.info(f"Final test MAE: {final_metrics['mae']:.4f}")
        logger.info(f"OOD large molecule MAE: {final_metrics['ood_large_molecule_mae']:.4f}")

        # Log final metrics to MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    'final_test_mae': final_metrics['mae'],
                    'final_test_rmse': final_metrics['rmse'],
                    'final_ood_mae': final_metrics['ood_large_molecule_mae'],
                })
            except Exception:
                pass

        return final_metrics

    def _evaluate_ood_large_molecules(self) -> Dict[str, float]:
        """Evaluate on out-of-distribution large molecules.

        Returns:
            OOD evaluation metrics.
        """
        self.model.eval()
        ood_loader = self.data_loader.get_ood_large_molecule_subset()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in ood_loader:
                batch = batch.to(self.device)
                targets = batch.y.float()

                spectral_features = self._extract_batch_spectral_features(batch)
                outputs = self.model(batch, spectral_features)

                predictions = outputs['prediction'].squeeze().cpu().numpy()
                all_predictions.extend(predictions if predictions.ndim > 0 else [predictions])
                all_targets.extend(targets.cpu().numpy())

        if len(all_predictions) > 0:
            return self.metrics.compute_metrics(
                np.array(all_predictions),
                np.array(all_targets)
            )
        else:
            logger.warning("No large molecules found for OOD evaluation")
            return {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}