"""Tests for training modules."""

import os
import tempfile
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from spectral_temporal_curriculum_molecular_gaps.training.trainer import CurriculumTrainer
from spectral_temporal_curriculum_molecular_gaps.evaluation.metrics import (
    MolecularPropertyMetrics,
    ConvergenceAnalyzer,
    AblationStudy,
)


class TestMolecularPropertyMetrics:
    """Test molecular property metrics calculation."""

    def test_initialization(self):
        """Test metrics calculator initialization."""
        metrics = MolecularPropertyMetrics()
        assert metrics.epsilon == 1e-8

    def test_compute_basic_metrics(self):
        """Test basic regression metrics computation."""
        metrics = MolecularPropertyMetrics()

        # Perfect predictions
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        results = metrics.compute_metrics(predictions, targets)

        assert results['mae'] == 0.0
        assert results['rmse'] == 0.0
        assert results['r2'] == 1.0
        assert results['pearson_r'] == 1.0

    def test_compute_metrics_with_noise(self):
        """Test metrics with noisy predictions."""
        np.random.seed(42)
        metrics = MolecularPropertyMetrics()

        targets = np.linspace(0, 10, 100)
        predictions = targets + np.random.normal(0, 0.1, 100)

        results = metrics.compute_metrics(predictions, targets)

        assert 0.0 < results['mae'] < 0.5
        assert 0.0 < results['rmse'] < 0.5
        assert 0.9 < results['r2'] < 1.0
        assert 0.9 < results['pearson_r'] < 1.0

    def test_chemical_accuracy_calculation(self):
        """Test chemical accuracy metric."""
        metrics = MolecularPropertyMetrics()

        # Half predictions within threshold, half outside
        predictions = np.array([1.0, 1.02, 1.05, 1.1])  # Errors: 0, 0.02, 0.05, 0.1
        targets = np.array([1.0, 1.0, 1.0, 1.0])

        results = metrics.compute_metrics(predictions, targets)

        # With threshold of 0.043 eV, first 2 should be within accuracy
        # But the 3rd (0.05) and 4th (0.1) are outside
        assert results['chemical_accuracy'] == 50.0  # 2 out of 4

    def test_error_distribution_stats(self):
        """Test error distribution statistics."""
        metrics = MolecularPropertyMetrics()

        predictions = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        results = metrics.compute_metrics(predictions, targets)

        assert 'error_mean' in results
        assert 'error_std' in results
        assert 'error_median' in results
        assert 'error_q25' in results
        assert 'error_q75' in results

    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        metrics = MolecularPropertyMetrics()

        # Sufficient data for bootstrap
        np.random.seed(42)
        targets = np.random.randn(100)
        predictions = targets + np.random.normal(0, 0.1, 100)

        results = metrics.compute_metrics(predictions, targets)

        assert 'mae_ci_lower' in results
        assert 'mae_ci_upper' in results
        assert results['mae_ci_lower'] < results['mae']
        assert results['mae'] < results['mae_ci_upper']

    def test_molecular_size_stratification(self):
        """Test molecular size stratified metrics."""
        metrics = MolecularPropertyMetrics()

        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.1, 1.9, 3.1, 3.9])
        molecular_sizes = np.array([5, 15, 25, 35])

        stratified = metrics.compute_molecular_size_stratified_metrics(
            predictions, targets, molecular_sizes,
            size_bins=[0, 20, 40]
        )

        assert 'size_0_20' in stratified
        assert 'size_20_40' in stratified
        assert stratified['size_0_20']['count'] == 2
        assert stratified['size_20_40']['count'] == 2

    def test_edge_cases(self):
        """Test edge cases in metrics computation."""
        metrics = MolecularPropertyMetrics()

        # Single data point
        predictions = np.array([1.0])
        targets = np.array([1.1])

        results = metrics.compute_metrics(predictions, targets)
        assert results['mae'] == 0.1
        assert not np.isnan(results['mae'])

        # Zero variance case
        predictions = np.array([1.0, 1.0, 1.0])
        targets = np.array([1.0, 1.0, 1.0])

        results = metrics.compute_metrics(predictions, targets)
        assert results['mae'] == 0.0
        assert results['rmse'] == 0.0


class TestConvergenceAnalyzer:
    """Test convergence analysis functionality."""

    def test_initialization(self):
        """Test convergence analyzer initialization."""
        analyzer = ConvergenceAnalyzer()
        assert analyzer is not None

    def test_convergence_speedup_analysis(self):
        """Test convergence speedup analysis."""
        analyzer = ConvergenceAnalyzer()

        # Simulate curriculum learning converging faster
        curriculum_history = {
            'val_mae': [0.5, 0.3, 0.15, 0.09, 0.08]  # Converges at epoch 3
        }

        baseline_history = {
            'val_mae': [0.5, 0.4, 0.3, 0.2, 0.15, 0.09, 0.08]  # Converges at epoch 5
        }

        results = analyzer.analyze_convergence_speedup(
            curriculum_history, baseline_history, target_mae=0.1
        )

        # Should detect speedup (5/3 ≈ 1.67)
        assert results['convergence_speedup_vs_baseline'] > 1.0
        assert 1.5 < results['convergence_speedup_vs_baseline'] < 2.0

    def test_curriculum_effectiveness_analysis(self):
        """Test curriculum effectiveness analysis."""
        analyzer = ConvergenceAnalyzer()

        # Simulate training history with curriculum learning
        training_history = {
            'val_mae': [0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.11, 0.10, 0.10],
            'curriculum_fraction': [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]
        }

        results = analyzer.analyze_curriculum_effectiveness(training_history)

        assert 'curriculum_phase_improvement' in results
        assert results['curriculum_phase_improvement'] > 0  # Should show improvement

    def test_epochs_to_target_helper(self):
        """Test helper function for finding epochs to target."""
        analyzer = ConvergenceAnalyzer()

        mae_history = [0.5, 0.3, 0.15, 0.08, 0.07]
        epochs = analyzer._epochs_to_target(mae_history, 0.1)

        assert epochs == 3  # First epoch to reach <= 0.1

        # Test case where target is not reached
        epochs_not_reached = analyzer._epochs_to_target(mae_history, 0.05)
        assert epochs_not_reached == -1


class TestAblationStudy:
    """Test ablation study functionality."""

    def test_initialization(self):
        """Test ablation study initialization."""
        study = AblationStudy()
        assert isinstance(study.metrics_calculator, MolecularPropertyMetrics)

    def test_spectral_scales_impact_analysis(self):
        """Test spectral scales impact analysis."""
        study = AblationStudy()

        # Simulate results with different numbers of scales
        results_by_scales = {
            1: {'mae': 0.15, 'r2': 0.85},
            2: {'mae': 0.12, 'r2': 0.88},
            4: {'mae': 0.10, 'r2': 0.90},  # Best
            8: {'mae': 0.11, 'r2': 0.89},
        }

        analysis = study.analyze_spectral_scales_impact(results_by_scales)

        assert analysis['optimal_num_scales'] == 4
        assert analysis['best_mae'] == 0.10
        assert analysis['improvement_over_single_scale'] > 0

    def test_curriculum_strategies_analysis(self):
        """Test curriculum strategies analysis."""
        study = AblationStudy()

        results_by_strategy = {
            'none': {'mae': 0.15, 'r2': 0.85},
            'linear': {'mae': 0.12, 'r2': 0.88},
            'exponential': {'mae': 0.10, 'r2': 0.90},  # Best
            'cosine': {'mae': 0.11, 'r2': 0.89},
        }

        analysis = study.analyze_curriculum_strategies(results_by_strategy)

        assert analysis['optimal_strategy'] == 'exponential'
        assert analysis['best_mae'] == 0.10
        assert analysis['improvement_over_no_curriculum'] > 0

    def test_statistical_significance_test(self):
        """Test statistical significance testing."""
        study = AblationStudy()

        np.random.seed(42)
        targets = np.random.randn(100)

        # Model A: slightly better
        predictions_a = targets + np.random.normal(0, 0.1, 100)

        # Model B: slightly worse
        predictions_b = targets + np.random.normal(0, 0.15, 100)

        results = study.statistical_significance_test(
            predictions_a, predictions_b, targets
        )

        assert 'p_value' in results
        assert 'cohens_d' in results
        assert 'significant_at_0_05' in results
        assert results['mean_error_a'] < results['mean_error_b']  # A should be better

    def test_ablation_report_generation(self):
        """Test ablation report generation."""
        study = AblationStudy()

        ablation_results = {
            'spectral_scales': {
                'optimal_num_scales': 4,
                'best_mae': 0.10,
                'improvement_over_single_scale': 25.0,
                'analysis': {
                    1: {'mae': 0.15, 'r2': 0.85},
                    4: {'mae': 0.10, 'r2': 0.90},
                }
            },
            'curriculum_strategies': {
                'optimal_strategy': 'exponential',
                'best_mae': 0.10,
                'improvement_over_no_curriculum': 20.0,
                'analysis': {
                    'none': {'mae': 0.15, 'r2': 0.85},
                    'exponential': {'mae': 0.10, 'r2': 0.90},
                }
            }
        }

        report = study.generate_ablation_report(ablation_results)

        assert "Ablation Study Report" in report
        assert "optimal_num_scales: 4" in report
        assert "optimal_strategy: exponential" in report
        assert isinstance(report, str)
        assert len(report) > 0


class TestCurriculumTrainerIntegration:
    """Integration tests for curriculum trainer."""

    @patch('spectral_temporal_curriculum_molecular_gaps.training.trainer.mlflow')
    def test_mlflow_integration_with_fallback(self, mock_mlflow, sample_model, sample_config, temp_dir):
        """Test MLflow integration with graceful fallback."""
        # Mock MLflow to raise an exception
        mock_mlflow.start_run.side_effect = Exception("MLflow not available")

        # Create mock data loader
        mock_data_loader = Mock()
        mock_data_loader.get_curriculum_dataloader.return_value = []
        mock_data_loader.get_validation_dataloader.return_value = []

        trainer = CurriculumTrainer(
            model=sample_model,
            data_loader=mock_data_loader,
            config=sample_config,
            device=torch.device('cpu'),
            checkpoint_dir=temp_dir
        )

        # Should not raise exception even if MLflow fails
        assert trainer is not None

    def test_optimizer_setup(self, sample_model, sample_config, temp_dir):
        """Test optimizer setup with different configurations."""
        mock_data_loader = Mock()

        # Test AdamW optimizer
        sample_config['training']['optimizer'] = 'adamw'
        trainer = CurriculumTrainer(
            model=sample_model,
            data_loader=mock_data_loader,
            config=sample_config,
            device=torch.device('cpu'),
            checkpoint_dir=temp_dir
        )
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

        # Test SGD optimizer
        sample_config['training']['optimizer'] = 'sgd'
        sample_config['training']['momentum'] = 0.9
        trainer = CurriculumTrainer(
            model=sample_model,
            data_loader=mock_data_loader,
            config=sample_config,
            device=torch.device('cpu'),
            checkpoint_dir=temp_dir
        )
        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_scheduler_setup(self, sample_model, sample_config, temp_dir):
        """Test learning rate scheduler setup."""
        mock_data_loader = Mock()

        # Test ReduceLROnPlateau
        sample_config['training']['lr_scheduler'] = 'reduce_on_plateau'
        trainer = CurriculumTrainer(
            model=sample_model,
            data_loader=mock_data_loader,
            config=sample_config,
            device=torch.device('cpu'),
            checkpoint_dir=temp_dir
        )
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        # Test CosineAnnealingLR
        sample_config['training']['lr_scheduler'] = 'cosine'
        trainer = CurriculumTrainer(
            model=sample_model,
            data_loader=mock_data_loader,
            config=sample_config,
            device=torch.device('cpu'),
            checkpoint_dir=temp_dir
        )
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

        # Test no scheduler
        sample_config['training']['lr_scheduler'] = 'none'
        trainer = CurriculumTrainer(
            model=sample_model,
            data_loader=mock_data_loader,
            config=sample_config,
            device=torch.device('cpu'),
            checkpoint_dir=temp_dir
        )
        assert trainer.scheduler is None

    def test_checkpoint_saving_and_loading(self, sample_model, sample_config, temp_dir):
        """Test checkpoint saving and loading."""
        mock_data_loader = Mock()

        trainer = CurriculumTrainer(
            model=sample_model,
            data_loader=mock_data_loader,
            config=sample_config,
            device=torch.device('cpu'),
            checkpoint_dir=temp_dir
        )

        # Save checkpoint
        trainer.current_epoch = 5
        trainer.best_val_loss = 0.15
        trainer._save_checkpoint(5, is_best=True)

        # Check files exist
        checkpoint_path = os.path.join(temp_dir, 'checkpoint_epoch_5.pt')
        best_path = os.path.join(temp_dir, 'best_model.pt')

        assert os.path.exists(checkpoint_path)
        assert os.path.exists(best_path)

        # Load checkpoint
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 5
        assert checkpoint['best_val_loss'] == 0.15

    def test_training_history_tracking(self, sample_model, sample_config, temp_dir):
        """Test training history tracking."""
        mock_data_loader = Mock()

        trainer = CurriculumTrainer(
            model=sample_model,
            data_loader=mock_data_loader,
            config=sample_config,
            device=torch.device('cpu'),
            checkpoint_dir=temp_dir
        )

        # Simulate logging metrics
        train_metrics = {'loss': 0.5, 'mae': 0.3}
        val_metrics = {'loss': 0.4, 'mae': 0.25}

        trainer._log_epoch_metrics(0, train_metrics, val_metrics, 0.5, 0.001)

        assert len(trainer.training_history['train_loss']) == 1
        assert len(trainer.training_history['val_loss']) == 1
        assert trainer.training_history['train_loss'][0] == 0.5
        assert trainer.training_history['val_loss'][0] == 0.4

    def test_error_handling(self, sample_model, sample_config, temp_dir):
        """Test error handling in trainer."""
        mock_data_loader = Mock()

        # Test with invalid optimizer
        sample_config['training']['optimizer'] = 'invalid_optimizer'

        with pytest.raises(ValueError, match="Unknown optimizer"):
            CurriculumTrainer(
                model=sample_model,
                data_loader=mock_data_loader,
                config=sample_config,
                device=torch.device('cpu'),
                checkpoint_dir=temp_dir
            )