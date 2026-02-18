"""Evaluation metrics and analysis tools for molecular property prediction."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class MolecularPropertyMetrics:
    """Comprehensive metrics for molecular property prediction evaluation."""

    def __init__(self) -> None:
        """Initialize molecular property metrics calculator."""
        self.epsilon = 1e-8  # Small constant to avoid division by zero

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            sample_weights: Optional sample weights.

        Returns:
            Dictionary containing various evaluation metrics.
        """
        # Flatten arrays to ensure proper shape
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Filter out NaN/Inf predictions
        valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
        if not np.all(valid_mask):
            n_invalid = int((~valid_mask).sum())
            import logging
            logging.warning(f"Filtered {n_invalid}/{len(pred_flat)} NaN/Inf predictions")
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]

        if sample_weights is not None:
            weights_flat = sample_weights.flatten()
            if not np.all(valid_mask):
                weights_flat = weights_flat[valid_mask]
        else:
            weights_flat = None

        metrics = {}

        # Basic regression metrics
        metrics['mae'] = self._compute_mae(pred_flat, target_flat, weights_flat)
        metrics['rmse'] = self._compute_rmse(pred_flat, target_flat, weights_flat)
        metrics['r2'] = self._compute_r2(pred_flat, target_flat, weights_flat)

        # Chemistry-specific metrics
        metrics['mape'] = self._compute_mape(pred_flat, target_flat, weights_flat)
        metrics['chemical_accuracy'] = self._compute_chemical_accuracy(pred_flat, target_flat)

        # Distribution-based metrics
        metrics['pearson_r'] = self._compute_pearson_correlation(pred_flat, target_flat)
        metrics['spearman_r'] = self._compute_spearman_correlation(pred_flat, target_flat)

        # Error distribution analysis
        error_stats = self._compute_error_distribution(pred_flat, target_flat)
        metrics.update(error_stats)

        # Confidence intervals
        confidence_metrics = self._compute_confidence_intervals(pred_flat, target_flat)
        metrics.update(confidence_metrics)

        return metrics

    def _compute_mae(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Compute Mean Absolute Error."""
        return float(mean_absolute_error(targets, predictions, sample_weight=weights))

    def _compute_rmse(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Compute Root Mean Square Error."""
        mse = mean_squared_error(targets, predictions, sample_weight=weights)
        return float(np.sqrt(mse))

    def _compute_r2(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Compute R-squared coefficient of determination."""
        return float(r2_score(targets, predictions, sample_weight=weights))

    def _compute_mape(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Compute Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = np.abs(targets) > self.epsilon
        if not np.any(mask):
            return 0.0

        targets_safe = targets[mask]
        predictions_safe = predictions[mask]

        if weights is not None:
            weights_safe = weights[mask]
        else:
            weights_safe = None

        ape = np.abs((targets_safe - predictions_safe) / targets_safe) * 100

        if weights_safe is not None:
            return float(np.average(ape, weights=weights_safe))
        else:
            return float(np.mean(ape))

    def _compute_chemical_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.043  # 1 kcal/mol in eV
    ) -> float:
        """Compute chemical accuracy (percentage within threshold).

        Args:
            predictions: Model predictions in eV.
            targets: Ground truth targets in eV.
            threshold: Accuracy threshold in eV (default: 1 kcal/mol).

        Returns:
            Percentage of predictions within chemical accuracy threshold.
        """
        errors = np.abs(predictions - targets)
        within_threshold = np.sum(errors <= threshold)
        return float(within_threshold / len(errors) * 100)

    def _compute_pearson_correlation(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Compute Pearson correlation coefficient."""
        if len(predictions) < 2:
            return 0.0
        correlation, _ = stats.pearsonr(predictions, targets)
        return float(correlation)

    def _compute_spearman_correlation(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Compute Spearman rank correlation coefficient."""
        if len(predictions) < 2:
            return 0.0
        correlation, _ = stats.spearmanr(predictions, targets)
        return float(correlation)

    def _compute_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute error distribution statistics."""
        errors = predictions - targets
        abs_errors = np.abs(errors)

        return {
            'error_mean': float(np.mean(errors)),
            'error_std': float(np.std(errors)),
            'error_median': float(np.median(errors)),
            'error_q25': float(np.percentile(abs_errors, 25)),
            'error_q75': float(np.percentile(abs_errors, 75)),
            'error_q95': float(np.percentile(abs_errors, 95)),
            'max_error': float(np.max(abs_errors)),
        }

    def _compute_confidence_intervals(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """Compute confidence intervals for metrics."""
        n = len(predictions)
        if n < 10:
            return {'mae_ci_lower': 0.0, 'mae_ci_upper': 0.0}

        # Bootstrap confidence intervals for MAE
        n_bootstrap = 1000
        bootstrap_maes = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            boot_pred = predictions[indices]
            boot_target = targets[indices]

            boot_mae = mean_absolute_error(boot_target, boot_pred)
            bootstrap_maes.append(boot_mae)

        alpha = 1.0 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        return {
            'mae_ci_lower': float(np.percentile(bootstrap_maes, lower_percentile)),
            'mae_ci_upper': float(np.percentile(bootstrap_maes, upper_percentile)),
        }

    def compute_molecular_size_stratified_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        molecular_sizes: np.ndarray,
        size_bins: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics stratified by molecular size.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            molecular_sizes: Number of atoms in each molecule.
            size_bins: Size bin boundaries.

        Returns:
            Dictionary of metrics for each size bin.
        """
        if size_bins is None:
            size_bins = [0, 20, 40, 60, 80, 1000]  # Default bins

        stratified_metrics = {}

        for i in range(len(size_bins) - 1):
            min_size = size_bins[i]
            max_size = size_bins[i + 1]

            # Create mask for current size bin
            mask = (molecular_sizes >= min_size) & (molecular_sizes < max_size)
            bin_name = f"size_{min_size}_{max_size}"

            if np.sum(mask) > 0:
                bin_predictions = predictions[mask]
                bin_targets = targets[mask]

                stratified_metrics[bin_name] = self.compute_metrics(
                    bin_predictions, bin_targets
                )
                stratified_metrics[bin_name]['count'] = int(np.sum(mask))
            else:
                stratified_metrics[bin_name] = {'count': 0}

        return stratified_metrics


class ConvergenceAnalyzer:
    """Analyze training convergence and curriculum learning effectiveness."""

    def __init__(self) -> None:
        """Initialize convergence analyzer."""
        pass

    def analyze_convergence_speedup(
        self,
        curriculum_history: Dict[str, List[float]],
        baseline_history: Dict[str, List[float]],
        target_mae: float = 0.1
    ) -> Dict[str, float]:
        """Analyze convergence speedup compared to baseline.

        Args:
            curriculum_history: Training history with curriculum learning.
            baseline_history: Training history without curriculum learning.
            target_mae: Target MAE for convergence analysis.

        Returns:
            Convergence analysis metrics.
        """
        metrics = {}

        # Find epochs to reach target MAE
        curriculum_epochs = self._epochs_to_target(
            curriculum_history['val_mae'], target_mae
        )
        baseline_epochs = self._epochs_to_target(
            baseline_history['val_mae'], target_mae
        )

        if curriculum_epochs > 0 and baseline_epochs > 0:
            speedup = baseline_epochs / curriculum_epochs
            metrics['convergence_speedup_vs_baseline'] = speedup
        else:
            metrics['convergence_speedup_vs_baseline'] = 1.0

        # Analyze final performance
        final_curriculum_mae = curriculum_history['val_mae'][-1]
        final_baseline_mae = baseline_history['val_mae'][-1]

        metrics['final_mae_improvement'] = (
            (final_baseline_mae - final_curriculum_mae) / final_baseline_mae * 100
        )

        # Analyze stability (variance in last 10 epochs)
        if len(curriculum_history['val_mae']) >= 10:
            curriculum_stability = np.std(curriculum_history['val_mae'][-10:])
            baseline_stability = np.std(baseline_history['val_mae'][-10:])

            metrics['stability_improvement'] = (
                (baseline_stability - curriculum_stability) / baseline_stability * 100
            )

        return metrics

    def _epochs_to_target(self, mae_history: List[float], target_mae: float) -> int:
        """Find number of epochs to reach target MAE."""
        for epoch, mae in enumerate(mae_history):
            if mae <= target_mae:
                return epoch + 1
        return -1  # Target not reached

    def analyze_curriculum_effectiveness(
        self,
        training_history: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Analyze effectiveness of curriculum learning strategy.

        Args:
            training_history: Training history with curriculum fractions.

        Returns:
            Curriculum effectiveness metrics.
        """
        metrics = {}

        val_mae = training_history['val_mae']
        curriculum_fractions = training_history['curriculum_fraction']

        # Analyze correlation between curriculum progress and improvement
        if len(val_mae) > 10:
            # Compute improvement rate during curriculum phase
            warmup_end = None
            for i, frac in enumerate(curriculum_fractions):
                if frac >= 0.99:  # Nearly full dataset
                    warmup_end = i
                    break

            if warmup_end is not None and warmup_end > 5:
                warmup_mae = val_mae[:warmup_end]
                post_warmup_mae = val_mae[warmup_end:]

                # Improvement during warmup
                warmup_improvement = (warmup_mae[0] - warmup_mae[-1]) / warmup_mae[0]
                metrics['curriculum_phase_improvement'] = warmup_improvement * 100

                # Compare improvement rates
                if len(post_warmup_mae) > 5:
                    post_warmup_improvement = (
                        post_warmup_mae[0] - post_warmup_mae[-1]
                    ) / post_warmup_mae[0]
                    metrics['post_curriculum_improvement'] = post_warmup_improvement * 100

        return metrics


class AblationStudy:
    """Perform ablation studies on model components."""

    def __init__(self) -> None:
        """Initialize ablation study analyzer."""
        self.metrics_calculator = MolecularPropertyMetrics()

    def analyze_spectral_scales_impact(
        self,
        results_by_scales: Dict[int, Dict[str, float]]
    ) -> Dict[str, Union[int, float]]:
        """Analyze impact of different numbers of spectral scales.

        Args:
            results_by_scales: Results dictionary keyed by number of scales.

        Returns:
            Analysis of spectral scales impact.
        """
        best_mae = float('inf')
        best_scales = 1

        for num_scales, metrics in results_by_scales.items():
            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_scales = num_scales

        # Compute improvement over single scale
        single_scale_mae = results_by_scales.get(1, {}).get('mae', best_mae)
        improvement = (single_scale_mae - best_mae) / single_scale_mae * 100

        return {
            'optimal_num_scales': best_scales,
            'best_mae': best_mae,
            'improvement_over_single_scale': improvement,
            'scales_analysis': results_by_scales
        }

    def analyze_curriculum_strategies(
        self,
        results_by_strategy: Dict[str, Dict[str, float]]
    ) -> Dict[str, Union[str, float, Dict]]:
        """Analyze different curriculum learning strategies.

        Args:
            results_by_strategy: Results dictionary keyed by strategy name.

        Returns:
            Analysis of curriculum strategies.
        """
        best_mae = float('inf')
        best_strategy = 'none'

        for strategy, metrics in results_by_strategy.items():
            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_strategy = strategy

        # Compare against no curriculum
        no_curriculum_mae = results_by_strategy.get('none', {}).get('mae', best_mae)
        improvement = (no_curriculum_mae - best_mae) / no_curriculum_mae * 100

        return {
            'optimal_strategy': best_strategy,
            'best_mae': best_mae,
            'improvement_over_no_curriculum': improvement,
            'strategy_analysis': results_by_strategy
        }

    def generate_ablation_report(
        self,
        ablation_results: Dict[str, Dict]
    ) -> str:
        """Generate comprehensive ablation study report.

        Args:
            ablation_results: Dictionary containing all ablation results.

        Returns:
            Formatted report string.
        """
        report_lines = [
            "# Ablation Study Report",
            "",
            "## Spectral Scales Analysis",
        ]

        if 'spectral_scales' in ablation_results:
            scales_results = ablation_results['spectral_scales']
            report_lines.extend([
                f"- Optimal number of scales: {scales_results['optimal_num_scales']}",
                f"- Best MAE achieved: {scales_results['best_mae']:.4f}",
                f"- Improvement over single scale: {scales_results['improvement_over_single_scale']:.2f}%",
                ""
            ])

        if 'curriculum_strategies' in ablation_results:
            curriculum_results = ablation_results['curriculum_strategies']
            report_lines.extend([
                "## Curriculum Learning Strategies",
                f"- Optimal strategy: {curriculum_results['optimal_strategy']}",
                f"- Best MAE achieved: {curriculum_results['best_mae']:.4f}",
                f"- Improvement over no curriculum: {curriculum_results['improvement_over_no_curriculum']:.2f}%",
                ""
            ])

        # Add detailed results tables
        for component, results in ablation_results.items():
            if isinstance(results, dict) and 'analysis' in results:
                report_lines.extend([
                    f"## {component.replace('_', ' ').title()} Detailed Results",
                    ""
                ])

                for variant, metrics in results['analysis'].items():
                    if isinstance(metrics, dict) and 'mae' in metrics:
                        report_lines.append(
                            f"- {variant}: MAE = {metrics['mae']:.4f}, "
                            f"R² = {metrics.get('r2', 0.0):.3f}"
                        )

        return "\n".join(report_lines)

    def statistical_significance_test(
        self,
        predictions_a: np.ndarray,
        predictions_b: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Test statistical significance between two models.

        Args:
            predictions_a: Predictions from model A.
            predictions_b: Predictions from model B.
            targets: Ground truth targets.

        Returns:
            Statistical test results.
        """
        # Compute absolute errors for both models
        errors_a = np.abs(predictions_a - targets)
        errors_b = np.abs(predictions_b - targets)

        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(errors_a, errors_b)

        # Effect size (Cohen's d)
        mean_diff = np.mean(errors_a) - np.mean(errors_b)
        pooled_std = np.sqrt((np.var(errors_a) + np.var(errors_b)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        return {
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant_at_0_05': p_value < 0.05,
            'mean_error_a': float(np.mean(errors_a)),
            'mean_error_b': float(np.mean(errors_b)),
        }