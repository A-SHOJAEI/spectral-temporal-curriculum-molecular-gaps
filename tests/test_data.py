"""Tests for data loading and preprocessing modules."""

import os
import tempfile
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from spectral_temporal_curriculum_molecular_gaps.data.preprocessing import (
    SpectralFeatureExtractor,
    MolecularComplexityCalculator,
    CurriculumScheduler,
)


class TestSpectralFeatureExtractor:
    """Test spectral feature extraction."""

    def test_initialization(self):
        """Test spectral feature extractor initialization."""
        extractor = SpectralFeatureExtractor(num_levels=4)
        assert extractor.num_levels == 4
        assert len(extractor.wavelet_scales) == 4

    def test_extract_features_simple_graph(self, sample_molecular_graph):
        """Test spectral feature extraction on simple graph."""
        extractor = SpectralFeatureExtractor(num_levels=2)
        features = extractor.extract_features(sample_molecular_graph)

        assert isinstance(features, list)
        assert len(features) == 2
        assert features[0].shape[0] == sample_molecular_graph.num_nodes
        assert features[0].shape[1] == sample_molecular_graph.x.shape[1]

    def test_extract_features_single_node(self):
        """Test spectral feature extraction on single node graph."""
        from torch_geometric.data import Data

        # Single node graph
        x = torch.randn(1, 9)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        extractor = SpectralFeatureExtractor(num_levels=2)
        features = extractor.extract_features(data)

        assert len(features) == 2
        assert features[0].shape == (1, 9)

    def test_extract_features_disconnected_graph(self):
        """Test spectral feature extraction on disconnected graph."""
        from torch_geometric.data import Data

        # Disconnected graph with 4 nodes, 2 components
        x = torch.randn(4, 9)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        extractor = SpectralFeatureExtractor(num_levels=2)
        features = extractor.extract_features(data)

        assert len(features) == 2
        assert features[0].shape == (4, 9)


class TestMolecularComplexityCalculator:
    """Test molecular complexity calculation."""

    def test_initialization(self):
        """Test molecular complexity calculator initialization."""
        calculator = MolecularComplexityCalculator()
        assert calculator is not None

    def test_calculate_complexity_simple_graph(self, sample_molecular_graph):
        """Test complexity calculation on simple graph."""
        calculator = MolecularComplexityCalculator()
        complexity = calculator.calculate_complexity(sample_molecular_graph)

        assert isinstance(complexity, float)
        assert complexity >= 0.0
        assert not np.isnan(complexity)

    def test_calculate_complexity_different_sizes(self):
        """Test complexity calculation scales with graph size."""
        from torch_geometric.data import Data

        calculator = MolecularComplexityCalculator()

        # Small graph
        x_small = torch.randn(3, 9)
        edge_index_small = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        data_small = Data(x=x_small, edge_index=edge_index_small)

        # Large graph
        x_large = torch.randn(20, 9)
        edge_index_large = torch.tensor([
            list(range(20)) + list(range(19)),
            list(range(1, 20)) + [0] + list(range(20))
        ], dtype=torch.long)
        data_large = Data(x=x_large, edge_index=edge_index_large)

        complexity_small = calculator.calculate_complexity(data_small)
        complexity_large = calculator.calculate_complexity(data_large)

        # Large graphs should generally have higher complexity
        assert complexity_large > complexity_small

    def test_calculate_graph_complexity(self, sample_molecular_graph):
        """Test graph-based complexity calculation."""
        calculator = MolecularComplexityCalculator()
        complexity = calculator._calculate_graph_complexity(sample_molecular_graph)

        assert isinstance(complexity, float)
        assert complexity >= 0.0

    @patch('spectral_temporal_curriculum_molecular_gaps.data.preprocessing.Chem')
    def test_rdkit_complexity_calculation_fallback(self, mock_chem, sample_molecular_graph):
        """Test fallback when RDKit conversion fails."""
        # Mock RDKit to fail conversion
        mock_chem.RWMol.return_value = None

        calculator = MolecularComplexityCalculator()
        complexity = calculator.calculate_complexity(sample_molecular_graph)

        # Should fallback to graph-based calculation
        assert isinstance(complexity, float)
        assert complexity >= 0.0


class TestCurriculumScheduler:
    """Test curriculum learning scheduler."""

    def test_initialization_default(self):
        """Test default scheduler initialization."""
        scheduler = CurriculumScheduler()
        assert scheduler.strategy == "linear"
        assert scheduler.warmup_epochs == 10
        assert scheduler.total_epochs == 100

    def test_initialization_custom(self):
        """Test custom scheduler initialization."""
        scheduler = CurriculumScheduler(
            strategy="exponential",
            warmup_epochs=5,
            total_epochs=50
        )
        assert scheduler.strategy == "exponential"
        assert scheduler.warmup_epochs == 5
        assert scheduler.total_epochs == 50

    def test_linear_schedule(self):
        """Test linear curriculum schedule."""
        scheduler = CurriculumScheduler(strategy="linear", warmup_epochs=10)

        # Test various epochs
        assert scheduler.get_curriculum_fraction(0) == 0.1
        assert scheduler.get_curriculum_fraction(5) == 0.55
        assert scheduler.get_curriculum_fraction(10) >= 0.99  # After warmup
        assert scheduler.get_curriculum_fraction(15) == 1.0

    def test_exponential_schedule(self):
        """Test exponential curriculum schedule."""
        scheduler = CurriculumScheduler(strategy="exponential", warmup_epochs=10)

        fractions = [scheduler.get_curriculum_fraction(epoch) for epoch in range(15)]

        # Should be monotonically increasing during warmup
        for i in range(9):
            assert fractions[i] <= fractions[i + 1]

        # Should be 1.0 after warmup
        assert fractions[10] >= 0.99
        assert fractions[14] == 1.0

    def test_cosine_schedule(self):
        """Test cosine curriculum schedule."""
        scheduler = CurriculumScheduler(strategy="cosine", warmup_epochs=10)

        fractions = [scheduler.get_curriculum_fraction(epoch) for epoch in range(15)]

        # Should be monotonically increasing during warmup
        for i in range(9):
            assert fractions[i] <= fractions[i + 1]

        # Should be 1.0 after warmup
        assert fractions[14] == 1.0

    def test_invalid_strategy(self):
        """Test error handling for invalid strategy."""
        scheduler = CurriculumScheduler(strategy="invalid")

        with pytest.raises(ValueError, match="Unknown curriculum strategy"):
            scheduler.get_curriculum_fraction(0)

    def test_schedule_properties(self):
        """Test general properties of curriculum schedules."""
        strategies = ["linear", "exponential", "cosine"]

        for strategy in strategies:
            scheduler = CurriculumScheduler(strategy=strategy, warmup_epochs=10)

            # Test boundary conditions
            assert 0.1 <= scheduler.get_curriculum_fraction(0) <= 0.2
            assert scheduler.get_curriculum_fraction(15) == 1.0

            # Test monotonicity during warmup
            for epoch in range(9):
                curr_frac = scheduler.get_curriculum_fraction(epoch)
                next_frac = scheduler.get_curriculum_fraction(epoch + 1)
                assert curr_frac <= next_frac, f"Non-monotonic at epoch {epoch} for {strategy}"


class TestDataLoadingIntegration:
    """Integration tests for data loading components."""

    def test_spectral_features_with_complexity(self, sample_molecular_graph):
        """Test integration between spectral features and complexity calculation."""
        extractor = SpectralFeatureExtractor(num_levels=2)
        calculator = MolecularComplexityCalculator()

        # Extract spectral features
        spectral_features = extractor.extract_features(sample_molecular_graph)

        # Calculate complexity
        complexity = calculator.calculate_complexity(sample_molecular_graph)

        assert len(spectral_features) == 2
        assert isinstance(complexity, float)
        assert complexity >= 0.0

    def test_curriculum_progression_consistency(self):
        """Test that curriculum progression is consistent across different schedulers."""
        warmup_epochs = 5

        schedulers = {
            "linear": CurriculumScheduler("linear", warmup_epochs),
            "exponential": CurriculumScheduler("exponential", warmup_epochs),
            "cosine": CurriculumScheduler("cosine", warmup_epochs),
        }

        # All should start low and end high
        for name, scheduler in schedulers.items():
            start_frac = scheduler.get_curriculum_fraction(0)
            mid_frac = scheduler.get_curriculum_fraction(warmup_epochs // 2)
            end_frac = scheduler.get_curriculum_fraction(warmup_epochs + 1)

            assert start_frac < mid_frac < end_frac, f"Inconsistent progression for {name}"
            assert end_frac == 1.0, f"Should reach 1.0 after warmup for {name}"

    def test_edge_cases_handling(self):
        """Test handling of edge cases in data processing."""
        # Test with minimal valid inputs
        calculator = MolecularComplexityCalculator()
        extractor = SpectralFeatureExtractor(num_levels=1)

        # Single node, no edges
        from torch_geometric.data import Data
        x = torch.randn(1, 9)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        complexity = calculator.calculate_complexity(data)
        features = extractor.extract_features(data)

        assert isinstance(complexity, float)
        assert len(features) == 1
        assert features[0].shape == (1, 9)