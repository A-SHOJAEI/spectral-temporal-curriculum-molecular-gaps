"""Tests for model implementations."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from spectral_temporal_curriculum_molecular_gaps.models.model import (
    SpectralConvLayer,
    MultiScaleSpectralEncoder,
    TemporalCurriculumHead,
    SpectralTemporalMolecularNet,
)


class TestSpectralConvLayer:
    """Test spectral convolution layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = SpectralConvLayer(
            in_channels=64,
            out_channels=128,
            num_scales=4,
            dropout=0.1
        )

        assert layer.in_channels == 64
        assert layer.out_channels == 128
        assert layer.num_scales == 4
        assert len(layer.scale_convs) == 4

    def test_forward_single_graph(self, sample_molecular_graph):
        """Test forward pass with single graph."""
        layer = SpectralConvLayer(
            in_channels=9,
            out_channels=64,
            num_scales=2,
            dropout=0.0  # Disable dropout for testing
        )

        x = sample_molecular_graph.x
        spectral_features = [
            torch.randn_like(x) * 0.1,
            torch.randn_like(x) * 0.1
        ]

        output = layer(x, spectral_features)

        assert output.shape == (x.size(0), 64)
        assert not torch.isnan(output).any()

    def test_forward_batch(self, sample_batch):
        """Test forward pass with batch of graphs."""
        layer = SpectralConvLayer(
            in_channels=9,
            out_channels=64,
            num_scales=2,
            dropout=0.0
        )

        x = sample_batch.x
        batch = sample_batch.batch
        spectral_features = [
            torch.randn_like(x) * 0.1,
            torch.randn_like(x) * 0.1
        ]

        output = layer(x, spectral_features, batch)

        assert output.shape == (x.size(0), 64)
        assert not torch.isnan(output).any()

    def test_parameter_reset(self):
        """Test parameter reset functionality."""
        layer = SpectralConvLayer(in_channels=32, out_channels=64, num_scales=2)

        # Store initial parameters
        initial_params = [p.clone() for p in layer.parameters()]

        # Reset parameters
        layer.reset_parameters()

        # Check that parameters changed
        current_params = list(layer.parameters())
        for initial, current in zip(initial_params, current_params):
            assert not torch.equal(initial, current)


class TestMultiScaleSpectralEncoder:
    """Test multi-scale spectral encoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = MultiScaleSpectralEncoder(
            input_dim=9,
            hidden_dim=128,
            num_layers=3,
            num_scales=4
        )

        assert encoder.input_dim == 9
        assert encoder.hidden_dim == 128
        assert encoder.num_layers == 3
        assert len(encoder.spectral_layers) == 3
        assert len(encoder.graph_layers) == 3

    def test_forward_pass(self, sample_batch):
        """Test forward pass through encoder."""
        encoder = MultiScaleSpectralEncoder(
            input_dim=9,
            hidden_dim=64,
            num_layers=2,
            num_scales=2,
            dropout=0.0
        )

        x = sample_batch.x
        edge_index = sample_batch.edge_index
        batch = sample_batch.batch
        spectral_features = [
            torch.randn_like(x) * 0.1,
            torch.randn_like(x) * 0.1
        ]

        output = encoder(x, edge_index, spectral_features, batch)

        assert output.shape == (x.size(0), 64)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, sample_batch):
        """Test gradient flow through encoder."""
        encoder = MultiScaleSpectralEncoder(
            input_dim=9,
            hidden_dim=32,
            num_layers=2,
            num_scales=2,
            dropout=0.0
        )

        x = sample_batch.x
        edge_index = sample_batch.edge_index
        batch = sample_batch.batch
        spectral_features = [
            torch.randn_like(x) * 0.1,
            torch.randn_like(x) * 0.1
        ]

        x.requires_grad_(True)
        output = encoder(x, edge_index, spectral_features, batch)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestTemporalCurriculumHead:
    """Test temporal curriculum head."""

    def test_initialization(self):
        """Test head initialization."""
        head = TemporalCurriculumHead(
            input_dim=128,
            hidden_dim=64,
            output_dim=1,
            num_curriculum_stages=4
        )

        assert head.input_dim == 128
        assert head.hidden_dim == 64
        assert head.output_dim == 1
        assert head.num_curriculum_stages == 4
        assert len(head.stage_heads) == 4

    def test_forward_training_with_stage(self):
        """Test forward pass during training with specified stage."""
        head = TemporalCurriculumHead(
            input_dim=64,
            hidden_dim=32,
            output_dim=1,
            num_curriculum_stages=3
        )

        x = torch.randn(4, 64)
        head.train()

        outputs = head(x, curriculum_stage=1, training=True)

        assert 'prediction' in outputs
        assert 'stage_logits' in outputs
        assert 'stage_probs' in outputs

        assert outputs['prediction'].shape == (4, 1)
        assert outputs['stage_logits'].shape == (4, 3)
        assert outputs['stage_probs'].shape == (4, 3)

    def test_forward_inference(self):
        """Test forward pass during inference."""
        head = TemporalCurriculumHead(
            input_dim=64,
            hidden_dim=32,
            output_dim=1,
            num_curriculum_stages=3
        )

        x = torch.randn(4, 64)
        head.eval()

        outputs = head(x, training=False)

        assert 'prediction' in outputs
        assert outputs['prediction'].shape == (4, 1)

    def test_stage_probability_normalization(self):
        """Test that stage probabilities are properly normalized."""
        head = TemporalCurriculumHead(
            input_dim=32,
            hidden_dim=16,
            num_curriculum_stages=4
        )

        x = torch.randn(2, 32)
        outputs = head(x)

        stage_probs = outputs['stage_probs']
        # Check probabilities sum to 1
        prob_sums = torch.sum(stage_probs, dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)


class TestSpectralTemporalMolecularNet:
    """Test complete molecular network."""

    def test_initialization_default(self, sample_config):
        """Test default model initialization."""
        model = SpectralTemporalMolecularNet(
            input_dim=sample_config['model']['input_dim'],
            hidden_dim=sample_config['model']['hidden_dim'],
            num_spectral_layers=sample_config['model']['num_spectral_layers'],
            num_scales=sample_config['model']['num_scales'],
        )

        assert isinstance(model.encoder, MultiScaleSpectralEncoder)
        assert isinstance(model.curriculum_head, TemporalCurriculumHead)

    def test_forward_pass(self, sample_model, sample_batch, sample_spectral_features):
        """Test complete forward pass."""
        model = sample_model
        outputs = model(sample_batch, sample_spectral_features)

        assert 'prediction' in outputs
        assert outputs['prediction'].shape[0] == sample_batch.num_graphs
        assert outputs['prediction'].shape[1] == 1

    def test_forward_with_curriculum_stage(self, sample_model, sample_batch, sample_spectral_features):
        """Test forward pass with curriculum stage."""
        model = sample_model
        model.train()

        outputs = model(sample_batch, sample_spectral_features, curriculum_stage=2)

        assert 'prediction' in outputs
        assert 'stage_logits' in outputs

    def test_loss_computation(self, sample_model, sample_batch, sample_spectral_features):
        """Test loss computation."""
        model = sample_model
        outputs = model(sample_batch, sample_spectral_features, curriculum_stage=1)

        targets = sample_batch.y.float()
        losses = model.compute_loss(outputs, targets, curriculum_stage=1)

        assert 'total' in losses
        assert 'regression' in losses
        assert 'stage_classification' in losses

        assert losses['total'].item() > 0
        assert losses['regression'].item() > 0

    def test_pooling_types(self, sample_batch, sample_spectral_features, sample_config):
        """Test different pooling types."""
        pooling_types = ['mean', 'max', 'add', 'attention']

        for pool_type in pooling_types:
            model = SpectralTemporalMolecularNet(
                input_dim=sample_config['model']['input_dim'],
                hidden_dim=sample_config['model']['hidden_dim'],
                num_spectral_layers=sample_config['model']['num_spectral_layers'],
                num_scales=sample_config['model']['num_scales'],
                pool_type=pool_type
            )

            # Should not raise any errors
            outputs = model(sample_batch, sample_spectral_features)
            assert outputs['prediction'].shape[0] == sample_batch.num_graphs

    def test_gradient_flow_complete_model(self, sample_model, sample_batch, sample_spectral_features):
        """Test gradient flow through complete model."""
        model = sample_model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Forward pass
        outputs = model(sample_batch, sample_spectral_features, curriculum_stage=0)
        targets = sample_batch.y.float()
        losses = model.compute_loss(outputs, targets, curriculum_stage=0)

        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()

        # Check gradients exist and are not NaN
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

        optimizer.step()

    def test_model_determinism(self, sample_batch, sample_spectral_features, sample_config, reproducible_random_state):
        """Test model determinism with fixed random seed."""
        # Create two identical models
        model1 = SpectralTemporalMolecularNet(
            input_dim=sample_config['model']['input_dim'],
            hidden_dim=sample_config['model']['hidden_dim'],
            num_spectral_layers=sample_config['model']['num_spectral_layers'],
            num_scales=sample_config['model']['num_scales'],
        )

        model2 = SpectralTemporalMolecularNet(
            input_dim=sample_config['model']['input_dim'],
            hidden_dim=sample_config['model']['hidden_dim'],
            num_spectral_layers=sample_config['model']['num_spectral_layers'],
            num_scales=sample_config['model']['num_scales'],
        )

        # Copy weights from model1 to model2
        model2.load_state_dict(model1.state_dict())

        # Both models should produce identical outputs
        model1.eval()
        model2.eval()

        with torch.no_grad():
            outputs1 = model1(sample_batch, sample_spectral_features)
            outputs2 = model2(sample_batch, sample_spectral_features)

        assert torch.allclose(outputs1['prediction'], outputs2['prediction'], atol=1e-6)

    def test_model_device_compatibility(self, sample_batch, sample_spectral_features, sample_config):
        """Test model device compatibility."""
        model = SpectralTemporalMolecularNet(
            input_dim=sample_config['model']['input_dim'],
            hidden_dim=32,  # Smaller for testing
            num_spectral_layers=1,
            num_scales=1,
        )

        # Test CPU
        model_cpu = model.to('cpu')
        batch_cpu = sample_batch.to('cpu')
        spectral_cpu = [feat.to('cpu') for feat in sample_spectral_features]

        outputs_cpu = model_cpu(batch_cpu, spectral_cpu)
        assert outputs_cpu['prediction'].device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            batch_cuda = sample_batch.to('cuda')
            spectral_cuda = [feat.to('cuda') for feat in sample_spectral_features]

            outputs_cuda = model_cuda(batch_cuda, spectral_cuda)
            assert outputs_cuda['prediction'].device.type == 'cuda'