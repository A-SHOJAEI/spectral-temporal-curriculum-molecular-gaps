"""Test configuration and fixtures."""

import os
import tempfile
import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch

# Configure PyTorch to allow numpy objects in torch.load for OGB datasets
# This is needed for PyTorch 2.6+ which changed weights_only default to True
try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
except (ImportError, AttributeError):
    # Fallback for different numpy versions
    pass

from spectral_temporal_curriculum_molecular_gaps.models.model import SpectralTemporalMolecularNet
from spectral_temporal_curriculum_molecular_gaps.utils.config import get_default_config


@pytest.fixture
def device():
    """Get test device (CPU for CI/CD)."""
    return torch.device('cpu')


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    config = get_default_config()
    # Use smaller dimensions for faster testing
    config['model']['hidden_dim'] = 64
    config['model']['num_spectral_layers'] = 2
    config['model']['num_scales'] = 2
    config['training']['batch_size'] = 4
    config['training']['num_epochs'] = 2
    return config


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_molecular_graph():
    """Create a sample molecular graph for testing."""
    # Simple benzene-like molecule
    num_nodes = 6
    x = torch.randn(num_nodes, 9)  # Atomic features
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],  # Source nodes
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]   # Target nodes
    ], dtype=torch.long)
    y = torch.tensor([5.2])  # HOMO-LUMO gap in eV

    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def sample_batch(sample_molecular_graph):
    """Create a batch of sample molecular graphs."""
    graphs = [sample_molecular_graph for _ in range(3)]
    # Modify targets slightly for each graph
    for i, graph in enumerate(graphs):
        graph.y = torch.tensor([5.2 + i * 0.1])

    return Batch.from_data_list(graphs)


@pytest.fixture
def sample_model(sample_config, device):
    """Create a sample model for testing."""
    model = SpectralTemporalMolecularNet(
        input_dim=sample_config['model']['input_dim'],
        hidden_dim=sample_config['model']['hidden_dim'],
        num_spectral_layers=sample_config['model']['num_spectral_layers'],
        num_scales=sample_config['model']['num_scales'],
        num_curriculum_stages=sample_config['model']['num_curriculum_stages'],
        dropout=sample_config['model']['dropout'],
        pool_type=sample_config['model']['pool_type'],
    )
    return model.to(device)


@pytest.fixture
def sample_spectral_features(sample_batch, sample_config):
    """Create sample spectral features for testing."""
    num_scales = sample_config['model']['num_scales']
    num_nodes = sample_batch.x.size(0)
    feature_dim = sample_batch.x.size(1)

    spectral_features = []
    for scale_idx in range(num_scales):
        # Create random spectral features for testing
        features = torch.randn(num_nodes, feature_dim) * 0.1
        spectral_features.append(features)

    return spectral_features


@pytest.fixture
def reproducible_random_state():
    """Set reproducible random state for tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    yield

    # Reset to default state after test
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@pytest.fixture
def mock_pcqm4mv2_data():
    """Mock PCQM4Mv2 dataset data for testing."""
    # Create synthetic molecular graphs
    molecules = []
    for i in range(10):
        num_nodes = np.random.randint(5, 15)
        x = torch.randn(num_nodes, 9)

        # Create random connected graph
        num_edges = num_nodes * 2
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # HOMO-LUMO gap (synthetic)
        y = torch.tensor([4.0 + np.random.randn() * 1.0])

        molecules.append(Data(x=x, edge_index=edge_index, y=y))

    return molecules


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )