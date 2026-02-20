# Spectral Temporal Curriculum Molecular Gaps

A research framework that combines spectral graph wavelets with curriculum learning to progressively train molecular property predictors on PCQM4Mv2. The approach orders molecules by structural complexity, using spectral decomposition to capture electron delocalization patterns critical for HOMO-LUMO gap prediction.

## Features

- **Spectral Graph Wavelets**: Multi-scale molecular representation using graph Laplacian eigendecomposition
- **Curriculum Learning**: Progressive training from simple to complex molecular structures
- **Real-time Spectral Features**: Dynamic spectral feature extraction with proper eigenvalue handling
- **Comprehensive Evaluation**: Bootstrap confidence intervals and out-of-distribution testing
- **Production Ready**: Type hints, error handling, checkpointing, and MLflow integration

## Quick Start

```bash
# Install dependencies
pip install -e .

# Train model with default configuration
python scripts/train.py

# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

## Installation

```bash
git clone https://github.com/A-SHOJAEI/spectral-temporal-curriculum-molecular-gaps.git
cd spectral-temporal-curriculum-molecular-gaps
pip install -e .
```

## Usage

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom config
python scripts/train.py --config configs/custom.yaml

# Resume from checkpoint
python scripts/train.py --resume checkpoints/checkpoint_epoch_50.pt

# Debug mode with reduced parameters
python scripts/train.py --debug
```

### Evaluation

```bash
# Comprehensive evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --output-dir results/

# Quick evaluation on test set only
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test-only
```

### Python API

```python
from spectral_temporal_curriculum_molecular_gaps import SpectralTemporalMolecularNet, CurriculumTrainer
from spectral_temporal_curriculum_molecular_gaps.data.loader import PCQM4Mv2CurriculumDataLoader
from spectral_temporal_curriculum_molecular_gaps.utils.config import load_config

# Load configuration
config = load_config("configs/default.yaml")

# Create data loader
data_loader = PCQM4Mv2CurriculumDataLoader(
    root=config["data"]["root_dir"],
    batch_size=config["training"]["batch_size"]
)

# Create model
model = SpectralTemporalMolecularNet(
    input_dim=config["model"]["input_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    num_spectral_layers=config["model"]["num_spectral_layers"],
    num_scales=config["model"]["num_scales"]
)

# Create trainer
trainer = CurriculumTrainer(
    model=model,
    data_loader=data_loader,
    config=config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Train
metrics = trainer.train()
```

## Architecture

### Core Components

1. **SpectralConvLayer**: Graph convolution layer using spectral decomposition
2. **CurriculumHead**: Stage-aware prediction head for curriculum learning
3. **AttentionPooling**: Learnable graph-level pooling mechanism
4. **CurriculumTrainer**: Training pipeline with automatic curriculum scheduling

### Spectral Features

The model extracts multi-scale spectral features using graph Laplacian eigendecomposition:

- Computes normalized graph Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
- Extracts smallest eigenvalues and eigenvectors
- Projects node features onto eigenvector space with spectral weighting
- Handles edge cases: isolated nodes, numerical instability, small graphs

## Configuration

Key parameters in `configs/default.yaml`:

```yaml
model:
  hidden_dim: 256
  num_spectral_layers: 4
  num_scales: 4
  dropout: 0.1

training:
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 32
  early_stopping_patience: 20

data:
  root_dir: "./data"
  curriculum_strategy: "complexity_based"
  num_spectral_scales: 4
```

## Results

Trained on PCQM4Mv2 (3.7M molecules) for HOMO-LUMO gap prediction with spectral curriculum learning.

### Test Set Metrics

| Metric | Value |
|--------|-------|
| MAE (eV) | 0.268 |
| RMSE (eV) | 0.379 |
| R² | 0.908 |
| Chemical Accuracy (%) | 11.81 |
| Test Predictions | 73,545 |

### Training History (10 epochs)

| Epoch | Train Loss | Val Loss | Val MAE | Curriculum Fraction |
|-------|-----------|----------|---------|-------------------|
| 1 | 0.701 | 0.507 | 0.507 | 0.10 |
| 2 | 0.471 | 0.500 | 0.500 | 0.19 |
| 3 | 0.457 | 0.393 | 0.393 | 0.28 |
| 4 | 0.354 | 0.300 | 0.300 | 0.37 |
| 5 | 0.311 | 0.275 | 0.275 | 0.46 |
| 6 | 0.351 | 0.307 | 0.307 | 0.55 |
| 7 | 0.291 | 0.277 | 0.277 | 0.64 |
| 8 | 0.280 | 0.289 | 0.289 | 0.73 |
| 9 | 0.317 | 0.271 | 0.271 | 0.82 |
| 10 | 0.272 | 0.268 | 0.268 | 0.91 |

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- PyTorch Geometric ≥ 2.4.0
- RDKit ≥ 2022.09.1
- NumPy, SciPy, scikit-learn
- Optional: MLflow (for experiment tracking)

See `pyproject.toml` for complete dependency list.

## Project Structure

```
spectral-temporal-curriculum-molecular-gaps/
├── src/spectral_temporal_curriculum_molecular_gaps/
│   ├── models/          # Model architectures
│   ├── training/        # Training pipeline
│   ├── data/           # Data loading and preprocessing
│   ├── evaluation/     # Metrics and analysis
│   └── utils/          # Configuration and utilities
├── scripts/            # Training and evaluation scripts
├── configs/            # Configuration files
├── tests/              # Unit tests
└── checkpoints/        # Model checkpoints (created during training)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/spectral_temporal_curriculum_molecular_gaps

# Type checking
mypy src/

# Code formatting
black src/ tests/
isort src/ tests/
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.