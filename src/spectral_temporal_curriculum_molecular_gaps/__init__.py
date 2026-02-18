"""Spectral Temporal Curriculum Molecular Gaps Package.

A research framework that combines spectral graph wavelets with curriculum learning
for molecular property prediction on PCQM4Mv2 dataset.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

__all__ = [
    "SpectralTemporalMolecularNet",
    "CurriculumTrainer",
    "PCQM4Mv2CurriculumDataLoader",
    "MolecularPropertyMetrics",
]


def __getattr__(name: str):
    """Lazy imports to avoid circular import issues."""
    if name == "SpectralTemporalMolecularNet":
        from .models.model import SpectralTemporalMolecularNet
        return SpectralTemporalMolecularNet
    elif name == "CurriculumTrainer":
        from .training.trainer import CurriculumTrainer
        return CurriculumTrainer
    elif name == "PCQM4Mv2CurriculumDataLoader":
        from .data.loader import PCQM4Mv2CurriculumDataLoader
        return PCQM4Mv2CurriculumDataLoader
    elif name == "MolecularPropertyMetrics":
        from .evaluation.metrics import MolecularPropertyMetrics
        return MolecularPropertyMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")