"""Residual extraction component for MetaGene-style models."""

from .config import ExtractionConfig
from .extractor import ResidualExtractionPipeline
from .model_adapter import FakeModelAdapter, TransformersModelAdapter

__all__ = [
    "ExtractionConfig",
    "ResidualExtractionPipeline",
    "FakeModelAdapter",
    "TransformersModelAdapter",
]

