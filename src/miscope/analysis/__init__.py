"""Analysis pipeline for training dynamics workbench.

This package provides:
- library/: Generic, reusable analysis functions (fourier, activations)
- analyzers/: Family-bound analyzers that compose library functions
- AnalysisPipeline: Orchestrates analysis across checkpoints
- AnalyzerRegistry: Discovers and instantiates analyzers
"""

from miscope.analysis.analyzers import AnalyzerRegistry
from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.bundle import TransformerLensBundle
from miscope.analysis.freshness import FreshnessReport, check_freshness
from miscope.analysis.pipeline import AnalysisPipeline
from miscope.analysis.protocols import (
    ActivationBundle,
    AnalysisRunConfig,
    Analyzer,
    CrossEpochAnalyzer,
    SecondaryAnalyzer,
)

__all__ = [
    "ActivationBundle",
    "Analyzer",
    "AnalyzerRegistry",
    "AnalysisPipeline",
    "AnalysisRunConfig",
    "ArtifactLoader",
    "CrossEpochAnalyzer",
    "FreshnessReport",
    "SecondaryAnalyzer",
    "TransformerLensBundle",
    "check_freshness",
]
