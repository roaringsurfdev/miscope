"""Analysis pipeline for training dynamics workbench.

This package provides:
- library/: Generic, reusable analysis functions (fourier, activations)
- analyzers/: Family-bound analyzers that compose library functions
- AnalysisPipeline: Orchestrates analysis across checkpoints
- AnalyzerRegistry: Discovers and instantiates analyzers
"""

from miscope.analysis.analyzers import AnalyzerRegistry
from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.freshness import FreshnessReport, check_freshness
from miscope.analysis.pipeline import AnalysisPipeline
from miscope.analysis.protocols import (
    AnalysisRunConfig,
    Analyzer,
    CrossEpochAnalyzer,
    SecondaryAnalyzer,
)

__all__ = [
    "Analyzer",
    "AnalyzerRegistry",
    "AnalysisPipeline",
    "AnalysisRunConfig",
    "ArtifactLoader",
    "CrossEpochAnalyzer",
    "FreshnessReport",
    "SecondaryAnalyzer",
    "check_freshness",
]
