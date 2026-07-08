"""Analysis pipeline for training dynamics workbench.

This package provides:
- library/: Generic, reusable analysis functions (fourier, activations)
- analyzers/: Family-bound analyzers that compose library functions
- AnalysisPipeline: Orchestrates analysis across checkpoints
- AnalyzerRegistry: Discovers and instantiates analyzers
"""

from miscope.analysis.analyzers import AnalyzerRegistry
from miscope.analysis.freshness import FreshnessReport, check_freshness
from miscope.analysis.pipeline import AnalysisPipeline
from miscope.analysis.planner import Plan, PlanItem, plan_analysis
from miscope.analysis.protocols import AnalysisRunConfig, Analyzer

# Note: ``ArtifactLoader`` is an internal storage primitive and is intentionally
# not re-exported here (REQ_125). Consumers should reach a configured loader
# through ``variant.artifacts`` (returns an instance) rather than importing the
# class directly. The class continues to live at
# ``miscope.analysis.artifact_loader`` for pipeline and cross-epoch analyzer use.

__all__ = [
    "Analyzer",
    "AnalyzerRegistry",
    "AnalysisPipeline",
    "AnalysisRunConfig",
    "FreshnessReport",
    "Plan",
    "PlanItem",
    "check_freshness",
    "plan_analysis",
]
