"""Individual analysis modules.

Analyzers compute analysis on model checkpoints and produce artifacts.
Each analyzer implements the Analyzer protocol and can be registered
with the AnalyzerRegistry for discovery.
"""

from miscope.analysis.analyzers.attention_freq import AttentionFreqAnalyzer
from miscope.analysis.analyzers.attention_patterns import AttentionPatternsAnalyzer
from miscope.analysis.analyzers.coarseness import CoarsenessAnalyzer
from miscope.analysis.analyzers.dominant_frequencies import DominantFrequenciesAnalyzer
from miscope.analysis.analyzers.effective_dimensionality import EffectiveDimensionalityAnalyzer
from miscope.analysis.analyzers.landscape_flatness import LandscapeFlatnessAnalyzer
from miscope.analysis.analyzers.neuron_activations import NeuronActivationsAnalyzer
from miscope.analysis.analyzers.neuron_dynamics import NeuronDynamicsAnalyzer
from miscope.analysis.analyzers.neuron_fourier import NeuronFourierAnalyzer
from miscope.analysis.analyzers.neuron_freq_clusters import NeuronFreqClustersAnalyzer
from miscope.analysis.analyzers.parameter_snapshot import ParameterSnapshotAnalyzer
from miscope.analysis.analyzers.parameter_trajectory_pca import ParameterTrajectoryPCA
from miscope.analysis.analyzers.registry import AnalyzerRegistry
from miscope.analysis.analyzers.repr_geometry import RepresentationalGeometryAnalyzer

__all__ = [
    "AnalyzerRegistry",
    "AttentionFreqAnalyzer",
    "NeuronFourierAnalyzer",
    "AttentionPatternsAnalyzer",
    "CoarsenessAnalyzer",
    "DominantFrequenciesAnalyzer",
    "EffectiveDimensionalityAnalyzer",
    "LandscapeFlatnessAnalyzer",
    "NeuronActivationsAnalyzer",
    "NeuronDynamicsAnalyzer",
    "NeuronFreqClustersAnalyzer",
    "ParameterSnapshotAnalyzer",
    "ParameterTrajectoryPCA",
    "RepresentationalGeometryAnalyzer",
]
