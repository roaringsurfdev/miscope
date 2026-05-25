"""Back-compat re-export shim for the analyzer Registry (REQ_120).

The Registry's canonical home is :mod:`miscope.analysis.registry`. This
module re-exports the public surface so existing imports
(``from miscope.analysis.analyzers.registry import AnalyzerRegistry``)
continue to work. New code should import from
:mod:`miscope.analysis.registry` directly.
"""

from __future__ import annotations

from miscope.analysis.registry import AnalyzerRegistry, register_analyzer

__all__ = ["AnalyzerRegistry", "register_analyzer", "register_default_analyzers"]


def register_default_analyzers() -> None:
    """Import every built-in analyzer module so registration runs.

    Each analyzer either self-registers via the ``@register_analyzer``
    decorator (REQ_120) or is registered via the legacy ``AnalyzerRegistry``
    class methods below. Importing the module is sufficient for the
    decorator path; the legacy ``register*`` calls cover analyzers that
    have not yet been migrated to a Spec.
    """
    from miscope.analysis.analyzers.activation_dmd import ActivationDMD
    from miscope.analysis.analyzers.attention_fourier import AttentionFourierAnalyzer
    from miscope.analysis.analyzers.attention_freq import AttentionFreqAnalyzer
    from miscope.analysis.analyzers.attention_patterns import AttentionPatternsAnalyzer
    from miscope.analysis.analyzers.centroid_dmd import CentroidDMD
    from miscope.analysis.analyzers.coarseness import CoarsenessAnalyzer
    from miscope.analysis.analyzers.dominant_frequencies import DominantFrequenciesAnalyzer
    from miscope.analysis.analyzers.effective_dimensionality import EffectiveDimensionalityAnalyzer
    from miscope.analysis.analyzers.fourier_frequency_quality import FourierFrequencyQualityAnalyzer
    from miscope.analysis.analyzers.fourier_nucleation import FourierNucleationAnalyzer
    from miscope.analysis.analyzers.freq_group_weight_geometry import (
        FreqGroupWeightGeometryAnalyzer,
    )
    from miscope.analysis.analyzers.global_centroid_pca import GlobalCentroidPCA
    from miscope.analysis.analyzers.gradient_site import GradientSiteAnalyzer
    from miscope.analysis.analyzers.input_trace import InputTraceAnalyzer
    from miscope.analysis.analyzers.input_trace_graduation import InputTraceGraduationAnalyzer
    from miscope.analysis.analyzers.intragroup_manifold import IntraGroupManifoldAnalyzer
    from miscope.analysis.analyzers.landscape_flatness import LandscapeFlatnessAnalyzer
    from miscope.analysis.analyzers.neuron_activations import NeuronActivationsAnalyzer
    from miscope.analysis.analyzers.neuron_dynamics import NeuronDynamicsAnalyzer
    from miscope.analysis.analyzers.neuron_fourier import NeuronFourierAnalyzer
    from miscope.analysis.analyzers.neuron_freq_clusters import NeuronFreqClustersAnalyzer
    from miscope.analysis.analyzers.neuron_group_pca import NeuronGroupPCAAnalyzer
    from miscope.analysis.analyzers.neuron_grouping import NeuronGrouping
    from miscope.analysis.analyzers.parameter_dmd import ParameterDMD
    from miscope.analysis.analyzers.parameter_snapshot import ParameterSnapshotAnalyzer
    from miscope.analysis.analyzers.parameter_trajectory_pca import ParameterTrajectoryPCA
    from miscope.analysis.analyzers.repr_geometry import RepresentationalGeometryAnalyzer
    from miscope.analysis.analyzers.transient_frequency import TransientFrequencyAnalyzer

    # Legacy registration path — no-op if the analyzer has already self-
    # registered via @register_analyzer.
    AnalyzerRegistry.register(AttentionFourierAnalyzer)
    AnalyzerRegistry.register(AttentionFreqAnalyzer)
    AnalyzerRegistry.register(AttentionPatternsAnalyzer)
    AnalyzerRegistry.register(DominantFrequenciesAnalyzer)
    AnalyzerRegistry.register(NeuronActivationsAnalyzer)
    AnalyzerRegistry.register(NeuronFreqClustersAnalyzer)
    AnalyzerRegistry.register(CoarsenessAnalyzer)
    AnalyzerRegistry.register(ParameterSnapshotAnalyzer)
    AnalyzerRegistry.register(EffectiveDimensionalityAnalyzer)
    AnalyzerRegistry.register(LandscapeFlatnessAnalyzer)
    AnalyzerRegistry.register(RepresentationalGeometryAnalyzer)
    AnalyzerRegistry.register(FourierNucleationAnalyzer)
    AnalyzerRegistry.register(InputTraceAnalyzer)

    AnalyzerRegistry.register_secondary(FourierFrequencyQualityAnalyzer)
    AnalyzerRegistry.register_secondary(NeuronFourierAnalyzer)
    AnalyzerRegistry.register_secondary(NeuronGrouping)

    AnalyzerRegistry.register_cross_epoch(ParameterTrajectoryPCA)
    AnalyzerRegistry.register_cross_epoch(NeuronDynamicsAnalyzer)
    AnalyzerRegistry.register_cross_epoch(GlobalCentroidPCA)
    AnalyzerRegistry.register_cross_epoch(CentroidDMD)
    AnalyzerRegistry.register_cross_epoch(ActivationDMD)
    AnalyzerRegistry.register_cross_epoch(ParameterDMD)
    AnalyzerRegistry.register_cross_epoch(GradientSiteAnalyzer)
    AnalyzerRegistry.register_cross_epoch(InputTraceGraduationAnalyzer)
    AnalyzerRegistry.register_cross_epoch(NeuronGroupPCAAnalyzer)
    AnalyzerRegistry.register_cross_epoch(FreqGroupWeightGeometryAnalyzer)
    AnalyzerRegistry.register_cross_epoch(IntraGroupManifoldAnalyzer)
    AnalyzerRegistry.register_cross_epoch(TransientFrequencyAnalyzer)


# Auto-register default analyzers on import
register_default_analyzers()
