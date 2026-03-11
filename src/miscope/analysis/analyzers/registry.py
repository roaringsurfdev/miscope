"""Analyzer registry for discovering and instantiating analyzers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from miscope.analysis.protocols import Analyzer, CrossEpochAnalyzer, SecondaryAnalyzer

if TYPE_CHECKING:
    from miscope.families.protocols import ModelFamily


class AnalyzerRegistry:
    """Registry of available analyzers.

    Analyzers can be registered by name and retrieved individually
    or filtered by what's valid for a given family.
    """

    _analyzers: dict[str, type] = {}
    _secondary_analyzers: dict[str, type] = {}
    _cross_epoch_analyzers: dict[str, type] = {}

    @classmethod
    def register(cls, analyzer_class: type) -> type:
        """Register an analyzer class.

        Can be used as a decorator:
            @AnalyzerRegistry.register
            class MyAnalyzer:
                name = "my_analyzer"
                ...

        Args:
            analyzer_class: Analyzer class with a 'name' attribute

        Returns:
            The analyzer class (for decorator usage)
        """
        name = getattr(analyzer_class, "name", None)
        if name is None:
            raise ValueError(f"Analyzer {analyzer_class} must have a 'name' attribute")
        cls._analyzers[name] = analyzer_class
        return analyzer_class

    @classmethod
    def get(cls, name: str) -> Analyzer:
        """Get an analyzer instance by name.

        Args:
            name: The analyzer's unique name

        Returns:
            New instance of the analyzer

        Raises:
            KeyError: If analyzer not found
        """
        if name not in cls._analyzers:
            raise KeyError(f"Analyzer '{name}' not found. Available: {list(cls._analyzers.keys())}")
        return cls._analyzers[name]()

    @classmethod
    def register_secondary(cls, analyzer_class: type) -> type:
        """Register a secondary analyzer class.

        Args:
            analyzer_class: SecondaryAnalyzer class with a 'name' attribute

        Returns:
            The analyzer class (for decorator usage)
        """
        name = getattr(analyzer_class, "name", None)
        if name is None:
            raise ValueError(f"Analyzer {analyzer_class} must have a 'name' attribute")
        cls._secondary_analyzers[name] = analyzer_class
        return analyzer_class

    @classmethod
    def get_secondary(cls, name: str) -> SecondaryAnalyzer:
        """Get a secondary analyzer instance by name."""
        if name not in cls._secondary_analyzers:
            raise KeyError(
                f"Secondary analyzer '{name}' not found. "
                f"Available: {list(cls._secondary_analyzers.keys())}"
            )
        return cls._secondary_analyzers[name]()

    @classmethod
    def get_secondary_for_family(
        cls,
        family: ModelFamily,
    ) -> list[SecondaryAnalyzer]:
        """Get all secondary analyzers valid for a family."""
        names = getattr(family, "secondary_analyzers", [])
        return [cls.get_secondary(name) for name in names if name in cls._secondary_analyzers]

    @classmethod
    def register_cross_epoch(cls, analyzer_class: type) -> type:
        """Register a cross-epoch analyzer class.

        Args:
            analyzer_class: CrossEpochAnalyzer class with a 'name' attribute

        Returns:
            The analyzer class (for decorator usage)
        """
        name = getattr(analyzer_class, "name", None)
        if name is None:
            raise ValueError(f"Analyzer {analyzer_class} must have a 'name' attribute")
        cls._cross_epoch_analyzers[name] = analyzer_class
        return analyzer_class

    @classmethod
    def get_cross_epoch(cls, name: str) -> CrossEpochAnalyzer:
        """Get a cross-epoch analyzer instance by name."""
        if name not in cls._cross_epoch_analyzers:
            raise KeyError(
                f"Cross-epoch analyzer '{name}' not found. "
                f"Available: {list(cls._cross_epoch_analyzers.keys())}"
            )
        return cls._cross_epoch_analyzers[name]()

    @classmethod
    def get_cross_epoch_for_family(
        cls,
        family: ModelFamily,
    ) -> list[CrossEpochAnalyzer]:
        """Get all cross-epoch analyzers valid for a family."""
        names = getattr(family, "cross_epoch_analyzers", [])
        return [cls.get_cross_epoch(name) for name in names if name in cls._cross_epoch_analyzers]

    @classmethod
    def get_for_family(cls, family: ModelFamily) -> list[Analyzer]:
        """Get all analyzers valid for a family.

        Args:
            family: ModelFamily instance with 'analyzers' attribute

        Returns:
            List of analyzer instances for analyzers listed in family.analyzers
        """
        return [cls.get(name) for name in family.analyzers if name in cls._analyzers]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered analyzer names.

        Returns:
            List of analyzer names
        """
        return list(cls._analyzers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an analyzer is registered.

        Args:
            name: Analyzer name to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._analyzers

    @classmethod
    def clear(cls) -> None:
        """Clear all registered analyzers. Mainly for testing."""
        cls._analyzers.clear()
        cls._secondary_analyzers.clear()
        cls._cross_epoch_analyzers.clear()


def register_default_analyzers() -> None:
    """Register the built-in analyzers."""
    from miscope.analysis.analyzers.attention_fourier import AttentionFourierAnalyzer
    from miscope.analysis.analyzers.attention_freq import AttentionFreqAnalyzer
    from miscope.analysis.analyzers.attention_patterns import AttentionPatternsAnalyzer
    from miscope.analysis.analyzers.centroid_dmd import CentroidDMD
    from miscope.analysis.analyzers.coarseness import CoarsenessAnalyzer
    from miscope.analysis.analyzers.dominant_frequencies import DominantFrequenciesAnalyzer
    from miscope.analysis.analyzers.effective_dimensionality import EffectiveDimensionalityAnalyzer
    from miscope.analysis.analyzers.fourier_frequency_quality import FourierFrequencyQualityAnalyzer
    from miscope.analysis.analyzers.fourier_nucleation import FourierNucleationAnalyzer
    from miscope.analysis.analyzers.global_centroid_pca import GlobalCentroidPCA
    from miscope.analysis.analyzers.landscape_flatness import LandscapeFlatnessAnalyzer
    from miscope.analysis.analyzers.neuron_activations import NeuronActivationsAnalyzer
    from miscope.analysis.analyzers.neuron_dynamics import NeuronDynamicsAnalyzer
    from miscope.analysis.analyzers.neuron_fourier import NeuronFourierAnalyzer
    from miscope.analysis.analyzers.neuron_freq_clusters import NeuronFreqClustersAnalyzer
    from miscope.analysis.analyzers.parameter_snapshot import ParameterSnapshotAnalyzer
    from miscope.analysis.analyzers.parameter_trajectory_pca import ParameterTrajectoryPCA
    from miscope.analysis.analyzers.repr_geometry import RepresentationalGeometryAnalyzer

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

    AnalyzerRegistry.register_secondary(FourierFrequencyQualityAnalyzer)
    AnalyzerRegistry.register_secondary(NeuronFourierAnalyzer)

    AnalyzerRegistry.register_cross_epoch(ParameterTrajectoryPCA)
    AnalyzerRegistry.register_cross_epoch(NeuronDynamicsAnalyzer)
    AnalyzerRegistry.register_cross_epoch(GlobalCentroidPCA)
    AnalyzerRegistry.register_cross_epoch(CentroidDMD)


# Auto-register default analyzers on import
register_default_analyzers()
