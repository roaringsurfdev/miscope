"""Protocol definitions for analysis modules.

REQ_121 phases:
    Phase 2A (this state): Legacy protocols (``Analyzer`` /
        ``SecondaryAnalyzer`` / ``CrossEpochAnalyzer``) coexist with a
        new ``UnifiedAnalyzer`` that takes ``(ResolvedInputs, context)``.
    Phase 2B: every analyzer migrated to ``UnifiedAnalyzer``.
    Phase 2C: legacy protocols deleted; ``UnifiedAnalyzer`` renamed to
        ``Analyzer``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import torch

if TYPE_CHECKING:
    from miscope.analysis.inputs import ResolvedInputs
    from miscope.architectures import ActivationCache, HookedModel


@dataclass
class ActivationContext:
    """Single-checkpoint analysis context passed to every primary analyzer.

    Constructed by the pipeline in ``_run_single_epoch``; families are not
    responsible for building it.

    Attributes:
        probe: The full analysis dataset tensor (e.g., all p² (a, b) pairs).
        analysis_params: Family-provided domain context — ``'params'``,
            ``'fourier_basis'``, ``'loss_fn'``, ``'labels'``, and any
            other family-specific precomputed values.
        model: Concrete ``HookedModel`` providing canonical-name weight
            access via ``model.get_weight(canonical_name)``.
        cache: Canonical-name-keyed activation cache from the same
            forward pass. Activations are read as
            ``cache[canonical_name]``.
        logits: Output logits from the same forward pass. Shape is
            architecture-dependent: ``(batch, seq_len, vocab_size)`` for
            transformers, ``(batch, vocab_size)`` for MLPs.
    """

    probe: torch.Tensor
    analysis_params: dict[str, Any]
    model: "HookedModel | None" = None
    cache: "ActivationCache | None" = None
    logits: torch.Tensor | None = None


@dataclass
class AnalysisRunConfig:
    """Configuration for an analysis run.

    Specifies what work the pipeline should perform. This is variant-agnostic;
    the same config can be applied to multiple variants.

    Attributes:
        analyzers: Which analyzers to run (by name). If empty, uses all
            analyzers registered for the variant's family.
        checkpoints: Which checkpoints to analyze. None means all available.
    """

    analyzers: list[str] = field(default_factory=list)
    """Which analyzers to run (by name). Empty list means all family analyzers."""

    checkpoints: list[int] | None = None
    """Which checkpoints to analyze. None means all available."""


@runtime_checkable
class Analyzer(Protocol):
    """Protocol defining the interface for all analyzers.

    Analyzers compute analysis on a single checkpoint and return
    artifact-ready numpy arrays.

    Analyzers receive a context dict prepared by the ModelFamily, which
    contains domain parameters and any precomputed values (e.g., fourier_basis).
    This allows the pipeline to be family-agnostic while analyzers can access
    the domain-specific values they need.

    Optional Summary Statistics (REQ_022):
        Analyzers may optionally implement two additional methods to produce
        summary statistics — small per-epoch values (scalars or small arrays)
        that are accumulated across checkpoints and saved as a single file.

        - get_summary_keys() -> list[str]:
            Declare the summary statistic keys this analyzer produces.
        - compute_summary(result, context) -> dict[str, float | np.ndarray]:
            Compute summary statistics from this epoch's analysis result.

        These methods are NOT part of the required protocol. The pipeline
        detects them via hasattr() to maintain backward compatibility with
        analyzers that only produce per-epoch artifacts.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer (used in artifact naming)."""
        ...

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """
        Run analysis on a single checkpoint.

        Args:
            ctx: Single-checkpoint context bundling the activation bundle,
                 probe tensor, and family-provided analysis parameters.

        Returns:
            Dict mapping artifact keys to numpy arrays.
            Keys become field names in the saved .npz file.
        """
        ...


@runtime_checkable
class SecondaryAnalyzer(Protocol):
    """Protocol for analyzers that derive results from existing per-epoch artifacts.

    Unlike Analyzer (which processes model checkpoints), SecondaryAnalyzers
    run after per-epoch primary analysis completes and consume the resulting
    artifacts to produce new per-epoch artifacts. No model loading occurs.

    Pipeline execution order:
        Phase 1:   Primary (Analyzer)       — model checkpoint → per-epoch artifact
        Phase 1.5: Secondary (this)         — primary artifact → per-epoch artifact
        Phase 2:   Cross-epoch (CrossEpochAnalyzer) — all epochs → cross_epoch.npz

    depends_on declares the single primary analyzer whose artifacts are consumed.
    The pipeline loads one epoch's artifact data and passes it to analyze().

    Results are stored with the same per-epoch pattern as primary analyzers:
        artifacts/{analyzer_name}/epoch_{NNNNN}.npz
    """

    @property
    def name(self) -> str:
        """Unique identifier (used in artifact naming)."""
        ...

    @property
    def depends_on(self) -> str:
        """Name of the primary analyzer whose per-epoch artifacts this consumes."""
        ...

    def analyze(
        self,
        artifact: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Run analysis on a single epoch's artifact data.

        Args:
            artifact: Dict of arrays from the dependency analyzer for this epoch.
            context: Family-provided analysis context (same as primary analyzers).

        Returns:
            Dict mapping artifact keys to numpy arrays.
        """
        ...


@runtime_checkable
class CrossEpochAnalyzer(Protocol):
    """Protocol for analyzers that operate across all checkpoints.

    Unlike Analyzer (which processes one checkpoint at a time),
    CrossEpochAnalyzers run after per-epoch analysis completes and
    consume the resulting artifacts to produce cross-epoch results.

    Examples: PCA trajectory projection, phase transition detection,
    representational similarity across training.

    Results are stored as a single file per analyzer:
        artifacts/{analyzer_name}/cross_epoch.npz
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer (used in artifact naming)."""
        ...

    @property
    def requires(self) -> list[str]:
        """Names of per-epoch analyzers whose artifacts this analyzer consumes."""
        ...

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Run cross-epoch analysis.

        Args:
            artifacts_dir: Root artifacts directory for the variant.
            epochs: Sorted list of available epoch numbers.
            context: Family-provided analysis context (same as per-epoch).

        Returns:
            Dict mapping artifact keys to numpy arrays.
            Keys become field names in the saved cross_epoch.npz file.
        """
        ...


@runtime_checkable
class UnifiedAnalyzer(Protocol):
    """Unified analyzer protocol (REQ_121).

    Replaces ``Analyzer`` / ``SecondaryAnalyzer`` / ``CrossEpochAnalyzer``.
    Each analyzer declares its inputs structurally on its ``SPEC``; the
    pipeline materializes whatever the Spec asks for and hands the
    analyzer a uniform :class:`miscope.analysis.inputs.ResolvedInputs`
    value.

    Output scope (``"per_epoch"`` vs ``"cross_epoch"``) is declared on
    the Spec, not implicit in the protocol. Per-epoch analyzers receive
    one ``ResolvedInputs`` per epoch and return one artifact dict per
    call; cross-epoch analyzers receive a single ``ResolvedInputs``
    covering all epochs and return one artifact dict.

    After Phase 2C, this protocol is renamed to ``Analyzer`` and the
    legacy three protocols above are deleted.
    """

    @property
    def name(self) -> str:
        """Unique identifier (used in artifact naming and Registry keys)."""
        ...

    def analyze(
        self,
        inputs: "ResolvedInputs",
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Run analysis given materialized inputs.

        Args:
            inputs: ``ResolvedInputs`` whose populated fields correspond
                to the analyzer's ``SPEC.inputs`` declaration plus any
                cross-epoch execution context (``artifacts_dir``,
                ``epochs``).
            context: Family-provided analysis context — same as the
                legacy protocols' ``context`` / ``analysis_params``.

        Returns:
            Dict mapping artifact keys to numpy arrays.
        """
        ...
