"""Protocol definitions for analysis modules."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class ActivationBundle(Protocol):
    """Architecture-agnostic interface to a model's activations and weights.

    Wraps a (model, cache, logits) tuple for a single forward pass. Analyzers
    call bundle methods instead of accessing TL objects directly, so the same
    analyzer code works against any architecture that provides a bundle.

    Implementations:
        TransformerLensBundle — wraps HookedTransformer + ActivationCache
        (future) MLPBundle    — wraps a plain PyTorch MLP with forward hooks
    """

    def mlp_post(self, layer: int, position: int) -> torch.Tensor:
        """Post-activation MLP neuron values. Returns (batch, d_mlp)."""
        ...

    def residual_stream(self, layer: int, position: int, location: str) -> torch.Tensor:
        """Residual stream at a given site. location: 'resid_pre', 'resid_post', 'attn_out'.
        Returns (batch, d_model)."""
        ...

    def attention_pattern(self, layer: int) -> torch.Tensor:
        """Attention weights (post-softmax). Returns (batch, n_heads, seq_to, seq_from).
        Raises NotImplementedError for non-transformer architectures."""
        ...

    def weight(self, name: str) -> torch.Tensor:
        """Named weight matrix. Supported names: W_E, W_pos, W_Q, W_K, W_V, W_O,
        W_in, W_out, W_U. Raises NotImplementedError for absent weights."""
        ...

    def logits(self, position: int) -> torch.Tensor:
        """Logits at a token position. Returns (batch, vocab_size)."""
        ...

    def supports_site(self, extractor: str) -> bool:
        """Return True if this bundle supports the given extraction type.

        extractor values: 'mlp', 'residual', 'attention'

        Analyzers should call this before attempting extraction to gracefully
        skip sites that the architecture does not provide.
        """
        ...


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
        bundle: ActivationBundle,
        probe: torch.Tensor,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """
        Run analysis on a single checkpoint.

        Args:
            bundle: Architecture-agnostic wrapper over the model and its activations
            probe: The analysis dataset tensor (e.g., full (a, b) grid)
            context: Family-provided analysis context containing:
                - 'params': Domain parameter values (e.g., {'prime': 113, 'seed': 42})
                - Family-specific precomputed values (e.g., 'fourier_basis')

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
