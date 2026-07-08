"""Protocol definitions for analysis modules.

A single unified ``Analyzer`` protocol covers every analyzer. Each analyzer
declares its inputs structurally on its ``SPEC`` (see
:mod:`miscope.analysis.spec`); the pipeline materializes whatever the Spec
asks for and hands the analyzer a uniform
:class:`miscope.analysis.inputs.ResolvedInputs` value.
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
    """Single-checkpoint analysis context — retained for the dashboard
    helpers and a handful of view-renderer call sites that still build it
    explicitly. Analyzers themselves consume ``ResolvedInputs`` (REQ_121).

    Attributes:
        probe: The full analysis dataset tensor.
        analysis_params: Family-provided domain context.
        model: Concrete ``HookedModel``.
        cache: Canonical-name-keyed activation cache.
        logits: Output logits from the same forward pass.
    """

    probe: torch.Tensor
    analysis_params: dict[str, Any]
    model: "HookedModel | None" = None
    cache: "ActivationCache | None" = None
    logits: torch.Tensor | None = None


@dataclass
class AnalysisRunConfig:
    """Configuration for an analysis run."""

    analyzers: list[str] = field(default_factory=list)
    """Which analyzers to run (by name). Empty list means all family analyzers."""

    checkpoints: list[int] | None = None
    """Which checkpoints to analyze. None means all available."""


@runtime_checkable
class Analyzer(Protocol):
    """Unified analyzer protocol (REQ_121).

    Each analyzer declares its inputs structurally on its ``SPEC``
    (:class:`miscope.analysis.spec.AnalyzerSpec`); the pipeline
    materializes whatever the Spec asks for and hands the analyzer a
    uniform :class:`miscope.analysis.inputs.ResolvedInputs` value.

    Output scope (``"per_epoch"`` vs ``"cross_epoch"``) is declared on
    the Spec, not implicit in the protocol. Per-epoch analyzers receive
    one ``ResolvedInputs`` per epoch and return one artifact dict per
    call; cross-epoch analyzers receive a single ``ResolvedInputs``
    covering all epochs and return one artifact dict.

    Optional Summary Statistics (REQ_022):
        Analyzers may implement two additional methods to produce
        summary statistics — small per-epoch values accumulated and
        saved as a single file:
        - ``get_summary_keys() -> list[str]``
        - ``compute_summary(result, context) -> dict[str, float | np.ndarray]``
        Detected via ``hasattr``; ``produces_summary=True`` on the Spec.
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
            inputs: ``ResolvedInputs`` populated by the pipeline according
                to the analyzer's ``SPEC.inputs`` declaration. For
                per-epoch analyzers, ``inputs.epoch`` is set and any
                ``ModelInput`` populates ``inputs.model`` / ``inputs.cache``
                / ``inputs.logits``. For cross-epoch analyzers,
                ``inputs.epochs`` is set. Declared ``ArtifactInput`` upstreams
                are read lazily through ``inputs.deps`` (REQ_128).
            context: Family-provided analysis context.

        Returns:
            Dict mapping artifact keys to numpy arrays.
        """
        ...
