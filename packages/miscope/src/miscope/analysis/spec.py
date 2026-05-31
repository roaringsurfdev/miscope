"""Analyzer Spec — declarative metadata for analyzers (REQ_120 + REQ_121).

A ``Spec`` is the small declarative object the Planner consults to know
*what an analyzer needs* without inspecting its ``.analyze()`` body.
Specs live next to the analyzers they describe (each module exports a
module-level ``SPEC``) and are registered through the ``@register_analyzer``
decorator in :mod:`miscope.analysis.registry`.

A Spec declares its input materializations structurally via ``inputs`` and
its output scope via ``output_scope``. Capability requirements — upstream
artifacts, whether model weights or an activation cache are needed — are
*derived* from those declarations, not authored separately. The internal
execution phase (the primary/secondary/cross-epoch grouping) is likewise
derived by the Planner via ``derive_category``; it is not part of the Spec's
public surface (REQ_132). The Pipeline materializes whatever the Spec asks
for and hands the analyzer a uniform
:class:`miscope.analysis.inputs.ResolvedInputs` value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from miscope.analysis.inputs import (
    InputSpec,
    derive_needs_activation_cache,
    derive_needs_model_weights,
    derive_required_artifacts,
)

OutputScope = Literal["per_epoch", "cross_epoch"]


@dataclass(frozen=True)
class AnalyzerSpec:
    """Declarative metadata describing an analyzer's needs.

    Author ``inputs`` and ``output_scope``; everything else is derived::

        AnalyzerSpec(
            name="x",
            output_scope="per_epoch",
            inputs=(ModelInput(needs_cache=False),),
            required_hooks=(),
        )

    Attributes:
        name: Unique identifier. Mirrors the analyzer's ``name`` property.
        output_scope: "per_epoch" (one artifact per epoch) or "cross_epoch"
            (one artifact across all epochs). Default "per_epoch".
        inputs: Declared input materializations.
        required_hooks: Canonical hook names the analyzer reads.
        produces_summary: Whether the analyzer implements the
            ``get_summary_keys`` / ``compute_summary`` surface (REQ_022).
    """

    name: str
    output_scope: OutputScope = "per_epoch"
    inputs: tuple[InputSpec, ...] = ()
    required_hooks: tuple[str, ...] = ()
    produces_summary: bool = False

    # ----- Derived properties ----------------------------------------------

    @property
    def requires(self) -> tuple[str, ...]:
        """Names of upstream analyzers — derived from ``inputs``."""
        return derive_required_artifacts(self.inputs)

    @property
    def requires_model_weights(self) -> bool:
        return derive_needs_model_weights(self.inputs)

    @property
    def requires_activation_cache(self) -> bool:
        return derive_needs_activation_cache(self.inputs)
