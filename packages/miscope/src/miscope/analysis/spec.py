"""Analyzer Spec — declarative metadata for analyzers (REQ_120 + REQ_121).

A ``Spec`` is the small declarative object the Planner consults to know
*what an analyzer needs* without inspecting its ``.analyze()`` body.
Specs live next to the analyzers they describe (each module exports a
module-level ``SPEC``) and are registered through the ``@register_analyzer``
decorator in :mod:`miscope.analysis.registry`.

REQ_121 introduces a unified ``inputs`` declaration alongside the
legacy ``category`` field. During Phase 2A coexistence, Specs may
author either:

    - Legacy style (REQ_120): ``category="primary"|"secondary"|"cross_epoch"``
      plus ``requires=(...)``, ``requires_model_weights=...``, etc.
    - Unified style (REQ_121): ``output_scope="per_epoch"|"cross_epoch"``
      plus ``inputs=(ModelInput(...), ArtifactInput(...), ...)``. Capability
      flags and ``requires`` become derived properties of ``inputs``.

The Pipeline routes unified-style Specs through a single ``.analyze(inputs,
context)`` path; legacy-style Specs continue to use the three-protocol
dispatcher until Phase 2C retires it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from miscope.analysis.inputs import (
    InputSpec,
    derive_category,
    derive_needs_activation_cache,
    derive_needs_model_weights,
    derive_required_artifacts,
)

Category = Literal["primary", "secondary", "cross_epoch"]
OutputScope = Literal["per_epoch", "cross_epoch"]


@dataclass(frozen=True)
class AnalyzerSpec:
    """Declarative metadata describing an analyzer's needs.

    Phase 2A authoring modes:

    Legacy (REQ_120) — keep ``inputs=()`` and author flags directly::

        AnalyzerSpec(
            name="x",
            category="primary",
            requires_model_weights=True,
            requires_activation_cache=False,
            required_hooks=("blocks.0.mlp.hook_out",),
        )

    Unified (REQ_121) — declare ``inputs``; flags and ``requires``
    become derived::

        AnalyzerSpec(
            name="x",
            output_scope="per_epoch",
            inputs=(ModelInput(needs_cache=False),),
            required_hooks=(),
        )

    The Pipeline detects unified Specs by ``bool(spec.inputs) or
    spec.output_scope != "per_epoch"`` (the default) — i.e. authoring
    *any* input declaration or non-default output scope opts the
    analyzer into the unified path.

    Attributes:
        name: Unique identifier. Mirrors the analyzer's ``name`` property.
        category: *Legacy authored field.* One of "primary", "secondary",
            "cross_epoch". Optional in unified style — derived from
            ``inputs`` + ``output_scope`` when omitted.
        output_scope: *Unified field.* "per_epoch" (one artifact per
            epoch) or "cross_epoch" (one artifact across all epochs).
            Default "per_epoch".
        inputs: *Unified field.* Declared input materializations.
            Empty for legacy Specs.
        requires: *Legacy authored field.* Names of upstream analyzers.
            Derived from ``inputs`` when ``inputs`` is non-empty.
        requires_model_weights / requires_activation_cache: *Legacy
            authored flags.* Derived from ``ModelInput`` entries in
            ``inputs`` when ``inputs`` is non-empty.
        required_hooks: Canonical hook names the analyzer reads.
        produces_summary: Whether the analyzer implements the
            ``get_summary_keys`` / ``compute_summary`` surface (REQ_022).
    """

    name: str
    category: Category | None = None
    output_scope: OutputScope = "per_epoch"
    inputs: tuple[InputSpec, ...] = ()
    requires: tuple[str, ...] = ()
    requires_model_weights: bool = True
    requires_activation_cache: bool = True
    required_hooks: tuple[str, ...] = ()
    produces_summary: bool = False

    # ----- Derived properties ----------------------------------------------

    @property
    def is_unified(self) -> bool:
        """True if this Spec was authored in REQ_121's unified style.

        Detection: ``category`` is ``None`` (legacy SPECs always author
        ``category``; unified SPECs author ``inputs`` + ``output_scope``).
        """
        return self.category is None

    @property
    def effective_category(self) -> Category:
        """The category to use for planner/pipeline grouping.

        Unified Specs derive from ``inputs`` + ``output_scope``; legacy
        Specs return their authored ``category``. A missing category
        on a legacy Spec is a programmer error caught at registration.
        """
        if self.is_unified:
            return derive_category(self.inputs, self.output_scope)  # type: ignore[return-value]
        assert self.category is not None, (
            f"Spec for {self.name!r} must declare either ``category`` (legacy) "
            f"or ``inputs`` / ``output_scope`` (unified)."
        )
        return self.category

    @property
    def effective_requires(self) -> tuple[str, ...]:
        """Names of upstream analyzers — derived from inputs if unified."""
        if self.is_unified:
            return derive_required_artifacts(self.inputs)
        return self.requires

    @property
    def effective_requires_model_weights(self) -> bool:
        if self.is_unified:
            return derive_needs_model_weights(self.inputs)
        return self.requires_model_weights

    @property
    def effective_requires_activation_cache(self) -> bool:
        if self.is_unified:
            return derive_needs_activation_cache(self.inputs)
        return self.requires_activation_cache


