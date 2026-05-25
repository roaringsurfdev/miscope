"""Analyzer Spec — declarative metadata for analyzers (REQ_120).

A ``Spec`` is the small declarative object the Planner consults to know
*what an analyzer needs* without inspecting its ``.analyze()`` body.
Specs live next to the analyzers they describe (each module exports a
module-level ``SPEC``) and are registered through the ``@register_analyzer``
decorator in :mod:`miscope.analysis.registry`.

This REQ preserves the three existing analyzer protocols
(``Analyzer`` / ``SecondaryAnalyzer`` / ``CrossEpochAnalyzer``); ``Spec`` is
metadata, not a fourth protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Category = Literal["primary", "secondary", "cross_epoch"]


@dataclass(frozen=True)
class AnalyzerSpec:
    """Declarative metadata describing an analyzer's needs.

    Attributes:
        name: Unique identifier. Mirrors the analyzer's ``name`` property
            and is the key under which artifacts are stored.
        category: Which of the three protocols this analyzer implements.
        requires: Names of upstream analyzers whose artifacts this
            analyzer consumes. Empty for primary; one item for secondary
            (mirrors ``depends_on``); multi-item for cross-epoch.
        requires_model_weights: ``True`` if ``.analyze()`` reads from
            ``ctx.model`` (only meaningful for primary). Secondary and
            cross-epoch analyzers do not load models — leave ``False``.
        requires_activation_cache: ``True`` if ``.analyze()`` reads from
            ``ctx.cache[...]`` or ``ctx.logits`` (only meaningful for
            primary). When ``False`` for all primary analyzers in a phase,
            the pipeline skips ``model.run_with_cache(probe)``.
        required_hooks: Canonical hook names this analyzer requires the
            model to publish. Empty if the analyzer doesn't read activations.
            The pipeline uses this for per-architecture compatibility
            filtering (analyzers whose required hooks aren't published get
            skipped).
        produces_summary: ``True`` if the analyzer implements the
            optional ``get_summary_keys`` / ``compute_summary`` surface
            (REQ_022) for per-epoch summary statistics.
    """

    name: str
    category: Category
    requires: tuple[str, ...] = ()
    requires_model_weights: bool = True
    requires_activation_cache: bool = True
    required_hooks: tuple[str, ...] = ()
    produces_summary: bool = False
