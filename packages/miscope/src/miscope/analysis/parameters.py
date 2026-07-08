"""Generation-parameter declarations, bindings, and run sets (REQ_138).

A *generation parameter* is a free analyst choice — or an upstream-derived value —
that **selects which artifact you get**. It is part of an artifact's identity, not
its context: two ``parameter_dmd`` runs at different reference epochs are not one
artifact recomputed, they are two distinct analytical objects. This is the cut that
``extra_context`` (one untyped, unrecorded, global ``dict``) failed to make — the
defect class behind the REQ_133 p101 regression, where a byte-affecting value was
supplied through an unrecorded side-channel and an unpinned recompute silently fell
back to a different default.

This module supplies the vocabulary; :mod:`miscope.analysis.recipe` does the
projection/resolution and :mod:`miscope.warehouse.run_sets` persists run sets.

The pieces
----------
- :class:`ParameterSpec` — declared on :class:`~miscope.analysis.spec.AnalyzerSpec`
  (``parameters=(...)``), parallel to the ``outputs`` discipline: name, dtype, scope
  (run-level vs analyzer-local), and a **default binding**.
- :class:`Binding` — :class:`LiteralBinding` (an analyst's free value) or
  :class:`ReferenceBinding` (a path into an upstream artifact field, resolved at run
  time). A default is itself a binding — there is no "unrecorded default", which is
  what makes the p101 failure mode structurally impossible.
- Selectors for reference bindings (kept minimal, grow on demand — REQ_138 OQ #2):
  :class:`Reducer` (reduce an upstream's *epoch inventory* to one epoch) and
  :class:`FieldIndex` (positional index into an upstream *output field*).
- :class:`Parameterization` — the researcher-facing run set: an ordered set of
  bindings plus an optional label. The empty parameterization is the default and
  addresses to today's storage path (see :mod:`miscope.analysis.recipe`).

Scope
-----
A binding is **run-level** (``analyzer is None``) — declared once at kickoff and
matched to every analyzer that declares a parameter of that name — or
**analyzer-local** (``analyzer == "<name>"``) — targeting one analyzer's declared
parameter. Analyzer-local addressing is what lets a run set pin
``parameter_dmd``'s reference epoch *alone* (the narrow-parameter proving ground)
without touching the two other analyzers that happen to declare the same parameter
name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

__all__ = [
    "ParameterScope",
    "Reducer",
    "FieldIndex",
    "Selector",
    "LiteralBinding",
    "ReferenceBinding",
    "Binding",
    "Parameterization",
    "EMPTY_PARAMETERIZATION",
    "ParameterSpec",
    "binding_key",
    "reference_default_sources",
]

ParameterScope = Literal["run", "analyzer"]


# ---------------------------------------------------------------------------
# Selectors — how a reference binding picks a value out of an upstream
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reducer:
    """Reduce an upstream analyzer's *epoch inventory* to a single epoch.

    The inventory-derived "last/max checkpoint" default. Its resolution depends on
    the checkpoint inventory, which is **mutable** (training can be extended), so a
    ``Reducer`` default floats: when training extends and ``max_epoch`` re-resolves
    to a later value, the floating-default artifact is replanned (REQ_138 freshness).
    """

    kind: Literal["max_epoch", "min_epoch"]


@dataclass(frozen=True)
class FieldIndex:
    """Positional index into an upstream analyzer's output field array.

    The derived-parameter walk: e.g. ``activation_dmd.regime_boundaries[0]`` pins a
    downstream window to a DMD-discovered regime boundary. Resolved at run time by
    reading the upstream field through the deps accessor.
    """

    field: str
    index: int


Selector = Union[Reducer, FieldIndex]  # noqa: UP007


# ---------------------------------------------------------------------------
# Bindings — a parameter bound to a value or an upstream reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiteralBinding:
    """An analyst's free, pinned value for a parameter.

    A literal pins a value that does not float with the checkpoint inventory: an
    explicitly pinned earlier epoch is *not* invalidated by training extension
    (its value still exists) — unlike a :class:`Reducer` default.
    """

    name: str
    value: Any
    analyzer: str | None = None
    """Consuming analyzer for analyzer-local scope; ``None`` for run-level."""


@dataclass(frozen=True)
class ReferenceBinding:
    """A parameter bound to a path into an upstream artifact (resolved at run time).

    ``source_analyzer`` is a real DAG edge: the consuming analyzer is ordered after
    it (REQ_133 topo-order), and the downstream recipe transitively includes the
    upstream's recipe.
    """

    name: str
    source_analyzer: str
    selector: Selector
    analyzer: str | None = None
    """Consuming analyzer for analyzer-local scope; ``None`` for run-level."""


Binding = Union[LiteralBinding, ReferenceBinding]  # noqa: UP007


def binding_key(binding: Binding) -> tuple[str | None, str]:
    """Identity of the parameter a binding targets: ``(analyzer, name)``.

    Run-level bindings key on ``(None, name)``; analyzer-local on
    ``("<analyzer>", name)``. Used to match a run set's bindings to declared
    :class:`ParameterSpec` parameters and to dedupe within a parameterization.
    """
    return (binding.analyzer, binding.name)


# ---------------------------------------------------------------------------
# ParameterSpec — declared on AnalyzerSpec.parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterSpec:
    """One generation parameter an analyzer consumes (declared on its Spec).

    Mirrors the :class:`~miscope.analysis.output_schema.OutputField` discipline: the
    analyzer declares what it reads, and reading an undeclared parameter at run time
    raises (the parameters mapping handed to ``analyze()`` is scoped to exactly the
    declared names — see :mod:`miscope.analysis.recipe`).

    Attributes:
        name: Parameter name; the key under ``inputs.parameters``.
        dtype: Element dtype string (``"int64"``, ``"float64"``, ``"str"`` …),
            validated at registry load against numpy like ``OutputField.dtype``.
        scope: ``"run"`` (declared at kickoff, propagates to every analyzer that
            declares the name) or ``"analyzer"`` (this analyzer's local parameter).
        default: The binding used when a run set supplies none. A literal value or
            a reference (e.g. ``Reducer("max_epoch")`` over an upstream). Never a
            silent code fallback.
    """

    name: str
    dtype: str
    scope: ParameterScope
    default: Binding


def reference_default_sources(parameters: tuple[ParameterSpec, ...]) -> tuple[str, ...]:
    """Upstream analyzers an analyzer reads via *reference-binding defaults*.

    A reference default is a real read-dependency (the resolver loads the source's
    inventory or a field), so it is a DAG edge the planner must order on — even when
    the source is not already an :class:`~miscope.analysis.inputs.ArtifactInput`. For
    the in-scope ``reference_epoch`` sites the source (``neuron_grouping``) is already
    an ArtifactInput, so this adds no new edge there; it keeps the derived-parameter
    walk (a reference into a *new* source) correct without a later refactor.
    """
    seen: dict[str, None] = {}
    for p in parameters:
        if isinstance(p.default, ReferenceBinding):
            seen.setdefault(p.default.source_analyzer, None)
    return tuple(seen)


# ---------------------------------------------------------------------------
# Parameterization — the run set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameterization:
    """A run set: an ordered set of bindings plus an optional human label.

    The empty parameterization is the default — every parameter resolves to its
    declared default, which addresses to today's storage path (no recipe segment).
    The researcher-facing handle and comparison/query unit (REQ_138); persisted in
    the run-set registry.
    """

    bindings: tuple[Binding, ...] = ()
    label: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.bindings

    def binding_for(self, analyzer: str, name: str) -> Binding | None:
        """The run set's binding for an analyzer's parameter, if any.

        An analyzer-local binding (``analyzer == "<name>"``) takes precedence over a
        run-level one (``analyzer is None``) of the same name, so a run can set a
        broad default and override it for one analyzer.
        """
        local = self._lookup(analyzer, name)
        return local if local is not None else self._lookup(None, name)

    def _lookup(self, analyzer: str | None, name: str) -> Binding | None:
        for b in self.bindings:
            if b.analyzer == analyzer and b.name == name:
                return b
        return None


EMPTY_PARAMETERIZATION = Parameterization()
