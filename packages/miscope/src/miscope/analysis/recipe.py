"""Recipe = the projection of a run set's bindings onto an artifact's closure (REQ_138).

The single storage rule:

    An artifact's **recipe** is the projection of the run set's bindings onto that
    artifact's **transitive upstream closure**.
      - empty projection  -> parameter-independent -> one shared copy (today's path)
      - projection matches an already-stored set -> reuse
      - projection is new -> write under the new recipe

A run set (:class:`~miscope.analysis.parameters.Parameterization`) holds only the
analyst-supplied (non-default) bindings, so the default/empty run set projects to an
**empty recipe for every analyzer** — which addresses to today's storage path, so the
existing artifacts never relocate (REQ_138 near-zero-migration). A binding lands in
analyzer *A*'s recipe when it targets a parameter declared by *A* or by any analyzer
in *A*'s transitive upstream closure; transitivity (a downstream recipe includes its
upstream's recipe) therefore falls out of the closure walk, not a separate fold.

Two pieces beyond the projection:

- :func:`recipe_signature` — addresses a recipe by its **binding spec**, not the
  resolved value (REQ_138 OQ #4 lean): a literal's value, a reference's
  ``(source, selector)``. The *resolved* value is recorded in the run-set registry
  for provenance and floating-default staleness, not in the address. Stable across
  runs/environments via canonical JSON (OQ #1: ints exact, floats quantized).
- :class:`RecipeResolver` — turns a binding into a concrete value at run time: a
  literal passes through; a :class:`~miscope.analysis.parameters.Reducer` reads the
  source's epoch inventory; a :class:`~miscope.analysis.parameters.FieldIndex` reads
  the source's output field. A default is a binding, so it resolves the same way —
  there is no unrecorded code fallback (the p101 fix).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miscope.analysis.parameters import (
    Binding,
    FieldIndex,
    LiteralBinding,
    Parameterization,
    Reducer,
    ReferenceBinding,
    binding_key,
    reference_default_sources,
)

if TYPE_CHECKING:
    from miscope.analysis.artifact_loader import ArtifactLoader
    from miscope.analysis.spec import AnalyzerSpec

_FLOAT_QUANTIZE = 12  # decimal places — OQ #1 stable float canonicalization


@dataclass(frozen=True)
class Recipe:
    """An artifact's storage recipe: the relevant non-default bindings, sorted.

    The empty recipe (no relevant bindings) is the canonical/default address —
    today's storage path. A non-empty recipe gets a stable :meth:`signature`
    segment so coexisting parameterizations never overwrite one another.
    """

    bindings: tuple[Binding, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.bindings

    def signature(self) -> str:
        """Stable short hash of the recipe's binding specs (empty -> ``""``)."""
        return recipe_signature(self)


def project_recipe(
    analyzer_name: str,
    parameterization: Parameterization,
    specs_by_name: dict[str, AnalyzerSpec],
) -> Recipe:
    """Project a run set's bindings onto ``analyzer_name``'s transitive closure.

    A binding is in the recipe iff some analyzer in ``closure ∪ {analyzer_name}``
    declares a parameter of that name and the binding's scope matches (run-level
    applies to any such analyzer; analyzer-local only to its named analyzer).
    """
    if parameterization.is_empty:
        return Recipe()
    closure = _closure(analyzer_name, specs_by_name) | {analyzer_name}
    # (analyzer, param_name) pairs declared anywhere in the closure.
    declared: set[tuple[str, str]] = {
        (name, p.name)
        for name in closure
        if name in specs_by_name
        for p in specs_by_name[name].parameters
    }
    relevant = [b for b in parameterization.bindings if _is_relevant(b, declared)]
    return Recipe(bindings=_canonical_order(relevant))


def _is_relevant(binding: Binding, declared: set[tuple[str, str]]) -> bool:
    """Whether a binding targets a parameter declared in the closure."""
    if binding.analyzer is not None:  # analyzer-local: exact (analyzer, name) match
        return (binding.analyzer, binding.name) in declared
    return any(name == binding.name for _, name in declared)  # run-level: any declarer


def _closure(analyzer_name: str, specs_by_name: dict[str, AnalyzerSpec]) -> set[str]:
    """Transitive upstream closure via ArtifactInput + reference-default edges."""
    seen: set[str] = set()
    stack = list(_edges(analyzer_name, specs_by_name))
    while stack:
        up = stack.pop()
        if up in seen:
            continue
        seen.add(up)
        stack.extend(_edges(up, specs_by_name))
    return seen


def _edges(analyzer_name: str, specs_by_name: dict[str, AnalyzerSpec]) -> tuple[str, ...]:
    """Direct upstream edges of one analyzer: declared inputs + reference defaults."""
    spec = specs_by_name.get(analyzer_name)
    if spec is None:
        return ()
    return tuple(dict.fromkeys((*spec.requires, *reference_default_sources(spec.parameters))))


def _canonical_order(bindings: list[Binding]) -> tuple[Binding, ...]:
    """Sort bindings by ``(analyzer, name)`` so a recipe's identity is order-free."""
    return tuple(sorted(bindings, key=lambda b: (b.analyzer or "", b.name)))


# ---------------------------------------------------------------------------
# Signature — address by binding spec (OQ #4), canonical + stable (OQ #1)
# ---------------------------------------------------------------------------


def recipe_signature(recipe: Recipe) -> str:
    """Short stable hash of a recipe's binding specs; ``""`` for the empty recipe."""
    if recipe.is_empty:
        return ""
    payload = [_binding_repr(b) for b in _canonical_order(list(recipe.bindings))]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _binding_repr(binding: Binding) -> dict[str, Any]:
    analyzer, name = binding_key(binding)
    base = {"analyzer": analyzer, "name": name}
    if isinstance(binding, LiteralBinding):
        return {**base, "kind": "literal", "value": _canon_value(binding.value)}
    return {
        **base,
        "kind": "reference",
        "source": binding.source_analyzer,
        "selector": _selector_repr(binding.selector),
    }


def _selector_repr(selector: object) -> dict[str, Any]:
    if isinstance(selector, Reducer):
        return {"reducer": selector.kind}
    if isinstance(selector, FieldIndex):
        return {"field": selector.field, "index": selector.index}
    raise TypeError(f"unknown selector type: {type(selector).__name__}")


def _canon_value(value: Any) -> Any:
    """Canonical, hashable representation of a literal value (OQ #1)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, _FLOAT_QUANTIZE)
    if isinstance(value, (list, tuple)):
        return [_canon_value(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Resolution — a binding -> a concrete value at run time
# ---------------------------------------------------------------------------


class RecipeResolutionError(RuntimeError):
    """Raised when a reference binding cannot be resolved against the artifacts."""


class RecipeResolver:
    """Resolves bindings to concrete values against a variant's artifacts.

    A literal passes through; a :class:`Reducer` reduces the source's epoch
    inventory; a :class:`FieldIndex` reads the source's (cross-epoch) output field.
    Defaults resolve the same way — no code fallback — which is what makes the p101
    silent-default divergence structurally impossible.
    """

    def __init__(self, loader: ArtifactLoader):
        self._loader = loader

    def resolve(self, binding: Binding) -> Any:
        if isinstance(binding, LiteralBinding):
            return binding.value
        selector = binding.selector
        if isinstance(selector, Reducer):
            return self._resolve_reducer(binding.source_analyzer, selector)
        if isinstance(selector, FieldIndex):
            return self._resolve_field_index(binding.source_analyzer, selector)
        raise RecipeResolutionError(f"unknown selector type: {type(selector).__name__}")

    def _resolve_reducer(self, source: str, selector: Reducer) -> int:
        epochs = self._loader.get_epochs(source)
        if not epochs:
            raise RecipeResolutionError(
                f"cannot resolve {selector.kind} over '{source}': no epochs on disk."
            )
        return int(max(epochs) if selector.kind == "max_epoch" else min(epochs))

    def _resolve_field_index(self, source: str, selector: FieldIndex) -> Any:
        try:
            data = self._loader.load_cross_epoch(source, fields=[selector.field])
        except FileNotFoundError as exc:
            raise RecipeResolutionError(
                f"cannot resolve {source}.{selector.field}[{selector.index}]: "
                f"no cross-epoch artifact for '{source}'."
            ) from exc
        return data[selector.field][selector.index]
