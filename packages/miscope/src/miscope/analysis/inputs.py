"""Unified input declarations for analyzers (REQ_121).

After REQ_121, every analyzer declares its inputs structurally on the
``Spec`` and receives a single :class:`ResolvedInputs` value in
``.analyze()``. The pipeline materializes whatever the Spec asks for
(model state, upstream artifacts at a chosen scope) and hands the
analyzer a uniform container.

Three input shapes:
    - :class:`ModelInput` — model state at the analyzer's current epoch.
      Primary-only; cross-epoch analyzers do not declare this.
    - :class:`ArtifactInput` — an upstream analyzer's artifact at the
      requested scope (``"epoch"`` for one-epoch dict, ``"all_epochs"``
      for the stacked cross-epoch form, ``"summary"`` for summary stats).

A ``CrossVariantInput`` is reserved for future work — not implemented
in v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from miscope.analysis.deps import ALL, DepsAccessor

if TYPE_CHECKING:
    import torch

    from miscope.architectures import ActivationCache, HookedModel

__all__ = [
    "ALL",
    "ArtifactInput",
    "DepsAccessor",
    "InputSpec",
    "ModelInput",
    "ResolvedInputs",
]


# ---------------------------------------------------------------------------
# Input declarations (authored on AnalyzerSpec.inputs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInput:
    """Declares need for model state at the analyzer's current epoch.

    The pipeline loads the model, runs the forward pass if needed, and
    populates ``ResolvedInputs.model`` / ``.cache`` / ``.logits``.

    Attributes:
        needs_weights: ``True`` if ``.analyze()`` reads ``inputs.model``.
            Almost always ``True`` for primary analyzers today.
        needs_cache: ``True`` if ``.analyze()`` reads ``inputs.cache`` or
            ``inputs.logits``. When ``False``, the pipeline skips
            ``model.run_with_cache(probe)`` for the epoch.
    """

    needs_weights: bool = True
    needs_cache: bool = True


@dataclass(frozen=True)
class ArtifactInput:
    """Declares a dependency on an upstream analyzer's artifact.

    A pure *declaration* of what the analyzer may read (used by the Planner
    for dependency resolution and to scope the analyzer's ``deps`` accessor).
    *How much* of the artifact is read, and at what shape, is chosen at the
    call site via the ``deps`` verb (load_epoch / stream / load_stack /
    load_cross_epoch) — REQ_128.

    Attributes:
        analyzer_name: Name of the upstream analyzer whose artifact is consumed.
    """

    analyzer_name: str


InputSpec = ModelInput | ArtifactInput


# ---------------------------------------------------------------------------
# Materialized container passed to .analyze()
# ---------------------------------------------------------------------------


@dataclass
class ResolvedInputs:
    """Materialized inputs for one ``.analyze()`` call.

    The pipeline fills in only those fields whose corresponding inputs were
    declared on the Spec; the rest stay ``None``.

    Model side (eager — from the forward pass the pipeline runs anyway):
        epoch: The current epoch (``None`` for cross-epoch analyzers).
        model, cache, logits, probe: Populated when a ``ModelInput`` is
            declared. ``cache`` and ``logits`` are ``None`` when the
            ``ModelInput`` opted out of the forward pass.
        epochs: All available checkpoint epochs (for cross-epoch analyzers).

    Artifact side (lazy — REQ_128):
        deps: Scoped accessor over the analyzer's declared ``ArtifactInput``
            upstreams. Analyzers load what they need, when they need it, via
            ``deps.load_epoch`` / ``stream`` / ``load_stack`` /
            ``load_cross_epoch``.

    Generation parameters (REQ_138):
        parameters: Resolved values for the analyzer's declared generation
            parameters, keyed by name. The pipeline resolves each declared
            parameter's binding (run-set override or its declared default) before
            ``analyze()``; reading a name the analyzer did not declare raises
            ``KeyError`` (the runtime half of the parameter discipline). Empty for
            an analyzer that declares no parameters.
    """

    epoch: int | None = None
    model: HookedModel | None = None
    cache: ActivationCache | None = None
    logits: torch.Tensor | None = None
    probe: torch.Tensor | None = None
    epochs: tuple[int, ...] | None = None
    deps: DepsAccessor | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Predicates over a Spec's ``inputs`` declaration
# ---------------------------------------------------------------------------


def derive_required_artifacts(inputs: tuple[InputSpec, ...]) -> tuple[str, ...]:
    """Return the names of upstream analyzers an analyzer reads, in order."""
    seen: dict[str, None] = {}
    for spec in inputs:
        if isinstance(spec, ArtifactInput):
            seen.setdefault(spec.analyzer_name, None)
    return tuple(seen)


def derive_has_model_input(inputs: tuple[InputSpec, ...]) -> bool:
    """True if the analyzer declares any ModelInput.

    Distinct from :func:`derive_needs_model_weights`: a ``ModelInput`` that
    opts out of weights (``needs_weights=False``) still makes this ``True``.
    The Planner uses this to decide a per-epoch analyzer's epoch coverage —
    a model-driven analyzer recomputes at every checkpoint, whereas a purely
    artifact-derived per-epoch analyzer follows its upstreams' epoch set
    (REQ_133, the former "secondary" target strategy).
    """
    return any(isinstance(i, ModelInput) for i in inputs)


def derive_needs_model_weights(inputs: tuple[InputSpec, ...]) -> bool:
    """True if any declared ModelInput.needs_weights is True."""
    return any(isinstance(i, ModelInput) and i.needs_weights for i in inputs)


def derive_needs_activation_cache(inputs: tuple[InputSpec, ...]) -> bool:
    """True if any declared ModelInput.needs_cache is True."""
    return any(isinstance(i, ModelInput) and i.needs_cache for i in inputs)


Category = Literal["primary", "secondary", "cross_epoch"]
"""Legacy execution-phase label. Derived from a Spec's ``inputs`` +
``output_scope`` (REQ_132) — never authored or exposed on the public API
surface.

REQ_133 dissolved "secondary" as an execution phase: the **only structural
axis** is now ``output_scope`` (a 2-valued ``"per_epoch"`` / ``"cross_epoch"``
key), and execution order within each scope is the topological order of the
``ArtifactInput`` DAG. There are zero cross-epoch→per-epoch edges across the
family specs (confirmed by the REQ_133 spike), so the topo-sort never crosses
scopes. ``derive_category`` is retained only for freshness's name-bucketing
(where "secondary" simply folds into the per-epoch bucket); the Planner no
longer branches on it — it branches on :func:`derive_has_model_input`."""


def derive_category(inputs: tuple[InputSpec, ...], output_scope: str) -> Category:
    """Derive the legacy execution-phase label from inputs + output_scope.

    A planner/pipeline-internal grouping key (REQ_132). Post-REQ_133 only
    freshness consults it, and only to split names into per-epoch vs
    cross-epoch buckets ("secondary" folds into per-epoch):
        - ``"cross_epoch"``: output_scope is "cross_epoch".
        - ``"secondary"``: per-epoch output AND every input is artifact-scoped
          (no ModelInput) AND at least one ArtifactInput declared.
        - ``"primary"``: per-epoch output with a ModelInput, OR no inputs at all.
    """
    if output_scope == "cross_epoch":
        return "cross_epoch"
    has_model = any(isinstance(i, ModelInput) for i in inputs)
    has_artifact = any(isinstance(i, ArtifactInput) for i in inputs)
    if has_artifact and not has_model:
        return "secondary"
    return "primary"
