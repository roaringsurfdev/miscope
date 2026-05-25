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
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    import numpy as np
    import torch

    from miscope.architectures import ActivationCache, HookedModel


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


ArtifactScope = Literal["epoch", "all_epochs", "summary"]


@dataclass(frozen=True)
class ArtifactInput:
    """Declares need for an upstream analyzer's artifact.

    Attributes:
        analyzer_name: Name of the upstream analyzer whose artifact is
            consumed. The Planner uses this for dependency resolution.
        scope: At what scope to materialize the artifact:
            - ``"epoch"`` — one epoch's per-epoch artifact dict. The
              pipeline iterates the upstream's completed epochs and
              calls the analyzer once per epoch.
            - ``"all_epochs"`` — the stacked form across all epochs
              (one materialization, used by cross-epoch analyzers).
            - ``"summary"`` — the summary.npz form, if available.
    """

    analyzer_name: str
    scope: ArtifactScope = "epoch"


InputSpec = Union[ModelInput, ArtifactInput]


# ---------------------------------------------------------------------------
# Materialized container passed to .analyze()
# ---------------------------------------------------------------------------


@dataclass
class ResolvedInputs:
    """Materialized inputs for one ``.analyze()`` call.

    The pipeline fills in only those fields whose corresponding inputs
    were declared on the Spec; the rest stay ``None`` or empty.

    Per-epoch fields:
        epoch: The current epoch (``None`` for cross-epoch analyzers).
        model, cache, logits, probe: Populated when a ``ModelInput`` is
            declared. ``cache`` and ``logits`` are ``None`` when the
            ``ModelInput`` opted out of the forward pass.
        artifacts: Per-epoch artifacts at ``scope="epoch"``, keyed by
            upstream analyzer name.

    Cross-epoch fields:
        cross_epoch_artifacts: Stacked artifacts at ``scope="all_epochs"``,
            keyed by upstream name. Each value is the dict from
            :meth:`ArtifactLoader.load`.
        summary_artifacts: Summary stats at ``scope="summary"``, keyed
            by upstream name.
        artifacts_dir: Variant artifacts directory — provided to
            cross-epoch analyzers that load checkpoints/artifacts
            directly (e.g. ``gradient_site``).
        epochs: All available checkpoint epochs (for cross-epoch).
    """

    epoch: int | None = None
    model: HookedModel | None = None
    cache: ActivationCache | None = None
    logits: torch.Tensor | None = None
    probe: torch.Tensor | None = None
    artifacts: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    cross_epoch_artifacts: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    summary_artifacts: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    artifacts_dir: str | None = None
    epochs: tuple[int, ...] | None = None


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


def derive_needs_model_weights(inputs: tuple[InputSpec, ...]) -> bool:
    """True if any declared ModelInput.needs_weights is True."""
    return any(isinstance(i, ModelInput) and i.needs_weights for i in inputs)


def derive_needs_activation_cache(inputs: tuple[InputSpec, ...]) -> bool:
    """True if any declared ModelInput.needs_cache is True."""
    return any(isinstance(i, ModelInput) and i.needs_cache for i in inputs)


def derive_category(inputs: tuple[InputSpec, ...], output_scope: str) -> str:
    """Derive a legacy category label from inputs + output_scope.

    Used for back-compat reporting and as a planner-internal grouping
    key during Phase 2A coexistence:
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
