"""Analysis planner — single source of truth for "what would run" (REQ_119).

The Planner consolidates work-decision logic that previously lived in three
places: ``AnalysisPipeline._build_work_queue``, the inline cross-epoch
freshness check inside ``AnalysisPipeline._run_cross_epoch_analyzers``, and
``freshness.cross_epoch_is_stale``. After this module, those decisions live
in one place; the pipeline consumes a ``Plan`` rather than rebuilding the
decision tree, and freshness reporting becomes a presentation layer over the
same logic.

Public surface:
    plan_analysis(variant, analyzers, force=False, checkpoints=None) -> Plan
    Plan, PlanItem (dataclasses)

The Plan is a passive description of what work the pipeline would do given
the current on-disk state. It has no side effects: no model loading, no
analyzer execution, no writes. Reads only artifact-directory metadata.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from miscope.families.variant import Variant


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanItem:
    """Single unit of planned work.

    Phase is determined by which ``Plan`` list this item lives in. Optional
    fields carry phase-specific context: ``depends_on`` for secondary,
    ``requires`` and ``blocked_by`` and ``reason`` for cross-epoch.

    REQ_120 capability flags (``requires_model_weights`` /
    ``requires_activation_cache``) default to ``None`` (unknown) for
    backwards compatibility with hand-constructed Analyzer instances. The
    pipeline treats ``None`` conservatively (assume True).

    Attributes:
        analyzer_name: Identifier used in artifact naming.
        epochs: Epochs to compute. Empty for cross-epoch ``blocked`` items.
        depends_on: For secondary items, name of the primary analyzer
            whose per-epoch artifacts are consumed. ``None`` otherwise.
        requires: For cross-epoch items, names of per-epoch analyzers
            whose artifacts must exist. Empty tuple otherwise.
        blocked_by: For cross-epoch items, names of required analyzers
            with no completed epochs on disk. Non-empty indicates the
            item cannot run as planned.
        reason: For cross-epoch items, short tag describing why the
            item is in the plan (``"missing"`` or ``"stale"``).
        requires_model_weights: From the analyzer's Spec. ``None`` if
            unknown (legacy Analyzer instance without a Spec).
        requires_activation_cache: From the analyzer's Spec. ``None`` if
            unknown. The pipeline uses ``Plan.needs_activation_cache`` to
            decide whether to skip the forward pass.
        required_hooks: From the analyzer's Spec. Used by the pipeline
            for per-architecture compatibility skipping.
    """

    analyzer_name: str
    epochs: tuple[int, ...] = ()
    depends_on: str | None = None
    requires: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    reason: str | None = None
    requires_model_weights: bool | None = None
    requires_activation_cache: bool | None = None
    required_hooks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """Description of analysis work to perform on a variant.

    Three-phase structure mirrors the pipeline:
        - ``per_epoch``: primary analyzers, one item per analyzer with
          missing epochs.
        - ``secondary``: secondary analyzers, one item per analyzer with
          unsatisfied dependency epochs.
        - ``cross_epoch``: cross-epoch analyzers, one item per analyzer
          that is missing, stale, or blocked.

    An empty list in a phase means no work for that phase.
    """

    variant_name: str
    available_checkpoints: tuple[int, ...] = ()
    target_epochs: tuple[int, ...] = ()
    per_epoch: list[PlanItem] = field(default_factory=list)
    secondary: list[PlanItem] = field(default_factory=list)
    cross_epoch: list[PlanItem] = field(default_factory=list)
    transitive_prerequisites: tuple[str, ...] = ()
    """REQ_120: cross-epoch items' missing dependencies that have a known
    Spec in the Registry. Suggested upstream analyzers to enqueue. Empty
    when nothing is blocked or no Specs are registered for the blockers."""

    @property
    def is_empty(self) -> bool:
        """True if no work in any phase."""
        return not (self.per_epoch or self.secondary or self.cross_epoch)

    @property
    def needs_activation_cache(self) -> bool:
        """REQ_120: True if any per-epoch analyzer in this Plan reads ctx.cache.

        Conservative default: items with ``requires_activation_cache=None``
        (legacy Analyzer instances without a Spec) count as needing cache.
        The pipeline uses this aggregate to decide whether to skip
        ``model.run_with_cache(probe)``.
        """
        return any(
            item.requires_activation_cache is None
            or item.requires_activation_cache
            for item in self.per_epoch
        )

    @property
    def needs_model_weights(self) -> bool:
        """REQ_120: True if any per-epoch analyzer reads ctx.model. Conservative
        default: ``None`` counts as ``True``. Today every primary analyzer
        accesses the model, so this is almost always ``True``; the flag
        exists for future analyzers that derive results purely from the probe."""
        return any(
            item.requires_model_weights is None or item.requires_model_weights
            for item in self.per_epoch
        )

    def format(self) -> str:
        """Return a human-readable plan summary."""
        lines = [
            f"Analysis plan: {self.variant_name}",
            f"Checkpoints:   {len(self.available_checkpoints)} available, "
            f"{len(self.target_epochs)} targeted",
            "",
            "Per-epoch analyzers:",
        ]
        if self.per_epoch:
            for item in sorted(self.per_epoch, key=lambda x: x.analyzer_name):
                lines.append(f"  ✗ {item.analyzer_name:<40} {len(item.epochs)} epoch(s)")
        else:
            lines.append("  (nothing to do)")

        lines.append("")
        lines.append("Secondary analyzers:")
        if self.secondary:
            for item in sorted(self.secondary, key=lambda x: x.analyzer_name):
                dep = f" (depends_on={item.depends_on})" if item.depends_on else ""
                lines.append(
                    f"  ✗ {item.analyzer_name:<40} {len(item.epochs)} epoch(s){dep}"
                )
        else:
            lines.append("  (nothing to do)")

        lines.append("")
        lines.append("Cross-epoch analyzers:")
        if self.cross_epoch:
            for item in sorted(self.cross_epoch, key=lambda x: x.analyzer_name):
                if item.blocked_by:
                    label = f"blocked: missing {', '.join(item.blocked_by)}"
                else:
                    label = item.reason or "needs rebuild"
                lines.append(f"  ✗ {item.analyzer_name:<40} {label}")
        else:
            lines.append("  (nothing to do)")

        if self.is_empty:
            lines.append("")
            lines.append("No analysis work to do.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form for logging and (future) dashboard preview."""
        return {
            "variant_name": self.variant_name,
            "available_checkpoints": list(self.available_checkpoints),
            "target_epochs": list(self.target_epochs),
            "per_epoch": [_item_to_dict(item) for item in self.per_epoch],
            "secondary": [_item_to_dict(item) for item in self.secondary],
            "cross_epoch": [_item_to_dict(item) for item in self.cross_epoch],
            "transitive_prerequisites": list(self.transitive_prerequisites),
            "needs_activation_cache": self.needs_activation_cache,
            "needs_model_weights": self.needs_model_weights,
        }


def _item_to_dict(item: PlanItem) -> dict[str, Any]:
    d = asdict(item)
    d["epochs"] = list(item.epochs)
    d["requires"] = list(item.requires)
    d["blocked_by"] = list(item.blocked_by)
    d["required_hooks"] = list(item.required_hooks)
    return d


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plan_analysis(
    variant: Variant,
    analyzers: Sequence[Any],
    force: bool = False,
    checkpoints: Sequence[int] | None = None,
) -> Plan:
    """Build a Plan describing what work the pipeline would perform.

    Args:
        variant: The variant to analyze.
        analyzers: Mixed sequence of ``Analyzer``, ``SecondaryAnalyzer``,
            ``CrossEpochAnalyzer`` instances, *or* ``AnalyzerSpec`` objects
            (REQ_120). Specs are classified by their ``category`` field;
            instances by attribute inspection (``analyze_across_epochs`` →
            cross-epoch, ``depends_on`` → secondary, else primary).
        force: If True, every applicable epoch is included regardless of
            on-disk state.
        checkpoints: Restrict per-epoch work to these epochs. ``None``
            means all available checkpoints. Cross-epoch always runs over
            all available checkpoints (current pipeline behavior).

    Returns:
        Plan describing per-epoch, secondary, and cross-epoch work. When
        Specs are provided, the Plan also carries capability flags per
        item (``requires_model_weights`` / ``requires_activation_cache``)
        and aggregate properties (``Plan.needs_activation_cache``) for
        the pipeline's load-decision optimization.
    """
    artifacts_dir = Path(variant.artifacts_dir)
    available = tuple(sorted(variant.get_available_checkpoints()))
    if checkpoints is not None:
        target_epochs = tuple(sorted(set(checkpoints) & set(available)))
    else:
        target_epochs = available

    per_epoch_items: list[PlanItem] = []
    secondary_items: list[PlanItem] = []
    cross_epoch_items: list[PlanItem] = []

    # Normalize each input into a uniform descriptor regardless of whether
    # it's an analyzer instance or a Spec. The descriptor carries enough
    # info to plan without further inspection.
    descriptors = [_describe(item) for item in analyzers]

    # Classify by category so secondary and cross-epoch planning can treat
    # the primary phase's planned outputs as "will be there" — i.e. the
    # Plan describes post-execution state, not pre-execution state.
    primary_descs = [d for d in descriptors if d.category == "primary"]
    secondary_descs = [d for d in descriptors if d.category == "secondary"]
    cross_epoch_descs = [d for d in descriptors if d.category == "cross_epoch"]

    projected_completed: dict[str, list[int]] = {}
    for desc in primary_descs:
        item = _plan_per_epoch_item(
            name=desc.name,
            artifacts_dir=artifacts_dir,
            target_epochs=target_epochs,
            force=force,
            requires_model_weights=desc.requires_model_weights,
            requires_activation_cache=desc.requires_activation_cache,
            required_hooks=desc.required_hooks,
        )
        if item is not None:
            per_epoch_items.append(item)
        current = set(scan_epoch_files(artifacts_dir / desc.name))
        projected_completed[desc.name] = sorted(current | set(target_epochs))

    for desc in secondary_descs:
        assert desc.depends_on is not None, "secondary descriptor must have depends_on"
        item = _plan_secondary_item(
            name=desc.name,
            depends_on=desc.depends_on,
            artifacts_dir=artifacts_dir,
            force=force,
            projected_completed=projected_completed,
        )
        if item is not None:
            secondary_items.append(item)
        dep_epochs = projected_completed.get(
            desc.depends_on, scan_epoch_files(artifacts_dir / desc.depends_on)
        )
        current = set(scan_epoch_files(artifacts_dir / desc.name))
        projected_completed[desc.name] = sorted(current | set(dep_epochs))

    for desc in cross_epoch_descs:
        item = _plan_cross_epoch_item(
            name=desc.name,
            requires=desc.requires,
            artifacts_dir=artifacts_dir,
            available_epochs=available,
            force=force,
            projected_completed=projected_completed,
        )
        if item is not None:
            cross_epoch_items.append(item)

    transitive = _collect_transitive_prerequisites(cross_epoch_items)

    return Plan(
        variant_name=variant.name,
        available_checkpoints=available,
        target_epochs=target_epochs,
        per_epoch=per_epoch_items,
        secondary=secondary_items,
        cross_epoch=cross_epoch_items,
        transitive_prerequisites=transitive,
    )


# ---------------------------------------------------------------------------
# Input normalization — Spec or Analyzer-instance → uniform descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AnalyzerDescriptor:
    """Internal planning descriptor — uniform shape for Spec or instance input."""

    name: str
    category: str  # "primary" | "secondary" | "cross_epoch"
    requires: tuple[str, ...] = ()
    depends_on: str | None = None
    requires_model_weights: bool | None = None
    requires_activation_cache: bool | None = None
    required_hooks: tuple[str, ...] = ()


def _describe(item: Any) -> _AnalyzerDescriptor:
    """Normalize either an AnalyzerSpec or an analyzer instance into a descriptor.

    Spec input: every field is taken verbatim from the Spec.
    Analyzer instance input: classify via protocol attributes; capability
    flags remain ``None`` (the pipeline defaults to conservative behavior).
    """
    # Import locally to avoid circular import with miscope.analysis.spec.
    from miscope.analysis.spec import AnalyzerSpec

    if isinstance(item, AnalyzerSpec):
        depends_on = item.requires[0] if item.category == "secondary" and item.requires else None
        return _AnalyzerDescriptor(
            name=item.name,
            category=item.category,
            requires=tuple(item.requires),
            depends_on=depends_on,
            requires_model_weights=item.requires_model_weights,
            requires_activation_cache=item.requires_activation_cache,
            required_hooks=tuple(item.required_hooks),
        )

    # Analyzer instance — classify by protocol attribute presence.
    if _is_cross_epoch(item):
        return _AnalyzerDescriptor(
            name=item.name,
            category="cross_epoch",
            requires=tuple(item.requires),
            required_hooks=tuple(getattr(item, "required_hooks", ()) or ()),
        )
    if _is_secondary(item):
        return _AnalyzerDescriptor(
            name=item.name,
            category="secondary",
            requires=(item.depends_on,),
            depends_on=item.depends_on,
        )
    return _AnalyzerDescriptor(
        name=item.name,
        category="primary",
        required_hooks=tuple(getattr(item, "required_hooks", ()) or ()),
    )


def _collect_transitive_prerequisites(
    cross_epoch_items: list[PlanItem],
) -> tuple[str, ...]:
    """Return blocked dependency names that have a registered Spec.

    REQ_120: the Planner can suggest upstream analyzers to enqueue when a
    cross-epoch item is blocked by a missing dependency that the Registry
    knows about. Default surfacing is informational (auto_queue=False).
    """
    # Import locally to avoid eager Registry initialization in test contexts
    # that haven't imported analyzers yet.
    try:
        from miscope.analysis.registry import AnalyzerRegistry
    except ImportError:
        return ()

    suggestions: list[str] = []
    seen: set[str] = set()
    for item in cross_epoch_items:
        for blocker in item.blocked_by:
            if blocker in seen:
                continue
            if AnalyzerRegistry.has_spec(blocker):
                suggestions.append(blocker)
                seen.add(blocker)
    return tuple(suggestions)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _is_cross_epoch(analyzer: Any) -> bool:
    return hasattr(analyzer, "analyze_across_epochs") and hasattr(analyzer, "requires")


def _is_secondary(analyzer: Any) -> bool:
    return hasattr(analyzer, "depends_on")


# ---------------------------------------------------------------------------
# Per-phase planning primitives
#
# These are the single source of truth for "what would run" decisions.
# They are imported by both ``plan_analysis`` (object-typed entry) and
# ``freshness.check_freshness`` (name-typed entry).
# ---------------------------------------------------------------------------


def _plan_per_epoch_item(
    name: str,
    artifacts_dir: Path,
    target_epochs: Sequence[int],
    force: bool,
    requires_model_weights: bool | None = None,
    requires_activation_cache: bool | None = None,
    required_hooks: tuple[str, ...] = (),
) -> PlanItem | None:
    """Decide which target epochs lack a per-epoch artifact for ``name``."""
    if force:
        missing = tuple(target_epochs)
    else:
        completed = set(scan_epoch_files(artifacts_dir / name))
        missing = tuple(e for e in target_epochs if e not in completed)
    if not missing:
        return None
    return PlanItem(
        analyzer_name=name,
        epochs=missing,
        requires_model_weights=requires_model_weights,
        requires_activation_cache=requires_activation_cache,
        required_hooks=required_hooks,
    )


def _plan_secondary_item(
    name: str,
    depends_on: str,
    artifacts_dir: Path,
    force: bool,
    projected_completed: dict[str, list[int]] | None = None,
) -> PlanItem | None:
    """Decide which epochs need a secondary artifact for ``name``.

    Secondary analyzers target the dependency's completed-epoch set rather
    than the variant's full checkpoint set. ``projected_completed`` lets
    the planner treat primary analyzers in the same plan as if they had
    already run — so the Plan describes post-execution state. If the
    dependency has no completed epochs (and is not in ``projected_completed``),
    the item is recorded as blocked.
    """
    projected_completed = projected_completed or {}
    if depends_on in projected_completed:
        dependency_epochs = list(projected_completed[depends_on])
    else:
        dependency_epochs = scan_epoch_files(artifacts_dir / depends_on)
    if not dependency_epochs:
        return PlanItem(
            analyzer_name=name,
            depends_on=depends_on,
            blocked_by=(depends_on,),
        )

    if force:
        target_epochs = tuple(dependency_epochs)
    else:
        completed = set(scan_epoch_files(artifacts_dir / name))
        target_epochs = tuple(e for e in dependency_epochs if e not in completed)

    if not target_epochs:
        return None
    return PlanItem(
        analyzer_name=name,
        epochs=target_epochs,
        depends_on=depends_on,
    )


def _plan_cross_epoch_item(
    name: str,
    requires: tuple[str, ...],
    artifacts_dir: Path,
    available_epochs: tuple[int, ...],
    force: bool,
    projected_completed: dict[str, list[int]] | None = None,
) -> PlanItem | None:
    """Decide whether a cross-epoch analyzer should run.

    Returns:
        None: artifact is fresh — no item emitted.
        PlanItem with non-empty ``blocked_by``: required dependency has no
            completed epochs (per-epoch or cross-epoch).
        PlanItem with ``reason="missing"``: artifact absent.
        PlanItem with ``reason="stale"``: artifact exists but is older than
            its dependencies or available checkpoints.

    ``projected_completed`` lets the planner treat earlier-phase analyzers
    in the same plan as if they had already run — so a cross-epoch is not
    flagged as blocked just because its primary dependency hasn't started.
    """
    projected_completed = projected_completed or {}
    # Blocked-by check: any required analyzer with zero completed epochs.
    # Mirrors AnalysisPipeline.get_completed_epochs semantics: per-epoch
    # files first, falling back to cross_epoch.npz with available_epochs.
    blocked: list[str] = []
    dep_epoch_counts: list[int] = []
    for required in requires:
        if required in projected_completed:
            completed = projected_completed[required]
        else:
            completed = get_completed_epochs(artifacts_dir, required, available_epochs)
        if not completed:
            blocked.append(required)
        else:
            dep_epoch_counts.append(len(completed))

    if blocked:
        return PlanItem(
            analyzer_name=name,
            requires=requires,
            blocked_by=tuple(blocked),
        )

    cross_epoch_path = artifacts_dir / name / "cross_epoch.npz"

    if force or not cross_epoch_path.exists():
        reason = "missing" if not cross_epoch_path.exists() else "forced"
        return PlanItem(
            analyzer_name=name,
            epochs=available_epochs,
            requires=requires,
            reason=reason,
        )

    covered = read_covered_epoch_count(cross_epoch_path)
    if covered < 0:
        # No epoch metadata — conservative rerun.
        return PlanItem(
            analyzer_name=name,
            epochs=available_epochs,
            requires=requires,
            reason="stale",
        )

    # Stale if dependency artifacts cover more epochs than the artifact does,
    # or if available checkpoints exceed what the artifact covers.
    max_dep_epochs = max(dep_epoch_counts) if dep_epoch_counts else 0
    threshold = max(max_dep_epochs, len(available_epochs))
    if threshold > covered:
        return PlanItem(
            analyzer_name=name,
            epochs=available_epochs,
            requires=requires,
            reason="stale",
        )

    return None


# ---------------------------------------------------------------------------
# Disk-state primitives — shared with freshness module
# ---------------------------------------------------------------------------


def scan_epoch_files(analyzer_dir: Path) -> list[int]:
    """Return sorted list of epoch numbers from ``epoch_*.npz`` files."""
    if not analyzer_dir.is_dir():
        return []
    epochs = []
    for fname in os.listdir(analyzer_dir):
        if fname.startswith("epoch_") and fname.endswith(".npz"):
            try:
                epochs.append(int(fname[len("epoch_") : -len(".npz")]))
            except ValueError:
                continue
    return sorted(epochs)


def read_covered_epoch_count(cross_epoch_path: Path) -> int:
    """Return the number of epochs stored in a cross-epoch artifact.

    Returns -1 if the artifact has no 'epochs' key (treated as unknown).
    """
    try:
        with np.load(cross_epoch_path, allow_pickle=False) as data:
            if "epochs" not in data:
                return -1
            return int(data["epochs"].shape[0])
    except Exception:
        return -1


def get_completed_epochs(
    artifacts_dir: Path,
    analyzer_name: str,
    available_epochs: Sequence[int],
) -> list[int]:
    """Return completed epochs for an analyzer.

    Per-epoch ``epoch_*.npz`` files take precedence. If none exist but a
    ``cross_epoch.npz`` does, the supplied ``available_epochs`` are
    returned so cross-epoch-to-cross-epoch dependencies are satisfied.
    This mirrors ``AnalysisPipeline.get_completed_epochs``.
    """
    analyzer_dir = artifacts_dir / analyzer_name
    if not analyzer_dir.is_dir():
        return []

    epochs = scan_epoch_files(analyzer_dir)
    if epochs:
        return epochs

    if (analyzer_dir / "cross_epoch.npz").exists():
        return sorted(available_epochs)

    return []
