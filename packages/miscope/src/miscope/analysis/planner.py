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

    Scope is determined by which ``Plan`` list this item lives in
    (``per_epoch`` or ``cross_epoch`` — REQ_133). Optional fields carry
    context: ``requires``/``depends_on`` name the item's ``ArtifactInput``
    upstreams, ``blocked_by`` names upstreams with no completed epochs, and
    ``reason`` tags why a cross-epoch item is in the plan.

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

    Two-scope structure (REQ_133): the only structural axis is
    ``output_scope``. "Secondary" is no longer a distinct phase — a per-epoch
    analyzer whose inputs are purely artifacts simply appears in ``per_epoch``,
    topologically ordered after its upstreams.

        - ``per_epoch``: every per-epoch analyzer (model-driven or purely
          artifact-derived) with missing epochs, in topological order of the
          ``ArtifactInput`` DAG.
        - ``cross_epoch``: cross-epoch analyzers that are missing, stale, or
          blocked, in topological order of the DAG.

    An empty list in a scope means no work for that scope.
    """

    variant_name: str
    available_checkpoints: tuple[int, ...] = ()
    target_epochs: tuple[int, ...] = ()
    per_epoch: list[PlanItem] = field(default_factory=list)
    cross_epoch: list[PlanItem] = field(default_factory=list)
    transitive_prerequisites: tuple[str, ...] = ()
    """REQ_120: cross-epoch items' missing dependencies that have a known
    Spec in the Registry. Suggested upstream analyzers to enqueue. Empty
    when nothing is blocked or no Specs are registered for the blockers."""

    @property
    def is_empty(self) -> bool:
        """True if no work in any scope."""
        return not (self.per_epoch or self.cross_epoch)

    @property
    def needs_activation_cache(self) -> bool:
        """REQ_120: True if any per-epoch analyzer in this Plan reads ctx.cache.

        Conservative default: items with ``requires_activation_cache=None``
        (legacy Analyzer instances without a Spec) count as needing cache.
        The pipeline uses this aggregate to decide whether to skip
        ``model.run_with_cache(probe)``.
        """
        return any(
            item.requires_activation_cache is None or item.requires_activation_cache
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
                if item.blocked_by:
                    label = f"blocked: missing {', '.join(item.blocked_by)}"
                else:
                    dep = f" (depends_on={item.depends_on})" if item.depends_on else ""
                    label = f"{len(item.epochs)} epoch(s){dep}"
                lines.append(f"  ✗ {item.analyzer_name:<40} {label}")
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
        analyzers: Mixed sequence of ``Analyzer`` instances *or*
            ``AnalyzerSpec`` objects. Specs are classified by their derived
            ``category``; instances by attribute inspection
            (``analyze_across_epochs`` → cross-epoch, ``depends_on`` →
            secondary, else primary).
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
    cross_epoch_items: list[PlanItem] = []

    # Normalize each input into a uniform descriptor regardless of whether
    # it's an analyzer instance or a Spec. The descriptor carries enough
    # info to plan without further inspection.
    descriptors = [_describe(item) for item in analyzers]

    # Split by the single structural axis — ``output_scope`` (REQ_133) — then
    # topologically order each scope by the ``ArtifactInput`` DAG. Execution
    # (and this Plan) describes post-execution state: an upstream planned in
    # the same run is treated as "will be there" via ``projected_completed``.
    per_epoch_descs = _topo_order([d for d in descriptors if d.output_scope == "per_epoch"])
    cross_epoch_descs = _topo_order([d for d in descriptors if d.output_scope == "cross_epoch"])

    projected_completed: dict[str, list[int]] = {}
    for desc in per_epoch_descs:
        covered, blocked = _per_epoch_target_epochs(
            desc, target_epochs, projected_completed, artifacts_dir
        )
        item = _plan_per_epoch_item(desc, artifacts_dir, covered, blocked, force)
        if item is not None:
            per_epoch_items.append(item)
        current = set(scan_epoch_files(artifacts_dir / desc.name))
        projected_completed[desc.name] = sorted(current | set(covered))

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
        # Finding 2 fix: a cross-epoch analyzer, once present or planned,
        # covers every available epoch — seed that so a downstream cross-epoch
        # analyzer reading it (cross→cross edge) is not falsely blocked in the
        # same pass. A *blocked* upstream keeps its real on-disk coverage so
        # its dependents stay blocked too.
        if item is not None and item.blocked_by:
            projected_completed[desc.name] = get_completed_epochs(
                artifacts_dir, desc.name, available
            )
        else:
            projected_completed[desc.name] = list(available)

    transitive = _collect_transitive_prerequisites(cross_epoch_items)

    return Plan(
        variant_name=variant.name,
        available_checkpoints=available,
        target_epochs=target_epochs,
        per_epoch=per_epoch_items,
        cross_epoch=cross_epoch_items,
        transitive_prerequisites=transitive,
    )


# ---------------------------------------------------------------------------
# Input normalization — Spec or Analyzer-instance → uniform descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AnalyzerDescriptor:
    """Internal planning descriptor — uniform shape for Spec or instance input.

    The two fields that drive ordering are ``output_scope`` (the structural
    axis the planner splits on) and ``requires`` (the ``ArtifactInput`` edges
    it topologically sorts). ``has_model_input`` selects a per-epoch analyzer's
    epoch-coverage strategy (REQ_133). ``depends_on`` is retained only for the
    human-readable plan summary.
    """

    name: str
    output_scope: str  # "per_epoch" | "cross_epoch"
    has_model_input: bool = False
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
        from miscope.analysis.inputs import derive_has_model_input
        from miscope.analysis.parameters import reference_default_sources

        has_model = derive_has_model_input(item.inputs)
        # A reference-binding default is a read-dependency (REQ_138): fold its source
        # into the topo-ordering edges. Deduped against ArtifactInput requires, so a
        # reference into an already-declared upstream (the in-scope reference_epoch
        # sites) adds no edge.
        requires = tuple(
            dict.fromkeys((*item.requires, *reference_default_sources(item.parameters)))
        )
        # depends_on is purely cosmetic now: the single upstream of a purely
        # artifact-derived per-epoch analyzer (the former "secondary" shape).
        depends_on = (
            requires[0]
            if item.output_scope == "per_epoch" and not has_model and requires
            else None
        )
        return _AnalyzerDescriptor(
            name=item.name,
            output_scope=item.output_scope,
            has_model_input=has_model,
            requires=requires,
            depends_on=depends_on,
            requires_model_weights=item.requires_model_weights,
            requires_activation_cache=item.requires_activation_cache,
            required_hooks=tuple(item.required_hooks),
        )

    # Analyzer instance — prefer the registered Spec when one exists.
    # REQ_121 migrated analyzers no longer match the legacy protocol
    # attribute set (the method is ``analyze``, not ``analyze_across_epochs``);
    # the registered Spec is the source of truth.
    try:
        from miscope.analysis.registry import AnalyzerRegistry

        if AnalyzerRegistry.has_spec(item.name):
            return _describe(AnalyzerRegistry.get_spec(item.name))
    except (ImportError, AttributeError):
        pass

    # Fallback: classify by protocol attribute presence (legacy analyzers).
    if _is_cross_epoch(item):
        return _AnalyzerDescriptor(
            name=item.name,
            output_scope="cross_epoch",
            requires=tuple(item.requires),
            required_hooks=tuple(getattr(item, "required_hooks", ()) or ()),
        )
    if _is_secondary(item):
        return _AnalyzerDescriptor(
            name=item.name,
            output_scope="per_epoch",
            has_model_input=False,
            requires=(item.depends_on,),
            depends_on=item.depends_on,
        )
    return _AnalyzerDescriptor(
        name=item.name,
        output_scope="per_epoch",
        has_model_input=True,
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
    # Legacy: ``analyze_across_epochs`` method. Migrated (REQ_121):
    # has a ``requires`` list attribute and no ``depends_on``.
    if hasattr(analyzer, "analyze_across_epochs") and hasattr(analyzer, "requires"):
        return True
    if hasattr(analyzer, "requires") and not hasattr(analyzer, "depends_on"):
        return True
    return False


def _is_secondary(analyzer: Any) -> bool:
    return hasattr(analyzer, "depends_on")


# ---------------------------------------------------------------------------
# Per-phase planning primitives
#
# These are the single source of truth for "what would run" decisions.
# They are imported by both ``plan_analysis`` (object-typed entry) and
# ``freshness.check_freshness`` (name-typed entry).
# ---------------------------------------------------------------------------


def _topo_order(descs: list[_AnalyzerDescriptor]) -> list[_AnalyzerDescriptor]:
    """Stable topological sort of descriptors by their ``ArtifactInput`` edges.

    REQ_133: the single ordering primitive for both scopes. An edge exists from
    a descriptor to each upstream named in its ``requires`` that is *also in
    this set* — i.e. a same-scope dependency that must be planned (and executed)
    first so ``projected_completed`` and the resulting execution order are both
    correct. ``requires`` naming an out-of-set analyzer (e.g. a cross-epoch
    analyzer depending on a per-epoch upstream) creates no edge here: that
    upstream lives in the other scope's earlier pass. Input order is preserved
    for independent descriptors. Raises on a cyclic dependency.
    """
    by_name = {d.name: d for d in descs}
    ordered: list[_AnalyzerDescriptor] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(desc: _AnalyzerDescriptor) -> None:
        if desc.name in visited:
            return
        if desc.name in visiting:
            raise ValueError(f"cyclic analyzer dependency involving '{desc.name}'")
        visiting.add(desc.name)
        for upstream in desc.requires:
            if upstream in by_name:
                visit(by_name[upstream])
        visiting.discard(desc.name)
        visited.add(desc.name)
        ordered.append(desc)

    for desc in descs:
        visit(desc)
    return ordered


def _per_epoch_target_epochs(
    desc: _AnalyzerDescriptor,
    target_epochs: Sequence[int],
    projected_completed: dict[str, list[int]],
    artifacts_dir: Path,
) -> tuple[list[int], tuple[str, ...]]:
    """Return ``(covered_epochs, blocked_by)`` for a per-epoch analyzer.

    A model-driven analyzer (or one with no inputs) recomputes at every target
    checkpoint. A purely artifact-derived per-epoch analyzer (the former
    "secondary" shape) instead follows the intersection of its per-epoch
    upstreams' completed epochs, and is blocked when any upstream has none.
    ``projected_completed`` lets an upstream planned in the same run count as
    "will be there". Per the REQ_133 spike, per-epoch analyzers never depend on
    a cross-epoch artifact, so every upstream here is itself per-epoch.
    """
    if desc.has_model_input or not desc.requires:
        return list(target_epochs), ()

    blocked: list[str] = []
    upstream_sets: list[set[int]] = []
    for upstream in desc.requires:
        epochs = projected_completed.get(upstream)
        if epochs is None:
            epochs = scan_epoch_files(artifacts_dir / upstream)
        if not epochs:
            blocked.append(upstream)
        else:
            upstream_sets.append(set(epochs))
    if blocked:
        return [], tuple(blocked)
    covered = sorted(set.intersection(*upstream_sets)) if upstream_sets else []
    return covered, ()


def _plan_per_epoch_item(
    desc: _AnalyzerDescriptor,
    artifacts_dir: Path,
    covered: list[int],
    blocked: tuple[str, ...],
    force: bool,
) -> PlanItem | None:
    """Build a per-epoch PlanItem from its computed coverage, or ``None``.

    ``covered``/``blocked`` come from :func:`_per_epoch_target_epochs`. A
    blocked analyzer yields an item with empty ``epochs`` and a non-empty
    ``blocked_by``. Otherwise the item carries the covered epochs still missing
    on disk (all of them under ``force``); ``None`` when nothing is missing.
    """
    if blocked:
        return PlanItem(
            analyzer_name=desc.name,
            requires=desc.requires,
            depends_on=desc.depends_on,
            blocked_by=blocked,
            requires_model_weights=desc.requires_model_weights,
            requires_activation_cache=desc.requires_activation_cache,
            required_hooks=desc.required_hooks,
        )
    if force:
        missing = tuple(covered)
    else:
        completed = set(scan_epoch_files(artifacts_dir / desc.name))
        missing = tuple(e for e in covered if e not in completed)
    if not missing:
        return None
    return PlanItem(
        analyzer_name=desc.name,
        epochs=missing,
        requires=desc.requires,
        depends_on=desc.depends_on,
        requires_model_weights=desc.requires_model_weights,
        requires_activation_cache=desc.requires_activation_cache,
        required_hooks=desc.required_hooks,
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
