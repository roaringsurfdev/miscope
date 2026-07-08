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

from miscope.analysis import signature as sig_mod
from miscope.analysis.artifact_loader import read_signature_manifest
from miscope.analysis.signature import CROSS_EPOCH_KEY, SigRecord

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

    signatures: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    """REQ_145: the post-run provenance signatures the pipeline stamps after a
    successful write. Keyed ``{analyzer_name: {epoch_str | CROSS_EPOCH_KEY ->
    SigRecord-as-json}}`` for every *planned* node (one entry per covered epoch).
    Computed once here so the planner's recompute decision and the pipeline's
    write-time stamp use one source — they cannot diverge. Internal execution
    detail: omitted from ``format()`` / ``to_dict()``."""

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
    recipe_map: dict[str, str] | None = None,
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

    # REQ_145: signature predicate state. ``projected`` carries each node's
    # post-run SigRecord per epoch so a downstream folds in its upstreams'
    # (projected-if-recomputed, else stored) signatures — forward propagation
    # does the invalidation. Checkpoint fingerprints are read once.
    ctx = _SigContext(
        artifacts_dir=artifacts_dir,
        recipe_map=recipe_map or {},
        checkpoint_fps={e: variant.checkpoint_fingerprint(e) for e in target_epochs},
        force=force,
    )
    plan_signatures: dict[str, dict[str, dict[str, Any]]] = {}

    projected_completed: dict[str, list[int]] = {}
    for desc in per_epoch_descs:
        covered, blocked = _per_epoch_target_epochs(
            desc, target_epochs, projected_completed, artifacts_dir, recipe_map
        )
        node_proj, recompute, reason = _signature_plan_per_epoch(ctx, desc, covered, blocked)
        ctx.projected[desc.name] = node_proj
        item = _plan_per_epoch_item(desc, recompute, blocked, reason)
        if item is not None:
            per_epoch_items.append(item)
            plan_signatures[desc.name] = {k: r.to_json() for k, r in node_proj.items()}
        current = set(scan_epoch_files(_scoped_dir(artifacts_dir, desc.name, recipe_map)))
        projected_completed[desc.name] = sorted(current | set(covered))

    for desc in cross_epoch_descs:
        item = _plan_cross_epoch_item(ctx, desc, available, projected_completed)
        if item is not None:
            cross_epoch_items.append(item)
            if not item.blocked_by and desc.name in ctx.projected:
                plan_signatures[desc.name] = {
                    k: r.to_json() for k, r in ctx.projected[desc.name].items()
                }
        # Finding 2 fix: a cross-epoch analyzer, once present or planned,
        # covers every available epoch — seed that so a downstream cross-epoch
        # analyzer reading it (cross→cross edge) is not falsely blocked in the
        # same pass. A *blocked* upstream keeps its real on-disk coverage so
        # its dependents stay blocked too.
        if item is not None and item.blocked_by:
            projected_completed[desc.name] = get_completed_epochs(
                artifacts_dir, desc.name, available, recipe_map
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
        signatures=plan_signatures,
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
    version: int = 1  # REQ_145: AnalyzerSpec.version (code-version signature component)


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
            requires[0] if item.output_scope == "per_epoch" and not has_model and requires else None
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
            version=item.version,
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
    # ``version`` defaults to 1 but is read off the instance when present so a
    # version-bearing legacy analyzer participates in the signature predicate.
    version = int(getattr(item, "version", 1))
    if _is_cross_epoch(item):
        return _AnalyzerDescriptor(
            name=item.name,
            output_scope="cross_epoch",
            requires=tuple(item.requires),
            required_hooks=tuple(getattr(item, "required_hooks", ()) or ()),
            version=version,
        )
    if _is_secondary(item):
        return _AnalyzerDescriptor(
            name=item.name,
            output_scope="per_epoch",
            has_model_input=False,
            requires=(item.depends_on,),
            depends_on=item.depends_on,
            version=version,
        )
    return _AnalyzerDescriptor(
        name=item.name,
        output_scope="per_epoch",
        has_model_input=True,
        required_hooks=tuple(getattr(item, "required_hooks", ()) or ()),
        version=version,
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
# Signature predicate (REQ_145) — recompute iff the input-derived signature changed
# ---------------------------------------------------------------------------


class _SigContext:
    """Carries the signature predicate's state across the planner's topo pass.

    ``projected`` holds each node's post-run :class:`SigRecord`s keyed by epoch
    (``str(epoch)`` or ``CROSS_EPOCH_KEY``) — the forward-propagation channel: a
    downstream reads its upstreams' projected (if recomputed) or stored (if fresh)
    signatures, so a changed upstream flows into a changed downstream signature.
    ``stored`` reads (and caches) each analyzer's on-disk signature manifest;
    ``checkpoint_fps`` is the per-epoch checkpoint fingerprint read once up front.
    """

    def __init__(
        self,
        artifacts_dir: Path,
        recipe_map: dict[str, str],
        checkpoint_fps: dict[int, str],
        force: bool,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.recipe_map = recipe_map
        self.checkpoint_fps = checkpoint_fps
        self.force = force
        self.projected: dict[str, dict[str, SigRecord]] = {}
        self._manifest_cache: dict[str, dict[str, SigRecord]] = {}

    def recipe_sig(self, name: str) -> str:
        return self.recipe_map.get(name, "")

    def stored(self, name: str) -> dict[str, SigRecord]:
        """Rehydrated on-disk signature manifest for ``name`` (cached, ``{}`` if absent)."""
        if name not in self._manifest_cache:
            raw = read_signature_manifest(str(self.artifacts_dir), name, self.recipe_sig(name))
            self._manifest_cache[name] = {
                key: rec
                for key, value in raw.items()
                if (rec := SigRecord.from_json(value)) is not None
            }
        return self._manifest_cache[name]

    def upstream_sig(self, upstream: str, key: str) -> str:
        """Projected (if planned) else stored signature for one upstream at one epoch."""
        proj = self.projected.get(upstream)
        if proj is not None and key in proj:
            return proj[key].sig
        rec = self.stored(upstream).get(key)
        return rec.sig if rec is not None else ""


def _would_per_epoch(ctx: _SigContext, desc: _AnalyzerDescriptor, epoch: int) -> SigRecord:
    """The signature a per-epoch node's artifact *would* carry after this run.

    Model-driven primary → keyed by the checkpoint fingerprint (skippable when the
    checkpoint and code/recipe are unchanged). Artifact-derived → keyed by its
    upstreams' signatures at this epoch. No-input → code/recipe only.
    """
    recipe = ctx.recipe_sig(desc.name)
    if desc.has_model_input:
        checkpoint = ctx.checkpoint_fps.get(epoch, f"{epoch}:absent")
        return sig_mod.build_record(code_version=desc.version, recipe=recipe, checkpoint=checkpoint)
    if not desc.requires:
        return sig_mod.build_record(code_version=desc.version, recipe=recipe)
    upstream = [ctx.upstream_sig(u, str(epoch)) for u in desc.requires]
    return sig_mod.build_record(code_version=desc.version, recipe=recipe, upstream_sigs=upstream)


def _signature_plan_per_epoch(
    ctx: _SigContext,
    desc: _AnalyzerDescriptor,
    covered: list[int],
    blocked: tuple[str, ...],
) -> tuple[dict[str, SigRecord], list[int], str | None]:
    """Decide which covered epochs to recompute by signature, and why.

    Returns ``(node_proj, recompute_epochs, reason)``. ``node_proj`` is the
    post-run signature for *every* covered epoch (fresh ones keep their value),
    so a downstream folds in the correct projection regardless of what reruns.
    """
    if blocked:
        return {}, [], None
    stored = ctx.stored(desc.name)
    node_proj: dict[str, SigRecord] = {}
    recompute: list[int] = []
    for epoch in covered:
        would = _would_per_epoch(ctx, desc, epoch)
        node_proj[str(epoch)] = would
        old = stored.get(str(epoch))
        if ctx.force or old is None or old.sig != would.sig:
            recompute.append(epoch)
    return node_proj, recompute, _per_epoch_reason(ctx, desc, recompute, stored, node_proj)


def _per_epoch_reason(
    ctx: _SigContext,
    desc: _AnalyzerDescriptor,
    recompute: list[int],
    stored: dict[str, SigRecord],
    node_proj: dict[str, SigRecord],
) -> str | None:
    """A short, honest reason for recomputing a per-epoch node (skip transparency)."""
    if not recompute:
        return None
    if ctx.force:
        return "forced"
    first = str(recompute[0])
    old = stored.get(first)
    if old is None:
        return "missing" if not stored else "stale: new epoch"
    changed = _changed_upstreams(ctx, desc.requires)
    if changed:
        return f"stale: upstream {changed[0]} changed"
    return sig_mod.explain_change(old, node_proj[first])


def _changed_upstreams(ctx: _SigContext, requires: tuple[str, ...]) -> list[str]:
    """Upstreams whose projected signature differs from what is stored (any epoch)."""
    out: list[str] = []
    for upstream in requires:
        proj = ctx.projected.get(upstream, {})
        stored = ctx.stored(upstream)
        if any(stored.get(key) is None or stored[key].sig != rec.sig for key, rec in proj.items()):
            out.append(upstream)
    return out


def _cross_upstream_sigs(ctx: _SigContext, requires: tuple[str, ...]) -> list[str]:
    """Every projected (if planned) else stored signature of a cross-epoch node's upstreams."""
    sigs: list[str] = []
    for upstream in requires:
        proj = ctx.projected.get(upstream)
        recs = proj if proj is not None else ctx.stored(upstream)
        sigs.extend(rec.sig for rec in recs.values())
    return sigs


def _cross_reason(
    ctx: _SigContext, desc: _AnalyzerDescriptor, stored: SigRecord | None, would: SigRecord
) -> str:
    """Reason a present cross-epoch artifact is stale (legacy/no-manifest → ``"stale"``)."""
    if stored is None:
        return "stale"
    changed = _changed_upstreams(ctx, desc.requires)
    if changed:
        return f"stale: upstream {changed[0]} changed"
    return sig_mod.explain_change(stored, would)


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


def _scoped_dir(artifacts_dir: Path, name: str, recipe_map: dict[str, str] | None) -> Path:
    """Recipe-scoped analyzer dir (REQ_138) via the storage primitive.

    An empty/absent recipe entry resolves to today's path, so the default
    (unparameterized) plan scans exactly the locations it always did.
    """
    from miscope.analysis.artifact_loader import analyzer_dir

    return Path(analyzer_dir(str(artifacts_dir), name, (recipe_map or {}).get(name, "")))


def _per_epoch_target_epochs(
    desc: _AnalyzerDescriptor,
    target_epochs: Sequence[int],
    projected_completed: dict[str, list[int]],
    artifacts_dir: Path,
    recipe_map: dict[str, str] | None = None,
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
            epochs = scan_epoch_files(_scoped_dir(artifacts_dir, upstream, recipe_map))
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
    recompute: list[int],
    blocked: tuple[str, ...],
    reason: str | None,
) -> PlanItem | None:
    """Build a per-epoch PlanItem from the signature-derived recompute set, or ``None``.

    ``recompute`` is the subset of covered epochs whose provenance signature
    changed (all of them under ``force``), from :func:`_signature_plan_per_epoch`.
    A blocked analyzer yields an item with empty ``epochs`` and a non-empty
    ``blocked_by``; ``None`` when nothing is stale.
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
    if not recompute:
        return None
    return PlanItem(
        analyzer_name=desc.name,
        epochs=tuple(recompute),
        requires=desc.requires,
        depends_on=desc.depends_on,
        reason=reason,
        requires_model_weights=desc.requires_model_weights,
        requires_activation_cache=desc.requires_activation_cache,
        required_hooks=desc.required_hooks,
    )


def _plan_cross_epoch_item(
    ctx: _SigContext,
    desc: _AnalyzerDescriptor,
    available_epochs: tuple[int, ...],
    projected_completed: dict[str, list[int]],
) -> PlanItem | None:
    """Decide whether a cross-epoch analyzer should run, by signature (REQ_145).

    Returns:
        None: artifact is fresh — signature unchanged.
        PlanItem with non-empty ``blocked_by``: a required dependency has no
            completed epochs (per-epoch or cross-epoch).
        PlanItem with ``reason="missing"``: ``cross_epoch.npz`` absent.
        PlanItem with ``reason="forced"``: ``force`` override.
        PlanItem with a ``"stale: …"`` reason: artifact present but its
            input-derived signature changed (or it predates signatures).

    ``projected_completed`` lets the planner treat earlier-phase analyzers in the
    same plan as if they had run — so a cross-epoch is not blocked just because
    its dependency hasn't started.
    """
    name = desc.name
    blocked = _cross_blocked(ctx, desc.requires, available_epochs, projected_completed)
    if blocked:
        ctx.projected[name] = {}
        return PlanItem(analyzer_name=name, requires=desc.requires, blocked_by=tuple(blocked))

    would = sig_mod.build_record(
        code_version=desc.version,
        recipe=ctx.recipe_sig(name),
        upstream_sigs=_cross_upstream_sigs(ctx, desc.requires),
    )
    ctx.projected[name] = {CROSS_EPOCH_KEY: would}

    cross_epoch_path = _scoped_dir(ctx.artifacts_dir, name, ctx.recipe_map) / "cross_epoch.npz"
    if not cross_epoch_path.exists():
        return PlanItem(
            analyzer_name=name, epochs=available_epochs, requires=desc.requires, reason="missing"
        )
    if ctx.force:
        return PlanItem(
            analyzer_name=name, epochs=available_epochs, requires=desc.requires, reason="forced"
        )
    stored = ctx.stored(name).get(CROSS_EPOCH_KEY)
    if stored is None or stored.sig != would.sig:
        return PlanItem(
            analyzer_name=name,
            epochs=available_epochs,
            requires=desc.requires,
            reason=_cross_reason(ctx, desc, stored, would),
        )
    return None


def _cross_blocked(
    ctx: _SigContext,
    requires: tuple[str, ...],
    available_epochs: tuple[int, ...],
    projected_completed: dict[str, list[int]],
) -> list[str]:
    """Required analyzers with zero completed epochs (the blocked-by set).

    Mirrors ``AnalysisPipeline.get_completed_epochs``: per-epoch files first,
    falling back to ``cross_epoch.npz`` with ``available_epochs``.
    """
    blocked: list[str] = []
    for required in requires:
        if required in projected_completed:
            completed = projected_completed[required]
        else:
            completed = get_completed_epochs(
                ctx.artifacts_dir, required, available_epochs, ctx.recipe_map
            )
        if not completed:
            blocked.append(required)
    return blocked


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
    recipe_map: dict[str, str] | None = None,
) -> list[int]:
    """Return completed epochs for an analyzer.

    Per-epoch ``epoch_*.npz`` files take precedence. If none exist but a
    ``cross_epoch.npz`` does, the supplied ``available_epochs`` are
    returned so cross-epoch-to-cross-epoch dependencies are satisfied.
    This mirrors ``AnalysisPipeline.get_completed_epochs``.
    """
    scoped = _scoped_dir(artifacts_dir, analyzer_name, recipe_map)
    if not scoped.is_dir():
        return []

    epochs = scan_epoch_files(scoped)
    if epochs:
        return epochs

    if (scoped / "cross_epoch.npz").exists():
        return sorted(available_epochs)

    return []
