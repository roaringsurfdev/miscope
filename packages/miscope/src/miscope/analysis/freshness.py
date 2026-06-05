"""Artifact freshness and staleness reporting (REQ_080).

Answers the question: for a given variant, which analyzers have complete
per-epoch coverage and which cross-epoch artifacts are out of date?

After REQ_119, this module is a thin wrapper over
:func:`miscope.analysis.planner.plan_analysis`: the Planner decides which
analyzers are stale or missing; this module presents the decision in the
``FreshnessReport`` shape that CLI and dashboard callers expect.

Taxonomy:
- *epoch-incomplete*: per-epoch artifact is missing checkpoints
- *epoch-stale*: cross-epoch artifact was built on fewer epochs than are available
- *summary-stale*: variant_summary.json is absent or older than most recent artifact

Public surface:
    check_freshness(variant, per_epoch_names, cross_epoch_names) -> FreshnessReport
    FreshnessReport.format() -> human-readable string table
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from miscope.analysis.planner import (
    plan_analysis,
    read_covered_epoch_count,
    scan_epoch_files,
)

# Re-export the disk-state primitives at their historical names so existing
# tests and callers continue to import them from ``freshness``.
_scan_epoch_files = scan_epoch_files
_read_covered_epoch_count = read_covered_epoch_count

from miscope.families.variant import Variant  # noqa: E402  (after re-export)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PerEpochFreshness:
    """Freshness status for a single per-epoch analyzer."""

    analyzer_name: str
    total_checkpoints: int
    artifact_epoch_count: int
    missing_epochs: list[int]

    @property
    def is_fresh(self) -> bool:
        return len(self.missing_epochs) == 0

    @property
    def status_label(self) -> str:
        if self.is_fresh:
            return "fresh"
        if self.artifact_epoch_count == 0:
            return "absent"
        return f"incomplete ({len(self.missing_epochs)} missing)"


@dataclass
class CrossEpochFreshness:
    """Freshness status for a single cross-epoch analyzer."""

    analyzer_name: str
    artifact_exists: bool
    available_checkpoints: int
    covered_epoch_count: int  # epochs stored in artifact; -1 if unknown (no epochs key)

    @property
    def is_fresh(self) -> bool:
        if not self.artifact_exists:
            return False
        if self.covered_epoch_count < 0:
            return False  # conservative: treat missing metadata as stale
        return self.covered_epoch_count >= self.available_checkpoints

    @property
    def status_label(self) -> str:
        if not self.artifact_exists:
            return "absent"
        if self.covered_epoch_count < 0:
            return "stale (no epoch metadata)"
        if self.is_fresh:
            return "fresh"
        gap = self.available_checkpoints - self.covered_epoch_count
        return f"stale ({gap} new epoch(s))"


@dataclass
class ReferenceFreshness:
    """Re-resolution status of an inventory-derived reference default (REQ_138).

    A floating ``max/min epoch`` default re-resolves against the *current* checkpoint
    inventory; when that resolved value moves past what the default-plane artifact
    recorded, the floating-default artifact is stale. An explicitly *pinned* earlier
    epoch lives under its own recipe and is never reported here — its value still
    exists, so it is unaffected by training extension.
    """

    analyzer_name: str
    parameter: str
    stored_value: int | None
    resolved_value: int | None

    @property
    def is_stale(self) -> bool:
        return (
            self.stored_value is not None
            and self.resolved_value is not None
            and self.stored_value != self.resolved_value
        )

    @property
    def status_label(self) -> str:
        if self.stored_value is None:
            return "no default artifact"
        if self.is_stale:
            return f"stale (recorded {self.stored_value} -> re-resolves {self.resolved_value})"
        return "fresh"


def check_reference_freshness(
    variant: Variant, specs: Sequence[Any] | None = None
) -> list[ReferenceFreshness]:
    """Re-resolve each inventory-derived reference default and compare to the artifact.

    Covers analyzers that declare a :class:`~miscope.analysis.parameters.Reducer`
    reference default *and* record the resolved value as an output field of the same
    name (the default-plane artifact's provenance). Operates only on the default
    plane — pinned recipes are unaffected by construction.
    """
    from miscope.analysis.artifact_loader import ArtifactLoader
    from miscope.analysis.parameters import Reducer, ReferenceBinding
    from miscope.analysis.recipe import RecipeResolver

    if specs is None:
        from miscope.analysis.registry import AnalyzerRegistry

        specs = AnalyzerRegistry.list_specs()

    loader = ArtifactLoader(str(variant.artifacts_dir))
    resolver = RecipeResolver(loader)
    out: list[ReferenceFreshness] = []
    for spec in specs:
        for param in getattr(spec, "parameters", ()):
            default = param.default
            if not (isinstance(default, ReferenceBinding) and isinstance(default.selector, Reducer)):
                continue
            stored = _stored_reference_value(loader, spec.name, param.name)
            resolved = None
            if stored is not None:
                try:
                    resolved = int(resolver.resolve(default))
                except Exception:
                    resolved = None
            out.append(
                ReferenceFreshness(
                    analyzer_name=spec.name,
                    parameter=param.name,
                    stored_value=stored,
                    resolved_value=resolved,
                )
            )
    return out


def _stored_reference_value(loader: Any, analyzer: str, field_name: str) -> int | None:
    """Read the default-plane artifact's recorded value for a reference parameter."""
    try:
        data = loader.load_cross_epoch(analyzer, fields=[field_name])
    except (FileNotFoundError, ValueError):
        return None
    try:
        return int(data[field_name])
    except (KeyError, TypeError, ValueError):
        return None


@dataclass
class FreshnessReport:
    """Full freshness snapshot for a variant."""

    variant_name: str
    checked_at: str  # ISO-8601 UTC timestamp
    total_checkpoints: int
    per_epoch: list[PerEpochFreshness] = field(default_factory=list)
    cross_epoch: list[CrossEpochFreshness] = field(default_factory=list)
    summary_stale: bool = False

    @property
    def any_stale(self) -> bool:
        return (
            any(not fe.is_fresh for fe in self.per_epoch)
            or any(not ce.is_fresh for ce in self.cross_epoch)
            or self.summary_stale
        )

    def format(self) -> str:
        """Return a human-readable freshness table."""
        lines = [
            f"Freshness report: {self.variant_name}",
            f"Checked at:       {self.checked_at}",
            f"Checkpoints:      {self.total_checkpoints}",
            "",
            "Per-epoch analyzers:",
        ]
        for fe in sorted(self.per_epoch, key=lambda x: x.analyzer_name):
            tick = "✓" if fe.is_fresh else "✗"
            lines.append(f"  {tick} {fe.analyzer_name:<40} {fe.status_label}")
        lines.append("")
        lines.append("Cross-epoch analyzers:")
        for ce in sorted(self.cross_epoch, key=lambda x: x.analyzer_name):
            tick = "✓" if ce.is_fresh else "✗"
            lines.append(f"  {tick} {ce.analyzer_name:<40} {ce.status_label}")
        lines.append("")
        summary_tick = "✗" if self.summary_stale else "✓"
        lines.append(f"  {summary_tick} variant_summary.json")
        if not self.any_stale:
            lines.append("")
            lines.append("All artifacts are fresh.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sentinel analyzer objects for plan_analysis
# ---------------------------------------------------------------------------


class _PerEpochSentinel:
    """Minimal Analyzer-shaped object used to feed plan_analysis from a name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def analyze(self, ctx: Any) -> dict[str, Any]:
        raise NotImplementedError  # never executed; planner is no-side-effect


class _CrossEpochSentinel:
    """Minimal CrossEpochAnalyzer-shaped object for plan_analysis."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.requires: tuple[str, ...] = ()

    def analyze_across_epochs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_freshness(
    variant: Variant,
    per_epoch_names: Sequence[str] | None = None,
    cross_epoch_names: Sequence[str] | None = None,
    analyzers: Sequence[Any] | None = None,
) -> FreshnessReport:
    """Build a freshness report for a variant.

    Thin wrapper over :func:`plan_analysis` (REQ_119): the Planner decides
    which analyzers are stale or missing; this function translates the
    Plan into the ``FreshnessReport`` shape callers expect.

    Args:
        variant: The variant to inspect.
        per_epoch_names: Per-epoch analyzer names to check. If None and
            ``analyzers`` is also None, all subdirectories containing
            ``epoch_*.npz`` files are checked.
        cross_epoch_names: Cross-epoch analyzer names to check. If None
            and ``analyzers`` is also None, all subdirectories containing
            ``cross_epoch.npz`` are checked.
        analyzers: Optional list of registered analyzer instances
            (mix of primary, secondary, cross-epoch). When provided,
            their names are unioned with on-disk discovery so that
            registered-but-never-run analyzers appear as "absent" *and*
            unregistered leftover artifacts remain visible. Ignored when
            explicit name lists are also supplied.

    Returns:
        FreshnessReport with per-epoch and cross-epoch freshness status.
    """
    artifacts_dir = Path(variant.artifacts_dir)
    available_checkpoints = sorted(variant.get_available_checkpoints())

    if analyzers is not None and per_epoch_names is None and cross_epoch_names is None:
        per_epoch_names, cross_epoch_names = _names_from_analyzers_with_disk_union(
            analyzers, artifacts_dir
        )

    per_epoch_resolved = _resolve_per_epoch_names(artifacts_dir, per_epoch_names)
    cross_epoch_resolved = _resolve_cross_epoch_names(artifacts_dir, cross_epoch_names)

    sentinels: list[Any] = []
    sentinels.extend(_PerEpochSentinel(name) for name in per_epoch_resolved)
    sentinels.extend(_CrossEpochSentinel(name) for name in cross_epoch_resolved)
    plan = plan_analysis(variant, sentinels, force=False)
    # REQ_133: every per-epoch artifact — model-driven or purely artifact-derived
    # (the former "secondary") — now lives in the single ``plan.per_epoch`` list.
    plan_per_epoch: dict[str, Any] = {item.analyzer_name: item for item in plan.per_epoch}
    plan_cross_epoch = {item.analyzer_name: item for item in plan.cross_epoch}

    per_epoch_results = [
        _build_per_epoch_freshness(
            artifacts_dir, name, available_checkpoints, plan_per_epoch.get(name)
        )
        for name in per_epoch_resolved
    ]
    cross_epoch_results = [
        _build_cross_epoch_freshness(
            artifacts_dir, name, len(available_checkpoints), plan_cross_epoch.get(name)
        )
        for name in cross_epoch_resolved
    ]
    summary_stale = _check_summary_stale(variant)

    return FreshnessReport(
        variant_name=variant.name,
        checked_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_checkpoints=len(available_checkpoints),
        per_epoch=per_epoch_results,
        cross_epoch=cross_epoch_results,
        summary_stale=summary_stale,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _names_from_analyzers_with_disk_union(
    analyzers: Sequence[Any],
    artifacts_dir: Path,
) -> tuple[list[str], list[str]]:
    """Split analyzer names by output scope and union with on-disk discovery.

    Primary and secondary analyzers both produce per-epoch artifacts and
    share the per-epoch name list. Cross-epoch analyzers go in their own
    list. Every analyzer satisfies the same ``Analyzer`` protocol —
    classification is by the registered Spec's derived ``category``.

    Disk-discovered names are unioned in so leftover artifacts from
    removed/renamed analyzers remain visible alongside registered-but-
    never-run analyzers. To prevent the same analyzer appearing in both
    lists, cross-epoch-classified registered names are excluded from the
    per-epoch list even if a per-epoch directory accidentally exists on
    disk (and vice versa).
    """
    from miscope.analysis.inputs import derive_category
    from miscope.analysis.registry import AnalyzerRegistry

    registered_per_epoch: set[str] = set()
    registered_cross_epoch: set[str] = set()
    for analyzer in analyzers:
        if AnalyzerRegistry.has_spec(analyzer.name):
            spec = AnalyzerRegistry.get_spec(analyzer.name)
            category = derive_category(spec.inputs, spec.output_scope)
            if category == "cross_epoch":
                registered_cross_epoch.add(analyzer.name)
            else:
                # primary or secondary — both produce per-epoch artifacts
                registered_per_epoch.add(analyzer.name)
        else:
            # No Spec: fall back to legacy attribute inspection.
            if hasattr(analyzer, "analyze_across_epochs") and hasattr(analyzer, "requires"):
                registered_cross_epoch.add(analyzer.name)
            else:
                registered_per_epoch.add(analyzer.name)

    discovered_per_epoch = set(_resolve_per_epoch_names(artifacts_dir, None))
    discovered_cross_epoch = set(_resolve_cross_epoch_names(artifacts_dir, None))

    # Keep a registered classification authoritative: don't double-list a
    # cross-epoch analyzer in the per-epoch bucket just because disk
    # discovery surfaced its directory.
    discovered_per_epoch -= registered_cross_epoch
    discovered_cross_epoch -= registered_per_epoch

    return (
        sorted(registered_per_epoch | discovered_per_epoch),
        sorted(registered_cross_epoch | discovered_cross_epoch),
    )


def _resolve_per_epoch_names(artifacts_dir: Path, names: Sequence[str] | None) -> list[str]:
    """Return per-epoch analyzer directory names to inspect.

    If names are explicit, return them as-is (even if empty on disk). If
    None, auto-discover by scanning artifacts_dir for directories that
    contain at least one ``epoch_*.npz`` file. Cross-epoch-only dirs are
    skipped during auto-discovery.
    """
    if names is not None:
        return list(names)

    if not artifacts_dir.exists():
        return []

    discovered = []
    for entry in artifacts_dir.iterdir():
        if not entry.is_dir():
            continue
        if scan_epoch_files(entry):
            discovered.append(entry.name)
    return discovered


def _resolve_cross_epoch_names(artifacts_dir: Path, names: Sequence[str] | None) -> list[str]:
    """Return cross-epoch analyzer directory names to inspect."""
    if names is not None:
        return list(names)

    if not artifacts_dir.exists():
        return []

    discovered = []
    for entry in artifacts_dir.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "cross_epoch.npz").exists():
            discovered.append(entry.name)
    return discovered


def _build_per_epoch_freshness(
    artifacts_dir: Path,
    name: str,
    available_checkpoints: list[int],
    plan_item: Any,
) -> PerEpochFreshness:
    """Compose a PerEpochFreshness entry from disk state + plan presence."""
    artifact_epochs = scan_epoch_files(artifacts_dir / name)
    missing = list(plan_item.epochs) if plan_item is not None else []
    return PerEpochFreshness(
        analyzer_name=name,
        total_checkpoints=len(available_checkpoints),
        artifact_epoch_count=len(artifact_epochs),
        missing_epochs=missing,
    )


def _build_cross_epoch_freshness(
    artifacts_dir: Path,
    name: str,
    n_checkpoints: int,
    plan_item: Any,  # noqa: ARG001 — kept for future Plan-derived fields
) -> CrossEpochFreshness:
    """Compose a CrossEpochFreshness entry from disk state."""
    cross_epoch_path = artifacts_dir / name / "cross_epoch.npz"
    if not cross_epoch_path.exists():
        return CrossEpochFreshness(
            analyzer_name=name,
            artifact_exists=False,
            available_checkpoints=n_checkpoints,
            covered_epoch_count=0,
        )
    covered = read_covered_epoch_count(cross_epoch_path)
    return CrossEpochFreshness(
        analyzer_name=name,
        artifact_exists=True,
        available_checkpoints=n_checkpoints,
        covered_epoch_count=covered,
    )


def _check_summary_stale(variant: Variant) -> bool:
    """Return True if variant_summary.json is absent or older than any artifact."""
    summary_path = variant.summary_path
    if not summary_path.exists():
        return True

    summary_mtime = summary_path.stat().st_mtime
    artifacts_dir = Path(variant.artifacts_dir)
    if not artifacts_dir.exists():
        return False

    for analyzer_dir in artifacts_dir.iterdir():
        if not analyzer_dir.is_dir():
            continue
        for artifact_file in analyzer_dir.iterdir():
            if artifact_file.suffix == ".npz":
                if artifact_file.stat().st_mtime > summary_mtime:
                    return True
    return False
