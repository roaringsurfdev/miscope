"""Artifact freshness and staleness reporting (REQ_080).

Answers the question: for a given variant, which analyzers have complete
per-epoch coverage and which cross-epoch artifacts are out of date?

Taxonomy:
- *epoch-incomplete*: per-epoch artifact is missing checkpoints at the tail
- *epoch-stale*: cross-epoch artifact was built on fewer epochs than are available
- *summary-stale*: variant_summary.json is absent or older than most recent artifact

Public surface:
    check_freshness(variant, per_epoch_names, cross_epoch_names) -> FreshnessReport
    FreshnessReport.format() -> human-readable string table
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from miscope.families.variant import Variant

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
# Public API
# ---------------------------------------------------------------------------


def check_freshness(
    variant: Variant,
    per_epoch_names: Sequence[str] | None = None,
    cross_epoch_names: Sequence[str] | None = None,
) -> FreshnessReport:
    """Build a freshness report for a variant.

    Args:
        variant: The variant to inspect.
        per_epoch_names: Per-epoch analyzer names to check. If None, all
            subdirectories containing epoch_*.npz files are checked.
        cross_epoch_names: Cross-epoch analyzer names to check. If None, all
            subdirectories containing cross_epoch.npz are checked.

    Returns:
        FreshnessReport with per-epoch and cross-epoch freshness status.
    """
    artifacts_dir = Path(variant.artifacts_dir)
    available_checkpoints = sorted(variant.get_available_checkpoints())
    checkpoint_set = set(available_checkpoints)

    per_epoch_results = _check_per_epoch(
        artifacts_dir, available_checkpoints, checkpoint_set, per_epoch_names
    )
    cross_epoch_results = _check_cross_epoch(
        artifacts_dir, len(available_checkpoints), cross_epoch_names
    )
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


def _check_per_epoch(
    artifacts_dir: Path,
    available_checkpoints: list[int],
    checkpoint_set: set[int],
    names: Sequence[str] | None,
) -> list[PerEpochFreshness]:
    """Check per-epoch artifact coverage for each analyzer."""
    if not artifacts_dir.exists():
        return []

    if names is not None:
        candidate_dirs = [artifacts_dir / n for n in names]
    else:
        candidate_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]

    results = []
    for analyzer_dir in candidate_dirs:
        if not analyzer_dir.is_dir():
            continue

        artifact_epochs = _scan_epoch_files(analyzer_dir)
        if not artifact_epochs:
            # Skip cross-epoch-only directories during auto-discovery.
            # When analyzer names are given explicitly, include even if empty.
            if names is None:
                continue

        artifact_epoch_set = set(artifact_epochs)
        missing = sorted(checkpoint_set - artifact_epoch_set)

        results.append(
            PerEpochFreshness(
                analyzer_name=analyzer_dir.name,
                total_checkpoints=len(available_checkpoints),
                artifact_epoch_count=len(artifact_epochs),
                missing_epochs=missing,
            )
        )

    return results


def _check_cross_epoch(
    artifacts_dir: Path,
    n_checkpoints: int,
    names: Sequence[str] | None,
) -> list[CrossEpochFreshness]:
    """Check cross-epoch artifact coverage for each analyzer."""
    if not artifacts_dir.exists():
        return []

    if names is not None:
        candidate_dirs = [artifacts_dir / n for n in names]
    else:
        candidate_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]

    results = []
    for analyzer_dir in candidate_dirs:
        if not analyzer_dir.is_dir():
            continue

        cross_epoch_path = analyzer_dir / "cross_epoch.npz"
        if not cross_epoch_path.exists():
            if names is not None:
                results.append(
                    CrossEpochFreshness(
                        analyzer_name=analyzer_dir.name,
                        artifact_exists=False,
                        available_checkpoints=n_checkpoints,
                        covered_epoch_count=0,
                    )
                )
            continue

        covered = _read_covered_epoch_count(cross_epoch_path)
        results.append(
            CrossEpochFreshness(
                analyzer_name=analyzer_dir.name,
                artifact_exists=True,
                available_checkpoints=n_checkpoints,
                covered_epoch_count=covered,
            )
        )

    return results


def _scan_epoch_files(analyzer_dir: Path) -> list[int]:
    """Return sorted list of epoch numbers from epoch_*.npz files."""
    epochs = []
    for fname in os.listdir(analyzer_dir):
        if fname.startswith("epoch_") and fname.endswith(".npz"):
            try:
                epochs.append(int(fname[6:-4]))
            except ValueError:
                continue
    return sorted(epochs)


def _read_covered_epoch_count(cross_epoch_path: Path) -> int:
    """Read the number of epochs stored in a cross-epoch artifact.

    Returns -1 if the artifact has no 'epochs' key (treated as unknown/stale).
    """
    try:
        with np.load(cross_epoch_path, allow_pickle=False) as data:
            if "epochs" not in data:
                return -1
            return int(data["epochs"].shape[0])
    except Exception:
        return -1


def _check_summary_stale(variant: Variant) -> bool:
    """Return True if variant_summary.json is absent or older than any artifact."""
    summary_path = variant.variant_dir / "variant_summary.json"
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


# ---------------------------------------------------------------------------
# Pipeline staleness check (used by AnalysisPipeline)
# ---------------------------------------------------------------------------


def cross_epoch_is_stale(
    cross_epoch_path: str | Path,
    required_analyzer_names: list[str],
    artifacts_dir: str | Path,
    available_epochs: list[int],
) -> bool:
    """Return True if a cross-epoch artifact should be rebuilt.

    An artifact is considered stale if the number of per-epoch dependency
    epochs available exceeds the number of epochs the artifact covers.
    This ensures cross-epoch analyzers rerun after incremental per-epoch runs.

    Args:
        cross_epoch_path: Path to the existing cross_epoch.npz file.
        required_analyzer_names: Names of per-epoch analyzers this depends on.
        artifacts_dir: Root artifacts directory for the variant.
        available_epochs: All checkpoint epochs for the variant.
    """
    covered = _read_covered_epoch_count(Path(cross_epoch_path))
    if covered < 0:
        return True  # No epoch metadata → conservative rerun

    # Find how many per-epoch dependency epochs exist on disk.
    artifacts_dir = Path(artifacts_dir)
    dep_epoch_counts = []
    for name in required_analyzer_names:
        dep_dir = artifacts_dir / name
        if dep_dir.is_dir():
            dep_epoch_counts.append(len(_scan_epoch_files(dep_dir)))

    if not dep_epoch_counts:
        return False  # No dependency data at all — nothing to rerun against

    max_dep_epochs = max(dep_epoch_counts)
    return max_dep_epochs > covered
