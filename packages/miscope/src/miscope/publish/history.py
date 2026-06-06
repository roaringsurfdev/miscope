"""Committed manifest history — the data-free baseline the gate reads in CI (REQ_110E).

Parquet never enters the repo, but a bundle's manifest *does*: it is small JSON and
it carries the captured schema. A bundle's published versions accumulate as
``{bundle}/manifest-{ver}.json`` files, and that committed history is what lets the
schema-stability gate run without the warehouse data present — CI compares the
newest committed manifest against its predecessor.

This module reads and integrity-checks that history. ``validate_history`` catches
the always-wrong cases (a manifest whose recorded version disagrees with its
filename, a duplicated version); ``schema_delta`` reports how the latest version's
schema moved from its predecessor (informational — a *new* version is allowed to
change schema; immutability binds a tag, not a bundle).
"""

from __future__ import annotations

from pathlib import Path

from miscope.publish.manifest import BundleManifest
from miscope.publish.schema_gate import GateReport, compare

MANIFEST_GLOB = "manifest-*.json"


def _version_key(version: str) -> tuple:
    """Numeric-aware sort key so ``v1.10`` orders after ``v1.9``."""
    stem = version.lstrip("vV")
    return tuple(int(p) if p.isdigit() else p for p in stem.split("."))


def load_history(history_dir: Path) -> list[BundleManifest]:
    """Every committed manifest in a bundle's history, oldest version first."""
    hdir = Path(history_dir)
    if not hdir.is_dir():
        return []
    manifests = [BundleManifest.read(p) for p in hdir.glob(MANIFEST_GLOB)]
    return sorted(manifests, key=lambda m: _version_key(m.bundle_version))


def validate_history(history_dir: Path) -> list[str]:
    """Integrity problems in a bundle's committed manifest history (empty == clean).

    Checks the structural invariants that are always wrong regardless of schema
    evolution: a manifest's recorded ``bundle_version`` must match its filename, and
    no version may appear twice.
    """
    hdir = Path(history_dir)
    problems: list[str] = []
    seen: set[str] = set()
    for path in sorted(hdir.glob(MANIFEST_GLOB)):
        manifest = BundleManifest.read(path)
        expected = f"manifest-{manifest.bundle_version}.json"
        if path.name != expected:
            problems.append(
                f"{path.name}: bundle_version '{manifest.bundle_version}' expects filename '{expected}'"
            )
        if manifest.bundle_version in seen:
            problems.append(f"{path.name}: duplicate version '{manifest.bundle_version}'")
        seen.add(manifest.bundle_version)
    return problems


def schema_delta(history_dir: Path) -> GateReport | None:
    """The latest committed version's schema delta from its predecessor (``None`` if <2)."""
    history = load_history(history_dir)
    if len(history) < 2:
        return None
    return compare(history[-2], history[-1])
