"""Shared configuration for the byte-regression harness (REQ_134).

Three scripts cooperate to maintain the v1.0.0 byte-regression safety net:

  - ``generate_regression_checksums.py`` — *generator*: snapshot reference
    checksums of the on-disk artifacts.
  - ``run_analysis_regression.py`` — *regen*: recompute artifacts on ``develop``
    when reference checksums need refreshing.
  - ``run_regression_check.py`` — *checker*: recompute into an isolated tree and
    compare every ``.npz`` against the reference checksums.

For the comparison to be meaningful, three things must agree across all three
scripts: the **analyzer set** under regression, the analyzers **excluded** from
it, and the **checksums file path**. They live here so no script re-derives them
(the drift REQ_134 fixes was three divergent copies of each).

Importable as ``regression_common`` because ``scripts/`` is on ``sys.path[0]``
when a script is run directly; tests reach it via the ``pythonpath`` pytest
setting.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from miscope.analysis import plan_analysis
from miscope.analysis.registry import AnalyzerRegistry

if TYPE_CHECKING:
    from miscope.analysis.planner import Plan
    from miscope.families.model_family import ModelFamily
    from miscope.families.variant import Variant

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAMILY = "modulo_addition_1layer"

# Single reference-checksums location, shared by generator and checker (CoS 3).
# The generator writes here; the checker reads here. Previously the two defaulted
# to different paths and a default checker invocation aborted immediately.
REFERENCE_CHECKSUMS_PATH = PROJECT_ROOT / "tests" / "regression" / "reference_checksums.json"

# Analyzers retired in REQ_102 and earlier. None are still registered, so they
# never appear in the canonical selection. They are listed only so the on-disk
# artifact scan ignores stale artifacts left in old results trees.
DEPRECATED_ANALYZERS = frozenset(
    {
        "effective_dimensionality",
        "centroid_dmd",
        "coarseness",
        "attention_freq",
        "attention_fourier",
        "neuron_fourier",
        "dominant_frequencies",
        "neuron_freq_norm",
    }
)

# The single exclude set, referenced by selection, checksum generation, and the
# EXTRA-artifact disk scan (CoS 2). Three categories of exclusion:
#   landscape_flatness  — declared + registered, but stochastic by design
#   fourier_nucleation  — declared + registered, but initialization-only
#   gradient_site       — registered but NOT family-declared, so never selected;
#                         listed so the disk scan ignores any stray artifacts
#   DEPRECATED_ANALYZERS — retired; ignored by the disk scan only
EXCLUDE_FROM_REGRESSION = DEPRECATED_ANALYZERS | frozenset(
    {
        "landscape_flatness",
        "fourier_nucleation",
        "gradient_site",
    }
)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def select_specs(family: ModelFamily) -> list:
    """The canonical regression analyzer set: family-declared minus excludes.

    This is the single source of truth for *which* analyzers are under
    regression. Because it derives from ``list_for_family`` it cannot fall
    behind the dependency graph — every declared ``ArtifactInput`` upstream is
    present (CoS 1, 4).
    """
    return [
        spec
        for spec in AnalyzerRegistry.list_for_family(family)
        if spec.name not in EXCLUDE_FROM_REGRESSION
    ]


def plan_regression(variant: Variant, force: bool) -> Plan:
    """Build a Plan over the canonical regression analyzer set for a variant."""
    return plan_analysis(variant, select_specs(variant.family), force=force)


def seed_artifact_overlay(source_artifacts_dir: Path, target_artifacts_dir: Path) -> int:
    """Mirror existing artifacts into the regression tree as symlinks.

    Recreates the directory structure under ``target_artifacts_dir`` with real
    directories and symlinks each ``.npz`` back to its counterpart under
    ``source_artifacts_dir``. This puts every existing artifact on disk *before*
    recompute, so cross-epoch analyzers whose upstream is another cross-epoch
    analyzer (``activation_dmd``/``intragroup_manifold``/``transient_frequency``)
    resolve via on-disk state rather than single-pass ordering (CoS 5).

    The recompute is non-destructive to the originals: the pipeline writes via
    ``os.replace(temp, target)``, which replaces a symlink with a fresh real file
    and leaves the symlink's target untouched. Excluded analyzers are not
    mirrored — nothing under regression depends on them.

    Returns the number of symlinks created.
    """
    if not source_artifacts_dir.exists():
        return 0
    count = 0
    for npz_path in source_artifacts_dir.rglob("*.npz"):
        rel = npz_path.relative_to(source_artifacts_dir)
        top_dir = rel.parts[0] if rel.parts else ""
        if top_dir in EXCLUDE_FROM_REGRESSION:
            continue
        link_path = target_artifacts_dir / rel
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if not link_path.exists():
            os.symlink(npz_path.resolve(), link_path)
            count += 1
    return count
