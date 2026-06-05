"""Prune artifact directories for deprecated/refactored analyzers.

Over the Analysis Atlas consolidation (REQ_111/117/118/126, ...), several
analyzers were renamed, absorbed, or retired. Their on-disk artifact
directories can linger under variants long after the analyzer stops being
registered. This script finds those orphaned directories and removes them.

Ground truth for *active* analyzers is each family's ``family.json``
(exposed as ``family.analyzers``). An artifact directory whose name is not
in its family's active analyzer list is treated as deprecated. The Atlas
(docs/analysis_atlas.md) explains *why* each one was retired; family.json is
the authoritative list this script enforces.

Artifact directories are reached only through the ``Variant.artifacts_dir``
accessor — no storage paths are composed here (storage layout is internal to
the API per PROJECT.md).

Safe by default: prints what it *would* delete. Pass ``--apply`` to remove.

Usage:
    uv run python scripts/prune_deprecated_artifacts.py            # dry run, all families
    uv run python scripts/prune_deprecated_artifacts.py --family modulo_addition_1layer
    uv run python scripts/prune_deprecated_artifacts.py --apply    # actually delete
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from miscope import list_families, load_family
from miscope.families.protocols import ModelFamily
from miscope.families.variant import Variant


def find_deprecated_artifact_dirs(variant: Variant, active: set[str]) -> list[Path]:
    """Return artifact subdirectories of ``variant`` not in the active set.

    Each immediate subdirectory of ``artifacts_dir`` is named for the analyzer
    that produced it; any name absent from ``active`` is deprecated.
    """
    artifacts_dir = variant.artifacts_dir
    if not artifacts_dir.exists():
        return []
    return sorted(
        d for d in artifacts_dir.iterdir() if d.is_dir() and d.name not in active
    )


def prune_family(family: ModelFamily, *, apply: bool) -> int:
    """Prune deprecated artifact dirs across all variants of one family.

    Returns the number of directories removed (or that would be removed).
    """
    active = set(family.analyzers)
    removed = 0

    # Intervention variants nest under their parent with their own artifacts,
    # so sweep them alongside the top-level variants.
    variants: list[Variant] = []
    for variant in family.variants:
        variants.append(variant)
        variants.extend(variant.interventions)

    for variant in variants:
        for stale_dir in find_deprecated_artifact_dirs(variant, active):
            action = "Deleting" if apply else "Would delete"
            print(f"  {action}: {variant.name}/artifacts/{stale_dir.name}")
            if apply:
                shutil.rmtree(stale_dir)
            removed += 1

    return removed


def run(family_name: str | None, *, apply: bool) -> None:
    family_names = [family_name] if family_name else list_families()

    total = 0
    for name in family_names:
        family = load_family(name)
        active = sorted(family.analyzers)
        print(f"\n{name} ({len(active)} active analyzers):")
        count = prune_family(family, apply=apply)
        if count == 0:
            print("  No deprecated artifact directories found.")
        total += count

    mode = "Removed" if apply else "Found (dry run)"
    print(f"\n{mode} {total} deprecated artifact director{'y' if total == 1 else 'ies'}.")
    if total and not apply:
        print("Re-run with --apply to delete them.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        default=None,
        help="Family name to prune. Defaults to all discovered families.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete directories. Without this flag, only reports.",
    )
    args = parser.parse_args()

    run(family_name=args.family, apply=args.apply)


if __name__ == "__main__":
    main()
