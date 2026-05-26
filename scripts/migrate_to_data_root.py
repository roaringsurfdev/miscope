"""One-shot migration: collapse ``results/`` + ``model_families/`` into ``data/``.

This script moves variant directories from the legacy ``results/{family}/`` tree
into the unified ``data/{family}/variants/`` tree, dropping the family prefix
from variant directory names so they line up with the shortened
``variant_pattern`` introduced in REQ_123.

Behaviour:

- Idempotent: re-running after a partial migration resumes cleanly. Variants
  already moved are skipped; previously-moved files are not overwritten.
- Non-destructive: ``model_families/`` and ``results/`` are left in place.
  After verifying the new tree, remove them manually.
- Reports its plan in dry-run mode (default); pass ``--apply`` to execute.

Usage:
    uv run python scripts/migrate_to_data_root.py            # dry run
    uv run python scripts/migrate_to_data_root.py --apply    # perform moves
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from miscope.config import get_config

# Maps each known family's legacy variant prefix to its new (no-prefix) form.
# Variants that don't start with the prefix are left alone.
_LEGACY_PREFIXES: dict[str, str] = {
    "modulo_addition_1layer": "modulo_addition_1layer_",
    "modulo_addition_2layer_mlp": "modulo_addition_2layer_mlp_",
    "modulo_addition_learned_emb_mlp": "modulo_addition_learned_emb_mlp_",
}

# Files at family root that should live in data/{family}/ rather than being
# moved per-variant. (family.json + ideal_frequency_sets.json are normally
# committed via REQ_123 itself; we still copy any missing files defensively.)
_FAMILY_ROOT_FILES = ("family.json", "ideal_frequency_sets.json")


def _short_variant_name(legacy_name: str, prefix: str) -> str:
    """Strip the family prefix from a legacy variant directory name.

    Raises ValueError if the name doesn't start with the prefix.
    """
    if not legacy_name.startswith(prefix):
        raise ValueError(f"{legacy_name!r} does not start with {prefix!r}")
    return legacy_name[len(prefix) :]


def _ensure_family_config(family: str, project_root: Path, data_root: Path) -> list[str]:
    """Ensure ``data/{family}/family.json`` (and friends) exist.

    Returns a list of human-readable action descriptions performed (or that
    would be performed in dry-run mode).
    """
    actions: list[str] = []
    legacy_family_dir = project_root / "model_families" / family
    new_family_dir = data_root / family
    new_family_dir.mkdir(parents=True, exist_ok=True)

    for filename in _FAMILY_ROOT_FILES:
        src = legacy_family_dir / filename
        dst = new_family_dir / filename
        if dst.exists():
            continue
        if not src.exists():
            continue
        actions.append(f"copy {src.relative_to(project_root)} -> {dst.relative_to(project_root)}")
    return actions


def _apply_family_config(family: str, project_root: Path, data_root: Path) -> None:
    """Copy family.json / ideal_frequency_sets.json into data/{family}/ if missing."""
    legacy_family_dir = project_root / "model_families" / family
    new_family_dir = data_root / family
    new_family_dir.mkdir(parents=True, exist_ok=True)

    for filename in _FAMILY_ROOT_FILES:
        src = legacy_family_dir / filename
        dst = new_family_dir / filename
        if dst.exists() or not src.exists():
            continue
        shutil.copy2(src, dst)


def _collect_variant_renames(
    family: str,
    project_root: Path,
    data_root: Path,
) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Return (renames, unmoved_warnings) for a family.

    ``renames`` lists (source, target) pairs where the source is the legacy
    variant directory and target is the destination under the new layout.
    ``unmoved_warnings`` lists items under ``results/{family}/`` that don't
    match the expected variant pattern.
    """
    legacy_family_dir = project_root / "results" / family
    target_variants_dir = data_root / family / "variants"

    renames: list[tuple[Path, Path]] = []
    warnings: list[str] = []

    if not legacy_family_dir.exists():
        return renames, warnings

    prefix = _LEGACY_PREFIXES.get(family, f"{family}_")

    for entry in sorted(legacy_family_dir.iterdir()):
        if entry.is_file():
            if entry.name == "variant_registry.json":
                # handled separately
                continue
            warnings.append(f"unmoved file {entry.relative_to(project_root)}")
            continue
        if not entry.is_dir():
            continue
        try:
            short_name = _short_variant_name(entry.name, prefix)
        except ValueError:
            warnings.append(f"unrecognised directory {entry.relative_to(project_root)}")
            continue
        renames.append((entry, target_variants_dir / short_name))

    return renames, warnings


def _move_directory(src: Path, dst: Path) -> None:
    """Move ``src`` into ``dst``. If ``dst`` already exists, do nothing."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _move_registry(family: str, project_root: Path, data_root: Path) -> str | None:
    """Move ``results/{family}/variant_registry.json`` into ``data/{family}/``.

    Returns a human-readable description if a move is needed (or was made).
    """
    src = project_root / "results" / family / "variant_registry.json"
    dst = data_root / family / "variant_registry.json"
    if not src.exists() or dst.exists():
        return None
    return f"move {src.relative_to(project_root)} -> {dst.relative_to(project_root)}"


def _families_present(project_root: Path) -> list[str]:
    """Return families discoverable under either legacy tree."""
    found = set()
    legacy_mf = project_root / "model_families"
    legacy_results = project_root / "results"
    for root in (legacy_mf, legacy_results):
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if re.match(r"^modulo_addition_[\w]+", child.name):
                found.add(child.name)
    return sorted(found)


def run(*, project_root: Path, data_root: Path, apply: bool) -> int:
    families = _families_present(project_root)
    if not families:
        print("Nothing to migrate: no legacy family directories found.")
        return 0

    total_renames = 0
    total_warnings = 0
    deletion_plan: list[str] = []

    for family in families:
        print(f"\n=== {family} ===")

        cfg_actions = _ensure_family_config(family, project_root, data_root)
        for desc in cfg_actions:
            print(f"  [config] {desc}")
        if apply:
            _apply_family_config(family, project_root, data_root)

        registry_desc = _move_registry(family, project_root, data_root)
        if registry_desc is not None:
            print(f"  [registry] {registry_desc}")
            if apply:
                src = project_root / "results" / family / "variant_registry.json"
                dst = data_root / family / "variant_registry.json"
                _move_directory(src, dst)

        renames, warnings = _collect_variant_renames(family, project_root, data_root)
        total_warnings += len(warnings)
        for w in warnings:
            print(f"  [warn] {w}")

        if not renames:
            print("  No variant directories to move.")
        for src, dst in renames:
            already = dst.exists()
            marker = "skip (target exists)" if already else "move"
            print(f"  [{marker}] {src.name} -> data/{family}/variants/{dst.name}")
            if apply and not already:
                _move_directory(src, dst)

        total_renames += sum(1 for _, dst in renames if not dst.exists() or apply)
        if (project_root / "model_families" / family).exists():
            deletion_plan.append(f"model_families/{family}")
        if (project_root / "results" / family).exists():
            deletion_plan.append(f"results/{family}")

    print("\n=== Summary ===")
    print(f"Families processed: {len(families)}")
    print(f"Variants planned/moved: {total_renames}")
    if total_warnings:
        print(f"Warnings: {total_warnings} (see [warn] lines above)")
    if deletion_plan:
        print("\nAfter you've verified the new tree, the following legacy paths can be removed:")
        for path in deletion_plan:
            print(f"  rm -rf {path}")
        print("(This script does NOT delete anything automatically.)")

    if not apply:
        print("\nDry run only. Re-run with --apply to perform the moves.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the moves (default: dry run).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing model_families/ and results/ (default: from cfg).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Destination data root (default: from cfg.data_root).",
    )
    args = parser.parse_args()

    cfg = get_config()
    project_root = args.project_root if args.project_root is not None else cfg.project_root
    data_root = args.data_root if args.data_root is not None else cfg.data_root

    return run(project_root=project_root, data_root=data_root, apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
