"""One-time migration: add _dseed{data_seed} suffix to variant directories.

REQ_061: data_seed is now a first-class domain parameter. All existing variant
directories must be renamed to encode the data_seed they were trained with.

Usage:
    uv run python scripts/migrate_dseed.py                  # dry run
    uv run python scripts/migrate_dseed.py --apply          # apply renames
    uv run python scripts/migrate_dseed.py --apply --data-seed 598  # explicit seed

Idempotent: directories already containing '_dseed' are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OLD_PATTERN = re.compile(r"^(modulo_addition_1layer_p\d+_seed\d+)$")


def load_data_seed_from_config(variant_dir: Path, fallback: int) -> int:
    """Read data_seed from config.json if present, otherwise return fallback."""
    config_path = variant_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return int(config.get("data_seed", fallback))
    return fallback


def collect_renames(family_dir: Path, default_data_seed: int) -> list[tuple[Path, Path]]:
    """Return (old_path, new_path) pairs for directories that need renaming."""
    renames = []
    for variant_dir in sorted(family_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        if "_dseed" in variant_dir.name:
            continue  # already migrated
        if not OLD_PATTERN.match(variant_dir.name):
            continue

        data_seed = load_data_seed_from_config(variant_dir, default_data_seed)
        new_name = f"{variant_dir.name}_dseed{data_seed}"
        renames.append((variant_dir, variant_dir.parent / new_name))

    return renames


def run(results_dir: Path, family_name: str, default_data_seed: int, apply: bool) -> None:
    family_dir = results_dir / family_name
    if not family_dir.exists():
        print(f"Family directory not found: {family_dir}")
        sys.exit(1)

    renames = collect_renames(family_dir, default_data_seed)

    if not renames:
        print("No directories require migration.")
        return

    label = "Applying" if apply else "Dry run —"
    for old_path, new_path in renames:
        print(f"  {label} rename: {old_path.name} → {new_path.name}")
        if apply:
            old_path.rename(new_path)

    if not apply:
        print(f"\n{len(renames)} director(y/ies) would be renamed. Pass --apply to execute.")
    else:
        print(f"\n{len(renames)} director(y/ies) renamed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply renames (default: dry run)")
    parser.add_argument(
        "--data-seed",
        type=int,
        default=598,
        help="Fallback data_seed value when config.json is absent (default: 598)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Path to results directory",
    )
    parser.add_argument(
        "--family",
        default="modulo_addition_1layer",
        help="Family subdirectory name",
    )
    args = parser.parse_args()

    run(
        results_dir=args.results_dir,
        family_name=args.family,
        default_data_seed=args.data_seed,
        apply=args.apply,
    )


if __name__ == "__main__":
    main()
