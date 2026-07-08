#!/usr/bin/env python
"""Materialize the columnar warehouse (long-format Parquet) for a variant or family.

The internal warehouse is deterministically regeneratable from the ``.npz``
artifacts — this script is the regeneration entry point (REQ_110A).

    # one variant
    uv run python scripts/materialize_warehouse.py modulo_addition_1layer \
        --prime 113 --seed 999 --data-seed 598

    # every trained variant in a family
    uv run python scripts/materialize_warehouse.py modulo_addition_1layer --all
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import miscope
from miscope.warehouse import materialize_variant_columnar, materialize_variant_derived

if TYPE_CHECKING:
    from miscope.families.variant import Variant


def _report(variant: Variant) -> None:
    rep = materialize_variant_columnar(variant)
    print(f"[{rep.variant_id}] {len(rep.files_written)} files across {len(rep.tables)} tables")
    for table, rows in sorted(rep.tables.items()):
        print(f"    {table:34} {rows:>10,} rows")
    if rep.skipped_analyzers:
        print(f"    skipped (no columnar artifacts): {', '.join(rep.skipped_analyzers)}")

    # Derived tables (REQ_141) query the columnar tables just written.
    drep = materialize_variant_derived(variant)
    if drep.tables or drep.views or drep.failed:
        print(f"    derived: {len(drep.tables)} materialized, {len(drep.views)} views")
        for table, rows in sorted(drep.tables.items()):
            print(f"    {table:34} {rows:>10,} rows (derived)")
        for name, err in sorted(drep.failed.items()):
            print(f"    derived FAILED {name}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", help="Model family name (e.g. modulo_addition_1layer)")
    parser.add_argument("--all", action="store_true", help="Materialize every trained variant")
    parser.add_argument("--prime", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-seed", type=int, dest="data_seed")
    args = parser.parse_args()

    family = miscope.load_family(args.family)
    if args.all:
        for variant in family.variants:
            _report(variant)
        return

    params = {
        k: v
        for k, v in (("prime", args.prime), ("seed", args.seed), ("data_seed", args.data_seed))
        if v is not None
    }
    if not params:
        parser.error("provide --prime/--seed/--data-seed, or --all")
    _report(family.get_variant(**params))


if __name__ == "__main__":
    main()
