"""Warehouse table source-signature manifest (REQ_145).

A per-variant sidecar mapping ``{table -> source_sig}`` under the warehouse root.
A table's *source signature* folds the signatures of the artifacts (or input
tables) that feed it; comparing it to the stored value decides whether a table
needs re-materializing. This is the warehouse half of the one freshness predicate
— it replaces the blanket wipe-and-rebuild with a surgical, per-table rebuild.

Path composition lives in :mod:`miscope.warehouse.paths` (storage-encapsulation
invariant 3); this module only does JSON I/O over that path.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from miscope.warehouse import paths

if TYPE_CHECKING:
    from miscope.families.variant import Variant


def read_table_signatures(variant: Variant) -> dict[str, str]:
    """Read the ``{table -> source_sig}`` manifest, or ``{}`` when absent/unreadable.

    A missing manifest (a warehouse built before REQ_145) yields ``{}`` so every
    table reads as stale — the conservative, correct default that triggers a
    one-time rebuild.
    """
    path = paths.warehouse_signatures_path(variant)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_table_signatures(variant: Variant, signatures: dict[str, str]) -> None:
    """Atomically overwrite the warehouse signature manifest (REQ_145)."""
    path = paths.warehouse_signatures_path(variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(signatures, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
