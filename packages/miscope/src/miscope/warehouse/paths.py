"""Warehouse path composition — the storage primitive for the columnar plane (REQ_110A).

Per the storage-encapsulation invariant, only storage primitives compose paths.
This module is the single place the internal columnar warehouse layout is built;
every other warehouse module reaches files through these functions, never through
``Path("...")`` literals.

Layout (sibling to the ``.npz`` ``artifacts/`` directory, gitignored by the
``data/*/*`` rule)::

    {variant_dir}/dataviews/
        {table}/{signature}.parquet     # one Parquet per (table, coord-signature)
        _catalog/{table}.parquet        # co-emitted columnar catalog rows

``{table}`` is a semantic table name (``pca_results``, ``frequency_spectrum``, …)
or, for the generic fallback, the analyzer name. ``{signature}`` encodes the
coord set of the rows in that file (e.g. ``by__variant_epoch_site_row_id``) so a
heterogeneous analyzer yields a few signature files, never one-per-epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miscope.analysis.output_schema import Coord
    from miscope.families.variant import Variant

DATAVIEWS_DIRNAME = "dataviews"
CATALOG_DIRNAME = "_catalog"
SEMANTIC_TOKEN = "long"  # single-file stem for a conformed semantic table


def warehouse_dir(variant: Variant) -> Path:
    """The variant's columnar warehouse root: ``{variant_dir}/dataviews/``."""
    return variant.variant_dir / DATAVIEWS_DIRNAME


def table_dir(variant: Variant, table: str) -> Path:
    """Directory holding one logical table's signature Parquet files."""
    return warehouse_dir(variant) / table


def signature_token(coords: tuple[Coord, ...]) -> str:
    """Stable file-stem token for a coord signature (e.g. ``by__variant_epoch_site``).

    The ``variant`` coord is kept in the token for legibility even though it is
    expanded to ``variant_id`` + family params as actual columns.
    """
    names = "_".join(c.value for c in coords) if coords else "scalar"
    return f"by__{names}"


def table_parquet_path(variant: Variant, table: str, coords: tuple[Coord, ...]) -> Path:
    """Path to the Parquet file for ``table`` rows with this coord signature."""
    return table_dir(variant, table) / f"{signature_token(coords)}.parquet"


def semantic_parquet_path(variant: Variant, table: str) -> Path:
    """Single-file path for a conformed semantic table (row-union of feeders)."""
    return table_dir(variant, table) / f"{SEMANTIC_TOKEN}.parquet"


def catalog_dir(variant: Variant) -> Path:
    """Directory holding the co-emitted columnar catalog Parquet relation."""
    return warehouse_dir(variant) / CATALOG_DIRNAME


def catalog_parquet_path(variant: Variant, table: str) -> Path:
    """Path to the columnar catalog rows co-emitted for ``table``."""
    return catalog_dir(variant) / f"{table}.parquet"
