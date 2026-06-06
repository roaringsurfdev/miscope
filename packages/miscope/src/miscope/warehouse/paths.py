"""Warehouse path composition — the storage primitive for the columnar plane (REQ_110A).

Per the storage-encapsulation invariant, only storage primitives compose paths.
This module is the single place the internal columnar warehouse layout is built;
every other warehouse module reaches files through these functions, never through
``Path("...")`` literals.

Layout (sibling to the ``.npz`` ``artifacts/`` directory, gitignored by the
``data/*/*`` rule)::

    {variant_dir}/dataviews/
        {table}/{signature}.parquet     # one Parquet per (table, coord-signature)
        _catalog/{table}.parquet        # co-emitted columnar catalog rows (110-A)
        _tensor_catalog/{analyzer}.parquet  # tensor descriptor rows (110-B)

``{table}`` is a semantic table name (``pca_results``, ``frequency_spectrum``, …)
or, for the generic fallback, the analyzer name. ``{signature}`` encodes the
coord set of the rows in that file (e.g. ``by__variant_epoch_site_row_id``) so a
heterogeneous analyzer yields a few signature files, never one-per-epoch.

The tensor catalog (110-B) is a sibling of the columnar catalog under the same
warehouse root, kept in its own ``_tensor_catalog/`` directory so the two halves
materialize independently — a columnar re-materialize preserves the tensor rows
and vice versa. Together the two ``_catalog`` directories are the shared catalog
relation (110-C unions them ``BY NAME`` over the ``kind`` discriminator).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miscope.analysis.output_schema import Coord
    from miscope.families.variant import Variant

DATAVIEWS_DIRNAME = "dataviews"
CATALOG_DIRNAME = "_catalog"
TENSOR_CATALOG_DIRNAME = "_tensor_catalog"
RUN_SETS_DIRNAME = "_run_sets"
SEMANTIC_TOKEN = "long"  # single-file stem for a conformed semantic table
DEFAULT_RUN_SET = "__default__"  # the empty/all-defaults parameterization (REQ_138)


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


def tensor_catalog_dir(variant: Variant) -> Path:
    """Directory holding the co-emitted tensor descriptor Parquet relation (110-B)."""
    return warehouse_dir(variant) / TENSOR_CATALOG_DIRNAME


def tensor_catalog_parquet_path(variant: Variant, analyzer: str) -> Path:
    """Path to the tensor descriptor rows co-emitted for one analyzer's blobs."""
    return tensor_catalog_dir(variant) / f"{analyzer}.parquet"


def run_sets_dir(variant: Variant) -> Path:
    """Directory holding the run-set registry relation for a variant (REQ_138)."""
    return warehouse_dir(variant) / RUN_SETS_DIRNAME


def run_sets_parquet_path(variant: Variant) -> Path:
    """Path to the variant's run-set registry Parquet (one file, append/merge)."""
    return run_sets_dir(variant) / "run_sets.parquet"


# ---------------------------------------------------------------------------
# Family-level globs — the cross-variant query surface (110-C).
#
# The query layer reads *across* every variant of a family, so the unit of
# addressing rises from a single variant to the family's `variants/` tree. The
# family owns the `variants/` segment (`family.variants_dir`); this module owns
# the `dataviews/` warehouse layout under it. Composing the two here keeps the
# DuckDB views (110-C) free of any path literal.
# ---------------------------------------------------------------------------


def _variants_glob(family: object) -> Path:
    """``{family.variants_dir}/*`` — one wildcard segment per variant dir."""
    return family.variants_dir / "*"  # type: ignore[attr-defined]


def family_table_glob(family: object, table: str) -> str:
    """Glob over every variant's Parquet files for ``table`` (cross-variant view)."""
    return str(_variants_glob(family) / DATAVIEWS_DIRNAME / table / "*.parquet")


def family_catalog_glob(family: object) -> str:
    """Glob over every variant's co-emitted columnar catalog Parquet (110-A rows)."""
    return str(_variants_glob(family) / DATAVIEWS_DIRNAME / CATALOG_DIRNAME / "*.parquet")


def family_tensor_catalog_glob(family: object) -> str:
    """Glob over every variant's co-emitted tensor descriptor Parquet (110-B rows)."""
    return str(_variants_glob(family) / DATAVIEWS_DIRNAME / TENSOR_CATALOG_DIRNAME / "*.parquet")


def family_run_sets_glob(family: object) -> str:
    """Glob over every variant's run-set registry Parquet (REQ_138 cross-variant view)."""
    return str(_variants_glob(family) / DATAVIEWS_DIRNAME / RUN_SETS_DIRNAME / "*.parquet")


def family_has_run_sets(family: object) -> bool:
    """Whether any variant has recorded a run set (so the query layer skips an empty glob)."""
    vdir: Path = family.variants_dir  # type: ignore[attr-defined]
    if not vdir.exists():
        return False
    return any(vdir.glob(f"*/{DATAVIEWS_DIRNAME}/{RUN_SETS_DIRNAME}/*.parquet"))


def list_family_tables(family: object) -> list[str]:
    """Table names materialized under any variant's warehouse (excludes catalog dirs).

    Filesystem scan across the family's variants — the table set is the union of
    what each variant has materialized, so a table present in only some variants
    still surfaces (its glob simply matches fewer files).
    """
    vdir: Path = family.variants_dir  # type: ignore[attr-defined]
    if not vdir.exists():
        return []
    seen: set[str] = set()
    for variant_dir in vdir.iterdir():
        wdir = variant_dir / DATAVIEWS_DIRNAME
        if not wdir.is_dir():
            continue
        for child in wdir.iterdir():
            if child.is_dir() and child.name not in (
                CATALOG_DIRNAME,
                TENSOR_CATALOG_DIRNAME,
                RUN_SETS_DIRNAME,
            ):
                seen.add(child.name)
    return sorted(seen)


def family_has_catalog_rows(family: object, *, tensor: bool) -> bool:
    """Whether any variant co-emitted catalog rows of the requested plane.

    Lets the query layer build the unified ``catalog`` view from only the planes
    that exist (a family may have one without the other), instead of globbing an
    empty pattern — which DuckDB treats as an error.
    """
    vdir: Path = family.variants_dir  # type: ignore[attr-defined]
    if not vdir.exists():
        return False
    sub = TENSOR_CATALOG_DIRNAME if tensor else CATALOG_DIRNAME
    return any(vdir.glob(f"*/{DATAVIEWS_DIRNAME}/{sub}/*.parquet"))


def flat_table_uri(root: str, table: str) -> str:
    """``{root}/{table}.parquet`` for a flat bundle (local dir or HTTP base URL).

    The published-bundle layout (110-E) is flat — one Parquet per table at a
    stable URL — so a string join is the whole address. ``root`` is a deployment
    value supplied by the caller (a Release asset base), not internal layout.
    """
    return f"{root.rstrip('/')}/{table}.parquet"
