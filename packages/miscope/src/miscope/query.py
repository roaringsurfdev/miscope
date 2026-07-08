"""DuckDB query surface over the warehouse (REQ_110C).

``miscope.query.open(...)`` binds a DuckDB connection to a Parquet root and
registers ergonomic views, so a consumer writes ``SELECT * FROM frequency_spectrum``
rather than a file glob. Two root shapes share one query surface:

- **Warehouse (family) mode** — ``open(family="modulo_addition_1layer")``. Each
  semantic/generic table becomes a view globbing that table across *every*
  variant's ``dataviews/`` dir (``variant_id`` is already a column, so the union
  is cross-variant with no reconciliation). A unified ``catalog`` view is the
  ``UNION ALL BY NAME`` of the columnar (110-A) and tensor (110-B) descriptor
  relations over the shared ``kind`` discriminator. Local only — globbing needs a
  filesystem. This is the day-to-day cross-variant surface.
- **Bundle mode** — ``open(root=<dir-or-url>, tables=[...])``. One view per flat
  ``{root}/{table}.parquet``. Works over local paths and HTTP URLs (DuckDB issues
  range requests via ``httpfs``), so a published bundle (110-E) is queryable by
  URL with the identical ``con.sql(...)`` surface.

The module composes no analyzer logic and re-implements no Parquet reads — DuckDB
handles scanning, range requests, and joins natively (a REQ_110 constraint). Path
composition stays in :mod:`miscope.warehouse.paths` (the storage primitive); this
module only turns those addresses into views.

Example::

    import miscope.query
    con = miscope.query.open(family="modulo_addition_1layer")
    df = con.sql('''
        SELECT variant_id, epoch, magnitude
        FROM frequency_spectrum
        WHERE site = 'mlp_out' AND frequency = 25
        ORDER BY variant_id, epoch
    ''').df()
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import duckdb

from miscope.warehouse import paths

if TYPE_CHECKING:
    from miscope.config import AppConfig

CATALOG_VIEW = "catalog"
RUN_SETS_VIEW = "run_sets"


@dataclass(frozen=True)
class QueryConnection:
    """A DuckDB connection with the warehouse's tables pre-registered as views.

    Thin wrapper: :meth:`sql` and :meth:`df` delegate straight to DuckDB, and
    :attr:`con` exposes the raw connection for anything the wrapper does not.
    Cross-table joins are plain SQL — DuckDB resolves them over the registered
    views with no custom join logic here.
    """

    con: duckdb.DuckDBPyConnection
    views: tuple[str, ...]

    def sql(self, query: str) -> duckdb.DuckDBPyRelation:
        """Run ``query`` and return the DuckDB relation (``.df()`` for a DataFrame)."""
        return self.con.sql(query)

    def df(self, query: str) -> Any:
        """Run ``query`` and return a pandas DataFrame (convenience over ``sql``)."""
        return self.con.sql(query).df()

    def tables(self) -> list[str]:
        """The view names registered on this connection."""
        return list(self.views)

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self.con.close()

    def __enter__(self) -> QueryConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open(  # noqa: A001 — deliberate: mirrors `duckdb.connect`/`sqlite3.connect` naming
    family: object | str | None = None,
    *,
    root: str | None = None,
    tables: Iterable[str] | None = None,
    config: AppConfig | None = None,
) -> QueryConnection:
    """Open a query connection bound to a warehouse (by family) or a bundle (by root).

    Exactly one of ``family`` or ``root`` selects the mode:

    - ``family`` (name or :class:`ModelFamily`): warehouse mode — views glob each
      table across every variant; a unified ``catalog`` view is added. Local.
    - ``root`` (local dir or base URL): bundle mode — one view per flat
      ``{root}/{table}.parquet``. ``tables`` is required for a URL root (a remote
      dir cannot be listed) and optional for a local dir (auto-discovered).

    ``tables`` restricts/declares which tables to register. ``config`` overrides
    the default :func:`miscope.get_config` when resolving a family by name.
    """
    if (family is None) == (root is None):
        raise ValueError("open() takes exactly one of `family=` or `root=`.")
    requested = tuple(tables) if tables is not None else None
    if family is not None:
        return _open_warehouse(_resolve_family(family, config), requested)
    return _open_bundle(root, requested)  # type: ignore[arg-type]


def open_variant(
    variant: object,
    tables: Iterable[str],
    *,
    run_set: str = paths.DEFAULT_RUN_SET,
) -> QueryConnection:
    """A connection scoped to one variant's tables, filtered to one run set (REQ_141).

    Registers a view per requested table over *that variant's own* Parquet files
    (not the cross-variant glob), filtered to ``run_set``, so a derived-table query
    runs against a single variant + parameterization plane — a per-variant
    aggregation never merges across variants. Tables absent for the variant are
    silently skipped (the caller decides whether missing inputs are fatal). This is
    the materialization-time scan; the consumer-facing surface is :func:`open`.
    """
    con = duckdb.connect()
    registered: list[str] = []
    for table in tables:
        tdir = paths.table_dir(variant, table)  # type: ignore[arg-type]
        if not tdir.is_dir():
            continue
        scan = _glob_scan(paths.variant_table_glob(variant, table))  # type: ignore[arg-type]
        _create_view(con, table, f"{scan} WHERE run_set = '{_escape(run_set)}'")
        registered.append(table)
    return QueryConnection(con=con, views=tuple(registered))


def _resolve_family(family: object | str, config: AppConfig | None) -> object:
    """A family name resolves through the sanctioned loader; an object passes through."""
    if isinstance(family, str):
        from miscope import load_family

        return load_family(family, config=config)
    return family


def _open_warehouse(family: object, requested: tuple[str, ...] | None) -> QueryConnection:
    """Register a glob view per table across the family's variants + a unified catalog."""
    available = paths.list_family_tables(family)
    names = [t for t in available if requested is None or t in requested]
    con = duckdb.connect()
    for table in names:
        _create_view(con, table, _glob_scan(paths.family_table_glob(family, table)))
    views = list(names)
    catalog_sql = _catalog_scan(family)
    if catalog_sql is not None:
        _create_view(con, CATALOG_VIEW, catalog_sql)
        views.append(CATALOG_VIEW)
    if paths.family_has_run_sets(family):
        _create_view(con, RUN_SETS_VIEW, _glob_scan(paths.family_run_sets_glob(family)))
        views.append(RUN_SETS_VIEW)
    _register_derived_views(con, views)
    return QueryConnection(con=con, views=tuple(views))


def _register_derived_views(con: duckdb.DuckDBPyConnection, views: list[str]) -> None:
    """Register declared view-mode derived tables as live DuckDB views (REQ_141).

    Materialized derived tables are discovered as ordinary base tables (their
    Parquet lives in the per-variant warehouse), so only ``materialized=False``
    specs need a live ``CREATE VIEW`` here — their query computes on read. A view
    whose input tables are not all present is skipped (its inputs were never
    materialized for this family). ``views`` is extended in place. Mutates nothing
    if no view-mode derived tables are registered.
    """
    from miscope.analysis.derived_table import DerivedTableRegistry

    for spec in sorted(DerivedTableRegistry.list_specs(), key=lambda d: d.name):
        if spec.materialized or spec.name in views:
            continue
        if not all(t in views for t in spec.input_tables):
            continue
        _create_view(con, spec.name, spec.query)
        views.append(spec.name)


def _open_bundle(root: str, requested: tuple[str, ...] | None) -> QueryConnection:
    """Register one flat-file view per table under ``root`` (local dir or base URL)."""
    names = list(requested) if requested is not None else _list_local_bundle_tables(root)
    con = duckdb.connect()
    if _is_url(root):
        _enable_httpfs(con)
    for table in names:
        _create_view(con, table, _flat_scan(paths.flat_table_uri(root, table)))
    return QueryConnection(con=con, views=tuple(names))


def _catalog_scan(family: object) -> str | None:
    """The ``UNION ALL BY NAME`` of whichever catalog planes the family co-emitted.

    Returns ``None`` when neither plane exists, so no empty ``catalog`` view is
    created (a glob with no matches is a DuckDB error, not an empty relation).
    """
    scans = []
    if paths.family_has_catalog_rows(family, tensor=False):
        scans.append(_glob_scan(paths.family_catalog_glob(family)))
    if paths.family_has_catalog_rows(family, tensor=True):
        scans.append(_glob_scan(paths.family_tensor_catalog_glob(family)))
    if not scans:
        return None
    return "\nUNION ALL BY NAME\n".join(scans)


def _glob_scan(glob: str) -> str:
    """A union-by-name scan over a multi-file glob (heterogeneous signatures reconcile)."""
    return f"SELECT * FROM read_parquet('{_escape(glob)}', union_by_name=true)"


def _flat_scan(uri: str) -> str:
    """A scan over a single flat Parquet file (one table per bundle file)."""
    return f"SELECT * FROM read_parquet('{_escape(uri)}')"


def _create_view(con: duckdb.DuckDBPyConnection, name: str, scan_sql: str) -> None:
    """Register ``scan_sql`` as a named view (quoted so any identifier is safe)."""
    con.execute(f'CREATE VIEW "{_escape(name)}" AS {scan_sql}')


def _escape(text: str) -> str:
    """Double single-quotes for embedding a string literal/identifier in SQL."""
    return text.replace("'", "''").replace('"', '""')


def _list_local_bundle_tables(root: str) -> list[str]:
    """Table names from a local flat bundle dir (the ``.parquet`` stems)."""
    from pathlib import Path

    rdir = Path(root)
    if not rdir.is_dir():
        raise FileNotFoundError(f"Bundle root '{root}' is not a local directory.")
    return sorted(p.stem for p in rdir.glob("*.parquet"))


def _is_url(root: str) -> bool:
    return root.startswith(("http://", "https://", "s3://"))


def _enable_httpfs(con: duckdb.DuckDBPyConnection) -> None:
    """Load ``httpfs`` so a URL root is read via range requests (lazy; URL roots only)."""
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
