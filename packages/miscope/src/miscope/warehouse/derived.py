"""Derived-table executor (REQ_141): materialize registered queries over the warehouse.

A :class:`~miscope.analysis.derived_table.DerivedTableSpec` declares a query over
already-materialized warehouse tables; this module runs it and lands the result in
the same per-variant columnar layout a semantic table uses
(``{variant}/dataviews/{name}/long.parquet`` + co-emitted catalog rows), so a
**materialized** derived table is discovered by :func:`miscope.query.open` for free
— indistinguishable to a consumer from a semantic table (CoS #2). A **view** derived
table (``materialized=False``) persists nothing; it is registered live by
``miscope.query.open`` and computes on read.

The query runs scoped to one variant + one run-set plane (:func:`miscope.query.open_variant`),
so a per-variant aggregation never merges across variants. Per-table isolation
mirrors the columnar materializer (REQ_140): a failing derived table is recorded
and skipped, never fatal to the pass. The warehouse stays decoupled from the
analysis engine — derived tables materialize from already-written warehouse tables,
just as the columnar plane materializes from artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING

import miscope.registry as reg
from miscope.analysis.derived_table import DerivedTableRegistry, DerivedTableSpec
from miscope.analysis.output_schema import Coord
from miscope.warehouse import catalog as catalog_mod
from miscope.warehouse import paths, schema

if TYPE_CHECKING:
    import pandas as pd

    from miscope.families.variant import Variant

logger = logging.getLogger(__name__)

# Canonical order for the derived table's coord signature (mirrors schema._COORD_ORDER,
# with VARIANT leading — it expands to variant_id + the family's param columns).
_COORD_SIGNATURE_ORDER = (
    Coord.VARIANT,
    Coord.EPOCH,
    Coord.SITE,
    Coord.HEAD,
    Coord.GROUP,
    Coord.FREQUENCY,
    Coord.NEURON,
    Coord.ROW_ID,
)


@dataclass
class DerivedMaterializeReport:
    """Summary of one variant's derived-table materialization."""

    variant_id: str
    tables: dict[str, int] = dc_field(default_factory=dict)  # materialized table -> rows
    views: list[str] = dc_field(default_factory=list)  # declared view-mode (not persisted)
    skipped: list[str] = dc_field(default_factory=list)  # inputs absent for this variant
    failed: dict[str, str] = dc_field(default_factory=dict)  # name -> error summary


def materialize_variant_derived(
    variant: Variant, run_set: str = paths.DEFAULT_RUN_SET
) -> DerivedMaterializeReport:
    """Materialize every registered derived table whose inputs exist for ``variant``.

    Derived tables are universal instruments (constraint 1) — not family-owned — so
    the scope is every registered spec, gated by whether its input tables are
    present for this variant. Ordered so a derived table that reads another derived
    table materializes after it.
    """
    report = DerivedMaterializeReport(variant_id=variant.name)
    specs = _ordered_specs(DerivedTableRegistry.list_specs())
    if not specs:
        return report
    variant_cols = _variant_columns(variant, run_set)
    for spec in specs:
        try:
            if not _inputs_present(variant, spec):
                report.skipped.append(spec.name)
                continue
            if not spec.materialized:
                report.views.append(spec.name)  # exposed live by miscope.query.open
                continue
            frame = _run_query(variant, spec, run_set)
            _write_derived(variant, spec, frame, variant_cols, report)
        except Exception as exc:  # noqa: BLE001 — quarantine one bad derived table (REQ_140)
            report.failed[spec.name] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "warehouse derived: table %r failed on variant %s, skipping (%s: %s)",
                spec.name,
                variant.name,
                type(exc).__name__,
                exc,
            )
    return report


def _ordered_specs(specs: list[DerivedTableSpec]) -> list[DerivedTableSpec]:
    """Specs ordered so a derived table reading another's output comes after it.

    A light topological pass over derived→derived edges (a derived table whose
    ``input_tables`` names another derived table). Non-derived inputs (semantic
    tables) impose no ordering. Ties broken by name for determinism.
    """
    by_name = {s.name: s for s in specs}
    ordered: list[DerivedTableSpec] = []
    placed: set[str] = set()

    def _place(spec: DerivedTableSpec, seen: frozenset[str]) -> None:
        if spec.name in placed or spec.name in seen:
            return
        for dep in sorted(spec.input_tables):
            if dep in by_name:
                _place(by_name[dep], seen | {spec.name})
        if spec.name not in placed:
            ordered.append(spec)
            placed.add(spec.name)

    for spec in sorted(specs, key=lambda s: s.name):
        _place(spec, frozenset())
    return ordered


def _inputs_present(variant: Variant, spec: DerivedTableSpec) -> bool:
    """Whether every input table has materialized Parquet for this variant."""
    return all(paths.table_dir(variant, t).is_dir() for t in spec.input_tables)


def _run_query(variant: Variant, spec: DerivedTableSpec, run_set: str) -> pd.DataFrame:
    """Run the spec's query scoped to one variant + run-set, return the value frame."""
    import miscope.query

    with miscope.query.open_variant(variant, spec.input_tables, run_set=run_set) as con:
        return con.df(spec.query)


def _write_derived(
    variant: Variant,
    spec: DerivedTableSpec,
    value_frame: pd.DataFrame,
    variant_cols: dict[str, object],
    report: DerivedMaterializeReport,
) -> None:
    """Attach key columns, write the single-file table, co-emit catalog rows."""
    signature = _coord_signature(spec)
    out = schema.assemble_table(value_frame, variant_cols, signature, {})
    path = paths.semantic_parquet_path(variant, spec.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    report.tables[spec.name] = len(out)
    catalog_mod.emit_columnar_rows(
        variant,
        spec.name,
        _coords_str(spec),
        paths.SEMANTIC_TOKEN,
        out,
        path,
        spec.output_names(),
        variant.name,
    )


def _coord_signature(spec: DerivedTableSpec) -> tuple[Coord, ...]:
    """The distinct coords across the spec's outputs, in canonical order."""
    present = {c for f in spec.outputs for c in f.coords}
    return tuple(c for c in _COORD_SIGNATURE_ORDER if c in present)


def _coords_str(spec: DerivedTableSpec) -> str:
    """Comma-joined join-key names for the catalog (VARIANT -> ``variant_id``)."""
    parts = ["variant_id" if c is Coord.VARIANT else c.value for c in _coord_signature(spec)]
    return ", ".join(parts)


def _variant_columns(variant: Variant, run_set: str) -> dict[str, object]:
    """Leading key columns: ``variant_id`` + family domain params + ``run_set``.

    Mirrors :func:`miscope.warehouse.writer._variant_columns` so the derived
    table's key columns line up with the rest of the warehouse in the union.
    """
    cols = reg.variant_key_columns(variant.family)
    values: dict[str, object] = {"variant_id": variant.name}
    params = variant.params
    for col in cols:
        if col != "variant_id":
            values[col] = params.get(col)
    values["run_set"] = run_set
    return values
