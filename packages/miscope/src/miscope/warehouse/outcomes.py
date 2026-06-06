"""Per-variant outcome rollup -> the ``variant_outcomes`` warehouse table (REQ_110D).

The summary engine (:class:`~miscope.analysis.variant_analysis_summary.VariantAnalysisSummary`)
writes ``variant_summary.json`` — a per-variant outcome snapshot (grokking onset,
homeless fraction, failure mode, learned frequencies, window metrics, …). REQ_110D
brings that snapshot onto the query surface so the cross-variant questions the CoS
names ("variants with ``homeless_fraction > 0.2``", "grokking onset across
variants") are one-line SQL, and so ``build_variant_registry`` becomes a SELECT
rather than a JSON glob.

Unlike the columnar tables in :mod:`.writer`, this is a *cross-analyzer rollup*,
not an analyzer's declared output — so it has its own co-emission seam here,
sourced from the already-written ``variant_summary.json`` (keeping the warehouse
decoupled from the analysis engine, mirroring "warehouse materializes from
artifacts"). It is **wide, one row per (variant, run_set)** — a deliberate
exception to long-format-canonical, because it replaces ``variant_registry.json``,
which is already one row per variant.

Shape: every scalar summary field is promoted to a queryable column (denormalized
for SQL convenience, as REQ_110 permits); the full snapshot is also carried as a
``summary_json`` string so the registry rebuild reconstructs each entry
byte-identically (registry consumers read nested window dicts and the
``performance_classification`` tuple, which do not flatten to scalar columns).
"""

from __future__ import annotations

import json

import pandas as pd

import miscope.registry as reg
from miscope.warehouse import catalog as catalog_mod
from miscope.warehouse import paths, schema

OUTCOMES_TABLE = "variant_outcomes"
SUMMARY_JSON_COLUMN = "summary_json"


def materialize_variant_outcomes(
    variant: object, run_set: str = paths.DEFAULT_RUN_SET
) -> int | None:
    """Write the variant's outcome row to the warehouse + co-emit its catalog rows.

    Returns the row count (always 1) or ``None`` when the variant has no
    ``variant_summary.json`` yet (a fresh variant analyzed but not summarized —
    the row appears on the next materialize once the summary exists).
    """
    summary_path = getattr(variant, "summary_path", None)
    if summary_path is None or not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    value_frame = _flatten_outcomes(summary)
    variant_cols = _variant_columns(variant, run_set)
    out = schema.assemble_table(value_frame, variant_cols, (), {})

    path = paths.semantic_parquet_path(variant, OUTCOMES_TABLE)  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

    catalog_mod.emit_columnar_rows(
        variant,
        OUTCOMES_TABLE,
        "",  # no coord join keys — keyed only by variant_id / run_set
        paths.SEMANTIC_TOKEN,
        out,
        path,
        tuple(value_frame.columns),
        variant.name,  # type: ignore[attr-defined]
    )
    return len(out)


def _flatten_outcomes(summary: dict) -> pd.DataFrame:
    """One-row frame: scalar summary fields promoted to columns + the JSON carrier.

    Non-scalar fields (lists of frequencies, nested window dicts, the
    ``performance_classification`` tuple) stay only inside ``summary_json``; the
    registry rebuild reads them back from there.
    """
    row: dict[str, object] = {
        key: value
        for key, value in summary.items()
        if value is None or isinstance(value, (int, float, str, bool))
    }
    row[SUMMARY_JSON_COLUMN] = json.dumps(summary)
    return pd.DataFrame([row])


def _variant_columns(variant: object, run_set: str) -> dict[str, object]:
    """Leading key columns: ``variant_id`` + family domain params + ``run_set``.

    Mirrors :func:`miscope.warehouse.writer._variant_columns` so this table's key
    columns line up with the rest of the warehouse in the cross-variant union.
    """
    cols = reg.variant_key_columns(variant.family)  # type: ignore[attr-defined]
    values: dict[str, object] = {"variant_id": variant.name}  # type: ignore[attr-defined]
    params = variant.params  # type: ignore[attr-defined]
    for col in cols:
        if col != "variant_id":
            values[col] = params.get(col)
    values["run_set"] = run_set
    return values
