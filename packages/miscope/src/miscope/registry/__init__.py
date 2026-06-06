"""``miscope.registry`` — the INFORMATION_SCHEMA for analysis (REQ_107).

A single canonical surface that answers *"what does this codebase compute, and
how do I consume it?"* before you write a new analyzer or re-derive something
that already exists. It enumerates every analyzer output field and every
DataView, with each field's ``kind`` (columnar|tensor), keying ``coords``, and a
one-line description, plus the producer→consumer edges between them.

First five minutes
------------------
::

    import miscope.registry as reg

    reg.analyzers()            # every analyzer output field (a DataFrame)
    reg.derived()              # every derived-table output field (REQ_141)
    reg.dataviews()            # every queryable DataView field
    reg.search("frequency")    # do we already have something for X?
    reg.field("dominant_freq") # who produces it, how it's keyed, who consumes it

In a notebook the table-returning calls render as DataFrames. ``reg.load()``
eagerly builds and validates the registry (the CI gate); it raises
:class:`RegistryError` if any analyzer is missing its output schema or a
DataView's declared source has drifted from its producer.

Why it exists
-------------
Even with clean layering, analyzers re-derive data their author didn't know
existed. The registry is the discoverability half of the fix: a field declared
once (on its ``AnalyzerSpec.outputs``) with its keying coords becomes a *join
target* — ``reg.field("dominant_freq")`` shows it is keyed by
``(variant, epoch, neuron)`` and produced by ``neuron_dynamics``, so a consumer
joins it instead of re-running the argmax. The registry enumerates and
cross-references; it does not explain — long-form lives in docstrings.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from miscope.registry._index import (
    FieldInfo,
    RegistryError,
    RegistryIndex,
    build_index,
    lookup_field,
    validate,
)

__all__ = [
    "analyzers",
    "dataviews",
    "derived",
    "field",
    "search",
    "load",
    "variant_key_columns",
    "index",
    "FieldInfo",
    "RegistryError",
    "RegistryIndex",
]


@lru_cache(maxsize=1)
def index() -> RegistryIndex:
    """The built registry index (cached). Lazily constructed on first use."""
    return build_index()


def load() -> RegistryIndex:
    """Build and validate the registry — the CI entry point and drift gate.

    Returns the validated index. Raises :class:`RegistryError` if any analyzer
    declares no output schema, a declared dtype is unknown, or a DataView source
    has drifted from its producer's declared schema/version.
    """
    idx = index()
    validate(idx)
    return idx


def variant_key_columns(family: object) -> tuple[str, ...]:
    """Family-owned expansion of the ``variant`` coord (``variant_id`` + params)."""
    return index().variant_key_columns(family)


# ---------------------------------------------------------------------------
# Tabular surfaces (DataFrames render in Jupyter via _repr_html_)
# ---------------------------------------------------------------------------


def analyzers() -> pd.DataFrame:
    """Every analyzer output field as a long-format DataFrame.

    Columns: ``analyzer, scope, version, field, kind, coords, dtype, description``.
    One row per declared output field — the columns of the analysis warehouse.
    """
    rows = [
        {
            "analyzer": s.name,
            "scope": s.output_scope,
            "version": s.version,
            "field": f.name,
            "kind": f.kind.value,
            "coords": ", ".join(f.coord_names),
            "dtype": f.dtype,
            "description": f.description,
        }
        for s in index().analyzers
        for f in s.outputs
    ]
    return pd.DataFrame(rows, columns=_ANALYZER_COLUMNS)


def dataviews() -> pd.DataFrame:
    """Every DataView field with its source dependencies as a DataFrame.

    Columns: ``dataview, field, kind, coords, sources, description``.
    """
    rows = []
    for dv in index().dataviews:
        src_str = "; ".join(f"{s.analyzer_name}({', '.join(s.fields)})" for s in dv.sources)
        for f in dv.schema.fields:
            rows.append(
                {
                    "dataview": dv.name,
                    "field": f.name,
                    "kind": f.kind.value if f.kind else "",
                    "coords": ", ".join(c.value for c in f.coords),
                    "sources": src_str,
                    "description": f.description,
                }
            )
    return pd.DataFrame(rows, columns=_DATAVIEW_COLUMNS)


def derived() -> pd.DataFrame:
    """Every derived-table output field as a long-format DataFrame (REQ_141).

    Columns: ``derived_table, version, materialized, field, kind, coords, dtype,
    input_tables, description``. One row per declared output field — the columns a
    derived table contributes to the warehouse, alongside :func:`analyzers`.
    """
    rows = [
        {
            "derived_table": d.name,
            "version": d.version,
            "materialized": d.materialized,
            "field": f.name,
            "kind": f.kind.value,
            "coords": ", ".join(f.coord_names),
            "dtype": f.dtype,
            "input_tables": ", ".join(d.input_tables),
            "description": f.description,
        }
        for d in index().derived
        for f in d.outputs
    ]
    return pd.DataFrame(rows, columns=_DERIVED_COLUMNS)


def search(query: str) -> pd.DataFrame:
    """Substring search over analyzer + derived + DataView fields.

    Returns a combined DataFrame with a ``surface`` column (``analyzer``,
    ``derived``, or ``dataview``) so related fields across all planes show together.
    """
    q = query.lower()

    def _match(*texts: str) -> bool:
        return any(q in t.lower() for t in texts)

    rows = []
    for s in index().analyzers:
        for f in s.outputs:
            if _match(s.name, f.name, f.description):
                rows.append(
                    {
                        "surface": "analyzer",
                        "name": s.name,
                        "field": f.name,
                        "kind": f.kind.value,
                        "coords": ", ".join(f.coord_names),
                        "description": f.description,
                    }
                )
    for d in index().derived:
        for f in d.outputs:
            if _match(d.name, f.name, f.description):
                rows.append(
                    {
                        "surface": "derived",
                        "name": d.name,
                        "field": f.name,
                        "kind": f.kind.value,
                        "coords": ", ".join(f.coord_names),
                        "description": f.description,
                    }
                )
    for dv in index().dataviews:
        for f in dv.schema.fields:
            if _match(dv.name, f.name, f.description):
                rows.append(
                    {
                        "surface": "dataview",
                        "name": dv.name,
                        "field": f.name,
                        "kind": f.kind.value if f.kind else "",
                        "coords": ", ".join(c.value for c in f.coords),
                        "description": f.description,
                    }
                )
    return pd.DataFrame(rows, columns=_SEARCH_COLUMNS)


def field(name: str) -> FieldInfo:
    """Reverse lookup for a field: producer, exposing DataViews, and consumers."""
    return lookup_field(index(), name)


_ANALYZER_COLUMNS = [
    "analyzer",
    "scope",
    "version",
    "field",
    "kind",
    "coords",
    "dtype",
    "description",
]
_DATAVIEW_COLUMNS = ["dataview", "field", "kind", "coords", "sources", "description"]
_DERIVED_COLUMNS = [
    "derived_table",
    "version",
    "materialized",
    "field",
    "kind",
    "coords",
    "dtype",
    "input_tables",
    "description",
]
_SEARCH_COLUMNS = ["surface", "name", "field", "kind", "coords", "description"]
