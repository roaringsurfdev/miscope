"""Columnar catalog co-emission (REQ_110A's half of the shared catalog relation).

The catalog is the queryable index over every field — columnar and tensor — with
its coordinate columns and a ``kind`` discriminator (REQ_110). 110-A populates the
``columnar`` rows; 110-B adds the ``tensor`` rows (``TensorRef``) and finalizes the
shared relation. Rows are **co-emitted with the Parquet payload** — written in the
same pass as the table they index, never by a later scan — so the index cannot
drift from the bytes by construction.

One catalog row per columnar *field* (value column): it records where the field's
payload lives (the table Parquet + the value column within it) and the coordinate
columns it is keyed by, so a consumer can discover and join it without reading the
payload. The row shape deliberately mirrors the flat columns of the catalog sketch
(``docs/requirements/drafts/catalog_design/catalog.py``); 110-B owns the final
schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from miscope.warehouse import paths


@dataclass(frozen=True)
class ColumnarCatalogRow:
    """One columnar field's catalog entry — an address, not a payload."""

    id: str  # stable surrogate: {variant_id}/{table}/{signature}/{field}
    kind: str  # always "columnar" here
    variant_id: str
    table: str
    field: str  # the value column within the table Parquet
    coords: str  # comma-joined coord names (the join keys)
    parquet_uri: str  # the table Parquet holding the payload
    n_rows: int


def emit_columnar_rows(
    variant: object,
    table: str,
    coords_str: str,
    sig_token: str,
    df: pd.DataFrame,
    parquet_path: Path,
    value_columns: tuple[str, ...],
    variant_id: str,
) -> None:
    """Append the catalog rows for one table's columnar value columns.

    Co-emitted in the same pass as ``parquet_path`` was written. ``coords_str`` is
    the comma-joined join-key names; ``sig_token`` disambiguates a table's files.
    """
    # Store the payload location relative to the warehouse root so the catalog
    # stays portable if the variant tree moves (the warehouse is regeneratable).
    rel_uri = parquet_path.relative_to(paths.warehouse_dir(variant)).as_posix()  # type: ignore[arg-type]
    rows = [
        ColumnarCatalogRow(
            id=f"{variant_id}/{table}/{sig_token}/{col}",
            kind="columnar",
            variant_id=variant_id,
            table=table,
            field=col,
            coords=coords_str,
            parquet_uri=rel_uri,
            n_rows=len(df),
        )
        for col in value_columns
    ]
    _append_catalog(variant, table, rows)


def _append_catalog(variant: object, table: str, rows: list[ColumnarCatalogRow]) -> None:
    """Write/merge a table's catalog Parquet (idempotent per (variant, table))."""
    path = paths.catalog_parquet_path(variant, table)  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([asdict(r) for r in rows])
    if path.exists():
        existing = pd.read_parquet(path, engine="pyarrow")
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["id"], keep="last", ignore_index=True)
    else:
        combined = new
    combined.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
