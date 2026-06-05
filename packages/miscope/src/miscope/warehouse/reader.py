"""In-memory long-format surface over the columnar warehouse (REQ_110A CoS #7).

``read_table(variant, table)`` returns a :class:`WarehouseTable` wrapping a
long-format ``pandas.DataFrame``. ``to_wide(index, columns, values)`` wraps
``pandas.pivot`` for plotting; cross-variant work is a plain
``pd.concat([t.df for t in tables])`` — a no-op because ``variant_id`` is a
column, so the long format needs no schema reconciliation.

Retrofitting the existing ``.npz``-backed DataViews onto this surface is 110-D's
consumer migration; this reader serves the new warehouse tables only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from miscope.analysis.output_schema import Coord
from miscope.warehouse import paths, schema


@dataclass(frozen=True)
class WarehouseTable:
    """A materialized warehouse table held as a long-format DataFrame."""

    name: str
    df: pd.DataFrame

    @property
    def natural_index(self) -> tuple[str, ...]:
        """The documented natural ``to_wide`` index for designed tables (else ())."""
        return schema.NATURAL_WIDE_INDEX.get(self.name, ())

    def to_wide(
        self,
        index: str | list[str],
        columns: str | list[str],
        values: str | list[str],
    ) -> pd.DataFrame:
        """Pivot to wide format (``pandas.pivot``). Long stays the canonical shape."""
        return self.df.pivot(index=index, columns=columns, values=values)


class WarehouseAccessor:
    """Variant-bound convenience surface over the columnar warehouse.

    Reached via ``variant.warehouse`` — mirrors ``variant.artifacts``. Reads the
    materialized tables and triggers (re)materialization.
    """

    def __init__(self, variant: object) -> None:
        self._variant = variant

    def materialize(self) -> object:
        """(Re)materialize this variant's columnar warehouse from its npz artifacts."""
        from miscope.warehouse.writer import materialize_variant_columnar

        return materialize_variant_columnar(self._variant)

    def tables(self) -> list[str]:
        """Materialized table names under the warehouse root."""
        wdir = paths.warehouse_dir(self._variant)  # type: ignore[arg-type]
        if not wdir.exists():
            return []
        return sorted(
            p.name for p in wdir.iterdir() if p.is_dir() and p.name != paths.CATALOG_DIRNAME
        )

    def table(self, name: str, signature: tuple[Coord, ...] | str | None = None) -> WarehouseTable:
        """Read a materialized table to long-format (see :func:`read_table`)."""
        return read_table(self._variant, name, signature)

    def signatures(self, table: str) -> list[str]:
        """Signature tokens materialized for ``table``."""
        return list_signatures(self._variant, table)


def list_signatures(variant: object, table: str) -> list[str]:
    """Signature tokens (Parquet stems) materialized for ``table``."""
    tdir = paths.table_dir(variant, table)  # type: ignore[arg-type]
    if not tdir.exists():
        return []
    return sorted(p.stem for p in tdir.glob("*.parquet"))


def read_table(
    variant: object,
    table: str,
    signature: tuple[Coord, ...] | str | None = None,
) -> WarehouseTable:
    """Read a materialized table to long-format.

    ``signature`` selects among an analyzer table's coord-signature files (a
    coords tuple or its token). For a single-signature table it may be omitted.
    """
    token = _resolve_token(variant, table, signature)
    path = paths.table_dir(variant, table) / f"{token}.parquet"  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(f"No warehouse table '{table}' signature '{token}' at {path}")
    return WarehouseTable(name=table, df=pd.read_parquet(path, engine="pyarrow"))


def _resolve_token(variant: object, table: str, signature: tuple[Coord, ...] | str | None) -> str:
    if isinstance(signature, str):
        return signature
    if signature is not None:
        return paths.signature_token(signature)
    available = list_signatures(variant, table)
    if len(available) == 1:
        return available[0]
    if not available:
        raise FileNotFoundError(f"No materialized signatures for table '{table}'")
    raise ValueError(
        f"Table '{table}' has multiple signatures {available}; pass signature= to select."
    )
