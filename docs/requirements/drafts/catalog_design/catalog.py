"""
catalog.py — sketch of a queryable metadata layer over mixed columnar + tensor data.

The whole point: you EXPLORE and JOIN over descriptors (cheap, in SQL, touching no
payload), the query returns a set of typed references, and a resolver materializes ONLY
the tensor references in that result — batching per container so each file opens once.

This is a sketch, not a framework: it shows the shape of the seam, not every edge case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np


class PayloadKind(str, Enum):
    COLUMNAR = "columnar"  # value already lives in the relational store; nothing to surface
    TENSOR = "tensor"      # dense array in an external blob; resolve on demand


@dataclass(frozen=True)
class TensorRef:
    """Everything a resolver needs to dereference one dense array. No data — only address + type.

    `codec` records how to open the container:
      "npz"            -> np.savez archive; members read one at a time, siblings untouched
      "npz_compressed" -> np.savez_compressed; the selected member is decompressed, siblings are not
      "npy"            -> single .npy array; memory-mappable without a full read
    """
    uri: str
    member: str               # key within the container; "" for a single-array .npy
    dtype: str                # e.g. "float32" — declared here, verified on load
    shape: tuple[int, ...]
    codec: str = "npz"


@dataclass(frozen=True)
class CatalogRow:
    """One logical artifact.

    Join keys are the relational handles — what you EXPLORE BY — and are always present and
    always queryable. `kind` says whether a payload has to be surfaced at all. The resolution
    payload (`tensor` or `value`) is whichever the kind calls for.
    """
    id: str                        # stable surrogate key; deterministic, survives recompute
    kind: PayloadKind

    # --- join keys: plain columns, full pushdown, never require touching a payload ---
    model: str
    seed: int
    dataset: str
    layer: str | None = None
    site: str | None = None        # e.g. residual-stream site
    epoch: int | None = None
    metric: str | None = None      # for columnar rows: which scalar/series this is

    # --- resolution payload: depends on kind ---
    tensor: TensorRef | None = None   # present iff kind is TENSOR
    value: float | None = None        # present iff kind is COLUMNAR and the payload is scalar
    #   (columnar *series* live in a sibling parquet keyed by `id`; the catalog stays the index)

    # ---- relational round-trip: frozen dataclass <-> flat columns DuckDB can store ----

    def to_flat(self) -> dict[str, Any]:
        t = self.tensor
        return {
            "id": self.id, "kind": self.kind.value,
            "model": self.model, "seed": self.seed, "dataset": self.dataset,
            "layer": self.layer, "site": self.site, "epoch": self.epoch,
            "metric": self.metric, "value": self.value,
            "tensor_uri": t.uri if t else None,
            "tensor_member": t.member if t else None,
            "tensor_dtype": t.dtype if t else None,
            "tensor_shape": list(t.shape) if t else None,
            "tensor_codec": t.codec if t else None,
        }

    @classmethod
    def from_row(cls, r: Mapping[str, Any]) -> "CatalogRow":
        kind = PayloadKind(r["kind"])
        tensor = None
        if kind is PayloadKind.TENSOR:
            tensor = TensorRef(
                uri=r["tensor_uri"], member=r["tensor_member"],
                dtype=r["tensor_dtype"], shape=tuple(r["tensor_shape"]),
                codec=r["tensor_codec"],
            )
        return cls(
            id=r["id"], kind=kind,
            model=r["model"], seed=r["seed"], dataset=r["dataset"],
            layer=r.get("layer"), site=r.get("site"), epoch=r.get("epoch"),
            metric=r.get("metric"), value=r.get("value"), tensor=tensor,
        )


class TensorResolver:
    """Turns catalog rows of kind=TENSOR into materialized ndarrays.

    Materializes ONLY the rows handed to it (columnar rows are ignored — nothing to surface),
    and batches reads per container so each archive opens exactly once.
    """

    def resolve(self, rows: Iterable[CatalogRow]) -> dict[str, np.ndarray]:
        by_uri: dict[str, list[CatalogRow]] = {}
        for r in rows:
            if r.kind is PayloadKind.TENSOR:          # "not everything needs surfacing"
                by_uri.setdefault(r.tensor.uri, []).append(r)

        out: dict[str, np.ndarray] = {}
        for uri, group in by_uri.items():
            out.update(self._read_container(uri, group))
        return out

    def _read_container(self, uri: str, group: list[CatalogRow]) -> dict[str, np.ndarray]:
        codec = group[0].tensor.codec
        result: dict[str, np.ndarray] = {}
        if codec in ("npz", "npz_compressed"):
            with np.load(uri) as archive:             # NpzFile: members read lazily, one at a time
                for r in group:
                    arr = archive[r.tensor.member]    # reads ONLY this member's bytes
                    result[r.id] = self._verify(arr, r.tensor)
        elif codec == "npy":
            arr = np.load(uri, mmap_mode="r")         # memory-mapped; no full read
            result[group[0].id] = self._verify(arr, group[0].tensor)
        else:
            raise ValueError(f"unknown codec {codec!r} for {uri}")
        return result

    @staticmethod
    def _verify(arr: np.ndarray, ref: TensorRef) -> np.ndarray:
        # Fail loud if the catalog and the bytes disagree. This is the reproducibility guard:
        # the catalog *asserts* shape/dtype; the resolver checks reality matches the assertion.
        if tuple(arr.shape) != tuple(ref.shape):
            raise ValueError(f"shape mismatch for {ref.uri}::{ref.member}: "
                             f"catalog {ref.shape} vs bytes {arr.shape}")
        if arr.dtype != np.dtype(ref.dtype):
            raise ValueError(f"dtype mismatch for {ref.uri}::{ref.member}: "
                             f"catalog {ref.dtype} vs bytes {arr.dtype}")
        return arr


# ---------------------------------------------------------------------------
# End-to-end flow: catalog is itself a queryable relation. Selection (incl. JOINs)
# happens in SQL over descriptors; only the selected tensors are ever materialized.
# ---------------------------------------------------------------------------

def _dicts(con, sql: str) -> list[dict[str, Any]]:
    """Run a query and return rows as dicts (column names from the cursor description)."""
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def example(catalog_parquet: str, runs_parquet: str) -> dict[str, np.ndarray]:
    import duckdb

    con = duckdb.connect()
    # The tensor-addressing fields are just columns, so DuckDB filters/joins on them with
    # full pushdown — no array is touched to answer a metadata question.
    con.execute(f"CREATE VIEW catalog AS SELECT * FROM '{catalog_parquet}'")
    con.execute(f"CREATE VIEW runs    AS SELECT * FROM '{runs_parquet}'")

    # EXPLORATION lives here: ad-hoc SQL over the catalog, joins included, over descriptors.
    # This selects WHICH tensors by joining the catalog against run config. No bytes read.
    selected = _dicts(con, """
        SELECT c.*
        FROM catalog c
        JOIN runs r USING (model, seed)
        WHERE c.kind = 'tensor'
          AND c.site = 'mlp_out'
          AND r.grokked = TRUE
          AND c.epoch BETWEEN 20000 AND 30000
        ORDER BY c.model, c.epoch          -- pin order: a reproducible result set
    """)

    rows = [CatalogRow.from_row(m) for m in selected]
    arrays = TensorResolver().resolve(rows)   # materializes ONLY the joined-and-filtered tensors
    return arrays                              # {id -> ndarray}, ready for PCA / DMD / Fourier
