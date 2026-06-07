"""Per-variant loss curves -> the ``losses`` warehouse table (REQ_144).

The variant summary engine's largest bucket-2 input is the training/test loss
series — loss extrema, threshold-crossing epochs, the second-descent onset. Today
the engine reads that series from ``variant.metadata["train_losses"]`` /
``["test_losses"]`` (checkpoint-run metadata), reaching around the warehouse. The
losses have **no analyzer source** — they are a property of the training run, not
an analytical lens — so unlike a columnar analyzer table they cannot be derived
from an artifact. REQ_144 brings them onto the query surface as a conformed table
co-emitted here, the same warehouse-level seam :mod:`.outcomes` uses (sourced from
the metadata, not an analyzer, keeping the warehouse decoupled from the engine).

Shape: **long, one row per ``(variant, epoch)``** — the dense per-epoch series
(every training epoch, not just checkpoint epochs), so ``MIN(train_loss)`` and its
``argmin`` epoch match the engine's direct indexing into the dense list value-for-value.

Freshness (REQ_145): the losses are content-addressed by
:func:`losses_source_signature`; the materializer threads that into the shared
table-signature manifest so the dense table is rewritten only when the series
changes (a retrain), not on every pass.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import miscope.registry as reg
from miscope.analysis import signature as sig_mod
from miscope.warehouse import catalog as catalog_mod
from miscope.warehouse import paths, schema

if TYPE_CHECKING:
    from miscope.families.variant import Variant

LOSSES_TABLE = "losses"
# Bump when the losses table's shape/columns change in a way that alters output
# bytes for an unchanged loss series — folds into the source signature.
LOSSES_VERSION = 1


def losses_source_signature(variant: Variant) -> str:
    """Content signature of a variant's loss series (REQ_145 source sig).

    Folds the table version with a digest of the raw train/test loss arrays, so
    the dense table is rebuilt exactly when the series changes (a retrain). Returns
    ``""`` when the variant carries no losses — the table is then absent, and a
    consumer that needs it skips this variant.
    """
    series = _loss_series(variant)
    if series is None:
        return ""
    train, test = series
    payload = np.concatenate([train, test]).tobytes()
    content = hashlib.sha256(payload).hexdigest()
    return sig_mod.compute_signature([f"losses=v{LOSSES_VERSION}", str(len(train)), content])


def materialize_variant_losses(
    variant: Variant, run_set: str = paths.DEFAULT_RUN_SET
) -> int | None:
    """Write the variant's dense loss table + co-emit its catalog rows (REQ_144).

    Returns the row count, or ``None`` when the variant carries no loss series.
    """
    series = _loss_series(variant)
    if series is None:
        return None
    train, test = series
    value_frame = pd.DataFrame(
        {
            "epoch": np.arange(len(train), dtype=np.int64),
            "train_loss": train,
            "test_loss": test,
        }
    )
    variant_cols = _variant_columns(variant, run_set)
    out = schema.assemble_table(value_frame, variant_cols, (), {})

    path = paths.semantic_parquet_path(variant, LOSSES_TABLE)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

    catalog_mod.emit_columnar_rows(
        variant,
        LOSSES_TABLE,
        "epoch",  # the single join-key coordinate
        paths.SEMANTIC_TOKEN,
        out,
        path,
        ("train_loss", "test_loss"),
        variant.name,
    )
    return len(out)


def _loss_series(variant: Variant) -> tuple[np.ndarray, np.ndarray] | None:
    """The variant's ``(train_losses, test_losses)`` as float64 arrays, or ``None``.

    Both series must be present and equal-length to form the dense per-epoch table.
    """
    metadata = getattr(variant, "metadata", None)
    if not metadata:
        return None
    train = metadata.get("train_losses")
    test = metadata.get("test_losses")
    if train is None or test is None:
        return None
    train_arr = np.asarray(train, dtype=np.float64)
    test_arr = np.asarray(test, dtype=np.float64)
    if train_arr.shape != test_arr.shape or train_arr.ndim != 1:
        return None
    return train_arr, test_arr


def _variant_columns(variant: Variant, run_set: str) -> dict[str, object]:
    """Leading key columns: ``variant_id`` + family domain params + ``run_set``.

    Mirrors :func:`miscope.warehouse.writer._variant_columns` so this table's key
    columns line up with the rest of the warehouse in the cross-variant union.
    """
    cols = reg.variant_key_columns(variant.family)
    values: dict[str, object] = {"variant_id": variant.name}
    params = variant.params
    for col in cols:
        if col != "variant_id":
            values[col] = params.get(col)
    values["run_set"] = run_set
    return values
