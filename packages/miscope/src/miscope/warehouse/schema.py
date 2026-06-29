"""Warehouse table assembly + the REQ_110 semantic-table schemas (REQ_110A).

Two responsibilities:

1. :func:`assemble_table` — attach the expanded ``variant`` columns, the
   ``group_type``/``operation_type`` discriminators, and the reserved nullable
   provenance columns to a value frame, in a canonical column order. Every
   long-format table carries the reserved slots so the schema is explicitly open
   (REQ_110: ``analyzer_version``, ``conditions_satisfied``, ``trust_tier``).

2. The discriminator vocabularies (:class:`GroupType`, :class:`OperationType`)
   and the natural ``to_wide`` index for each designed semantic table — the
   columns a consumer pivots on. The semantic tables themselves are populated by
   :mod:`.mapping`; this module owns their shared shape.
"""

from __future__ import annotations

from enum import Enum
from typing import cast

import pandas as pd

from miscope.analysis.output_schema import Coord

# Discriminator + reserved column names (REQ_110).
GROUP_TYPE = "group_type"
OPERATION_TYPE = "operation_type"
RESERVED_PROVENANCE = ("analyzer_version", "conditions_satisfied", "trust_tier")

# Canonical coord-column order for a long table (after the variant columns).
# `head` follows `site` — it is an attention-site sub-axis (REQ_136).
_COORD_ORDER = (
    "window",
    "epoch",
    "site",
    "head",
    "group",
    "frequency",
    "neuron",
    "row_id",
    "pc_index",
)


class GroupType(str, Enum):
    """What a ``group`` column refers to — the REQ_110 ``group_type`` enum."""

    WEIGHT_MATRIX = "weight_matrix"
    WEIGHT_COMPONENT = "weight_component"
    ACTIVATION_SITE = "activation_site"
    FREQUENCY_GROUP = "frequency_group"
    SINGLE_CENTROID = "single_centroid"
    CENTROID_GROUP = "centroid_group"


class OperationType(str, Enum):
    """The measurement / transform that produced a row — REQ_110 ``operation_type``."""

    PCA = "pca"
    PCA_SUMMARY = "pca_summary"
    PCA_ROLLING = "pca_rolling"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    CIRCULARITY = "circularity"
    FOURIER_ALIGNMENT = "fourier_alignment"
    CURVATURE = "curvature"
    SIGMOIDALITY = "sigmoidality"


# Natural ``to_wide`` index per designed semantic table (REQ_110A CoS #7 / #2).
NATURAL_WIDE_INDEX: dict[str, tuple[str, ...]] = {
    "pca_results": ("variant_id", "epoch", "group", "operation_type", "pc_index"),
    "pca_projections": ("variant_id", "epoch", "group", "operation_type", "row_id"),
    "frequency_spectrum": ("variant_id", "epoch", "site", "frequency"),
    "neuron_frequency_attribution": ("variant_id", "epoch", "neuron", "frequency"),
    "shape_characterizations": ("variant_id", "epoch", "group", "operation_type"),
    # REQ_156: Layer 4 circuit object table (site = circuit discriminator, per head).
    "circuit_spectra": ("variant_id", "epoch", "site", "head"),
}


def assemble_table(
    value_frame: pd.DataFrame,
    variant_cols: dict[str, object],
    signature: tuple[Coord, ...],  # noqa: ARG001 — kept for caller symmetry / future ordering
    discriminators: dict[str, object],
) -> pd.DataFrame:
    """Attach variant columns, discriminators, and reserved provenance; order columns.

    ``discriminators`` may carry ``group_type``/``operation_type`` constants (and,
    later, provenance). Absent slots are written as nullable columns.
    """
    df = value_frame.reset_index(drop=True).copy()

    # Semantic reshapers emit per-row discriminators as columns already; only fill
    # a discriminator from the constant when the frame doesn't carry it.
    for col in (GROUP_TYPE, OPERATION_TYPE):
        if col not in df.columns:
            df[col] = discriminators.get(col)
    for col in RESERVED_PROVENANCE:
        df[col] = discriminators.get(col)

    # Variant columns lead (insert in reverse so order is preserved at the front).
    for col, val in reversed(list(variant_cols.items())):
        if col in df.columns:
            df = df.drop(columns=[col])
        df.insert(0, col, val)

    return cast(pd.DataFrame, df[_ordered_columns(df, tuple(variant_cols.keys()))])


def _ordered_columns(df: pd.DataFrame, variant_cols: tuple[str, ...]) -> list[str]:
    """Variant cols, coords, discriminators, value columns, reserved provenance."""
    lead = list(variant_cols)
    coords = [c for c in _COORD_ORDER if c in df.columns and c not in lead]
    disc = [GROUP_TYPE, OPERATION_TYPE]
    reserved = list(RESERVED_PROVENANCE)
    fixed = set(lead + coords + disc + reserved)
    values = [c for c in df.columns if c not in fixed]
    return lead + coords + disc + values + reserved
