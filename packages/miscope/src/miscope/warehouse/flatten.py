"""Flatten one declared field's array into long-format rows — REQ_110A.

Given a columnar :class:`OutputField`, its on-disk array, the prefix-coord values
parsed by :mod:`.decompose`, and the epoch context, produce a long-format
``pandas.DataFrame`` with one column per coordinate plus one value column named
for the field. Array-axis coords (``neuron``, ``row_id``, ``frequency``, and —
for cross-epoch analyzers — ``epoch``) are unrolled from the array's shape;
``site``/``group`` prefix coords and a per-epoch ``epoch`` are constant columns.

The flattener is generic: it consumes only the declared coords and the array. All
per-analyzer key-composition knowledge lives in :mod:`.decompose`; all
data-dependent axis labels (group frequencies, qualified frequencies, epoch
values) are passed in by the writer as ``axis_label_values``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from miscope.analysis.output_schema import Coord, OutputField

# Coords whose axis is labelled positionally (0..n-1) unless the writer supplies
# data-dependent labels.
_POSITIONAL_DEFAULT = frozenset(
    {Coord.NEURON, Coord.ROW_ID, Coord.FREQUENCY, Coord.GROUP, Coord.HEAD}
)


def flatten_field(
    field: OutputField,
    array: np.ndarray,
    prefix_coords: dict[str, str],
    *,
    epoch: int | None,
    epoch_is_axis: bool,
    axis_label_values: dict[Coord, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Long-format rows for one field's array.

    Args:
        field: The declared columnar output field.
        array: Its on-disk array (the npz member assigned to this field).
        prefix_coords: ``{coord_value: token}`` parsed from the npz key (constant
            ``site``/``group`` columns); only coords the field declares.
        epoch: The epoch this array belongs to (per-epoch analyzers); used as a
            constant column when ``epoch_is_axis`` is False.
        epoch_is_axis: True for cross-epoch analyzers whose array carries the
            epoch axis (labelled via ``axis_label_values[EPOCH]``).
        axis_label_values: Data-dependent labels for ``group``/``frequency``/
            ``epoch`` axes, keyed by coord; positional fallback otherwise.
    """
    labels = axis_label_values or {}
    axis_coords = _axis_coords(field, prefix_coords, epoch_is_axis)
    arr = _conform_rank(np.asarray(array), axis_coords)
    axis_indices = _axis_index_grids(arr, axis_coords)

    columns: dict[str, np.ndarray] = {}
    for coord, idx in zip(axis_coords, axis_indices, strict=True):
        columns[coord.value] = _label_axis(coord, idx, arr.shape, labels)

    n = arr.size
    for coord_value, token in prefix_coords.items():
        columns[coord_value] = np.repeat(token, n)
    if not epoch_is_axis and Coord.EPOCH in field.coords and epoch is not None:
        columns[Coord.EPOCH.value] = np.repeat(np.int64(epoch), n)

    columns[field.name] = arr.ravel(order="C")
    return pd.DataFrame(columns)


def _axis_coords(
    field: OutputField, prefix_coords: dict[str, str], epoch_is_axis: bool
) -> tuple[Coord, ...]:
    """Declared coords that are array axes, in declaration (= array-axis) order."""
    out: list[Coord] = []
    for c in field.coords:
        if c is Coord.VARIANT:
            continue
        if c.value in prefix_coords:
            continue
        if c is Coord.EPOCH and not epoch_is_axis:
            continue
        out.append(c)
    return tuple(out)


def _conform_rank(arr: np.ndarray, axis_coords: tuple[Coord, ...]) -> np.ndarray:
    """Reconcile array rank with the declared axis count.

    The attention head axis that ``weight_spectra.sv`` and
    ``weight_basis_projection.dominant_frequency`` carry is now a first-class
    ``head`` coordinate (REQ_136), emitted uniform-rank, so it maps to its own
    column rather than being folded. The fold below remains a defensive fallback:
    if an array still carries *more* axes than its field declares, the undeclared
    leading axes collapse into the last declared axis (lossless in values; the
    composite index keys the row). Fewer axes than declared is a genuine
    declaration error and raises.
    """
    k = len(axis_coords)
    if k == 0:
        if arr.size != 1:
            raise ValueError(f"scalar field expected size 1, got shape {arr.shape}")
        return arr
    if arr.ndim < k:
        raise ValueError(
            f"field axes {[c.value for c in axis_coords]} expect {k}D array, got shape {arr.shape}"
        )
    if arr.ndim > k:
        return arr.reshape(*arr.shape[: k - 1], -1)
    return arr


def _axis_index_grids(arr: np.ndarray, axis_coords: tuple[Coord, ...]) -> list[np.ndarray]:
    """Per-axis flattened index arrays aligned with ``arr.ravel('C')``."""
    if not axis_coords:
        return []
    grids = np.indices(arr.shape)  # (ndim, *shape)
    return [g.ravel(order="C") for g in grids]


def _label_axis(
    coord: Coord,
    idx: np.ndarray,
    shape: tuple[int, ...],
    labels: dict[Coord, np.ndarray],
) -> np.ndarray:
    """Map positional axis indices to labels (data-dependent if supplied)."""
    if coord in labels and coord not in (Coord.NEURON, Coord.ROW_ID):
        label_vals = np.asarray(labels[coord])
        return label_vals[idx]
    if coord in _POSITIONAL_DEFAULT or coord is Coord.EPOCH:
        return idx.astype(np.int64)
    return idx.astype(np.int64)
