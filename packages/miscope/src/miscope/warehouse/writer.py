"""Columnar materializer: npz artifacts -> long-format Parquet warehouse (REQ_110A).

``materialize_variant_columnar(variant)`` is the entry point. For each analyzer
with declared ``columnar`` outputs and artifacts on disk it:

1. reads the npz (through the sanctioned ``variant.artifacts`` accessor),
2. decomposes keys -> ``(field, prefix coords)`` (:mod:`.decompose`),
3. flattens each columnar field to long rows (:mod:`.flatten`),
4. routes fields to a semantic table (:mod:`.mapping`) or the generic
   analyzer-named fallback,
5. groups rows by coord signature and writes one Parquet per
   ``(table, signature)`` — ``epoch`` is always a column, never a file — plus the
   co-emitted columnar catalog rows (:mod:`.catalog`), in the same pass.

``tensor``-kind fields are skipped (owned by 110-B); the ``.npz`` blobs are
untouched. Driven entirely from the registry's declared schema, so it round-trips
deterministically from checkpoints regardless of what happens to be on disk.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dc_field

import pandas as pd

import miscope.registry as reg
from miscope.analysis.output_schema import Coord, FieldKind, OutputField
from miscope.warehouse import catalog as catalog_mod
from miscope.warehouse import mapping, mapping_semantic, paths, schema
from miscope.warehouse.decompose import KeyMatch, assign_keys, get_decomp
from miscope.warehouse.flatten import flatten_field


@dataclass
class MaterializeReport:
    """Summary of one variant's columnar materialization."""

    variant_id: str
    files_written: list[str] = dc_field(default_factory=list)
    tables: dict[str, int] = dc_field(default_factory=dict)  # table -> row count
    skipped_analyzers: list[str] = dc_field(default_factory=list)


# A field's long frame before variant columns / discriminators are attached.
@dataclass
class _FieldFrame:
    field: OutputField
    frame: pd.DataFrame


def materialize_variant_columnar(variant: object) -> MaterializeReport:
    """Materialize all of a variant's columnar analyzer outputs to Parquet."""
    report = MaterializeReport(variant_id=variant.name)  # type: ignore[attr-defined]
    # Deterministic regeneration: wipe the prior columnar outputs so a field that
    # moved tables (generic <-> semantic) leaves no stale Parquet behind. The wipe
    # is selective — it preserves the 110-B tensor catalog so the two halves of
    # the shared catalog relation materialize independently (either order).
    _wipe_columnar_outputs(variant)
    variant_cols = _variant_columns(variant)
    semantic: dict[str, _SemanticAcc] = defaultdict(_SemanticAcc)

    for spec in reg.index().analyzers:
        if not any(f.kind is FieldKind.COLUMNAR for f in spec.outputs):
            continue
        # Availability is decided per scope inside _build_field_frames (per-epoch
        # checks get_epochs; cross-epoch checks cross_epoch.npz) — the loader's
        # get_available_analyzers only counts epoch_* dirs, missing cross-epoch ones.
        field_frames = _build_field_frames(variant, spec)
        if not field_frames:
            report.skipped_analyzers.append(spec.name)
            continue
        claimed = _collect_semantic(spec.name, field_frames, semantic)
        generic = [ff for ff in field_frames if ff.field.name not in claimed]
        _emit_generic(variant, spec.name, generic, variant_cols, report)

    _emit_semantic(variant, semantic, variant_cols, report)
    return report


# ---------------------------------------------------------------------------
# Loading + flattening into per-field long frames
# ---------------------------------------------------------------------------


def _build_field_frames(variant: object, spec: object) -> list[_FieldFrame]:
    decomp = get_decomp(spec.name)  # type: ignore[attr-defined]
    if spec.output_scope == "per_epoch":  # type: ignore[attr-defined]
        return _per_epoch_frames(variant, spec, decomp)
    return _cross_epoch_frames(variant, spec, decomp)


def _per_epoch_frames(variant: object, spec: object, decomp: object) -> list[_FieldFrame]:
    loader = variant.artifacts  # type: ignore[attr-defined]
    epochs = loader.get_epochs(spec.name)  # type: ignore[attr-defined]
    if not epochs:
        return []
    parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    fields: dict[str, OutputField] = {}
    loose: set[str] = set()
    for epoch in epochs:
        npz = loader.load_epoch(spec.name, epoch)  # type: ignore[attr-defined]
        labels = _resolve_labels(decomp, npz, epoch_axis=False)
        for m in _columnar_matches(spec, npz):
            df = flatten_field(
                m.field,
                npz[m.npz_key],
                m.prefix_coords,
                epoch=epoch,
                epoch_is_axis=False,
                axis_label_values=labels,
            )
            if m.loose:
                df = df.drop_duplicates()
                loose.add(m.field.name)
            parts[m.field.name].append(df)
            fields[m.field.name] = m.field
    return _concat_parts(parts, fields, loose)


def _cross_epoch_frames(variant: object, spec: object, decomp: object) -> list[_FieldFrame]:
    loader = variant.artifacts  # type: ignore[attr-defined]
    try:
        npz = loader.load_cross_epoch(spec.name)  # type: ignore[attr-defined]
    except FileNotFoundError:
        return []
    labels = _resolve_labels(decomp, npz, epoch_axis=True)
    parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    fields: dict[str, OutputField] = {}
    loose: set[str] = set()
    for m in _columnar_matches(spec, npz):
        epoch_is_axis = Coord.EPOCH in m.field.coords
        df = flatten_field(
            m.field,
            npz[m.npz_key],
            m.prefix_coords,
            epoch=None,
            epoch_is_axis=epoch_is_axis,
            axis_label_values=labels,
        )
        if m.loose:
            df = df.drop_duplicates()
            loose.add(m.field.name)
        parts[m.field.name].append(df)
        fields[m.field.name] = m.field
    return _concat_parts(parts, fields, loose)


def _columnar_matches(spec: object, npz: dict) -> list[KeyMatch]:
    matches = assign_keys(spec.name, spec.outputs, tuple(npz.keys()))  # type: ignore[attr-defined]
    return [m for m in matches if m.field.kind is FieldKind.COLUMNAR]


def _concat_parts(
    parts: dict[str, list[pd.DataFrame]],
    fields: dict[str, OutputField],
    loose: set[str],
) -> list[_FieldFrame]:
    out: list[_FieldFrame] = []
    for name, frames in parts.items():
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        if name in loose:
            df = df.drop_duplicates(ignore_index=True)
        out.append(_FieldFrame(field=fields[name], frame=df))
    return out


def _resolve_labels(decomp: object, npz: dict, *, epoch_axis: bool) -> dict[Coord, object]:
    labels: dict[Coord, object] = {}
    for coord, field_name in decomp.axis_labels.items():  # type: ignore[attr-defined]
        if field_name in npz:
            labels[coord] = npz[field_name]
    if epoch_axis and "epochs" in npz:
        labels[Coord.EPOCH] = npz["epochs"]
    return labels


# ---------------------------------------------------------------------------
# Routing + writing
# ---------------------------------------------------------------------------


@dataclass
class _SemanticAcc:
    """Row accumulator for one conformed semantic table across all feeder analyzers."""

    frames: list[pd.DataFrame] = dc_field(default_factory=list)
    value_columns: set[str] = dc_field(default_factory=set)


def _collect_semantic(
    analyzer_name: str,
    field_frames: list[_FieldFrame],
    semantic: dict[str, _SemanticAcc],
) -> set[str]:
    """Apply this analyzer's claims; accumulate semantic rows. Returns claimed fields."""
    by_name = {ff.field.name: ff.frame for ff in field_frames}
    claimed: set[str] = set()
    for claim in mapping_semantic.claims_for(analyzer_name):
        subset = {name: by_name[name] for name in claim.fields if name in by_name}
        if not subset:
            continue
        rows = claim.reshaper(subset)
        if rows.empty:
            continue
        acc = semantic[claim.table]
        acc.frames.append(rows)
        acc.value_columns |= _value_columns(rows)
        claimed |= subset.keys()
    return claimed


def _emit_generic(
    variant: object,
    analyzer_name: str,
    field_frames: list[_FieldFrame],
    variant_cols: dict[str, object],
    report: MaterializeReport,
) -> None:
    """Write the unclaimed fields as generic tables: one Parquet per coord signature."""
    groups: dict[tuple[Coord, ...], list[_FieldFrame]] = defaultdict(list)
    for ff in field_frames:
        groups[ff.field.coords].append(ff)
    for signature, members in groups.items():
        merged = _merge_on_coords([m.frame for m in members])
        value_columns = tuple(m.field.name for m in members)
        discriminators = mapping.generic_discriminators(analyzer_name, merged.columns)
        out = schema.assemble_table(merged, variant_cols, signature, discriminators)
        coords_str = ", ".join(c.value for c in signature)
        _write_table(
            variant,
            analyzer_name,
            paths.table_parquet_path(variant, analyzer_name, signature),
            coords_str,
            paths.signature_token(signature),
            out,
            value_columns,
            report,
        )


def _emit_semantic(
    variant: object,
    semantic: dict[str, _SemanticAcc],
    variant_cols: dict[str, object],
    report: MaterializeReport,
) -> None:
    """Row-union each semantic table's feeder frames and write one Parquet per table."""
    for table, acc in semantic.items():
        big = pd.concat(acc.frames, ignore_index=True)
        _normalize_label_columns(big)
        out = schema.assemble_table(big, variant_cols, (), {})
        coords_str = ", ".join(schema.NATURAL_WIDE_INDEX.get(table, ()))
        _write_table(
            variant,
            table,
            paths.semantic_parquet_path(variant, table),
            coords_str,
            paths.SEMANTIC_TOKEN,
            out,
            tuple(sorted(acc.value_columns)),
            report,
        )


def _normalize_label_columns(df: pd.DataFrame) -> None:
    """Coerce ``group``/``site`` to nullable strings (REQ_110: group is a string).

    A union across feeders mixes site names with frequency-group ints in one
    column; Parquet needs a uniform type. Nulls are preserved (not stringified).
    """
    for col in ("group", "site"):
        if col in df.columns:
            df[col] = df[col].map(lambda x: None if pd.isna(x) else str(x))


def _value_columns(frame: pd.DataFrame) -> set[str]:
    """Value columns of a semantic frame (everything but coords + discriminators)."""
    known = set(schema._COORD_ORDER) | {schema.GROUP_TYPE, schema.OPERATION_TYPE}
    return {c for c in frame.columns if c not in known}


def _merge_on_coords(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine value frames sharing a coord signature into one wide-on-value table.

    Frames join on their shared coord columns. Variant-only scalar fields carry no
    coord column (a single constant row each); those concat horizontally.
    """
    if len(frames) == 1:
        return frames[0]
    merged = frames[0]
    for nxt in frames[1:]:
        join_cols = [c for c in merged.columns if c in nxt.columns]
        if join_cols:
            merged = merged.merge(nxt, on=join_cols, how="outer")
        else:
            merged = pd.concat([merged.reset_index(drop=True), nxt.reset_index(drop=True)], axis=1)
    return merged


def _write_table(
    variant: object,
    table: str,
    path: object,
    coords_str: str,
    sig_token: str,
    df: pd.DataFrame,
    value_columns: tuple[str, ...],
    report: MaterializeReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    report.files_written.append(str(path))
    report.tables[table] = report.tables.get(table, 0) + len(df)
    catalog_mod.emit_columnar_rows(
        variant,
        table,
        coords_str,
        sig_token,
        df,
        path,
        value_columns,
        variant.name,  # type: ignore[attr-defined]
    )


def _wipe_columnar_outputs(variant: object) -> None:
    """Remove the columnar warehouse children, preserving the tensor catalog (110-B)."""
    wdir = paths.warehouse_dir(variant)  # type: ignore[arg-type]
    if not wdir.exists():
        return
    for child in wdir.iterdir():
        if child.name == paths.TENSOR_CATALOG_DIRNAME:
            continue
        shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()


def _variant_columns(variant: object) -> dict[str, object]:
    """The expanded ``variant`` coord: ``variant_id`` + family domain params."""
    cols = reg.variant_key_columns(variant.family)  # type: ignore[attr-defined]
    values: dict[str, object] = {"variant_id": variant.name}  # type: ignore[attr-defined]
    params = variant.params  # type: ignore[attr-defined]
    for col in cols:
        if col != "variant_id":
            values[col] = params.get(col)
    return values
