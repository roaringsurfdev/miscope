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

import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING

import pandas as pd

import miscope.registry as reg
from miscope.analysis import signature as sig_mod
from miscope.analysis.artifact_loader import read_signature_manifest
from miscope.analysis.output_schema import Coord, FieldKind, OutputField
from miscope.analysis.registry import AnalyzerRegistry
from miscope.warehouse import catalog as catalog_mod
from miscope.warehouse import mapping, mapping_semantic, paths, schema
from miscope.warehouse.decompose import KeyMatch, assign_keys, get_decomp
from miscope.warehouse.flatten import flatten_field
from miscope.warehouse.losses import LOSSES_TABLE, losses_source_signature
from miscope.warehouse.signatures import read_table_signatures, write_table_signatures

# Bump when the columnar mapping/flattening logic changes in a way that alters
# output bytes for unchanged artifacts — folds into every table's source signature
# so a materializer change invalidates the warehouse without an artifact change.
MATERIALIZER_VERSION = 1

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from miscope.analysis.spec import AnalyzerSpec
    from miscope.families.variant import Variant
    from miscope.warehouse.decompose import AnalyzerDecomp

logger = logging.getLogger(__name__)


@dataclass
class MaterializeReport:
    """Summary of one variant's columnar materialization."""

    variant_id: str
    files_written: list[str] = dc_field(default_factory=list)
    tables: dict[str, int] = dc_field(default_factory=dict)  # table -> row count
    skipped_analyzers: list[str] = dc_field(default_factory=list)  # no data / empty frames
    failed_analyzers: dict[str, str] = dc_field(default_factory=dict)  # name -> error summary


# A field's long frame before variant columns / discriminators are attached.
@dataclass
class _FieldFrame:
    field: OutputField
    frame: pd.DataFrame


def materialize_variant_columnar(
    variant: Variant, run_set: str = paths.DEFAULT_RUN_SET, force: bool = False
) -> MaterializeReport:
    """Materialize a variant's columnar analyzer outputs to Parquet (REQ_110A/REQ_145).

    Surgical by default: each table's **source signature** folds the materializer
    version and its feeder analyzers' signatures; a table is rebuilt only when its
    source signature changed (or its Parquet is absent), and only the analyzers
    feeding a stale table are read. ``force=True`` ignores signatures and rebuilds
    every table — the explicit "rebuild regardless" override that replaces the old
    blanket wipe-and-rebuild.

    Every long-format row and catalog row carries a ``run_set`` coordinate column
    (REQ_138) so the query surface gains the parameterization dimension once.
    """
    report = MaterializeReport(variant_id=variant.name)
    specs = _scoped_specs(variant)
    columnar_specs = [s for s in specs if any(f.kind is FieldKind.COLUMNAR for f in s.outputs)]

    analyzer_sigs = {s.name: _analyzer_signature(variant, s.name) for s in columnar_specs}
    feeders = _table_feeders(columnar_specs)
    new_sigs = {t: _table_source_sig(analyzer_sigs, fs) for t, fs in feeders.items()}
    # The losses table (REQ_144) is a warehouse-level co-emission, not a columnar
    # analyzer table — its source signature is the loss series' content (REQ_145).
    losses_sig = losses_source_signature(variant)
    if losses_sig:
        new_sigs[LOSSES_TABLE] = losses_sig
    stored_sigs = read_table_signatures(variant)
    stale = _stale_tables(variant, new_sigs, stored_sigs, force)
    _wipe_stale_and_removed(variant, stale, set(new_sigs))

    needed = {a for t in stale for a in feeders[t]}
    variant_cols = _variant_columns(variant, run_set)
    semantic: dict[str, _SemanticAcc] = defaultdict(_SemanticAcc)

    for spec in columnar_specs:
        if spec.name not in needed:
            continue
        # Per-analyzer isolation (REQ_140): one malformed on-disk artifact is
        # recorded and skipped, never fatal to the whole columnar pass.
        try:
            field_frames = _build_field_frames(variant, spec)
            if not field_frames:
                report.skipped_analyzers.append(spec.name)
                continue
            claimed = _collect_semantic(spec.name, field_frames, semantic)
            generic = [ff for ff in field_frames if ff.field.name not in claimed]
            _emit_generic(variant, spec.name, generic, variant_cols, report, stale)
        except Exception as exc:  # noqa: BLE001 — quarantine one bad artifact
            report.failed_analyzers[spec.name] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "warehouse materialize: analyzer %r failed on variant %s, skipping (%s: %s)",
                spec.name,
                variant.name,
                type(exc).__name__,
                exc,
            )

    _emit_semantic(variant, semantic, variant_cols, report, stale)
    _emit_losses(variant, run_set, report, stale)
    write_table_signatures(variant, new_sigs)
    return report


# ---------------------------------------------------------------------------
# Surgical re-materialize: per-table source signatures (REQ_145)
# ---------------------------------------------------------------------------


def _analyzer_signature(variant: Variant, name: str) -> str:
    """Order-independent digest of an analyzer's stamped per-epoch/cross signatures.

    Reads the default-plane signature manifest (the plane the columnar warehouse
    materializes). An unstamped (legacy) or absent manifest yields ``""`` — so the
    table reads stale until the artifacts are signature-stamped (the one-time rebuild).
    """
    raw = read_signature_manifest(variant.artifacts.artifacts_dir, name, "")
    sigs = [v["sig"] for v in raw.values() if isinstance(v, dict) and "sig" in v]
    return sig_mod.digest(sigs) if sigs else ""


def _table_feeders(columnar_specs: list[AnalyzerSpec]) -> dict[str, set[str]]:
    """Map each materialized table to the analyzers that feed it.

    A semantic table is fed by every analyzer whose claim targets it; the generic
    fallback table (named for the analyzer) is fed by that analyzer iff it has a
    columnar field not claimed by any semantic table.
    """
    feeders: dict[str, set[str]] = defaultdict(set)
    for spec in columnar_specs:
        columnar_fields = {f.name for f in spec.outputs if f.kind is FieldKind.COLUMNAR}
        claimed: set[str] = set()
        for claim in mapping_semantic.claims_for(spec.name):
            feeders[claim.table].add(spec.name)
            claimed |= set(claim.fields)
        if columnar_fields - claimed:
            feeders[spec.name].add(spec.name)
    return feeders


def _table_source_sig(analyzer_sigs: dict[str, str], feeder_set: set[str]) -> str:
    """A table's source signature: materializer version + its feeders' signatures."""
    components = [f"mat={MATERIALIZER_VERSION}"]
    components.extend(sorted(analyzer_sigs.get(a, "") for a in feeder_set))
    return sig_mod.compute_signature(components)


def _table_materialized(variant: Variant, table: str) -> bool:
    """Whether a table has at least one Parquet on disk."""
    tdir = paths.table_dir(variant, table)
    return tdir.is_dir() and any(tdir.glob("*.parquet"))


def _stale_tables(
    variant: Variant, new_sigs: dict[str, str], stored_sigs: dict[str, str], force: bool
) -> set[str]:
    """Tables to rebuild: signature changed, or Parquet missing (all under ``force``)."""
    if force:
        return set(new_sigs)
    return {
        table
        for table, sig in new_sigs.items()
        if stored_sigs.get(table) != sig or not _table_materialized(variant, table)
    }


def _wipe_table(variant: Variant, table: str) -> None:
    """Remove one table's Parquet dir and its co-emitted columnar catalog rows."""
    tdir = paths.table_dir(variant, table)
    if tdir.exists():
        shutil.rmtree(tdir, ignore_errors=True)
    catalog_path = paths.catalog_parquet_path(variant, table)
    if catalog_path.exists():
        catalog_path.unlink()


def _wipe_stale_and_removed(variant: Variant, stale: set[str], current_tables: set[str]) -> None:
    """Clear stale tables and any on-disk table no longer produced (feeders gone)."""
    for table in stale:
        _wipe_table(variant, table)
    wdir = paths.warehouse_dir(variant)
    if not wdir.exists():
        return
    reserved = {
        paths.CATALOG_DIRNAME,
        paths.TENSOR_CATALOG_DIRNAME,
        paths.RUN_SETS_DIRNAME,
    }
    # ``variant_outcomes`` is now a derived table (REQ_144) owned by the surgical
    # derived pass; protect it from the columnar pass's removed-table sweep exactly
    # as the old co-emitted outcomes table was protected.
    keep = current_tables | {"variant_outcomes"}
    for child in wdir.iterdir():
        if child.is_dir() and child.name not in reserved and child.name not in keep:
            _wipe_table(variant, child.name)


def _scoped_specs(variant: Variant) -> list[AnalyzerSpec]:
    """Analyzer specs in scope for materialization — the family's declared set.

    REQ_140: ``family.json`` (via ``AnalyzerRegistry.list_for_family``) is the
    single source of truth for a family's analyzer scope — the same source the
    run plan uses. Iterating the global registry instead pulled in
    registered-but-undeclared analyzers (e.g. deprecated ``gradient_site``),
    whose leftover artifacts aborted the pass. Sorted by name so output ordering
    matches the prior registry iteration (byte-identical on baselines).
    """
    specs = AnalyzerRegistry.list_for_family(variant.family)
    return sorted(specs, key=lambda s: s.name)


def _emit_losses(
    variant: Variant, run_set: str, report: MaterializeReport, stale: set[str]
) -> None:
    """Co-emit the dense per-epoch losses table (REQ_144), if stale this pass.

    A warehouse-level co-emission sourced from checkpoint metadata, not an analyzer
    output — written here so it lives in the same per-variant warehouse and shares
    the signature manifest. Surgical (REQ_145): rebuilt only when its loss-series
    source signature changed, never on every pass like the 1-row outcomes table.
    """
    from miscope.warehouse.losses import materialize_variant_losses

    if LOSSES_TABLE not in stale:
        return
    rows = materialize_variant_losses(variant, run_set)
    if rows:
        report.tables[LOSSES_TABLE] = rows
        report.files_written.append(str(paths.semantic_parquet_path(variant, LOSSES_TABLE)))


# ---------------------------------------------------------------------------
# Loading + flattening into per-field long frames
# ---------------------------------------------------------------------------


def _build_field_frames(variant: Variant, spec: AnalyzerSpec) -> list[_FieldFrame]:
    decomp = get_decomp(spec.name)
    if spec.output_scope == "per_epoch":
        return _per_epoch_frames(variant, spec, decomp)
    return _cross_epoch_frames(variant, spec, decomp)


def _per_epoch_frames(
    variant: Variant, spec: AnalyzerSpec, decomp: AnalyzerDecomp
) -> list[_FieldFrame]:
    loader = variant.artifacts
    epochs = loader.get_epochs(spec.name)
    if not epochs:
        return []
    parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    fields: dict[str, OutputField] = {}
    loose: set[str] = set()
    for epoch in epochs:
        npz = loader.load_epoch(spec.name, epoch)
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


def _cross_epoch_frames(
    variant: Variant, spec: AnalyzerSpec, decomp: AnalyzerDecomp
) -> list[_FieldFrame]:
    loader = variant.artifacts
    try:
        npz = loader.load_cross_epoch(spec.name)
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


def _columnar_matches(spec: AnalyzerSpec, npz: dict) -> list[KeyMatch]:
    matches = assign_keys(spec.name, spec.outputs, tuple(npz.keys()))
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


def _resolve_labels(
    decomp: AnalyzerDecomp, npz: dict, *, epoch_axis: bool
) -> dict[Coord, np.ndarray]:
    labels: dict[Coord, np.ndarray] = {}
    for coord, field_name in decomp.axis_labels.items():
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
    variant: Variant,
    analyzer_name: str,
    field_frames: list[_FieldFrame],
    variant_cols: dict[str, object],
    report: MaterializeReport,
    stale: set[str],
) -> None:
    """Write the unclaimed fields as generic tables: one Parquet per coord signature.

    Skipped when the analyzer's generic table is signature-fresh (REQ_145) — it is
    only built here because the analyzer also feeds a *stale* table.
    """
    if analyzer_name not in stale:
        return
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
    variant: Variant,
    semantic: dict[str, _SemanticAcc],
    variant_cols: dict[str, object],
    report: MaterializeReport,
    stale: set[str],
) -> None:
    """Row-union each stale semantic table's feeder frames and write one Parquet per table.

    Only stale tables are emitted (REQ_145); a fresh semantic table whose feeders
    were read incidentally (because they also feed a stale table) is left untouched.
    """
    for table, acc in semantic.items():
        if table not in stale:
            continue
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
    variant: Variant,
    table: str,
    path: Path,
    coords_str: str,
    sig_token: str,
    df: pd.DataFrame,
    value_columns: tuple[str, ...],
    report: MaterializeReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        variant.name,
    )


def _variant_columns(variant: Variant, run_set: str) -> dict[str, object]:
    """Leading key columns: ``variant_id`` + family domain params + ``run_set``.

    ``run_set`` (REQ_138) is the parameterization coordinate every row is keyed by;
    it leads alongside the variant identity so cross-parameterization comparison is
    ``WHERE run_set = ...`` / ``GROUP BY run_set``.
    """
    cols = reg.variant_key_columns(variant.family)  # type: ignore[attr-defined]
    values: dict[str, object] = {"variant_id": variant.name}  # type: ignore[attr-defined]
    params = variant.params  # type: ignore[attr-defined]
    for col in cols:
        if col != "variant_id":
            values[col] = params.get(col)
    values["run_set"] = run_set
    return values
