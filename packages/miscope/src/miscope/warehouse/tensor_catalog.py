"""Tensor descriptor catalog + resolver (REQ_110B — the non-columnar half).

``.npz`` tensors (raw weights, activations, bases) stay as blobs — the format is
**unchanged** — but they stop being *opaque*. Each ``tensor``-kind field
(declared per REQ_107) is indexed here as a coordinate-keyed *descriptor*: an
address (``TensorRef``) plus the instance coordinates it is keyed by. You then
**select** tensors by filtering/joining the descriptor relation in SQL (touching
zero array bytes), and a :class:`TensorResolver` materializes only the selected
blobs — batched per container so each archive opens exactly once.

Three load-bearing properties (REQ_110B CoS):

- **Address-only.** A :class:`TensorRef` carries ``uri``/``member``/``dtype``/
  ``shape``/``codec`` — never payload. The catalog is an index, not a store.
- **Co-emission.** Descriptor rows are written in the same pass that reads each
  blob's header (:func:`materialize_variant_tensors`), driven by the registry's
  declared ``tensor`` fields — never a later drift-prone scan. The descriptor's
  ``shape`` is read from the blob header (REQ_107 declares ``dtype`` but not the
  concrete shape); the declared ``dtype`` is cross-checked at index time.
- **Reproducibility guard.** The catalog *asserts* shape/dtype; the resolver
  *checks* the bytes match on load and fails loud on mismatch. That is what lets
  the index be trusted without being authoritative — the blob bytes stay so.

Grain: one descriptor per blob *member*. The descriptor's coordinate columns are
its **instance** keys — ``epoch`` (for per-epoch analyzers) and the ``site`` /
``group`` prefix coords carried in the npz key (parsed via :mod:`.decompose`).
Array-axis coords (``neuron``/``row_id``/``frequency``/``head``, and ``epoch``
for a cross-epoch stack) live *inside* the array and are recorded in ``shape``,
not as columns.
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import IO, Any, cast

import numpy as np
import numpy.lib.format as npformat
import pandas as pd

import miscope.registry as reg
from miscope.analysis.output_schema import FieldKind, OutputField
from miscope.warehouse import paths
from miscope.warehouse.decompose import KeyMatch, assign_keys

# npy header version -> the matching header reader (numpy 2.x dropped the private
# ``_read_array_header``; the versioned public readers are stable).
_HEADER_READERS = {
    (1, 0): npformat.read_array_header_1_0,
    (2, 0): npformat.read_array_header_2_0,
}


@dataclass(frozen=True)
class TensorRef:
    """Everything a resolver needs to dereference one dense array — address + type, no data.

    ``uri`` is relative to the variant directory (portable across tree moves; the
    warehouse is regeneratable). ``codec`` records how to open the container:

      ``npz``            -- ``np.savez`` archive; members read one at a time
      ``npz_compressed`` -- ``np.savez_compressed``; selected member decompressed
      ``npy``            -- single ``.npy`` array; memory-mappable, no full read
    """

    uri: str
    member: str  # key within the container; "" for a single-array .npy
    dtype: str  # declared here, verified against the bytes on load
    shape: tuple[int, ...]
    codec: str = "npz_compressed"


@dataclass(frozen=True)
class TensorCatalogRow:
    """One blob member's descriptor — an instance-keyed address, not a payload.

    Coordinate columns (``epoch``/``site``/``group``) are the descriptor's
    instance keys, always queryable without touching a payload. ``kind`` is
    always ``"tensor"`` (the discriminator shared with the columnar catalog).
    """

    id: str  # stable surrogate: {variant_id}/{analyzer}/{field}[/epoch=..][/site=..]
    kind: str
    variant_id: str
    analyzer: str
    field: str
    coords: str  # comma-joined declared coord names of the field (introspection)
    ref: TensorRef
    epoch: int | None = None
    site: str | None = None
    group: str | None = None
    group_type: str | None = None

    def to_flat(self) -> dict[str, object]:
        """Flatten to the columns DuckDB / pyarrow store (TensorRef inlined)."""
        return {
            "id": self.id,
            "kind": self.kind,
            "variant_id": self.variant_id,
            "analyzer": self.analyzer,
            "field": self.field,
            "coords": self.coords,
            "epoch": self.epoch,
            "site": self.site,
            "group": self.group,
            "group_type": self.group_type,
            "tensor_uri": self.ref.uri,
            "tensor_member": self.ref.member,
            "tensor_dtype": self.ref.dtype,
            "tensor_shape": list(self.ref.shape),
            "tensor_codec": self.ref.codec,
        }

    @classmethod
    def from_flat(cls, r: dict[str, Any]) -> TensorCatalogRow:
        """Rebuild a row from a flat record (the inverse of :meth:`to_flat`)."""
        ref = TensorRef(
            uri=r["tensor_uri"],
            member=r["tensor_member"],
            dtype=r["tensor_dtype"],
            shape=tuple(int(d) for d in r["tensor_shape"]),
            codec=r["tensor_codec"],
        )
        return cls(
            id=r["id"],
            kind=r["kind"],
            variant_id=r["variant_id"],
            analyzer=r["analyzer"],
            field=r["field"],
            coords=r["coords"],
            ref=ref,
            epoch=_opt_int(r.get("epoch")),
            site=_opt_str(r.get("site")),
            group=_opt_str(r.get("group")),
            group_type=_opt_str(r.get("group_type")),
        )


# ---------------------------------------------------------------------------
# Co-emission: build the descriptor relation from declared tensor fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DtypeDrift:
    """A tensor field whose blob dtype differs from its REQ_107 declaration."""

    analyzer: str
    field: str
    declared: str
    actual: str


@dataclass
class TensorCatalogReport:
    """Summary of one variant's tensor-catalog materialization."""

    variant_id: str
    files_written: list[str] = dc_field(default_factory=list)
    n_descriptors: int = 0
    dtype_drift: list[DtypeDrift] = dc_field(default_factory=list)


def materialize_variant_tensors(variant: object) -> TensorCatalogReport:
    """Index every declared ``tensor`` field's blobs as descriptor rows.

    Driven by the registry (REQ_107): for each analyzer declaring a ``tensor``
    output, locate its blobs through the sanctioned ``variant.artifacts``
    accessor, read each member's header (shape/dtype, no array bytes), and
    co-emit one descriptor Parquet per analyzer. Deterministic: the prior tensor
    catalog is wiped first.

    The descriptor records the **actual** on-disk shape/dtype — the blob bytes
    are authoritative, so the catalog faithfully addresses them and ``resolve``
    works. Where the bytes' dtype disagrees with the analyzer's REQ_107
    declaration, the mismatch is surfaced as a :class:`DtypeDrift` finding on the
    report (a declaration to reconcile), never silently dropped and never fatal.
    """
    report = TensorCatalogReport(variant_id=variant.name)  # type: ignore[attr-defined]
    tdir = paths.tensor_catalog_dir(variant)  # type: ignore[arg-type]
    shutil.rmtree(tdir, ignore_errors=True)
    variant_cols = _variant_columns(variant)
    for spec in reg.index().analyzers:
        if not any(f.kind is FieldKind.TENSOR for f in spec.outputs):
            continue
        rows = _descriptor_rows(variant, spec, variant_cols, report)
        if rows:
            report.files_written.append(_write_descriptors(variant, spec.name, rows, variant_cols))
            report.n_descriptors += len(rows)
    return report


def _descriptor_rows(
    variant: object, spec: object, variant_cols: dict[str, object], report: TensorCatalogReport
) -> list[TensorCatalogRow]:
    """Descriptor rows for one analyzer's tensor blobs (per-epoch or cross-epoch)."""
    loader = variant.artifacts  # type: ignore[attr-defined]
    if spec.output_scope == "per_epoch":  # type: ignore[attr-defined]
        rows: list[TensorCatalogRow] = []
        for epoch in loader.get_epochs(spec.name):  # type: ignore[attr-defined]
            path = Path(loader.artifact_path(spec.name, epoch))  # type: ignore[attr-defined]
            rows += _rows_for_container(variant, spec, variant_cols, path, epoch, report)
        return rows
    path = Path(loader.artifact_path(spec.name))  # type: ignore[attr-defined]
    if not path.exists():
        return []
    return _rows_for_container(variant, spec, variant_cols, path, None, report)


def _rows_for_container(
    variant: object,
    spec: object,
    variant_cols: dict[str, object],
    path: Path,
    epoch: int | None,
    report: TensorCatalogReport,
) -> list[TensorCatalogRow]:
    """Descriptors for the tensor members of one ``.npz`` container."""
    members = _inspect_npz(path)
    matches = assign_keys(spec.name, spec.outputs, tuple(members))  # type: ignore[attr-defined]
    rel_uri = path.relative_to(variant.variant_dir).as_posix()  # type: ignore[attr-defined]
    rows = []
    for m in matches:
        if m.field.kind is not FieldKind.TENSOR:
            continue
        meta = members[m.npz_key]
        _record_dtype_drift(spec.name, m.field, meta.dtype, report)  # type: ignore[attr-defined]
        ref = TensorRef(rel_uri, m.npz_key, meta.dtype, meta.shape, meta.codec)
        rows.append(_build_row(variant_cols, spec.name, m, ref, epoch))  # type: ignore[attr-defined]
    return rows


def _build_row(
    variant_cols: dict[str, object],
    analyzer: str,
    match: KeyMatch,
    ref: TensorRef,
    epoch: int | None,
) -> TensorCatalogRow:
    """Assemble one descriptor row from a key match + its blob address."""
    site = match.prefix_coords.get("site")
    group = match.prefix_coords.get("group")
    return TensorCatalogRow(
        id=_row_id(variant_cols["variant_id"], analyzer, match.field.name, epoch, site, group),
        kind="tensor",
        variant_id=variant_cols["variant_id"],  # type: ignore[arg-type]
        analyzer=analyzer,
        field=match.field.name,
        coords=", ".join(match.field.coord_names),
        ref=ref,
        epoch=epoch,
        site=site,
        group=group,
    )


def _row_id(
    variant_id: object,
    analyzer: str,
    field: str,
    epoch: int | None,
    site: str | None,
    group: str | None,
) -> str:
    """Deterministic surrogate key — survives recompute, stable across runs."""
    parts = [str(variant_id), analyzer, field]
    if epoch is not None:
        parts.append(f"epoch={epoch:05d}")
    if site is not None:
        parts.append(f"site={site}")
    if group is not None:
        parts.append(f"group={group}")
    return "/".join(parts)


def _write_descriptors(
    variant: object, analyzer: str, rows: list[TensorCatalogRow], variant_cols: dict[str, object]
) -> str:
    """Write one analyzer's descriptor rows to its tensor-catalog Parquet.

    The family's domain-parameter columns (beyond ``variant_id``, which the row
    already carries) are attached here so in-family filters (``WHERE prime > 100``)
    work directly against the descriptor relation.
    """
    path = paths.tensor_catalog_parquet_path(variant, analyzer)  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_flat() for r in rows])
    params = {col: val for col, val in variant_cols.items() if col != "variant_id"}
    for col, val in reversed(list(params.items())):  # keep declared param order
        # variant_id is unique, so get_loc returns an int position.
        variant_id_pos = cast(int, df.columns.get_loc("variant_id"))
        df.insert(variant_id_pos + 1, col, val)
    _stabilize_coord_dtypes(df)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return str(path)


# Nullable coordinate columns whose Parquet type must be pinned even when a given
# analyzer leaves them all-null — otherwise pyarrow infers a ``null`` type and a
# cross-file DuckDB union (110-C) fails to reconcile it against a typed sibling.
_NULLABLE_STR_COORDS = ("site", "group", "group_type")


def _stabilize_coord_dtypes(df: pd.DataFrame) -> None:
    """Pin nullable coord columns to stable types so cross-file unions reconcile."""
    df["epoch"] = df["epoch"].astype("Int64")
    for col in _NULLABLE_STR_COORDS:
        df[col] = df[col].astype("string")


def _variant_columns(variant: object) -> dict[str, object]:
    """The expanded ``variant`` coord: ``variant_id`` + family domain params."""
    cols = reg.variant_key_columns(variant.family)  # type: ignore[attr-defined]
    values: dict[str, object] = {"variant_id": variant.name}  # type: ignore[attr-defined]
    params = variant.params  # type: ignore[attr-defined]
    for col in cols:
        if col != "variant_id":
            values[col] = params.get(col)
    return values


# ---------------------------------------------------------------------------
# Reading the descriptor relation (selection is SQL/filter over these rows)
# ---------------------------------------------------------------------------


def read_tensor_catalog(variant: object) -> pd.DataFrame:
    """The variant's tensor descriptor relation as one DataFrame (no payload).

    This is the queryable surface: filter/join it (in pandas or via DuckDB over
    :func:`paths.tensor_catalog_dir`) to *select* tensors. No array byte is
    touched to answer a metadata question. Empty frame if nothing materialized.
    """
    tdir = paths.tensor_catalog_dir(variant)  # type: ignore[arg-type]
    if not tdir.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(p, engine="pyarrow") for p in sorted(tdir.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rows_from_frame(df: pd.DataFrame) -> list[TensorCatalogRow]:
    """Rebuild descriptor rows from a (possibly SQL-filtered) descriptor frame."""
    return [TensorCatalogRow.from_flat(rec) for rec in df.to_dict("records")]


class TensorCatalogAccessor:
    """Variant-bound surface over the tensor descriptor catalog (REQ_110B).

    Reached via ``variant.tensor_catalog`` — mirrors ``variant.warehouse``.
    ``materialize()`` indexes the blobs; ``descriptors()`` returns the queryable
    relation (no payload); ``resolve(rows)`` materializes selected descriptors,
    binding the variant directory as the resolution root.
    """

    def __init__(self, variant: object) -> None:
        self._variant = variant

    def materialize(self) -> TensorCatalogReport:
        """(Re)index this variant's tensor blobs as descriptor rows."""
        return materialize_variant_tensors(self._variant)

    def descriptors(self) -> pd.DataFrame:
        """The descriptor relation as a DataFrame — select over it, touching no payload."""
        return read_tensor_catalog(self._variant)

    def resolve(self, rows: pd.DataFrame | list[TensorCatalogRow]) -> dict[str, np.ndarray]:
        """Materialize selected descriptors to ``{id -> ndarray}`` (bytes verified)."""
        if isinstance(rows, pd.DataFrame):
            rows = rows_from_frame(rows)
        root = self._variant.variant_dir  # type: ignore[attr-defined]
        return TensorResolver(root).resolve(rows)


# ---------------------------------------------------------------------------
# Resolution: materialize only the selected descriptors
# ---------------------------------------------------------------------------


class TensorResolver:
    """Turns selected descriptor rows into materialized ndarrays.

    Materializes only the rows handed to it, batched per container so each
    archive opens exactly once (``NpzFile`` members read lazily; ``.npy``
    memory-mapped). ``root`` resolves a descriptor's variant-relative ``uri`` to
    a file (the variant directory for the internal warehouse).
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else None

    def resolve(self, rows: list[TensorCatalogRow]) -> dict[str, np.ndarray]:
        """Materialize ``{id -> ndarray}`` for the given descriptor rows."""
        by_uri: dict[str, list[TensorCatalogRow]] = {}
        for r in rows:
            by_uri.setdefault(r.ref.uri, []).append(r)
        out: dict[str, np.ndarray] = {}
        for uri, group in by_uri.items():
            out.update(self._read_container(uri, group))
        return out

    def _read_container(self, uri: str, group: list[TensorCatalogRow]) -> dict[str, np.ndarray]:
        path = self._resolve_uri(uri)
        codec = group[0].ref.codec
        if codec == "npy":
            arr = np.load(path, mmap_mode="r")  # memory-mapped; no full read
            return {group[0].id: _verify(arr, group[0].ref)}
        with np.load(path) as archive:  # NpzFile: members read lazily, one at a time
            return {r.id: _verify(archive[r.ref.member], r.ref) for r in group}

    def _resolve_uri(self, uri: str) -> Path:
        p = Path(uri)
        if p.is_absolute() or self._root is None:
            return p
        return self._root / p


def _verify(arr: np.ndarray, ref: TensorRef) -> np.ndarray:
    """Reproducibility guard: fail loud if the bytes disagree with the descriptor."""
    if tuple(arr.shape) != tuple(ref.shape):
        raise ValueError(
            f"shape mismatch for {ref.uri}::{ref.member}: "
            f"descriptor {tuple(ref.shape)} vs bytes {tuple(arr.shape)}"
        )
    if arr.dtype != np.dtype(ref.dtype):
        raise ValueError(
            f"dtype mismatch for {ref.uri}::{ref.member}: "
            f"descriptor {ref.dtype} vs bytes {arr.dtype}"
        )
    return arr


# ---------------------------------------------------------------------------
# npz header inspection (shape/dtype/codec without touching array bytes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MemberMeta:
    shape: tuple[int, ...]
    dtype: str
    codec: str


def _inspect_npz(path: Path) -> dict[str, _MemberMeta]:
    """Map each member name -> (shape, dtype, codec), reading only npy headers.

    One zip open per container; per member, only the ``.npy`` header bytes are
    read (the array payload is never decompressed). ``codec`` is read from the
    member's zip compression type, so it is honest about how the blob was stored.
    """
    out: dict[str, _MemberMeta] = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            member = info.filename[: -len(".npy")]
            codec = "npz_compressed" if info.compress_type != zipfile.ZIP_STORED else "npz"
            with zf.open(info) as fp:
                shape, dtype = _read_header(fp)
            out[member] = _MemberMeta(shape, dtype, codec)
    return out


def _read_header(fp: IO[bytes]) -> tuple[tuple[int, ...], str]:
    """Read an npy stream's header -> (shape, dtype string). Header bytes only."""
    version = npformat.read_magic(fp)
    reader = _HEADER_READERS.get(version)
    if reader is None:
        raise ValueError(f"unsupported npy header version {version}")
    shape, _fortran, dtype = reader(fp)
    return tuple(shape), str(dtype)


def _record_dtype_drift(
    analyzer: str, field: OutputField, blob_dtype: str, report: TensorCatalogReport
) -> None:
    """Note a declared-vs-actual dtype mismatch as a finding (non-fatal).

    The bytes are authoritative, so the descriptor takes the actual dtype; this
    records the divergence from the REQ_107 declaration for the report to surface
    (a declaration to reconcile), rather than failing the materialize.
    """
    if np.dtype(field.dtype) == np.dtype(blob_dtype):
        return
    drift = DtypeDrift(analyzer, field.name, field.dtype, blob_dtype)
    if drift not in report.dtype_drift:  # one finding per (analyzer, field), not per row
        report.dtype_drift.append(drift)


def _opt_int(v: Any) -> int | None:
    return None if v is None or pd.isna(v) else int(v)


def _opt_str(v: Any) -> str | None:
    return None if v is None or pd.isna(v) else str(v)
