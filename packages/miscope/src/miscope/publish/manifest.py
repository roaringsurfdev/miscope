"""Bundle manifest — the citable, schema-bearing record of a published bundle (REQ_110E).

A published bundle is a curated, frozen set of flat Parquet files attached to a
``data-*`` GitHub Release. The ``manifest.json`` that travels with it is the
bundle's identity: ``bundle_version``, ``mint_date``, ``miscope_version``, the
referencing article(s), a description, and — per table — the file name, row
count, content hash, column schema, and which reserved provenance columns were
populated at mint time.

The manifest is the unit the schema-stability gate reads (a published manifest is
the baseline a rebuild is compared against) and the unit a citation points at
(URL + manifest + content hash). It is small JSON, so a manifest *history* is
committed to the repo even though the Parquet it describes never is — that
committed history is what gives the gate a baseline without the data in CI.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import pyarrow.parquet as pq

from miscope.warehouse.schema import RESERVED_PROVENANCE

MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class TableManifest:
    """One published table's frozen record within a bundle manifest."""

    name: str
    file: str  # bundle-relative Parquet filename
    rows: int
    content_hash: str  # "sha256:<hex>"
    schema: dict[str, str]  # column name -> Arrow type string (e.g. "int64", "string")
    populated_provenance: tuple[str, ...]  # reserved provenance cols carrying any non-null value

    def to_dict(self) -> dict:
        return {**asdict(self), "populated_provenance": list(self.populated_provenance)}

    @classmethod
    def from_dict(cls, data: dict) -> TableManifest:
        return cls(
            name=data["name"],
            file=data["file"],
            rows=int(data["rows"]),
            content_hash=data["content_hash"],
            schema=dict(data["schema"]),
            populated_provenance=tuple(data.get("populated_provenance", ())),
        )


@dataclass(frozen=True)
class BundleManifest:
    """The full manifest for one published bundle version (citable + gate baseline)."""

    bundle_name: str
    bundle_version: str
    mint_date: str  # ISO 8601, UTC
    miscope_version: str
    description: str
    articles: tuple[str, ...]
    tables: tuple[TableManifest, ...]

    @property
    def tag(self) -> str:
        """The ``data-*`` Release tag this manifest publishes under."""
        return f"data-{self.bundle_name}-{self.bundle_version}"

    def table(self, name: str) -> TableManifest | None:
        """The table record by name (``None`` if the bundle does not include it)."""
        return next((t for t in self.tables if t.name == name), None)

    def to_json(self, *, indent: int = 2) -> str:
        payload = {
            "bundle_name": self.bundle_name,
            "bundle_version": self.bundle_version,
            "mint_date": self.mint_date,
            "miscope_version": self.miscope_version,
            "description": self.description,
            "articles": list(self.articles),
            "tables": [t.to_dict() for t in self.tables],
        }
        return json.dumps(payload, indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> BundleManifest:
        data = json.loads(text)
        return cls(
            bundle_name=data["bundle_name"],
            bundle_version=data["bundle_version"],
            mint_date=data["mint_date"],
            miscope_version=data["miscope_version"],
            description=data.get("description", ""),
            articles=tuple(data.get("articles", ())),
            tables=tuple(TableManifest.from_dict(t) for t in data["tables"]),
        )

    def write(self, path: Path) -> None:
        """Write the manifest JSON to ``path`` (the bundle's ``manifest.json``)."""
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> BundleManifest:
        """Read a manifest from a JSON file (a committed baseline or a built bundle)."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Capture helpers — derive a TableManifest from a written Parquet file
# ---------------------------------------------------------------------------


def hash_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file as ``sha256:<hex>`` (the bundle's citable fingerprint)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def capture_schema(path: Path) -> dict[str, str]:
    """Column name -> Arrow type string for a Parquet file (the gate's comparison unit)."""
    arrow_schema = pq.read_schema(Path(path))
    return {field.name: str(field.type) for field in arrow_schema}


def populated_provenance(path: Path) -> tuple[str, ...]:
    """Which reserved provenance columns carry any non-null value (manifest record).

    Uses Parquet row-group statistics when present (no full scan); falls back to
    reading just the candidate columns when a writer omitted null counts. In v1.0
    these columns are reserved-but-empty, so this is typically ``()`` — the
    manifest records that fact so a later bundle can show provenance landing.
    """
    parquet = pq.ParquetFile(Path(path))
    present = [c for c in RESERVED_PROVENANCE if c in parquet.schema_arrow.names]
    if not present:
        return ()
    return tuple(c for c in present if _column_has_value(parquet, c))


def _column_has_value(parquet: pq.ParquetFile, column: str) -> bool:
    """Whether ``column`` holds at least one non-null value (stats-first, scan fallback)."""
    total = parquet.metadata.num_rows
    if total == 0:
        return False
    null_count = _null_count_from_stats(parquet, column)
    if null_count is not None:
        return null_count < total
    table = parquet.read(columns=[column])
    return table.column(0).null_count < total


def _null_count_from_stats(parquet: pq.ParquetFile, column: str) -> int | None:
    """Summed null count across row groups, or ``None`` if any group lacks statistics."""
    col_index = parquet.schema_arrow.names.index(column)
    total_nulls = 0
    for group in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(group).column(col_index).statistics
        if stats is None or not stats.has_null_count:
            return None
        total_nulls += stats.null_count
    return total_nulls


def miscope_version() -> str:
    """Installed ``miscope`` version for the manifest (``0+unknown`` if unresolved)."""
    try:
        return _pkg_version("miscope")
    except PackageNotFoundError:
        return "0+unknown"


def utc_now_iso() -> str:
    """Current UTC timestamp as an ISO 8601 string (the manifest ``mint_date``)."""
    return datetime.now(UTC).isoformat(timespec="seconds")
