"""Bundle build — freeze curated warehouse tables into a publishable bundle (REQ_110E).

``build_bundle(spec, source, out_dir)`` is the core: for each table in the spec it
runs the (subsetting) SQL over a :class:`~miscope.query.QueryConnection`, writes a
flat zstd Parquet (the published shape — one file per table, queryable by URL),
captures its schema + content hash + row count into a :class:`TableManifest`,
optionally runs the schema-stability gate against a baseline manifest, and writes
``manifest.json``. ``build_bundle_for_family`` is the convenience that opens the
family warehouse as the source.

The internal-vs-published distinction is structural here: the source is the churn-
free internal warehouse; the output is a frozen, hashed, manifest-bearing directory
destined for a ``data-*`` Release. Nothing about a build mutates the warehouse, and
the build never reaches past the query surface for data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from miscope.publish import manifest as manifest_mod
from miscope.publish.manifest import BundleManifest, TableManifest
from miscope.publish.schema_gate import GateReport, enforce
from miscope.publish.spec import BundleSpec

if TYPE_CHECKING:
    from miscope.config import AppConfig
    from miscope.query import QueryConnection

# Published bundles favour file size over write speed (REQ_110: zstd for bundles).
BUNDLE_COMPRESSION = "zstd"


@dataclass(frozen=True)
class BuildResult:
    """The outcome of a bundle build: the manifest, its directory, and the gate verdict."""

    manifest: BundleManifest
    out_dir: Path
    gate: GateReport | None  # None when no baseline was supplied

    @property
    def parquet_paths(self) -> list[Path]:
        """The written Parquet files (the Release assets, alongside the manifest)."""
        return [self.out_dir / t.file for t in self.manifest.tables]

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / manifest_mod.MANIFEST_FILENAME


def build_bundle(
    spec: BundleSpec,
    source: QueryConnection,
    out_dir: Path,
    *,
    baseline: BundleManifest | None = None,
) -> BuildResult:
    """Build a frozen bundle from ``spec`` over ``source`` into ``out_dir``.

    When ``baseline`` is given, the schema-stability gate runs *before* the
    manifest is written: a breaking or un-versioned change raises and no
    ``manifest.json`` is produced, so a failed build never looks publishable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = tuple(_write_table(spec, table, source, out_dir) for table in spec.tables)
    candidate = BundleManifest(
        bundle_name=spec.name,
        bundle_version=spec.version,
        mint_date=manifest_mod.utc_now_iso(),
        miscope_version=manifest_mod.miscope_version(),
        description=spec.description,
        articles=spec.articles,
        tables=tables,
    )
    report = enforce(baseline, candidate) if baseline is not None else None
    candidate.write(out_dir / manifest_mod.MANIFEST_FILENAME)
    return BuildResult(manifest=candidate, out_dir=out_dir, gate=report)


def build_bundle_for_family(
    spec: BundleSpec,
    out_dir: Path,
    *,
    config: AppConfig | None = None,
    baseline: BundleManifest | None = None,
) -> BuildResult:
    """Build a bundle with the spec's family warehouse opened as the source."""
    import miscope.query as query

    with query.open(family=spec.family, config=config) as source:
        return build_bundle(spec, source, out_dir, baseline=baseline)


def _write_table(spec: BundleSpec, table, source: QueryConnection, out_dir: Path) -> TableManifest:
    """Materialize one spec table to a flat zstd Parquet and capture its manifest row."""
    path = out_dir / f"{table.name}.parquet"
    source.sql(table.sql()).write_parquet(str(path), compression=BUNDLE_COMPRESSION)
    return TableManifest(
        name=table.name,
        file=path.name,
        rows=pq.ParquetFile(path).metadata.num_rows,
        content_hash=manifest_mod.hash_file(path),
        schema=manifest_mod.capture_schema(path),
        populated_provenance=manifest_mod.populated_provenance(path),
    )
