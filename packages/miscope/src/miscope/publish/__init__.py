"""``miscope.publish`` — curated, frozen, citable data bundles for publication (REQ_110E).

The internal warehouse (``miscope.warehouse``) churns freely; a *published bundle*
is its immutable counterpart — a per-article subset of warehouse tables, frozen as
flat zstd Parquet, hashed, manifest-bearing, and attached to a ``data-*`` GitHub
Release. This module is the build + publish surface for that boundary:

- :class:`BundleSpec` — the declarative, TOML-backed description of a bundle's
  contents (which family, which tables, which article).
- :func:`build_bundle` / :func:`build_bundle_for_family` — freeze the spec's tables
  over the :mod:`miscope.query` surface into ``{table}.parquet`` + ``manifest.json``.
- :class:`BundleManifest` — the citable record (version, mint date, per-table schema
  + content hash); its committed history is the gate's baseline.
- :func:`enforce` — the schema-stability gate: breaking changes refused, additive
  changes allowed only under a new version, published tags immutable.
- :func:`create_release` — publish the built assets via ``gh``.
- :func:`verify_bundle_url` — prove the published bundle is range-request queryable.

The build never mutates the warehouse and never reaches past the query surface for
data; publication requires the explicit build + Release path, so internal changes
never silently become published.
"""

from __future__ import annotations

from miscope.publish.build import (
    BuildResult,
    build_bundle,
    build_bundle_for_family,
)
from miscope.publish.history import load_history, schema_delta, validate_history
from miscope.publish.manifest import BundleManifest, TableManifest
from miscope.publish.release import create_release, release_argv, render_notes
from miscope.publish.schema_gate import (
    GateReport,
    SchemaBreakError,
    SchemaChange,
    TableDelta,
    compare,
    enforce,
)
from miscope.publish.spec import BundleSpec, BundleTable
from miscope.publish.verify import VerifyResult, verify_bundle_url

__all__ = [
    "BundleSpec",
    "BundleTable",
    "build_bundle",
    "build_bundle_for_family",
    "BuildResult",
    "BundleManifest",
    "TableManifest",
    "enforce",
    "compare",
    "GateReport",
    "TableDelta",
    "SchemaChange",
    "SchemaBreakError",
    "create_release",
    "release_argv",
    "render_notes",
    "verify_bundle_url",
    "VerifyResult",
    "load_history",
    "validate_history",
    "schema_delta",
]
