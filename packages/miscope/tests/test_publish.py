"""Publication bundles: build, manifest, schema gate, history (REQ_110E).

Builds derived bundles over a *bundle-mode* source connection (flat Parquet in a
temp dir) rather than a full family warehouse — the build only ever reaches data
through the query surface, so a flat source exercises the same path without the
heavy fixture. Asserts the structural internal-vs-published distinction (frozen
files + hashed manifest), the schema-stability gate (identical / additive-needs-bump
/ breaking-fails), and the committed manifest-history validation the CI gate reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import miscope.query as query
from miscope.publish import (
    BundleManifest,
    BundleSpec,
    TableManifest,
    build_bundle,
    compare,
    enforce,
    load_history,
    render_notes,
    schema_delta,
    validate_history,
)
from miscope.publish.manifest import MANIFEST_FILENAME
from miscope.publish.release import create_release
from miscope.publish.schema_gate import SchemaBreakError, SchemaChange

# ---------------------------------------------------------------------------
# Fixtures: a flat-Parquet source the build can query (bundle mode)
# ---------------------------------------------------------------------------


def _write_source(root: Path) -> None:
    """A two-table flat source. `shape_characterizations` carries reserved provenance.

    `analyzer_version` is populated, `trust_tier` is reserved-but-null — so the
    manifest should record exactly `analyzer_version` as populated provenance.
    """
    root.mkdir(parents=True, exist_ok=True)
    shapes = pd.DataFrame(
        {
            "variant_id": ["p113", "p113", "p109"],
            "epoch": [5000, 6000, 5000],
            "group": ["MLP", "MLP", "MLP"],
            "operation_type": ["circularity", "circularity", "curvature"],
            "value": [0.91, 0.93, 0.40],
            "analyzer_version": ["1.0", "1.0", "1.0"],
            "trust_tier": [None, None, None],
        }
    )
    shapes.to_parquet(root / "shape_characterizations.parquet", index=False)
    freq = pd.DataFrame(
        {
            "variant_id": ["p113", "p113"],
            "epoch": [5000, 5000],
            "site": ["mlp_out", "mlp_out"],
            "frequency": [25, 42],
            "magnitude": [0.7, 0.3],
        }
    )
    freq.to_parquet(root / "frequency_spectrum.parquet", index=False)


@pytest.fixture
def source(tmp_path):
    _write_source(tmp_path / "source")
    with query.open(root=str(tmp_path / "source")) as con:
        yield con


def _spec(version: str = "v1.0", *, query_str: str | None = None) -> BundleSpec:
    return BundleSpec.from_dict(
        {
            "name": "ring-geometry",
            "version": version,
            "family": "modulo_addition_1layer",
            "description": "Circularity backing the ring finding.",
            "articles": ["ring-geometry"],
            "tables": [
                {"name": "shape_characterizations", "query": query_str}
                if query_str
                else {"name": "shape_characterizations"},
                {"name": "frequency_spectrum"},
            ],
        }
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def test_build_writes_flat_parquet_and_manifest(source, tmp_path):
    out = tmp_path / "dist"
    result = build_bundle(_spec(), source, out)

    # Flat shape: one Parquet per table + a manifest.
    assert (out / "shape_characterizations.parquet").exists()
    assert (out / "frequency_spectrum.parquet").exists()
    assert (out / MANIFEST_FILENAME).exists()

    m = result.manifest
    assert m.tag == "data-ring-geometry-v1.0"
    shapes = m.table("shape_characterizations")
    assert shapes.rows == 3
    assert shapes.content_hash.startswith("sha256:")
    assert shapes.schema["variant_id"] == "string"
    assert "value" in shapes.schema
    # Reserved provenance: analyzer_version populated, trust_tier null.
    assert shapes.populated_provenance == ("analyzer_version",)


def test_build_subset_query_filters_rows(source, tmp_path):
    spec = _spec(
        query_str="SELECT * FROM shape_characterizations WHERE operation_type = 'circularity'"
    )
    result = build_bundle(spec, source, tmp_path / "dist")
    assert result.manifest.table("shape_characterizations").rows == 2  # curvature row dropped


def test_built_bundle_is_queryable_in_bundle_mode(source, tmp_path):
    out = tmp_path / "dist"
    build_bundle(_spec(), source, out)
    with query.open(root=str(out), tables=["frequency_spectrum"]) as con:
        n = con.df("SELECT COUNT(*) AS n FROM frequency_spectrum")["n"].iloc[0]
    assert n == 2


def test_manifest_json_roundtrip(source, tmp_path):
    result = build_bundle(_spec(), source, tmp_path / "dist")
    restored = BundleManifest.from_json(result.manifest.to_json())
    assert restored == result.manifest
    # Manifest is valid JSON on disk.
    json.loads((result.out_dir / MANIFEST_FILENAME).read_text())


# ---------------------------------------------------------------------------
# Schema-stability gate
# ---------------------------------------------------------------------------


def _manifest(version: str, schema: dict[str, str]) -> BundleManifest:
    return BundleManifest(
        bundle_name="ring-geometry",
        bundle_version=version,
        mint_date="2026-06-05T00:00:00+00:00",
        miscope_version="0.1.0",
        description="",
        articles=("ring-geometry",),
        tables=(
            TableManifest(
                name="shape_characterizations",
                file="shape_characterizations.parquet",
                rows=3,
                content_hash="sha256:deadbeef",
                schema=schema,
                populated_provenance=(),
            ),
        ),
    )


BASE_SCHEMA = {"variant_id": "string", "epoch": "int64", "value": "double"}


def test_gate_identical_allows_same_version():
    base = _manifest("v1.0", BASE_SCHEMA)
    report = enforce(base, _manifest("v1.0", BASE_SCHEMA))
    assert report.change is SchemaChange.IDENTICAL


def test_gate_additive_requires_version_bump():
    base = _manifest("v1.0", BASE_SCHEMA)
    additive = {**BASE_SCHEMA, "spread": "double"}
    with pytest.raises(SchemaBreakError, match="additive"):
        enforce(base, _manifest("v1.0", additive))
    # Same additive change under a new version is allowed.
    report = enforce(base, _manifest("v1.1", additive))
    assert report.change is SchemaChange.ADDITIVE


def test_gate_breaking_rename_fails_even_with_bump():
    base = _manifest("v1.0", BASE_SCHEMA)
    renamed = {
        "variant_id": "string",
        "epoch": "int64",
        "circularity": "double",
    }  # value -> circularity
    with pytest.raises(SchemaBreakError, match="breaks the published schema"):
        enforce(base, _manifest("v2.0", renamed))


def test_gate_breaking_retype_fails():
    base = _manifest("v1.0", BASE_SCHEMA)
    retyped = {**BASE_SCHEMA, "epoch": "string"}
    report = compare(base, _manifest("v2.0", retyped))
    assert report.change is SchemaChange.BREAKING
    assert "epoch: int64->string" in "; ".join(report.breaking_reasons())


# ---------------------------------------------------------------------------
# Release command assembly (pure; no gh invocation)
# ---------------------------------------------------------------------------


def test_release_argv_and_notes(source, tmp_path):
    result = build_bundle(_spec(), source, tmp_path / "dist")
    notes = render_notes(result.manifest)
    assert "data-ring-geometry-v1.0" not in notes  # notes are human text, tag is the gh arg
    assert "shape_characterizations" in notes

    argv = create_release(result, draft=True, dry_run=True)  # dry run: assembles, never runs gh
    assert argv[:4] == ["gh", "release", "create", "data-ring-geometry-v1.0"]
    assert "--draft" in argv
    assert str(result.manifest_path) in argv
    assert any(p.name == "NOTES.md" for p in result.out_dir.iterdir())


# ---------------------------------------------------------------------------
# Committed manifest history (the CI gate's data-free baseline)
# ---------------------------------------------------------------------------


def test_history_validation_and_delta(tmp_path):
    hist = tmp_path / "ring-geometry"
    hist.mkdir()
    _manifest("v1.0", BASE_SCHEMA).write(hist / "manifest-v1.0.json")
    _manifest("v1.1", {**BASE_SCHEMA, "spread": "double"}).write(hist / "manifest-v1.1.json")

    assert validate_history(hist) == []
    versions = [m.bundle_version for m in load_history(hist)]
    assert versions == ["v1.0", "v1.1"]  # numeric-aware order
    delta = schema_delta(hist)
    assert delta.change is SchemaChange.ADDITIVE


def test_history_validation_flags_filename_version_mismatch(tmp_path):
    hist = tmp_path / "ring-geometry"
    hist.mkdir()
    # Recorded version v2.0 but filed under v1.0 — an integrity problem.
    _manifest("v2.0", BASE_SCHEMA).write(hist / "manifest-v1.0.json")
    problems = validate_history(hist)
    assert problems and "expects filename" in problems[0]
