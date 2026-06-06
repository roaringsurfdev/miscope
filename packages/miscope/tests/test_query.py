"""DuckDB query surface over the warehouse (REQ_110C).

Builds a two-variant family with a fully materialized warehouse — both planes:
columnar tables + ``_catalog`` (110-A) and tensor descriptors + ``_tensor_catalog``
(110-B) — then asserts the query contract: ergonomic per-table views globbed
across variants, a unified ``catalog`` view that ``UNION ALL BY NAME``s the two
planes over the ``kind`` discriminator, cross-variant aggregation, cross-table
joins, the three canonical one-line questions, and the bundle (flat-file) mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import miscope.query as query
from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.warehouse import materialize_variant_columnar, materialize_variant_tensors


class _FakeFamily:
    name = "modulo_addition_1layer"
    domain_parameters = {"prime": None, "seed": None, "data_seed": None}
    # Declared analyzer scope (REQ_140): the materializer iterates this set.
    analyzers = (
        "fourier_frequency_quality",
        "neuron_frequency_attribution",
        "neuron_dynamics",
        "parameter_snapshot",
    )

    def __init__(self, variants_dir: Path) -> None:
        self.variants_dir = variants_dir


class _FakeVariant:
    """Duck-typed Variant exposing only what the warehouse materializers use."""

    def __init__(self, family: _FakeFamily, name: str, params: dict[str, int]) -> None:
        self.family = family
        self.variant_dir = family.variants_dir / name
        self.name = name
        self.params = params
        (self.variant_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    @property
    def artifacts(self) -> ArtifactLoader:
        return ArtifactLoader(str(self.variant_dir / "artifacts"))


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(""), **arrays)


def _seed_columnar(variant: _FakeVariant, n_neurons: int = 4) -> None:
    """Per-epoch scalars + a cross-epoch semantic claim (neuron_frequency_attribution)."""
    art = variant.variant_dir / "artifacts"
    for epoch in (0, 100):
        _write_npz(
            art / "fourier_frequency_quality" / f"epoch_{epoch:05d}.npz",
            quality_score=np.float32(0.5 + epoch / 1000),
            coverage_hard=np.float32(0.9),
            active_frequencies=np.array([1, 5, 9], dtype=np.int32),
            k=np.int32(3),
            reconstruction_error=np.float32(0.01),
        )
    # REQ_141: per-epoch attribution feeds the neuron_frequency_attribution table.
    dominant_by_epoch = {0: [0, 5, 9, 5], 100: [5, 5, 9, 0]}
    frac_by_epoch = {0: [0.1, 0.4, 0.7, 0.3], 100: [0.5, 0.4, 0.8, 0.2]}
    for epoch in (0, 100):
        _write_npz(
            art / "neuron_frequency_attribution" / f"epoch_{epoch:05d}.npz",
            dominant_freq=np.array(dominant_by_epoch[epoch], dtype=np.int64),
            max_frac=np.array(frac_by_epoch[epoch], dtype=np.float64),
        )
    _write_npz(
        art / "neuron_dynamics" / "cross_epoch.npz",
        epochs=np.array([0, 100], dtype=np.int64),
        switch_counts=np.arange(n_neurons, dtype=np.int32),
        commitment_epochs=np.full(n_neurons, 100.0, dtype=np.float64),
        threshold=np.float64(0.05),
    )


def _seed_tensors(variant: _FakeVariant) -> None:
    """A couple of real tensor-declaring analyzers so the tensor catalog has rows."""
    art = variant.variant_dir / "artifacts"
    for epoch in (0, 100):
        _write_npz(
            art / "parameter_snapshot" / f"epoch_{epoch:05d}.npz",
            W_E=np.arange(12, dtype=np.float32).reshape(3, 4) + epoch,
            W_in=np.ones((4, 5), dtype=np.float32) * epoch,
        )


def _make_variant(family: _FakeFamily, name: str, params: dict[str, int]) -> _FakeVariant:
    v = _FakeVariant(family, name, params)
    _seed_columnar(v)
    _seed_tensors(v)
    materialize_variant_columnar(v)
    materialize_variant_tensors(v)
    return v


@pytest.fixture
def family(tmp_path: Path) -> _FakeFamily:
    fam = _FakeFamily(tmp_path / "variants")
    _make_variant(fam, "p5_seed1_dseed2", {"prime": 5, "seed": 1, "data_seed": 2})
    _make_variant(fam, "p7_seed1_dseed2", {"prime": 7, "seed": 1, "data_seed": 2})
    return fam


def test_views_registered_per_table_plus_unified_catalog(family):
    with query.open(family=family) as con:
        names = set(con.tables())
    # every materialized table is an ergonomic view, and the two catalog planes
    # collapse into one `catalog` view (not two `_catalog`/`_tensor_catalog` ones).
    assert {"fourier_frequency_quality", "neuron_dynamics", "neuron_frequency_attribution"} <= names
    assert "catalog" in names
    assert "_catalog" not in names and "_tensor_catalog" not in names


def test_cross_variant_query_is_a_plain_select(family):
    """variant_id is a column, so one SELECT spans every variant with no concat."""
    with query.open(family=family) as con:
        df = con.df(
            "SELECT variant_id, COUNT(*) AS n FROM neuron_frequency_attribution "
            "GROUP BY variant_id ORDER BY variant_id"
        )
    assert list(df["variant_id"]) == ["p5_seed1_dseed2", "p7_seed1_dseed2"]
    assert (df["n"] > 0).all()


def test_catalog_unions_both_planes_by_name(family):
    """The shared relation is the UNION ALL BY NAME of columnar + tensor rows."""
    with query.open(family=family) as con:
        by_kind = con.df("SELECT kind, COUNT(*) AS n FROM catalog GROUP BY kind")
        # plane-specific columns coexist (null-padded) — proof it is a by-name union.
        cols = con.df(
            "SELECT parquet_uri, tensor_uri FROM catalog "
            "WHERE kind='tensor' AND tensor_uri IS NOT NULL LIMIT 1"
        )
    kinds = dict(zip(by_kind["kind"], by_kind["n"], strict=True))
    assert kinds.get("columnar", 0) > 0 and kinds.get("tensor", 0) > 0
    # a tensor row carries its address but no columnar payload pointer.
    assert cols["parquet_uri"].isna().all() and cols["tensor_uri"].notna().all()


def test_cross_table_join_runs_in_duckdb(family):
    """Joins are plain SQL over the views — no custom join logic in miscope.query."""
    with query.open(family=family) as con:
        joined = con.df(
            "SELECT a.variant_id, a.neuron, a.frequency "
            "FROM neuron_frequency_attribution a "
            "JOIN fourier_frequency_quality f "
            "  ON a.variant_id = f.variant_id AND a.epoch = f.epoch "
            "WHERE a.dominant"
        )
    assert len(joined) > 0
    assert set(joined["variant_id"]) == {"p5_seed1_dseed2", "p7_seed1_dseed2"}


def test_canonical_one_line_question(family):
    """A canonical cross-variant question expressed as one-line SQL (validation CoS)."""
    with query.open(family=family) as con:
        ranked = con.df(
            "SELECT variant_id, neuron, frac_explained "
            "FROM neuron_frequency_attribution "
            "WHERE epoch = 0 AND dominant ORDER BY frac_explained DESC"
        )
    assert list(ranked.columns) == ["variant_id", "neuron", "frac_explained"]
    assert ranked["frac_explained"].is_monotonic_decreasing


def test_tables_filter_restricts_registered_views(family):
    """`tables=` gates the data-table views; the `catalog` index stays (it spans all)."""
    with query.open(family=family, tables=["neuron_frequency_attribution"]) as con:
        assert con.tables() == ["neuron_frequency_attribution", "catalog"]
        # the unrequested table is absent...
        with pytest.raises(Exception):  # noqa: B017 — any DuckDB "table not found"
            con.sql("SELECT * FROM fourier_frequency_quality")
        # ...but the whole-warehouse catalog index is always available.
        assert con.df("SELECT COUNT(*) AS n FROM catalog")["n"].iloc[0] > 0


def test_bundle_mode_over_flat_local_files(family, tmp_path):
    """Bundle mode: one view per flat {root}/{table}.parquet (the published shape)."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with query.open(family=family) as con:
        nfa = con.df("SELECT * FROM neuron_frequency_attribution")
    nfa.to_parquet(bundle / "ring_geometry.parquet", index=False)

    with query.open(root=str(bundle)) as con:
        assert con.tables() == ["ring_geometry"]
        out = con.df("SELECT variant_id, COUNT(*) AS n FROM ring_geometry GROUP BY variant_id")
    assert set(out["variant_id"]) == {"p5_seed1_dseed2", "p7_seed1_dseed2"}


def test_open_requires_exactly_one_root_selector(family):
    with pytest.raises(ValueError, match="exactly one"):
        query.open()
    with pytest.raises(ValueError, match="exactly one"):
        query.open(family=family, root="/tmp/whatever")
