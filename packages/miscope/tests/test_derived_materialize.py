"""Derived-table materialization + query exposure (REQ_141, CoS #2/#3).

Materializes a synthetic variant's columnar warehouse, registers a derived table
over the conformed ``neuron_frequency_attribution`` table, and asserts: the
materialized derived table lands in the per-variant layout with co-emitted catalog
rows (so the query surface discovers it), per-table isolation quarantines a broken
derived table, and a declared view-mode table is registered live by the query layer.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest

import miscope.analysis.derived_table as dt_mod
import miscope.query as query
from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.derived_table import DerivedTableSpec, register_derived_table
from miscope.analysis.output_schema import Coord, OutputField
from miscope.warehouse import (
    materialize_variant_columnar,
    materialize_variant_derived,
    paths,
    read_table,
)


class _FakeFamily:
    name = "modulo_addition_1layer"
    domain_parameters = {"prime": None, "seed": None, "data_seed": None}
    analyzers = ("neuron_dynamics",)


class _FakeVariant:
    def __init__(self, root: Path, name: str, params: dict[str, int]) -> None:
        self.variant_dir = root / name
        self.name = name
        self.params = params
        self.family = _FakeFamily()
        (self.variant_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    @property
    def artifacts(self) -> ArtifactLoader:
        return ArtifactLoader(str(self.variant_dir / "artifacts"))

    @property
    def summary_path(self) -> Path:
        return self.variant_dir / "variant_summary.json"


def _seed_neuron_dynamics(variant: _FakeVariant) -> None:
    art = variant.variant_dir / "artifacts" / "neuron_dynamics"
    art.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        art / "cross_epoch",
        epochs=np.array([0, 100], dtype=np.int64),
        dominant_freq=np.array([[0, 5, 9, 5], [5, 5, 9, 0]], dtype=np.int64),
        max_frac=np.array([[0.1, 0.4, 0.7, 0.3], [0.5, 0.4, 0.8, 0.2]], dtype=np.float32),
        switch_counts=np.arange(4, dtype=np.int32),
        commitment_epochs=np.full(4, 100.0, dtype=np.float64),
        threshold=np.float64(0.05),
    )


@pytest.fixture
def materialized_variant(tmp_path: Path) -> _FakeVariant:
    v = _FakeVariant(tmp_path, "p5_seed1_dseed2", {"prime": 5, "seed": 1, "data_seed": 2})
    _seed_neuron_dynamics(v)
    materialize_variant_columnar(v)  # writes neuron_frequency_attribution
    return v


@pytest.fixture
def isolated_derived_registry():
    """Save/restore the global derived registry so tests don't leak registrations."""
    saved = dict(dt_mod._derived)
    dt_mod._derived.clear()
    try:
        yield
    finally:
        dt_mod._derived.clear()
        dt_mod._derived.update(saved)


def _committed_counts_spec() -> DerivedTableSpec:
    return DerivedTableSpec(
        name="committed_counts_test",
        query=(
            "SELECT epoch, frequency, COUNT(*) AS committed_count "
            "FROM neuron_frequency_attribution WHERE frac_explained >= 0.70 "
            "GROUP BY epoch, frequency ORDER BY epoch, frequency"
        ),
        input_tables=("neuron_frequency_attribution",),
        outputs=(
            OutputField.columnar(
                "committed_count",
                "int64",
                (Coord.VARIANT, Coord.EPOCH, Coord.FREQUENCY),
                "Neurons committed to each frequency per epoch.",
            ),
        ),
    )


def test_materialized_derived_table_lands_and_is_queryable(
    materialized_variant: _FakeVariant, isolated_derived_registry
):
    register_derived_table(_committed_counts_spec())
    report = materialize_variant_derived(materialized_variant)

    assert report.tables == {"committed_counts_test": 2}
    assert not report.failed

    # Lands in the per-variant semantic layout (discoverable by miscope.query).
    path = paths.semantic_parquet_path(materialized_variant, "committed_counts_test")
    assert path.exists()

    table = read_table(materialized_variant, "committed_counts_test")
    df = table.df.sort_values(["epoch", "frequency"]).reset_index(drop=True)
    # frac_explained >= 0.70 hits neuron 2 (freq 9) at both epochs.
    assert list(df["epoch"]) == [0, 100]
    assert list(df["frequency"]) == [9, 9]
    assert list(df["committed_count"]) == [1, 1]
    # Key columns line up with the rest of the warehouse.
    assert list(df["variant_id"]) == ["p5_seed1_dseed2", "p5_seed1_dseed2"]
    assert set(df["run_set"]) == {paths.DEFAULT_RUN_SET}


def test_catalog_rows_co_emitted_for_derived_table(
    materialized_variant: _FakeVariant, isolated_derived_registry
):
    register_derived_table(_committed_counts_spec())
    materialize_variant_derived(materialized_variant)

    catalog_path = paths.catalog_parquet_path(materialized_variant, "committed_counts_test")
    assert catalog_path.exists()
    import pandas as pd

    cat = pd.read_parquet(catalog_path)
    row = cat[cat["field"] == "committed_count"].iloc[0]
    assert row["kind"] == "columnar"
    assert row["table"] == "committed_counts_test"
    assert row["coords"] == "variant_id, epoch, frequency"


def test_broken_derived_table_is_isolated(
    materialized_variant: _FakeVariant, isolated_derived_registry
):
    """A failing derived table is recorded and skipped; a good one still materializes."""
    register_derived_table(_committed_counts_spec())
    register_derived_table(
        DerivedTableSpec(
            name="broken",
            query="SELECT no_such_column FROM neuron_frequency_attribution",
            input_tables=("neuron_frequency_attribution",),
            outputs=(OutputField.columnar("x", "int64", (Coord.VARIANT,), "x"),),
        )
    )
    report = materialize_variant_derived(materialized_variant)

    assert "committed_counts_test" in report.tables
    assert "broken" in report.failed


def test_inputs_absent_skips_without_failing(
    materialized_variant: _FakeVariant, isolated_derived_registry
):
    register_derived_table(
        DerivedTableSpec(
            name="needs_missing",
            query="SELECT * FROM not_materialized",
            input_tables=("not_materialized",),
            outputs=(OutputField.columnar("x", "int64", (Coord.VARIANT,), "x"),),
        )
    )
    report = materialize_variant_derived(materialized_variant)
    assert report.skipped == ["needs_missing"]
    assert not report.failed


def test_view_mode_derived_table_registered_live(isolated_derived_registry):
    """A declared view-mode derived table is registered as a live DuckDB view."""
    con = duckdb.connect()
    con.execute(
        "CREATE VIEW neuron_frequency_attribution AS "
        "SELECT * FROM (VALUES (0, 9), (0, 9), (100, 9)) AS t(epoch, frequency)"
    )
    register_derived_table(
        DerivedTableSpec(
            name="view_counts",
            query=("SELECT epoch, COUNT(*) AS c FROM neuron_frequency_attribution GROUP BY epoch"),
            input_tables=("neuron_frequency_attribution",),
            outputs=(OutputField.columnar("c", "int64", (Coord.VARIANT, Coord.EPOCH), "count"),),
            materialized=False,
        )
    )
    views = ["neuron_frequency_attribution"]
    query._register_derived_views(con, views)

    assert "view_counts" in views
    out = con.sql("SELECT epoch, c FROM view_counts ORDER BY epoch").df()
    assert list(out["epoch"]) == [0, 100]
    assert list(out["c"]) == [2, 1]


def test_view_mode_not_materialized(materialized_variant: _FakeVariant, isolated_derived_registry):
    """A view-mode table persists nothing and is reported as a view, not a table."""
    register_derived_table(
        DerivedTableSpec(
            name="view_only",
            query="SELECT epoch, COUNT(*) AS c FROM neuron_frequency_attribution GROUP BY epoch",
            input_tables=("neuron_frequency_attribution",),
            outputs=(OutputField.columnar("c", "int64", (Coord.VARIANT, Coord.EPOCH), "count"),),
            materialized=False,
        )
    )
    report = materialize_variant_derived(materialized_variant)
    assert report.views == ["view_only"]
    assert "view_only" not in report.tables
    assert not paths.semantic_parquet_path(materialized_variant, "view_only").exists()
