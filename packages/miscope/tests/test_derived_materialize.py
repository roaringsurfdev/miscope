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
    analyzers = ("neuron_frequency_attribution",)


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


def _seed_attribution(variant: _FakeVariant) -> None:
    """Per-epoch neuron_frequency_attribution artifacts (the conformed source)."""
    art = variant.variant_dir / "artifacts" / "neuron_frequency_attribution"
    art.mkdir(parents=True, exist_ok=True)
    dominant_by_epoch = {0: [0, 5, 9, 5], 100: [5, 5, 9, 0]}
    frac_by_epoch = {0: [0.1, 0.4, 0.7, 0.3], 100: [0.5, 0.4, 0.8, 0.2]}
    for epoch in (0, 100):
        np.savez_compressed(
            art / f"epoch_{epoch:05d}",
            dominant_freq=np.array(dominant_by_epoch[epoch], dtype=np.int64),
            max_frac=np.array(frac_by_epoch[epoch], dtype=np.float64),
        )


@pytest.fixture
def materialized_variant(tmp_path: Path) -> _FakeVariant:
    v = _FakeVariant(tmp_path, "p5_seed1_dseed2", {"prime": 5, "seed": 1, "data_seed": 2})
    _seed_attribution(v)
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


def _seed_transient_pattern(variant: _FakeVariant) -> None:
    """Per-epoch attribution with one transient (freq 3) and one final (freq 7) freq.

    d_mlp=20. Epoch 100: neurons 0-4 commit to freq 3, neurons 5-9 to freq 7.
    Epoch 200: freq 3 cohort abandons (max_frac drops), freq 7 cohort holds — so
    freq 3 is transient with 5 homeless, freq 7 is final.
    """
    art = variant.variant_dir / "artifacts" / "neuron_frequency_attribution"
    art.mkdir(parents=True, exist_ok=True)
    d_mlp = 20
    plan = {
        0: ([0] * d_mlp, [0.5] * d_mlp),  # nothing committed
        100: ([3] * 5 + [7] * 5 + [0] * 10, [0.9] * 10 + [0.5] * 10),
        200: ([3] * 5 + [7] * 5 + [0] * 10, [0.5] * 5 + [0.9] * 5 + [0.5] * 10),
    }
    for epoch, (dom, frac) in plan.items():
        np.savez_compressed(
            art / f"epoch_{epoch:05d}",
            dominant_freq=np.array(dom, dtype=np.int64),
            max_frac=np.array(frac, dtype=np.float64),
        )


def test_transient_derived_tables_reproduce_analyzer(tmp_path: Path):
    """The real transient derived tables compute committed/peak/final/homeless/members.

    Exercises the registered committed_counts / transient_frequencies /
    transient_peak_members tables end to end (the bucket-2 proof), replacing the
    retired analyzer's unit coverage.
    """
    v = _FakeVariant(tmp_path, "p23_seed1_dseed2", {"prime": 23, "seed": 1, "data_seed": 2})
    _seed_transient_pattern(v)
    materialize_variant_columnar(v)
    report = materialize_variant_derived(v)

    assert {"committed_counts", "transient_frequencies", "transient_peak_members"} <= set(
        report.tables
    )

    tf = read_table(v, "transient_frequencies").df.sort_values("frequency").reset_index(drop=True)
    assert list(tf["frequency"]) == [3, 7]
    row3 = tf[tf.frequency == 3].iloc[0]
    row7 = tf[tf.frequency == 7].iloc[0]
    assert row3["peak_epoch"] == 100 and row3["peak_count"] == 5
    assert not bool(row3["is_final"]) and row3["homeless_count"] == 5  # transient, all homeless
    assert row7["peak_epoch"] == 100 and row7["peak_count"] == 5
    assert bool(row7["is_final"]) and row7["homeless_count"] == 0  # final, none homeless

    members = read_table(v, "transient_peak_members").df
    freq3_members = sorted(members[members.frequency == 3]["member_neuron"])
    freq7_members = sorted(members[members.frequency == 7]["member_neuron"])
    assert freq3_members == [0, 1, 2, 3, 4]
    assert freq7_members == [5, 6, 7, 8, 9]


def test_derived_rematerializes_when_input_recomputed(tmp_path: Path):
    """REQ_141 freshness: recomputing an input table rebuilds its derived tables.

    Derived tables join the materialize DAG downstream of their inputs (no separate
    staleness mechanism): re-running the columnar+derived pass after the attribution
    artifact changes yields updated derived tables.
    """
    v = _FakeVariant(tmp_path, "p23_seed9_dseed9", {"prime": 23, "seed": 9, "data_seed": 9})
    _seed_transient_pattern(v)
    materialize_variant_columnar(v)
    materialize_variant_derived(v)
    before = read_table(v, "committed_counts").df
    final7 = before[(before.epoch == 200) & (before.frequency == 7)]["committed_counts"]
    assert int(final7.iloc[0]) == 5

    # Recompute the input: at the final epoch, freq 7's cohort abandons too.
    art = v.variant_dir / "artifacts" / "neuron_frequency_attribution"
    np.savez_compressed(
        art / "epoch_00200",
        dominant_freq=np.array([3] * 5 + [7] * 5 + [0] * 10, dtype=np.int64),
        max_frac=np.array([0.5] * 20, dtype=np.float64),  # nobody committed at final
    )
    materialize_variant_columnar(v)
    materialize_variant_derived(v)
    after = read_table(v, "committed_counts").df
    # Epoch 200 now has no committed neurons -> the row is gone (rebuilt, not stale).
    assert after[(after.epoch == 200)].empty


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
