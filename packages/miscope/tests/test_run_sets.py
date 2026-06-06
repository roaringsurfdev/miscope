"""Tests for the run-set registry relation + query surface (REQ_138, Phase 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.parameters import LiteralBinding, Parameterization, ParameterSpec
from miscope.analysis.spec import AnalyzerSpec
from miscope.warehouse import paths, read_run_sets, record_run_set, run_set_id
from miscope.warehouse.writer import materialize_variant_columnar


class _FakeFamily:
    name = "modulo_addition_1layer"
    domain_parameters = {"prime": None, "seed": None, "data_seed": None}
    # Declared analyzer scope (REQ_140): the materializer iterates this set.
    analyzers = ("fourier_frequency_quality",)

    @property
    def variants_dir(self) -> Path:
        return self._variants_dir  # set by the fixture


class _FakeVariant:
    def __init__(self, root: Path, name: str = "p5_seed1_dseed2") -> None:
        self.variant_dir = root / "variants" / name
        self.name = name
        self.params = {"prime": 5, "seed": 1, "data_seed": 2}
        self.family = _FakeFamily()
        self.family._variants_dir = root / "variants"
        (self.variant_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    @property
    def artifacts(self) -> ArtifactLoader:
        return ArtifactLoader(str(self.variant_dir / "artifacts"))


def _seed_one_analyzer(variant: _FakeVariant) -> None:
    art = variant.variant_dir / "artifacts"
    for epoch in (0, 100):
        p = art / "fourier_frequency_quality" / f"epoch_{epoch:05d}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p.with_suffix(""),
            quality_score=np.float32(0.5 + epoch / 1000),
            coverage_hard=np.float32(0.9),
            active_frequencies=np.array([1, 5, 9], dtype=np.int32),
            k=np.int32(3),
            reconstruction_error=np.float32(0.01),
        )


@pytest.fixture
def variant(tmp_path: Path) -> _FakeVariant:
    v = _FakeVariant(tmp_path)
    _seed_one_analyzer(v)
    return v


def _specs() -> dict[str, AnalyzerSpec]:
    from miscope.analysis.inputs import ArtifactInput, ModelInput

    grouping = AnalyzerSpec(name="grouping", inputs=(ModelInput(),))
    dmd = AnalyzerSpec(
        name="dmd",
        output_scope="cross_epoch",
        inputs=(ArtifactInput("grouping"),),
        parameters=(
            ParameterSpec(
                "reference_epoch", "int64", "analyzer", LiteralBinding("reference_epoch", 0)
            ),
        ),
    )
    return {s.name: s for s in (grouping, dmd)}


# ---------------------------------------------------------------------------
# Registry relation
# ---------------------------------------------------------------------------


def test_run_set_id_is_recipe_derived_and_default_for_empty():
    assert run_set_id(Parameterization()) == paths.DEFAULT_RUN_SET
    a = Parameterization(bindings=(LiteralBinding("reference_epoch", 20000, analyzer="dmd"),))
    b = Parameterization(bindings=(LiteralBinding("reference_epoch", 30000, analyzer="dmd"),))
    assert run_set_id(a) not in (run_set_id(b), paths.DEFAULT_RUN_SET)


def test_record_and_read_run_set_round_trips(variant):
    pz = Parameterization(
        bindings=(LiteralBinding("reference_epoch", 20000, analyzer="dmd"),), label="early-ref"
    )
    record_run_set(variant, pz, {"dmd": "abc123"}, {"dmd": {"reference_epoch": 20000}}, _specs())
    df = read_run_sets(variant)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["analyzer"] == "dmd"
    assert row["label"] == "early-ref"
    assert row["recipe_signature"] == "abc123"
    assert "20000" in row["bindings_json"]


def test_record_run_set_is_idempotent_on_run_set_analyzer(variant):
    pz = Parameterization(bindings=(LiteralBinding("reference_epoch", 20000, analyzer="dmd"),))
    for _ in range(2):
        record_run_set(variant, pz, {"dmd": "sig"}, {"dmd": {"reference_epoch": 20000}}, _specs())
    assert len(read_run_sets(variant)) == 1


def test_empty_parameterization_records_nothing(variant):
    record_run_set(variant, Parameterization(), {}, {}, _specs())
    assert read_run_sets(variant).empty


# ---------------------------------------------------------------------------
# Query surface
# ---------------------------------------------------------------------------


def test_run_set_column_present_on_default_columnar_plane(variant):
    """The run_set coordinate is materialized once on every table + catalog row."""
    materialize_variant_columnar(variant)
    scalars = pd.read_parquet(
        next(paths.table_dir(variant, "fourier_frequency_quality").glob("*.parquet"))
    )
    assert "run_set" in scalars.columns
    assert (scalars["run_set"] == paths.DEFAULT_RUN_SET).all()
    cat = pd.read_parquet(next(paths.catalog_dir(variant).glob("*.parquet")))
    assert "run_set" in cat.columns


def test_run_sets_view_registered_and_queryable(variant):
    import miscope.query

    materialize_variant_columnar(variant)
    pz = Parameterization(
        bindings=(LiteralBinding("reference_epoch", 20000, analyzer="dmd"),), label="early-ref"
    )
    record_run_set(variant, pz, {"dmd": "abc123"}, {"dmd": {"reference_epoch": 20000}}, _specs())

    con = miscope.query.open(family=variant.family)
    try:
        assert "run_sets" in con.tables()
        df = con.df("SELECT DISTINCT label FROM run_sets")
        assert "early-ref" in set(df["label"])
    finally:
        con.close()
