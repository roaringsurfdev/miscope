"""Writer + reader integration on a synthetic variant (REQ_110A).

Builds a tiny ``.npz`` artifact tree for two analyzers (one per-epoch generic, one
cross-epoch with a semantic claim), materializes it, and asserts the contract:
schema-driven routing, dtype fidelity, one Parquet per coord signature, semantic
table population, catalog co-emission, ``to_wide``, and cross-variant ``concat``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.warehouse import paths, read_table
from miscope.warehouse.reader import WarehouseAccessor
from miscope.warehouse.writer import materialize_variant_columnar


class _FakeFamily:
    name = "modulo_addition_1layer"
    domain_parameters = {"prime": None, "seed": None, "data_seed": None}


class _FakeVariant:
    """Duck-typed Variant exposing only what the warehouse writer/reader use."""

    def __init__(self, root: Path, name: str, params: dict[str, int]) -> None:
        self.variant_dir = root / name
        self.name = name
        self.params = params
        self.family = _FakeFamily()
        (self.variant_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    @property
    def artifacts(self) -> ArtifactLoader:
        return ArtifactLoader(str(self.variant_dir / "artifacts"))


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(""), **arrays)


def _seed_artifacts(variant: _FakeVariant, n_neurons: int = 4) -> None:
    art = variant.variant_dir / "artifacts"
    # per-epoch, generic (FLAT): fourier_frequency_quality
    for epoch in (0, 100):
        _write_npz(
            art / "fourier_frequency_quality" / f"epoch_{epoch:05d}.npz",
            quality_score=np.float32(0.5 + epoch / 1000),
            coverage_hard=np.float32(0.9),
            active_frequencies=np.array([1, 5, 9], dtype=np.int32),
            k=np.int32(3),
            reconstruction_error=np.float32(0.01),
        )
    # cross-epoch with a semantic claim: neuron_dynamics
    epochs = np.array([0, 100], dtype=np.int64)
    _write_npz(
        art / "neuron_dynamics" / "cross_epoch.npz",
        epochs=epochs,
        dominant_freq=np.array([[0, 5, 9, 5], [5, 5, 9, 0]], dtype=np.int64),
        max_frac=np.array([[0.1, 0.4, 0.7, 0.3], [0.5, 0.4, 0.8, 0.2]], dtype=np.float32),
        switch_counts=np.arange(n_neurons, dtype=np.int32),
        commitment_epochs=np.full(n_neurons, 100.0, dtype=np.float64),
        threshold=np.float64(0.05),
    )


@pytest.fixture
def variant(tmp_path: Path) -> _FakeVariant:
    v = _FakeVariant(tmp_path, "p5_seed1_dseed2", {"prime": 5, "seed": 1, "data_seed": 2})
    _seed_artifacts(v)
    return v


def _seed_weight_spectra(variant: _FakeVariant) -> None:
    """Per-epoch `sv` artifacts with the REQ_136 uniform head axis.

    Non-attention sites carry a singleton head axis ``(1, n_sv)``; attention sites
    carry one row per head ``(n_heads, n_sv)``.
    """
    art = variant.variant_dir / "artifacts"
    for epoch in (0, 100):
        _write_npz(
            art / "weight_spectra" / f"epoch_{epoch:05d}.npz",
            sv_W_E=np.array([[3.0, 2.0, 1.0]], dtype=np.float32),  # (1, 3)
            sv_W_Q=np.array([[3.0, 2.0], [1.0, 1.0], [2.0, 1.0], [1.0, 0.5]], dtype=np.float32),
        )


def test_generic_table_one_file_per_signature_with_dtype_fidelity(variant):
    materialize_variant_columnar(variant)
    sigs = set(paths.table_dir(variant, "fourier_frequency_quality").glob("*.parquet"))
    stems = {p.stem for p in sigs}
    # scalars share one signature; the frequency-keyed field gets its own file.
    assert stems == {"by__variant_epoch", "by__variant_epoch_frequency"}

    scalars = read_table(variant, "fourier_frequency_quality", "by__variant_epoch").df
    assert {"variant_id", "prime", "seed", "data_seed", "epoch"} <= set(scalars.columns)
    assert str(scalars["quality_score"].dtype) == "float32"
    assert str(scalars["k"].dtype) == "int32"
    # epoch is a column, not a file — both epochs present in one table.
    assert sorted(scalars["epoch"].unique()) == [0, 100]


def test_tensor_fields_are_not_emitted(variant):
    # neuron_dynamics is fully columnar here; assert no stray tensor columns leak by
    # checking the generic table only carries declared columnar value columns.
    materialize_variant_columnar(variant)
    df = read_table(variant, "neuron_dynamics", "by__variant_neuron").df
    assert {"switch_counts", "commitment_epochs"} <= set(df.columns)


def test_semantic_claim_populates_neuron_frequency_attribution(variant):
    materialize_variant_columnar(variant)
    nfa = read_table(variant, "neuron_frequency_attribution").df  # single 'long' file
    assert {"variant_id", "epoch", "neuron", "frequency", "frac_explained", "dominant"} <= set(
        nfa.columns
    )
    assert nfa["dominant"].all()
    # claimed dominant_freq -> frequency; max_frac -> frac_explained
    row = nfa[(nfa.epoch == 0) & (nfa.neuron == 2)].iloc[0]
    assert row["frequency"] == 9
    assert row["frac_explained"] == pytest.approx(0.7, rel=1e-3)


def test_claimed_fields_excluded_from_generic_table(variant):
    materialize_variant_columnar(variant)
    cols = set()
    for tok in WarehouseAccessor(variant).signatures("neuron_dynamics"):
        cols |= set(read_table(variant, "neuron_dynamics", tok).df.columns)
    # dominant_freq / max_frac are claimed by neuron_frequency_attribution.
    assert "dominant_freq" not in cols
    assert "max_frac" not in cols
    assert {"switch_counts", "threshold", "epochs"} <= cols


def test_catalog_co_emitted(variant):
    materialize_variant_columnar(variant)
    cat = pd.read_parquet(paths.catalog_parquet_path(variant, "fourier_frequency_quality"))
    assert (cat["kind"] == "columnar").all()
    assert {"quality_score", "active_frequencies"} <= set(cat["field"])
    # uri is relative to the warehouse root (portable).
    assert not Path(cat["parquet_uri"].iloc[0]).is_absolute()


def test_to_wide_pivots(variant):
    materialize_variant_columnar(variant)
    scalars = read_table(variant, "fourier_frequency_quality", "by__variant_epoch")
    wide = scalars.to_wide(index="variant_id", columns="epoch", values="quality_score")
    assert list(wide.columns) == [0, 100]


def test_weight_spectra_head_is_its_own_column(variant):
    """REQ_136: `sv` head axis maps to a `head` column; row_id is the SV index alone."""
    _seed_weight_spectra(variant)
    materialize_variant_columnar(variant)
    df = read_table(variant, "weight_spectra", "by__variant_epoch_site_head_row_id").df
    assert {"site", "head", "row_id", "sv"} <= set(df.columns)
    # Non-attention site W_E: a single head (0); row_id spans the 3 singular values.
    # (Bracket access throughout: ``df.head`` is the DataFrame method, not the column.)
    we = df[df["site"] == "W_E"]
    assert sorted(we["head"].unique()) == [0]
    assert sorted(we["row_id"].unique()) == [0, 1, 2]
    # Attention site W_Q: four heads, each with its own SV index axis.
    wq = df[df["site"] == "W_Q"]
    assert sorted(wq["head"].unique()) == [0, 1, 2, 3]
    assert sorted(wq["row_id"].unique()) == [0, 1]
    # head/row_id together recover a head's singular value (no conflation).
    val = wq[(wq["head"] == 2) & (wq["row_id"] == 0) & (wq["epoch"] == 0)]["sv"].iloc[0]
    assert val == pytest.approx(2.0)


def test_cross_variant_concat_is_a_noop(tmp_path):
    v1 = _FakeVariant(tmp_path, "p5_seed1_dseed2", {"prime": 5, "seed": 1, "data_seed": 2})
    v2 = _FakeVariant(tmp_path, "p7_seed1_dseed2", {"prime": 7, "seed": 1, "data_seed": 2})
    for v in (v1, v2):
        _seed_artifacts(v)
        materialize_variant_columnar(v)
    combined = pd.concat(
        [read_table(v, "neuron_frequency_attribution").df for v in (v1, v2)],
        ignore_index=True,
    )
    assert set(combined["variant_id"].unique()) == {"p5_seed1_dseed2", "p7_seed1_dseed2"}
    # long format needs no reconciliation: one schema, variant_id distinguishes rows.
    assert combined.groupby("variant_id").size().nunique() == 1
