"""Tensor catalog + resolver on a synthetic variant (REQ_110B).

Seeds a tiny ``.npz`` blob tree for real registered tensor-declaring analyzers
(``parameter_snapshot`` per-epoch; ``weight_basis_projection`` per-epoch + site;
``parameter_trajectory`` cross-epoch + group), materializes the descriptor
relation, and asserts the contract: address-only descriptors, instance-coordinate
columns, declared-vs-actual dtype drift surfaced as a finding (non-fatal),
SQL-over-descriptors touching zero array bytes, per-container batched resolution,
the shape/dtype reproducibility guard, and a write -> select -> resolve round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.warehouse import (
    materialize_variant_tensors,
    paths,
    read_tensor_catalog,
    tensor_catalog,
)
from miscope.warehouse.tensor_catalog import TensorCatalogAccessor, TensorResolver


class _FakeFamily:
    name = "modulo_addition_1layer"
    domain_parameters = {"prime": None, "seed": None, "data_seed": None}
    # Declared analyzer scope (REQ_140): the columnar materializer iterates this set.
    analyzers = ("parameter_snapshot", "weight_basis_projection", "parameter_trajectory")


class _FakeVariant:
    """Duck-typed Variant exposing only what the tensor catalog uses."""

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
    def tensor_catalog(self) -> TensorCatalogAccessor:
        return TensorCatalogAccessor(self)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(""), **arrays)


# Reference payloads, keyed (analyzer, container-relative member) -> array.
def _seed_blobs(variant: _FakeVariant) -> dict[tuple[str, str], np.ndarray]:
    art = variant.variant_dir / "artifacts"
    ref: dict[tuple[str, str], np.ndarray] = {}
    # parameter_snapshot: per-epoch, FLAT keys, float32, keyed (variant, epoch).
    for epoch in (0, 100):
        we = np.arange(12, dtype=np.float32).reshape(3, 4) + epoch
        win = np.ones((4, 5), dtype=np.float32) * epoch
        _write_npz(art / "parameter_snapshot" / f"epoch_{epoch:05d}.npz", W_E=we, W_in=win)
        ref[("parameter_snapshot", f"e{epoch}_W_E")] = we
        ref[("parameter_snapshot", f"e{epoch}_W_in")] = win
    # weight_basis_projection: per-epoch, PREFIX_US site, float64, key {site}_{field}.
    power = np.array([[0.2, 0.8], [0.5, 0.5]], dtype=np.float64)
    _write_npz(art / "weight_basis_projection" / "epoch_00000.npz", mlp_out_power=power)
    ref[("weight_basis_projection", "mlp_out_power")] = power
    # parameter_trajectory: cross-epoch, DUS group, float64, key {group}__projections.
    proj = np.linspace(0, 1, 6, dtype=np.float64).reshape(2, 3)
    _write_npz(art / "parameter_trajectory" / "cross_epoch.npz", group_8__projections=proj)
    ref[("parameter_trajectory", "group_8__projections")] = proj
    return ref


@pytest.fixture
def variant(tmp_path: Path) -> _FakeVariant:
    v = _FakeVariant(tmp_path, "p5_seed1_dseed2", {"prime": 5, "seed": 1, "data_seed": 2})
    _seed_blobs(v)
    return v


def test_descriptors_are_address_only_with_instance_coords(variant):
    materialize_variant_tensors(variant)
    df = read_tensor_catalog(variant)
    assert (df["kind"] == "tensor").all()
    # the relation carries the instance-coordinate spine + the family params.
    assert {"variant_id", "prime", "seed", "data_seed", "epoch", "site", "group"} <= set(df.columns)
    # address-only: a uri/member/dtype/shape/codec, never a payload column.
    assert {"tensor_uri", "tensor_member", "tensor_dtype", "tensor_shape", "tensor_codec"} <= set(
        df.columns
    )
    # parameter_snapshot: epoch is the instance key; site/group are null (matrix axes).
    psnap = df[df["analyzer"] == "parameter_snapshot"]
    assert sorted(psnap["epoch"].unique()) == [0, 100]
    assert psnap["site"].isna().all() and psnap["group"].isna().all()
    # uri is variant-relative (portable), pointing into the untouched artifacts tree.
    assert not Path(psnap["tensor_uri"].iloc[0]).is_absolute()
    assert psnap["tensor_uri"].iloc[0].startswith("artifacts/parameter_snapshot/")


def test_prefix_coords_become_instance_columns(variant):
    materialize_variant_tensors(variant)
    df = read_tensor_catalog(variant)
    # site prefix (per-epoch analyzer) -> a concrete site column + epoch.
    wbp = df[df["analyzer"] == "weight_basis_projection"].iloc[0]
    assert wbp["site"] == "mlp_out" and wbp["field"] == "power" and wbp["epoch"] == 0
    # group prefix (cross-epoch analyzer) -> group set, epoch null (epoch is an axis).
    ptraj = df[df["analyzer"] == "parameter_trajectory"].iloc[0]
    assert ptraj["group"] == "group_8" and ptraj["field"] == "projections"
    assert pd.isna(ptraj["epoch"])  # epoch is an array axis here, not an instance key


def test_shape_and_codec_read_from_blob_header(variant):
    materialize_variant_tensors(variant)
    df = read_tensor_catalog(variant)
    we = df[(df["analyzer"] == "parameter_snapshot") & (df["field"] == "W_E")].iloc[0]
    assert list(we["tensor_shape"]) == [3, 4]  # the matrix shape, read from the npy header
    assert we["tensor_dtype"] == "float32"
    assert we["tensor_codec"] == "npz_compressed"  # honest about savez_compressed


def test_declared_dtype_drift_is_a_finding_not_fatal(tmp_path):
    """Bytes are authoritative: a declared/actual dtype mismatch is surfaced, not raised."""
    v = _FakeVariant(tmp_path, "p5_seed1_dseed2", {"prime": 5, "seed": 1, "data_seed": 2})
    # parameter_snapshot declares float32; write the blob as float64.
    _write_npz(
        v.variant_dir / "artifacts" / "parameter_snapshot" / "epoch_00000.npz",
        W_E=np.zeros((3, 4), dtype=np.float64),
    )
    report = materialize_variant_tensors(v)
    assert report.n_descriptors == 1  # catalog still built
    drift = [d for d in report.dtype_drift if d.field == "W_E"]
    assert drift and drift[0].declared == "float32" and drift[0].actual == "float64"
    # the descriptor records the *actual* dtype, so the array still resolves.
    df = read_tensor_catalog(v)
    arr = v.tensor_catalog.resolve(df)
    assert next(iter(arr.values())).dtype == np.float64


def test_selection_touches_zero_array_bytes(variant):
    """A SQL filter over descriptors returns rows even with the blobs deleted."""
    import duckdb

    materialize_variant_tensors(variant)
    # Delete every blob: only the descriptor index remains.
    for npz in (variant.variant_dir / "artifacts").rglob("*.npz"):
        npz.unlink()

    tdir = paths.tensor_catalog_dir(variant)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW tensors AS SELECT * FROM '{tdir.as_posix()}/*.parquet'")
    selected = con.execute(
        "SELECT * FROM tensors WHERE kind='tensor' AND analyzer='parameter_snapshot' "
        "AND epoch BETWEEN 0 AND 50 ORDER BY field"
    ).df()
    # selection succeeded without any payload; only epoch-0 snapshots match.
    assert sorted(selected["field"]) == ["W_E", "W_in"]
    assert (selected["epoch"] == 0).all()
    # but resolving now fails — the bytes really are gone (proves selection read none).
    with pytest.raises(FileNotFoundError):
        variant.tensor_catalog.resolve(selected)


def test_resolver_opens_each_container_once(variant, monkeypatch):
    materialize_variant_tensors(variant)
    df = read_tensor_catalog(variant)
    epoch0 = df[(df["analyzer"] == "parameter_snapshot") & (df["epoch"] == 0)]
    assert len(epoch0) == 2  # W_E and W_in share one container

    real_load = tensor_catalog.np.load
    calls = {"n": 0}

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(tensor_catalog.np, "load", counting_load)
    out = variant.tensor_catalog.resolve(epoch0)
    assert set(out) == set(epoch0["id"])  # both members materialized
    assert calls["n"] == 1  # one archive open for both members


def test_resolver_verifies_shape_against_bytes(variant):
    materialize_variant_tensors(variant)
    df = read_tensor_catalog(variant)
    import dataclasses

    we = df[(df["analyzer"] == "parameter_snapshot") & (df["field"] == "W_E")].iloc[0]
    # Inject a wrong shape into the descriptor; the resolver must reject the bytes.
    row = tensor_catalog.TensorCatalogRow.from_flat(we.to_dict())
    bad = dataclasses.replace(row, ref=dataclasses.replace(row.ref, shape=(99, 99)))
    with pytest.raises(ValueError, match="shape mismatch"):
        TensorResolver(variant.variant_dir).resolve([bad])


def test_round_trip_reconstructs_original_arrays(variant):
    ref = _seed_blobs(variant)  # re-seed to capture reference arrays
    materialize_variant_tensors(variant)
    df = read_tensor_catalog(variant)
    arrays = variant.tensor_catalog.resolve(df)  # materialize everything selected

    by_id = {r["id"]: r for _, r in df.iterrows()}
    # spot-check one per analyzer against the reference payloads.
    we0 = next(i for i, r in by_id.items() if r["field"] == "W_E" and r["epoch"] == 0)
    np.testing.assert_array_equal(arrays[we0], ref[("parameter_snapshot", "e0_W_E")])
    power = next(i for i, r in by_id.items() if r["field"] == "power")
    np.testing.assert_array_equal(arrays[power], ref[("weight_basis_projection", "mlp_out_power")])
    proj = next(i for i, r in by_id.items() if r["field"] == "projections")
    np.testing.assert_array_equal(arrays[proj], ref[("parameter_trajectory", "group_8__projections")])


def test_tensor_catalog_survives_columnar_rematerialize(variant):
    """The two catalog halves are independent: a columnar wipe preserves tensors."""
    from miscope.warehouse.writer import materialize_variant_columnar

    materialize_variant_tensors(variant)
    before = len(read_tensor_catalog(variant))
    assert before > 0
    materialize_variant_columnar(variant)  # 110-A wipes its own outputs only
    assert len(read_tensor_catalog(variant)) == before
    assert paths.tensor_catalog_dir(variant).exists()
