"""Tests for the discoverability registry (REQ_107).

Covers the registry surface (analyzers/dataviews/field/search), the load-time
contract (every analyzer declares an output schema; dtypes are real), and drift
detection (a DataView source that consumes an undeclared field, or requires a
newer producer version, fails loud).
"""

from __future__ import annotations

import dataclasses
import types

import pytest

import miscope.registry as reg
from miscope.registry._index import RegistryError, RegistryIndex, validate
from miscope.views.dataview_catalog import DataViewSource

# ---------------------------------------------------------------------------
# Load gate + surfaces
# ---------------------------------------------------------------------------


def test_ci_load_gate_succeeds():
    """The mechanical CI signal: import + load() must succeed."""
    idx = reg.load()
    assert len(idx.analyzers) >= 24
    assert len(idx.dataviews) >= 1


def test_every_analyzer_declares_outputs():
    idx = reg.load()
    undeclared = [s.name for s in idx.analyzers if not s.outputs]
    assert undeclared == [], f"analyzers missing output schema: {undeclared}"


def test_analyzers_surface_columns():
    df = reg.analyzers()
    assert list(df.columns) == [
        "analyzer",
        "scope",
        "version",
        "field",
        "kind",
        "coords",
        "dtype",
        "description",
    ]
    # One row per declared field; the columns of the analysis warehouse.
    assert len(df) == sum(len(s.outputs) for s in reg.load().analyzers)
    assert set(df["kind"]) <= {"columnar", "tensor"}


def test_dataviews_surface():
    df = reg.dataviews()
    assert list(df.columns) == ["dataview", "field", "kind", "coords", "sources", "description"]
    assert (df["dataview"] == "neuron_dynamics.raw").any()


def test_search_matches_names_and_descriptions():
    hits = reg.search("frequency")
    assert len(hits) > 0
    assert {"analyzer", "dataview"} & set(hits["surface"])
    # A frequency-tuning analyzer surfaces.
    assert (hits["name"] == "fourier_frequency_quality").any()


def test_search_is_case_insensitive():
    assert len(reg.search("FREQUENCY")) == len(reg.search("frequency"))


def test_field_reverse_lookup_reports_producer_coords_and_consumers():
    # REQ_141: dominant_freq is now produced by the per-epoch
    # neuron_frequency_attribution analyzer; neuron_dynamics consumes it.
    info = reg.field("dominant_freq")
    producers = {an for an, _ in info.producers}
    assert "neuron_frequency_attribution" in producers
    (_, fld) = next(f for f in info.producers if f[0] == "neuron_frequency_attribution")
    assert fld.coord_names == ("variant", "epoch", "neuron")
    assert fld.kind.value == "columnar"
    # Consumers: the neuron_dynamics.raw DataView (field-level) and the
    # neuron_dynamics analyzer (artifact-level — it streams the attribution).
    assert "neuron_dynamics.raw" in info.dataview_consumers
    assert "neuron_dynamics" in info.analyzer_consumers


def test_field_unknown_returns_empty_fieldinfo():
    info = reg.field("definitely_not_a_field")
    assert info.producers == ()
    assert "<not found>" in repr(info)


# ---------------------------------------------------------------------------
# Enforcement + drift detection
# ---------------------------------------------------------------------------


def test_enforcement_empty_outputs_fails_loud():
    idx = reg.load()
    broken_spec = dataclasses.replace(idx.analyzers[0], outputs=())
    broken = RegistryIndex(analyzers=(broken_spec, *idx.analyzers[1:]), dataviews=idx.dataviews)
    with pytest.raises(RegistryError, match="declares no output fields"):
        validate(broken)


def test_drift_undeclared_consumed_field_fails():
    idx = reg.load()
    dv = idx.dataviews[0]
    drifted = dataclasses.replace(
        dv, sources=(DataViewSource("neuron_dynamics", ("not_a_real_field",)),)
    )
    broken = RegistryIndex(analyzers=idx.analyzers, dataviews=(drifted,))
    with pytest.raises(RegistryError, match="does not declare it"):
        validate(broken)


def test_drift_version_mismatch_fails():
    idx = reg.load()
    dv = idx.dataviews[0]
    drifted = dataclasses.replace(
        dv, sources=(DataViewSource("neuron_dynamics", ("epochs",), min_version=99),)
    )
    broken = RegistryIndex(analyzers=idx.analyzers, dataviews=(drifted,))
    with pytest.raises(RegistryError, match=r">= v99"):
        validate(broken)


def test_drift_unregistered_producer_fails():
    idx = reg.load()
    dv = idx.dataviews[0]
    drifted = dataclasses.replace(dv, sources=(DataViewSource("ghost_analyzer", ("x",)),))
    broken = RegistryIndex(analyzers=idx.analyzers, dataviews=(drifted,))
    with pytest.raises(RegistryError, match="not registered"):
        validate(broken)


def test_bad_dtype_fails():
    idx = reg.load()
    spec = idx.analyzers[0]
    bad_field = dataclasses.replace(spec.outputs[0], dtype="not_a_dtype")
    bad_spec = dataclasses.replace(spec, outputs=(bad_field, *spec.outputs[1:]))
    broken = RegistryIndex(analyzers=(bad_spec, *idx.analyzers[1:]), dataviews=idx.dataviews)
    with pytest.raises(RegistryError, match="unknown dtype"):
        validate(broken)


# ---------------------------------------------------------------------------
# Family-owned variant key
# ---------------------------------------------------------------------------


def test_variant_key_columns_is_family_owned():
    family = types.SimpleNamespace(domain_parameters={"prime": {}, "seed": {}, "data_seed": {}})
    assert reg.variant_key_columns(family) == ("variant_id", "prime", "seed", "data_seed")


def test_declared_coords_use_canonical_vocabulary():
    """Every declared coord is from the canonical Coord vocabulary."""
    from miscope.analysis.output_schema import Coord

    valid = set(Coord)
    for spec in reg.load().analyzers:
        for f in spec.outputs:
            assert set(f.coords) <= valid, f"{spec.name}.{f.name} has non-canonical coords"
