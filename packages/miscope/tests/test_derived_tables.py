"""DerivedTable primitive + registry wiring (REQ_141, CoS #1).

A derived table is a first-class output producer alongside an analyzer: it carries
a declared output schema + version, the registry reports it as the producer of its
fields, and ``registry.load()`` rejects one missing its schema — exactly as for an
analyzer (the codified-analysis invariant). These tests pin that contract.
"""

from __future__ import annotations

import pytest

import miscope.registry as reg
from miscope.analysis.derived_table import (
    DerivedTableRegistry,
    DerivedTableSpec,
    register_derived_table,
)
from miscope.analysis.output_schema import Coord, FieldKind, OutputField
from miscope.analysis.spec import SchemaProducer
from miscope.registry._index import (
    RegistryError,
    RegistryIndex,
    build_index,
    lookup_field,
    validate,
)


def _committed_counts_table() -> DerivedTableSpec:
    """A fixture derived table mirroring the REQ_141 bucket-2 committed-count slice."""
    return DerivedTableSpec(
        name="committed_counts_test",
        query=(
            "SELECT epoch, frequency, COUNT(*) AS committed_count "
            "FROM neuron_frequency_attribution WHERE frac_explained >= 0.70 "
            "GROUP BY epoch, frequency"
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
        version=1,
        materialized=True,
    )


def test_derived_table_satisfies_schema_producer():
    """DerivedTableSpec is structurally a SchemaProducer (the shared registry seam)."""
    spec = _committed_counts_table()
    assert isinstance(spec, SchemaProducer)
    assert spec.name == "committed_counts_test"
    assert spec.version == 1
    assert spec.output_names() == ("committed_count",)


def test_field_reverse_lookup_reports_derived_producer():
    """registry.field reports a derived table as producer, with its keying coords."""
    spec = _committed_counts_table()
    idx = RegistryIndex(analyzers=(), dataviews=(), derived=(spec,))
    info = lookup_field(idx, "committed_count")

    assert info.derived_producers, "expected a derived producer for the field"
    name, field = info.derived_producers[0]
    assert name == "committed_counts_test"
    assert field.kind is FieldKind.COLUMNAR
    assert field.coord_names == ("variant", "epoch", "frequency")
    # repr names the derived table as producer (parity with analyzer producers).
    assert "derived table 'committed_counts_test'" in repr(info)


def test_validate_accepts_well_formed_derived_table():
    spec = _committed_counts_table()
    validate(RegistryIndex(analyzers=(), dataviews=(), derived=(spec,)))  # no raise


def test_validate_rejects_derived_table_missing_schema():
    """A derived table with no declared outputs fails load, same as an analyzer."""
    bare = DerivedTableSpec(
        name="schemaless", query="SELECT 1", input_tables=("some_table",), outputs=()
    )
    with pytest.raises(RegistryError, match="declares no output fields"):
        validate(RegistryIndex(analyzers=(), dataviews=(), derived=(bare,)))


def test_validate_rejects_derived_table_missing_input_tables():
    """A derived table must declare the warehouse tables it queries."""
    no_inputs = DerivedTableSpec(
        name="no_inputs",
        query="SELECT 1",
        input_tables=(),
        outputs=(OutputField.columnar("x", "int64", (Coord.VARIANT,), "a value"),),
    )
    with pytest.raises(RegistryError, match="no input_tables"):
        validate(RegistryIndex(analyzers=(), dataviews=(), derived=(no_inputs,)))


def test_validate_rejects_derived_table_bad_dtype():
    bad = DerivedTableSpec(
        name="bad_dtype",
        query="SELECT 1",
        input_tables=("t",),
        outputs=(OutputField.columnar("x", "not_a_dtype", (Coord.VARIANT,), "bad"),),
    )
    with pytest.raises(RegistryError, match="unknown dtype"):
        validate(RegistryIndex(analyzers=(), dataviews=(), derived=(bad,)))


def test_registered_derived_table_is_collected_and_searchable():
    """A registered derived table is collected by build_index and found by search."""
    DerivedTableRegistry.clear()
    try:
        register_derived_table(_committed_counts_table())
        idx = build_index()
        assert "committed_counts_test" in {d.name for d in idx.derived}
        validate(idx)  # the real registry + our fixture still loads clean

        info = lookup_field(idx, "committed_count")
        assert info.derived_producers[0][0] == "committed_counts_test"
    finally:
        DerivedTableRegistry.clear()


def test_real_registry_still_loads():
    """The CI load gate passes with the derived-table surface wired in (none yet)."""
    DerivedTableRegistry.clear()
    idx = reg.load()
    assert isinstance(idx.derived, tuple)
    # derived() surface is constructible (empty until the REQ_141 slice lands).
    assert list(reg.derived().columns)[:2] == ["derived_table", "version"]
