"""Tests for generation-parameter declarations and run sets (REQ_138, Phase 1).

Covers the parameter vocabulary (specs, bindings, selectors, parameterization
lookup) and the registry load-time well-formedness gate that parallels the
``outputs`` discipline.
"""

from __future__ import annotations

import dataclasses

import pytest

import miscope.registry as reg
from miscope.analysis.parameters import (
    EMPTY_PARAMETERIZATION,
    FieldIndex,
    LiteralBinding,
    Parameterization,
    ParameterSpec,
    Reducer,
    ReferenceBinding,
    binding_key,
)
from miscope.registry._index import RegistryError, RegistryIndex, validate

# ---------------------------------------------------------------------------
# Binding / parameterization vocabulary
# ---------------------------------------------------------------------------


def test_binding_key_distinguishes_run_level_from_analyzer_local():
    run_level = LiteralBinding(name="reference_epoch", value=20000)
    local = LiteralBinding(name="reference_epoch", value=20000, analyzer="parameter_dmd")
    assert binding_key(run_level) == (None, "reference_epoch")
    assert binding_key(local) == ("parameter_dmd", "reference_epoch")
    assert binding_key(run_level) != binding_key(local)


def test_empty_parameterization_is_default():
    assert EMPTY_PARAMETERIZATION.is_empty
    assert EMPTY_PARAMETERIZATION.binding_for("parameter_dmd", "reference_epoch") is None


def test_analyzer_local_binding_overrides_run_level():
    run_level = LiteralBinding(name="reference_epoch", value=10000)
    local = LiteralBinding(name="reference_epoch", value=20000, analyzer="parameter_dmd")
    p = Parameterization(bindings=(run_level, local))
    # parameter_dmd sees the local override; another analyzer sees the run-level value.
    assert p.binding_for("parameter_dmd", "reference_epoch") is local
    assert p.binding_for("neuron_group_pca", "reference_epoch") is run_level


def test_reference_binding_carries_source_and_selector():
    b = ReferenceBinding(
        name="reference_epoch",
        source_analyzer="neuron_grouping",
        selector=Reducer("max_epoch"),
    )
    assert b.source_analyzer == "neuron_grouping"
    assert isinstance(b.selector, Reducer)

    walk = ReferenceBinding(
        name="window_start",
        source_analyzer="activation_dmd",
        selector=FieldIndex("regime_boundaries", 0),
    )
    assert walk.selector == FieldIndex("regime_boundaries", 0)


# ---------------------------------------------------------------------------
# Registry load-time well-formedness gate (parallel to the outputs discipline)
# ---------------------------------------------------------------------------


def _spec_with_parameters(idx: RegistryIndex, params: tuple[ParameterSpec, ...]) -> RegistryIndex:
    """Return an index whose first analyzer carries ``params`` (for failure tests)."""
    patched = dataclasses.replace(idx.analyzers[0], parameters=params)
    return RegistryIndex(analyzers=(patched, *idx.analyzers[1:]), dataviews=idx.dataviews)


def test_real_registry_validates_with_declared_parameters():
    # The bundled registry (which gains reference_epoch params in Phase 3) must load.
    reg.load()


def test_bad_parameter_dtype_fails_load():
    idx = reg.load()
    bad = ParameterSpec("p", dtype="not_a_dtype", scope="analyzer", default=LiteralBinding("p", 1))
    with pytest.raises(RegistryError, match="unknown.*dtype"):
        validate(_spec_with_parameters(idx, (bad,)))


def test_bad_parameter_scope_fails_load():
    idx = reg.load()
    bad = ParameterSpec(
        "p",
        dtype="int64",
        scope="global",
        default=LiteralBinding("p", 1),  # type: ignore[arg-type]
    )
    with pytest.raises(RegistryError, match="unknown scope"):
        validate(_spec_with_parameters(idx, (bad,)))


def test_reference_default_to_unregistered_analyzer_fails_load():
    idx = reg.load()
    bad = ParameterSpec(
        "p",
        dtype="int64",
        scope="analyzer",
        default=ReferenceBinding("p", "ghost_analyzer", Reducer("max_epoch")),
    )
    with pytest.raises(RegistryError, match="not registered"):
        validate(_spec_with_parameters(idx, (bad,)))
