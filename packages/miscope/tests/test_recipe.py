"""Tests for recipe projection, signature, and resolution (REQ_138, Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from miscope.analysis.parameters import (
    EMPTY_PARAMETERIZATION,
    FieldIndex,
    LiteralBinding,
    Parameterization,
    ParameterSpec,
    Reducer,
    ReferenceBinding,
)
from miscope.analysis.recipe import (
    Recipe,
    RecipeResolutionError,
    RecipeResolver,
    project_recipe,
    recipe_signature,
)
from miscope.analysis.spec import AnalyzerSpec

# ---------------------------------------------------------------------------
# A tiny synthetic analyzer graph:  grouping -> dmd (declares reference_epoch)
#                                   grouping -> downstream (reads dmd)
# ---------------------------------------------------------------------------

REF_EPOCH = ParameterSpec(
    "reference_epoch",
    dtype="int64",
    scope="analyzer",
    default=ReferenceBinding("reference_epoch", "grouping", Reducer("max_epoch")),
)


def _specs() -> dict[str, AnalyzerSpec]:
    from miscope.analysis.inputs import ArtifactInput, ModelInput

    grouping = AnalyzerSpec(name="grouping", inputs=(ModelInput(),))
    dmd = AnalyzerSpec(
        name="dmd",
        output_scope="cross_epoch",
        inputs=(ArtifactInput("grouping"),),
        parameters=(REF_EPOCH,),
    )
    downstream = AnalyzerSpec(
        name="downstream", output_scope="cross_epoch", inputs=(ArtifactInput("dmd"),)
    )
    unrelated = AnalyzerSpec(name="unrelated", inputs=(ModelInput(),))
    return {s.name: s for s in (grouping, dmd, downstream, unrelated)}


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_empty_parameterization_projects_to_empty_recipe():
    specs = _specs()
    for name in specs:
        assert project_recipe(name, EMPTY_PARAMETERIZATION, specs).is_empty


def test_local_binding_lands_only_on_declaring_analyzer_and_its_dependents():
    specs = _specs()
    pin = Parameterization(bindings=(LiteralBinding("reference_epoch", 20000, analyzer="dmd"),))
    # dmd declares it -> recipe; downstream reads dmd (closure) -> recipe (transitive);
    # grouping is upstream of dmd, not downstream -> empty; unrelated -> empty.
    assert not project_recipe("dmd", pin, specs).is_empty
    assert not project_recipe("downstream", pin, specs).is_empty
    assert project_recipe("grouping", pin, specs).is_empty
    assert project_recipe("unrelated", pin, specs).is_empty


def test_transitive_recipe_matches_upstream_recipe():
    specs = _specs()
    pin = Parameterization(bindings=(LiteralBinding("reference_epoch", 20000, analyzer="dmd"),))
    # downstream's recipe includes dmd's binding -> same signature as dmd's recipe.
    assert (
        project_recipe("downstream", pin, specs).signature()
        == project_recipe("dmd", pin, specs).signature()
    )


# ---------------------------------------------------------------------------
# Signature  (address by binding spec, stable)
# ---------------------------------------------------------------------------


def test_signature_empty_recipe_is_blank():
    assert recipe_signature(Recipe()) == ""


def test_signature_distinguishes_pinned_values():
    a = Recipe((LiteralBinding("reference_epoch", 20000, analyzer="dmd"),))
    b = Recipe((LiteralBinding("reference_epoch", 30000, analyzer="dmd"),))
    assert a.signature() != b.signature()


def test_signature_is_order_independent_and_stable():
    b1 = LiteralBinding("x", 1, analyzer="dmd")
    b2 = LiteralBinding("y", 2, analyzer="dmd")
    assert recipe_signature(Recipe((b1, b2))) == recipe_signature(Recipe((b2, b1)))


def test_signature_addresses_by_spec_not_resolved_value():
    # Two run sets using the same reference binding get one address; the resolved
    # value (which may move with upstream) is recorded in provenance, not the path.
    ref = ReferenceBinding("reference_epoch", "grouping", Reducer("max_epoch"), analyzer="dmd")
    assert recipe_signature(Recipe((ref,))) == recipe_signature(Recipe((ref,)))
    other = ReferenceBinding("reference_epoch", "grouping", Reducer("min_epoch"), analyzer="dmd")
    assert recipe_signature(Recipe((ref,))) != recipe_signature(Recipe((other,)))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class _FakeLoader:
    def __init__(self, epochs=None, cross=None):
        self._epochs = epochs or {}
        self._cross = cross or {}

    def get_epochs(self, name):
        return self._epochs.get(name, [])

    def load_cross_epoch(self, name, fields=None):
        if name not in self._cross:
            raise FileNotFoundError(name)
        return self._cross[name]


def test_resolve_literal_passes_through():
    r = RecipeResolver(_FakeLoader())
    assert r.resolve(LiteralBinding("reference_epoch", 20000)) == 20000


def test_resolve_reducer_reads_inventory():
    loader = _FakeLoader(epochs={"grouping": [0, 5000, 50000]})
    r = RecipeResolver(loader)
    assert r.resolve(ReferenceBinding("e", "grouping", Reducer("max_epoch"))) == 50000
    assert r.resolve(ReferenceBinding("e", "grouping", Reducer("min_epoch"))) == 0


def test_resolve_reducer_empty_inventory_raises():
    r = RecipeResolver(_FakeLoader(epochs={"grouping": []}))
    with pytest.raises(RecipeResolutionError, match="no epochs"):
        r.resolve(ReferenceBinding("e", "grouping", Reducer("max_epoch")))


def test_resolve_field_index_reads_cross_epoch_field():
    loader = _FakeLoader(cross={"dmd": {"regime_boundaries": np.array([8000, 16000])}})
    r = RecipeResolver(loader)
    val = r.resolve(ReferenceBinding("w", "dmd", FieldIndex("regime_boundaries", 1)))
    assert val == 16000
