"""Tests for unified input declarations (REQ_121).

Coverage:
- Input declaration types: ModelInput, ArtifactInput frozen-ness and defaults.
- ResolvedInputs container shape.
- Spec derived properties (category / requires / capability flags) from inputs.
- Pipeline dispatch: Specs route to .analyze(inputs, context).
- Analyzer protocol is_instance check.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from miscope.analysis.inputs import (
    ArtifactInput,
    ModelInput,
    ResolvedInputs,
    derive_category,
    derive_needs_activation_cache,
    derive_needs_model_weights,
    derive_required_artifacts,
)
from miscope.analysis.planner import plan_analysis
from miscope.analysis.protocols import Analyzer
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

# ---------------------------------------------------------------------------
# Input declarations
# ---------------------------------------------------------------------------


def test_model_input_defaults():
    mi = ModelInput()
    assert mi.needs_weights is True
    assert mi.needs_cache is True


def test_model_input_weights_only():
    mi = ModelInput(needs_weights=True, needs_cache=False)
    assert mi.needs_weights is True
    assert mi.needs_cache is False


def test_artifact_input_defaults():
    ai = ArtifactInput("parameter_snapshot")
    assert ai.analyzer_name == "parameter_snapshot"


def test_input_types_are_frozen():
    mi = ModelInput()
    ai = ArtifactInput("x")
    for instance in (mi, ai):
        with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError
            instance.needs_weights = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------


def test_derive_required_artifacts_orders_first_seen():
    inputs = (
        ModelInput(),
        ArtifactInput("b"),
        ArtifactInput("a"),
        ArtifactInput("b"),  # duplicate ignored
    )
    assert derive_required_artifacts(inputs) == ("b", "a")


def test_derive_needs_model_weights():
    assert derive_needs_model_weights((ModelInput(needs_weights=True),)) is True
    assert derive_needs_model_weights((ModelInput(needs_weights=False),)) is False
    assert derive_needs_model_weights((ArtifactInput("x"),)) is False


def test_derive_needs_activation_cache():
    assert derive_needs_activation_cache((ModelInput(needs_cache=True),)) is True
    assert derive_needs_activation_cache((ModelInput(needs_cache=False),)) is False
    assert derive_needs_activation_cache(()) is False


def test_derive_category_dispatch():
    assert derive_category((ModelInput(),), "per_epoch") == "primary"
    assert derive_category((ArtifactInput("x"),), "per_epoch") == "secondary"
    assert derive_category((ArtifactInput("x"),), "cross_epoch") == "cross_epoch"
    assert derive_category((), "per_epoch") == "primary"
    assert derive_category((), "cross_epoch") == "cross_epoch"


# ---------------------------------------------------------------------------
# Spec derived properties (all derived from ``inputs`` + ``output_scope``)
# ---------------------------------------------------------------------------


def test_primary_spec_derives_from_inputs():
    spec = AnalyzerSpec(
        name="u",
        output_scope="per_epoch",
        inputs=(ModelInput(needs_cache=False),),
    )
    assert derive_category(spec.inputs, spec.output_scope) == "primary"
    assert spec.requires == ()
    assert spec.requires_model_weights is True
    assert spec.requires_activation_cache is False


def test_secondary_spec_derives_category_from_inputs():
    spec = AnalyzerSpec(
        name="u_sec",
        output_scope="per_epoch",
        inputs=(ArtifactInput("upstream"),),
    )
    assert derive_category(spec.inputs, spec.output_scope) == "secondary"
    assert spec.requires == ("upstream",)


def test_cross_epoch_spec_derives_category():
    spec = AnalyzerSpec(
        name="u_ce",
        output_scope="cross_epoch",
        inputs=(ArtifactInput("upstream"),),
    )
    assert derive_category(spec.inputs, spec.output_scope) == "cross_epoch"


def test_inputless_spec_derives_conservatively():
    spec = AnalyzerSpec(name="x", output_scope="per_epoch")
    assert derive_category(spec.inputs, spec.output_scope) == "primary"
    assert spec.requires == ()
    assert spec.requires_model_weights is False
    assert spec.requires_activation_cache is False


# ---------------------------------------------------------------------------
# Analyzer protocol check
# ---------------------------------------------------------------------------


def test_analyzer_protocol_isinstance():
    class U:
        name = "u"

        def analyze(self, inputs, context):
            return {}

    class NotU:
        name = "n"
        # missing analyze

    assert isinstance(U(), Analyzer)
    assert not isinstance(NotU(), Analyzer)


# ---------------------------------------------------------------------------
# Pipeline dispatch — unified analyzer flow
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_variant():
    import json
    import tempfile

    from miscope.families.discovery import discover_families

    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir)
        family_dir = data_root / "modulo_addition_1layer"
        family_dir.mkdir()
        (family_dir / "variants").mkdir()
        (family_dir / "family.json").write_text(
            json.dumps(
                {
                    "name": "modulo_addition_1layer",
                    "display_name": "Modulo Addition (1 Layer)",
                    "description": "",
                    "architecture": {
                        "n_layers": 1,
                        "n_heads": 4,
                        "d_model": 128,
                        "d_head": 32,
                        "d_mlp": 512,
                        "act_fn": "relu",
                        "normalization_type": None,
                        "n_ctx": 3,
                    },
                    "domain_parameters": {
                        "prime": {"type": "int", "description": "", "default": 113},
                        "seed": {"type": "int", "description": "", "default": 999},
                    },
                    "analyzers": ["dominant_frequencies"],
                    "visualizations": [],
                    "analysis_dataset": {"type": "modulo_addition_grid"},
                    "variant_pattern": "p{prime}_seed{seed}",
                }
            )
        )
        families = discover_families(data_root=data_root)
        family = families["modulo_addition_1layer"]
        variant = family.create_variant({"prime": 17, "seed": 42, "data_seed": 598})
        variant.train(num_epochs=10, checkpoint_epochs=[0, 9], device="cpu")
        yield variant


@pytest.fixture
def registry_snapshot():
    """Snapshot Spec/factory dicts; restore them after the test.

    Tests below register one-off analyzers via the decorator; this fixture
    keeps those registrations from leaking to other test files (which
    expect the canonical default Specs unchanged).
    """
    from miscope.analysis import registry as reg_mod

    saved_specs = dict(reg_mod._specs)
    saved_factories = dict(reg_mod._factories)
    yield
    reg_mod._specs.clear()
    reg_mod._specs.update(saved_specs)
    reg_mod._factories.clear()
    reg_mod._factories.update(saved_factories)


def test_pipeline_unified_per_epoch_dispatch(trained_variant, registry_snapshot):
    """Unified Spec → analyzer receives ResolvedInputs, not ActivationContext."""
    from miscope.analysis import AnalysisPipeline

    spec = AnalyzerSpec(
        name="unified_primary_test",
        output_scope="per_epoch",
        inputs=(ModelInput(needs_cache=False),),
    )

    received_inputs: list[ResolvedInputs] = []

    @register_analyzer(spec)
    class _UnifiedPrimary:
        name = "unified_primary_test"

        def analyze(self, inputs, context):
            received_inputs.append(inputs)
            return {"sentinel": np.array([1.0], dtype=np.float32)}

    plan = plan_analysis(trained_variant, [spec])
    AnalysisPipeline(trained_variant).run(plan=plan)
    assert len(received_inputs) > 0
    sample = received_inputs[0]
    assert isinstance(sample, ResolvedInputs)
    assert sample.epoch is not None
    assert sample.model is not None  # ModelInput(needs_weights=True default)
    assert sample.cache is None  # needs_cache=False
    assert sample.logits is None


def test_pipeline_unified_cross_epoch_dispatch(trained_variant, registry_snapshot):
    """Unified cross-epoch Spec → analyzer receives ResolvedInputs with
    epochs set and the declared upstream reachable lazily via deps."""
    from miscope.analysis import AnalysisPipeline

    # First, register and run a primary analyzer the cross-epoch depends on.
    prim_spec = AnalyzerSpec(
        name="u_primary_for_ce",
        output_scope="per_epoch",
        inputs=(ModelInput(needs_cache=False),),
    )
    ce_spec = AnalyzerSpec(
        name="u_cross_epoch_test",
        output_scope="cross_epoch",
        inputs=(ArtifactInput("u_primary_for_ce"),),
    )

    received: list[ResolvedInputs] = []

    @register_analyzer(prim_spec)
    class _PrimForCE:
        name = "u_primary_for_ce"

        def analyze(self, inputs, context):
            return {"data": np.ones((3,), dtype=np.float32)}

    @register_analyzer(ce_spec)
    class _UnifiedCE:
        name = "u_cross_epoch_test"

        def analyze(self, inputs, context):
            received.append(inputs)
            return {"summary": np.array([1.0], dtype=np.float32)}

    plan = plan_analysis(trained_variant, [prim_spec, ce_spec])
    AnalysisPipeline(trained_variant).run(plan=plan)
    assert len(received) == 1
    sample = received[0]
    assert sample.epoch is None
    assert sample.epochs is not None
    assert sample.deps is not None
    # The declared upstream is reachable lazily through deps (no eager dict).
    stacked = sample.deps.load_stack("u_primary_for_ce", fields=["data"])
    np.testing.assert_array_equal(stacked["data"][0], np.ones((3,), dtype=np.float32))


# Phase 2C: the legacy dispatch path was retired in REQ_121, so
# ``test_pipeline_legacy_still_dispatches_via_ctx`` no longer applies —
# every analyzer goes through the unified path.


def test_materialize_all_epochs_survives_shape_mismatch(
    trained_variant, registry_snapshot, tmp_path
):
    """Regression: a cross-epoch analyzer that declares an upstream whose
    per-epoch artifacts vary in shape across epochs (e.g., a legacy partial-run
    state) must still run. Post-REQ_128 the pipeline pre-materializes nothing,
    so a ragged upstream can never crash it during input construction — the
    analyzer reaches its body and reads (or not) lazily through ``deps``.
    """
    import os

    from miscope.analysis import AnalysisPipeline

    # Write a "ragged" upstream artifact: two per-epoch files with different
    # shapes for the same key. ``loader.load`` would np.stack and crash.
    ragged_name = "ragged_upstream_test"
    ragged_dir = os.path.join(trained_variant.artifacts_dir, ragged_name)
    os.makedirs(ragged_dir, exist_ok=True)
    np.savez(
        os.path.join(ragged_dir, "epoch_00000.npz"),
        data=np.ones((3,), dtype=np.float32),
    )
    np.savez(
        os.path.join(ragged_dir, "epoch_00009.npz"),
        data=np.ones((5,), dtype=np.float32),  # different shape
    )

    ce_spec = AnalyzerSpec(
        name="downstream_test",
        output_scope="cross_epoch",
        inputs=(ArtifactInput(ragged_name),),
    )
    received: list[ResolvedInputs] = []

    @register_analyzer(ce_spec)
    class _Downstream:
        name = "downstream_test"

        def analyze(self, inputs, context):
            received.append(inputs)
            return {"ok": np.array([1])}

    plan = plan_analysis(trained_variant, [ce_spec])
    # Should NOT raise even though the upstream stack would fail.
    AnalysisPipeline(trained_variant).run(plan=plan)
    assert len(received) == 1
    # The analyzer reached its body with a scoped deps accessor; the ragged
    # upstream was never eagerly stacked, so nothing crashed.
    assert received[0].deps is not None


# ---------------------------------------------------------------------------
# Plan still surfaces capability flags from unified inputs
# ---------------------------------------------------------------------------


def _make_variant(tmp_path: Path, checkpoints: list[int]) -> MagicMock:
    variant = MagicMock()
    variant.name = "v"
    variant.artifacts_dir = str(tmp_path / "artifacts")
    variant.get_available_checkpoints.return_value = list(checkpoints)
    return variant


def test_plan_from_unified_spec_surfaces_cache_flag(tmp_path):
    spec = AnalyzerSpec(
        name="u",
        output_scope="per_epoch",
        inputs=(ModelInput(needs_cache=False),),
    )
    variant = _make_variant(tmp_path, [0, 100])
    plan = plan_analysis(variant, [spec])
    assert plan.per_epoch[0].requires_activation_cache is False
    assert plan.needs_activation_cache is False
