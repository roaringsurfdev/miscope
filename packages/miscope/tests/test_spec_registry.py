"""Tests for REQ_120: AnalyzerSpec + Registry + Planner/Pipeline integration.

CoS coverage:
- Spec dataclass: shape, defaults, frozen-ness, derived properties.
- Registry: @register_analyzer decorator, query API, Spec/class name
  mismatch detection.
- Planner: accepts Specs, surfaces capability flags, computes
  transitive_prerequisites, Plan.needs_activation_cache aggregate.
- Pipeline: skips run_with_cache when Plan.needs_activation_cache is False;
  Spec-only analyzers in a Plan get instantiated from Registry; spec-less
  Analyzer instances still default to cache-on.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from miscope.analysis.inputs import ArtifactInput, ModelInput, derive_category
from miscope.analysis.planner import plan_analysis
from miscope.analysis.registry import AnalyzerRegistry, register_analyzer
from miscope.analysis.spec import AnalyzerSpec

# ---------------------------------------------------------------------------
# AnalyzerSpec
# ---------------------------------------------------------------------------


def test_spec_defaults():
    """A bare Spec declares no inputs, so capability flags derive to False."""
    spec = AnalyzerSpec(name="foo")
    assert spec.name == "foo"
    assert derive_category(spec.inputs, spec.output_scope) == "primary"
    assert spec.requires == ()
    assert spec.requires_model_weights is False
    assert spec.requires_activation_cache is False
    assert spec.required_hooks == ()
    assert spec.produces_summary is False


def test_spec_derives_flags_from_model_input():
    spec = AnalyzerSpec(name="foo", inputs=(ModelInput(needs_weights=True, needs_cache=True),))
    assert spec.requires_model_weights is True
    assert spec.requires_activation_cache is True


def test_spec_is_frozen():
    spec = AnalyzerSpec(name="foo")
    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError
        spec.name = "bar"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_registry():
    """Snapshot + restore the registry around each test."""
    from miscope.analysis import registry as reg_mod

    saved_specs = dict(reg_mod._specs)
    saved_factories = dict(reg_mod._factories)
    AnalyzerRegistry.clear()
    yield AnalyzerRegistry
    AnalyzerRegistry.clear()
    reg_mod._specs.update(saved_specs)
    reg_mod._factories.update(saved_factories)


# ---------------------------------------------------------------------------
# Registry: decorator + query API
# ---------------------------------------------------------------------------


def test_decorator_registers_spec_and_factory(fresh_registry):
    spec = AnalyzerSpec(name="foo", inputs=(ModelInput(),))

    @register_analyzer(spec)
    class FooAnalyzer:
        name = "foo"

        def analyze(self, inputs, context):
            return {}

    assert fresh_registry.get_spec("foo") is spec
    assert isinstance(fresh_registry.create("foo"), FooAnalyzer)


def test_decorator_rejects_name_mismatch(fresh_registry):
    spec = AnalyzerSpec(name="foo", inputs=(ModelInput(),))
    with pytest.raises(ValueError, match="name mismatch"):

        @register_analyzer(spec)
        class WrongName:
            name = "bar"

            def analyze(self, inputs, context):
                return {}


def test_specs_route_by_derived_category(fresh_registry):
    """Category is derived from inputs + output_scope, not a query surface (REQ_132)."""

    @register_analyzer(AnalyzerSpec(name="p1", inputs=(ModelInput(),)))
    class P1:
        name = "p1"

        def analyze(self, inputs, context):
            return {}

    @register_analyzer(AnalyzerSpec(name="c1", output_scope="cross_epoch"))
    class C1:
        name = "c1"
        requires = []

        def analyze(self, inputs, context):
            return {}

    by_category: dict[str, list[str]] = {}
    for spec in fresh_registry.list_specs():
        cat = derive_category(spec.inputs, spec.output_scope)
        by_category.setdefault(cat, []).append(spec.name)
    assert "p1" in by_category["primary"] and "c1" not in by_category["primary"]
    assert "c1" in by_category["cross_epoch"] and "p1" not in by_category["cross_epoch"]


# ---------------------------------------------------------------------------
# Planner integration: Specs in, capability flags on PlanItem + Plan aggregate
# ---------------------------------------------------------------------------


def _make_variant(tmp_path: Path, checkpoints: list[int]) -> MagicMock:
    variant = MagicMock()
    variant.name = "v"
    variant.artifacts_dir = str(tmp_path / "artifacts")
    variant.get_available_checkpoints.return_value = list(checkpoints)
    return variant


def test_plan_with_spec_carries_capability_flags(tmp_path):
    variant = _make_variant(tmp_path, [0, 100])
    spec = AnalyzerSpec(
        name="weights_only",
        inputs=(ModelInput(needs_weights=True, needs_cache=False),),
    )

    plan = plan_analysis(variant, [spec])
    item = plan.per_epoch[0]
    assert item.analyzer_name == "weights_only"
    assert item.requires_activation_cache is False
    assert item.requires_model_weights is True


def test_plan_with_instance_leaves_capability_flags_none(tmp_path):
    """Spec-less Analyzer instances yield ``None`` flags, which the
    Pipeline treats conservatively as ``True``."""

    class Legacy:
        name = "legacy"

        def analyze(self, inputs, context):
            return {}

    variant = _make_variant(tmp_path, [0])
    plan = plan_analysis(variant, [Legacy()])
    item = plan.per_epoch[0]
    assert item.requires_model_weights is None
    assert item.requires_activation_cache is None


def test_plan_needs_activation_cache_or_aggregate(tmp_path):
    variant = _make_variant(tmp_path, [0])
    weights_only = AnalyzerSpec(name="a", inputs=(ModelInput(needs_cache=False),))
    cache_reader = AnalyzerSpec(name="b", inputs=(ModelInput(needs_cache=True),))

    plan_a = plan_analysis(variant, [weights_only])
    assert plan_a.needs_activation_cache is False

    plan_b = plan_analysis(variant, [cache_reader])
    assert plan_b.needs_activation_cache is True

    plan_both = plan_analysis(variant, [weights_only, cache_reader])
    assert plan_both.needs_activation_cache is True  # OR aggregate


def test_plan_needs_activation_cache_defaults_true_for_legacy(tmp_path):
    class Legacy:
        name = "legacy"

        def analyze(self, inputs, context):
            return {}

    variant = _make_variant(tmp_path, [0])
    plan = plan_analysis(variant, [Legacy()])
    # Unknown → treated as True
    assert plan.needs_activation_cache is True


def test_plan_transitive_prerequisites_when_registered(tmp_path, fresh_registry):
    """A cross-epoch item blocked by a Registry-known dep gets surfaced."""
    fresh_registry.clear()

    @register_analyzer(AnalyzerSpec(name="prim", inputs=(ModelInput(),)))
    class Prim:
        name = "prim"

        def analyze(self, inputs, context):
            return {}

    variant = _make_variant(tmp_path, [0, 100])
    cross_spec = AnalyzerSpec(
        name="cross",
        output_scope="cross_epoch",
        inputs=(ArtifactInput("prim"),),
    )
    # Don't pass prim — cross gets blocked
    plan = plan_analysis(variant, [cross_spec])
    assert plan.cross_epoch[0].blocked_by == ("prim",)
    assert plan.transitive_prerequisites == ("prim",)


def test_plan_no_transitive_when_blocker_unregistered(tmp_path, fresh_registry):
    """A blocker with no Spec is omitted from transitive_prerequisites."""
    fresh_registry.clear()

    variant = _make_variant(tmp_path, [0])
    cross_spec = AnalyzerSpec(
        name="cross",
        output_scope="cross_epoch",
        inputs=(ArtifactInput("ghost"),),
    )
    plan = plan_analysis(variant, [cross_spec])
    assert plan.cross_epoch[0].blocked_by == ("ghost",)
    assert plan.transitive_prerequisites == ()


# ---------------------------------------------------------------------------
# Pipeline integration: skip run_with_cache + absorb Spec-only items
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_variant():
    """Reuse the existing trained_variant fixture pattern from test_analysis_pipeline."""
    import json
    import tempfile

    from miscope.families.discovery import discover_families

    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir)
        family_dir = data_root / "modulo_addition_1layer"
        family_dir.mkdir()
        (family_dir / "variants").mkdir()
        family_json = {
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
        (family_dir / "family.json").write_text(json.dumps(family_json))
        families = discover_families(data_root=data_root)
        family = families["modulo_addition_1layer"]
        variant = family.create_variant({"prime": 17, "seed": 42, "data_seed": 598})
        variant.train(num_epochs=10, checkpoint_epochs=[0, 9], device="cpu")
        yield variant


class _WeightsOnlyAnalyzer:
    """Test double: claims weights-only via its Spec, fails on cache access."""

    name = "weights_only_test"

    def analyze(self, inputs, context) -> dict[str, np.ndarray]:
        # Asserts the pipeline did not load the cache.
        assert inputs.cache is None, "expected cache to be None when Spec says no cache"
        assert inputs.logits is None, "expected logits to be None when Spec says no cache"
        assert inputs.model is not None, "model should still be loaded for weights access"
        return {"sentinel": np.array([1.0], dtype=np.float32)}


def test_pipeline_skips_forward_pass_when_plan_says_so(trained_variant):
    """When every active analyzer's Spec says ``requires_activation_cache=False``,
    the pipeline must not call ``model.run_with_cache(probe)``."""
    from miscope.analysis import AnalysisPipeline

    spec = AnalyzerSpec(
        name="weights_only_test",
        inputs=(ModelInput(needs_weights=True, needs_cache=False),),
    )

    @register_analyzer(spec)
    class _Stub(_WeightsOnlyAnalyzer):
        pass

    try:
        plan = plan_analysis(trained_variant, [spec])
        assert plan.needs_activation_cache is False

        pipeline = AnalysisPipeline(trained_variant)

        # Patch run_with_cache to detect if the pipeline calls it.
        from miscope.architectures.hooked_transformer import HookedTransformer

        with patch.object(HookedTransformer, "run_with_cache", autospec=True) as mock_rwc:
            pipeline.run(plan=plan)

        assert mock_rwc.call_count == 0, (
            "Pipeline ran the forward pass even though no analyzer needed cache"
        )
    finally:
        AnalyzerRegistry.clear()
        from miscope.analysis.analyzers.registry import register_default_analyzers

        register_default_analyzers()


def test_pipeline_runs_forward_pass_when_cache_needed(trained_variant):
    """Smoke check: when an analyzer's Spec needs cache, the forward pass runs."""
    from miscope.analysis import AnalysisPipeline

    spec = AnalyzerSpec(
        name="cache_reader_test",
        inputs=(ModelInput(needs_cache=True),),
    )

    @register_analyzer(spec)
    class _CacheReader:
        name = "cache_reader_test"

        def analyze(self, inputs, context) -> dict[str, np.ndarray]:
            assert inputs.cache is not None
            return {"sentinel": np.array([1.0], dtype=np.float32)}

    try:
        plan = plan_analysis(trained_variant, [spec])
        assert plan.needs_activation_cache is True
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.run(plan=plan)  # would raise inside analyzer if cache absent

        # Confirm artifacts were written
        analyzer_dir = Path(trained_variant.artifacts_dir) / "cache_reader_test"
        assert any(
            p.name.startswith("epoch_") and p.suffix == ".npz" for p in analyzer_dir.iterdir()
        )
    finally:
        AnalyzerRegistry.clear()
        from miscope.analysis.analyzers.registry import register_default_analyzers

        register_default_analyzers()


def test_pipeline_absorbs_spec_only_plan_items(trained_variant):
    """When the caller hands the pipeline a Plan built from Specs without
    pre-registering them, the pipeline instantiates them from the Registry
    so the existing per-phase loops can run them."""
    from miscope.analysis import AnalysisPipeline

    spec = AnalyzerSpec(
        name="absorb_test",
        inputs=(ModelInput(needs_weights=False, needs_cache=False),),
    )

    @register_analyzer(spec)
    class _Absorb:
        name = "absorb_test"

        def analyze(self, inputs, context) -> dict[str, np.ndarray]:
            return {"sentinel": np.array([42.0], dtype=np.float32)}

    try:
        plan = plan_analysis(trained_variant, [spec])
        pipeline = AnalysisPipeline(trained_variant)  # nothing registered
        assert len(pipeline._analyzers) == 0
        pipeline.run(plan=plan)
        # Pipeline instantiated absorb_test from the Registry
        assert any(a.name == "absorb_test" for a in pipeline._analyzers)
        artifact_dir = Path(trained_variant.artifacts_dir) / "absorb_test"
        assert artifact_dir.is_dir()
        assert any(p.suffix == ".npz" for p in artifact_dir.iterdir())
    finally:
        AnalyzerRegistry.clear()
        from miscope.analysis.analyzers.registry import register_default_analyzers

        register_default_analyzers()


def test_pipeline_register_requires_spec(trained_variant):
    """REQ_132: a registered Spec is mandatory — registering a spec-less
    analyzer raises rather than silently running on the old conservative path."""
    from miscope.analysis import AnalysisPipeline

    class NoSpec:
        name = "no_spec_analyzer"

        def analyze(self, inputs, context):
            return {}

    pipeline = AnalysisPipeline(trained_variant)
    with pytest.raises(ValueError, match="no registered Spec"):
        pipeline.register(NoSpec())


# ---------------------------------------------------------------------------
# Spec ↔ class consistency audit (REQ_120 CoS: registry consistency)
# ---------------------------------------------------------------------------


def test_every_registered_factory_produces_matching_name():
    """For each Spec in the Registry, calling its factory yields an analyzer
    whose ``.name`` matches the Spec's ``name``. Catches drift."""
    # Ensure default registrations have run.
    from miscope.analysis.analyzers.registry import register_default_analyzers

    register_default_analyzers()

    for spec in AnalyzerRegistry.list_specs():
        analyzer = AnalyzerRegistry.create(spec.name)
        assert analyzer.name == spec.name, (
            f"Spec.name={spec.name!r} but instance.name={analyzer.name!r}"
        )


def test_every_analyzer_module_has_spec():
    """REQ_120 CoS audit: every analyzer module under analyzers/ defines a
    SPEC attribute and registers via the decorator."""
    import importlib
    import pkgutil

    import miscope.analysis.analyzers as analyzers_pkg
    from miscope.analysis.analyzers.registry import register_default_analyzers

    register_default_analyzers()

    missing = []
    for module_info in pkgutil.iter_modules(analyzers_pkg.__path__):
        if module_info.name in {"registry", "__init__"}:
            continue
        full_name = f"{analyzers_pkg.__name__}.{module_info.name}"
        module = importlib.import_module(full_name)
        if not hasattr(module, "SPEC"):
            missing.append(module_info.name)
    assert missing == [], f"Analyzer modules without SPEC: {missing}"


def test_every_registered_analyzer_conforms_to_protocol():
    """Every registered analyzer satisfies the unified ``Analyzer`` protocol."""
    from miscope.analysis.analyzers.registry import register_default_analyzers
    from miscope.analysis.protocols import Analyzer

    register_default_analyzers()

    mismatches = []
    for spec in AnalyzerRegistry.list_specs():
        analyzer = AnalyzerRegistry.create(spec.name)
        if not isinstance(analyzer, Analyzer):
            mismatches.append(f"{spec.name}: instance fails Analyzer protocol check")
    assert mismatches == [], "Analyzer protocol mismatches:\n" + "\n".join(mismatches)


def test_secondary_spec_requires_matches_depends_on():
    """Secondary Specs must list their dependency under ``requires`` (single item)
    and that item must match the analyzer's ``depends_on`` attribute."""
    from miscope.analysis.analyzers.registry import register_default_analyzers

    register_default_analyzers()

    mismatches = []
    secondary_specs = [
        s
        for s in AnalyzerRegistry.list_specs()
        if derive_category(s.inputs, s.output_scope) == "secondary"
    ]
    for spec in secondary_specs:
        analyzer = AnalyzerRegistry.create(spec.name)
        if len(spec.requires) != 1:
            mismatches.append(
                f"{spec.name}: secondary Spec.requires has {len(spec.requires)} items "
                f"(expected 1)"
            )
            continue
        if spec.requires[0] != analyzer.depends_on:
            mismatches.append(
                f"{spec.name}: Spec.requires={spec.requires[0]!r} != "
                f"depends_on={analyzer.depends_on!r}"
            )
    assert mismatches == [], "\n".join(mismatches)


def test_cross_epoch_spec_requires_matches_class_requires():
    """Cross-epoch Specs' ``requires`` tuple must match the analyzer class's
    ``requires`` attribute (as a list)."""
    from miscope.analysis.analyzers.registry import register_default_analyzers

    register_default_analyzers()

    mismatches = []
    cross_epoch_specs = [
        s
        for s in AnalyzerRegistry.list_specs()
        if derive_category(s.inputs, s.output_scope) == "cross_epoch"
    ]
    for spec in cross_epoch_specs:
        analyzer = AnalyzerRegistry.create(spec.name)
        class_requires = tuple(getattr(analyzer, "requires", ()) or ())
        if class_requires != spec.requires:
            mismatches.append(
                f"{spec.name}: class.requires={class_requires} != Spec.requires={spec.requires}"
            )
    assert mismatches == [], "\n".join(mismatches)
