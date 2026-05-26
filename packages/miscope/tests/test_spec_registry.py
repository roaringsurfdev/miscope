"""Tests for REQ_120: AnalyzerSpec + Registry + Planner/Pipeline integration.

CoS coverage:
- Spec dataclass: shape, defaults, frozen-ness.
- Registry: @register_analyzer decorator, query API, legacy-API back-compat,
  Spec/class name mismatch detection.
- Planner: accepts Specs, surfaces capability flags, computes
  transitive_prerequisites, Plan.needs_activation_cache aggregate.
- Pipeline: skips run_with_cache when Plan.needs_activation_cache is False;
  Spec-only analyzers in a Plan get instantiated from Registry; legacy
  Analyzer instances still default to cache-on.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from miscope.analysis.planner import plan_analysis
from miscope.analysis.registry import AnalyzerRegistry, register_analyzer
from miscope.analysis.spec import AnalyzerSpec

# ---------------------------------------------------------------------------
# AnalyzerSpec
# ---------------------------------------------------------------------------


def test_spec_defaults():
    spec = AnalyzerSpec(name="foo", category="primary")
    assert spec.name == "foo"
    assert spec.category == "primary"
    assert spec.effective_requires == ()
    assert spec.effective_requires_model_weights is True
    assert spec.effective_requires_activation_cache is True
    assert spec.required_hooks == ()
    assert spec.produces_summary is False


def test_spec_is_frozen():
    spec = AnalyzerSpec(name="foo", category="primary")
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
    spec = AnalyzerSpec(name="foo", category="primary")

    @register_analyzer(spec)
    class FooAnalyzer:
        name = "foo"

        def analyze(self, inputs, context):
            return {}

    assert fresh_registry.get_spec("foo") is spec
    assert isinstance(fresh_registry.create("foo"), FooAnalyzer)


def test_decorator_rejects_name_mismatch(fresh_registry):
    spec = AnalyzerSpec(name="foo", category="primary")
    with pytest.raises(ValueError, match="name mismatch"):

        @register_analyzer(spec)
        class WrongName:
            name = "bar"

            def analyze(self, inputs, context):
                return {}


def test_list_specs_by_category(fresh_registry):
    @register_analyzer(AnalyzerSpec(name="p1", category="primary"))
    class P1:
        name = "p1"

        def analyze(self, inputs, context):
            return {}

    @register_analyzer(AnalyzerSpec(name="c1", category="cross_epoch"))
    class C1:
        name = "c1"
        requires = []

        def analyze_across_epochs(self, *args):
            return {}

    primaries = [s.name for s in fresh_registry.list_specs_by_category("primary")]
    cross = [s.name for s in fresh_registry.list_specs_by_category("cross_epoch")]
    assert "p1" in primaries and "c1" not in primaries
    assert "c1" in cross and "p1" not in cross


def test_legacy_register_synthesizes_default_spec(fresh_registry):
    class LegacyAnalyzer:
        name = "legacy"

        def analyze(self, inputs, context):
            return {}

    fresh_registry.register(LegacyAnalyzer)
    spec = fresh_registry.get_spec("legacy")
    # Default Spec: conservative — assume weights + cache
    assert spec.category == "primary"
    assert spec.effective_requires_model_weights is True
    assert spec.effective_requires_activation_cache is True


def test_legacy_register_secondary_infers_requires_from_depends_on(fresh_registry):
    class LegacySecondary:
        name = "leg_sec"
        depends_on = "leg_prim"

        def analyze(self, artifact, ctx):
            return {}

    fresh_registry.register_secondary(LegacySecondary)
    spec = fresh_registry.get_spec("leg_sec")
    assert spec.category == "secondary"
    assert spec.effective_requires == ("leg_prim",)


def test_legacy_register_cross_epoch_infers_requires(fresh_registry):
    class LegacyCross:
        name = "leg_cross"
        requires = ["upstream_a", "upstream_b"]

        def analyze_across_epochs(self, *args):
            return {}

    fresh_registry.register_cross_epoch(LegacyCross)
    spec = fresh_registry.get_spec("leg_cross")
    assert spec.category == "cross_epoch"
    assert spec.effective_requires == ("upstream_a", "upstream_b")


def test_decorator_wins_over_subsequent_legacy_register(fresh_registry):
    """Decorator-registered Spec is preserved when legacy register() runs after."""
    explicit = AnalyzerSpec(
        name="dual",
        category="primary",
        requires_activation_cache=False,
    )

    @register_analyzer(explicit)
    class Dual:
        name = "dual"

        def analyze(self, inputs, context):
            return {}

    fresh_registry.register(Dual)  # legacy call — should be a no-op
    assert fresh_registry.get_spec("dual") is explicit
    assert fresh_registry.get_spec("dual").requires_activation_cache is False


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
        category="primary",
        requires_model_weights=True,
        requires_activation_cache=False,
    )

    plan = plan_analysis(variant, [spec])
    item = plan.per_epoch[0]
    assert item.analyzer_name == "weights_only"
    assert item.requires_activation_cache is False
    assert item.requires_model_weights is True


def test_plan_with_instance_leaves_capability_flags_none(tmp_path):
    """Legacy Analyzer instances (no Spec) yield ``None`` flags, which the
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
    weights_only = AnalyzerSpec(name="a", category="primary", requires_activation_cache=False)
    cache_reader = AnalyzerSpec(name="b", category="primary", requires_activation_cache=True)

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

    @register_analyzer(AnalyzerSpec(name="prim", category="primary"))
    class Prim:
        name = "prim"

        def analyze(self, inputs, context):
            return {}

    variant = _make_variant(tmp_path, [0, 100])
    cross_spec = AnalyzerSpec(
        name="cross",
        category="cross_epoch",
        requires=("prim",),
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
        category="cross_epoch",
        requires=("ghost",),
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
        category="primary",
        requires_activation_cache=False,
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
        category="primary",
        requires_activation_cache=True,
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
        category="primary",
        requires_activation_cache=False,
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


def test_spec_category_matches_protocol_class():
    """REQ_120 CoS audit: each Spec's category corresponds to the analyzer
    class's actual protocol (isinstance check on a freshly-instantiated analyzer)."""
    from miscope.analysis.analyzers.registry import register_default_analyzers
    from miscope.analysis.protocols import (
        Analyzer,
        CrossEpochAnalyzer,
        SecondaryAnalyzer,
    )

    register_default_analyzers()

    mismatches = []
    for spec in AnalyzerRegistry.list_specs():
        analyzer = AnalyzerRegistry.create(spec.name)

        if spec.category == "cross_epoch":
            ok = isinstance(analyzer, CrossEpochAnalyzer)
        elif spec.category == "secondary":
            ok = isinstance(analyzer, SecondaryAnalyzer)
        else:  # primary
            ok = isinstance(analyzer, Analyzer)
        if not ok:
            mismatches.append(
                f"{spec.name}: declared {spec.category!r}, instance fails protocol check"
            )
    assert mismatches == [], "Spec category ↔ protocol mismatches:\n" + "\n".join(mismatches)


def test_secondary_spec_requires_matches_depends_on():
    """Secondary Specs must list their dependency under ``requires`` (single item)
    and that item must match the analyzer's ``depends_on`` attribute."""
    from miscope.analysis.analyzers.registry import register_default_analyzers

    register_default_analyzers()

    mismatches = []
    for spec in AnalyzerRegistry.list_specs_by_category("secondary"):
        analyzer = AnalyzerRegistry.create(spec.name)
        if len(spec.effective_requires) != 1:
            mismatches.append(
                f"{spec.name}: secondary Spec.requires has {len(spec.effective_requires)} items "
                f"(expected 1)"
            )
            continue
        if spec.effective_requires[0] != analyzer.depends_on:
            mismatches.append(
                f"{spec.name}: Spec.requires={spec.effective_requires[0]!r} != "
                f"depends_on={analyzer.depends_on!r}"
            )
    assert mismatches == [], "\n".join(mismatches)


def test_cross_epoch_spec_requires_matches_class_requires():
    """Cross-epoch Specs' ``requires`` tuple must match the analyzer class's
    ``requires`` attribute (as a list)."""
    from miscope.analysis.analyzers.registry import register_default_analyzers

    register_default_analyzers()

    mismatches = []
    for spec in AnalyzerRegistry.list_specs_by_category("cross_epoch"):
        analyzer = AnalyzerRegistry.create(spec.name)
        class_requires = tuple(getattr(analyzer, "requires", ()) or ())
        if class_requires != spec.effective_requires:
            mismatches.append(
                f"{spec.name}: class.requires={class_requires} != Spec.requires={spec.effective_requires}"
            )
    assert mismatches == [], "\n".join(mismatches)
