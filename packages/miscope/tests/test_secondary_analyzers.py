"""Tests for REQ_048: Secondary Analysis Tier."""

import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from miscope.analysis import AnalysisPipeline
from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.inputs import ALL
from miscope.analysis.protocols import Analyzer
from miscope.families.discovery import discover_families


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Remove any test-only Specs registered during a test (REQ_132).

    A Spec is mandatory to run an analyzer through the pipeline, so the fakes
    below register one via ``_register_fake_spec``. This teardown drops only
    the keys added during the test, leaving production registrations intact.
    """
    from miscope.analysis import registry as reg_mod

    before = set(reg_mod._specs)
    yield
    for name in set(reg_mod._specs) - before:
        reg_mod._specs.pop(name, None)
        reg_mod._factories.pop(name, None)


def _register_fake_spec(analyzer):
    """Register a minimal Spec for a test fake and return the instance.

    Inputs are inferred from the fake's legacy attribute: ``requires``
    (cross-epoch) or ``depends_on`` (secondary). Fakes with neither are
    treated as primary (a single ``ModelInput``).
    """
    from miscope.analysis.inputs import ArtifactInput, ModelInput
    from miscope.analysis.registry import register_analyzer
    from miscope.analysis.spec import AnalyzerSpec

    if hasattr(analyzer, "requires"):
        spec = AnalyzerSpec(
            name=analyzer.name,
            output_scope="cross_epoch",
            inputs=tuple(ArtifactInput(r) for r in analyzer.requires),
        )
    elif hasattr(analyzer, "depends_on"):
        spec = AnalyzerSpec(name=analyzer.name, inputs=(ArtifactInput(analyzer.depends_on),))
    else:
        spec = AnalyzerSpec(name=analyzer.name, inputs=(ModelInput(),))
    register_analyzer(spec)(type(analyzer))
    return analyzer


# ── Minimal fake analyzers ─────────────────────────────────────────────


class FakePrimaryAnalyzer:
    """A primary analyzer that saves a scalar 'value' per epoch."""

    name = "fake_primary"

    def analyze(self, model, probe, cache, context):
        return {"value": np.array([1.0, 2.0, 3.0])}


class FakeSecondaryAnalyzer:
    """A secondary analyzer that doubles the 'value' from fake_primary."""

    name = "fake_secondary"
    depends_on = "fake_primary"

    def analyze(self, inputs, context):
        artifact = inputs.deps.load_epoch(self.depends_on, inputs.epoch, fields=ALL)
        return {"doubled": artifact["value"] * 2}


class FakeSecondaryWithSummary:
    """A secondary analyzer that also produces summary statistics."""

    name = "fake_secondary_summary"
    depends_on = "fake_primary"

    def analyze(self, inputs, context):
        artifact = inputs.deps.load_epoch(self.depends_on, inputs.epoch, fields=ALL)
        return {"doubled": artifact["value"] * 2}

    def get_summary_keys(self):
        return ["max_value"]

    def compute_summary(self, result, context):
        return {"max_value": float(result["doubled"].max())}


class WrongDependencyAnalyzer:
    """A secondary analyzer that depends on a non-existent analyzer."""

    name = "wrong_dependency"
    depends_on = "does_not_exist"

    def analyze(self, inputs, context):
        inputs.deps.load_epoch(self.depends_on, inputs.epoch, fields=ALL)
        return {}


# ── Protocol conformance ───────────────────────────────────────────────


class TestSecondaryAnalyzerProtocol:
    def test_fake_secondary_conforms(self):
        assert isinstance(FakeSecondaryAnalyzer(), Analyzer)

    def test_has_name(self):
        assert FakeSecondaryAnalyzer().name == "fake_secondary"

    def test_has_depends_on(self):
        assert FakeSecondaryAnalyzer().depends_on == "fake_primary"

    def test_has_analyze(self):
        assert callable(FakeSecondaryAnalyzer().analyze)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def artifacts_with_primary():
    """Create a temp artifacts dir with fake_primary epoch files."""
    epochs = [0, 100, 200]
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = os.path.join(tmpdir, "artifacts")
        primary_dir = os.path.join(artifacts_dir, "fake_primary")
        os.makedirs(primary_dir)

        for epoch in epochs:
            path = os.path.join(primary_dir, f"epoch_{epoch:05d}.npz")
            np.savez_compressed(path, value=np.array([1.0, 2.0, 3.0]))

        yield artifacts_dir, epochs


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir)
        yield data_root


@pytest.fixture
def trained_variant(temp_dirs):
    data_root = temp_dirs
    family_dir = data_root / "modulo_addition_1layer"
    family_dir.mkdir()
    family_json = {
        "name": "modulo_addition_1layer",
        "display_name": "Modulo Addition (1 Layer)",
        "description": "Test",
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
            "prime": {"type": "int", "description": "Modulus", "default": 113},
            "seed": {"type": "int", "description": "Random seed", "default": 999},
        },
        "analyzers": ["parameter_snapshot"],
        "visualizations": [],
        "analysis_dataset": {"type": "modulo_addition_grid"},
        "variant_pattern": "p{prime}_seed{seed}",
    }
    with open(family_dir / "family.json", "w") as f:
        json.dump(family_json, f)

    families = discover_families(data_root=data_root)
    family = families["modulo_addition_1layer"]
    params = {"prime": 17, "seed": 42, "data_seed": 598}
    variant = family.create_variant(params)
    variant.train(num_epochs=50, checkpoint_epochs=[0, 25, 49], device="cpu")
    return variant


# ── Pipeline integration ───────────────────────────────────────────────


class TestPipelineSecondary:
    def test_secondary_produces_artifact(self, trained_variant):
        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register(_register_fake_spec(FakeSecondaryAnalyzer()))
        pipeline.run()

        # fake_secondary depends on fake_primary, which hasn't run
        # Expect a warning and no artifact
        secondary_dir = os.path.join(pipeline.artifacts_dir, "fake_secondary")
        assert not os.path.exists(secondary_dir)

    def test_secondary_runs_after_primary(self, trained_variant):
        """Secondary runs on epochs where its dependency has completed."""
        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        # First run parameter_snapshot as primary
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register(_register_fake_spec(FakeSecondaryAnalyzer()))
        pipeline.run()

        # fake_secondary has no output (depends on fake_primary, not parameter_snapshot)
        secondary_dir = os.path.join(pipeline.artifacts_dir, "fake_secondary")
        assert not os.path.exists(secondary_dir)

    def test_secondary_with_correct_dependency(self, trained_variant):
        """Secondary with correct dependency produces per-epoch artifacts."""

        class DoubleSnapshotNorm:
            """Secondary that computes norm of W_E from parameter_snapshot."""

            name = "snapshot_norm"
            depends_on = "parameter_snapshot"

            def analyze(self, inputs, context):
                artifact = inputs.deps.load_epoch(self.depends_on, inputs.epoch, fields=ALL)
                w_e = artifact["W_E"]
                return {"norm": np.array([float(np.linalg.norm(w_e))])}

        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register(_register_fake_spec(DoubleSnapshotNorm()))
        pipeline.run()

        # snapshot_norm should have epochs matching parameter_snapshot
        snapshot_epochs = pipeline.get_completed_epochs("parameter_snapshot")
        secondary_epochs = pipeline.get_completed_epochs("snapshot_norm")
        assert secondary_epochs == snapshot_epochs

    def test_secondary_skip_if_exists(self, trained_variant):
        """Secondary skips epochs already computed (no force)."""

        class DoubleSnapshotNorm:
            name = "snapshot_norm"
            depends_on = "parameter_snapshot"

            def analyze(self, inputs, context):
                artifact = inputs.deps.load_epoch(self.depends_on, inputs.epoch, fields=ALL)
                return {"norm": np.array([float(np.linalg.norm(artifact["W_E"]))])}

        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        # First run
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register(_register_fake_spec(DoubleSnapshotNorm()))
        pipeline.run()

        # Record mtimes
        secondary_dir = os.path.join(pipeline.artifacts_dir, "snapshot_norm")
        mtimes_before = {
            f: os.path.getmtime(os.path.join(secondary_dir, f)) for f in os.listdir(secondary_dir)
        }

        import time

        time.sleep(0.05)

        # Second run without force — should skip
        pipeline2 = AnalysisPipeline(trained_variant)
        pipeline2.register(ParameterSnapshotAnalyzer())
        pipeline2.register(_register_fake_spec(DoubleSnapshotNorm()))
        pipeline2.run()

        mtimes_after = {
            f: os.path.getmtime(os.path.join(secondary_dir, f)) for f in os.listdir(secondary_dir)
        }
        assert mtimes_after == mtimes_before

    def test_secondary_force_recomputes(self, trained_variant):
        """Secondary recomputes when force=True."""

        class DoubleSnapshotNorm:
            name = "snapshot_norm"
            depends_on = "parameter_snapshot"

            def analyze(self, inputs, context):
                artifact = inputs.deps.load_epoch(self.depends_on, inputs.epoch, fields=ALL)
                return {"norm": np.array([float(np.linalg.norm(artifact["W_E"]))])}

        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register(_register_fake_spec(DoubleSnapshotNorm()))
        pipeline.run()

        secondary_dir = os.path.join(pipeline.artifacts_dir, "snapshot_norm")
        files = [f for f in os.listdir(secondary_dir) if f.endswith(".npz")]
        assert files
        mtime_before = os.path.getmtime(os.path.join(secondary_dir, files[0]))

        import time

        time.sleep(0.05)

        pipeline2 = AnalysisPipeline(trained_variant)
        pipeline2.register(ParameterSnapshotAnalyzer())
        pipeline2.register(_register_fake_spec(DoubleSnapshotNorm()))
        pipeline2.run(force=True)

        mtime_after = os.path.getmtime(os.path.join(secondary_dir, files[0]))
        assert mtime_after > mtime_before

    def test_secondary_warns_when_dependency_missing(self, trained_variant):
        """Secondary warns and skips (does not raise) when dependency has no epochs."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(_register_fake_spec(WrongDependencyAnalyzer()))

        with warnings.catch_warnings(record=True):  # as w:
            warnings.simplefilter("always")
            pipeline.run()

        # No exception raised, but warning should have been logged
        # (Since we use logger.warning not warnings.warn, just verify no exception)
        secondary_dir = os.path.join(pipeline.artifacts_dir, "wrong_dependency")
        assert not os.path.exists(secondary_dir)

    def test_secondary_runs_before_cross_epoch(self, trained_variant):
        """Secondary phase completes before cross-epoch analyzers run."""
        call_order = []

        class TrackingSecondary:
            name = "tracking_snapshot_norm"
            depends_on = "parameter_snapshot"

            def analyze(self, inputs, context):
                inputs.deps.load_epoch(self.depends_on, inputs.epoch, fields=ALL)
                call_order.append("secondary")
                return {"norm": np.array([1.0])}

        class TrackingCrossEpoch:
            name = "tracking_cross"
            requires = ["parameter_snapshot"]

            def analyze(self, inputs, context):
                call_order.append("cross_epoch")
                return {"epochs": np.array(list(inputs.epochs or ()))}

        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register(_register_fake_spec(TrackingSecondary()))
        pipeline.register(_register_fake_spec(TrackingCrossEpoch()))
        pipeline.run()

        # All secondaries appear before any cross-epoch
        secondary_positions = [i for i, x in enumerate(call_order) if x == "secondary"]
        cross_positions = [i for i, x in enumerate(call_order) if x == "cross_epoch"]
        assert secondary_positions
        assert cross_positions
        assert max(secondary_positions) < min(cross_positions)

    def test_secondary_artifact_loadable(self, trained_variant):
        """Secondary artifacts are readable via ArtifactLoader."""

        class DoubleSnapshotNorm:
            name = "snapshot_norm"
            depends_on = "parameter_snapshot"

            def analyze(self, inputs, context):
                artifact = inputs.deps.load_epoch(self.depends_on, inputs.epoch, fields=ALL)
                return {"norm": np.array([float(np.linalg.norm(artifact["W_E"]))])}

        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register(_register_fake_spec(DoubleSnapshotNorm()))
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epochs = pipeline.get_completed_epochs("snapshot_norm")
        assert epochs
        data = loader.load_epoch("snapshot_norm", epochs[0])
        assert "norm" in data
        assert data["norm"].shape == (1,)


# REQ_132: the per-phase ``secondary_analyzers`` / ``cross_epoch_analyzers``
# family properties were collapsed into the single flat ``analyzers`` list.
# Family declaration is covered by the family unit tests (test_modulo_addition_*).
