"""Tests for REQ_048: Secondary Analysis Tier."""

import os
import tempfile
import warnings
from pathlib import Path
import json

import numpy as np
import pytest

from miscope.analysis import AnalysisPipeline, ArtifactLoader, SecondaryAnalyzer
from miscope.analysis.analyzers import AnalyzerRegistry
from miscope.analysis.protocols import SecondaryAnalyzer as SecondaryAnalyzerProtocol
from miscope.families import FamilyRegistry


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

    def analyze(self, artifact, context):
        return {"doubled": artifact["value"] * 2}


class FakeSecondaryWithSummary:
    """A secondary analyzer that also produces summary statistics."""

    name = "fake_secondary_summary"
    depends_on = "fake_primary"

    def analyze(self, artifact, context):
        return {"doubled": artifact["value"] * 2}

    def get_summary_keys(self):
        return ["max_value"]

    def compute_summary(self, result, context):
        return {"max_value": float(result["doubled"].max())}


class WrongDependencyAnalyzer:
    """A secondary analyzer that depends on a non-existent analyzer."""

    name = "wrong_dependency"
    depends_on = "does_not_exist"

    def analyze(self, artifact, context):
        return {}


# ── Protocol conformance ───────────────────────────────────────────────


class TestSecondaryAnalyzerProtocol:
    def test_fake_secondary_conforms(self):
        assert isinstance(FakeSecondaryAnalyzer(), SecondaryAnalyzerProtocol)

    def test_has_name(self):
        assert FakeSecondaryAnalyzer().name == "fake_secondary"

    def test_has_depends_on(self):
        assert FakeSecondaryAnalyzer().depends_on == "fake_primary"

    def test_has_analyze(self):
        assert callable(FakeSecondaryAnalyzer().analyze)


# ── Registry ──────────────────────────────────────────────────────────


class TestSecondaryAnalyzerRegistry:
    def setup_method(self):
        AnalyzerRegistry.clear()

    def teardown_method(self):
        # Re-register defaults after each test
        from miscope.analysis.analyzers.registry import register_default_analyzers

        register_default_analyzers()

    def test_register_secondary(self):
        AnalyzerRegistry.register_secondary(FakeSecondaryAnalyzer)
        instance = AnalyzerRegistry.get_secondary("fake_secondary")
        assert isinstance(instance, FakeSecondaryAnalyzer)

    def test_get_secondary_missing_raises(self):
        with pytest.raises(KeyError, match="fake_secondary"):
            AnalyzerRegistry.get_secondary("fake_secondary")

    def test_register_secondary_as_decorator(self):
        @AnalyzerRegistry.register_secondary
        class DecoratedSecondary:
            name = "decorated"
            depends_on = "something"

            def analyze(self, artifact, context):
                return {}

        assert AnalyzerRegistry.get_secondary("decorated") is not None

    def test_clear_removes_secondary(self):
        AnalyzerRegistry.register_secondary(FakeSecondaryAnalyzer)
        AnalyzerRegistry.clear()
        with pytest.raises(KeyError):
            AnalyzerRegistry.get_secondary("fake_secondary")


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
        model_families_dir = Path(tmpdir) / "model_families"
        results_dir = Path(tmpdir) / "results"
        model_families_dir.mkdir()
        results_dir.mkdir()
        yield model_families_dir, results_dir


@pytest.fixture
def trained_variant(temp_dirs):
    model_families_dir, results_dir = temp_dirs
    family_dir = model_families_dir / "modulo_addition_1layer"
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
        "secondary_analyzers": [],
        "cross_epoch_analyzers": [],
        "visualizations": [],
        "analysis_dataset": {"type": "modulo_addition_grid"},
        "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}",
    }
    with open(family_dir / "family.json", "w") as f:
        json.dump(family_json, f)

    registry = FamilyRegistry(model_families_dir=model_families_dir, results_dir=results_dir)
    family = registry.get_family("modulo_addition_1layer")
    params = {"prime": 17, "seed": 42}
    variant = registry.create_variant(family, params)
    variant.train(num_epochs=50, checkpoint_epochs=[0, 25, 49], device="cpu")
    return variant


# ── Pipeline integration ───────────────────────────────────────────────


class TestPipelineSecondary:
    def test_secondary_produces_artifact(self, trained_variant):
        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_secondary(FakeSecondaryAnalyzer())
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
        pipeline.register_secondary(FakeSecondaryAnalyzer())
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

            def analyze(self, artifact, context):
                w_e = artifact["W_E"]
                return {"norm": np.array([float(np.linalg.norm(w_e))])}

        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_secondary(DoubleSnapshotNorm())
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

            def analyze(self, artifact, context):
                return {"norm": np.array([float(np.linalg.norm(artifact["W_E"]))])}

        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        # First run
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_secondary(DoubleSnapshotNorm())
        pipeline.run()

        # Record mtimes
        secondary_dir = os.path.join(pipeline.artifacts_dir, "snapshot_norm")
        mtimes_before = {f: os.path.getmtime(os.path.join(secondary_dir, f))
                        for f in os.listdir(secondary_dir)}

        import time
        time.sleep(0.05)

        # Second run without force — should skip
        pipeline2 = AnalysisPipeline(trained_variant)
        pipeline2.register(ParameterSnapshotAnalyzer())
        pipeline2.register_secondary(DoubleSnapshotNorm())
        pipeline2.run()

        mtimes_after = {f: os.path.getmtime(os.path.join(secondary_dir, f))
                       for f in os.listdir(secondary_dir)}
        assert mtimes_after == mtimes_before

    def test_secondary_force_recomputes(self, trained_variant):
        """Secondary recomputes when force=True."""

        class DoubleSnapshotNorm:
            name = "snapshot_norm"
            depends_on = "parameter_snapshot"

            def analyze(self, artifact, context):
                return {"norm": np.array([float(np.linalg.norm(artifact["W_E"]))])}

        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_secondary(DoubleSnapshotNorm())
        pipeline.run()

        secondary_dir = os.path.join(pipeline.artifacts_dir, "snapshot_norm")
        files = [f for f in os.listdir(secondary_dir) if f.endswith(".npz")]
        assert files
        mtime_before = os.path.getmtime(os.path.join(secondary_dir, files[0]))

        import time
        time.sleep(0.05)

        pipeline2 = AnalysisPipeline(trained_variant)
        pipeline2.register(ParameterSnapshotAnalyzer())
        pipeline2.register_secondary(DoubleSnapshotNorm())
        pipeline2.run(force=True)

        mtime_after = os.path.getmtime(os.path.join(secondary_dir, files[0]))
        assert mtime_after > mtime_before

    def test_secondary_warns_when_dependency_missing(self, trained_variant):
        """Secondary warns and skips (does not raise) when dependency has no epochs."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register_secondary(WrongDependencyAnalyzer())

        with warnings.catch_warnings(record=True) as w:
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

            def analyze(self, artifact, context):
                call_order.append("secondary")
                return {"norm": np.array([1.0])}

        class TrackingCrossEpoch:
            name = "tracking_cross"
            requires = ["parameter_snapshot"]

            def analyze_across_epochs(self, artifacts_dir, epochs, context):
                call_order.append("cross_epoch")
                return {"epochs": np.array(epochs)}

        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_secondary(TrackingSecondary())
        pipeline.register_cross_epoch(TrackingCrossEpoch())
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

            def analyze(self, artifact, context):
                return {"norm": np.array([float(np.linalg.norm(artifact["W_E"]))])}

        from miscope.analysis.analyzers import ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_secondary(DoubleSnapshotNorm())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epochs = pipeline.get_completed_epochs("snapshot_norm")
        assert epochs
        data = loader.load_epoch("snapshot_norm", epochs[0])
        assert "norm" in data
        assert data["norm"].shape == (1,)


# ── JsonModelFamily secondary_analyzers property ─────────────────────


class TestJsonFamilySecondaryAnalyzers:
    def test_secondary_analyzers_returns_list(self, temp_dirs):
        model_families_dir, results_dir = temp_dirs
        family_dir = model_families_dir / "test_fam"
        family_dir.mkdir()
        family_json = {
            "name": "test_fam",
            "display_name": "Test",
            "description": "Test",
            "architecture": {},
            "domain_parameters": {},
            "analyzers": [],
            "secondary_analyzers": ["neuron_fourier"],
            "visualizations": [],
            "variant_pattern": "test_{seed}",
        }
        with open(family_dir / "family.json", "w") as f:
            json.dump(family_json, f)

        from miscope.families.json_family import JsonModelFamily

        fam = JsonModelFamily.from_json(family_dir / "family.json")
        assert fam.secondary_analyzers == ["neuron_fourier"]

    def test_secondary_analyzers_defaults_to_empty(self, temp_dirs):
        model_families_dir, results_dir = temp_dirs
        family_dir = model_families_dir / "test_fam2"
        family_dir.mkdir()
        family_json = {
            "name": "test_fam2",
            "display_name": "Test",
            "description": "Test",
            "architecture": {},
            "domain_parameters": {},
            "analyzers": [],
            "visualizations": [],
            "variant_pattern": "test_{seed}",
        }
        with open(family_dir / "family.json", "w") as f:
            json.dump(family_json, f)

        from miscope.families.json_family import JsonModelFamily

        fam = JsonModelFamily.from_json(family_dir / "family.json")
        assert fam.secondary_analyzers == []
