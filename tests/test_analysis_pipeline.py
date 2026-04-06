"""Tests for REQ_003_001: Core Infrastructure, REQ_021b Pipeline Refinement,
and REQ_022 Summary Statistics."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from miscope.analysis import AnalysisPipeline, AnalysisRunConfig, Analyzer
from miscope.families import FamilyRegistry


class MockAnalyzer:
    """Mock analyzer for testing pipeline mechanics."""

    def __init__(self, name: str = "mock"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def analyze(self, bundle, probe, context: dict[str, Any]) -> dict[str, np.ndarray]:
        """Mock analysis - returns simple test data."""
        return {"data": np.ones((10,), dtype=np.float32)}


class TestAnalyzerProtocol:
    """Tests for Analyzer protocol."""

    def test_mock_analyzer_conforms_to_protocol(self):
        """MockAnalyzer implements Analyzer protocol."""
        analyzer = MockAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_protocol_requires_name_property(self):
        """Analyzer requires name property."""
        analyzer = MockAnalyzer("test_name")
        assert analyzer.name == "test_name"

    def test_protocol_requires_analyze_method(self):
        """Analyzer requires analyze method."""
        analyzer = MockAnalyzer()
        assert callable(analyzer.analyze)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for model_families and results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_families_dir = Path(tmpdir) / "model_families"
        results_dir = Path(tmpdir) / "results"
        model_families_dir.mkdir()
        results_dir.mkdir()
        yield model_families_dir, results_dir


@pytest.fixture
def registry_with_family(temp_dirs):
    """Create a registry with the modulo addition family."""
    model_families_dir, results_dir = temp_dirs

    # Copy family.json to temp location
    family_dir = model_families_dir / "modulo_addition_1layer"
    family_dir.mkdir()

    family_json = {
        "name": "modulo_addition_1layer",
        "display_name": "Modulo Addition (1 Layer)",
        "description": "Single-layer transformer for modular arithmetic",
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
        "analyzers": ["dominant_frequencies", "neuron_activations", "neuron_freq_norm"],
        "visualizations": [],
        "analysis_dataset": {"type": "modulo_addition_grid"},
        "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}",
    }
    with open(family_dir / "family.json", "w") as f:
        json.dump(family_json, f)

    registry = FamilyRegistry(
        model_families_dir=model_families_dir,
        results_dir=results_dir,
    )
    return registry, results_dir


@pytest.fixture
def trained_variant(registry_with_family):
    """Create a trained variant with minimal training."""
    registry, results_dir = registry_with_family
    family = registry.get_family("modulo_addition_1layer")
    params = {"prime": 17, "seed": 42, "data_seed": 598}
    variant = registry.create_variant(family, params)

    # Train with minimal epochs
    variant.train(
        num_epochs=50,
        checkpoint_epochs=[0, 25, 49],
        device="cpu",
    )
    return variant


class TestAnalysisPipelineInstantiation:
    """Tests for AnalysisPipeline instantiation."""

    def test_pipeline_instantiation(self, trained_variant):
        """Can instantiate AnalysisPipeline with variant."""
        pipeline = AnalysisPipeline(trained_variant)
        assert pipeline is not None
        assert pipeline.variant is trained_variant

    def test_pipeline_creates_artifacts_directory(self, trained_variant):
        """Pipeline ensures artifacts directory exists."""
        pipeline = AnalysisPipeline(trained_variant)
        assert os.path.exists(pipeline.artifacts_dir)

    def test_pipeline_register_returns_self(self, trained_variant):
        """register() returns self for chaining."""
        pipeline = AnalysisPipeline(trained_variant)
        result = pipeline.register(MockAnalyzer())
        assert result is pipeline

    def test_pipeline_register_chaining(self, trained_variant):
        """Can chain multiple register calls."""
        pipeline = AnalysisPipeline(trained_variant)
        result = pipeline.register(MockAnalyzer("a")).register(MockAnalyzer("b"))
        assert result is pipeline
        assert len(pipeline._analyzers) == 2


class TestAnalysisPipelineRun:
    """Tests for AnalysisPipeline.run()."""

    def test_run_with_no_analyzers_does_nothing(self, trained_variant):
        """Running with no analyzers doesn't crash."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.run()

    def test_run_creates_manifest(self, trained_variant):
        """Running pipeline creates manifest.json."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        manifest_path = os.path.join(pipeline.artifacts_dir, "manifest.json")
        assert os.path.exists(manifest_path)

    def test_run_saves_artifacts(self, trained_variant):
        """Running pipeline saves per-epoch artifact files."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer("test_analyzer"))
        pipeline.run()

        # Per-epoch storage: artifacts/{analyzer_name}/epoch_{NNNNN}.npz
        analyzer_dir = os.path.join(pipeline.artifacts_dir, "test_analyzer")
        assert os.path.isdir(analyzer_dir)

        checkpoints = trained_variant.get_available_checkpoints()
        for epoch in checkpoints:
            epoch_file = os.path.join(analyzer_dir, f"epoch_{epoch:05d}.npz")
            assert os.path.exists(epoch_file)

    def test_run_processes_all_checkpoints(self, trained_variant):
        """Pipeline processes all available checkpoints."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        completed = pipeline.get_completed_epochs("mock")
        expected = trained_variant.get_available_checkpoints()
        assert completed == expected

    def test_run_with_specific_checkpoints_via_config(self, trained_variant):
        """Can specify subset of checkpoints to process via config."""
        config = AnalysisRunConfig(checkpoints=[0, 25])
        pipeline = AnalysisPipeline(trained_variant, config)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        completed = pipeline.get_completed_epochs("mock")
        assert completed == [0, 25]

    def test_manifest_structure(self, trained_variant):
        """Manifest has correct structure."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer("test"))
        pipeline.run()

        manifest_path = os.path.join(pipeline.artifacts_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "analyzers" in manifest
        assert "test" in manifest["analyzers"]
        assert "epochs_completed" in manifest["analyzers"]["test"]
        assert "variant_params" in manifest
        assert manifest["variant_params"]["prime"] == 17
        assert "family_name" in manifest
        assert manifest["family_name"] == "modulo_addition_1layer"


class TestAnalysisPipelineResumability:
    """Tests for pipeline resumability."""

    def test_skips_completed_epochs(self, trained_variant):
        """Pipeline skips already-completed epochs."""
        config1 = AnalysisRunConfig(checkpoints=[0, 25])
        pipeline1 = AnalysisPipeline(trained_variant, config1)
        pipeline1.register(MockAnalyzer())
        pipeline1.run()

        pipeline2 = AnalysisPipeline(trained_variant)
        pipeline2.register(MockAnalyzer())

        completed_before = pipeline2.get_completed_epochs("mock")
        assert completed_before == [0, 25]

        pipeline2.run()

        completed_after = pipeline2.get_completed_epochs("mock")
        assert completed_after == [0, 25, 49]

    def test_force_recomputes_all(self, trained_variant):
        """force=True recomputes even completed epochs."""

        class CountingAnalyzer:
            def __init__(self):
                self.call_count = 0

            @property
            def name(self):
                return "counting"

            def analyze(self, bundle, probe, context):
                self.call_count += 1
                return {"data": np.ones((5,))}

        pipeline1 = AnalysisPipeline(trained_variant)
        analyzer1 = CountingAnalyzer()
        pipeline1.register(analyzer1)
        pipeline1.run()
        first_count = analyzer1.call_count

        pipeline2 = AnalysisPipeline(trained_variant)
        analyzer2 = CountingAnalyzer()
        pipeline2.register(analyzer2)
        pipeline2.run(force=True)

        assert analyzer2.call_count == first_count

    def test_manifest_persists_between_sessions(self, trained_variant):
        """Manifest is loaded correctly in new pipeline instance."""
        config = AnalysisRunConfig(checkpoints=[0])
        pipeline1 = AnalysisPipeline(trained_variant, config)
        pipeline1.register(MockAnalyzer())
        pipeline1.run()

        del pipeline1

        pipeline2 = AnalysisPipeline(trained_variant)
        completed = pipeline2.get_completed_epochs("mock")
        assert completed == [0]


class TestAnalysisPipelineArtifactLoading:
    """Tests for loading pipeline artifacts via ArtifactLoader."""

    def test_artifacts_loadable_by_loader(self, trained_variant):
        """Artifacts created by pipeline are discoverable by ArtifactLoader."""
        from miscope.analysis import ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epochs = loader.get_epochs("mock")
        assert len(epochs) > 0

    def test_load_epoch_returns_data(self, trained_variant):
        """Can load a single epoch's data via ArtifactLoader."""
        from miscope.analysis import ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epochs = loader.get_epochs("mock")
        epoch_data = loader.load_epoch("mock", epochs[0])

        assert isinstance(epoch_data, dict)
        assert "data" in epoch_data
        assert epoch_data["data"].shape == (10,)

    def test_load_all_epochs_stacked(self, trained_variant):
        """Can load all epochs stacked via ArtifactLoader.load()."""
        from miscope.analysis import ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        artifact = loader.load("mock")

        assert isinstance(artifact, dict)
        assert "epochs" in artifact
        assert "data" in artifact
        expected_epochs = np.array([0, 25, 49])
        np.testing.assert_array_equal(artifact["epochs"], expected_epochs)

    def test_load_stacked_data_shape(self, trained_variant):
        """Stacked data has correct shape (n_epochs, ...)."""
        from miscope.analysis import ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        artifact = loader.load("mock")
        assert artifact["data"].shape[0] == 3
        assert artifact["data"].shape[1] == 10

    def test_load_nonexistent_raises(self, trained_variant):
        """Loading nonexistent analyzer raises FileNotFoundError."""
        from miscope.analysis import ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        loader = ArtifactLoader(pipeline.artifacts_dir)

        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent")


class TestAnalysisPipelineMultipleAnalyzers:
    """Tests for multiple analyzers."""

    def test_multiple_analyzers_produce_separate_artifacts(self, trained_variant):
        """Each analyzer creates its own artifact directory."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer("analyzer_a"))
        pipeline.register(MockAnalyzer("analyzer_b"))
        pipeline.run()

        assert os.path.isdir(os.path.join(pipeline.artifacts_dir, "analyzer_a"))
        assert os.path.isdir(os.path.join(pipeline.artifacts_dir, "analyzer_b"))

    def test_manifest_tracks_all_analyzers(self, trained_variant):
        """Manifest tracks completion for all analyzers."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer("a"))
        pipeline.register(MockAnalyzer("b"))
        pipeline.run()

        manifest_path = os.path.join(pipeline.artifacts_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "a" in manifest["analyzers"]
        assert "b" in manifest["analyzers"]


class TestAnalysisRunConfig:
    """Tests for AnalysisRunConfig."""

    def test_default_config(self):
        """Default config has empty analyzers and None checkpoints."""
        config = AnalysisRunConfig()
        assert config.analyzers == []
        assert config.checkpoints is None

    def test_config_with_analyzers(self):
        """Can specify analyzers in config."""
        config = AnalysisRunConfig(analyzers=["a", "b"])
        assert config.analyzers == ["a", "b"]

    def test_config_with_checkpoints(self):
        """Can specify checkpoints in config."""
        config = AnalysisRunConfig(checkpoints=[0, 100, 200])
        assert config.checkpoints == [0, 100, 200]


class SummaryMockAnalyzer:
    """Mock analyzer that produces both artifacts and summary statistics."""

    def __init__(self, name: str = "summary_mock"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def analyze(self, bundle, probe, context: dict[str, Any]) -> dict[str, np.ndarray]:
        """Returns per-epoch artifact data."""
        return {"data": np.random.rand(10).astype(np.float32)}

    def get_summary_keys(self) -> list[str]:
        return ["mean_val", "max_val"]

    def compute_summary(
        self, result: dict[str, np.ndarray], context: dict[str, Any]
    ) -> dict[str, float]:
        return {
            "mean_val": float(np.mean(result["data"])),
            "max_val": float(np.max(result["data"])),
        }


class TestPipelineSummaryStatistics:
    """Tests for REQ_022: Summary statistics collection and persistence."""

    def test_summary_file_created(self, trained_variant):
        """Pipeline creates summary.npz for analyzer with summary support."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(SummaryMockAnalyzer())
        pipeline.run()

        summary_path = os.path.join(pipeline.artifacts_dir, "summary_mock", "summary.npz")
        assert os.path.exists(summary_path)

    def test_summary_file_contents(self, trained_variant):
        """Summary file contains epochs and declared summary keys."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(SummaryMockAnalyzer())
        pipeline.run()

        summary_path = os.path.join(pipeline.artifacts_dir, "summary_mock", "summary.npz")
        summary = dict(np.load(summary_path))

        assert "epochs" in summary
        assert "mean_val" in summary
        assert "max_val" in summary

        expected_epochs = sorted(trained_variant.get_available_checkpoints())
        np.testing.assert_array_equal(summary["epochs"], expected_epochs)
        assert summary["mean_val"].shape == (len(expected_epochs),)
        assert summary["max_val"].shape == (len(expected_epochs),)

    def test_no_summary_for_regular_analyzer(self, trained_variant):
        """MockAnalyzer (no summary methods) does not produce summary.npz."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        summary_path = os.path.join(pipeline.artifacts_dir, "mock", "summary.npz")
        assert not os.path.exists(summary_path)

    def test_mixed_analyzers(self, trained_variant):
        """Pipeline handles mix of summary and non-summary analyzers."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer("regular"))
        pipeline.register(SummaryMockAnalyzer("with_summary"))
        pipeline.run()

        # Regular analyzer: no summary file
        assert not os.path.exists(os.path.join(pipeline.artifacts_dir, "regular", "summary.npz"))
        # Summary analyzer: has summary file
        assert os.path.exists(os.path.join(pipeline.artifacts_dir, "with_summary", "summary.npz"))
        # Both produce per-epoch artifacts
        assert len(os.listdir(os.path.join(pipeline.artifacts_dir, "regular"))) > 0
        assert any(
            f.startswith("epoch_")
            for f in os.listdir(os.path.join(pipeline.artifacts_dir, "with_summary"))
        )

    def test_mock_analyzer_still_conforms_to_protocol(self):
        """MockAnalyzer without summary methods still conforms to Analyzer protocol."""
        analyzer = MockAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_summary_gap_filling(self, trained_variant):
        """Incremental run merges new summaries with existing ones."""
        # First run: analyze epochs [0, 25] only
        config1 = AnalysisRunConfig(checkpoints=[0, 25])
        pipeline1 = AnalysisPipeline(trained_variant, config1)
        pipeline1.register(SummaryMockAnalyzer())
        pipeline1.run()

        summary_path = os.path.join(pipeline1.artifacts_dir, "summary_mock", "summary.npz")
        summary1 = dict(np.load(summary_path))
        np.testing.assert_array_equal(summary1["epochs"], [0, 25])

        # Second run: analyze all (should add epoch 49)
        pipeline2 = AnalysisPipeline(trained_variant)
        pipeline2.register(SummaryMockAnalyzer())
        pipeline2.run()

        summary2 = dict(np.load(summary_path))
        np.testing.assert_array_equal(summary2["epochs"], [0, 25, 49])
        assert summary2["mean_val"].shape == (3,)
        assert summary2["max_val"].shape == (3,)

    def test_summary_force_recompute(self, trained_variant):
        """force=True rewrites summary from scratch."""
        pipeline1 = AnalysisPipeline(trained_variant)
        pipeline1.register(SummaryMockAnalyzer())
        pipeline1.run()

        summary_path = os.path.join(pipeline1.artifacts_dir, "summary_mock", "summary.npz")
        summary1 = dict(np.load(summary_path))

        # Force recompute (random data, so values will differ)
        pipeline2 = AnalysisPipeline(trained_variant)
        pipeline2.register(SummaryMockAnalyzer())
        pipeline2.run(force=True)

        summary2 = dict(np.load(summary_path))
        np.testing.assert_array_equal(summary2["epochs"], summary1["epochs"])

    def test_summary_loadable_by_artifact_loader(self, trained_variant):
        """Summary statistics are loadable via ArtifactLoader."""
        from miscope.analysis import ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(SummaryMockAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        assert loader.has_summary("summary_mock")

        summary = loader.load_summary("summary_mock")
        assert "epochs" in summary
        assert "mean_val" in summary
        assert "max_val" in summary

    def test_summary_with_specific_checkpoints(self, trained_variant):
        """Summary only contains epochs that were analyzed."""
        config = AnalysisRunConfig(checkpoints=[0, 49])
        pipeline = AnalysisPipeline(trained_variant, config)
        pipeline.register(SummaryMockAnalyzer())
        pipeline.run()

        summary_path = os.path.join(pipeline.artifacts_dir, "summary_mock", "summary.npz")
        summary = dict(np.load(summary_path))
        np.testing.assert_array_equal(summary["epochs"], [0, 49])


class TestGetCompletedEpochsCrossEpoch:
    """get_completed_epochs recognizes cross-epoch artifacts as satisfying dependencies."""

    def test_returns_empty_when_no_artifacts(self, trained_variant):
        """Returns empty list when analyzer directory is absent."""
        pipeline = AnalysisPipeline(trained_variant)
        result = pipeline.get_completed_epochs("nonexistent_analyzer")
        assert result == []

    def test_returns_per_epoch_files_when_present(self, trained_variant):
        """Returns epoch numbers from epoch_NNNNN.npz files as before."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer("epoch_test"))
        pipeline.run()
        completed = pipeline.get_completed_epochs("epoch_test")
        assert len(completed) > 0
        assert all(isinstance(e, int) for e in completed)

    def test_cross_epoch_artifact_satisfies_dependency(self, trained_variant):
        """When only cross_epoch.npz exists, returns available checkpoint epochs."""
        pipeline = AnalysisPipeline(trained_variant)

        # Manually write a cross_epoch.npz with no per-epoch files
        cross_only_dir = os.path.join(pipeline.artifacts_dir, "cross_only_analyzer")
        os.makedirs(cross_only_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(cross_only_dir, "cross_epoch.npz"),
            data=np.array([1.0]),
        )

        result = pipeline.get_completed_epochs("cross_only_analyzer")
        expected = sorted(trained_variant.get_available_checkpoints())
        assert result == expected

    def test_per_epoch_takes_precedence_over_cross_epoch(self, trained_variant):
        """When both per-epoch and cross_epoch files exist, per-epoch files win."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(MockAnalyzer("mixed_analyzer"))
        pipeline.run()

        mixed_dir = os.path.join(pipeline.artifacts_dir, "mixed_analyzer")
        np.savez_compressed(
            os.path.join(mixed_dir, "cross_epoch.npz"),
            data=np.array([1.0]),
        )

        # Should still return just the per-epoch files, not all checkpoints
        result = pipeline.get_completed_epochs("mixed_analyzer")
        expected = sorted(trained_variant.get_available_checkpoints())
        assert result == expected
