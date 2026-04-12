"""Tests for REQ_038: Cross-Epoch Analyzers."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.analysis import AnalysisPipeline, ArtifactLoader, CrossEpochAnalyzer
from miscope.analysis.analyzers import (
    AnalyzerRegistry,
    ParameterSnapshotAnalyzer,
    ParameterTrajectoryPCA,
)
from miscope.analysis.analyzers.parameter_trajectory_pca import _GROUPS
from miscope.analysis.library.trajectory import compute_parameter_velocity, compute_pca_trajectory
from miscope.families import FamilyRegistry
from miscope.visualization.renderers.parameter_trajectory import (
    get_group_label,
    render_component_velocity,
    render_explained_variance,
    render_parameter_trajectory,
    render_parameter_velocity,
    render_trajectory_3d,
    render_trajectory_pc1_pc3,
    render_trajectory_pc2_pc3,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _make_snapshot(d: int = 16, scale: float = 1.0, seed: int = 0) -> dict[str, np.ndarray]:
    """Create a fake snapshot dict matching WEIGHT_MATRIX_NAMES shapes."""
    rng = np.random.default_rng(seed)
    return {
        "W_E": rng.normal(size=(10, d), scale=scale).astype(np.float32),
        "W_pos": rng.normal(size=(3, d), scale=scale).astype(np.float32),
        "W_Q": rng.normal(size=(4, d // 4, d), scale=scale).astype(np.float32),
        "W_K": rng.normal(size=(4, d // 4, d), scale=scale).astype(np.float32),
        "W_V": rng.normal(size=(4, d // 4, d), scale=scale).astype(np.float32),
        "W_O": rng.normal(size=(4, d, d // 4), scale=scale).astype(np.float32),
        "W_in": rng.normal(size=(d, d * 4), scale=scale).astype(np.float32),
        "W_out": rng.normal(size=(d * 4, d), scale=scale).astype(np.float32),
        "W_U": rng.normal(size=(d, 10), scale=scale).astype(np.float32),
    }


def _make_snapshot_sequence(n_epochs: int = 5, d: int = 16) -> list[dict[str, np.ndarray]]:
    """Create a sequence of snapshots with progressive drift."""
    return [_make_snapshot(d=d, scale=1.0 + i * 0.1, seed=i) for i in range(n_epochs)]


@pytest.fixture
def artifacts_with_snapshots():
    """Create a temp artifacts dir with parameter_snapshot epoch files."""
    epochs = [0, 100, 200, 300, 400]
    snapshots = _make_snapshot_sequence(len(epochs))

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = os.path.join(tmpdir, "artifacts")
        snapshot_dir = os.path.join(artifacts_dir, "parameter_snapshot")
        os.makedirs(snapshot_dir)

        for epoch, snapshot in zip(epochs, snapshots):
            path = os.path.join(snapshot_dir, f"epoch_{epoch:05d}.npz")
            np.savez_compressed(path, **snapshot)  # type: ignore[arg-type]

        yield artifacts_dir, epochs, snapshots


# ── Protocol tests ────────────────────────────────────────────────────


class TestCrossEpochAnalyzerProtocol:
    """Tests for CrossEpochAnalyzer protocol conformance."""

    def test_parameter_trajectory_pca_conforms(self):
        """ParameterTrajectoryPCA satisfies CrossEpochAnalyzer protocol."""
        analyzer = ParameterTrajectoryPCA()
        assert isinstance(analyzer, CrossEpochAnalyzer)

    def test_has_name(self):
        analyzer = ParameterTrajectoryPCA()
        assert analyzer.name == "parameter_trajectory"

    def test_has_requires(self):
        analyzer = ParameterTrajectoryPCA()
        assert analyzer.requires == ["parameter_snapshot"]

    def test_has_analyze_method(self):
        analyzer = ParameterTrajectoryPCA()
        assert callable(analyzer.analyze_across_epochs)

    def test_registered_in_registry(self):
        assert "parameter_trajectory" in [name for name in AnalyzerRegistry._cross_epoch_analyzers]


# ── Analyzer output tests ────────────────────────────────────────────


class TestParameterTrajectoryPCA:
    """Tests for ParameterTrajectoryPCA analyzer."""

    def test_returns_dict(self, artifacts_with_snapshots):
        artifacts_dir, epochs, _ = artifacts_with_snapshots
        analyzer = ParameterTrajectoryPCA()
        result = analyzer.analyze_across_epochs(artifacts_dir, epochs, {})
        assert isinstance(result, dict)

    def test_contains_epochs(self, artifacts_with_snapshots):
        artifacts_dir, epochs, _ = artifacts_with_snapshots
        analyzer = ParameterTrajectoryPCA()
        result = analyzer.analyze_across_epochs(artifacts_dir, epochs, {})
        np.testing.assert_array_equal(result["epochs"], epochs)

    def test_contains_all_groups(self, artifacts_with_snapshots):
        artifacts_dir, epochs, _ = artifacts_with_snapshots
        analyzer = ParameterTrajectoryPCA()
        result = analyzer.analyze_across_epochs(artifacts_dir, epochs, {})
        # Groups whose weight matrices are absent from the snapshot are skipped.
        # Only assert groups that are actually present in the result.
        present_groups = {k.split("__")[0] for k in result if "__projections" in k}
        for group_name in present_groups:
            assert f"{group_name}__projections" in result
            assert f"{group_name}__explained_variance_ratio" in result
            assert f"{group_name}__explained_variance" in result
            assert f"{group_name}__velocity" in result
        # The "all" group must always be present
        assert "all__projections" in result

    def test_projections_shape(self, artifacts_with_snapshots):
        artifacts_dir, epochs, _ = artifacts_with_snapshots
        analyzer = ParameterTrajectoryPCA()
        result = analyzer.analyze_across_epochs(artifacts_dir, epochs, {})
        n = len(epochs)
        k = min(10, n)
        assert result["all__projections"].shape == (n, k)
        assert result["all__explained_variance_ratio"].shape == (k,)
        assert result["all__velocity"].shape == (n - 1,)

    def test_numerical_equivalence_with_library(self, artifacts_with_snapshots):
        """Cross-epoch results match direct library function calls."""
        artifacts_dir, epochs, snapshots = artifacts_with_snapshots
        analyzer = ParameterTrajectoryPCA()
        result = analyzer.analyze_across_epochs(artifacts_dir, epochs, {})

        # Compare "all" group with direct library call
        direct_pca = compute_pca_trajectory(snapshots, None, n_components=min(10, len(epochs)))
        np.testing.assert_allclose(
            result["all__projections"],
            direct_pca["projections"],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            result["all__explained_variance_ratio"],
            direct_pca["explained_variance_ratio"],
            atol=1e-5,
        )

        direct_vel = compute_parameter_velocity(snapshots, None, epochs=epochs)
        np.testing.assert_allclose(result["all__velocity"], direct_vel, atol=1e-5)

    def test_component_groups_differ(self, artifacts_with_snapshots):
        """Different component groups produce different PCA results."""
        artifacts_dir, epochs, _ = artifacts_with_snapshots
        analyzer = ParameterTrajectoryPCA()
        result = analyzer.analyze_across_epochs(artifacts_dir, epochs, {})
        assert not np.allclose(
            result["all__projections"],
            result["mlp__projections"],
        )


# ── ArtifactLoader tests ─────────────────────────────────────────────


class TestArtifactLoaderCrossEpoch:
    """Tests for ArtifactLoader cross-epoch methods."""

    def test_has_cross_epoch_false_when_missing(self, artifacts_with_snapshots):
        artifacts_dir, _, _ = artifacts_with_snapshots
        loader = ArtifactLoader(artifacts_dir)
        assert not loader.has_cross_epoch("parameter_trajectory")

    def test_has_cross_epoch_true_when_present(self, artifacts_with_snapshots):
        artifacts_dir, epochs, _ = artifacts_with_snapshots
        # Create the cross-epoch file
        analyzer = ParameterTrajectoryPCA()
        result = analyzer.analyze_across_epochs(artifacts_dir, epochs, {})
        out_dir = os.path.join(artifacts_dir, "parameter_trajectory")
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "cross_epoch.npz"), **result)  # type: ignore[arg-type]

        loader = ArtifactLoader(artifacts_dir)
        assert loader.has_cross_epoch("parameter_trajectory")

    def test_load_cross_epoch(self, artifacts_with_snapshots):
        artifacts_dir, epochs, _ = artifacts_with_snapshots
        analyzer = ParameterTrajectoryPCA()
        result = analyzer.analyze_across_epochs(artifacts_dir, epochs, {})
        out_dir = os.path.join(artifacts_dir, "parameter_trajectory")
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "cross_epoch.npz"), **result)  # type: ignore[arg-type]

        loader = ArtifactLoader(artifacts_dir)
        loaded = loader.load_cross_epoch("parameter_trajectory")
        np.testing.assert_array_equal(loaded["epochs"], epochs)
        assert "all__projections" in loaded

    def test_load_cross_epoch_file_not_found(self, artifacts_with_snapshots):
        artifacts_dir, _, _ = artifacts_with_snapshots
        loader = ArtifactLoader(artifacts_dir)
        with pytest.raises(FileNotFoundError):
            loader.load_cross_epoch("parameter_trajectory")


# ── Pipeline integration tests ────────────────────────────────────────


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
        "cross_epoch_analyzers": ["parameter_trajectory"],
        "visualizations": [],
        "analysis_dataset": {"type": "modulo_addition_grid"},
        "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}",
    }
    with open(family_dir / "family.json", "w") as f:
        json.dump(family_json, f)

    registry = FamilyRegistry(model_families_dir=model_families_dir, results_dir=results_dir)
    family = registry.get_family("modulo_addition_1layer")
    params = {"prime": 17, "seed": 42, "data_seed": 598}
    variant = registry.create_variant(family, params)
    variant.train(num_epochs=50, checkpoint_epochs=[0, 25, 49], device="cpu")
    return variant


class TestPipelineCrossEpoch:
    """Integration tests for pipeline Phase 2."""

    def test_cross_epoch_produces_artifact(self, trained_variant):
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_cross_epoch(ParameterTrajectoryPCA())
        pipeline.run()

        cross_epoch_path = os.path.join(
            pipeline.artifacts_dir,
            "parameter_trajectory",
            "cross_epoch.npz",
        )
        assert os.path.exists(cross_epoch_path)

    def test_cross_epoch_loadable(self, trained_variant):
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_cross_epoch(ParameterTrajectoryPCA())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        assert loader.has_cross_epoch("parameter_trajectory")
        data = loader.load_cross_epoch("parameter_trajectory")
        np.testing.assert_array_equal(data["epochs"], [0, 25, 49])
        assert data["all__projections"].shape[0] == 3

    def test_cross_epoch_skips_if_exists(self, trained_variant):
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_cross_epoch(ParameterTrajectoryPCA())
        pipeline.run()

        # Modify the cross-epoch file to detect if it gets overwritten
        cross_epoch_path = os.path.join(
            pipeline.artifacts_dir,
            "parameter_trajectory",
            "cross_epoch.npz",
        )
        mtime_before = os.path.getmtime(cross_epoch_path)

        # Run again without force — should skip
        import time

        time.sleep(0.05)
        pipeline2 = AnalysisPipeline(trained_variant)
        pipeline2.register(ParameterSnapshotAnalyzer())
        pipeline2.register_cross_epoch(ParameterTrajectoryPCA())
        pipeline2.run()

        mtime_after = os.path.getmtime(cross_epoch_path)
        assert mtime_after == mtime_before

    def test_cross_epoch_force_recomputes(self, trained_variant):
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_cross_epoch(ParameterTrajectoryPCA())
        pipeline.run()

        cross_epoch_path = os.path.join(
            pipeline.artifacts_dir,
            "parameter_trajectory",
            "cross_epoch.npz",
        )
        mtime_before = os.path.getmtime(cross_epoch_path)

        import time

        time.sleep(0.05)
        pipeline2 = AnalysisPipeline(trained_variant)
        pipeline2.register(ParameterSnapshotAnalyzer())
        pipeline2.register_cross_epoch(ParameterTrajectoryPCA())
        pipeline2.run(force=True)

        mtime_after = os.path.getmtime(cross_epoch_path)
        assert mtime_after > mtime_before

    def test_cross_epoch_fails_without_dependency(self, trained_variant):
        """Cross-epoch analyzer fails if required per-epoch analyzer hasn't run."""
        pipeline = AnalysisPipeline(trained_variant)
        # Don't register ParameterSnapshotAnalyzer — only register cross-epoch
        pipeline.register_cross_epoch(ParameterTrajectoryPCA())
        with pytest.raises(RuntimeError, match="requires.*parameter_snapshot"):
            pipeline.run()


# ── Updated renderer tests ────────────────────────────────────────────


class TestRenderersWithPrecomputedData:
    """Tests for trajectory renderers with precomputed PCA data."""

    @pytest.fixture
    def precomputed_data(self):
        """Create precomputed cross-epoch data for renderer tests."""
        snapshots = _make_snapshot_sequence(10)
        epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        pca_result = compute_pca_trajectory(snapshots, None, n_components=10)
        velocity = compute_parameter_velocity(snapshots, None, epochs=epochs)

        # Build full cross-epoch data dict — skip groups absent from mock snapshots
        first = snapshots[0] if snapshots else {}
        cross_epoch_data = {"epochs": np.array(epochs)}
        for group_name, components in _GROUPS.items():
            if components is not None and not any(k in first for k in components):
                continue
            pca = compute_pca_trajectory(snapshots, components, n_components=10)
            vel = compute_parameter_velocity(snapshots, components, epochs=epochs)
            cross_epoch_data[f"{group_name}__projections"] = pca["projections"]
            cross_epoch_data[f"{group_name}__explained_variance_ratio"] = pca[
                "explained_variance_ratio"
            ]
            cross_epoch_data[f"{group_name}__explained_variance"] = pca["explained_variance"]
            cross_epoch_data[f"{group_name}__velocity"] = vel

        return pca_result, velocity, epochs, cross_epoch_data

    def test_render_parameter_trajectory(self, precomputed_data):
        pca_result, _, epochs, _ = precomputed_data
        fig = render_parameter_trajectory(pca_result, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)

    def test_render_trajectory_3d(self, precomputed_data):
        pca_result, _, epochs, _ = precomputed_data
        fig = render_trajectory_3d(pca_result, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)
        scatter3d_traces = [t for t in fig.data if isinstance(t, go.Scatter3d)]
        assert len(scatter3d_traces) >= 2

    def test_render_trajectory_pc1_pc3(self, precomputed_data):
        pca_result, _, epochs, _ = precomputed_data
        fig = render_trajectory_pc1_pc3(pca_result, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)
        assert "PC1" in fig.layout.xaxis.title.text  # type: ignore[attr-defined]
        assert "PC3" in fig.layout.yaxis.title.text  # type: ignore[attr-defined]

    def test_render_trajectory_pc2_pc3(self, precomputed_data):
        pca_result, _, epochs, _ = precomputed_data
        fig = render_trajectory_pc2_pc3(pca_result, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)
        assert "PC2" in fig.layout.xaxis.title.text  # type: ignore[attr-defined]
        assert "PC3" in fig.layout.yaxis.title.text  # type: ignore[attr-defined]

    def test_render_explained_variance(self, precomputed_data):
        pca_result, _, _, _ = precomputed_data
        fig = render_explained_variance(pca_result)
        assert isinstance(fig, go.Figure)

    def test_render_parameter_velocity(self, precomputed_data):
        _, velocity, epochs, _ = precomputed_data
        fig = render_parameter_velocity(velocity, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)

    def test_render_component_velocity(self, precomputed_data):
        _, _, epochs, cross_epoch_data = precomputed_data
        fig = render_component_velocity(cross_epoch_data, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)
        line_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(line_traces) == 3  # embedding, attention, mlp

    def test_group_label(self):
        assert get_group_label("all") == "All Parameters"
        assert get_group_label("mlp") == "Mlp"
        assert get_group_label("embedding") == "Embedding"

    def test_highlights_current_epoch(self, precomputed_data):
        pca_result, _, epochs, _ = precomputed_data
        for renderer in [
            render_trajectory_3d,
            render_trajectory_pc1_pc3,
            render_trajectory_pc2_pc3,
        ]:
            fig = renderer(pca_result, epochs, current_epoch=50)
            assert len(fig.data) == 3, f"{renderer.__name__} missing highlight"  # type: ignore[arg-type]

    def test_no_highlight_for_missing_epoch(self, precomputed_data):
        pca_result, _, epochs, _ = precomputed_data
        for renderer in [
            render_trajectory_3d,
            render_trajectory_pc1_pc3,
            render_trajectory_pc2_pc3,
        ]:
            fig = renderer(pca_result, epochs, current_epoch=999)
            assert len(fig.data) == 2, f"{renderer.__name__} unexpected highlight"  # type: ignore[arg-type]
