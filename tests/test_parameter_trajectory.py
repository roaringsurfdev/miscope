"""Tests for REQ_029/REQ_032: Parameter Space Trajectory Projections."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.analysis import AnalysisPipeline, Analyzer, ArtifactLoader
from miscope.analysis.analyzers import ParameterSnapshotAnalyzer
from miscope.analysis.bundle import TransformerLensBundle
from miscope.analysis.library.trajectory import (
    compute_parameter_velocity,
    compute_pca_trajectory,
    flatten_snapshot,
)
from miscope.analysis.library.weights import (
    COMPONENT_GROUPS,
    WEIGHT_MATRIX_NAMES,
    extract_parameter_snapshot,
)
from miscope.analysis.protocols import ActivationContext
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
    snapshots = []
    for i in range(n_epochs):
        snapshots.append(_make_snapshot(d=d, scale=1.0 + i * 0.1, seed=i))
    return snapshots


# ── Library: weights.py tests ─────────────────────────────────────────


class TestWeightMatrixConstants:
    """Tests for weight matrix name constants."""

    def test_weight_matrix_names_count(self):
        """All 9 transformer weight matrices are listed."""
        assert len(WEIGHT_MATRIX_NAMES) == 9

    def test_component_groups_cover_all_weights(self):
        """Component groups cover all transformer weight matrix names (plus arch-specific extras)."""
        all_from_groups = set()
        for components in COMPONENT_GROUPS.values():
            all_from_groups.update(components)
        # All transformer names must be covered; arch-specific names (embed_a, embed_b)
        # may also appear in COMPONENT_GROUPS without being in WEIGHT_MATRIX_NAMES.
        assert set(WEIGHT_MATRIX_NAMES).issubset(all_from_groups)

    def test_component_groups_are_disjoint(self):
        """Component groups don't overlap."""
        seen = set()
        for components in COMPONENT_GROUPS.values():
            for c in components:
                assert c not in seen, f"{c} appears in multiple groups"
                seen.add(c)


class TestExtractParameterSnapshot:
    """Tests for extract_parameter_snapshot with a real model."""

    @pytest.fixture
    def model(self):
        """Create a minimal HookedTransformer."""
        from transformer_lens import HookedTransformer, HookedTransformerConfig

        cfg = HookedTransformerConfig(
            d_model=32,
            d_head=8,
            n_heads=4,
            n_layers=1,
            d_vocab=10,
            d_mlp=128,
            n_ctx=3,
            act_fn="relu",
        )
        return HookedTransformer(cfg)

    def test_returns_dict(self, model):
        """Returns a dict."""
        snapshot = extract_parameter_snapshot(TransformerLensBundle(model, None, None))  # type: ignore
        assert isinstance(snapshot, dict)

    def test_contains_all_weight_names(self, model):
        """Result contains all expected weight matrix names."""
        snapshot = extract_parameter_snapshot(TransformerLensBundle(model, None, None))  # type: ignore
        for name in WEIGHT_MATRIX_NAMES:
            assert name in snapshot, f"Missing weight: {name}"

    def test_values_are_numpy(self, model):
        """All values are numpy arrays."""
        snapshot = extract_parameter_snapshot(TransformerLensBundle(model, None, None))  # type: ignore
        for name, arr in snapshot.items():
            assert isinstance(arr, np.ndarray), f"{name} is not numpy"

    def test_shapes_match_model(self, model):
        """Extracted shapes match model architecture."""
        snapshot = extract_parameter_snapshot(TransformerLensBundle(model, None, None))  # type: ignore
        assert snapshot["W_E"].shape == (10, 32)  # (d_vocab, d_model)
        assert snapshot["W_pos"].shape == (3, 32)  # (n_ctx, d_model)
        assert snapshot["W_Q"].shape == (4, 32, 8)  # (n_heads, d_model, d_head)
        assert snapshot["W_in"].shape == (32, 128)  # (d_model, d_mlp)
        assert snapshot["W_out"].shape == (128, 32)  # (d_mlp, d_model)
        assert snapshot["W_U"].shape == (32, 10)  # (d_model, d_vocab)


# ── Library: trajectory.py tests ──────────────────────────────────────


class TestFlattenSnapshot:
    """Tests for flatten_snapshot function."""

    def test_returns_1d_array(self):
        """Output is a 1D numpy array."""
        snapshot = _make_snapshot()
        result = flatten_snapshot(snapshot)
        assert result.ndim == 1

    def test_default_includes_all(self):
        """None components includes all weight matrices."""
        snapshot = _make_snapshot()
        result = flatten_snapshot(snapshot)
        expected_size = sum(snapshot[k].size for k in WEIGHT_MATRIX_NAMES)
        assert result.shape[0] == expected_size

    def test_component_filter(self):
        """Specifying components filters the result."""
        snapshot = _make_snapshot()
        result = flatten_snapshot(snapshot, components=["W_E"])
        assert result.shape[0] == snapshot["W_E"].size

    def test_component_group_filter(self):
        """Component groups can be used as filters."""
        snapshot = _make_snapshot()
        result = flatten_snapshot(snapshot, components=COMPONENT_GROUPS["mlp"])
        expected_size = snapshot["W_in"].size + snapshot["W_out"].size
        assert result.shape[0] == expected_size


class TestComputePcaTrajectory:
    """Tests for compute_pca_trajectory function."""

    def test_returns_dict_with_expected_keys(self):
        """Output dict contains projections and variance info."""
        snapshots = _make_snapshot_sequence(5)
        result = compute_pca_trajectory(snapshots, n_components=3)
        assert "projections" in result
        assert "explained_variance_ratio" in result
        assert "explained_variance" in result

    def test_projections_shape(self):
        """Projections shape is (n_epochs, n_components)."""
        snapshots = _make_snapshot_sequence(5)
        result = compute_pca_trajectory(snapshots, n_components=3)
        assert result["projections"].shape == (5, 3)

    def test_variance_ratio_shape(self):
        """Variance ratio has n_components entries."""
        snapshots = _make_snapshot_sequence(5)
        result = compute_pca_trajectory(snapshots, n_components=3)
        assert result["explained_variance_ratio"].shape == (3,)

    def test_variance_ratios_sum_at_most_one(self):
        """Variance ratios sum to at most 1."""
        snapshots = _make_snapshot_sequence(10)
        result = compute_pca_trajectory(snapshots, n_components=5)
        assert result["explained_variance_ratio"].sum() <= 1.0 + 1e-6

    def test_n_components_clamped_to_samples(self):
        """n_components is clamped when fewer samples than requested."""
        snapshots = _make_snapshot_sequence(2)
        result = compute_pca_trajectory(snapshots, n_components=10)
        assert result["projections"].shape[1] == 2

    def test_component_filter(self):
        """Component filter restricts which weights are used."""
        snapshots = _make_snapshot_sequence(5)
        result_all = compute_pca_trajectory(snapshots, n_components=2)
        result_mlp = compute_pca_trajectory(
            snapshots, components=COMPONENT_GROUPS["mlp"], n_components=2
        )
        # Different inputs should produce different projections
        assert not np.allclose(result_all["projections"], result_mlp["projections"])


class TestComputeParameterVelocity:
    """Tests for compute_parameter_velocity function."""

    def test_output_length(self):
        """Velocity has n_epochs - 1 entries."""
        snapshots = _make_snapshot_sequence(5)
        velocity = compute_parameter_velocity(snapshots)
        assert velocity.shape == (4,)

    def test_values_are_nonnegative(self):
        """Velocities (L2 norms) are non-negative."""
        snapshots = _make_snapshot_sequence(5)
        velocity = compute_parameter_velocity(snapshots)
        assert np.all(velocity >= 0)

    def test_zero_velocity_for_identical_snapshots(self):
        """Velocity is zero when snapshots are identical."""
        snapshot = _make_snapshot(seed=42)
        snapshots = [snapshot, snapshot, snapshot]
        velocity = compute_parameter_velocity(snapshots)
        np.testing.assert_array_almost_equal(velocity, [0.0, 0.0])

    def test_normalized_by_epoch_gap(self):
        """Velocity is divided by epoch gap when epochs are provided."""
        snapshots = _make_snapshot_sequence(3)
        raw = compute_parameter_velocity(snapshots)
        # Non-uniform epochs: gaps of 10 and 100
        normalized = compute_parameter_velocity(snapshots, epochs=[0, 10, 110])
        np.testing.assert_allclose(normalized[0], raw[0] / 10)
        np.testing.assert_allclose(normalized[1], raw[1] / 100)

    def test_uniform_epochs_scale_evenly(self):
        """Uniform epoch spacing scales all velocities equally."""
        snapshots = _make_snapshot_sequence(3)
        raw = compute_parameter_velocity(snapshots)
        normalized = compute_parameter_velocity(snapshots, epochs=[0, 500, 1000])
        np.testing.assert_allclose(normalized, raw / 500)

    def test_component_filter(self):
        """Component filter restricts which weights are compared."""
        snapshots = _make_snapshot_sequence(3)
        vel_all = compute_parameter_velocity(snapshots)
        vel_mlp = compute_parameter_velocity(snapshots, components=COMPONENT_GROUPS["mlp"])
        # Different subsets = different velocities
        assert not np.allclose(vel_all, vel_mlp)


# ── Analyzer protocol tests ──────────────────────────────────────────


class TestParameterSnapshotAnalyzerProtocol:
    """Tests for protocol conformance."""

    def test_conforms_to_analyzer_protocol(self):
        """ParameterSnapshotAnalyzer implements Analyzer protocol."""
        analyzer = ParameterSnapshotAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_has_name(self):
        """Analyzer has correct name."""
        analyzer = ParameterSnapshotAnalyzer()
        assert analyzer.name == "parameter_snapshot"

    def test_has_analyze_method(self):
        """Analyzer has analyze method."""
        analyzer = ParameterSnapshotAnalyzer()
        assert callable(analyzer.analyze)

    def test_registered_in_registry(self):
        """Analyzer is registered in default registry."""
        from miscope.analysis.analyzers import AnalyzerRegistry

        assert AnalyzerRegistry.is_registered("parameter_snapshot")


# ── Renderer tests ────────────────────────────────────────────────────


class TestRenderers:
    """Tests for parameter trajectory renderers with precomputed data."""

    @pytest.fixture
    def trajectory_data(self):
        """Create precomputed PCA result and velocity for renderer tests."""
        snapshots = _make_snapshot_sequence(10)
        epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        pca_result = compute_pca_trajectory(snapshots, n_components=10)
        velocity = compute_parameter_velocity(snapshots, epochs=epochs)
        return pca_result, velocity, epochs

    @pytest.fixture
    def cross_epoch_data(self):
        """Create precomputed cross-epoch data with all component groups present in snapshots."""
        snapshots = _make_snapshot_sequence(10)
        epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        data = {"epochs": np.array(epochs)}
        first = snapshots[0] if snapshots else {}
        groups = {"all": None, **COMPONENT_GROUPS}
        for group_name, components in groups.items():
            # Skip groups whose keys are absent from the snapshots (e.g. learned_embeddings)
            if components is not None and not any(k in first for k in components):
                continue
            pca = compute_pca_trajectory(snapshots, components, n_components=10)
            vel = compute_parameter_velocity(snapshots, components, epochs=epochs)
            data[f"{group_name}__projections"] = pca["projections"]
            data[f"{group_name}__explained_variance_ratio"] = pca["explained_variance_ratio"]
            data[f"{group_name}__explained_variance"] = pca["explained_variance"]
            data[f"{group_name}__velocity"] = vel
        return data, epochs

    def test_render_parameter_trajectory_returns_figure(self, trajectory_data):
        """render_parameter_trajectory returns a Plotly Figure."""
        pca_result, _, epochs = trajectory_data
        fig = render_parameter_trajectory(pca_result, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)

    def test_render_parameter_trajectory_with_group_label(self, trajectory_data):
        """render_parameter_trajectory works with group_label."""
        pca_result, _, epochs = trajectory_data
        fig = render_parameter_trajectory(pca_result, epochs, current_epoch=50, group_label="Mlp")
        assert isinstance(fig, go.Figure)

    def test_render_explained_variance_returns_figure(self, trajectory_data):
        """render_explained_variance returns a Plotly Figure."""
        pca_result, _, _ = trajectory_data
        fig = render_explained_variance(pca_result)
        assert isinstance(fig, go.Figure)

    def test_render_parameter_velocity_returns_figure(self, trajectory_data):
        """render_parameter_velocity returns a Plotly Figure."""
        _, velocity, epochs = trajectory_data
        fig = render_parameter_velocity(velocity, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)

    def test_render_component_velocity_returns_figure(self, cross_epoch_data):
        """render_component_velocity returns a Plotly Figure."""
        data, epochs = cross_epoch_data
        fig = render_component_velocity(data, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)

    def test_render_component_velocity_has_three_traces(self, cross_epoch_data):
        """Component velocity renders one trace per group."""
        data, epochs = cross_epoch_data
        fig = render_component_velocity(data, epochs, current_epoch=50)
        # 3 component groups (embedding, attention, mlp) as line traces
        line_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(line_traces) == 3

    # ── REQ_032: 3D and additional 2D projections ──────────────────

    def test_render_trajectory_3d_returns_figure(self, trajectory_data):
        """render_trajectory_3d returns a Plotly Figure."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_3d(pca_result, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)

    def test_render_trajectory_3d_uses_scatter3d(self, trajectory_data):
        """3D renderer uses Scatter3d traces (not Scatter)."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_3d(pca_result, epochs, current_epoch=50)
        scatter3d_traces = [t for t in fig.data if isinstance(t, go.Scatter3d)]
        assert len(scatter3d_traces) >= 2  # path + points (+ optional highlight)

    def test_render_trajectory_3d_projects_3_dimensions(self, trajectory_data):
        """3D renderer data spans 3 coordinate axes."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_3d(pca_result, epochs, current_epoch=50)
        # The points trace (second trace) should have x, y, and z data
        points_trace = fig.data[1]
        assert isinstance(points_trace, go.Scatter3d)
        assert points_trace.x is not None
        assert points_trace.y is not None
        assert points_trace.z is not None

    def test_render_trajectory_3d_with_group_label(self, trajectory_data):
        """3D renderer accepts group_label parameter."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_3d(pca_result, epochs, current_epoch=50, group_label="Mlp")
        assert isinstance(fig, go.Figure)

    def test_render_trajectory_3d_highlights_epoch(self, trajectory_data):
        """3D renderer includes highlight marker for current epoch."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_3d(pca_result, epochs, current_epoch=50)
        # 3 traces: path, points, highlight
        assert len(fig.data) == 3  # type: ignore[arg-type]

    def test_render_trajectory_3d_axis_labels(self, trajectory_data):
        """3D renderer labels all three axes with PC number and variance."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_3d(pca_result, epochs, current_epoch=50)
        scene = fig.layout.scene  # type: ignore[union-attr]
        assert "PC1" in scene.xaxis.title.text
        assert "PC2" in scene.yaxis.title.text
        assert "PC3" in scene.zaxis.title.text

    def test_render_trajectory_pc1_pc3_returns_figure(self, trajectory_data):
        """render_trajectory_pc1_pc3 returns a Plotly Figure."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_pc1_pc3(pca_result, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)

    def test_render_trajectory_pc1_pc3_axis_labels(self, trajectory_data):
        """PC1 vs PC3 renderer uses correct axis labels."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_pc1_pc3(pca_result, epochs, current_epoch=50)
        assert "PC1" in fig.layout.xaxis.title.text  # type: ignore[attr-defined]
        assert "PC3" in fig.layout.yaxis.title.text  # type: ignore[attr-defined]

    def test_render_trajectory_pc2_pc3_returns_figure(self, trajectory_data):
        """render_trajectory_pc2_pc3 returns a Plotly Figure."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_pc2_pc3(pca_result, epochs, current_epoch=50)
        assert isinstance(fig, go.Figure)

    def test_render_trajectory_pc2_pc3_axis_labels(self, trajectory_data):
        """PC2 vs PC3 renderer uses correct axis labels."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_pc2_pc3(pca_result, epochs, current_epoch=50)
        assert "PC2" in fig.layout.xaxis.title.text  # type: ignore[attr-defined]
        assert "PC3" in fig.layout.yaxis.title.text  # type: ignore[attr-defined]

    def test_render_trajectory_pc1_pc3_with_group_label(self, trajectory_data):
        """PC1 vs PC3 renderer accepts group_label parameter."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_pc1_pc3(pca_result, epochs, current_epoch=50, group_label="Mlp")
        assert isinstance(fig, go.Figure)

    def test_render_trajectory_pc2_pc3_with_group_label(self, trajectory_data):
        """PC2 vs PC3 renderer accepts group_label parameter."""
        pca_result, _, epochs = trajectory_data
        fig = render_trajectory_pc2_pc3(pca_result, epochs, current_epoch=50, group_label="Mlp")
        assert isinstance(fig, go.Figure)

    def test_renderers_highlight_current_epoch(self, trajectory_data):
        """All trajectory renderers include highlight when epoch exists."""
        pca_result, _, epochs = trajectory_data
        for renderer in [
            render_trajectory_3d,
            render_trajectory_pc1_pc3,
            render_trajectory_pc2_pc3,
        ]:
            fig = renderer(pca_result, epochs, current_epoch=50)
            # 3 traces: path, points, highlight
            assert len(fig.data) == 3, f"{renderer.__name__} missing highlight"  # type: ignore[arg-type]

    def test_renderers_no_highlight_for_missing_epoch(self, trajectory_data):
        """Renderers skip highlight when epoch not in list."""
        pca_result, _, epochs = trajectory_data
        for renderer in [
            render_trajectory_3d,
            render_trajectory_pc1_pc3,
            render_trajectory_pc2_pc3,
        ]:
            fig = renderer(pca_result, epochs, current_epoch=999)
            # 2 traces: path, points only
            assert len(fig.data) == 2, f"{renderer.__name__} unexpected highlight"  # type: ignore[arg-type]

    def test_get_group_label_all(self):
        """'all' group gives 'All Parameters' label."""
        assert get_group_label("all") == "All Parameters"

    def test_get_group_label_named_group(self):
        """Named group gives capitalized group name."""
        assert get_group_label("mlp") == "Mlp"

    def test_get_group_label_unknown(self):
        """Unknown group gets capitalized fallback."""
        assert get_group_label("custom") == "Custom"


# ── Integration tests ─────────────────────────────────────────────────


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
    """Create a registry with the modulo addition family including parameter_snapshot."""
    model_families_dir, results_dir = temp_dirs

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
        "analyzers": ["parameter_snapshot"],
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

    variant.train(
        num_epochs=50,
        checkpoint_epochs=[0, 25, 49],
        device="cpu",
    )
    return variant


class TestParameterSnapshotAnalyzerOutput:
    """Tests for analyzer output with a real model."""

    @pytest.fixture
    def model_with_context(self, trained_variant):
        """Create model, run forward pass, return model, probe, cache, context."""
        import torch

        device = "cpu"
        family = trained_variant.family
        params = trained_variant.params

        probe = family.generate_analysis_dataset(params, device=device)
        context = family.prepare_analysis_context(params, device)

        model = family.create_model(params, device=device)
        state_dict = trained_variant.load_checkpoint(49)
        model.load_state_dict(state_dict)

        with torch.inference_mode():
            logits, cache = model.run_with_cache(probe)

        bundle = TransformerLensBundle(model, cache, logits)
        return bundle, probe, context

    def test_returns_dict(self, model_with_context):
        """analyze returns a dict."""
        bundle, probe, context = model_with_context
        analyzer = ParameterSnapshotAnalyzer()
        result = analyzer.analyze(
            ActivationContext(bundle=bundle, probe=probe, analysis_params=context)
        )
        assert isinstance(result, dict)

    def test_returns_all_weight_names(self, model_with_context):
        """Result contains all weight matrix names."""
        bundle, probe, context = model_with_context
        analyzer = ParameterSnapshotAnalyzer()
        result = analyzer.analyze(
            ActivationContext(bundle=bundle, probe=probe, analysis_params=context)
        )
        for name in WEIGHT_MATRIX_NAMES:
            assert name in result, f"Missing weight: {name}"

    def test_weight_shapes(self, model_with_context):
        """Weight matrices have correct shapes for p=17 model."""
        bundle, probe, context = model_with_context
        analyzer = ParameterSnapshotAnalyzer()
        result = analyzer.analyze(
            ActivationContext(bundle=bundle, probe=probe, analysis_params=context)
        )
        # d_model=128, d_mlp=512, n_heads=4, d_head=32
        assert result["W_E"].shape[1] == 128  # d_model
        assert result["W_in"].shape == (128, 512)  # (d_model, d_mlp)
        assert result["W_out"].shape == (512, 128)  # (d_mlp, d_model)


class TestParameterSnapshotIntegration:
    """Integration tests with AnalysisPipeline."""

    def test_pipeline_creates_artifact(self, trained_variant):
        """Pipeline creates per-epoch artifact files."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.run()

        analyzer_dir = os.path.join(pipeline.artifacts_dir, "parameter_snapshot")
        assert os.path.isdir(analyzer_dir)

    def test_artifact_contains_epochs(self, trained_variant):
        """Artifact loader discovers correct epochs."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        artifact = loader.load("parameter_snapshot")
        assert "epochs" in artifact
        np.testing.assert_array_equal(artifact["epochs"], [0, 25, 49])

    def test_per_epoch_artifact_contains_weights(self, trained_variant):
        """Per-epoch artifact contains all weight matrices."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epoch_data = loader.load_epoch("parameter_snapshot", 0)
        for name in WEIGHT_MATRIX_NAMES:
            assert name in epoch_data, f"Missing weight in artifact: {name}"

    def test_snapshots_usable_for_pca(self, trained_variant):
        """Loaded snapshots can be used with compute_pca_trajectory."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epochs = [0, 25, 49]
        snapshots = [loader.load_epoch("parameter_snapshot", e) for e in epochs]

        result = compute_pca_trajectory(snapshots, n_components=2)
        assert result["projections"].shape == (3, 2)

    def test_snapshots_usable_for_velocity(self, trained_variant):
        """Loaded snapshots can be used with compute_parameter_velocity."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epochs = [0, 25, 49]
        snapshots = [loader.load_epoch("parameter_snapshot", e) for e in epochs]

        velocity = compute_parameter_velocity(snapshots)
        assert velocity.shape == (2,)
        assert np.all(velocity >= 0)
