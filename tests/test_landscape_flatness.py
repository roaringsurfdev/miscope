"""Tests for REQ_031: Loss Landscape Flatness."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from miscope.analysis import AnalysisPipeline, Analyzer, ArtifactLoader
from miscope.analysis.analyzers import LandscapeFlatnessAnalyzer
from miscope.analysis.analyzers.landscape_flatness import FLATNESS_SUMMARY_KEYS
from miscope.analysis.bundle import TransformerLensBundle
from miscope.analysis.library.landscape import compute_landscape_flatness
from miscope.analysis.protocols import ActivationContext
from miscope.families import FamilyRegistry
from miscope.visualization.renderers.landscape_flatness import (
    FLATNESS_METRICS,
    render_flatness_trajectory,
    render_perturbation_distribution,
)

# ── Library: compute_landscape_flatness tests ────────────────────────


@pytest.fixture
def small_model():
    """Create a minimal HookedTransformer for testing."""
    cfg = HookedTransformerConfig(
        d_model=32,
        d_head=8,
        n_heads=4,
        n_layers=1,
        d_vocab=10,
        d_mlp=64,
        n_ctx=3,
        act_fn="relu",
    )
    return HookedTransformer(cfg)


@pytest.fixture
def dummy_probe():
    """Create a dummy probe tensor."""
    return torch.randint(0, 10, (20, 3))


def make_loss_fn():
    """Create a simple loss function for testing."""

    def loss_fn(model, probe):
        with torch.no_grad():
            logits = model(probe)
        return float(logits[:, -1].mean())

    return loss_fn


class TestComputeLandscapeFlatness:
    """Tests for the compute_landscape_flatness library function."""

    def test_returns_correct_keys(self, small_model, dummy_probe):
        """Result contains baseline_loss, delta_losses, epsilon."""
        result = compute_landscape_flatness(
            small_model, dummy_probe, make_loss_fn(), n_directions=5, seed=42
        )
        assert "baseline_loss" in result
        assert "delta_losses" in result
        assert "epsilon" in result

    def test_delta_losses_length(self, small_model, dummy_probe):
        """delta_losses has length n_directions."""
        n = 7
        result = compute_landscape_flatness(
            small_model, dummy_probe, make_loss_fn(), n_directions=n, seed=42
        )
        assert len(result["delta_losses"]) == n

    def test_baseline_loss_is_scalar(self, small_model, dummy_probe):
        """baseline_loss is a scalar array."""
        result = compute_landscape_flatness(
            small_model, dummy_probe, make_loss_fn(), n_directions=3, seed=42
        )
        assert result["baseline_loss"].ndim == 0

    def test_epsilon_stored_correctly(self, small_model, dummy_probe):
        """Stored epsilon matches the input value."""
        result = compute_landscape_flatness(
            small_model,
            dummy_probe,
            make_loss_fn(),
            n_directions=3,
            epsilon=0.05,
            seed=42,
        )
        assert float(result["epsilon"]) == pytest.approx(0.05)

    def test_weights_unchanged_after_call(self, small_model, dummy_probe):
        """Model weights are identical before and after call."""
        original = {k: v.clone() for k, v in small_model.state_dict().items()}
        compute_landscape_flatness(
            small_model, dummy_probe, make_loss_fn(), n_directions=5, seed=42
        )
        for k, v in small_model.state_dict().items():
            assert torch.allclose(v, original[k]), f"Weight {k} changed"

    def test_seed_reproducibility(self, small_model, dummy_probe):
        """Same seed produces identical results."""
        r1 = compute_landscape_flatness(
            small_model, dummy_probe, make_loss_fn(), n_directions=5, seed=123
        )
        r2 = compute_landscape_flatness(
            small_model, dummy_probe, make_loss_fn(), n_directions=5, seed=123
        )
        np.testing.assert_array_almost_equal(r1["delta_losses"], r2["delta_losses"])

    def test_different_seeds_differ(self, small_model, dummy_probe):
        """Different seeds produce different results."""
        r1 = compute_landscape_flatness(
            small_model, dummy_probe, make_loss_fn(), n_directions=10, seed=1
        )
        r2 = compute_landscape_flatness(
            small_model, dummy_probe, make_loss_fn(), n_directions=10, seed=2
        )
        assert not np.allclose(r1["delta_losses"], r2["delta_losses"])


# ── Analyzer protocol tests ──────────────────────────────────────────


class TestLandscapeFlatnessAnalyzerProtocol:
    """Tests for protocol conformance."""

    def test_conforms_to_analyzer_protocol(self):
        """LandscapeFlatnessAnalyzer implements Analyzer protocol."""
        analyzer = LandscapeFlatnessAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_has_correct_name(self):
        """Analyzer has correct name."""
        analyzer = LandscapeFlatnessAnalyzer()
        assert analyzer.name == "landscape_flatness"

    def test_registered_in_registry(self):
        """Analyzer is registered in default registry."""
        from miscope.analysis.analyzers import AnalyzerRegistry

        assert AnalyzerRegistry.is_registered("landscape_flatness")

    def test_configurable_constructor(self):
        """Constructor accepts n_directions, epsilon, seed."""
        analyzer = LandscapeFlatnessAnalyzer(n_directions=100, epsilon=0.05, seed=42)
        assert analyzer.n_directions == 100
        assert analyzer.epsilon == 0.05
        assert analyzer.seed == 42

    def test_raises_without_loss_fn(self, small_model, dummy_probe):
        """Raises ValueError if loss_fn not in context."""
        from transformer_lens.ActivationCache import ActivationCache

        analyzer = LandscapeFlatnessAnalyzer(n_directions=3)
        cache = ActivationCache({}, small_model)
        with pytest.raises(ValueError, match="loss_fn"):
            analyzer.analyze(
                ActivationContext(
                    bundle=TransformerLensBundle(small_model, cache, None),  # type: ignore
                    probe=dummy_probe,
                    analysis_params={},
                )
            )

    def test_summary_keys(self):
        """get_summary_keys returns the 7 expected keys."""
        analyzer = LandscapeFlatnessAnalyzer()
        keys = analyzer.get_summary_keys()
        assert keys == FLATNESS_SUMMARY_KEYS
        assert len(keys) == 7

    def test_compute_summary_returns_all_keys(self):
        """compute_summary produces all 7 summary statistics."""
        analyzer = LandscapeFlatnessAnalyzer()
        result = {
            "baseline_loss": np.array(2.0),
            "delta_losses": np.array([0.1, 0.5, -0.2, 0.8, 0.05]),
            "epsilon": np.array(0.1),
        }
        summary = analyzer.compute_summary(result, {})
        for key in FLATNESS_SUMMARY_KEYS:
            assert key in summary, f"Missing summary key: {key}"

    def test_flatness_ratio_computation(self):
        """Flatness ratio correctly counts directions below threshold."""
        analyzer = LandscapeFlatnessAnalyzer()
        # baseline=1.0, threshold = 0.1 * 1.0 = 0.1
        # delta_losses: 0.05, 0.15, 0.08, 0.2 → 2 below threshold
        result = {
            "baseline_loss": np.array(1.0),
            "delta_losses": np.array([0.05, 0.15, 0.08, 0.2]),
            "epsilon": np.array(0.1),
        }
        summary = analyzer.compute_summary(result, {})
        assert summary["flatness_ratio"] == pytest.approx(0.5)

    def test_flatness_ratio_zero_baseline(self):
        """Flatness ratio uses floor of 0.1 when baseline is zero."""
        analyzer = LandscapeFlatnessAnalyzer()
        result = {
            "baseline_loss": np.array(0.0),
            "delta_losses": np.array([0.05, 0.15, 0.08]),
            "epsilon": np.array(0.1),
        }
        summary = analyzer.compute_summary(result, {})
        # threshold = 0.1 (floor), 0.05 and 0.08 below → 2/3
        assert summary["flatness_ratio"] == pytest.approx(2.0 / 3.0)


# ── Renderer tests ────────────────────────────────────────────────────


class TestRenderFlatnessTrajectory:
    """Tests for render_flatness_trajectory."""

    @pytest.fixture
    def summary_data(self):
        """Create mock summary data."""
        epochs = np.array([0, 100, 200, 300, 400])
        return {
            "epochs": epochs,
            "mean_delta_loss": np.array([0.5, 0.4, 0.3, 0.2, 0.1]),
            "median_delta_loss": np.array([0.45, 0.35, 0.25, 0.18, 0.08]),
            "max_delta_loss": np.array([1.0, 0.8, 0.6, 0.5, 0.3]),
            "std_delta_loss": np.array([0.2, 0.18, 0.15, 0.1, 0.05]),
            "p90_delta_loss": np.array([0.8, 0.7, 0.5, 0.35, 0.2]),
            "flatness_ratio": np.array([0.3, 0.4, 0.5, 0.7, 0.9]),
            "baseline_loss": np.array([3.0, 2.5, 1.5, 0.8, 0.3]),
        }

    def test_returns_figure(self, summary_data):
        """Returns a Plotly Figure."""
        fig = render_flatness_trajectory(summary_data, current_epoch=200)
        assert isinstance(fig, go.Figure)

    def test_title_reflects_metric(self, summary_data):
        """Title includes the metric display name."""
        fig = render_flatness_trajectory(summary_data, current_epoch=200, metric="p90_delta_loss")
        assert "P90" in fig.layout.title.text  # type: ignore[attr-defined]

    def test_has_secondary_axis(self, summary_data):
        """Figure has secondary y-axis for baseline loss."""
        fig = render_flatness_trajectory(summary_data, current_epoch=200)
        assert fig.layout.yaxis2  # type: ignore[attr-defined] is not None

    def test_custom_title(self, summary_data):
        """Custom title is applied."""
        fig = render_flatness_trajectory(summary_data, current_epoch=200, title="Custom Title")
        assert fig.layout.title.text  # type: ignore[attr-defined] == "Custom Title"

    def test_all_metrics_render(self, summary_data):
        """All FLATNESS_METRICS render without error."""
        for metric in FLATNESS_METRICS:
            fig = render_flatness_trajectory(summary_data, current_epoch=200, metric=metric)
            assert isinstance(fig, go.Figure)


class TestRenderPerturbationDistribution:
    """Tests for render_perturbation_distribution."""

    @pytest.fixture
    def epoch_data(self):
        """Create mock per-epoch data."""
        rng = np.random.default_rng(42)
        return {
            "delta_losses": rng.normal(0.3, 0.15, size=50).astype(np.float32),
            "baseline_loss": np.array(2.0, dtype=np.float32),
            "epsilon": np.array(0.1, dtype=np.float32),
        }

    def test_returns_figure(self, epoch_data):
        """Returns a Plotly Figure."""
        fig = render_perturbation_distribution(epoch_data, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_has_histogram_trace(self, epoch_data):
        """Figure contains a Histogram trace."""
        fig = render_perturbation_distribution(epoch_data, epoch=100)
        hist_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(hist_traces) == 1

    def test_title_includes_epoch(self, epoch_data):
        """Title includes the epoch number."""
        fig = render_perturbation_distribution(epoch_data, epoch=250)
        assert "250" in fig.layout.title.text  # type: ignore[attr-defined]

    def test_title_includes_flatness_ratio(self, epoch_data):
        """Title includes flatness ratio."""
        fig = render_perturbation_distribution(epoch_data, epoch=100)
        assert "Flatness Ratio" in fig.layout.title.text  # type: ignore[attr-defined]

    def test_custom_title(self, epoch_data):
        """Custom title is applied."""
        fig = render_perturbation_distribution(epoch_data, epoch=100, title="Custom")
        assert fig.layout.title.text  # type: ignore[attr-defined] == "Custom"


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
    """Create a registry with landscape_flatness analyzer configured."""
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
        "analyzers": ["landscape_flatness"],
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


class TestLandscapeFlatnessIntegration:
    """Integration tests with AnalysisPipeline."""

    def test_pipeline_creates_artifact(self, trained_variant):
        """Pipeline creates per-epoch artifact files."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(LandscapeFlatnessAnalyzer(n_directions=5, seed=42))
        pipeline.run()

        analyzer_dir = os.path.join(pipeline.artifacts_dir, "landscape_flatness")
        assert os.path.isdir(analyzer_dir)

    def test_artifact_has_correct_epochs(self, trained_variant):
        """Artifact loader discovers correct epochs."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(LandscapeFlatnessAnalyzer(n_directions=5, seed=42))
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epochs = loader.get_epochs("landscape_flatness")
        assert epochs == [0, 25, 49]

    def test_per_epoch_contains_correct_keys(self, trained_variant):
        """Per-epoch artifact contains baseline_loss, delta_losses, epsilon."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(LandscapeFlatnessAnalyzer(n_directions=5, seed=42))
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epoch_data = loader.load_epoch("landscape_flatness", 0)
        assert "baseline_loss" in epoch_data
        assert "delta_losses" in epoch_data
        assert "epsilon" in epoch_data
        assert len(epoch_data["delta_losses"]) == 5

    def test_summary_contains_all_keys(self, trained_variant):
        """Summary file contains all 7 keys plus epochs."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(LandscapeFlatnessAnalyzer(n_directions=5, seed=42))
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        summary = loader.load_summary("landscape_flatness")
        assert "epochs" in summary
        for key in FLATNESS_SUMMARY_KEYS:
            assert key in summary, f"Missing summary key: {key}"

    def test_renderers_work_with_real_data(self, trained_variant):
        """Both renderers work with real pipeline output."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(LandscapeFlatnessAnalyzer(n_directions=5, seed=42))
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)

        # Trajectory from summary
        summary = loader.load_summary("landscape_flatness")
        fig_traj = render_flatness_trajectory(summary, current_epoch=25)
        assert isinstance(fig_traj, go.Figure)

        # Distribution from per-epoch
        epoch_data = loader.load_epoch("landscape_flatness", 25)
        fig_dist = render_perturbation_distribution(epoch_data, epoch=25)
        assert isinstance(fig_dist, go.Figure)
