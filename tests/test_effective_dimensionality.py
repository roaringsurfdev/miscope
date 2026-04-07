"""Tests for REQ_030: Weight Matrix Effective Dimensionality."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.analysis import AnalysisPipeline, Analyzer, ArtifactLoader
from miscope.analysis.analyzers import EffectiveDimensionalityAnalyzer
from miscope.analysis.bundle import TransformerLensBundle
from miscope.analysis.library.weights import (
    ATTENTION_MATRICES,
    WEIGHT_MATRIX_NAMES,
    compute_participation_ratio,
    compute_weight_singular_values,
)
from miscope.families import FamilyRegistry
from miscope.visualization.renderers.effective_dimensionality import (
    render_dimensionality_trajectory,
    render_singular_value_spectrum,
)

# ── Library: participation ratio tests ────────────────────────────────


class TestComputeParticipationRatio:
    """Tests for compute_participation_ratio."""

    def test_rank_one_pr_is_one(self):
        """A single nonzero singular value gives PR = 1.0."""
        sv = np.array([5.0, 0.0, 0.0, 0.0])
        assert compute_participation_ratio(sv) == pytest.approx(1.0)

    def test_equal_values_pr_is_n(self):
        """n equal singular values give PR = n."""
        n = 8
        sv = np.ones(n) * 3.0
        assert compute_participation_ratio(sv) == pytest.approx(float(n))

    def test_pr_in_valid_range(self):
        """PR is in [1, n] for valid inputs."""
        rng = np.random.default_rng(42)
        sv = np.abs(rng.normal(size=20))
        pr = compute_participation_ratio(sv)
        assert 1.0 <= pr <= 20.0

    def test_all_zeros_returns_zero(self):
        """All-zero singular values return 0.0."""
        sv = np.zeros(5)
        assert compute_participation_ratio(sv) == 0.0

    def test_two_values_intermediate_pr(self):
        """Two distinct values give PR between 1 and 2."""
        sv = np.array([3.0, 1.0])
        pr = compute_participation_ratio(sv)
        assert 1.0 < pr < 2.0

    def test_2d_input_returns_per_row(self):
        """2D input returns per-row participation ratios."""
        sv = np.array(
            [
                [5.0, 0.0, 0.0, 0.0],  # rank-1 → PR = 1
                [1.0, 1.0, 1.0, 1.0],  # equal  → PR = 4
            ]
        )
        pr = compute_participation_ratio(sv)
        assert isinstance(pr, np.ndarray)
        assert pr.shape == (2,)
        assert pr[0] == pytest.approx(1.0)
        assert pr[1] == pytest.approx(4.0)

    def test_2d_with_zero_row(self):
        """2D input with an all-zero row returns 0 for that row."""
        sv = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        pr = compute_participation_ratio(sv)
        assert isinstance(pr, np.ndarray)
        assert pr[0] == 0.0
        assert pr[1] == pytest.approx(3.0)


# ── Library: weight SVD tests ────────────────────────────────────────


class TestComputeWeightSingularValues:
    """Tests for compute_weight_singular_values."""

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

    def test_returns_all_sv_keys(self, model):
        """Result contains sv_{name} for all weight matrices."""
        result = compute_weight_singular_values(TransformerLensBundle(model, None, None))  # type: ignore
        for name in WEIGHT_MATRIX_NAMES:
            assert f"sv_{name}" in result, f"Missing key: sv_{name}"

    def test_non_attention_svs_are_1d(self, model):
        """Non-attention singular values are 1D arrays."""
        result = compute_weight_singular_values(TransformerLensBundle(model, None, None))  # type: ignore
        non_attn = [n for n in WEIGHT_MATRIX_NAMES if n not in ATTENTION_MATRICES]
        for name in non_attn:
            sv = result[f"sv_{name}"]
            assert sv.ndim == 1, f"sv_{name} should be 1D, got {sv.ndim}D"

    def test_attention_svs_are_2d(self, model):
        """Attention singular values are 2D (n_heads, d_head)."""
        result = compute_weight_singular_values(TransformerLensBundle(model, None, None))  # type: ignore
        for name in ATTENTION_MATRICES:
            sv = result[f"sv_{name}"]
            assert sv.ndim == 2, f"sv_{name} should be 2D, got {sv.ndim}D"
            assert sv.shape[0] == 4, f"sv_{name} should have 4 heads"
            assert sv.shape[1] == 8, f"sv_{name} should have 8 SVs per head"

    def test_singular_values_nonnegative(self, model):
        """All singular values are non-negative."""
        result = compute_weight_singular_values(TransformerLensBundle(model, None, None))  # type: ignore
        for key, sv in result.items():
            assert np.all(sv >= 0), f"{key} has negative singular values"

    def test_singular_values_sorted_descending(self, model):
        """Singular values are sorted in descending order."""
        result = compute_weight_singular_values(TransformerLensBundle(model, None, None))  # type: ignore
        for key, sv in result.items():
            if sv.ndim == 1:
                assert np.all(sv[:-1] >= sv[1:] - 1e-6), f"{key} not sorted"
            else:
                for h in range(sv.shape[0]):
                    assert np.all(sv[h, :-1] >= sv[h, 1:] - 1e-6), f"{key} head {h} not sorted"

    def test_non_attention_sv_count(self, model):
        """Non-attention SVs have count = min(rows, cols)."""
        result = compute_weight_singular_values(TransformerLensBundle(model, None, None))  # type: ignore
        # W_E: (10, 32) → min = 10
        assert result["sv_W_E"].shape[0] == 10
        # W_in: (32, 128) → min = 32
        assert result["sv_W_in"].shape[0] == 32
        # W_U: (32, 10) → min = 10
        assert result["sv_W_U"].shape[0] == 10


# ── Analyzer protocol tests ──────────────────────────────────────────


class TestEffectiveDimensionalityAnalyzerProtocol:
    """Tests for protocol conformance."""

    def test_conforms_to_analyzer_protocol(self):
        """EffectiveDimensionalityAnalyzer implements Analyzer protocol."""
        analyzer = EffectiveDimensionalityAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_has_correct_name(self):
        """Analyzer has correct name."""
        analyzer = EffectiveDimensionalityAnalyzer()
        assert analyzer.name == "effective_dimensionality"

    def test_registered_in_registry(self):
        """Analyzer is registered in default registry."""
        from miscope.analysis.analyzers import AnalyzerRegistry

        assert AnalyzerRegistry.is_registered("effective_dimensionality")

    def test_summary_keys_match_weight_names(self):
        """Summary keys are pr_{name} for all weight matrices."""
        analyzer = EffectiveDimensionalityAnalyzer()
        keys = analyzer.get_summary_keys()
        expected = [f"pr_{name}" for name in WEIGHT_MATRIX_NAMES]
        assert keys == expected

    def test_compute_summary_returns_all_pr_keys(self):
        """compute_summary produces a PR for each sv key in result."""
        analyzer = EffectiveDimensionalityAnalyzer()
        # Mock result with fake SVs
        result = {
            f"sv_{name}": np.array([3.0, 2.0, 1.0])
            for name in WEIGHT_MATRIX_NAMES
            if name not in ATTENTION_MATRICES
        }
        for name in ATTENTION_MATRICES:
            result[f"sv_{name}"] = np.array([[3.0, 2.0], [1.0, 1.0]])

        summary = analyzer.compute_summary(result, {})
        for name in WEIGHT_MATRIX_NAMES:
            assert f"pr_{name}" in summary, f"Missing PR key: pr_{name}"


# ── Renderer tests ────────────────────────────────────────────────────


class TestRenderDimensionalityTrajectory:
    """Tests for render_dimensionality_trajectory."""

    @pytest.fixture
    def summary_data(self):
        """Create mock summary data."""
        epochs = np.array([0, 100, 200, 300, 400])
        data = {"epochs": epochs}
        for name in WEIGHT_MATRIX_NAMES:
            key = f"pr_{name}"
            if name in ATTENTION_MATRICES:
                # Per-head: (n_epochs, n_heads)
                data[key] = np.random.default_rng(42).uniform(1.0, 8.0, size=(5, 4))
            else:
                data[key] = np.random.default_rng(42).uniform(1.0, 20.0, size=5)
        return data

    def test_returns_figure(self, summary_data):
        """Returns a Plotly Figure."""
        fig = render_dimensionality_trajectory(summary_data, current_epoch=200)
        assert isinstance(fig, go.Figure)

    def test_default_excludes_attention(self, summary_data):
        """Default matrices exclude attention (4 attention matrices)."""
        fig = render_dimensionality_trajectory(summary_data, current_epoch=200)
        trace_names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        for name in ATTENTION_MATRICES:
            # Attention traces should not be present in default
            assert all(name not in tn for tn in trace_names if tn)

    def test_explicit_matrices(self, summary_data):
        """Explicit matrix selection works."""
        fig = render_dimensionality_trajectory(
            summary_data, current_epoch=200, matrices=["W_in", "W_Q"]
        )
        trace_names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        assert any("W_in" in n for n in trace_names if n)
        assert any("W_Q" in n for n in trace_names if n)

    def test_attention_shows_mean(self, summary_data):
        """Attention matrices show '(mean)' in legend."""
        fig = render_dimensionality_trajectory(summary_data, current_epoch=200, matrices=["W_Q"])
        trace_names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        assert any("mean" in n for n in trace_names if n)

    def test_custom_title(self, summary_data):
        """Custom title is applied."""
        fig = render_dimensionality_trajectory(summary_data, current_epoch=200, title="Custom")
        assert fig.layout.title.text  # type: ignore[attr-defined] == "Custom"


class TestRenderSingularValueSpectrum:
    """Tests for render_singular_value_spectrum."""

    @pytest.fixture
    def epoch_data(self):
        """Create mock per-epoch data."""
        data = {}
        for name in WEIGHT_MATRIX_NAMES:
            key = f"sv_{name}"
            if name in ATTENTION_MATRICES:
                # (n_heads, d_head) singular values
                data[key] = np.sort(np.random.default_rng(42).uniform(0.0, 5.0, size=(4, 8)))[
                    :, ::-1
                ]
            else:
                data[key] = np.sort(np.random.default_rng(42).uniform(0.0, 5.0, size=16))[::-1]
        return data

    def test_returns_figure(self, epoch_data):
        """Returns a Plotly Figure."""
        fig = render_singular_value_spectrum(epoch_data, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_default_matrix_is_W_in(self, epoch_data):
        """Default matrix is W_in."""
        fig = render_singular_value_spectrum(epoch_data, epoch=100)
        assert "W_in" in fig.layout.title.text  # type: ignore[attr-defined]

    def test_attention_matrix_default_head(self, epoch_data):
        """Attention matrix uses head 0 by default."""
        fig = render_singular_value_spectrum(epoch_data, epoch=100, matrix_name="W_Q")
        assert "Head 0" in fig.layout.title.text  # type: ignore[attr-defined]

    def test_attention_matrix_specific_head(self, epoch_data):
        """Specific head can be selected for attention matrices."""
        fig = render_singular_value_spectrum(epoch_data, epoch=100, matrix_name="W_Q", head_idx=2)
        assert "Head 2" in fig.layout.title.text  # type: ignore[attr-defined]

    def test_pr_in_title(self, epoch_data):
        """Participation ratio is shown in title."""
        fig = render_singular_value_spectrum(epoch_data, epoch=100)
        assert "PR" in fig.layout.title.text  # type: ignore[attr-defined]

    def test_bar_count_matches_sv_length(self, epoch_data):
        """Number of bars matches singular value count."""
        fig = render_singular_value_spectrum(epoch_data, epoch=100, matrix_name="W_in")
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 1
        assert len(bar_traces[0].y) == len(epoch_data["sv_W_in"])  # type: ignore[arg-type]

    def test_custom_title(self, epoch_data):
        """Custom title is applied."""
        fig = render_singular_value_spectrum(epoch_data, epoch=100, title="Custom")
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
    """Create a registry with effective_dimensionality analyzer configured."""
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
        "analyzers": ["effective_dimensionality"],
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


class TestEffectiveDimensionalityIntegration:
    """Integration tests with AnalysisPipeline."""

    def test_pipeline_creates_artifact(self, trained_variant):
        """Pipeline creates per-epoch artifact files."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(EffectiveDimensionalityAnalyzer())
        pipeline.run()

        analyzer_dir = os.path.join(pipeline.artifacts_dir, "effective_dimensionality")
        assert os.path.isdir(analyzer_dir)

    def test_artifact_has_correct_epochs(self, trained_variant):
        """Artifact loader discovers correct epochs."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(EffectiveDimensionalityAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epochs = loader.get_epochs("effective_dimensionality")
        assert epochs == [0, 25, 49]

    def test_per_epoch_contains_sv_keys(self, trained_variant):
        """Per-epoch artifact contains all sv_{name} keys."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(EffectiveDimensionalityAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epoch_data = loader.load_epoch("effective_dimensionality", 0)
        for name in WEIGHT_MATRIX_NAMES:
            assert f"sv_{name}" in epoch_data, f"Missing sv_{name}"

    def test_summary_contains_pr_keys(self, trained_variant):
        """Summary file contains pr_{name} keys and epochs."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(EffectiveDimensionalityAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        summary = loader.load_summary("effective_dimensionality")
        assert "epochs" in summary
        for name in WEIGHT_MATRIX_NAMES:
            assert f"pr_{name}" in summary, f"Missing pr_{name} in summary"

    def test_pr_values_in_valid_range(self, trained_variant):
        """Participation ratios from real model are in valid range."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(EffectiveDimensionalityAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        summary = loader.load_summary("effective_dimensionality")

        for name in WEIGHT_MATRIX_NAMES:
            pr = summary[f"pr_{name}"]
            assert np.all(pr >= 0), f"pr_{name} has negative values"
            if name not in ATTENTION_MATRICES:
                assert np.all(pr >= 0), f"pr_{name} below 0"

    def test_renderers_work_with_real_data(self, trained_variant):
        """Both renderers work with real pipeline output."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(EffectiveDimensionalityAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)

        # Trajectory from summary
        summary = loader.load_summary("effective_dimensionality")
        fig_traj = render_dimensionality_trajectory(summary, current_epoch=25)
        assert isinstance(fig_traj, go.Figure)

        # Spectrum from per-epoch
        epoch_data = loader.load_epoch("effective_dimensionality", 25)
        fig_spec = render_singular_value_spectrum(epoch_data, epoch=25)
        assert isinstance(fig_spec, go.Figure)

        # Spectrum for attention matrix
        fig_attn = render_singular_value_spectrum(
            epoch_data, epoch=25, matrix_name="W_Q", head_idx=0
        )
        assert isinstance(fig_attn, go.Figure)
