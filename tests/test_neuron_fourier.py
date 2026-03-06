"""Tests for REQ_049: Neuron Fourier Decomposition."""

import json
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.analysis.analyzers import NeuronFourierAnalyzer
from miscope.analysis.analyzers.registry import AnalyzerRegistry
from miscope.analysis.library.fourier import extract_frequency_pairs, get_fourier_basis
from miscope.analysis.protocols import SecondaryAnalyzer

# ── Test parameters ────────────────────────────────────────────────────

PRIME = 17  # small prime for fast tests
D_MODEL = 32
D_MLP = 16
K = (PRIME - 1) // 2  # 8 frequency pairs for p=17


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_snapshot(
    p: int = PRIME, d_model: int = D_MODEL, d_mlp: int = D_MLP, seed: int = 0
) -> dict[str, np.ndarray]:
    """Create a fake parameter_snapshot artifact."""
    rng = np.random.default_rng(seed)
    return {
        "W_E": rng.normal(size=(p + 1, d_model)).astype(np.float32),
        "W_pos": rng.normal(size=(3, d_model)).astype(np.float32),
        "W_in": rng.normal(size=(d_model, d_mlp)).astype(np.float32),
        "W_out": rng.normal(size=(d_mlp, d_model)).astype(np.float32),
        "W_U": rng.normal(size=(d_model, p)).astype(np.float32),
    }


def _make_context(p: int = PRIME) -> dict:
    """Create a fake analysis context with fourier_basis."""
    fourier_basis, _ = get_fourier_basis(p, device="cpu")
    return {"params": {"prime": p}, "fourier_basis": fourier_basis}


@pytest.fixture
def snapshot():
    return _make_snapshot()


@pytest.fixture
def context():
    return _make_context()


# ── Protocol conformance ───────────────────────────────────────────────


class TestNeuronFourierProtocol:
    def test_conforms_to_secondary_analyzer(self):
        assert isinstance(NeuronFourierAnalyzer(), SecondaryAnalyzer)

    def test_name(self):
        assert NeuronFourierAnalyzer().name == "neuron_fourier"

    def test_depends_on(self):
        assert NeuronFourierAnalyzer().depends_on == "parameter_snapshot"

    def test_registered_in_registry(self):
        assert "neuron_fourier" in AnalyzerRegistry._secondary_analyzers


# ── Output shape and dtype ─────────────────────────────────────────────


class TestNeuronFourierOutputShape:
    def test_alpha_mk_shape(self, snapshot, context):
        result = NeuronFourierAnalyzer().analyze(snapshot, context)
        assert result["alpha_mk"].shape == (D_MLP, K)

    def test_phi_mk_shape(self, snapshot, context):
        result = NeuronFourierAnalyzer().analyze(snapshot, context)
        assert result["phi_mk"].shape == (D_MLP, K)

    def test_beta_mk_shape(self, snapshot, context):
        result = NeuronFourierAnalyzer().analyze(snapshot, context)
        assert result["beta_mk"].shape == (D_MLP, K)

    def test_psi_mk_shape(self, snapshot, context):
        result = NeuronFourierAnalyzer().analyze(snapshot, context)
        assert result["psi_mk"].shape == (D_MLP, K)

    def test_freq_indices_shape(self, snapshot, context):
        result = NeuronFourierAnalyzer().analyze(snapshot, context)
        assert result["freq_indices"].shape == (K,)

    def test_freq_indices_values(self, snapshot, context):
        result = NeuronFourierAnalyzer().analyze(snapshot, context)
        np.testing.assert_array_equal(result["freq_indices"], np.arange(1, K + 1))

    def test_float32_dtype(self, snapshot, context):
        result = NeuronFourierAnalyzer().analyze(snapshot, context)
        for key in ("alpha_mk", "phi_mk", "beta_mk", "psi_mk"):
            assert result[key].dtype == np.float32, f"{key} should be float32"

    def test_int32_dtype_freq_indices(self, snapshot, context):
        result = NeuronFourierAnalyzer().analyze(snapshot, context)
        assert result["freq_indices"].dtype == np.int32


# ── Numerical properties ───────────────────────────────────────────────


class TestNeuronFourierNumerical:
    def test_magnitudes_non_negative(self, snapshot, context):
        result = NeuronFourierAnalyzer().analyze(snapshot, context)
        assert (result["alpha_mk"] >= 0).all()
        assert (result["beta_mk"] >= 0).all()

    def test_phases_in_valid_range(self, snapshot, context):
        result = NeuronFourierAnalyzer().analyze(snapshot, context)
        assert (result["phi_mk"] >= -np.pi).all() and (result["phi_mk"] <= np.pi).all()
        assert (result["psi_mk"] >= -np.pi).all() and (result["psi_mk"] <= np.pi).all()

    def test_perfectly_specialized_neuron(self):
        """A neuron whose weights are a pure sinusoid at frequency k* should have
        magnitude concentrated at k*, near zero everywhere else."""
        p = 17
        d_model = p
        d_mlp = 4
        k_star = 3  # target frequency

        # Build W_E = identity (token x maps to standard basis vector e_x)
        W_E = np.zeros((p + 1, d_model), dtype=np.float32)
        W_E[:p] = np.eye(p, dtype=np.float32)  # identity for token positions

        # W_in: neuron 0 gets a pure cos(k*) pattern across token dimension
        # W_U = identity for output dimension
        # This makes theta_0 = W_E[:p] @ W_in[:, 0]
        #                     = I @ w_in_col_0 = w_in_col_0
        # We want theta_0 to be proportional to cos(k*, .)
        omega = 2 * np.pi * k_star / p
        cos_signal = np.array([np.cos(omega * j) for j in range(p)], dtype=np.float32)
        cos_signal = cos_signal / np.linalg.norm(cos_signal)

        W_in = np.zeros((d_model, d_mlp), dtype=np.float32)
        W_in[:p, 0] = cos_signal  # only neuron 0 has signal

        # W_out and W_U as identity-like (output side)
        W_out = np.zeros((d_mlp, d_model), dtype=np.float32)
        W_out[0, :p] = cos_signal  # neuron 0 output also cos(k*)

        W_U = np.eye(d_model, p, dtype=np.float32)

        snapshot = {
            "W_E": W_E,
            "W_in": W_in,
            "W_out": W_out,
            "W_U": W_U,
        }
        context = _make_context(p)

        result = NeuronFourierAnalyzer().analyze(snapshot, context)

        alpha_neuron0 = result["alpha_mk"][0]  # (K,) magnitudes for neuron 0
        dominant_freq_idx = int(np.argmax(alpha_neuron0))
        # freq_indices are 1-based, so k* should be at index k*-1
        assert dominant_freq_idx == k_star - 1, (
            f"Expected dominant frequency at index {k_star - 1}, got {dominant_freq_idx}"
        )


# ── extract_frequency_pairs unit tests ────────────────────────────────


class TestExtractFrequencyPairs:
    def test_output_shapes(self):
        p = 17
        K = (p - 1) // 2
        M = 8
        rng = np.random.default_rng(42)
        coeffs = rng.normal(size=(p, M)).astype(np.float32)
        mags, phases = extract_frequency_pairs(coeffs, p)
        assert mags.shape == (M, K)
        assert phases.shape == (M, K)

    def test_magnitudes_non_negative(self):
        p = 13
        M = 5
        rng = np.random.default_rng(0)
        coeffs = rng.normal(size=(p, M)).astype(np.float32)
        mags, _ = extract_frequency_pairs(coeffs, p)
        assert (mags >= 0).all()

    def test_phases_in_range(self):
        p = 13
        M = 5
        rng = np.random.default_rng(1)
        coeffs = rng.normal(size=(p, M)).astype(np.float32)
        _, phases = extract_frequency_pairs(coeffs, p)
        assert (phases >= -np.pi).all() and (phases <= np.pi).all()

    def test_pure_cos_gives_correct_magnitude(self):
        """A coefficient vector that is purely cos(k*) should give magnitude ≈ 1
        at k* and ≈ 0 elsewhere (using normalized basis)."""
        p = 17
        k_star = 3
        # K = (p - 1) // 2

        basis, _ = get_fourier_basis(p)
        basis_np = basis.numpy()  # (p, p)

        # Pure cos(k*) signal in token space
        omega = 2 * np.pi * k_star / p
        signal = np.array([[np.cos(omega * j)] for j in range(p)], dtype=np.float32)  # (p, 1)

        # Project: fourier_coeffs = basis_np @ signal → (p, 1)
        coeffs = basis_np @ signal  # (p, 1)

        mags, phases = extract_frequency_pairs(coeffs, p)  # (1, K)
        mags_neuron = mags[0]  # (K,)

        # k_star is at index k_star - 1 (1-based)
        dominant_idx = int(np.argmax(mags_neuron))
        assert dominant_idx == k_star - 1

        # Magnitude at dominant frequency should be much larger than others
        dominant_mag = mags_neuron[dominant_idx]
        other_mags = np.concatenate([mags_neuron[:dominant_idx], mags_neuron[dominant_idx + 1 :]])
        assert dominant_mag > 5 * other_mags.max() if len(other_mags) > 0 else True

    def test_phase_convention(self):
        """Verify atan2(-sin, cos) convention."""
        p = 13
        k = 2
        # Build a coefficient vector with only sin_k component nonzero
        coeffs = np.zeros((p, 1), dtype=np.float32)
        sin_idx = 2 * k - 1
        coeffs[sin_idx, 0] = 1.0  # pure sin component

        _, phases = extract_frequency_pairs(coeffs, p)
        # atan2(-sin_coeff, cos_coeff) = atan2(-1, 0) = -π/2
        expected_phase = np.arctan2(-1.0, 0.0)
        assert abs(phases[0, k - 1] - expected_phase) < 1e-5


# ── Renderer tests ─────────────────────────────────────────────────────


class TestNeuronFourierRenderers:
    @pytest.fixture
    def fake_artifact(self):
        rng = np.random.default_rng(42)
        return {
            "alpha_mk": rng.uniform(0, 1, size=(D_MLP, K)).astype(np.float32),
            "phi_mk": rng.uniform(-np.pi, np.pi, size=(D_MLP, K)).astype(np.float32),
            "beta_mk": rng.uniform(0, 1, size=(D_MLP, K)).astype(np.float32),
            "psi_mk": rng.uniform(-np.pi, np.pi, size=(D_MLP, K)).astype(np.float32),
            "freq_indices": np.arange(1, K + 1, dtype=np.int32),
        }

    def test_render_input_heatmap_returns_figure(self, fake_artifact):
        from miscope.visualization.renderers.neuron_fourier import render_neuron_fourier_heatmap

        fig = render_neuron_fourier_heatmap(fake_artifact, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_render_output_heatmap_returns_figure(self, fake_artifact):
        from miscope.visualization.renderers.neuron_fourier import (
            render_neuron_fourier_heatmap_output,
        )

        fig = render_neuron_fourier_heatmap_output(fake_artifact, epoch=100)
        assert isinstance(fig, go.Figure)


"""
    def test_heatmap_has_correct_dimensions(self, fake_artifact):
        from miscope.visualization.renderers.neuron_fourier import render_neuron_fourier_heatmap

        fig = render_neuron_fourier_heatmap(fake_artifact, epoch=0)
        heatmap = fig.data[0]
        assert heatmap.z.shape == (D_MLP, K)

    def test_input_and_output_differ(self, fake_artifact):
        from miscope.visualization.renderers.neuron_fourier import (
            render_neuron_fourier_heatmap,
            render_neuron_fourier_heatmap_output,
        )

        fig_in = render_neuron_fourier_heatmap(fake_artifact, epoch=0)
        fig_out = render_neuron_fourier_heatmap_output(fake_artifact, epoch=0)
        # Different data (alpha vs beta)
        assert not np.allclose(fig_in.data[0].z, fig_out.data[0].z)
    def test_custom_title(self, fake_artifact):
        from miscope.visualization.renderers.neuron_fourier import render_neuron_fourier_heatmap

        fig = render_neuron_fourier_heatmap(fake_artifact, epoch=0, title="My Title")
        assert "My Title" in fig.layout.title.text
"""


# ── View catalog integration ───────────────────────────────────────────


class TestNeuronFourierViews:
    def test_views_registered(self):
        from miscope.views.catalog import _catalog

        names = _catalog.names()
        assert "activations.mlp.neuron_fourier_heatmap" in names
        assert "activations.mlp.neuron_fourier_heatmap_output" in names


# ── Pipeline integration ───────────────────────────────────────────────


class TestNeuronFourierPipelineIntegration:
    @pytest.fixture
    def temp_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_families_dir = Path(tmpdir) / "model_families"
            results_dir = Path(tmpdir) / "results"
            model_families_dir.mkdir()
            results_dir.mkdir()
            yield model_families_dir, results_dir

    @pytest.fixture
    def trained_variant(self, temp_dirs):
        from miscope.families import FamilyRegistry

        model_families_dir, results_dir = temp_dirs
        family_dir = model_families_dir / "modulo_addition_1layer"
        family_dir.mkdir()
        family_json = {
            "name": "modulo_addition_1layer",
            "display_name": "Test",
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
                "seed": {"type": "int", "description": "Seed", "default": 999},
            },
            "analyzers": ["parameter_snapshot"],
            "secondary_analyzers": ["neuron_fourier"],
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
        variant.train(num_epochs=30, checkpoint_epochs=[0, 15, 29], device="cpu")
        return variant

    def test_pipeline_produces_neuron_fourier_artifacts(self, trained_variant):
        from miscope.analysis import AnalysisPipeline
        from miscope.analysis.analyzers import NeuronFourierAnalyzer, ParameterSnapshotAnalyzer

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_secondary(NeuronFourierAnalyzer())
        pipeline.run()

        epochs = pipeline.get_completed_epochs("neuron_fourier")
        snapshot_epochs = pipeline.get_completed_epochs("parameter_snapshot")
        assert epochs == snapshot_epochs

    def test_artifacts_have_correct_shapes(self, trained_variant):
        from miscope.analysis import AnalysisPipeline, ArtifactLoader
        from miscope.analysis.analyzers import NeuronFourierAnalyzer, ParameterSnapshotAnalyzer

        p = 17
        d_mlp = 512
        K = (p - 1) // 2

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register_secondary(NeuronFourierAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epochs = pipeline.get_completed_epochs("neuron_fourier")
        data = loader.load_epoch("neuron_fourier", epochs[0])

        assert data["alpha_mk"].shape == (d_mlp, K)
        assert data["beta_mk"].shape == (d_mlp, K)
        assert data["freq_indices"].shape == (K,)
        assert (data["alpha_mk"] >= 0).all()
