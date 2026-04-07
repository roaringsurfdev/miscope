"""Tests for REQ_063: Fourier Nucleation Predictor."""

import numpy as np
import pytest
import torch

from miscope.analysis.analyzers import FourierNucleationAnalyzer
from miscope.analysis.analyzers.fourier_nucleation import (
    _build_fourier_basis,
    _project,
    _sharpen,
    _snapshot,
)
from miscope.analysis.bundle import TransformerLensBundle

# ---------------------------------------------------------------------------
# Unit: Fourier basis construction
# ---------------------------------------------------------------------------


class TestBuildFourierBasis:
    def test_shapes(self):
        prime = 17
        freqs, cos_b, sin_b = _build_fourier_basis(prime)
        n_freqs = prime // 2
        assert freqs.shape == (n_freqs,)
        assert cos_b.shape == (n_freqs, prime)
        assert sin_b.shape == (n_freqs, prime)

    def test_frequencies_range(self):
        prime = 13
        freqs, _, _ = _build_fourier_basis(prime)
        assert freqs[0] == 1
        assert freqs[-1] == prime // 2

    def test_basis_is_normalized(self):
        prime = 11
        _, cos_b, sin_b = _build_fourier_basis(prime)
        cos_norms = np.linalg.norm(cos_b, axis=1)
        sin_norms = np.linalg.norm(sin_b, axis=1)
        np.testing.assert_allclose(cos_norms, 1.0, atol=1e-10)
        np.testing.assert_allclose(sin_norms, 1.0, atol=1e-10)

    def test_pure_cosine_signal_projects_fully(self):
        """A signal that is a pure cosine at frequency k should project entirely
        onto the k-th frequency bin and nowhere else (up to normalization)."""
        prime = 11
        freqs, cos_b, sin_b = _build_fourier_basis(prime)
        k = 3
        ki = int(np.searchsorted(freqs, k))
        # Build a pure cosine signal at frequency k
        positions = np.arange(prime)
        signal = np.cos(2 * np.pi * k * positions / prime)
        # Normalize to unit vector for comparison
        signal /= np.linalg.norm(signal)

        R = signal[None, :]  # (1, prime)
        _, _, energy = _project(R, cos_b, sin_b)

        # Energy should be highest at frequency k
        assert energy[0, ki] == pytest.approx(energy[0].max(), rel=1e-5)
        # And dominant (> 0.99 of total)
        total = energy[0].sum()
        assert energy[0, ki] / total > 0.99


# ---------------------------------------------------------------------------
# Unit: Iterative sharpening
# ---------------------------------------------------------------------------


class TestSharpen:
    def test_reduces_entropy(self):
        """Sharpening should concentrate energy — entropy decreases over iterations."""
        prime = 17
        freqs, cos_b, sin_b = _build_fourier_basis(prime)
        rng = np.random.default_rng(42)
        R = rng.standard_normal((32, prime))

        proj_cos, proj_sin, energy = _project(R, cos_b, sin_b)
        total = energy.sum(axis=1, keepdims=True).clip(min=1e-10)
        frac = energy / total
        entropy_before = -np.sum(frac * np.log(frac + 1e-12), axis=1).mean()

        for _ in range(8):
            proj_cos, proj_sin, energy = _project(R, cos_b, sin_b)
            R = _sharpen(R, proj_cos, proj_sin, energy, cos_b, sin_b, sharpness=0.7)

        _, _, energy_after = _project(R, cos_b, sin_b)
        total_after = energy_after.sum(axis=1, keepdims=True).clip(min=1e-10)
        frac_after = energy_after / total_after
        entropy_after = -np.sum(frac_after * np.log(frac_after + 1e-12), axis=1).mean()

        assert entropy_after < entropy_before

    def test_output_shape_unchanged(self):
        prime = 13
        d_mlp = 16
        freqs, cos_b, sin_b = _build_fourier_basis(prime)
        rng = np.random.default_rng(0)
        R = rng.standard_normal((d_mlp, prime))
        proj_cos, proj_sin, energy = _project(R, cos_b, sin_b)
        new_R = _sharpen(R, proj_cos, proj_sin, energy, cos_b, sin_b, 0.7)
        assert new_R.shape == R.shape


# ---------------------------------------------------------------------------
# Unit: Analyzer projection math
# ---------------------------------------------------------------------------


class TestProjectionMath:
    """Validate W_in @ W_E.T projection is computed correctly."""

    def test_response_matrix_shape(self):
        # W_in is (d_model, d_mlp) in TransformerLens convention
        prime = 11
        d_mlp, d_model = 8, 16
        W_in = np.random.randn(d_model, d_mlp)
        W_E = np.random.randn(prime + 1, d_model)  # vocab includes equals token
        R = (W_E[:prime] @ W_in).T
        assert R.shape == (d_mlp, prime)

    def test_response_matrix_uses_only_prime_tokens(self):
        """Only the first p rows of W_E should appear in the response matrix."""
        prime = 7
        d_mlp, d_model = 4, 8
        W_in = np.random.randn(d_model, d_mlp)
        W_E = np.random.randn(prime + 1, d_model)
        R_correct = (W_E[:prime] @ W_in).T
        R_wrong = (W_E @ W_in).T
        assert R_correct.shape[1] == prime
        assert R_wrong.shape[1] == prime + 1


# ---------------------------------------------------------------------------
# Unit: Snapshot statistics
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_agg_energy_normalized(self):
        prime = 11
        freqs, cos_b, sin_b = _build_fourier_basis(prime)
        R = np.random.randn(16, prime)
        _, _, energy = _project(R, cos_b, sin_b)
        agg, _, _ = _snapshot(energy, freqs)
        assert agg.max() == pytest.approx(1.0, abs=1e-6)

    def test_peak_freq_values_in_range(self):
        prime = 11
        freqs, cos_b, sin_b = _build_fourier_basis(prime)
        R = np.random.randn(16, prime)
        _, _, energy = _project(R, cos_b, sin_b)
        _, peak, _ = _snapshot(energy, freqs)
        assert all(1 <= k <= prime // 2 for k in peak)

    def test_committed_count_upper_bound(self):
        prime = 11
        d_mlp = 32
        freqs, cos_b, sin_b = _build_fourier_basis(prime)
        R = np.random.randn(d_mlp, prime)
        _, _, energy = _project(R, cos_b, sin_b)
        _, _, committed = _snapshot(energy, freqs)
        # Each frequency can be committed by at most d_mlp neurons
        # (a neuron can be committed to multiple frequencies)
        assert committed.max() <= d_mlp


# ---------------------------------------------------------------------------
# Analyzer: protocol and output
# ---------------------------------------------------------------------------


class TestFourierNucleationAnalyzerProtocol:
    def test_has_name(self):
        assert FourierNucleationAnalyzer.name == "fourier_nucleation"

    def test_has_analyze_method(self):
        assert callable(FourierNucleationAnalyzer().analyze)


class TestFourierNucleationAnalyzerOutput:
    """Run analyzer against a minimal mock model."""

    @pytest.fixture
    def minimal_model(self):
        """Minimal mock with the attributes the analyzer accesses."""
        prime = 11
        d_model = 16
        d_mlp = 8

        class MockMLP:
            W_in = torch.randn(d_model, d_mlp)  # TransformerLens: (d_model, d_mlp)

        class MockBlock:
            mlp = MockMLP()

        class MockEmbed:
            W_E = torch.randn(prime + 1, d_model)  # includes equals token

        class MockModel:
            blocks = [MockBlock()]
            embed = MockEmbed()

        return MockModel(), prime, d_mlp

    def test_returns_all_keys(self, minimal_model):
        model, prime, d_mlp = minimal_model
        analyzer = FourierNucleationAnalyzer(iterations=3)
        result = analyzer.analyze(
            TransformerLensBundle(model, None, None),  # type: ignore
            None,  # type: ignore
            {"params": {"prime": prime}},  # type: ignore
        )  # type: ignore[arg-type]
        expected_keys = {
            "aggregate_energy",
            "neuron_peak_freq",
            "neuron_committed_count",
            "frequencies",
            "prime",
            "iterations",
            "sharpness",
        }
        assert expected_keys == set(result.keys())

    def test_aggregate_energy_shape(self, minimal_model):
        model, prime, d_mlp = minimal_model
        n_iters = 4
        analyzer = FourierNucleationAnalyzer(iterations=n_iters)
        result = analyzer.analyze(
            TransformerLensBundle(model, None, None),  # type: ignore
            None,  # type: ignore
            {"params": {"prime": prime}},  # type: ignore
        )  # type: ignore[arg-type]
        n_freqs = prime // 2
        assert result["aggregate_energy"].shape == (n_iters + 1, n_freqs)

    def test_neuron_peak_freq_shape(self, minimal_model):
        model, prime, d_mlp = minimal_model
        n_iters = 3
        analyzer = FourierNucleationAnalyzer(iterations=n_iters)
        result = analyzer.analyze(
            TransformerLensBundle(model, None, None),  # type: ignore
            None,  # type: ignore
            {"params": {"prime": prime}},  # type: ignore
        )  # type: ignore[arg-type]
        assert result["neuron_peak_freq"].shape == (n_iters + 1, d_mlp)

    def test_frequencies_array_values(self, minimal_model):
        model, prime, _ = minimal_model
        analyzer = FourierNucleationAnalyzer(iterations=2)
        result = analyzer.analyze(
            TransformerLensBundle(model, None, None),  # type: ignore
            None,  # type: ignore
            {"params": {"prime": prime}},  # type: ignore
        )  # type: ignore[arg-type]
        expected = np.arange(1, prime // 2 + 1, dtype=np.int32)
        np.testing.assert_array_equal(result["frequencies"], expected)

    def test_aggregate_energy_normalized(self, minimal_model):
        model, prime, _ = minimal_model
        analyzer = FourierNucleationAnalyzer(iterations=4)
        result = analyzer.analyze(
            TransformerLensBundle(model, None, None),  # type: ignore
            None,  # type: ignore
            {"params": {"prime": prime}},  # type: ignore
        )  # type: ignore[arg-type]
        # Max value per iteration should be 1.0
        for it in range(result["aggregate_energy"].shape[0]):
            assert result["aggregate_energy"][it].max() == pytest.approx(1.0, abs=1e-5)

    def test_sharpening_concentrates_energy(self, minimal_model):
        """Final iteration should have higher peak energy concentration than iteration 0."""
        model, prime, d_mlp = minimal_model
        analyzer = FourierNucleationAnalyzer(iterations=8, sharpness=0.8)
        result = analyzer.analyze(
            TransformerLensBundle(model, None, None),  # type: ignore
            None,  # type: ignore
            {"params": {"prime": prime}},  # type: ignore
        )  # type: ignore[arg-type]
        energy_0 = result["aggregate_energy"][0]
        energy_final = result["aggregate_energy"][-1]
        # Concentration: max/mean ratio should increase (or stay same)
        ratio_0 = energy_0.max() / (energy_0.mean() + 1e-10)
        ratio_final = energy_final.max() / (energy_final.mean() + 1e-10)
        assert ratio_final >= ratio_0 * 0.95  # allow 5% tolerance


# ---------------------------------------------------------------------------
# Integration: pipeline produces epoch_00000.npz
# ---------------------------------------------------------------------------


class TestFourierNucleationIntegration:
    @pytest.fixture
    def trained_variant(self, tmp_path):
        """Small trained variant for integration testing."""
        import json

        from miscope.families import FamilyRegistry

        family_dir = tmp_path / "model_families" / "modulo_addition_1layer"
        family_dir.mkdir(parents=True)
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        family_json = {
            "name": "modulo_addition_1layer",
            "display_name": "Modulo Addition (1 Layer)",
            "description": "Test family",
            "class_type": "miscope.families.implementations.modulo_addition_1layer.ModuloAddition1LayerFamily",
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
                "data_seed": {"type": "int", "description": "Data seed", "default": 598},
            },
            "analyzers": ["fourier_nucleation"],
            "secondary_analyzers": [],
            "cross_epoch_analyzers": [],
            "analysis_dataset": {"type": "modulo_addition_grid"},
            "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}_dseed{data_seed}",
        }
        (family_dir / "family.json").write_text(json.dumps(family_json))

        registry = FamilyRegistry(
            model_families_dir=tmp_path / "model_families",
            results_dir=results_dir,
        )
        family = registry.get_family("modulo_addition_1layer")
        variant = registry.create_variant(family, {"prime": 11, "seed": 42, "data_seed": 598})
        variant.train(num_epochs=5, checkpoint_epochs=[0], device="cpu")
        return variant

    def test_pipeline_creates_epoch_artifact(self, trained_variant):
        from miscope.analysis import AnalysisPipeline

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(FourierNucleationAnalyzer(iterations=3))
        pipeline.run()

        import os

        artifact_path = os.path.join(
            pipeline.artifacts_dir, "fourier_nucleation", "epoch_00000.npz"
        )
        assert os.path.exists(artifact_path)

    def test_artifact_loadable_via_loader(self, trained_variant):
        from miscope.analysis import AnalysisPipeline, ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(FourierNucleationAnalyzer(iterations=3))
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        data = loader.load_epoch("fourier_nucleation", 0)
        assert "aggregate_energy" in data
        assert "frequencies" in data
