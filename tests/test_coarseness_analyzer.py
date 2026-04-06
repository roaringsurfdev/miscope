"""Tests for REQ_023: Coarseness Analyzer."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from miscope.analysis import AnalysisPipeline, Analyzer, ArtifactLoader
from miscope.analysis.analyzers import CoarsenessAnalyzer
from miscope.analysis.bundle import TransformerLensBundle
from miscope.analysis.library.fourier import compute_neuron_coarseness
from miscope.families import FamilyRegistry

# ── Library function tests ──────────────────────────────────────────────


class TestComputeNeuronCoarseness:
    """Tests for compute_neuron_coarseness library function."""

    def test_values_in_unit_interval(self):
        """Coarseness values are in [0, 1] for valid inputs."""
        n_freq, d_mlp = 8, 32
        # Create normalized fractions (columns sum to 1)
        raw = torch.rand(n_freq, d_mlp)
        freq_fractions = raw / raw.sum(dim=0, keepdim=True)

        coarseness = compute_neuron_coarseness(freq_fractions, n_low_freqs=3)

        assert coarseness.shape == (d_mlp,)
        assert torch.all(coarseness >= 0.0)
        assert torch.all(coarseness <= 1.0)

    def test_high_coarseness_for_low_freq_energy(self):
        """Neurons with energy concentrated in low frequencies have high coarseness."""
        n_freq, d_mlp = 8, 4
        freq_fractions = torch.zeros(n_freq, d_mlp)
        # All energy in frequency k=1 (index 0)
        freq_fractions[0, :] = 1.0

        coarseness = compute_neuron_coarseness(freq_fractions, n_low_freqs=3)

        assert torch.all(coarseness == 1.0)

    def test_low_coarseness_for_high_freq_energy(self):
        """Neurons with energy in high frequencies have low coarseness."""
        n_freq, d_mlp = 8, 4
        freq_fractions = torch.zeros(n_freq, d_mlp)
        # All energy in highest frequencies (beyond k=3)
        freq_fractions[5:, :] = 1.0 / 3.0

        coarseness = compute_neuron_coarseness(freq_fractions, n_low_freqs=3)

        assert torch.all(coarseness == 0.0)

    def test_output_shape(self):
        """Output shape matches d_mlp dimension."""
        freq_fractions = torch.rand(10, 64)
        coarseness = compute_neuron_coarseness(freq_fractions, n_low_freqs=3)
        assert coarseness.shape == (64,)

    def test_n_low_freqs_parameter(self):
        """Changing n_low_freqs changes the result."""
        n_freq, d_mlp = 8, 4
        freq_fractions = torch.zeros(n_freq, d_mlp)
        # Spread energy across first 5 frequencies equally
        freq_fractions[:5, :] = 0.2

        coarseness_3 = compute_neuron_coarseness(freq_fractions, n_low_freqs=3)
        coarseness_5 = compute_neuron_coarseness(freq_fractions, n_low_freqs=5)

        assert torch.allclose(coarseness_3, torch.tensor([0.6]))
        assert torch.allclose(coarseness_5, torch.tensor([1.0]))

    def test_n_low_freqs_clamped_to_available(self):
        """n_low_freqs larger than available frequencies is handled gracefully."""
        freq_fractions = torch.ones(3, 4) / 3.0
        coarseness = compute_neuron_coarseness(freq_fractions, n_low_freqs=10)
        assert torch.allclose(coarseness, torch.tensor([1.0]))


# ── Analyzer protocol tests ────────────────────────────────────────────


class TestCoarsenessAnalyzerProtocol:
    """Tests for protocol conformance."""

    def test_conforms_to_analyzer_protocol(self):
        """CoarsenessAnalyzer implements Analyzer protocol."""
        analyzer = CoarsenessAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_has_name(self):
        """Analyzer has correct name."""
        analyzer = CoarsenessAnalyzer()
        assert analyzer.name == "coarseness"

    def test_has_analyze_method(self):
        """Analyzer has analyze method."""
        analyzer = CoarsenessAnalyzer()
        assert callable(analyzer.analyze)

    def test_has_summary_methods(self):
        """Analyzer has summary statistics methods."""
        analyzer = CoarsenessAnalyzer()
        assert callable(analyzer.get_summary_keys)
        assert callable(analyzer.compute_summary)


class TestCoarsenessAnalyzerSummary:
    """Tests for summary statistics."""

    def test_summary_keys(self):
        """get_summary_keys returns expected keys."""
        analyzer = CoarsenessAnalyzer()
        keys = analyzer.get_summary_keys()
        assert "mean_coarseness" in keys
        assert "std_coarseness" in keys
        assert "median_coarseness" in keys
        assert "p25_coarseness" in keys
        assert "p75_coarseness" in keys
        assert "blob_count" in keys
        assert "coarseness_hist" in keys

    def test_summary_keys_match_compute_output(self):
        """Keys from get_summary_keys match keys from compute_summary."""
        analyzer = CoarsenessAnalyzer()
        keys = set(analyzer.get_summary_keys())
        result = {"coarseness": np.random.rand(512).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert set(summary.keys()) == keys

    def test_summary_scalar_shapes(self):
        """Scalar summary values are floats."""
        analyzer = CoarsenessAnalyzer()
        result = {"coarseness": np.random.rand(512).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})

        for key in [
            "mean_coarseness",
            "std_coarseness",
            "median_coarseness",
            "p25_coarseness",
            "p75_coarseness",
            "blob_count",
        ]:
            assert isinstance(summary[key], float), f"{key} should be float"

    def test_summary_histogram_shape(self):
        """Histogram has 20 bins."""
        analyzer = CoarsenessAnalyzer()
        result = {"coarseness": np.random.rand(512).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})
        hist = summary["coarseness_hist"]
        assert isinstance(hist, np.ndarray)
        assert hist.shape == (20,)

    def test_blob_count_with_known_data(self):
        """blob_count correctly counts neurons above threshold."""
        analyzer = CoarsenessAnalyzer(blob_threshold=0.7)
        # 10 neurons: 3 above 0.7, 7 below
        coarseness = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        result = {"coarseness": coarseness}
        summary = analyzer.compute_summary(result, {})
        assert summary["blob_count"] == 4.0  # 0.7, 0.8, 0.9, 0.95


# ── Integration tests ──────────────────────────────────────────────────


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
    """Create a registry with the modulo addition family including coarseness."""
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
        "analyzers": [
            "dominant_frequencies",
            "neuron_activations",
            "neuron_freq_norm",
            "coarseness",
        ],
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


class TestCoarsenessAnalyzerOutput:
    """Tests for analyzer output shape and values."""

    @pytest.fixture
    def model_with_context(self, trained_variant):
        """Create model, run forward pass, return model, probe, cache, and context."""
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
        analyzer = CoarsenessAnalyzer()
        result = analyzer.analyze(bundle, probe, context)
        assert isinstance(result, dict)

    def test_returns_coarseness_key(self, model_with_context):
        """Result contains 'coarseness' key."""
        bundle, probe, context = model_with_context
        analyzer = CoarsenessAnalyzer()
        result = analyzer.analyze(bundle, probe, context)
        assert "coarseness" in result

    def test_coarseness_is_numpy_array(self, model_with_context):
        """Coarseness is a numpy array."""
        bundle, probe, context = model_with_context
        analyzer = CoarsenessAnalyzer()
        result = analyzer.analyze(bundle, probe, context)
        assert isinstance(result["coarseness"], np.ndarray)

    def test_coarseness_shape(self, model_with_context):
        """Coarseness has shape (d_mlp,)."""
        bundle, probe, context = model_with_context
        analyzer = CoarsenessAnalyzer()
        result = analyzer.analyze(bundle, probe, context)
        # d_mlp = 512 from architecture
        assert result["coarseness"].shape == (512,)

    def test_coarseness_values_in_unit_interval(self, model_with_context):
        """Coarseness values are in [0, 1]."""
        bundle, probe, context = model_with_context
        analyzer = CoarsenessAnalyzer()
        result = analyzer.analyze(bundle, probe, context)
        assert np.all(result["coarseness"] >= 0.0)
        assert np.all(result["coarseness"] <= 1.0)


class TestCoarsenessAnalyzerIntegration:
    """Integration tests with AnalysisPipeline."""

    def test_pipeline_creates_artifact(self, trained_variant):
        """Pipeline creates per-epoch artifact files."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(CoarsenessAnalyzer())
        pipeline.run()

        analyzer_dir = os.path.join(pipeline.artifacts_dir, "coarseness")
        assert os.path.isdir(analyzer_dir)

    def test_artifact_contains_epochs(self, trained_variant):
        """Artifact loader discovers correct epochs."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(CoarsenessAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        artifact = loader.load("coarseness")
        assert "epochs" in artifact
        np.testing.assert_array_equal(artifact["epochs"], [0, 25, 49])

    def test_per_epoch_artifact_shape(self, trained_variant):
        """Per-epoch artifact has correct shape."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(CoarsenessAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epoch_data = loader.load_epoch("coarseness", 0)
        assert "coarseness" in epoch_data
        assert epoch_data["coarseness"].shape == (512,)

    def test_summary_file_created(self, trained_variant):
        """Pipeline creates summary.npz for coarseness analyzer."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(CoarsenessAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        assert loader.has_summary("coarseness")

    def test_summary_contents(self, trained_variant):
        """Summary file contains expected keys and shapes."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(CoarsenessAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        summary = loader.load_summary("coarseness")

        assert "epochs" in summary
        np.testing.assert_array_equal(summary["epochs"], [0, 25, 49])

        # Scalar stats: one value per epoch
        for key in [
            "mean_coarseness",
            "std_coarseness",
            "median_coarseness",
            "p25_coarseness",
            "p75_coarseness",
            "blob_count",
        ]:
            assert key in summary, f"Missing key: {key}"
            assert summary[key].shape == (3,), f"{key} wrong shape"

        # Histogram: (n_epochs, 20)
        assert "coarseness_hist" in summary
        assert summary["coarseness_hist"].shape == (3, 20)
