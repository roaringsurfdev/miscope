"""Tests for REQ_003_002: Dominant Frequencies Analyzer."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from miscope.analysis import AnalysisPipeline, Analyzer
from miscope.analysis.analyzers import DominantFrequenciesAnalyzer
from miscope.analysis.bundle import TransformerLensBundle
from miscope.analysis.protocols import ActivationContext
from miscope.families import FamilyRegistry


class TestDominantFrequenciesAnalyzerProtocol:
    """Tests for protocol conformance."""

    def test_conforms_to_analyzer_protocol(self):
        """DominantFrequenciesAnalyzer implements Analyzer protocol."""
        analyzer = DominantFrequenciesAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_has_name_property(self):
        """Analyzer has correct name property."""
        analyzer = DominantFrequenciesAnalyzer()
        assert analyzer.name == "dominant_frequencies"

    def test_has_analyze_method(self):
        """Analyzer has analyze method."""
        analyzer = DominantFrequenciesAnalyzer()
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

    variant.train(
        num_epochs=50,
        checkpoint_epochs=[0, 25, 49],
        device="cpu",
    )
    return variant


class TestDominantFrequenciesAnalyzerOutput:
    """Tests for analyzer output shape and values."""

    @pytest.fixture
    def model_with_context(self, trained_variant):
        """Create model, run forward pass, return model, probe, cache, and context."""
        device = "cpu"  # Force CPU for consistent testing
        family = trained_variant.family
        params = trained_variant.params

        # Generate probe and context on same device
        probe = family.generate_analysis_dataset(params, device=device)
        context = family.prepare_analysis_context(params, device)

        # Create model on the same device and load checkpoint weights
        model = family.create_model(params, device=device)
        state_dict = trained_variant.load_checkpoint(49)
        model.load_state_dict(state_dict)

        # Run forward pass
        with torch.inference_mode():
            logits, cache = model.run_with_cache(probe)

        bundle = TransformerLensBundle(model, cache, logits)
        return bundle, probe, context

    def test_returns_dict(self, model_with_context):
        """analyze returns a dict."""
        bundle, probe, context = model_with_context
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(
            ActivationContext(bundle=bundle, probe=probe, analysis_params=context)
        )
        assert isinstance(result, dict)

    def test_returns_coefficients_key(self, model_with_context):
        """Result contains 'coefficients' key."""
        bundle, probe, context = model_with_context
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(
            ActivationContext(bundle=bundle, probe=probe, analysis_params=context)
        )
        assert "coefficients" in result

    def test_coefficients_is_numpy_array(self, model_with_context):
        """Coefficients is a numpy array."""
        bundle, probe, context = model_with_context
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(
            ActivationContext(bundle=bundle, probe=probe, analysis_params=context)
        )
        assert isinstance(result["coefficients"], np.ndarray)

    def test_coefficients_shape(self, model_with_context):
        """Coefficients has correct shape (n_fourier_components,)."""
        bundle, probe, context = model_with_context
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(
            ActivationContext(bundle=bundle, probe=probe, analysis_params=context)
        )

        fourier_basis = context["fourier_basis"]
        n_components = fourier_basis.shape[0]
        assert result["coefficients"].shape == (n_components,)

    def test_coefficients_are_non_negative(self, model_with_context):
        """Coefficient values are non-negative (they are norms)."""
        bundle, probe, context = model_with_context
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(
            ActivationContext(bundle=bundle, probe=probe, analysis_params=context)
        )
        assert np.all(result["coefficients"] >= 0)

    def test_coefficients_on_cpu(self, model_with_context):
        """Coefficients are on CPU (numpy array, not tensor)."""
        bundle, probe, context = model_with_context
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(
            ActivationContext(bundle=bundle, probe=probe, analysis_params=context)
        )
        assert not isinstance(result["coefficients"], torch.Tensor)


class TestDominantFrequenciesAnalyzerIntegration:
    """Integration tests with AnalysisPipeline."""

    def test_pipeline_creates_artifact(self, trained_variant):
        """Pipeline creates per-epoch artifact files."""
        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        analyzer_dir = os.path.join(pipeline.artifacts_dir, "dominant_frequencies")
        assert os.path.isdir(analyzer_dir)

    def test_artifact_contains_epochs(self, trained_variant):
        """Artifact loader discovers correct epochs."""
        from miscope.analysis import ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        artifact = loader.load("dominant_frequencies")
        assert "epochs" in artifact
        np.testing.assert_array_equal(artifact["epochs"], [0, 25, 49])

    def test_artifact_contains_coefficients(self, trained_variant):
        """Artifact contains coefficients array."""
        from miscope.analysis import ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        epoch_data = loader.load_epoch("dominant_frequencies", 0)
        assert "coefficients" in epoch_data

    def test_artifact_coefficients_shape(self, trained_variant):
        """Artifact coefficients have correct shape."""
        from miscope.analysis import ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        prime = trained_variant.params["prime"]
        n_components = 2 * (prime // 2) + 1

        # Single epoch: (n_components,)
        epoch_data = loader.load_epoch("dominant_frequencies", 0)
        assert epoch_data["coefficients"].shape == (n_components,)

        # Stacked: (n_epochs, n_components)
        artifact = loader.load("dominant_frequencies")
        n_epochs = len(trained_variant.get_available_checkpoints())
        assert artifact["coefficients"].shape == (n_epochs, n_components)

    def test_coefficients_change_across_epochs(self, trained_variant):
        """Coefficients differ between epochs (training changes weights)."""
        from miscope.analysis import ArtifactLoader

        pipeline = AnalysisPipeline(trained_variant)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        loader = ArtifactLoader(pipeline.artifacts_dir)
        artifact = loader.load("dominant_frequencies")
        coefficients = artifact["coefficients"]

        # First and last epoch should have different coefficients
        assert not np.allclose(coefficients[0], coefficients[-1])
