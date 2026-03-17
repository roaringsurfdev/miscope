"""Tests for Modulo Addition 1-Layer family implementation (REQ_021c)."""
# pyright: reportArgumentType=false
# pyright: reportInvalidTypeForm=false
# pyright: reportReturnType=false
# pyright: reportAttributeAccessIssue=false

import tempfile
from pathlib import Path

import pytest
import torch

from miscope.analysis.analyzers import AnalyzerRegistry
from miscope.families import FamilyRegistry, VariantState
from miscope.families.implementations import ModuloAddition1LayerFamily


@pytest.fixture
def temp_project_dir() -> Path:
    """Create a temporary project directory with model_families."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Copy family.json to temp location
        family_dir = root / "model_families" / "modulo_addition_1layer"
        family_dir.mkdir(parents=True)

        # Create family.json
        import json

        config = {
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
                "data_seed": {
                    "type": "int",
                    "description": "Random seed for train/test split",
                    "default": 598,
                },
            },
            "analyzers": ["dominant_frequencies", "neuron_activations", "neuron_freq_norm"],
            "visualizations": ["dominant_frequencies_bar"],
            "analysis_dataset": {"type": "modulo_addition_grid"},
            "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}_dseed{data_seed}",
        }
        with open(family_dir / "family.json", "w") as f:
            json.dump(config, f)

        # Create results directory
        (root / "results").mkdir()

        yield root


@pytest.fixture
def registry(temp_project_dir) -> FamilyRegistry:
    """Create a FamilyRegistry with the temp project."""
    return FamilyRegistry(
        model_families_dir=temp_project_dir / "model_families",
        results_dir=temp_project_dir / "results",
    )


# --- Family Discovery Tests ---


class TestFamilyDiscovery:
    """Tests for family discovery and loading."""

    def test_registry_discovers_family(self, registry):
        """Test that registry discovers the modulo_addition_1layer family."""
        assert "modulo_addition_1layer" in registry
        assert len(registry) == 1

    def test_family_is_correct_implementation(self, registry):
        """Test that the family is loaded as ModuloAddition1LayerFamily."""
        family = registry.get_family("modulo_addition_1layer")
        assert isinstance(family, ModuloAddition1LayerFamily)

    def test_family_properties(self, registry):
        """Test that family properties are correct."""
        family = registry.get_family("modulo_addition_1layer")

        assert family.name == "modulo_addition_1layer"
        assert family.display_name == "Modulo Addition (1 Layer)"
        assert family.architecture["n_layers"] == 1
        assert family.architecture["d_mlp"] == 512
        assert "dominant_frequencies" in family.analyzers


# --- Model Creation Tests ---


class TestModelCreation:
    """Tests for model creation."""

    def test_create_model_basic(self, registry):
        """Test basic model creation."""
        family = registry.get_family("modulo_addition_1layer")
        model = family.create_model({"prime": 11, "seed": 42})

        assert model is not None
        assert model.cfg.n_layers == 1
        assert model.cfg.d_vocab == 12  # p + 1
        assert model.cfg.d_vocab_out == 11  # p

    def test_create_model_vocab_size(self, registry):
        """Test that vocabulary size is correct for different primes."""
        family = registry.get_family("modulo_addition_1layer")

        for p in [7, 11, 17]:
            model = family.create_model({"prime": p})
            assert model.cfg.d_vocab == p + 1
            assert model.cfg.d_vocab_out == p

    def test_create_model_architecture(self, registry):
        """Test that model architecture matches family config."""
        family = registry.get_family("modulo_addition_1layer")
        model = family.create_model({"prime": 11})

        assert model.cfg.n_layers == 1
        assert model.cfg.n_heads == 4
        assert model.cfg.d_model == 128
        assert model.cfg.d_mlp == 512
        assert model.cfg.act_fn == "relu"

    def test_create_model_biases_disabled(self, registry):
        """Test that biases are disabled (requires_grad=False)."""
        family = registry.get_family("modulo_addition_1layer")
        model = family.create_model({"prime": 11})

        for name, param in model.named_parameters():
            if "b_" in name:
                assert not param.requires_grad


# --- Dataset Generation Tests ---


class TestDatasetGeneration:
    """Tests for analysis dataset generation."""

    def test_generate_dataset_shape(self, registry):
        """Test that dataset has correct shape."""
        family = registry.get_family("modulo_addition_1layer")
        p = 11
        dataset = family.generate_analysis_dataset({"prime": p})

        assert dataset.shape == (p * p, 3)

    def test_generate_dataset_content(self, registry):
        """Test that dataset contains correct values."""
        family = registry.get_family("modulo_addition_1layer")
        p = 5
        dataset = family.generate_analysis_dataset({"prime": p})

        # First row should be [0, 0, p]
        assert dataset[0, 0] == 0
        assert dataset[0, 1] == 0
        assert dataset[0, 2] == p  # equals token

        # All values in columns 0 and 1 should be in [0, p-1]
        assert (dataset[:, 0] >= 0).all()
        assert (dataset[:, 0] < p).all()
        assert (dataset[:, 1] >= 0).all()
        assert (dataset[:, 1] < p).all()

        # Column 2 should all be equals token (p)
        assert (dataset[:, 2] == p).all()

    def test_get_labels(self, registry):
        """Test that labels are correct."""
        family = registry.get_family("modulo_addition_1layer")
        p = 7
        labels = family.get_labels({"prime": p})

        assert labels.shape == (p * p,)
        assert (labels >= 0).all()
        assert (labels < p).all()

        # Verify a few specific cases
        dataset = family.generate_analysis_dataset({"prime": p})
        for i in range(min(10, len(labels))):
            a, b = dataset[i, 0].item(), dataset[i, 1].item()
            expected = (a + b) % p
            assert labels[i].item() == expected


# --- Variant Tests ---


class TestVariantIntegration:
    """Tests for variant creation and discovery."""

    def test_create_variant(self, registry):
        """Test creating a variant."""
        variant = registry.create_variant(
            "modulo_addition_1layer", {"prime": 113, "seed": 42, "data_seed": 598}
        )

        assert variant.name == "modulo_addition_1layer_p113_seed42_dseed598"
        assert variant.state == VariantState.UNTRAINED

    def test_variant_directory_structure(self, registry, temp_project_dir):
        """Test variant directory paths."""
        variant = registry.create_variant(
            "modulo_addition_1layer", {"prime": 113, "seed": 42, "data_seed": 598}
        )

        expected_base = temp_project_dir / "results" / "modulo_addition_1layer"
        assert variant.variant_dir == expected_base / "modulo_addition_1layer_p113_seed42_dseed598"

    def test_discover_variants(self, registry, temp_project_dir):
        """Test discovering existing variants."""
        # Create a variant directory
        variant_dir = (
            temp_project_dir
            / "results"
            / "modulo_addition_1layer"
            / "modulo_addition_1layer_p17_seed123_dseed598"
        )
        variant_dir.mkdir(parents=True)
        (variant_dir / "checkpoints").mkdir()
        (variant_dir / "checkpoints" / "checkpoint_epoch_00100.safetensors").touch()

        variants = registry.get_variants("modulo_addition_1layer")

        assert len(variants) == 1
        assert variants[0].name == "modulo_addition_1layer_p17_seed123_dseed598"
        assert variants[0].params == {"prime": 17, "seed": 123, "data_seed": 598}
        assert variants[0].state == VariantState.TRAINED


# --- Analyzer Integration Tests ---


class TestAnalyzerIntegration:
    """Tests for analyzer integration with the family."""

    def test_get_analyzers_for_family(self, registry):
        """Test getting analyzers for the family."""
        family = registry.get_family("modulo_addition_1layer")
        analyzers = AnalyzerRegistry.get_for_family(family)

        assert len(analyzers) == 3
        analyzer_names = {a.name for a in analyzers}
        assert "dominant_frequencies" in analyzer_names
        assert "neuron_activations" in analyzer_names
        assert "neuron_freq_norm" in analyzer_names

    def test_run_dominant_frequencies_analyzer(self, registry):
        """Test running the dominant frequencies analyzer."""
        family = registry.get_family("modulo_addition_1layer")
        params = {"prime": 7, "seed": 42}

        model = family.create_model(params)
        dataset = family.generate_analysis_dataset(params)
        context = family.prepare_analysis_context(params, model.cfg.device)

        with torch.inference_mode():
            _, cache = model.run_with_cache(dataset)

        analyzer = AnalyzerRegistry.get("dominant_frequencies")
        result = analyzer.analyze(model, dataset, cache, context)

        assert "coefficients" in result
        assert result["coefficients"].shape[0] == context["fourier_basis"].shape[0]

    def test_run_neuron_activations_analyzer(self, registry):
        """Test running the neuron activations analyzer."""
        family = registry.get_family("modulo_addition_1layer")
        params = {"prime": 7, "seed": 42}

        model = family.create_model(params)
        dataset = family.generate_analysis_dataset(params)
        context = family.prepare_analysis_context(params, model.cfg.device)

        with torch.inference_mode():
            _, cache = model.run_with_cache(dataset)

        analyzer = AnalyzerRegistry.get("neuron_activations")
        result = analyzer.analyze(model, dataset, cache, context)

        assert "activations" in result
        p = params["prime"]
        d_mlp = model.cfg.d_mlp
        assert result["activations"].shape == (d_mlp, p, p)

    def test_run_neuron_freq_norm_analyzer(self, registry):
        """Test running the neuron frequency clusters analyzer."""
        family = registry.get_family("modulo_addition_1layer")
        params = {"prime": 7, "seed": 42}

        model = family.create_model(params)
        dataset = family.generate_analysis_dataset(params)
        context = family.prepare_analysis_context(params, model.cfg.device)

        with torch.inference_mode():
            _, cache = model.run_with_cache(dataset)

        analyzer = AnalyzerRegistry.get("neuron_freq_norm")
        result = analyzer.analyze(model, dataset, cache, context)

        assert "norm_matrix" in result
        p = params["prime"]
        d_mlp = model.cfg.d_mlp
        assert result["norm_matrix"].shape == (p // 2, d_mlp)


# --- End-to-End Test ---


class TestEndToEnd:
    """End-to-end integration test."""

    def test_full_workflow(self, registry):
        """Test complete workflow: family -> model -> dataset -> analysis."""
        # 1. Get family from registry
        family = registry.get_family("modulo_addition_1layer")
        assert isinstance(family, ModuloAddition1LayerFamily)

        # 2. Create a variant
        params = {"prime": 7, "seed": 42, "data_seed": 598}
        variant = registry.create_variant(family, params)
        assert variant.name == "modulo_addition_1layer_p7_seed42_dseed598"

        # 3. Create model
        model = family.create_model(params)
        assert model.cfg.d_vocab == 8

        # 4. Generate dataset
        dataset = family.generate_analysis_dataset(params)
        assert dataset.shape == (49, 3)

        # 5. Run forward pass
        with torch.inference_mode():
            logits, cache = model.run_with_cache(dataset)
        assert logits.shape == (49, 3, 7)

        # 6. Get and run analyzers
        context = family.prepare_analysis_context(params, model.cfg.device)
        analyzers = AnalyzerRegistry.get_for_family(family)

        for analyzer in analyzers:
            result = analyzer.analyze(model, dataset, cache, context)
            assert len(result) > 0


# --- Real Family.json Test ---


class TestRealFamilyJson:
    """Test with the actual family.json file."""

    def test_load_real_family_json(self):
        """Test loading the real family.json file."""
        from miscope.families.implementations.modulo_addition_1layer import (
            load_modulo_addition_1layer_family,
        )

        # This will fail if the file doesn't exist or is invalid
        family = load_modulo_addition_1layer_family()

        assert family.name == "modulo_addition_1layer"
        assert family.display_name == "Modulo Addition (1 Layer)"
        assert isinstance(family, ModuloAddition1LayerFamily)
