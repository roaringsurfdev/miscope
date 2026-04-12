"""Tests for the notebook research API (REQ_037)."""
# pyright: reportArgumentType=false
# pyright: reportReturnType=false

import json
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from miscope import LoadedFamily, list_families, load_family
from miscope.config import AppConfig
from miscope.families import BaseModelFamily, Variant

# --- Fixtures ---


@pytest.fixture
def sample_family_config() -> dict:
    """Sample family.json configuration."""
    return {
        "name": "test_family",
        "display_name": "Test Family",
        "description": "A test family for unit tests",
        "architecture": {
            "n_layers": 1,
            "n_heads": 4,
            "d_model": 128,
            "d_mlp": 512,
            "act_fn": "relu",
        },
        "domain_parameters": {
            "prime": {"type": "int", "description": "Modulus", "default": 113},
            "seed": {"type": "int", "description": "Random seed", "default": 999},
        },
        "analyzers": ["dominant_frequencies", "neuron_activations"],
        "visualizations": ["freq_bar", "activation_heatmap"],
        "analysis_dataset": {"type": "test_grid"},
        "variant_pattern": "test_family_p{prime}_seed{seed}",
    }


@pytest.fixture
def temp_project(sample_family_config) -> Generator[Path, None, None]:
    """Create a temporary project with family config, trained variant, and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create model_families directory
        family_dir = root / "model_families" / "test_family"
        family_dir.mkdir(parents=True)
        with open(family_dir / "family.json", "w") as f:
            json.dump(sample_family_config, f)

        # Create a trained variant with checkpoints
        variant_dir = root / "results" / "test_family" / "test_family_p113_seed999"
        checkpoints_dir = variant_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        (checkpoints_dir / "checkpoint_epoch_00100.safetensors").touch()
        (checkpoints_dir / "checkpoint_epoch_00500.safetensors").touch()
        (checkpoints_dir / "checkpoint_epoch_01000.safetensors").touch()

        # Create metadata.json
        metadata = {
            "train_losses": [2.5, 2.0, 1.5, 1.0, 0.5],
            "test_losses": [3.0, 2.8, 2.5, 2.0, 1.0],
            "checkpoint_epochs": [100, 500, 1000],
            "num_epochs": 5,
        }
        with open(variant_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create config.json
        config = {
            "n_layers": 1,
            "n_heads": 4,
            "d_model": 128,
            "d_mlp": 512,
            "prime": 113,
            "seed": 999,
        }
        with open(variant_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create artifacts with a real .npz file
        artifacts_dir = variant_dir / "artifacts" / "dominant_frequencies"
        artifacts_dir.mkdir(parents=True)
        np.savez(
            artifacts_dir / "epoch_00100.npz",
            coefficients=np.random.rand(10),
        )

        # Create a second variant (untrained)
        untrained_dir = root / "results" / "test_family" / "test_family_p97_seed42"
        untrained_dir.mkdir(parents=True)

        # Create a second trained variant
        variant2_dir = root / "results" / "test_family" / "test_family_p113_seed485"
        variant2_checkpoints = variant2_dir / "checkpoints"
        variant2_checkpoints.mkdir(parents=True)
        (variant2_checkpoints / "checkpoint_epoch_00100.safetensors").touch()

        yield root


@pytest.fixture
def test_config(temp_project) -> AppConfig:
    """AppConfig pointing at the temp project."""
    return AppConfig(
        project_root=temp_project,
        results_dir=temp_project / "results",
        model_families_dir=temp_project / "model_families",
    )


# --- Entry Point Tests ---


class TestLoadFamily:
    """Tests for load_family() entry point."""

    def test_load_family_returns_loaded_family(self, test_config):
        """load_family returns a LoadedFamily instance."""
        family = load_family("test_family", config=test_config)
        assert isinstance(family, LoadedFamily)
        assert family.name == "test_family"

    def test_load_family_not_found(self, test_config):
        """load_family raises KeyError for unknown family."""
        with pytest.raises(KeyError, match="not found"):
            load_family("nonexistent", config=test_config)

    def test_load_family_display_name(self, test_config):
        """LoadedFamily exposes display name."""
        family = load_family("test_family", config=test_config)
        assert family.display_name == "Test Family"

    def test_load_family_description(self, test_config):
        """LoadedFamily exposes description."""
        family = load_family("test_family", config=test_config)
        assert family.description == "A test family for unit tests"

    def test_load_family_repr(self, test_config):
        """LoadedFamily has useful repr."""
        family = load_family("test_family", config=test_config)
        r = repr(family)
        assert "test_family" in r


class TestListFamilies:
    """Tests for list_families() entry point."""

    def test_list_families(self, test_config):
        """list_families returns available family names."""
        names = list_families(config=test_config)
        assert "test_family" in names


# --- Variant Access Tests ---


class TestGetVariant:
    """Tests for LoadedFamily.get_variant()."""

    def test_get_variant_by_params(self, test_config):
        """get_variant returns trained variant by parameter values."""
        family = load_family("test_family", config=test_config)
        variant = family.get_variant(prime=113, seed=999)

        assert isinstance(variant, Variant)
        assert variant.params == {"prime": 113, "seed": 999}

    def test_get_variant_not_found(self, test_config):
        """get_variant raises ValueError for untrained/nonexistent variant."""
        family = load_family("test_family", config=test_config)

        with pytest.raises(ValueError, match="not found or not trained"):
            family.get_variant(prime=199, seed=123)

    def test_get_variant_untrained(self, test_config):
        """get_variant raises ValueError for untrained variant (dir exists, no checkpoints)."""
        family = load_family("test_family", config=test_config)

        with pytest.raises(ValueError, match="not found or not trained"):
            family.get_variant(prime=97, seed=42)

    def test_list_variants(self, test_config):
        """list_variants discovers all variants with directories."""
        family = load_family("test_family", config=test_config)
        variants = family.list_variants()

        # Should find 3 variant directories (1 trained, 1 analyzed, 1 untrained)
        assert len(variants) == 3
        names = {v.name for v in variants}
        assert "test_family_p113_seed999" in names

    def test_list_variant_parameters(self, test_config):
        """list_variant_parameters returns list of param dicts."""
        family = load_family("test_family", config=test_config)
        param_list = family.list_variant_parameters()

        assert isinstance(param_list, list)
        assert len(param_list) == 3
        assert {"prime": 113, "seed": 999} in param_list


# --- Variant Hub: Convenience Properties ---


class TestVariantConvenience:
    """Tests for Variant convenience properties (REQ_037)."""

    def test_artifacts_returns_loader(self, test_config):
        """variant.artifacts returns an ArtifactLoader."""
        from miscope.analysis.artifact_loader import ArtifactLoader

        family = load_family("test_family", config=test_config)
        variant = family.get_variant(prime=113, seed=999)
        loader = variant.artifacts

        assert isinstance(loader, ArtifactLoader)

    def test_artifacts_can_load_epoch(self, test_config):
        """ArtifactLoader from variant can load artifacts."""
        family = load_family("test_family", config=test_config)
        variant = family.get_variant(prime=113, seed=999)

        epoch_data = variant.artifacts.load_epoch("dominant_frequencies", 100)
        assert "coefficients" in epoch_data

    def test_metadata_returns_dict(self, test_config):
        """variant.metadata returns parsed metadata dict."""
        family = load_family("test_family", config=test_config)
        variant = family.get_variant(prime=113, seed=999)

        meta = variant.metadata
        assert isinstance(meta, dict)
        assert "train_losses" in meta
        assert "test_losses" in meta

    def test_model_config_returns_dict(self, test_config):
        """variant.model_config returns parsed config dict."""
        family = load_family("test_family", config=test_config)
        variant = family.get_variant(prime=113, seed=999)

        cfg = variant.model_config
        assert isinstance(cfg, dict)
        assert cfg["d_model"] == 128
        assert cfg["prime"] == 113

    def test_train_losses_shortcut(self, test_config):
        """variant.train_losses returns loss array."""
        family = load_family("test_family", config=test_config)
        variant = family.get_variant(prime=113, seed=999)

        losses = variant.train_losses
        assert isinstance(losses, list)
        assert len(losses) == 5
        assert losses[0] == 2.5

    def test_test_losses_shortcut(self, test_config):
        """variant.test_losses returns loss array."""
        family = load_family("test_family", config=test_config)
        variant = family.get_variant(prime=113, seed=999)

        losses = variant.test_losses
        assert isinstance(losses, list)
        assert len(losses) == 5
        assert losses[0] == 3.0

    def test_metadata_file_not_found(self, test_config):
        """variant.metadata raises FileNotFoundError when missing."""
        family = load_family("test_family", config=test_config)
        # p113/seed485 has checkpoints but no metadata.json
        variant = family.get_variant(prime=113, seed=485)

        with pytest.raises(FileNotFoundError):
            _ = variant.metadata

    def test_model_config_file_not_found(self, test_config):
        """variant.model_config raises FileNotFoundError when missing."""
        family = load_family("test_family", config=test_config)
        variant = family.get_variant(prime=113, seed=485)

        with pytest.raises(FileNotFoundError):
            _ = variant.model_config


# --- Variant Hub: Forward Pass Conveniences ---


class TestMakeProbe:
    """Tests for make_probe via ModuloAddition family."""

    def test_make_probe_modular_addition(self):
        """make_probe appends equals token for modular addition."""
        from miscope.families.implementations.modulo_addition_1layer import (
            ModuloAddition1LayerFamily,
        )

        # Create a minimal family instance for testing
        config = {
            "name": "modulo_addition_1layer",
            "display_name": "Modulo Addition 1-Layer",
            "description": "Test",
            "architecture": {"n_layers": 1, "n_heads": 4, "d_model": 128, "d_mlp": 512},
            "domain_parameters": {
                "prime": {"type": "int", "default": 113},
                "seed": {"type": "int", "default": 999},
            },
            "analyzers": [],
            "visualizations": [],
            "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}",
        }
        family = ModuloAddition1LayerFamily(config)

        probe = family.make_probe({"prime": 113}, [[3, 29]])
        assert probe.shape == (1, 3)
        assert probe[0, 0].item() == 3
        assert probe[0, 1].item() == 29
        assert probe[0, 2].item() == 113  # equals token = p

    def test_make_probe_multiple_inputs(self):
        """make_probe handles multiple input pairs."""
        from miscope.families.implementations.modulo_addition_1layer import (
            ModuloAddition1LayerFamily,
        )

        config = {
            "name": "modulo_addition_1layer",
            "display_name": "Test",
            "description": "Test",
            "architecture": {},
            "domain_parameters": {"prime": {"type": "int"}, "seed": {"type": "int"}},
            "analyzers": [],
            "visualizations": [],
            "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}",
        }
        family = ModuloAddition1LayerFamily(config)

        probe = family.make_probe({"prime": 97}, [[3, 29], [5, 7], [0, 0]])
        assert probe.shape == (3, 3)
        assert probe[2, 2].item() == 97

    def test_make_probe_base_not_implemented(self, sample_family_config):
        """BaseModelFamily.make_probe raises NotImplementedError."""
        family = BaseModelFamily(sample_family_config)

        with pytest.raises(NotImplementedError):
            family.make_probe({"prime": 113}, [[3, 29]])

    def test_variant_make_probe_delegates(self):
        """Variant.make_probe delegates to family."""
        from miscope.families.implementations.modulo_addition_1layer import (
            ModuloAddition1LayerFamily,
        )

        config = {
            "name": "modulo_addition_1layer",
            "display_name": "Test",
            "description": "Test",
            "architecture": {},
            "domain_parameters": {"prime": {"type": "int"}, "seed": {"type": "int"}},
            "analyzers": [],
            "visualizations": [],
            "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}",
        }
        family = ModuloAddition1LayerFamily(config)
        variant = Variant(family, {"prime": 113, "seed": 999}, Path("/tmp/results"))

        probe = variant.make_probe([[3, 29]])
        assert probe.shape == (1, 3)
        assert probe[0, 2].item() == 113


class TestAnalysisDataset:
    """Tests for Variant.analysis_dataset()."""

    def test_analysis_dataset_delegates(self):
        """Variant.analysis_dataset delegates to family."""
        from miscope.families.implementations.modulo_addition_1layer import (
            ModuloAddition1LayerFamily,
        )

        config = {
            "name": "modulo_addition_1layer",
            "display_name": "Test",
            "description": "Test",
            "architecture": {"n_layers": 1, "n_heads": 4, "d_model": 128, "d_mlp": 512},
            "domain_parameters": {"prime": {"type": "int"}, "seed": {"type": "int"}},
            "analyzers": [],
            "visualizations": [],
            "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}",
        }
        family = ModuloAddition1LayerFamily(config)
        variant = Variant(family, {"prime": 7, "seed": 42}, Path("/tmp/results"))

        dataset = variant.analysis_dataset()
        # For prime=7, should be 7^2 = 49 rows, 3 columns
        assert dataset.shape == (49, 3)
        # Equals token should be 7 for all rows
        assert (dataset[:, 2] == 7).all()
