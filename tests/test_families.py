"""Tests for the families module (REQ_021a)."""
# pyright: reportArgumentType=false
# pyright: reportInvalidTypeForm=false
# pyright: reportReturnType=false
# pyright: reportIndexIssue=false
# pyright: reportTypedDictNotRequiredAccess=false

import json
import tempfile
from pathlib import Path

import pytest

from miscope.families import (
    FamilyRegistry,
    JsonModelFamily,
    Variant,
    VariantState,
)

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
        "analysis_dataset": {"type": "test_grid", "description": "Test dataset"},
        "variant_pattern": "test_family_p{prime}_seed{seed}",
    }


@pytest.fixture
def temp_project_dir(sample_family_config) -> Path:
    """Create a temporary project directory with family and results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create model_families directory with family.json
        family_dir = root / "model_families" / "test_family"
        family_dir.mkdir(parents=True)
        with open(family_dir / "family.json", "w") as f:
            json.dump(sample_family_config, f)

        # Create results directory with variant
        results_dir = root / "results" / "test_family" / "test_family_p113_seed42"
        results_dir.mkdir(parents=True)

        yield root


@pytest.fixture
def temp_project_with_checkpoints(temp_project_dir) -> Path:
    """Create a project dir with checkpoint files."""
    variant_dir = temp_project_dir / "results" / "test_family" / "test_family_p113_seed42"
    checkpoints_dir = variant_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Create fake checkpoint files (empty for testing)
    (checkpoints_dir / "checkpoint_epoch_00100.safetensors").touch()
    (checkpoints_dir / "checkpoint_epoch_00500.safetensors").touch()
    (checkpoints_dir / "checkpoint_epoch_01000.safetensors").touch()

    return temp_project_dir


@pytest.fixture
def temp_project_with_artifacts(temp_project_with_checkpoints) -> Path:
    """Create a project dir with both checkpoints and artifacts."""
    variant_dir = (
        temp_project_with_checkpoints / "results" / "test_family" / "test_family_p113_seed42"
    )
    artifacts_dir = variant_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Create fake artifact file
    (artifacts_dir / "dominant_frequencies_epoch_00100.npz").touch()

    return temp_project_with_checkpoints


# --- JsonModelFamily Tests ---


class TestJsonModelFamily:
    """Tests for JsonModelFamily class."""

    def test_from_json(self, temp_project_dir):
        """Test loading family from JSON file."""
        family_json = temp_project_dir / "model_families" / "test_family" / "family.json"
        family = JsonModelFamily.from_json(family_json)

        assert family.name == "test_family"
        assert family.display_name == "Test Family"
        assert family.description == "A test family for unit tests"

    def test_properties(self, sample_family_config):
        """Test all property accessors."""
        family = JsonModelFamily(sample_family_config)

        assert family.name == "test_family"
        assert family.display_name == "Test Family"
        assert family.architecture["n_layers"] == 1
        assert family.architecture["d_model"] == 128
        assert "prime" in family.domain_parameters
        assert family.domain_parameters["prime"]["type"] == "int"
        assert family.analyzers == ["dominant_frequencies", "neuron_activations"]
        assert family.variant_pattern == "test_family_p{prime}_seed{seed}"

    def test_get_variant_directory_name(self, sample_family_config):
        """Test variant directory name generation."""
        family = JsonModelFamily(sample_family_config)

        name = family.get_variant_directory_name({"prime": 113, "seed": 42})
        assert name == "test_family_p113_seed42"

        name = family.get_variant_directory_name({"prime": 97, "seed": 999})
        assert name == "test_family_p97_seed999"

    def test_get_default_params(self, sample_family_config):
        """Test getting default parameters."""
        family = JsonModelFamily(sample_family_config)
        defaults = family.get_default_params()

        assert defaults == {"prime": 113, "seed": 999}

    def test_missing_required_fields(self):
        """Test validation fails for missing required fields."""
        incomplete_config = {
            "name": "test",
            # Missing other required fields
        }

        with pytest.raises(KeyError) as exc_info:
            JsonModelFamily(incomplete_config)

        assert "Missing required fields" in str(exc_info.value)

    def test_create_model_not_implemented(self, sample_family_config):
        """Test that create_model raises NotImplementedError."""
        family = JsonModelFamily(sample_family_config)

        with pytest.raises(NotImplementedError):
            family.create_model({"prime": 113, "seed": 42})


# --- Variant Tests ---


class TestVariant:
    """Tests for Variant class."""

    def test_variant_properties(self, temp_project_dir, sample_family_config):
        """Test Variant property accessors."""
        family = JsonModelFamily(sample_family_config)
        results_dir = temp_project_dir / "results"
        variant = Variant(family, {"prime": 113, "seed": 42}, results_dir)

        assert variant.name == "test_family_p113_seed42"
        assert variant.family.name == "test_family"
        assert variant.params == {"prime": 113, "seed": 42}

    def test_variant_paths(self, temp_project_dir, sample_family_config):
        """Test Variant path resolution."""
        family = JsonModelFamily(sample_family_config)
        results_dir = temp_project_dir / "results"
        variant = Variant(family, {"prime": 113, "seed": 42}, results_dir)

        assert variant.variant_dir == results_dir / "test_family" / "test_family_p113_seed42"
        assert variant.checkpoints_dir == variant.variant_dir / "checkpoints"
        assert variant.artifacts_dir == variant.variant_dir / "artifacts"

    def test_variant_state_untrained(self, temp_project_dir, sample_family_config):
        """Test state detection for untrained variant."""
        family = JsonModelFamily(sample_family_config)
        results_dir = temp_project_dir / "results"
        # Create variant with non-existent params (no directory)
        variant = Variant(family, {"prime": 97, "seed": 123}, results_dir)

        assert variant.state == VariantState.UNTRAINED

    def test_variant_state_trained(self, temp_project_with_checkpoints, sample_family_config):
        """Test state detection for trained variant (has checkpoints, no artifacts)."""
        family = JsonModelFamily(sample_family_config)
        results_dir = temp_project_with_checkpoints / "results"
        variant = Variant(family, {"prime": 113, "seed": 42}, results_dir)

        assert variant.state == VariantState.TRAINED

    def test_variant_state_analyzed(self, temp_project_with_artifacts, sample_family_config):
        """Test state detection for analyzed variant."""
        family = JsonModelFamily(sample_family_config)
        results_dir = temp_project_with_artifacts / "results"
        variant = Variant(family, {"prime": 113, "seed": 42}, results_dir)

        assert variant.state == VariantState.ANALYZED

    def test_get_available_checkpoints(self, temp_project_with_checkpoints, sample_family_config):
        """Test checkpoint discovery."""
        family = JsonModelFamily(sample_family_config)
        results_dir = temp_project_with_checkpoints / "results"
        variant = Variant(family, {"prime": 113, "seed": 42}, results_dir)

        checkpoints = variant.get_available_checkpoints()
        assert checkpoints == [100, 500, 1000]

    def test_variant_equality(self, sample_family_config):
        """Test Variant equality comparison."""
        family = JsonModelFamily(sample_family_config)
        results_dir = Path("/tmp/results")

        v1 = Variant(family, {"prime": 113, "seed": 42}, results_dir)
        v2 = Variant(family, {"prime": 113, "seed": 42}, results_dir)
        v3 = Variant(family, {"prime": 97, "seed": 42}, results_dir)

        assert v1 == v2
        assert v1 != v3

    def test_variant_hash(self, sample_family_config):
        """Test Variant is hashable (can be used in sets/dicts)."""
        family = JsonModelFamily(sample_family_config)
        results_dir = Path("/tmp/results")

        v1 = Variant(family, {"prime": 113, "seed": 42}, results_dir)
        v2 = Variant(family, {"prime": 113, "seed": 42}, results_dir)

        variant_set = {v1, v2}
        assert len(variant_set) == 1


# --- FamilyRegistry Tests ---


class TestFamilyRegistry:
    """Tests for FamilyRegistry class."""

    def test_load_families(self, temp_project_dir):
        """Test family discovery from directory."""
        registry = FamilyRegistry(
            model_families_dir=temp_project_dir / "model_families",
            results_dir=temp_project_dir / "results",
        )

        assert len(registry) == 1
        assert "test_family" in registry

    def test_get_family(self, temp_project_dir):
        """Test getting a family by name."""
        registry = FamilyRegistry(
            model_families_dir=temp_project_dir / "model_families",
            results_dir=temp_project_dir / "results",
        )

        family = registry.get_family("test_family")
        assert family.name == "test_family"

    def test_get_family_not_found(self, temp_project_dir):
        """Test error when family not found."""
        registry = FamilyRegistry(
            model_families_dir=temp_project_dir / "model_families",
            results_dir=temp_project_dir / "results",
        )

        with pytest.raises(KeyError) as exc_info:
            registry.get_family("nonexistent")

        assert "not found" in str(exc_info.value)

    def test_list_families(self, temp_project_dir):
        """Test listing all families."""
        registry = FamilyRegistry(
            model_families_dir=temp_project_dir / "model_families",
            results_dir=temp_project_dir / "results",
        )

        families = registry.list_families()
        assert len(families) == 1
        assert families[0].name == "test_family"

    def test_get_variants(self, temp_project_dir):
        """Test variant discovery for a family."""
        registry = FamilyRegistry(
            model_families_dir=temp_project_dir / "model_families",
            results_dir=temp_project_dir / "results",
        )

        variants = registry.get_variants("test_family")
        assert len(variants) == 1
        assert variants[0].name == "test_family_p113_seed42"
        assert variants[0].params == {"prime": 113, "seed": 42}

    def test_get_variants_multiple(self, temp_project_dir):
        """Test discovering multiple variants."""
        # Create additional variant directories
        results_base = temp_project_dir / "results" / "test_family"
        (results_base / "test_family_p97_seed42").mkdir()
        (results_base / "test_family_p113_seed999").mkdir()

        registry = FamilyRegistry(
            model_families_dir=temp_project_dir / "model_families",
            results_dir=temp_project_dir / "results",
        )

        variants = registry.get_variants("test_family")
        assert len(variants) == 3

        variant_names = {v.name for v in variants}
        assert "test_family_p113_seed42" in variant_names
        assert "test_family_p97_seed42" in variant_names
        assert "test_family_p113_seed999" in variant_names

    def test_create_variant(self, temp_project_dir):
        """Test creating a new variant instance."""
        registry = FamilyRegistry(
            model_families_dir=temp_project_dir / "model_families",
            results_dir=temp_project_dir / "results",
        )

        variant = registry.create_variant("test_family", {"prime": 97, "seed": 123})
        assert variant.name == "test_family_p97_seed123"
        assert variant.state == VariantState.UNTRAINED

    def test_empty_model_families_dir(self):
        """Test registry with no families directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FamilyRegistry(
                model_families_dir=Path(tmpdir) / "nonexistent",
                results_dir=Path(tmpdir) / "results",
            )

            assert len(registry) == 0
            assert registry.list_families() == []


# --- Path Resolution Tests ---


class TestPathResolution:
    """Tests for consistent path resolution using ModelFamily.name."""

    def test_family_name_as_directory_key(self, temp_project_dir, sample_family_config):
        """Test that ModelFamily.name is used as directory key."""
        family = JsonModelFamily(sample_family_config)
        results_dir = temp_project_dir / "results"
        variant = Variant(family, {"prime": 113, "seed": 42}, results_dir)

        # Family name should be the first directory under results/
        assert variant.variant_dir.parent.name == family.name
        assert variant.variant_dir.parent == results_dir / family.name

    def test_consistent_path_structure(self, temp_project_dir, sample_family_config):
        """Test that path structure follows convention."""
        family = JsonModelFamily(sample_family_config)
        results_dir = temp_project_dir / "results"
        variant = Variant(family, {"prime": 113, "seed": 42}, results_dir)

        # Expected structure: results/{family.name}/{variant_name}/
        expected_variant_dir = results_dir / "test_family" / "test_family_p113_seed42"
        assert variant.variant_dir == expected_variant_dir

        # Subdirectories
        assert variant.checkpoints_dir == expected_variant_dir / "checkpoints"
        assert variant.artifacts_dir == expected_variant_dir / "artifacts"


# ---------------------------------------------------------------------------
# InterventionVariant Tests (REQ_071)
# ---------------------------------------------------------------------------


class TestInterventionVariant:
    """Tests for InterventionVariant and Variant.interventions / create_intervention_variant."""

    @pytest.fixture
    def base_variant(self, sample_family_config, tmp_path):
        family = JsonModelFamily(sample_family_config)
        results_dir = tmp_path / "results"
        return Variant(family, {"prime": 59, "seed": 485, "data_seed": 598}, results_dir)

    @pytest.fixture
    def iv_config(self):
        return {
            "type": "frequency_gain",
            "label": "v1",
            "target_frequencies": [4, 10, 12],
            "gain": {"4": 0.3, "10": 0.3, "12": 0.3},
            "epoch_start": 1500,
            "epoch_end": 6500,
            "ramp_epochs": 200,
        }

    def test_name_uses_label(self, base_variant, iv_config):
        from miscope.families.intervention_variant import InterventionVariant

        iv = InterventionVariant(base_variant, iv_config)
        assert iv.name == "v1"

    def test_name_falls_back_to_hash(self, base_variant):
        from miscope.families.intervention_variant import (
            InterventionVariant,
            compute_intervention_id,
        )

        config = {
            "type": "frequency_gain",
            "target_frequencies": [4],
            "gain": {"4": 0.3},
            "epoch_start": 1500,
            "epoch_end": 6500,
        }
        iv = InterventionVariant(base_variant, config)
        assert iv.name == compute_intervention_id(config)
        assert len(iv.name) == 8

    def test_variant_dir_nested_under_parent(self, base_variant, iv_config):
        from miscope.families.intervention_variant import InterventionVariant

        iv = InterventionVariant(base_variant, iv_config)
        assert iv.variant_dir == base_variant.variant_dir / "interventions" / "v1"

    def test_checkpoints_dir(self, base_variant, iv_config):
        from miscope.families.intervention_variant import InterventionVariant

        iv = InterventionVariant(base_variant, iv_config)
        assert iv.checkpoints_dir == iv.variant_dir / "checkpoints"

    def test_parent_reference(self, base_variant, iv_config):
        from miscope.families.intervention_variant import InterventionVariant

        iv = InterventionVariant(base_variant, iv_config)
        assert iv.parent is base_variant

    def test_intervention_config_copy(self, base_variant, iv_config):
        from miscope.families.intervention_variant import InterventionVariant

        iv = InterventionVariant(base_variant, iv_config)
        returned = iv.intervention_config
        assert returned == iv_config
        returned["label"] = "mutated"
        assert iv.intervention_config["label"] == "v1"  # original unchanged

    def test_create_intervention_variant_returns_iv(self, base_variant, iv_config):
        from miscope.families.intervention_variant import InterventionVariant

        iv = base_variant.create_intervention_variant(iv_config)
        assert isinstance(iv, InterventionVariant)
        assert iv.name == "v1"

    def test_create_intervention_variant_raises_if_exists(self, base_variant, iv_config, tmp_path):
        iv = base_variant.create_intervention_variant(iv_config)
        iv.variant_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="already exists"):
            base_variant.create_intervention_variant(iv_config)

    def test_interventions_empty_when_no_directory(self, base_variant):
        assert base_variant.interventions == []

    def test_interventions_discovers_from_filesystem(self, base_variant, iv_config):
        import json as _json

        # Simulate a migrated intervention: write config.json into interventions/v1/
        iv_dir = base_variant.variant_dir / "interventions" / "v1"
        iv_dir.mkdir(parents=True)
        config_on_disk = {**base_variant.params, "intervention": iv_config}
        with open(iv_dir / "config.json", "w") as f:
            _json.dump(config_on_disk, f)

        ivs = base_variant.interventions
        assert len(ivs) == 1
        assert ivs[0].name == "v1"
        assert ivs[0].intervention_config["label"] == "v1"
        assert ivs[0].parent.name == base_variant.name

    def test_hash_stable(self, base_variant, iv_config):
        from miscope.families.intervention_variant import InterventionVariant

        iv1 = InterventionVariant(base_variant, iv_config)
        iv2 = InterventionVariant(base_variant, iv_config)
        assert hash(iv1) == hash(iv2)

    def test_eq(self, base_variant, iv_config):
        from miscope.families.intervention_variant import InterventionVariant

        iv1 = InterventionVariant(base_variant, iv_config)
        iv2 = InterventionVariant(base_variant, iv_config)
        assert iv1 == iv2
