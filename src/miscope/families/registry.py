"""FamilyRegistry for discovering and managing model families."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from miscope.families.json_family import JsonModelFamily
from miscope.families.protocols import ModelFamily
from miscope.families.variant import Variant

# Mapping of family names to their implementation classes
# Families not in this map will use JsonModelFamily (config-only)
_FAMILY_IMPLEMENTATIONS: dict[str, type[JsonModelFamily]] = {}


def register_family_implementation(name: str, cls: type[JsonModelFamily]) -> None:
    """Register a family implementation class.

    Args:
        name: Family name (must match family.json "name" field)
        cls: Implementation class (must be subclass of JsonModelFamily)
    """
    _FAMILY_IMPLEMENTATIONS[name] = cls


def _register_default_implementations() -> None:
    """Register built-in family implementations."""
    from miscope.families.implementations.modulo_addition_1layer import ModuloAddition1LayerFamily
    from miscope.families.implementations.two_layer_mlp import TwoLayerMLPFamily

    register_family_implementation("modulo_addition_1layer", ModuloAddition1LayerFamily)
    register_family_implementation("modulo_addition_2layer_mlp", TwoLayerMLPFamily)


# Register default implementations on module load
_register_default_implementations()


class FamilyRegistry:
    """Discovers and manages registered model families.

    Families are discovered from the model_families/ directory.
    Each subdirectory with a family.json file is registered as a family.

    If a family has a registered implementation class (via register_family_implementation),
    that class is used instead of the base JsonModelFamily.

    Variants are discovered from the results/ directory based on
    the family's variant_pattern.
    """

    def __init__(
        self,
        model_families_dir: Path | str,
        results_dir: Path | str,
    ):
        """Initialize the registry.

        Args:
            model_families_dir: Path to model_families/ directory
            results_dir: Path to results/ directory
        """
        self._model_families_dir = Path(model_families_dir)
        self._results_dir = Path(results_dir)
        self._families: dict[str, ModelFamily] = {}
        self._load_families()

    def _load_families(self) -> None:
        """Discover and load all families from model_families directory."""
        if not self._model_families_dir.exists():
            return

        for family_dir in self._model_families_dir.iterdir():
            if not family_dir.is_dir():
                continue

            family_json = family_dir / "family.json"
            if not family_json.exists():
                continue

            try:
                # Peek at the name to determine which class to use
                with open(family_json) as f:
                    config = json.load(f)
                family_name = config.get("name", "")

                # Use registered implementation if available
                impl_class = _FAMILY_IMPLEMENTATIONS.get(family_name, JsonModelFamily)
                family = impl_class.from_json(family_json)
                self._families[family.name] = family
            except (json.JSONDecodeError, KeyError) as e:
                # Log warning but continue loading other families
                print(f"Warning: Failed to load family from {family_json}: {e}")

    def get_family(self, name: str) -> ModelFamily:
        """Get a family by name.

        Args:
            name: The family name (e.g., "modulo_addition_1layer")

        Returns:
            The ModelFamily instance

        Raises:
            KeyError: If family not found
        """
        if name not in self._families:
            raise KeyError(f"Family '{name}' not found. Available: {list(self._families.keys())}")
        return self._families[name]

    def list_families(self) -> list[ModelFamily]:
        """List all registered families.

        Returns:
            List of all registered ModelFamily instances
        """
        return list(self._families.values())

    def get_family_names(self) -> list[str]:
        """Get names of all registered families.

        Returns:
            List of family names
        """
        return list(self._families.keys())

    def get_variants(self, family: ModelFamily | str) -> list[Variant]:
        """Discover variants for a family from results directory.

        Args:
            family: The ModelFamily instance or family name

        Returns:
            List of discovered Variant instances
        """
        if isinstance(family, str):
            family = self.get_family(family)

        family_results_dir = self._results_dir / family.name
        if not family_results_dir.exists():
            return []

        variants = []
        pattern_regex = self._pattern_to_regex(family.variant_pattern, family.domain_parameters)

        for variant_dir in family_results_dir.iterdir():
            if not variant_dir.is_dir():
                continue

            match = pattern_regex.match(variant_dir.name)
            if match:
                params = self._extract_params(match, family.domain_parameters)
                variant = Variant(family, params, self._results_dir)
                variants.append(variant)

        return variants

    def _pattern_to_regex(self, pattern: str, domain_parameters: dict[str, Any]) -> re.Pattern[str]:
        """Convert variant pattern to regex for matching.

        Args:
            pattern: Pattern like "modulo_addition_1layer_p{prime}_seed{seed}"
            domain_parameters: Parameter specs to determine types

        Returns:
            Compiled regex pattern
        """
        # Escape regex special characters except our placeholders
        regex_pattern = re.escape(pattern)

        # Replace escaped placeholders with capture groups
        for param_name, spec in domain_parameters.items():
            placeholder = re.escape("{" + param_name + "}")
            param_type = spec.get("type", "str")

            if param_type == "int":
                capture_group = f"(?P<{param_name}>\\d+)"
            elif param_type == "float":
                capture_group = f"(?P<{param_name}>\\d+\\.?\\d*)"
            else:
                capture_group = f"(?P<{param_name}>[^_]+)"

            regex_pattern = regex_pattern.replace(placeholder, capture_group)

        return re.compile(f"^{regex_pattern}$")

    def _extract_params(
        self, match: re.Match[str], domain_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract typed parameters from regex match.

        Args:
            match: Regex match object
            domain_parameters: Parameter specs for type conversion

        Returns:
            Dict of parameter name to typed value
        """
        params = {}
        for param_name, spec in domain_parameters.items():
            raw_value = match.group(param_name)
            param_type = spec.get("type", "str")

            if param_type == "int":
                params[param_name] = int(raw_value)
            elif param_type == "float":
                params[param_name] = float(raw_value)
            else:
                params[param_name] = raw_value

        return params

    def create_variant(self, family: ModelFamily | str, params: dict[str, Any]) -> Variant:
        """Create a new Variant instance (does not create directories).

        Args:
            family: The ModelFamily instance or family name
            params: Domain parameter values

        Returns:
            New Variant instance
        """
        if isinstance(family, str):
            family = self.get_family(family)
        return Variant(family, params, self._results_dir)

    def __len__(self) -> int:
        return len(self._families)

    def __contains__(self, name: str) -> bool:
        return name in self._families
