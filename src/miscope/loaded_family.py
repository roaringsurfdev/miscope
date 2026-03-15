"""LoadedFamily — notebook-friendly wrapper around ModelFamily + FamilyRegistry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miscope.families.protocols import ModelFamily
from miscope.families.registry import FamilyRegistry
from miscope.families.types import VariantState
from miscope.families.variant import Variant


class LoadedFamily:
    """A model family loaded and ready for notebook exploration.

    Wraps a ModelFamily with registry access so variants can be
    looked up by domain parameters without knowing file paths.

    Returned by ``tdw.load_family()``. Not intended to be
    instantiated directly.
    """

    def __init__(self, family: ModelFamily, registry: FamilyRegistry):
        self._family = family
        self._registry = registry

    @property
    def name(self) -> str:
        """Family identifier (e.g., 'modulo_addition_1layer')."""
        return self._family.name

    @property
    def display_name(self) -> str:
        """Human-readable family name."""
        return self._family.display_name

    @property
    def description(self) -> str:
        """Brief description of the family."""
        return self._family.description

    @property
    def family(self) -> ModelFamily:
        """The underlying ModelFamily instance."""
        return self._family

    def get_variant(self, **params: Any) -> Variant:
        """Get a trained variant by domain parameter values.

        Args:
            **params: Domain parameters (e.g., prime=113, seed=999)

        Returns:
            Variant object with convenience access to checkpoints,
            artifacts, metadata, and forward passes.

        Raises:
            ValueError: If variant doesn't exist or isn't trained

        Example:
            variant = family.get_variant(prime=113, seed=999)
        """
        variant = self._registry.create_variant(self._family, params)

        if variant.state == VariantState.UNTRAINED:
            available = self.list_variant_parameters()
            raise ValueError(
                f"Variant with params {params} not found or not trained. "
                f"Available variants: {available}"
            )

        return variant

    def list_variants(self) -> list[Variant]:
        """List all available variants for this family.

        Returns:
            List of Variant objects (trained or analyzed)
        """
        return self._registry.get_variants(self._family)

    def list_variant_parameters(self) -> list[dict[str, Any]]:
        """List parameter combinations for all available variants.

        Returns:
            List of parameter dicts (e.g., [{"prime": 113, "seed": 999}, ...])
        """
        return [v.params for v in self.list_variants()]

    def create_intervention_variant(
        self,
        prime: int,
        seed: int,
        data_seed: int,
        intervention_config: dict[str, Any],
        results_dir: Path | str | None = None,
    ) -> Variant:
        """Create an intervention variant nested under the specified parent variant.

        Args:
            prime: Modulus for the addition task
            seed: Random seed for model initialization
            data_seed: Random seed for train/test split
            intervention_config: Intervention parameter dict
            results_dir: Unused; kept for backwards compatibility.
        """
        parent = self.get_variant(prime=prime, seed=seed, data_seed=data_seed)
        return parent.create_intervention_variant(intervention_config)

    def __repr__(self) -> str:
        n_variants = len(self.list_variants())
        return f"LoadedFamily({self.name!r}, {n_variants} variants)"
