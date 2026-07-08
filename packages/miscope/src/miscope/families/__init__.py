"""Model family abstractions for MIScope.

This module provides the core abstractions for grouping structurally
similar models that share analysis logic.

Key concepts:
- ModelFamily: Protocol defining what a family must provide
- Variant: A specific trained model within a family
- discovery: Filesystem helpers for locating and loading families

Example usage:
    from miscope import load_family

    family = load_family("modulo_addition_1layer")

    # Look up a trained variant
    variant = family.get_variant(prime=113, seed=999, data_seed=598)

    # Iterate all variants
    for v in family.variants:
        print(f"{v.name}: {v.state.value}")

    # Construct a new variant (for training)
    variant = family.create_variant({"prime": 113, "seed": 42, "data_seed": 598})
"""

from miscope.families.base_model_family import BaseModelFamily
from miscope.families.discovery import (
    discover_families,
    list_family_dirs,
    load_family_from_dir,
    register_family_implementation,
)

# Importing implementations triggers self-registration with discovery.
from miscope.families.implementations import ModuloAddition1LayerFamily  # noqa: F401
from miscope.families.intervention_variant import InterventionVariant
from miscope.families.protocols import ModelFamily
from miscope.families.types import (
    AnalysisDatasetSpec,
    ArchitectureSpec,
    ParameterSpec,
    VariantState,
)
from miscope.families.variant import TrainingResult, Variant

__all__ = [
    # Protocols
    "ModelFamily",
    # Classes
    "BaseModelFamily",
    "InterventionVariant",
    "ModuloAddition1LayerFamily",
    "TrainingResult",
    "Variant",
    # Types
    "AnalysisDatasetSpec",
    "ArchitectureSpec",
    "ParameterSpec",
    "VariantState",
    # Discovery helpers
    "discover_families",
    "list_family_dirs",
    "load_family_from_dir",
    "register_family_implementation",
]
