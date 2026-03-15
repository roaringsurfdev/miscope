"""Model family abstractions for the Training Dynamics Workbench.

This module provides the core abstractions for grouping structurally
similar models that share analysis logic.

Key concepts:
- ModelFamily: Protocol defining what a family must provide
- Variant: A specific trained model within a family
- FamilyRegistry: Discovers families and variants from filesystem

Example usage:
    from miscope.families import FamilyRegistry, Variant, VariantState

    # Initialize registry
    registry = FamilyRegistry(
        model_families_dir="model_families",
        results_dir="results"
    )

    # Get a family
    family = registry.get_family("modulo_addition_1layer")

    # Discover variants
    variants = registry.get_variants(family)
    for variant in variants:
        print(f"{variant.name}: {variant.state.value}")

    # Create a new variant
    variant = registry.create_variant(
        family,
        {"prime": 113, "seed": 42}
    )

    # Create a model (requires family with implementation)
    model = family.create_model({"prime": 113, "seed": 42})
"""

# Import implementations to trigger registration
from miscope.families.implementations import ModuloAddition1LayerFamily  # noqa: F401
from miscope.families.intervention_variant import InterventionVariant
from miscope.families.json_family import JsonModelFamily
from miscope.families.protocols import ModelFamily
from miscope.families.registry import FamilyRegistry
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
    "FamilyRegistry",
    "InterventionVariant",
    "JsonModelFamily",
    "ModuloAddition1LayerFamily",
    "TrainingResult",
    "Variant",
    # Types
    "AnalysisDatasetSpec",
    "ArchitectureSpec",
    "ParameterSpec",
    "VariantState",
]
