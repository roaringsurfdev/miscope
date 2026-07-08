"""Family-specific implementations.

This module contains concrete ModelFamily implementations that provide
the create_model() and generate_analysis_dataset() methods.

Each implementation module registers itself with
``miscope.families.discovery`` on import, so importing this package is
enough to make all built-in families discoverable.
"""

from miscope.families.implementations.modulo_addition_1layer import ModuloAddition1LayerFamily
from miscope.families.implementations.modulo_addition_2l_mlp import ModuloAddition2LMLPFamily
from miscope.families.implementations.modulo_addition_embed_mlp import (
    ModuloAdditionEmbedMLPFamily,
)

__all__ = [
    "ModuloAddition1LayerFamily",
    "ModuloAddition2LMLPFamily",
    "ModuloAdditionEmbedMLPFamily",
]
