"""Server-side state management for the Dash dashboard.

Unlike Gradio's gr.State (serialized per-request), Dash uses dcc.Store
for client-side JSON state. Heavy data (loss arrays, artifact loader)
stays server-side in this module to avoid serialization overhead.

Single-user assumption: one global ServerState instance. This matches
the Gradio dashboard's model and is appropriate for a local analysis tool.
"""

from __future__ import annotations

from pathlib import Path

from miscope import EpochContext, catalog
from miscope.families import FamilyRegistry, Variant

# ---------------------------------------------------------------------------
# Family Registry (singleton)
# ---------------------------------------------------------------------------

_registry: FamilyRegistry | None = None


def get_registry() -> FamilyRegistry:
    """Get or create the global FamilyRegistry instance."""
    global _registry
    if _registry is None:
        _registry = FamilyRegistry(
            model_families_dir=Path("model_families"),
            results_dir=Path("results"),
        )
    return _registry


def refresh_registry() -> None:
    """Force registry to reload from filesystem."""
    global _registry
    _registry = FamilyRegistry(
        model_families_dir=Path("model_families"),
        results_dir=Path("results"),
    )

# ---------------------------------------------------------------------------
# Variant Data in Server State
# ---------------------------------------------------------------------------

class VariantState:
    family_name: str | None = None
    variant: Variant
    context: EpochContext
    available_epochs: list[int] = [0]
    available_views: list[str] = catalog.names()

    def load_variant(self, family_name: str, variant_name: str) -> bool:
        """Load a variant's metadata and discover its artifacts.

        Returns True if the variant was found and loaded.
        """
        registry = get_registry()

        try:
            family = registry.get_family(family_name)
        except KeyError:
            return False

        variants = registry.get_variants(family)
        variant = None
        for v in variants:
            if v.name == variant_name:
                variant = v
                break

        if variant is None:
            return False

        self.variant = variant
        self.available_epochs = variant.get_available_checkpoints()
        self.context = variant.at(0)
        
        return True
    
    def load_epoch(self, epoch: int) -> bool:
        if epoch in self.available_epochs:
            self.context = self.variant.at(epoch)

        #TODO: Add error handling. This is currently just proof-of-concept 
        # for wiring architecture
        return True
        

    def get_nearest_epoch_index(self, epoch: int) -> int:
        """Find the slider index closest to the given epoch number."""
        if not self.available_epochs:
            return 0
        distances = [abs(e - epoch) for e in self.available_epochs]
        return distances.index(min(distances))

# Global server state instance
variant_state = VariantState()