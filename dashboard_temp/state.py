"""Server-side state management for the Dash dashboard.

Unlike Gradio's gr.State (serialized per-request), Dash uses dcc.Store
for client-side JSON state. Heavy data (loss arrays, artifact loader)
stays server-side in this module to avoid serialization overhead.

Single-user assumption: one global ServerState instance. This matches
the Gradio dashboard's model and is appropriate for a local analysis tool.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miscope.analysis import ArtifactLoader
from miscope import catalog, load_family, EpochContext
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
        

# ---------------------------------------------------------------------------
# Server-side state
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
    """Heavy state that stays on the server (not serialized to browser).

    Holds loss arrays, artifact loader, and variant metadata.
    Client-side dcc.Store holds only lightweight selection state
    (family name, variant name, epoch index, neuron index, etc.).
    """

    # Variant metadata
    variant: Variant | None = None
    artifacts_dir: str | None = None
    available_analyzers: list[str] = field(default_factory=list)
    available_epochs: list[int] = field(default_factory=list)

    # Loss data (lists of floats, too large for dcc.Store)
    train_losses: list[float] | None = None
    test_losses: list[float] | None = None

    # Model config
    model_config: dict[str, Any] | None = None
    n_neurons: int = 512
    n_heads: int = 4

    # Cached cross-epoch trajectory data (REQ_038)
    _trajectory_data: dict[str, Any] | None = field(default=None, repr=False)

    def clear(self) -> None:
        """Reset all state."""
        self.variant = None
        self.artifacts_dir = None
        self.available_analyzers = []
        self.available_epochs = []
        self.train_losses = None
        self.test_losses = None
        self.model_config = None
        self.n_neurons = 512
        self.n_heads = 4
        self._trajectory_data = None

    def get_epoch_at_index(self, idx: int) -> int:
        """Map slider index to actual epoch number."""
        if self.available_epochs and 0 <= idx < len(self.available_epochs):
            return self.available_epochs[idx]
        return 0

    def nearest_epoch_index(self, epoch: int) -> int:
        """Find the slider index closest to the given epoch number."""
        if not self.available_epochs:
            return 0
        distances = [abs(e - epoch) for e in self.available_epochs]
        return distances.index(min(distances))

    def get_loader(self) -> ArtifactLoader | None:
        """Create an ArtifactLoader for the current variant."""
        if self.artifacts_dir:
            return ArtifactLoader(self.artifacts_dir)
        return None

    def get_trajectory_data(self) -> dict[str, Any] | None:
        """Load precomputed trajectory analysis (REQ_038 cross-epoch artifact).

        Returns the full cross-epoch data dict with group-prefixed keys,
        or None if no precomputed data exists.
        """
        if self._trajectory_data is not None:
            return self._trajectory_data

        if not self.artifacts_dir:
            return None

        loader = ArtifactLoader(self.artifacts_dir)
        if not loader.has_cross_epoch("parameter_trajectory"):
            return None

        try:
            self._trajectory_data = loader.load_cross_epoch("parameter_trajectory")
            return self._trajectory_data
        except FileNotFoundError:
            return None

    def load_variant(self, family_name: str, variant_name: str) -> bool:
        """Load a variant's metadata and discover its artifacts.

        Returns True if the variant was found and loaded.
        """
        self.clear()
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

        # Load loss data from metadata
        if variant.metadata_path.exists():
            with open(variant.metadata_path) as f:
                metadata = json.load(f)
            self.train_losses = metadata.get("train_losses", [])
            self.test_losses = metadata.get("test_losses", [])

        # Load config (includes d_mlp for neuron count, n_heads for attention)
        if variant.config_path.exists():
            with open(variant.config_path) as f:
                config = json.load(f)
            self.model_config = config
            self.n_neurons = config.get("d_mlp", 512)
            self.n_heads = config.get("n_heads", 4)

        # Discover artifacts
        artifacts_dir = variant.artifacts_dir
        if artifacts_dir.exists():
            try:
                loader = ArtifactLoader(str(artifacts_dir))
                self.available_analyzers = loader.get_available_analyzers()
                self.artifacts_dir = str(artifacts_dir)

                for analyzer_name in self.available_analyzers:
                    epochs = loader.get_epochs(analyzer_name)
                    if epochs:
                        self.available_epochs = epochs
                        break
            except Exception:
                pass

        return True


# Global server state instance
server_state = ServerState()
variant_state = VariantState()