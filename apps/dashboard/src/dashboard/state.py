"""Server-side state management for the Dash dashboard.

Unlike Gradio's gr.State (serialized per-request), Dash uses dcc.Store
for client-side JSON state. Heavy data (loss arrays, artifact loader)
stays server-side in this module to avoid serialization overhead.

Single-user assumption: one global ServerState instance. This matches
the Gradio dashboard's model and is appropriate for a local analysis tool.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from miscope import EpochContext, catalog
from miscope.config import get_config
from miscope.families import InterventionVariant, ModelFamily, Variant
from miscope.families.discovery import discover_families

# ---------------------------------------------------------------------------
# Job progress tracking (thread-safe)
# ---------------------------------------------------------------------------


@dataclass
class JobProgress:
    """Server-side mutable progress state for a running job.

    Used with threading.Thread + dcc.Interval polling pattern.
    Single-user assumption makes global instances safe.
    """

    running: bool = False
    progress: float = 0.0
    message: str = ""
    result: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        with self._lock:
            self.running = True
            self.progress = 0.0
            self.message = "Starting..."
            self.result = ""

    def update(self, progress: float, message: str) -> None:
        with self._lock:
            self.progress = progress
            self.message = message

    def finish(self, result: str) -> None:
        with self._lock:
            self.running = False
            self.progress = 1.0
            self.message = "Complete"
            self.result = result

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "progress": self.progress,
                "message": self.message,
                "result": self.result,
            }


training_progress = JobProgress()
analysis_progress = JobProgress()


# ---------------------------------------------------------------------------
# Family cache (singleton)
# ---------------------------------------------------------------------------

_families: dict[str, ModelFamily] | None = None


def _load_families() -> dict[str, ModelFamily]:
    return discover_families(data_root=get_config().data_root)


def get_families() -> dict[str, ModelFamily]:
    """Return all discovered families keyed by name (cached)."""
    global _families
    if _families is None:
        _families = _load_families()
    return _families


def refresh_families() -> None:
    """Force the family cache to reload from filesystem."""
    global _families
    _families = _load_families()


# ---------------------------------------------------------------------------
# Variant Data in Server State
# ---------------------------------------------------------------------------


class VariantServerState:
    """Server-side state for a loaded variant.

    Distinct from miscope.families.VariantState, which tracks training status.
    This class holds Dash-specific infrastructure: artifact loader access,
    EpochContext cursor, and pre-computed availability lists.
    """

    family_name: str | None = None
    variant: Variant
    context: EpochContext
    available_epochs: list[int] = [0]
    available_views: list[str] = []
    intervention_name: str | None = None
    interventions: list[InterventionVariant] = []
    intervention: InterventionVariant | None = None

    def load_variant(
        self, family_name: str, variant_name: str, intervention_name: str | None = None
    ) -> bool:
        """Load a variant's metadata and discover its artifacts.

        Computes available_views once against the variant's actual artifacts.
        Returns True if the variant was found and loaded.
        """
        families = get_families()

        family = families.get(family_name)
        if family is None:
            return False

        variant = next((v for v in family.variants if v.name == variant_name), None)
        if variant is None:
            return False

        intervention = None
        if intervention_name is not None:
            for iv in variant.interventions:
                if iv.name == intervention_name:
                    intervention = iv

        self.variant = variant
        self.intervention_name = intervention_name
        self.interventions = variant.interventions
        self.intervention = intervention

        if self.intervention is not None:
            self.available_epochs = self.intervention.get_available_checkpoints()
            self.context = self.intervention.at(0)
            self.available_views = catalog.available_names_for(self.intervention)
        else:
            self.available_epochs = variant.get_available_checkpoints()
            self.context = variant.at(0)
            self.available_views = catalog.available_names_for(variant)

        return True

    def load_epoch(self, epoch: int) -> bool:
        if epoch in self.available_epochs:
            if self.intervention is not None:
                self.context = self.intervention.at(epoch)
            else:
                self.context = self.variant.at(epoch)

        # TODO: Add error handling. This is currently just proof-of-concept
        # for wiring architecture
        return True

    def get_nearest_epoch_index(self, epoch: int) -> int:
        """Find the slider index closest to the given epoch number."""
        if not self.available_epochs:
            return 0
        distances = [abs(e - epoch) for e in self.available_epochs]
        return distances.index(min(distances))


# Global server state instance
variant_server_state = VariantServerState()
