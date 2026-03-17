"""InterventionVariant — a Variant produced by applying a training hook.

An intervention variant is nested under its parent Variant's directory:

    {parent.variant_dir}/interventions/{name}/

It shares the parent's family, domain parameters, and full Variant API
(train, load_model_at_checkpoint, at().view(), etc.) and adds a parent
reference and intervention config access.

The intervention directory name is either:
- The ``label`` field from the intervention config (e.g. "v1", "v2")
- A short deterministic hash of the config (8 hex chars) if no label is set

Usage::

    # Discover existing interventions on a variant
    parent = registry.get_variants(family)[0]
    for iv in parent.interventions:
        print(iv.name, iv.intervention_config.get("label"))

    # Create a new intervention variant
    iv = parent.create_intervention_variant({
        "type": "frequency_gain",
        "label": "v4",
        "target_frequencies": [4, 10],
        "gain": {4: 0.5, 10: 0.5},
        "epoch_start": 1500,
        "epoch_end": 6500,
        "ramp_epochs": 200,
    })
    iv.train(training_hook=hook)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from miscope.families.variant import Variant

if TYPE_CHECKING:
    pass


def compute_intervention_id(intervention_config: dict[str, Any]) -> str:
    """Compute a short deterministic ID from an intervention config dict.

    The ID is the first 8 hex characters of the SHA-256 hash of the
    canonically serialised config (sorted keys, no whitespace).  The
    same config always produces the same ID regardless of key insertion
    order.

    Args:
        intervention_config: Intervention parameter dict.

    Returns:
        8-character lowercase hex string.
    """
    canonical = json.dumps(intervention_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


class InterventionVariant(Variant):
    """A trained model variant produced by applying an intervention during training.

    Lives nested under a parent Variant's directory::

        {parent.variant_dir}/interventions/{name}/

    Inherits the full Variant API — train(), load_model_at_checkpoint(),
    at(epoch).view(), artifacts, metadata, etc.  All domain logic
    (model creation, dataset generation, analysis context) is delegated
    to the parent's family.

    The name is derived from the intervention config's optional ``label``
    field; if absent, the 8-char config hash is used instead.

    Labels must be unique within the parent's ``interventions/`` directory.
    Uniqueness is enforced when calling ``parent.create_intervention_variant()``.
    """

    def __init__(
        self,
        parent: Variant,
        intervention_config: dict[str, Any],
    ):
        """Initialise an InterventionVariant.

        Args:
            parent: The Variant this intervention was applied to.
            intervention_config: Intervention parameter dict.  Should
                include at minimum: type, target_frequencies, gain,
                epoch_start, epoch_end.  Optional: label, ramp_epochs.
        """
        # Merge parent params with the intervention config so that
        # _save_config() serialises the full picture to config.json.
        params = {**parent.params, "intervention": intervention_config}
        super().__init__(
            family=parent.family,
            params=params,
            results_dir=parent._results_dir,
        )
        self._parent = parent
        self._intervention_config = intervention_config

    # --- Identity ---

    @property
    def name(self) -> str:
        """Directory name: label if set, otherwise 8-char config hash."""
        label = self._intervention_config.get("label")
        if label:
            return str(label)
        return compute_intervention_id(self._intervention_config)

    @property
    def variant_dir(self) -> Path:
        """Path nested under the parent's interventions/ subdirectory."""
        return self._parent.variant_dir / "interventions" / self.name

    # --- Domain access ---

    @property
    def parent(self) -> Variant:
        """The baseline Variant this intervention was applied to."""
        return self._parent

    @property
    def intervention_config(self) -> dict[str, Any]:
        """Intervention parameter dict (copy)."""
        return self._intervention_config.copy()

    # --- Representation ---

    def __repr__(self) -> str:
        return (
            f"InterventionVariant("
            f"parent={self._parent.name!r}, "
            f"name={self.name!r}, "
            f"state={self.state.value})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InterventionVariant):
            return NotImplemented
        return (
            self._parent == other._parent
            and self._intervention_config == other._intervention_config
        )

    def __hash__(self) -> int:
        # Avoid hashing the intervention dict (contains unhashable nested dicts)
        return hash((self._parent.family.name, self._parent.name, self.name))
