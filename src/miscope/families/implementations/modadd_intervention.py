"""Modulo Addition Intervention family.

Provides isolated training runs with intervention hooks on top of the
Modulo Addition 1-Layer task. Results are stored separately from baseline
variants and cannot overwrite them.

Model creation, dataset generation, and analysis context are delegated to
ModuloAddition1LayerFamily — the intervention family does not reimplement
the domain logic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from miscope.families.implementations.modulo_addition_1layer import ModuloAddition1LayerFamily
from miscope.families.json_family import JsonModelFamily
from miscope.families.variant import Variant


def compute_intervention_id(intervention_config: dict[str, Any]) -> str:
    """Compute a short deterministic ID from an intervention config dict.

    The ID is the first 8 hex characters of the SHA-256 hash of the
    canonically serialized config (sorted keys, no whitespace). The same
    config always produces the same ID.

    Args:
        intervention_config: Intervention parameter dict

    Returns:
        8-character lowercase hex string
    """
    canonical = json.dumps(intervention_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


class ModAddInterventionFamily(JsonModelFamily):
    """Model family for intervention runs on the Modulo Addition 1-Layer task.

    This family is distinct from ModuloAddition1LayerFamily — it shares the
    same model architecture and domain logic (via composition), but has its
    own name, results directory root, and variant identity scheme.

    Variant names include a short intervention ID derived from the intervention
    config hash, keeping filenames tractable regardless of config complexity.
    The full intervention spec is stored in config.json for reproducibility.

    Usage:
        family = registry.get_family("modadd_intervention")
        variant = family.create_intervention_variant(
            prime=59, seed=485, data_seed=598,
            intervention_config={
                "type": "frequency_gain",
                "target_heads": "all",
                "target_frequencies": [4, 10, 12, 17, 22],
                "gain": {4: 0.3, 10: 0.3, 12: 0.3, 17: 0.3, 22: 0.3},
                "epoch_start": 1500,
                "epoch_end": 6500,
                "ramp_epochs": 200,
            },
            results_dir=results_dir,
        )
    """

    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path | None = None,
        baseline_family: ModuloAddition1LayerFamily | None = None,
    ):
        super().__init__(config, config_path)
        if baseline_family is not None:
            self._baseline = baseline_family
        else:
            # Load baseline from sibling directory relative to this family's config
            baseline_json = (
                config_path.parent.parent / "modulo_addition_1layer" / "family.json"
                if config_path is not None
                else Path("model_families/modulo_addition_1layer/family.json")
            )
            self._baseline = ModuloAddition1LayerFamily.from_json(baseline_json)

    @classmethod
    def from_json(cls, path: Path | str) -> ModAddInterventionFamily:
        path = Path(path)
        with open(path) as f:
            config = json.load(f)
        return cls(config, config_path=path)

    # --- Domain logic delegated to baseline family ---

    def create_model(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> Any:
        return self._baseline.create_model(params, device=device)

    def generate_analysis_dataset(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        return self._baseline.generate_analysis_dataset(params, device=device)

    def generate_training_dataset(
        self,
        params: dict[str, Any],
        training_fraction: float = 0.3,
        data_seed: int = 598,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._baseline.generate_training_dataset(
            params,
            training_fraction=training_fraction,
            data_seed=data_seed,
            device=device,
        )

    def get_training_config(self) -> dict[str, Any]:
        return self._baseline.get_training_config()

    def prepare_analysis_context(
        self,
        params: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        return self._baseline.prepare_analysis_context(params, device=device)

    def make_probe(
        self,
        params: dict[str, Any],
        inputs: list[list[int]],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        return self._baseline.make_probe(params, inputs, device=device)

    # --- Intervention variant factory ---

    def create_intervention_variant(
        self,
        prime: int,
        seed: int,
        data_seed: int,
        intervention_config: dict[str, Any],
        results_dir: Path | str,
    ) -> Variant:
        """Create a Variant with a specific intervention config.

        The intervention ID is computed from the config hash and included in
        the variant's params so it appears in the directory name. The full
        intervention config is also stored in params so it is written to
        config.json by Variant.train().

        Args:
            prime: Modulus for the addition task
            seed: Random seed for model initialization
            data_seed: Random seed for train/test split
            intervention_config: Intervention parameter dict. Must include
                at minimum: type, target_heads, target_frequencies, gain,
                epoch_start, epoch_end, ramp_epochs.
            results_dir: Root results directory

        Returns:
            Variant ready for training via variant.train(training_hook=...)
        """
        intervention_id = compute_intervention_id(intervention_config)
        params: dict[str, Any] = {
            "prime": prime,
            "seed": seed,
            "data_seed": data_seed,
            "intervention_id": intervention_id,
            "intervention": intervention_config,
        }
        return Variant(self, params, Path(results_dir))
