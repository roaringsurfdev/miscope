"""Landscape Flatness Analyzer (REQ_031).

Measures local loss landscape flatness by randomly perturbing model
weights and observing loss changes. Flat regions tolerate large
perturbations; sharp regions don't.

Per-epoch artifacts contain the full delta_losses distribution.
Summary statistics provide flatness metrics for trajectory visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from miscope.analysis.library.landscape import compute_landscape_flatness

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle

FLATNESS_SUMMARY_KEYS = [
    "mean_delta_loss",
    "median_delta_loss",
    "max_delta_loss",
    "std_delta_loss",
    "p90_delta_loss",
    "flatness_ratio",
    "baseline_loss",
]


class LandscapeFlatnessAnalyzer:
    """Measures local loss landscape flatness via random perturbation.

    For each checkpoint, samples random unit-norm direction vectors in
    parameter space, perturbs weights by epsilon, and records how much
    the loss changes. Flat basins show small delta_losses.

    Configurable via constructor:
        n_directions: Number of perturbation directions (default 50).
        epsilon: Perturbation magnitude (default 0.1).
        seed: Optional RNG seed for reproducibility.
    """

    name = "landscape_flatness"
    description = "Measures loss landscape flatness via random weight perturbation"
    architecture_support = ["transformer"]

    def __init__(
        self,
        n_directions: int = 50,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        self.n_directions = n_directions
        self.epsilon = epsilon
        self.seed = seed

    def analyze(
        self,
        bundle: ActivationBundle,
        probe: torch.Tensor,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute landscape flatness for a single checkpoint.

        Args:
            bundle: Activation bundle. Uses raw_model for parameter perturbation.
            probe: The analysis dataset tensor.
            context: Must contain 'loss_fn' key.

        Returns:
            Dict with baseline_loss, delta_losses, epsilon.

        Raises:
            ValueError: If 'loss_fn' not found in context.
        """
        if "loss_fn" not in context:
            raise ValueError(
                "LandscapeFlatnessAnalyzer requires 'loss_fn' in analysis "
                "context. Ensure the model family's "
                "prepare_analysis_context() provides it."
            )

        # Landscape flatness requires direct parameter manipulation — the bundle
        # protocol doesn't cover perturbation. TransformerLensBundle.raw_model
        # provides the escape hatch for this transformer-specific use case.
        model = bundle.raw_model  # type: ignore[attr-defined]
        return compute_landscape_flatness(
            model=model,
            probe=probe,
            loss_fn=context["loss_fn"],
            n_directions=self.n_directions,
            epsilon=self.epsilon,
            seed=self.seed,
        )

    def get_summary_keys(self) -> list[str]:
        """Declare summary statistic keys."""
        return FLATNESS_SUMMARY_KEYS

    def compute_summary(
        self,
        result: dict[str, np.ndarray],
        context: dict[str, Any],
    ) -> dict[str, float | np.ndarray]:
        """Compute summary statistics from per-epoch flatness data.

        Args:
            result: Dict with baseline_loss, delta_losses, epsilon.
            context: Analysis context (unused).

        Returns:
            Dict with mean, median, max, std, p90, flatness_ratio,
            and baseline_loss.
        """
        delta = result["delta_losses"]
        baseline = float(result["baseline_loss"])

        threshold = 0.1 * baseline if baseline > 0 else 0.1
        flat_count = np.sum(delta < threshold)
        flatness_ratio = float(flat_count / len(delta))

        return {
            "mean_delta_loss": float(np.mean(delta)),
            "median_delta_loss": float(np.median(delta)),
            "max_delta_loss": float(np.max(delta)),
            "std_delta_loss": float(np.std(delta)),
            "p90_delta_loss": float(np.percentile(delta, 90)),
            "flatness_ratio": flatness_ratio,
            "baseline_loss": baseline,
        }
