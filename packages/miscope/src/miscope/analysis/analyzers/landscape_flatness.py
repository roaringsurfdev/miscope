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

from miscope.analysis.library.landscape import compute_landscape_flatness
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext

FLATNESS_SUMMARY_KEYS = [
    "mean_delta_loss",
    "median_delta_loss",
    "max_delta_loss",
    "std_delta_loss",
    "p90_delta_loss",
    "flatness_ratio",
    "baseline_loss",
]


SPEC = AnalyzerSpec(
    name="landscape_flatness",
    category="primary",
    requires_model_weights=True,
    # Probes the model with ctx.probe (re-runs forward passes internally
    # via compute_landscape_flatness) but does NOT read ctx.cache directly.
    # Pipeline-loaded cache is unused.
    requires_activation_cache=False,
    required_hooks=(),
    produces_summary=True,
)


@register_analyzer(SPEC)
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
    # Reads model parameters directly via state_dict / parameters() — no
    # canonical-name surface used. Runs on any HookedModel architecture.
    required_hooks: list[str] = []

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
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """Compute landscape flatness for a single checkpoint.

        Args:
            ctx: Analysis context with bundle, probe, and analysis_params.
                 analysis_params must contain 'loss_fn'.

        Returns:
            Dict with baseline_loss, delta_losses, epsilon.

        Raises:
            ValueError: If 'loss_fn' not found in analysis_params.
        """
        assert ctx.model is not None  # type-narrowing for pyright
        if "loss_fn" not in ctx.analysis_params:
            raise ValueError(
                "LandscapeFlatnessAnalyzer requires 'loss_fn' in analysis "
                "context. Ensure the model family's "
                "prepare_analysis_context() provides it."
            )

        # Landscape flatness requires direct parameter manipulation
        # (state_dict / parameters / load_state_dict). ``ctx.model`` is
        # the HookedModel; nn.Module surface is sufficient for the
        # perturbation logic in compute_landscape_flatness.
        return compute_landscape_flatness(
            model=ctx.model,
            probe=ctx.probe,
            loss_fn=ctx.analysis_params["loss_fn"],
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
