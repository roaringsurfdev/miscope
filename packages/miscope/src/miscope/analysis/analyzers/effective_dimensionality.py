"""Effective Dimensionality Analyzer.

Computes singular values of all trainable weight matrices per checkpoint.
Stores full singular value spectra for downstream metrics (participation
ratio, stable rank, etc.). Summary statistics provide participation ratios
for trajectory visualization without loading per-epoch artifacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from miscope.analysis.library.weights import (
    WEIGHT_MATRIX_NAMES,
    compute_participation_ratio,
    compute_weight_singular_values,
)
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext


SPEC = AnalyzerSpec(
    name="effective_dimensionality",
    category="primary",
    requires_model_weights=True,
    requires_activation_cache=False,  # reads weights only; no cache access
    required_hooks=(),
    produces_summary=True,
)


@register_analyzer(SPEC)
class EffectiveDimensionalityAnalyzer:
    """Computes per-matrix singular value spectra across training.

    For each checkpoint, extracts all 9 weight matrices and computes
    their singular values. Attention matrices are decomposed per head.

    Per-epoch artifacts contain singular value arrays (sv_W_E, sv_W_Q, etc.).
    Summary statistics contain participation ratios (pr_W_E, pr_W_Q, etc.).
    """

    name = "effective_dimensionality"
    description = "Computes weight matrix singular values for dimensionality analysis"
    # Reads weights only — runs on any architecture.
    required_hooks: list[str] = []

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """Compute singular values of all trainable weight matrices.

        Args:
            ctx: Analysis context with model (probe and analysis_params unused).

        Returns:
            Dict mapping sv_{name} to singular value arrays.
        """
        assert ctx.model is not None  # type-narrowing for pyright
        return compute_weight_singular_values(ctx.model)

    def get_summary_keys(self) -> list[str]:
        """Declare participation ratio summary keys."""
        return [f"pr_{name}" for name in WEIGHT_MATRIX_NAMES]

    def compute_summary(
        self, result: dict[str, np.ndarray], context: dict[str, Any]
    ) -> dict[str, float | np.ndarray]:
        """Compute participation ratios from singular values.

        Args:
            result: Dict with sv_{name} arrays from analyze()
            context: Analysis context (unused)

        Returns:
            Dict mapping pr_{name} to participation ratio (scalar or per-head array).
        """
        summary = {}
        for name in WEIGHT_MATRIX_NAMES:
            sv_key = f"sv_{name}"
            pr_key = f"pr_{name}"
            if sv_key in result:
                summary[pr_key] = compute_participation_ratio(result[sv_key])
        return summary
