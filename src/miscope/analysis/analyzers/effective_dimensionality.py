"""Effective Dimensionality Analyzer.

Computes singular values of all trainable weight matrices per checkpoint.
Stores full singular value spectra for downstream metrics (participation
ratio, stable rank, etc.). Summary statistics provide participation ratios
for trajectory visualization without loading per-epoch artifacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from miscope.analysis.library.weights import (
    WEIGHT_MATRIX_NAMES,
    compute_participation_ratio,
    compute_weight_singular_values,
)

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle


class EffectiveDimensionalityAnalyzer:
    """Computes per-matrix singular value spectra across training.

    For each checkpoint, extracts all 9 weight matrices and computes
    their singular values. Attention matrices are decomposed per head.

    Per-epoch artifacts contain singular value arrays (sv_W_E, sv_W_Q, etc.).
    Summary statistics contain participation ratios (pr_W_E, pr_W_Q, etc.).
    """

    name = "effective_dimensionality"
    description = "Computes weight matrix singular values for dimensionality analysis"

    def analyze(
        self,
        bundle: ActivationBundle,
        probe: torch.Tensor,  # noqa: ARG002
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Compute singular values of all trainable weight matrices.

        Args:
            bundle: Activation bundle with checkpoint weights.
            probe: Unused (protocol conformance).
            context: Unused (protocol conformance).

        Returns:
            Dict mapping sv_{name} to singular value arrays.
        """
        return compute_weight_singular_values(bundle)

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
