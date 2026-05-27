"""Weight Spectra Analyzer (REQ_111).

Successor to ``effective_dimensionality``. Same per-epoch shape, plus
left/right singular vectors retained alongside the singular values so
downstream views (subspace alignment, spectral basis projections) can
work from artifacts directly rather than recomputing SVD on demand.

Routes through :func:`miscope.analysis.library.pca.compute_svd` via the
:func:`compute_weight_spectra` helper — no inline ``np.linalg.svd`` in
``analyze()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.inputs import ModelInput, ResolvedInputs
from miscope.analysis.library.weights import (
    WEIGHT_MATRIX_NAMES,
    compute_participation_ratio,
    compute_weight_spectra,
)
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

SPEC = AnalyzerSpec(
    name="weight_spectra",
    output_scope="per_epoch",
    inputs=(ModelInput(needs_weights=True, needs_cache=False),),
    required_hooks=(),
    produces_summary=True,
)


@register_analyzer(SPEC)
class WeightSpectraAnalyzer:
    """Per-matrix SVD of all trainable weight matrices across training.

    Per-epoch artifacts contain, for each available weight matrix:

    - ``sv_{name}`` — singular values (same as legacy ``effective_dimensionality``).
    - ``u_{name}`` — left singular vectors.
    - ``vt_{name}`` — right singular vectors.

    Attention matrices (W_Q/W_K/W_V/W_O) decompose per head, so all three
    components carry a leading ``n_heads`` axis.

    Summary statistics provide participation ratios per matrix.
    """

    name = "weight_spectra"
    description = (
        "Per-matrix SVD (singular values + left/right vectors) for "
        "spectral and subspace analyses across training."
    )

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute full SVD of all trainable weight matrices."""
        assert inputs.model is not None  # type-narrowing for pyright
        spectra = compute_weight_spectra(inputs.model)
        result: dict[str, np.ndarray] = {}
        for name, (u, s, vt) in spectra.items():
            result[f"sv_{name}"] = s
            result[f"u_{name}"] = u
            result[f"vt_{name}"] = vt
        return result

    def get_summary_keys(self) -> list[str]:
        """Declare participation ratio summary keys."""
        return [f"pr_{name}" for name in WEIGHT_MATRIX_NAMES]

    def compute_summary(
        self, result: dict[str, np.ndarray], context: dict[str, Any]
    ) -> dict[str, float | np.ndarray]:
        """Compute participation ratios from singular values."""
        summary: dict[str, float | np.ndarray] = {}
        for name in WEIGHT_MATRIX_NAMES:
            sv_key = f"sv_{name}"
            if sv_key in result:
                summary[f"pr_{name}"] = compute_participation_ratio(result[sv_key])
        return summary
