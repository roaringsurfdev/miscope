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
    ATTENTION_MATRICES,
    WEIGHT_MATRIX_NAMES,
    compute_participation_ratio,
    compute_weight_spectra,
)
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

# Per weight matrix: singular values + SVD factor matrices. The on-disk keys are
# {component}_{site} (e.g. sv_W_E, u_W_E, vt_W_E); declared here as logical
# fields keyed by `site` (the weight matrix). 110-A flattens the composition.
# Attention sites (W_Q/K/V/O) decompose per head; `sv` therefore carries a `head`
# coordinate (REQ_136) — emitted uniform-rank as (n_heads, n_sv), with n_heads=1
# for non-attention sites so the field has one declared shape. The u/vt tensors
# keep their natural per-head shape; a tensor's internal axes are not coords.
SPEC = AnalyzerSpec(
    name="weight_spectra",
    output_scope="per_epoch",
    inputs=(ModelInput(needs_weights=True, needs_cache=False),),
    required_hooks=(),
    produces_summary=True,
    outputs=(
        F.columnar(
            "sv",
            "float32",
            ("variant", "epoch", "site", "head", "row_id"),
            "Singular values of a weight matrix, descending (head = attention head, "
            "head=0 for non-attention sites; row_id = SV index).",
        ),
        F.tensor(
            "u",
            "float32",
            ("variant", "epoch", "site"),
            "Left singular vectors (U) of a weight matrix.",
        ),
        F.tensor(
            "vt",
            "float32",
            ("variant", "epoch", "site"),
            "Right singular vectors (Vᵀ) of a weight matrix.",
        ),
    ),
)


@register_analyzer(SPEC)
class WeightSpectraAnalyzer:
    """Per-matrix SVD of all trainable weight matrices across training.

    Per-epoch artifacts contain, for each available weight matrix:

    - ``sv_{name}`` — singular values (same as legacy ``effective_dimensionality``).
    - ``u_{name}`` — left singular vectors.
    - ``vt_{name}`` — right singular vectors.

    Attention matrices (W_Q/W_K/W_V/W_O) decompose per head, so all three
    components carry a leading ``n_heads`` axis. ``sv`` is emitted uniform-rank
    as ``(n_heads, n_sv)`` for every site — ``n_heads=1`` for non-attention
    sites — so its declared ``head`` coordinate (REQ_136) keys one shape rather
    than folding the head index into ``row_id``.

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
            # Uniform-rank `sv` (REQ_136): attention S is already (n_heads, n_sv);
            # give non-attention S a singleton head axis so the field has one shape.
            result[f"sv_{name}"] = s if s.ndim == 2 else s[np.newaxis, :]
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
            if sv_key not in result:
                continue
            sv = result[sv_key]
            # `sv` is uniform-rank (n_heads, n_sv). Attention keeps one PR per head
            # (a per-head array); non-attention collapses its singleton head axis so
            # PR stays the scalar that downstream summaries expect.
            if name not in ATTENTION_MATRICES:
                sv = sv[0]
            summary[f"pr_{name}"] = compute_participation_ratio(sv)
        return summary
