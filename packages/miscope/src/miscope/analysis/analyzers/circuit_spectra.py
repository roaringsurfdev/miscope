"""REQ_152 / REQ_154: Circuit spectra analyzer.

A *fresh* universal spectral instrument over family-declared **circuit
composition sites** (``circuit_spectra_sites``). For each site — a composed
end-to-end weight circuit such as the full OV path ``W_U W_O W_V W_E`` or the full
QK form ``W_E^T W_Q^T W_K W_E`` (`data_model_master.md` Layer 4) — it measures
gauge-invariant invariants of the composed operand alone: the operator norm, the
effective rank (participation ratio of the singular values), and the eigenspectrum
**copying score**. The dense composed matrix and its eigenvalues are emitted as
tensor refs (blob plane); only the scalar invariants flatten to columns. Each
circuit is one ``site`` row, so the data model's distinct Layer 4 objects
(FullOVCircuit, FullQKCircuit, …) are discriminator-keyed slices of one table.

The copying score is OV-meaningful (a head's copy-vs-transform tendency); it is
computed uniformly for every circuit site and simply not interpreted for non-OV
circuits — cheap and harmless, since the consumer selects the columns it needs.

Boundaries (REQ_152 constraints):

- The composition is the **family's** responsibility — this analyzer embeds no
  per-circuit weight math. It reads ``context["circuit_spectra_sites"]`` exactly
  as ``weight_basis_projection`` reads its Fourier sites.
- Singular values come from the REQ_109 :func:`compute_svd` primitive and the
  :func:`compute_participation_ratio` primitive — no inline ``np.linalg.svd``.
  Only the **copying score** (eigenspectrum positivity) is new here.
- The Fourier ``dominant_frequency`` of the same circuit is supplied separately by
  the universal ``weight_basis_projection`` instrument over the matching site — it
  is never re-derived in this analyzer (invariant 1: instruments are universal).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.inputs import ALL, ArtifactInput, ResolvedInputs
from miscope.analysis.library.pca import compute_svd
from miscope.analysis.library.weights import compute_participation_ratio
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec
from miscope.core.basis_projection import BasisProjectionSite

# Per circuit site: three columnar spectral invariants (per head) + two tensor
# fields (the composed matrix and its eigenvalues). On-disk keys are
# {site}_{field} (PREFIX_US; see warehouse/decompose.py). Each columnar field is
# emitted uniform-rank as (n_heads,) with n_heads=1 for head-less circuits, so its
# declared `head` coordinate (REQ_136) keys one shape.
_COLUMNAR = (
    (
        "copying_score",
        "Eigenspectrum copying score Σ max(Re(λ),0) / Σ|λ| of the composed "
        "circuit, per head (∈ [0, 1]; high = copy-like, low = transform-like).",
    ),
    (
        "effective_rank",
        "Participation ratio of the composed circuit's singular values, per head.",
    ),
    (
        "operator_norm",
        "Largest singular value of the composed circuit, per head.",
    ),
)

SPEC = AnalyzerSpec(
    name="circuit_spectra",
    output_scope="per_epoch",
    inputs=(ArtifactInput("parameter_snapshot"),),
    version=1,
    outputs=(
        *(
            F.columnar(name, "float64", ("variant", "epoch", "site", "head"), desc)
            for name, desc in _COLUMNAR
        ),
        F.tensor(
            "circuit_matrix",
            "float64",
            ("variant", "epoch", "site"),
            "The composed end-to-end circuit matrix, per head (square in token space).",
        ),
        F.tensor(
            "eigenvalues",
            "complex128",
            ("variant", "epoch", "site"),
            "Per-head eigenvalues of the composed circuit (generally complex).",
        ),
    ),
)


@register_analyzer(SPEC)
class CircuitSpectraAnalyzer:
    """Spectral invariants of family-declared composed circuits (REQ_152 / REQ_154).

    For each :class:`BasisProjectionSite` the family declares in
    ``context["circuit_spectra_sites"]``, runs the site's composer on the
    ``parameter_snapshot`` artifact, then computes per-head SVD-derived invariants
    and the eigenspectrum copying score. Outputs are namespaced by site
    (``{site_name}_*`` keys).
    """

    name = "circuit_spectra"
    description = "Spectral invariants + copying score of family-declared composed circuits"
    depends_on = "parameter_snapshot"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Measure each declared circuit site for the current epoch."""
        assert inputs.deps is not None and inputs.epoch is not None
        sites: tuple[BasisProjectionSite, ...] = context.get("circuit_spectra_sites", ())
        if not sites:
            return {}
        snapshot = inputs.deps.load_epoch("parameter_snapshot", inputs.epoch, fields=ALL)

        result: dict[str, np.ndarray] = {}
        for site in sites:
            matrix = np.asarray(site.compose(snapshot, context))
            for key, value in _circuit_spectra(matrix).items():
                result[f"{site.name}_{key}"] = value
        return result


def _circuit_spectra(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Per-head spectral invariants of a stack of square circuit matrices.

    ``matrix`` is ``(n_heads, p, p)``; a head-less ``(p, p)`` circuit is given a
    singleton head axis (uniform-rank, head=0) so every field has one shape.
    """
    if matrix.ndim == 2:
        matrix = matrix[np.newaxis]
    n_heads, dim = matrix.shape[0], matrix.shape[-1]
    copying = np.empty(n_heads, dtype=np.float64)
    effective_rank = np.empty(n_heads, dtype=np.float64)
    operator_norm = np.empty(n_heads, dtype=np.float64)
    eigenvalues = np.empty((n_heads, dim), dtype=np.complex128)
    for h in range(n_heads):
        head_matrix = matrix[h]
        singular_values = compute_svd(head_matrix).singular_values
        operator_norm[h] = float(singular_values[0]) if singular_values.size else 0.0
        effective_rank[h] = float(compute_participation_ratio(singular_values))
        eigenvalues[h] = np.linalg.eigvals(head_matrix)
        copying[h] = _copying_score(eigenvalues[h])
    return {
        "copying_score": copying,
        "effective_rank": effective_rank,
        "operator_norm": operator_norm,
        "circuit_matrix": matrix.astype(np.float64),
        "eigenvalues": eigenvalues,
    }


def _copying_score(eigenvalues: np.ndarray) -> float:
    """Σ max(Re(λ), 0) / Σ |λ| — copy-vs-transform tendency of a circuit (∈ [0, 1]).

    The classic OV-circuit diagnostic (Elhage et al.): a head whose OV eigenvalues
    are positive-real copies its input token; negative/complex eigenvalues mark a
    transforming head. Bounded in ``[0, 1]`` since ``max(Re(λ), 0) ≤ |λ|``.
    """
    total = float(np.abs(eigenvalues).sum())
    if total == 0.0:
        return 0.0
    return float(np.maximum(np.real(eigenvalues), 0.0).sum() / total)
