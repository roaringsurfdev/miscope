"""Fourier Frequency Quality Analyzer (REQ_052).

Secondary analyzer that scores the model's dominant frequency selection
against the ideal Fourier structure of the mod-p addition task.

Quality is the R² of projecting the ideal p×p×p logit tensor onto the
2D Fourier subspace spanned by the dominant frequency indices. Analytically
this equals 2m/p for m complete sin/cos frequency pairs, but is computed
numerically so partial pairs are handled correctly.
"""

from typing import Any

import numpy as np


class FourierFrequencyQualityAnalyzer:
    """Scores dominant frequency selection against the mod-p addition ideal.

    A secondary analyzer that reads per-epoch dominant_frequencies artifacts
    and computes how much of the ideal mod-p addition logit tensor is
    explained by the model's selected frequency subset.

    Quality score is R²: ||T_F||² / ||T||², where T is the ideal p×p×p
    logit tensor and T_F is its projection onto the 2D Fourier subspace
    defined by the dominant frequency indices.

    Dominant frequencies are those with coefficient > threshold_factor × mean,
    using a relative threshold that adapts to the per-epoch distribution.
    At initialization all coefficients are roughly equal, so nothing qualifies
    as "dominant" and quality starts near 0. As the model concentrates energy
    into a few frequencies, those spike above the mean and quality rises.
    """

    name = "fourier_frequency_quality"
    depends_on = "dominant_frequencies"

    def __init__(self, threshold_factor: float = 3.0):
        self.threshold_factor = threshold_factor

    def analyze(
        self,
        artifact: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute frequency quality score for one epoch.

        Args:
            artifact: dominant_frequencies artifact containing 'coefficients'
                      array of shape (p,).
            context: Analysis context containing 'params' (with 'prime') and
                     'fourier_basis' (normalized torch.Tensor, shape (p, p)).

        Returns:
            Dict with:
                quality_score:        scalar R² in [0, 1]
                dominant_frequencies: int array of dominant basis indices
                k:                    number of dominant basis vectors
                reconstruction_error: scalar (1 - quality_score)
        """
        p = context["params"]["prime"]
        fourier_basis = context["fourier_basis"].cpu().numpy()  # (p, p)
        coefficients = artifact["coefficients"]  # (p,)

        threshold = self.threshold_factor * float(np.mean(coefficients))
        dominant_indices = np.where(coefficients > threshold)[0]
        k = int(len(dominant_indices))

        quality_score = _compute_quality_score(p, dominant_indices, fourier_basis)

        return {
            "quality_score": np.float32(quality_score),
            "dominant_frequencies": dominant_indices.astype(np.int32),
            "k": np.int32(k),
            "reconstruction_error": np.float32(1.0 - quality_score),
        }

    def get_summary_keys(self) -> list[str]:
        return ["quality_score", "reconstruction_error", "k"]

    def compute_summary(self, result: dict[str, Any], context: dict[str, Any]) -> dict[str, float]:
        return {
            "quality_score": float(result["quality_score"]),
            "reconstruction_error": float(result["reconstruction_error"]),
            "k": float(result["k"]),
        }


def _compute_quality_score(
    p: int,
    dominant_indices: np.ndarray,
    fourier_basis: np.ndarray,
) -> float:
    """Compute R² of projecting the ideal mod-p tensor onto the dominant subspace.

    Projects T_oh[a, b, c] = 1 iff (a+b)%p==c onto the 2D Fourier subspace
    spanned by {F[i] ⊗ F[j] : i, j ∈ dominant_indices}, where F is the
    orthonormal Fourier basis (rows).

    Avoids building the full p×p×p tensor by exploiting the structure:
        T_Fa[i, b, c] = F[i, (c-b)%p]

    Args:
        p: Prime modulus.
        dominant_indices: Array of basis vector indices above threshold.
        fourier_basis: Shape (p, p), rows are orthonormal basis vectors.

    Returns:
        R² quality score in [0, 1].
    """
    if len(dominant_indices) == 0:
        return 0.0

    F_restricted = fourier_basis[dominant_indices, :]  # (m, p)

    # T_Fa[i, b, c] = Σ_a F_restricted[i, a] * T_oh[a, b, c]
    #              = F_restricted[i, (c-b)%p]   (since only a=(c-b)%p contributes)
    b_grid = np.arange(p)[:, None]  # (p, 1)
    c_grid = np.arange(p)[None, :]  # (1, p)
    a_idx = (c_grid - b_grid) % p  # (p, p): a_idx[b, c] = (c-b)%p

    T_Fa = F_restricted[:, a_idx]  # (m, p, p): T_Fa[i, b, c] = F_restricted[i, a_idx[b,c]]

    # T_2D[i, j, c] = Σ_b F_restricted[j, b] * T_Fa[i, b, c]
    T_2D = np.einsum("jb,ibc->ijc", F_restricted, T_Fa)  # (m, m, p)

    projected_energy = float(np.sum(T_2D**2))
    total_energy = float(p**2)  # ||T_oh||_F^2 = p^2 (one 1 per input pair)

    return projected_energy / total_energy
