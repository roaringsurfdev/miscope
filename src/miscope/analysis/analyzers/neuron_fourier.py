"""Neuron Fourier Decomposition Analyzer (REQ_049).

Secondary analyzer that computes per-neuron Fourier decomposition of MLP
weights (W_in and W_out) from parameter_snapshot artifacts. Produces the
per-epoch Fourier magnitude and phase spectra required for phase alignment,
IPR, lottery ticket, and neuron specialization analyses.

Architecture support:
  - transformer: θ_m = W_E[:p] @ W_in[:, m],  ξ_m = W_out[m, :] @ W_U
  - mlp:         θ_m = avg(W_in[m, :p], W_in[m, p:]),  ξ_m = W_out[:, m]
"""

from typing import Any

import numpy as np

from miscope.analysis.library import compose_neuron_fourier_weights, extract_frequency_pairs


class NeuronFourierAnalyzer:
    """Computes per-neuron Fourier decomposition of MLP weights.

    A secondary analyzer that derives per-neuron Fourier magnitudes and
    phases for both MLP layers from parameter_snapshot artifacts.

    Dispatches on architecture based on artifact contents (presence of W_E).
    For each epoch, computes effective weight vectors in token/output space
    and projects onto the family Fourier basis, extracting per-frequency-pair
    magnitudes (α_mk, β_mk) and phases (φ_mk, ψ_mk) following the
    He et al. (2026) convention.
    """

    name = "neuron_fourier"
    depends_on = "parameter_snapshot"
    architecture_support = ["transformer", "mlp"]

    def analyze(
        self,
        artifact: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute per-neuron Fourier decomposition for one epoch.

        Args:
            artifact: parameter_snapshot artifact. Transformer artifacts contain
                      W_E, W_in, W_out, W_U; MLP artifacts contain W_in, W_out.
            context: Analysis context containing 'params' (with 'prime') and
                     'fourier_basis' (normalized torch.Tensor, shape (p, p)).

        Returns:
            Dict with:
                alpha_mk:    shape (M, K) — input layer Fourier magnitudes
                phi_mk:      shape (M, K) — input layer Fourier phases ∈ (-π, π]
                beta_mk:     shape (M, K) — output layer Fourier magnitudes
                psi_mk:      shape (M, K) — output layer Fourier phases ∈ (-π, π]
                freq_indices: shape (K,)  — frequency indices k = 1 ... K
            where M = n_neurons (d_mlp), K = (p-1)//2 frequency pairs.
        """
        p = context["params"]["prime"]
        fourier_basis = context["fourier_basis"].cpu().numpy()  # (p, p)

        theta, xi = compose_neuron_fourier_weights(artifact, p)  # both (p, M)

        G = fourier_basis @ theta  # (p, M) — Fourier coefficients of θ_m
        R = fourier_basis @ xi  # (p, M) — Fourier coefficients of ξ_m

        alpha_mk, phi_mk = extract_frequency_pairs(G, p)
        beta_mk, psi_mk = extract_frequency_pairs(R, p)

        k_count = (p - 1) // 2
        freq_indices = np.arange(1, k_count + 1)

        return {
            "alpha_mk": alpha_mk.astype(np.float32),
            "phi_mk": phi_mk.astype(np.float32),
            "beta_mk": beta_mk.astype(np.float32),
            "psi_mk": psi_mk.astype(np.float32),
            "freq_indices": freq_indices.astype(np.int32),
        }
