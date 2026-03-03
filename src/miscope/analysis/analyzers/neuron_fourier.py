"""Neuron Fourier Decomposition Analyzer (REQ_049).

Secondary analyzer that computes per-neuron Fourier decomposition of MLP
weights (W_in and W_out) from parameter_snapshot artifacts. Produces the
per-epoch Fourier magnitude and phase spectra required for phase alignment,
IPR, lottery ticket, and neuron specialization analyses.
"""

from typing import Any

import numpy as np

from miscope.analysis.library import extract_frequency_pairs


class NeuronFourierAnalyzer:
    """Computes per-neuron Fourier decomposition of MLP weights.

    A secondary analyzer that derives per-neuron Fourier magnitudes and
    phases for both MLP layers from parameter_snapshot artifacts.

    For each epoch, computes effective weight vectors in token/output space:
        θ_m = W_E[:p] @ W_in[:, m]   — input sensitivity per token
        ξ_m = W_out[m, :] @ W_U      — output contribution per logit

    Projects onto the family Fourier basis and extracts per-frequency-pair
    magnitudes (α_mk, β_mk) and phases (φ_mk, ψ_mk) following the
    He et al. (2026) convention.
    """

    name = "neuron_fourier"
    depends_on = "parameter_snapshot"

    def analyze(
        self,
        artifact: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute per-neuron Fourier decomposition for one epoch.

        Args:
            artifact: parameter_snapshot artifact containing W_E, W_in,
                      W_out, W_U as numpy arrays.
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

        W_E = artifact["W_E"]  # (p+1, d_model) — includes equals token
        W_in = artifact["W_in"]  # (d_model, d_mlp)
        W_out = artifact["W_out"]  # (d_mlp, d_model)
        W_U = artifact["W_U"]  # (d_model, p)

        theta = _compose_input_weights(W_E, W_in, p)  # (p, M)
        xi = _compose_output_weights(W_out, W_U)  # (M, p)

        G = fourier_basis @ theta  # (p, M) — Fourier coefficients of θ_m
        R = fourier_basis @ xi.T  # (p, M) — Fourier coefficients of ξ_m

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


def _compose_input_weights(
    W_E: np.ndarray,
    W_in: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Compute effective input weights in token space.

    θ_m = W_E[:p] @ W_in[:, m]

    Args:
        W_E:   shape (p+1, d_model) — embedding matrix including equals token
        W_in:  shape (d_model, d_mlp) — MLP input projection
        prime: The prime p (number of token inputs, excluding equals token)

    Returns:
        shape (p, d_mlp) — column m is the effective input weight θ_m ∈ ℝ^p
    """
    return W_E[:prime] @ W_in  # (p, d_mlp)


def _compose_output_weights(
    W_out: np.ndarray,
    W_U: np.ndarray,
) -> np.ndarray:
    """Compute effective output weights in logit space.

    ξ_m = W_out[m, :] @ W_U

    Args:
        W_out: shape (d_mlp, d_model) — MLP output projection
        W_U:   shape (d_model, p) — unembedding matrix

    Returns:
        shape (d_mlp, p) — row m is the effective output weight ξ_m ∈ ℝ^p
    """
    return W_out @ W_U  # (d_mlp, p)
