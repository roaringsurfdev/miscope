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

from miscope.analysis.inputs import ArtifactInput, ResolvedInputs
from miscope.analysis.library import compose_neuron_fourier_weights, extract_frequency_pairs
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

SPEC = AnalyzerSpec(
    name="neuron_fourier",
    output_scope="per_epoch",
    inputs=(ArtifactInput("parameter_snapshot", scope="epoch"),),
)


@register_analyzer(SPEC)
class NeuronFourierAnalyzer:
    """Computes per-neuron Fourier decomposition of MLP weights."""

    name = "neuron_fourier"
    depends_on = "parameter_snapshot"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute per-neuron Fourier decomposition for one epoch."""
        artifact = inputs.artifacts["parameter_snapshot"]
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
