"""REQ_055: Attention Head Fourier Decomposition Analyzer.

Computes the Fourier decomposition of each attention head's QK^T product
and value projection. For the Fourier algorithm for modular addition, the
QK^T matrix should be dominated by a single frequency component when Q and K
are jointly aligned in the same Fourier subspace.

Output per epoch: qk_freq_norms (n_heads, n_freq), v_freq_norms (n_heads, n_freq).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle


class AttentionFourierAnalyzer:
    """Per-epoch Fourier decomposition of attention head weight matrices.

    Projects each head's Q and K through the token embedding matrix,
    computes QK^T, and decomposes it in the prime-based Fourier basis.
    Does the same for V. Stores per-head, per-frequency energy fractions.

    Probe and cache are accepted for protocol conformance but unused —
    this analyzer operates purely on weight matrices.
    """

    name = "attention_fourier"
    description = "Fourier decomposition of QK^T and V per attention head"
    architecture_support = ["transformer"]

    def analyze(
        self,
        bundle: ActivationBundle,
        probe: torch.Tensor,  # noqa: ARG002
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Decompose each head's QK^T and V into Fourier frequency fractions.

        Args:
            bundle: Activation bundle with checkpoint weights.
            probe: Unused (protocol conformance).
            context: Must include 'fourier_basis': Tensor (p+1, p).

        Returns:
            Dict with:
            - qk_freq_norms: (n_heads, n_freq) — fraction of QK^T energy per frequency.
            - v_freq_norms: (n_heads, n_freq) — fraction of V energy per frequency.
        """
        fourier_basis = context["fourier_basis"]  # (p+1, p)
        p = fourier_basis.shape[1]
        n_freq = p // 2

        W_E_tok = bundle.weight("W_E").detach()[:p]  # (p, d_model) — token rows only
        W_Q = bundle.weight("W_Q").detach()  # (n_heads, d_model, d_head)
        W_K = bundle.weight("W_K").detach()  # (n_heads, d_model, d_head)
        W_V = bundle.weight("W_V").detach()  # (n_heads, d_model, d_head)

        n_heads = W_Q.shape[0]
        qk_freq_norms = np.zeros((n_heads, n_freq), dtype=np.float32)
        v_freq_norms = np.zeros((n_heads, n_freq), dtype=np.float32)

        for h in range(n_heads):
            qk_freq_norms[h] = _compute_qk_freq_fractions(
                W_E_tok, W_Q[h], W_K[h], fourier_basis, n_freq
            )
            v_freq_norms[h] = _compute_v_freq_fractions(W_E_tok, W_V[h], fourier_basis, n_freq)

        return {
            "qk_freq_norms": qk_freq_norms,
            "v_freq_norms": v_freq_norms,
        }


def _compute_qk_freq_fractions(
    W_E_tok: torch.Tensor,
    W_Q_h: torch.Tensor,
    W_K_h: torch.Tensor,
    F: torch.Tensor,
    n_freq: int,
) -> np.ndarray:
    """Compute per-frequency energy fractions of QK^T in Fourier space."""
    Q = W_E_tok @ W_Q_h  # (p, d_head)
    K = W_E_tok @ W_K_h  # (p, d_head)
    QK = Q @ K.T  # (p, p)
    QK_fourier = F @ QK @ F.T  # (p+1, p+1)
    energies = _qk_block_norms(QK_fourier, n_freq)
    total = energies.sum().clamp(min=1e-10)
    return (energies / total).cpu().numpy()


def _compute_v_freq_fractions(
    W_E_tok: torch.Tensor,
    W_V_h: torch.Tensor,
    F: torch.Tensor,
    n_freq: int,
) -> np.ndarray:
    """Compute per-frequency energy fractions of V projection in Fourier space."""
    V = W_E_tok @ W_V_h  # (p, d_head)
    V_fourier = F @ V  # (p+1, d_head)
    energies = _v_band_norms(V_fourier, n_freq)
    total = energies.sum().clamp(min=1e-10)
    return (energies / total).cpu().numpy()


def _qk_block_norms(QK_fourier: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Extract Frobenius norms of per-frequency 2x2 blocks from QK_fourier.

    For frequency k (1-indexed): sin idx = 2k-1, cos idx = 2k.
    The 2x2 block at these indices captures the k-th frequency component.
    """
    energies = torch.zeros(n_freq, dtype=QK_fourier.dtype, device=QK_fourier.device)
    n = QK_fourier.shape[0]
    for k in range(1, n_freq + 1):
        si, ci = 2 * k - 1, 2 * k
        if ci < n:
            block = QK_fourier[[si, ci], :][:, [si, ci]]
            energies[k - 1] = block.pow(2).sum().sqrt()
    return energies


def _v_band_norms(V_fourier: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Extract per-frequency band norms from V_fourier (p+1, d_head)."""
    energies = torch.zeros(n_freq, dtype=V_fourier.dtype, device=V_fourier.device)
    n = V_fourier.shape[0]
    for k in range(1, n_freq + 1):
        si, ci = 2 * k - 1, 2 * k
        if ci < n:
            band = V_fourier[[si, ci], :]
            energies[k - 1] = band.pow(2).sum().sqrt()
    return energies
