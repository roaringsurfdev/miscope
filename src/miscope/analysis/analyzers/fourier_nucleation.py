"""Fourier Nucleation Analyzer (REQ_063).

Projects neuron response profiles onto the Fourier basis to surface latent
frequency bias in MLP weights. For a given checkpoint, computes how each
neuron's response across token values aligns with each Fourier frequency.
Iterative sharpening amplifies dominant spectral features.

The key projection: W_in @ W_E.T gives the neuron response matrix in token
space — (d_mlp, p) — where each row describes how strongly neuron n responds
to each token value 0..p-1. This is the right space to project onto the
Fourier basis, since Fourier frequencies are defined over token values.

This analyzer is intended to run at epoch 0 (initialization). The dashboard
view hardcodes loading epoch 0. Running at later epochs is valid and produces
a cross-epoch frequency commitment trajectory.
"""

from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache

_COMMIT_THRESHOLD = 0.15  # Fraction of neuron energy at one frequency to count as "committed"


def _build_fourier_basis(prime: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build normalized sin/cos basis for Fourier analysis over token values.

    Args:
        prime: The modulus p. Frequencies k run from 1 to p//2.

    Returns:
        Tuple of (frequencies, cos_basis, sin_basis):
            frequencies: int array of shape (n_freqs,), values 1..p//2
            cos_basis: float64 array of shape (n_freqs, p), normalized
            sin_basis: float64 array of shape (n_freqs, p), normalized
    """
    frequencies = np.arange(1, prime // 2 + 1)
    positions = np.arange(prime)
    angles = 2 * np.pi * frequencies[:, None] * positions[None, :] / prime
    cos_basis = np.cos(angles)
    sin_basis = np.sin(angles)
    cos_basis /= np.linalg.norm(cos_basis, axis=1, keepdims=True)
    sin_basis /= np.linalg.norm(sin_basis, axis=1, keepdims=True)
    return frequencies, cos_basis, sin_basis


def _project(R: np.ndarray, cos_basis: np.ndarray, sin_basis: np.ndarray) -> np.ndarray:
    """Project neuron response matrix onto Fourier basis.

    Args:
        R: Neuron response matrix of shape (d_mlp, p)
        cos_basis: Normalized cosine basis of shape (n_freqs, p)
        sin_basis: Normalized sine basis of shape (n_freqs, p)

    Returns:
        energy: float64 array of shape (d_mlp, n_freqs) — squared projection magnitudes
    """
    proj_cos = R @ cos_basis.T  # (d_mlp, n_freqs)
    proj_sin = R @ sin_basis.T  # (d_mlp, n_freqs)
    return proj_cos, proj_sin, proj_cos**2 + proj_sin**2


def _sharpen(
    R: np.ndarray,
    proj_cos: np.ndarray,
    proj_sin: np.ndarray,
    energy: np.ndarray,
    cos_basis: np.ndarray,
    sin_basis: np.ndarray,
    sharpness: float,
) -> np.ndarray:
    """Threshold and reconstruct neuron response matrix.

    Keeps only the top (1 - sharpness) fraction of frequencies by energy
    for each neuron, then reconstructs from those Fourier components.

    Args:
        R: Current response matrix (d_mlp, p)
        proj_cos: Cosine projections (d_mlp, n_freqs)
        proj_sin: Sine projections (d_mlp, n_freqs)
        energy: Energy per (neuron, frequency) (d_mlp, n_freqs)
        cos_basis: Normalized cosine basis (n_freqs, p)
        sin_basis: Normalized sine basis (n_freqs, p)
        sharpness: Fraction of frequencies to zero out (0 = no sharpening, 1 = keep only peak)

    Returns:
        new_R: Sharpened response matrix (d_mlp, p)
    """
    n_freqs = energy.shape[1]
    keep_count = max(1, int(n_freqs * (1.0 - sharpness)))

    # Per-neuron threshold: energy at the keep_count-th rank (descending)
    sorted_energy = np.sort(energy, axis=1)[:, ::-1]  # (d_mlp, n_freqs) desc
    threshold = sorted_energy[:, keep_count - 1:keep_count]  # (d_mlp, 1)

    keep_mask = (energy >= threshold).astype(np.float64)  # (d_mlp, n_freqs)

    masked_cos = proj_cos * keep_mask  # (d_mlp, n_freqs)
    masked_sin = proj_sin * keep_mask

    return masked_cos @ cos_basis + masked_sin @ sin_basis  # (d_mlp, p)


def _snapshot(
    energy: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-iteration summary statistics from energy matrix.

    Args:
        energy: (d_mlp, n_freqs) energy per neuron per frequency
        frequencies: (n_freqs,) frequency values

    Returns:
        agg_energy: (n_freqs,) normalized aggregate energy across neurons
        peak_freq: (d_mlp,) peak frequency assignment per neuron
        committed_count: (n_freqs,) count of neurons with >15% energy at each frequency
    """
    agg = energy.sum(axis=0)
    agg_max = agg.max()
    agg_energy = (agg / agg_max if agg_max > 0 else agg).astype(np.float32)

    peak_idx = energy.argmax(axis=1)
    peak_freq = frequencies[peak_idx].astype(np.int32)

    total_energy = energy.sum(axis=1, keepdims=True).clip(min=1e-10)
    fraction = energy / total_energy
    committed_count = (fraction > _COMMIT_THRESHOLD).sum(axis=0).astype(np.int32)

    return agg_energy, peak_freq, committed_count


class FourierNucleationAnalyzer:
    """Surfaces latent Fourier frequency bias in MLP weights via iterative projection.

    For each checkpoint, computes the neuron response matrix W_in @ W_E.T (d_mlp × p),
    then projects each row onto the Fourier basis for the task's prime p.
    Iterative sharpening amplifies dominant spectral features and suppresses noise.

    Intended primarily for epoch 0 (initialization), but valid at any epoch.
    The dashboard nucleation view hardcodes loading epoch 0.
    """

    name = "fourier_nucleation"
    description = "Iterative Fourier projection of MLP neuron response profiles"

    def __init__(self, iterations: int = 12, sharpness: float = 0.7):
        self.iterations = iterations
        self.sharpness = sharpness

    def analyze(
        self,
        model: HookedTransformer,
        probe: torch.Tensor,
        cache: ActivationCache,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Project neuron response profiles onto Fourier basis, iteratively sharpening.

        Args:
            model: The model loaded with checkpoint weights
            probe: Unused (protocol conformance)
            cache: Unused (protocol conformance)
            context: Analysis context; must contain context["params"]["prime"]

        Returns:
            Dict with keys: aggregate_energy, neuron_peak_freq, neuron_committed_count,
            frequencies, prime, iterations, sharpness
        """
        prime = int(context["params"]["prime"])

        W_in = model.blocks[0].mlp.W_in.detach().cpu().numpy()  # (d_mlp, d_model)
        W_E = model.embed.W_E.detach().cpu().numpy()  # (vocab_size, d_model)

        # W_in is (d_model, d_mlp) in TransformerLens convention.
        # Neuron response to each token: (W_E[:prime] @ W_in).T = (d_mlp, prime)
        R = (W_E[:prime] @ W_in)  # (prime, d_mlp) → transposed below
        R = R.T  # (d_mlp, prime)

        frequencies, cos_basis, sin_basis = _build_fourier_basis(prime)
        n_iters = self.iterations
        n_freqs = len(frequencies)
        d_mlp = R.shape[0]

        all_agg_energy = np.zeros((n_iters + 1, n_freqs), dtype=np.float32)
        all_peak_freq = np.zeros((n_iters + 1, d_mlp), dtype=np.int32)
        all_committed_count = np.zeros((n_iters + 1, n_freqs), dtype=np.int32)

        current_R = R.copy()

        for it in range(n_iters + 1):
            proj_cos, proj_sin, energy = _project(current_R, cos_basis, sin_basis)
            agg, peak, committed = _snapshot(energy, frequencies)

            all_agg_energy[it] = agg
            all_peak_freq[it] = peak
            all_committed_count[it] = committed

            if it < n_iters:
                current_R = _sharpen(
                    current_R, proj_cos, proj_sin, energy,
                    cos_basis, sin_basis, self.sharpness,
                )

        return {
            "aggregate_energy": all_agg_energy,
            "neuron_peak_freq": all_peak_freq,
            "neuron_committed_count": all_committed_count,
            "frequencies": frequencies.astype(np.int32),
            "prime": np.array(prime, dtype=np.int32),
            "iterations": np.array(n_iters, dtype=np.int32),
            "sharpness": np.array(self.sharpness, dtype=np.float32),
        }
