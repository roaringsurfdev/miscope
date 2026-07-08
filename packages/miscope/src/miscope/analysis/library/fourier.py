"""Fourier analysis utilities for modular arithmetic.

This module provides functions for computing Fourier bases and projections
used in analyzing models trained on modular arithmetic tasks.

Note: The Fourier basis used here is the discrete Fourier basis for
modular arithmetic (period p), not standard FFT. This is appropriate
for analyzing models that learn modular structure.
"""

import numpy as np
import torch


def get_fourier_basis(
    p: int, device: torch.device | str | None = None
) -> tuple[torch.Tensor, list[str]]:
    """Generate normalized Fourier basis for modular arithmetic with period p.

    Creates a basis of sine and cosine functions at frequencies 1 through p//2,
    plus a constant term. The basis is normalized so each component has unit norm.

    Args:
        p: The modulus (period) for the Fourier basis
        device: Device to place the tensor on (e.g., "cuda", "cpu")

    Returns:
        Tuple of:
        - bases: Tensor of shape (p+1, p) containing the normalized basis vectors.
                 First row is constant, then alternating sin/cos pairs.
        - basis_names: List of human-readable names for each basis component.

    Example:
        >>> basis, names = get_fourier_basis(113, "cuda")
        >>> print(basis.shape)  # (114, 113)
        >>> print(names[:5])  # ['Constant', 'sin k=1', 'cos k=1', 'sin k=2', 'cos k=2']
    """
    bases = []
    basis_names = []
    n = (p // 2) + 1

    # Constant component
    bases.append(torch.ones(p))
    basis_names.append("Constant")

    # Sin and cos components for each frequency
    for frequency in range(1, n):
        theta_range = torch.arange(p) * 2 * torch.pi * frequency / p

        bases.append(torch.sin(theta_range))
        basis_names.append(f"sin k={frequency}")

        bases.append(torch.cos(theta_range))
        basis_names.append(f"cos k={frequency}")

    bases = torch.stack(bases, dim=0)

    if device:
        bases = bases.to(device)

    # Normalize each basis vector
    bases = bases / bases.norm(dim=-1, keepdim=True)

    return bases, basis_names


def project_onto_fourier_basis(weights: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project weight matrix onto Fourier basis and return coefficient norms.

    Computes (basis @ weights).norm(dim=-1), giving the magnitude of each
    Fourier component in the weight representation.

    Args:
        weights: Weight tensor of shape (p, d) where p is the vocabulary/embedding size
        basis: Fourier basis of shape (n_components, p)

    Returns:
        Tensor of shape (n_components,) containing the norm of each coefficient
    """
    coefficients = basis @ weights
    return coefficients.norm(dim=-1)


def compute_2d_fourier_transform(activations: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Apply 2D Fourier transform to activation grid using given basis.

    For activations of shape (..., p, p), computes basis @ activations @ basis.T

    Args:
        activations: Tensor of shape (..., p, p) representing activations over input grid
        basis: Fourier basis of shape (n_components, p)

    Returns:
        Tensor of shape (..., n_components, n_components) in Fourier space
    """
    # basis @ activations @ basis.T
    # activations: (..., p, p)
    # basis: (n_components, p)
    # result: (..., n_components, n_components)
    return basis @ activations @ basis.T


def get_dominant_frequency_indices(coefficients: torch.Tensor, threshold: float = 1.0) -> list[int]:
    """Find indices of Fourier components with coefficients above threshold.

    Args:
        coefficients: Tensor of coefficient norms
        threshold: Minimum coefficient norm to be considered dominant

    Returns:
        List of indices where coefficients exceed threshold
    """
    is_dominant = coefficients > threshold
    dominant_indices = torch.argwhere(is_dominant)
    return dominant_indices.flatten().tolist()


def compute_frequency_variance_fractions(fourier_activations: torch.Tensor, p: int) -> torch.Tensor:
    """Compute fraction of variance explained by each frequency.

    For each neuron (or batch dimension), computes what fraction of the
    total variance in Fourier space is explained by each frequency k.

    Args:
        fourier_activations: Tensor of shape (n_neurons, n_components, n_components)
                            representing 2D Fourier-transformed activations
        p: The modulus (used to determine number of frequencies)

    Returns:
        Tensor of shape (n_frequencies, n_neurons) where n_frequencies = p // 2
    """
    n_neurons = fourier_activations.shape[0]
    n_frequencies = p // 2
    device = fourier_activations.device

    # Center by removing DC component
    fourier_activations = fourier_activations.clone()
    fourier_activations[:, 0, 0] = 0.0

    # Compute variance explained by each frequency
    neuron_freq_norm = torch.zeros(n_frequencies, n_neurons, device=device)

    for freq in range(n_frequencies):
        # Indices for sin and cos of this frequency
        # freq=0 corresponds to frequency 1 (indices 1, 2)
        # freq=k corresponds to frequency k+1 (indices 2k+1, 2k+2)
        for x in [0, 2 * (freq + 1) - 1, 2 * (freq + 1)]:
            for y in [0, 2 * (freq + 1) - 1, 2 * (freq + 1)]:
                if x < fourier_activations.shape[1] and y < fourier_activations.shape[2]:
                    neuron_freq_norm[freq] += fourier_activations[:, x, y] ** 2

    # Normalize by total variance
    total_variance = fourier_activations.pow(2).sum(dim=[-1, -2])
    total_variance = torch.clamp(total_variance, min=1e-10)
    neuron_freq_norm = neuron_freq_norm / total_variance[None, :]

    return neuron_freq_norm


def compute_neuron_coarseness(
    freq_fractions: torch.Tensor,
    n_low_freqs: int = 3,
) -> torch.Tensor:
    """Compute coarseness (low-frequency energy ratio) per neuron.

    Coarseness quantifies how much of a neuron's activation variance is
    explained by low modular frequencies. High coarseness indicates "blob"
    neurons (large coherent activation regions), while low coarseness
    indicates "plaid" neurons (fine-grained checkerboard patterns).

    Args:
        freq_fractions: Per-neuron variance fractions from
            compute_frequency_variance_fractions(), shape (n_frequencies, d_mlp).
        n_low_freqs: Number of lowest frequencies to consider "low".
            Default 3 captures frequencies k=1, 2, 3.

    Returns:
        Tensor of shape (d_mlp,) with coarseness values in [0, 1].
    """
    k = min(n_low_freqs, freq_fractions.shape[0])
    return freq_fractions[:k].sum(dim=0)


def compose_neuron_fourier_weights(
    artifact: dict,
    prime: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose effective input and output weight vectors for Fourier decomposition.

    Dispatches on architecture based on artifact contents:
      - Transformer: theta_m = W_E[:p] @ W_in[:, m]  (embedding composed with MLP input)
                     xi_m    = W_out[m, :] @ W_U      (MLP output composed with unembedding)
      - MLP:         theta_m = (W_in[m, :p] + W_in[m, p:]) / 2  (average of a/b input halves)
                     xi_m    = W_out[:, m]             (column m of output weight)

    Args:
        artifact: parameter_snapshot dict. Transformer artifacts contain 'W_E' and 'W_U';
                  MLP artifacts contain only 'W_in' and 'W_out'.
        prime: The prime p. Used to split MLP W_in into a/b halves.

    Returns:
        (theta, xi): both shape (prime, M) where M = n_neurons.
            theta[:, m] is neuron m's effective input sensitivity per token.
            xi[:, m] is neuron m's effective output contribution per logit.
    """
    W_in = artifact["W_in"]
    W_out = artifact["W_out"]

    if "W_E" in artifact:
        # Transformer path
        W_E = artifact["W_E"]  # (p+1, d_model)
        W_U = artifact["W_U"]  # (d_model, p)
        theta = W_E[:prime] @ W_in  # (p, d_mlp)
        xi = (W_out @ W_U).T  # (p, d_mlp)
    else:
        # MLP path: W_in is (d_hidden, 2p), W_out is (p, d_hidden)
        theta = (W_in[:, :prime] + W_in[:, prime:]).T / 2  # (p, d_hidden)
        xi = W_out  # (p, d_hidden) — already in (p, M) form

    return theta, xi


def extract_frequency_pairs(
    fourier_coeffs: np.ndarray,
    prime: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-frequency-pair magnitudes and phases from Fourier coefficients.

    Interprets a projected coefficient array (produced using the normalized basis
    from get_fourier_basis()) and returns per-frequency-pair magnitudes and phases
    for all K = (prime-1)//2 non-DC frequency pairs.

    Basis index convention (matches get_fourier_basis()):
        Index 0:     Constant (DC) — excluded from output
        Index 2k-1:  sin(k) component for k = 1, ..., K
        Index 2k:    cos(k) component for k = 1, ..., K

    Phase convention (He et al. 2026, §5.1):
        φ_k = atan2(-sin_coeff, cos_coeff)

    Args:
        fourier_coeffs: Array of shape (prime, M) where M is the number of
            weight vectors. Column j contains the prime Fourier coefficients
            for weight vector j. Rows correspond to the normalized basis components.
        prime: The prime p (determines K = (p-1)//2 frequency pairs).

    Returns:
        Tuple (magnitudes, phases):
            magnitudes: shape (M, K) — per (weight vector, frequency) magnitude.
                        magnitudes[m, k] = ||(sin_k_coeff, cos_k_coeff)||_2
            phases:     shape (M, K) — per (weight vector, frequency) phase in (-π, π].
                        phases[m, k] = atan2(-sin_k_coeff, cos_k_coeff)
    """
    k_count = (prime - 1) // 2

    sin_idx = np.array([2 * k - 1 for k in range(1, k_count + 1)])
    cos_idx = np.array([2 * k for k in range(1, k_count + 1)])

    sin_coeffs = fourier_coeffs[sin_idx, :]  # (K, M)
    cos_coeffs = fourier_coeffs[cos_idx, :]  # (K, M)

    magnitudes = np.sqrt(sin_coeffs**2 + cos_coeffs**2).T  # (M, K)
    phases = np.arctan2(-sin_coeffs, cos_coeffs).T  # (M, K)

    return magnitudes, phases
