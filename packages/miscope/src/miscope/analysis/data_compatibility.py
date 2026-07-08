"""Fourier Data Compatibility Analyzer (REQ_064).

Measures how well each Fourier frequency can be learned from a given
train/test split. Pure computation on task structure — no model weights,
no checkpoints.

The training split is reconstructed exactly using the same logic as the
training pipeline: torch.manual_seed(data_seed) + torch.randperm(p*p),
first int(p*p*training_fraction) indices form the training set.

The key metric per frequency k is the 2×2 Gram matrix of the restricted
Fourier basis on the training set. Its condition number measures how
well the cos and sin components of frequency k are independently
represented — a high condition number means the model effectively has
only one degree of freedom at that frequency, which limits its ability
to learn an arbitrary phase alignment.
"""

import numpy as np
import torch

_N_PHASE_BINS = 20
_CONDITION_CLIP = 1e-10  # Prevents division by zero in condition number


def compute_data_compatibility(
    prime: int,
    data_seed: int,
    training_fraction: float = 0.3,
) -> dict:
    """Compute per-frequency data compatibility for a train/test split.

    Reconstructs the training set from (prime, data_seed) using the exact
    same logic as the training pipeline. For each Fourier frequency k,
    computes how well the training set supports learning that frequency.

    Args:
        prime: The modulus p for the modular addition task.
        data_seed: Random seed used to generate the train/test split.
        training_fraction: Fraction of pairs used for training (default 0.3).

    Returns:
        Dict with keys:
            frequencies: int array (n_freqs,), values 1..p//2
            condition_number: float array (n_freqs,)
            condition_score: float array (n_freqs,), in [0, 1]
            phase_uniformity: float array (n_freqs,), in [0, 1]
            compatibility_score: float array (n_freqs,), composite [0, 1]
            prime: int
            data_seed: int
            training_fraction: float
            n_training_pairs: int
    """
    s_vals = _reconstruct_training_sums(prime, data_seed, training_fraction)
    n_train = len(s_vals)

    frequencies = np.arange(1, prime // 2 + 1)
    condition_number = _compute_condition_numbers(frequencies, s_vals, prime)
    condition_score = 1.0 / (1.0 + np.log10(np.maximum(1.0, condition_number)))
    phase_uniformity = _compute_phase_uniformity(frequencies, s_vals, prime)
    compatibility_score = 0.5 * condition_score + 0.5 * phase_uniformity

    return {
        "frequencies": frequencies.astype(np.int32),
        "condition_number": condition_number.astype(np.float32),
        "condition_score": condition_score.astype(np.float32),
        "phase_uniformity": phase_uniformity.astype(np.float32),
        "compatibility_score": compatibility_score.astype(np.float32),
        "prime": prime,
        "data_seed": data_seed,
        "training_fraction": training_fraction,
        "n_training_pairs": n_train,
    }


def _reconstruct_training_sums(
    prime: int,
    data_seed: int,
    training_fraction: float,
) -> np.ndarray:
    """Reconstruct the training set and return sum residues s = (a+b) % p.

    Matches training pipeline exactly:
        torch.manual_seed(data_seed)
        indices = torch.randperm(p*p)
        train_indices = indices[:int(p*p*training_fraction)]

    Args:
        prime: The modulus p.
        data_seed: Random seed for the split.
        training_fraction: Fraction of pairs used for training.

    Returns:
        s_vals: int array of shape (n_train,), residues for each training pair.
    """
    torch.manual_seed(data_seed)
    all_indices = torch.randperm(prime * prime).numpy()
    n_train = int(prime * prime * training_fraction)
    train_indices = all_indices[:n_train]

    a_vals = train_indices // prime
    b_vals = train_indices % prime
    return (a_vals + b_vals) % prime


def _compute_condition_numbers(
    frequencies: np.ndarray,
    s_vals: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Compute Gram matrix condition numbers for all frequencies.

    For each k, builds the 2×2 Gram matrix of the restricted Fourier basis
    on the training set and returns its eigenvalue ratio.

    Args:
        frequencies: Array of frequency values, shape (n_freqs,).
        s_vals: Residues (a+b)%p for each training pair, shape (n_train,).
        prime: The modulus p.

    Returns:
        condition_number: float array of shape (n_freqs,).
    """
    angles = 2 * np.pi * frequencies[:, None] * s_vals[None, :] / prime  # (n_freqs, n_train)
    cos_vals = np.cos(angles)  # (n_freqs, n_train)
    sin_vals = np.sin(angles)  # (n_freqs, n_train)

    g_aa = (cos_vals**2).sum(axis=1)  # (n_freqs,)
    g_ab = (cos_vals * sin_vals).sum(axis=1)
    g_bb = (sin_vals**2).sum(axis=1)

    # 2×2 symmetric eigenvalues: (tr/2) ± sqrt((tr/2 - a)^2 + b^2)
    half_tr = (g_aa + g_bb) / 2.0
    disc = np.sqrt(np.maximum(((g_aa - g_bb) / 2.0) ** 2 + g_ab**2, 0.0))
    lambda_max = half_tr + disc
    lambda_min = half_tr - disc

    return lambda_max / np.maximum(lambda_min, _CONDITION_CLIP)


def _compute_phase_uniformity(
    frequencies: np.ndarray,
    s_vals: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Compute normalized phase entropy for all frequencies.

    For each k, builds a histogram of the phase 2πk·s/p over the training
    set and computes its entropy normalized by the maximum entropy.

    Args:
        frequencies: Array of frequency values, shape (n_freqs,).
        s_vals: Residues for each training pair, shape (n_train,).
        prime: The modulus p.

    Returns:
        phase_uniformity: float array of shape (n_freqs,), in [0, 1].
    """
    max_entropy = np.log(_N_PHASE_BINS)
    uniformity = np.zeros(len(frequencies), dtype=np.float64)

    for ki, k in enumerate(frequencies):
        phases = (2 * np.pi * k * s_vals / prime) % (2 * np.pi)
        counts, _ = np.histogram(phases, bins=_N_PHASE_BINS, range=(0.0, 2 * np.pi))
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        uniformity[ki] = entropy / max_entropy

    return uniformity
