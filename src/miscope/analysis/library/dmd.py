"""REQ_051: Standard Dynamic Mode Decomposition (DMD) library functions.

Implements exact DMD for discrete-time state trajectories. Designed for
centroid class trajectories in global PCA space, but applicable to any
real-valued time series with consistent state dimensionality.

Reference: Tu et al. (2014), "On Dynamic Mode Decomposition: Theory and
Applications." Journal of Computational Dynamics.

Key outputs:
- DMD eigenvalues: oscillatory structure of centroid dynamics
- DMD modes: spatial patterns associated with each dynamical mode
- Residual norms: per-step linear prediction error (primary grokking signal)
"""

import numpy as np


def compute_dmd(
    trajectory: np.ndarray,
    energy_threshold: float = 0.99,
) -> dict[str, np.ndarray]:
    """Standard exact DMD on a discrete-time state trajectory.

    Constructs consecutive snapshot pairs from the trajectory, computes
    the reduced DMD operator via SVD, and extracts eigenvalues, modes,
    and per-step linear prediction residuals.

    Truncation strategy (flagged for review): energy-based, retaining
    the smallest rank r such that the top-r singular values of the
    snapshot matrix account for >= energy_threshold of total energy
    (sum of squared singular values). This is more conservative than
    a spectral gap criterion — appropriate here since we're studying
    the residual, which is sensitive to rank choice.

    Args:
        trajectory: State trajectory of shape (n_steps, state_dim).
            Rows are consecutive time steps; columns are state variables.
            Must have at least 2 steps (n_steps >= 2).
        energy_threshold: Cumulative energy fraction for SVD rank truncation.
            Default 0.99. Higher retains more modes; lower is more compressed.

    Returns:
        Dict with:
            "eigenvalues":      (n_modes,) complex128 — DMD eigenvalues λ_i
            "modes":            (state_dim, n_modes) complex128 — DMD modes Φ_i
            "amplitudes":       (n_modes,) complex128 — initial amplitudes α_i
            "residual_norms":   (n_steps-1,) float64 — per-step prediction error
            "singular_values":  (min(state_dim, n_steps-1),) float64 — SVD spectrum
            "n_modes":          scalar int64 — number of retained modes
    """
    n_steps, state_dim = trajectory.shape
    # n_pairs = n_steps - 1

    X = trajectory[:-1].T  # (state_dim, n_pairs) — "from" snapshots
    Xp = trajectory[1:].T  # (state_dim, n_pairs) — "to" snapshots

    U, s, Vt = np.linalg.svd(X, full_matrices=False)  # U: (d,k), s: (k,), Vt: (k,n)

    r = _truncation_rank(s, energy_threshold)

    U_r = U[:, :r]  # (state_dim, r)
    s_r = s[:r]  # (r,)
    Vt_r = Vt[:r, :]  # (r, n_pairs)

    # Reduced DMD operator: Ã = U_r^T X' V_r Σ_r^{-1}
    A_tilde = (U_r.T @ Xp) @ Vt_r.T @ np.diag(1.0 / s_r)  # (r, r)

    eigenvalues, W = np.linalg.eig(A_tilde)  # eigenvalues: (r,), W: (r, r)

    # Exact DMD modes: Φ = X' V_r Σ_r^{-1} W
    Phi = (Xp @ Vt_r.T) @ np.diag(1.0 / s_r) @ W  # (state_dim, r)

    # Amplitudes: project initial state onto DMD modes
    amplitudes, _, _, _ = np.linalg.lstsq(Phi, trajectory[0], rcond=None)

    residual_norms = _compute_residual_norms(trajectory, Phi, eigenvalues)

    return {
        "eigenvalues": eigenvalues,
        "modes": Phi,
        "amplitudes": amplitudes,
        "residual_norms": residual_norms,
        "singular_values": s,
        "n_modes": np.array(r, dtype=np.int64),
    }


def dmd_reconstruct(
    eigenvalues: np.ndarray,
    modes: np.ndarray,
    amplitudes: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Reconstruct a DMD trajectory from modes and eigenvalues.

    Evaluates x̂_t = Re(Σ_i α_i λ_i^t φ_i) for t = 0 ... n_steps-1.

    Args:
        eigenvalues: (n_modes,) complex — DMD eigenvalues
        modes:       (state_dim, n_modes) complex — DMD modes
        amplitudes:  (n_modes,) complex — initial amplitudes
        n_steps:     Number of time steps to reconstruct

    Returns:
        Reconstructed trajectory of shape (n_steps, state_dim), real-valued.
    """
    # Power series: lambda^t for t = 0..n_steps-1
    t_idx = np.arange(n_steps)
    # eigenvalues[:, None] ** t_idx[None, :] → (n_modes, n_steps)
    lambda_powers = eigenvalues[:, np.newaxis] ** t_idx[np.newaxis, :]  # (n_modes, n_steps)
    # Weighted modes: α_i * φ_i * λ_i^t → sum over modes → (state_dim, n_steps)
    reconstruction = modes @ (amplitudes[:, np.newaxis] * lambda_powers)
    return reconstruction.real.T  # (n_steps, state_dim)


# --- Private helpers ---


def _truncation_rank(singular_values: np.ndarray, energy_threshold: float) -> int:
    """Determine SVD truncation rank by cumulative energy fraction."""
    energy = singular_values**2
    total = energy.sum()
    if total < 1e-12:
        return 1
    cumulative = np.cumsum(energy) / total
    passing = np.where(cumulative >= energy_threshold)[0]
    return int(passing[0]) + 1 if len(passing) > 0 else len(singular_values)


def _compute_residual_norms(
    trajectory: np.ndarray,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Per-step DMD prediction residuals, vectorized over all time steps.

    For each step t: residual_t = ||x_{t+1} - Re(Φ diag(λ) Φ† x_t)||

    Args:
        trajectory: (n_steps, state_dim) real-valued state trajectory
        modes:      (state_dim, n_modes) complex DMD modes
        eigenvalues:(n_modes,) complex DMD eigenvalues

    Returns:
        residual_norms: (n_steps-1,) float64
    """
    Phi_pinv = np.linalg.pinv(modes)  # (n_modes, state_dim)

    # Project all "from" states into DMD modal coordinates
    Z = Phi_pinv @ trajectory[:-1].T  # (n_modes, n_pairs)
    # Advance one step by multiplying each mode coordinate by its eigenvalue
    Z_next = eigenvalues[:, np.newaxis] * Z  # (n_modes, n_pairs)
    # Reconstruct predicted "to" states (real part)
    X_pred = (modes @ Z_next).real  # (state_dim, n_pairs)

    # Residual norms along state dimension
    diff = trajectory[1:].T - X_pred  # (state_dim, n_pairs)
    return np.linalg.norm(diff, axis=0)  # (n_pairs,)
