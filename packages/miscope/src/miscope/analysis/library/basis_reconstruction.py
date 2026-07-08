"""Reduce a 2D activation basis-projection to the per-frequency energy norm.

This is the single reduction the ``activation_frequency_norm`` analyzer applies
to each site's joint Fourier projection before persisting. It descends from the
retired ``neuron_freq_clusters`` / ``attention_freq`` analyzers: it rebuilds the
legacy ``norm_matrix`` ``(n_freq, n_units)`` — the per-(unit, frequency) fraction
of non-DC Fourier variance — which is bitwise-validated against those legacy
artifacts at parity tolerance (``rtol=1e-3``) in
``test_activation_frequency_norm.py``.

Historically this reduction was duplicated across four consumers (three analyzers
plus the view adapter), each re-loading the heavy ``(n_units, K, K)`` joint cube
and the two marginal-power arrays. ``activation_frequency_norm`` now performs the
reduction once, in-memory, and persists only the small ``(n_freq, n_units)``
result — so this helper takes arrays directly rather than an on-disk field dict.
"""

from __future__ import annotations

import numpy as np


def reduce_to_freq_norm(
    joint_power: np.ndarray,
    marginal_a_power: np.ndarray,
    marginal_b_power: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Rebuild the legacy per-frequency variance-fraction matrix.

    Legacy formula (sum of the 8 cells of the ``(k+1)``-th 3×3 cross of the 2D
    Fourier transform, divided by total non-DC Fourier variance ≡ total signal
    variance by Parseval)::

        numerator_k[n] = joint_power[n, k, k]
                       + prime * marginal_a_power[n, k]
                       + prime * marginal_b_power[n, k]
        freq_norm[k, n] = numerator_k[n] / total_non_dc_variance[n]

    The ``prime`` factor restores the per-axis marginal energy to the same scale
    as the 2D diagonal: the marginals are projections of a mean-over-the-other-
    axis, which divides energy by ``prime``.

    Args:
        joint_power: ``(n_units, K, K)`` joint 2D power cube.
        marginal_a_power: ``(n_units, K)`` per-axis-a marginal power.
        marginal_b_power: ``(n_units, K)`` per-axis-b marginal power.
        prime: the family prime (marginal energy scale factor).

    Returns:
        ``(n_freq, n_units)`` array matching the legacy ``norm_matrix`` layout.
    """
    k = joint_power.shape[-1]
    diag_joint = joint_power[..., np.arange(k), np.arange(k)]  # (n_units, K)
    numerator = diag_joint + prime * (marginal_a_power + marginal_b_power)

    total_non_dc = (
        joint_power.sum(axis=(-1, -2))
        + prime * marginal_a_power.sum(axis=-1)
        + prime * marginal_b_power.sum(axis=-1)
    )
    total_clipped = np.maximum(total_non_dc, 1e-10)
    reconstructed = numerator / total_clipped[..., None]  # (n_units, K)
    return reconstructed.T  # (n_freq, n_units) to match legacy norm_matrix layout
