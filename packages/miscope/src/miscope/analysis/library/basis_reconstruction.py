"""Reconstruct the legacy ``neuron_freq_norm`` matrix from basis projections.

REQ_131: the consumers ``neuron_dynamics``, ``neuron_group_pca``, and
``freq_group_weight_geometry`` were re-pointed off the specialized
``neuron_freq_norm`` artifact (produced by the retired ``neuron_freq_clusters``)
and onto the generic ``activation_basis_projection``. They all read a single
``norm_matrix`` ``(n_freq, n_units)`` and use it identically — ``argmax``/``max``
over the frequency axis. This helper rebuilds that matrix from the generic
analyzer's ``mlp_out`` site outputs, so the re-point is behavior-preserving.

The reconstruction is bitwise-validated against the legacy artifact at parity
tolerance (``rtol=1e-3``) in ``test_activation_basis_projection.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# The activation_basis_projection fields needed to reconstruct norm_matrix for
# the mlp_out site. Shared by all REQ_131 consumers so the selective-loading
# contract stays in one place.
NEURON_FREQ_NORM_FIELDS: list[str] = [
    "mlp_out_power",
    "mlp_out_axis_a_marginal_power",
    "mlp_out_axis_b_marginal_power",
]


def reconstruct_neuron_freq_norm(
    projection: Mapping[str, np.ndarray],
    prime: int,
    site_prefix: str = "mlp_out",
) -> np.ndarray:
    """Rebuild the legacy per-frequency variance fraction matrix.

    Legacy formula (sum of the 8 cells of the ``(k+1)``-th 3×3 cross of the 2D
    Fourier transform, divided by total non-DC Fourier variance ≡ total signal
    variance by Parseval)::

        numerator_k[n] = power_joint[n, k, k]
                       + prime * axis_a_marginal_power[n, k]
                       + prime * axis_b_marginal_power[n, k]
        norm_matrix[k, n] = numerator_k[n] / total_non_dc_variance[n]

    Args:
        projection: an ``activation_basis_projection`` epoch dict containing the
            ``{site_prefix}_power`` joint cube and the two per-axis marginal
            power arrays (see :data:`NEURON_FREQ_NORM_FIELDS`).
        prime: the family prime (marginal energy is scaled by ``prime`` to match
            the legacy mix of 2D-diagonal and per-axis-marginal energy).
        site_prefix: activation site name (default ``"mlp_out"``).

    Returns:
        ``(n_freq, n_units)`` array matching the legacy ``norm_matrix`` layout.
    """
    joint_power = projection[f"{site_prefix}_power"]  # (n_units, K, K)
    marginal_a_power = projection[f"{site_prefix}_axis_a_marginal_power"]  # (n_units, K)
    marginal_b_power = projection[f"{site_prefix}_axis_b_marginal_power"]  # (n_units, K)

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
