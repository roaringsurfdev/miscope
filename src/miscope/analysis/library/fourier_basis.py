"""Periodic Fourier basis primitive.

Pure-input Fourier decomposition for signals of fixed period N. Mirrors the
PCA pattern: ``get_fourier_basis(period)`` returns a typed
:class:`PeriodicFourierBasis` (precomputed, reusable across many projections);
``project_onto_fourier_basis(X, basis)`` returns a typed
:class:`FourierResult` carrying coefficients, magnitudes, phases, power,
fractional power, and dominant frequency.

Per the REQ_106 layering rule, the caller is responsible for composing the
right matrix to project (e.g. extracting ``W_E`` rows along the token axis,
or assembling a 2D Lissajous trajectory). The primitive doesn't know about
``Variant``, weight matrices, or domain-specific compositions.

The basis covers only the *non-trivial* frequencies ``k = 1, ..., (N-1)//2``.
The DC component (mean) is excluded from the basis; callers compute it
directly as ``X.mean(axis=period_axis)`` if they need it. This keeps the
cosine and sine bases symmetric in shape — the DC sine is zero everywhere
and would otherwise be a fictitious basis row.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicFourierBasis:
    """Precomputed Fourier basis for signals of fixed period.

    Built once via :func:`get_fourier_basis`, then reused across many
    projections via :func:`project_onto_fourier_basis`.

    Attributes:
        period: Length of one full period of the signal (e.g. the prime
            ``p`` in modular arithmetic).
        n_frequencies: Number of non-trivial frequencies retained,
            ``(period - 1) // 2``.
        cos_basis: ``(n_frequencies, period)`` unit-normalized cosine
            basis vectors. Row ``i`` is :math:`\\cos(2\\pi k_i t / N)`
            for ``t = 0, ..., N-1``.
        sin_basis: ``(n_frequencies, period)`` unit-normalized sine basis
            vectors. Row ``i`` is :math:`\\sin(2\\pi k_i t / N)`.
        frequencies: ``(n_frequencies,)`` integer frequency labels
            ``[1, 2, ..., n_frequencies]``.
    """

    period: int
    n_frequencies: int
    cos_basis: np.ndarray
    sin_basis: np.ndarray
    frequencies: np.ndarray


@dataclass(frozen=True)
class FourierResult:
    """Result of projecting an array onto a periodic Fourier basis.

    All "frequency-side" arrays carry the frequency axis at the same
    position the input's ``period_axis`` was — frequency replaces period
    in the output shape. All other axes are preserved as "unit" axes.

    Attributes:
        cos_coeffs: signed projection onto each cosine basis vector.
        sin_coeffs: signed projection onto each sine basis vector.
        magnitudes: ``sqrt(cos² + sin²)`` — per-unit, per-frequency amplitude.
        phases: ``arctan2(sin, cos)`` — per-unit, per-frequency phase
            in ``[-π, π]``.
        power: ``cos² + sin²`` — per-unit, per-frequency power.
        fractional_power: power normalized across frequencies per unit.
            Each unit's row sums to 1 (or 0 if total power is zero).
        dominant_frequency: integer frequency label of the highest-magnitude
            frequency per unit. Shape is the input's shape with
            ``period_axis`` removed.
        period_axis: which axis of the input ``X`` was the period axis.
            The output's frequency axis is at the same position.
    """

    cos_coeffs: np.ndarray
    sin_coeffs: np.ndarray
    magnitudes: np.ndarray
    phases: np.ndarray
    power: np.ndarray
    fractional_power: np.ndarray
    dominant_frequency: np.ndarray
    period_axis: int


def get_fourier_basis(period: int) -> PeriodicFourierBasis:
    """Build a periodic Fourier basis for signals of length ``period``.

    The basis covers non-trivial frequencies ``k = 1, ..., (period-1)//2``.
    For odd ``period`` (e.g. an odd prime), this is a complete spanning
    set for zero-mean periodic signals. The DC component is *not* in the
    basis; the caller computes ``X.mean(axis=period_axis)`` separately if
    they need it.

    Each basis vector is unit-normalized so that projections are the inner
    products (caller's signal magnitudes are preserved up to the ``1/√N``
    normalization implicit in the unit basis).

    Args:
        period: Signal period (must be ``>= 3`` for at least one
            non-trivial frequency).

    Returns:
        :class:`PeriodicFourierBasis`.
    """
    if period < 3:
        raise ValueError(
            f"period must be >= 3 to have at least one non-trivial frequency; got {period}"
        )

    n_frequencies = (period - 1) // 2
    frequencies = np.arange(1, n_frequencies + 1)

    t = np.arange(period)
    angles = 2.0 * np.pi * frequencies[:, np.newaxis] * t[np.newaxis, :] / period

    cos_basis = np.cos(angles)
    sin_basis = np.sin(angles)

    cos_basis = cos_basis / np.linalg.norm(cos_basis, axis=-1, keepdims=True)
    sin_basis = sin_basis / np.linalg.norm(sin_basis, axis=-1, keepdims=True)

    return PeriodicFourierBasis(
        period=period,
        n_frequencies=n_frequencies,
        cos_basis=cos_basis,
        sin_basis=sin_basis,
        frequencies=frequencies,
    )


def project_onto_fourier_basis(
    X: np.ndarray,
    basis: PeriodicFourierBasis,
    period_axis: int = -1,
) -> FourierResult:
    """Project an array onto a periodic Fourier basis.

    The axis named by ``period_axis`` must have length ``basis.period``.
    All other axes are treated as independent "unit" axes — projection is
    computed independently per unit. The frequency axis in the output
    occupies the same position as ``period_axis`` in the input.

    Args:
        X: Array containing periodic signals along ``period_axis``.
        basis: Pre-computed :class:`PeriodicFourierBasis`.
        period_axis: Axis of ``X`` with length ``basis.period``. Default
            ``-1`` (last axis).

    Returns:
        :class:`FourierResult` with per-unit, per-frequency coefficients
        and derived metrics.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[period_axis] != basis.period:
        raise ValueError(
            f"X.shape[{period_axis}]={X.shape[period_axis]} != basis.period={basis.period}"
        )

    # Move the period axis to the last position for matmul, then move the
    # resulting frequency axis back to where the period axis was.
    X_pa = np.moveaxis(X, period_axis, -1)  # (..., period)
    cos_coeffs_pa = X_pa @ basis.cos_basis.T  # (..., n_frequencies)
    sin_coeffs_pa = X_pa @ basis.sin_basis.T

    cos_coeffs = np.moveaxis(cos_coeffs_pa, -1, period_axis)
    sin_coeffs = np.moveaxis(sin_coeffs_pa, -1, period_axis)

    magnitudes = np.sqrt(cos_coeffs**2 + sin_coeffs**2)
    phases = np.arctan2(sin_coeffs, cos_coeffs)
    power = cos_coeffs**2 + sin_coeffs**2

    total_power = power.sum(axis=period_axis, keepdims=True)
    fractional_power = np.where(
        total_power > 0,
        power / np.maximum(total_power, 1e-12),
        np.zeros_like(power),
    )

    dominant_idx = np.argmax(magnitudes, axis=period_axis)
    dominant_frequency = basis.frequencies[dominant_idx]

    return FourierResult(
        cos_coeffs=cos_coeffs,
        sin_coeffs=sin_coeffs,
        magnitudes=magnitudes,
        phases=phases,
        power=power,
        fractional_power=fractional_power,
        dominant_frequency=dominant_frequency,
        period_axis=period_axis,
    )
