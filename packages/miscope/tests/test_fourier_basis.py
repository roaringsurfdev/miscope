"""Unit tests for the periodic Fourier basis primitive (REQ_109)."""

import numpy as np
import pytest

from miscope.analysis.library.fourier_basis import (
    FourierResult,
    PeriodicFourierBasis,
    SpecializationMetrics,
    compute_specialization,
    get_fourier_basis,
    project_onto_fourier_basis,
)

# ── Basis construction ───────────────────────────────────────────────


class TestGetFourierBasis:
    def test_returns_typed_basis(self):
        assert isinstance(get_fourier_basis(7), PeriodicFourierBasis)

    def test_n_frequencies_for_odd_prime(self):
        basis = get_fourier_basis(113)
        assert basis.n_frequencies == 56  # (113 - 1) // 2

    def test_basis_shapes(self):
        basis = get_fourier_basis(11)
        assert basis.cos_basis.shape == (5, 11)
        assert basis.sin_basis.shape == (5, 11)
        assert basis.frequencies.shape == (5,)

    def test_frequency_labels_start_at_one(self):
        basis = get_fourier_basis(11)
        np.testing.assert_array_equal(basis.frequencies, [1, 2, 3, 4, 5])

    def test_basis_rows_unit_normalized(self):
        basis = get_fourier_basis(13)
        cos_norms = np.linalg.norm(basis.cos_basis, axis=-1)
        sin_norms = np.linalg.norm(basis.sin_basis, axis=-1)
        np.testing.assert_allclose(cos_norms, 1.0, atol=1e-10)
        np.testing.assert_allclose(sin_norms, 1.0, atol=1e-10)

    def test_cos_sin_orthogonal_at_same_frequency(self):
        # cos(k·t) and sin(k·t) are orthogonal over a full period.
        basis = get_fourier_basis(17)
        for i in range(basis.n_frequencies):
            inner = float(basis.cos_basis[i] @ basis.sin_basis[i])
            assert abs(inner) < 1e-10

    def test_different_frequencies_orthogonal(self):
        # cos(k₁·t) and cos(k₂·t) are orthogonal for k₁ != k₂.
        basis = get_fourier_basis(19)
        gram = basis.cos_basis @ basis.cos_basis.T
        np.testing.assert_allclose(gram, np.eye(basis.n_frequencies), atol=1e-10)

    def test_rejects_period_too_small(self):
        with pytest.raises(ValueError, match="period must be >= 3"):
            get_fourier_basis(2)


# ── Projection on known signals ───────────────────────────────────────


class TestProjectOntoFourierBasis:
    def test_returns_typed_result(self):
        basis = get_fourier_basis(7)
        result = project_onto_fourier_basis(np.zeros(7), basis)
        assert isinstance(result, FourierResult)

    def test_pure_cosine_concentrates_power(self):
        # X(t) = cos(2π · k · t / N) — all power should land on frequency k
        period = 31
        k_target = 5
        basis = get_fourier_basis(period)
        t = np.arange(period)
        signal = np.cos(2 * np.pi * k_target * t / period)

        result = project_onto_fourier_basis(signal, basis)
        target_idx = k_target - 1  # frequencies are 1-indexed
        # Fractional power: ~1.0 at target, ~0 elsewhere
        assert result.fractional_power[target_idx] == pytest.approx(1.0, abs=1e-10)
        assert result.fractional_power.sum() == pytest.approx(1.0, abs=1e-10)
        # Sin coefficient at target is zero (it's a pure cosine)
        assert abs(result.sin_coeffs[target_idx]) < 1e-10

    def test_pure_sine_concentrates_power(self):
        period = 31
        k_target = 7
        basis = get_fourier_basis(period)
        t = np.arange(period)
        signal = np.sin(2 * np.pi * k_target * t / period)

        result = project_onto_fourier_basis(signal, basis)
        target_idx = k_target - 1
        assert result.fractional_power[target_idx] == pytest.approx(1.0, abs=1e-10)
        # Cos coefficient at target is zero
        assert abs(result.cos_coeffs[target_idx]) < 1e-10

    def test_dominant_frequency_label(self):
        # dominant_frequency returns the frequency LABEL, not the array index
        period = 31
        k_target = 7
        basis = get_fourier_basis(period)
        t = np.arange(period)
        signal = np.cos(2 * np.pi * k_target * t / period)
        result = project_onto_fourier_basis(signal, basis)
        assert result.dominant_frequency == k_target

    def test_phase_of_pure_cosine(self):
        # Pure cosine has phase 0 at its frequency
        period = 23
        k_target = 3
        basis = get_fourier_basis(period)
        t = np.arange(period)
        signal = np.cos(2 * np.pi * k_target * t / period)
        result = project_onto_fourier_basis(signal, basis)
        assert result.phases[k_target - 1] == pytest.approx(0.0, abs=1e-10)

    def test_phase_of_pure_sine(self):
        # Pure sine has phase π/2 at its frequency
        period = 23
        k_target = 3
        basis = get_fourier_basis(period)
        t = np.arange(period)
        signal = np.sin(2 * np.pi * k_target * t / period)
        result = project_onto_fourier_basis(signal, basis)
        assert result.phases[k_target - 1] == pytest.approx(np.pi / 2, abs=1e-10)

    def test_phase_shift(self):
        # X(t) = cos(2π · k · t / N + φ) splits between cos and sin coefficients
        # such that arctan2(sin_coeff, cos_coeff) ≈ -φ  (since phase from atan2(b,a)
        # of cos·a + sin·b form recovers shift sign).
        period = 31
        k_target = 4
        phi = np.pi / 3
        basis = get_fourier_basis(period)
        t = np.arange(period)
        signal = np.cos(2 * np.pi * k_target * t / period + phi)
        result = project_onto_fourier_basis(signal, basis)
        # cos(α + φ) = cos(α)cos(φ) − sin(α)sin(φ), so cos_coeff > 0 and sin_coeff < 0
        # for φ ∈ (0, π/2). Recovered phase is −φ.
        assert result.phases[k_target - 1] == pytest.approx(-phi, abs=1e-10)

    def test_zero_input_zero_power(self):
        period = 11
        basis = get_fourier_basis(period)
        result = project_onto_fourier_basis(np.zeros(period), basis)
        np.testing.assert_array_equal(result.power, np.zeros(basis.n_frequencies))
        # fractional_power: zero divided by zero falls back to zeros
        np.testing.assert_array_equal(result.fractional_power, np.zeros(basis.n_frequencies))

    def test_dc_signal_zero_power_in_non_trivial_frequencies(self):
        # Constant signal has all power in DC, which is excluded from the basis
        period = 13
        basis = get_fourier_basis(period)
        signal = np.full(period, 7.0)
        result = project_onto_fourier_basis(signal, basis)
        np.testing.assert_allclose(result.power, 0.0, atol=1e-10)

    def test_rejects_period_mismatch(self):
        basis = get_fourier_basis(11)
        with pytest.raises(ValueError, match="!= basis.period"):
            project_onto_fourier_basis(np.zeros(13), basis)


# ── Multi-unit projection (axis handling) ────────────────────────────


class TestProjectMultiUnit:
    def test_per_unit_independence(self):
        # Two neurons, two different target frequencies — projection runs
        # independently per neuron
        period = 31
        basis = get_fourier_basis(period)
        t = np.arange(period)
        X = np.stack(
            [
                np.cos(2 * np.pi * 4 * t / period),
                np.sin(2 * np.pi * 9 * t / period),
            ]
        )  # shape (2, period)
        result = project_onto_fourier_basis(X, basis, period_axis=-1)

        assert result.dominant_frequency.shape == (2,)
        assert result.dominant_frequency[0] == 4
        assert result.dominant_frequency[1] == 9

    def test_output_shape_period_axis_last(self):
        period = 31
        basis = get_fourier_basis(period)
        X = np.zeros((5, 8, period))
        result = project_onto_fourier_basis(X, basis, period_axis=-1)
        assert result.cos_coeffs.shape == (5, 8, basis.n_frequencies)
        assert result.dominant_frequency.shape == (5, 8)

    def test_output_shape_period_axis_first(self):
        period = 31
        basis = get_fourier_basis(period)
        X = np.zeros((period, 8, 5))
        result = project_onto_fourier_basis(X, basis, period_axis=0)
        # Frequency axis replaces period axis at the same position
        assert result.cos_coeffs.shape == (basis.n_frequencies, 8, 5)
        assert result.dominant_frequency.shape == (8, 5)

    def test_period_axis_middle(self):
        period = 17
        basis = get_fourier_basis(period)
        X = np.zeros((4, period, 3))
        result = project_onto_fourier_basis(X, basis, period_axis=1)
        assert result.cos_coeffs.shape == (4, basis.n_frequencies, 3)

    def test_fractional_power_normalized_per_unit(self):
        period = 31
        basis = get_fourier_basis(period)
        t = np.arange(period)
        rng = np.random.default_rng(0)
        # Three neurons with random Fourier content
        X = np.stack(
            [
                rng.normal() * np.cos(2 * np.pi * k * t / period)
                + rng.normal() * np.sin(2 * np.pi * k * t / period)
                for k in (3, 5, 7)
            ]
        )
        result = project_onto_fourier_basis(X, basis, period_axis=-1)
        per_unit_sum = result.fractional_power.sum(axis=-1)
        np.testing.assert_allclose(per_unit_sum, 1.0, atol=1e-10)


# ── Specialization summary metrics ───────────────────────────────────


class TestComputeSpecialization:
    def test_returns_typed_result(self):
        # 3 units across 5 frequencies, no specialization
        fp = np.full((3, 5), 0.2)
        metrics = compute_specialization(fp)
        assert isinstance(metrics, SpecializationMetrics)

    def test_records_threshold_and_n_frequencies(self):
        fp = np.full((4, 7), 1 / 7)
        metrics = compute_specialization(fp, threshold=0.85)
        assert metrics.threshold == 0.85
        assert metrics.n_frequencies == 7

    def test_per_unit_max_and_dominant(self):
        # Three units, each fully specialized to a known bin
        fp = np.zeros((3, 5))
        fp[0, 1] = 1.0
        fp[1, 4] = 1.0
        fp[2, 0] = 1.0
        metrics = compute_specialization(fp, threshold=0.9)
        np.testing.assert_array_equal(metrics.max_fractional_power, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(metrics.dominant_frequency_idx, [1, 4, 0])
        np.testing.assert_array_equal(metrics.specialized_mask, [True, True, True])

    def test_threshold_filters_specialized_units(self):
        # Two specialized (max=1.0), one not specialized (max=0.5)
        fp = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0],
            ]
        )
        metrics = compute_specialization(fp, threshold=0.9)
        assert metrics.specialized_count_total == 2
        np.testing.assert_array_equal(metrics.specialized_mask, [True, True, False])

    def test_specialized_count_per_frequency(self):
        # Five units; dominants on bins [0, 0, 2, 2, 4], all above threshold
        fp = np.zeros((5, 5))
        fp[0, 0] = 1.0
        fp[1, 0] = 1.0
        fp[2, 2] = 1.0
        fp[3, 2] = 1.0
        fp[4, 4] = 1.0
        metrics = compute_specialization(fp, threshold=0.9)
        np.testing.assert_array_equal(metrics.specialized_count_per_frequency, [2, 0, 2, 0, 1])
        assert metrics.specialized_count_total == 5

    def test_unspecialized_units_excluded_from_per_frequency_count(self):
        # Two units have dominant bin 1 but only one passes threshold
        fp = np.array(
            [
                [0.0, 0.95, 0.05, 0.0, 0.0],  # specialized at bin 1
                [0.4, 0.4, 0.2, 0.0, 0.0],  # dominant at bin 0 (or 1; argmax → 0), not specialized
            ]
        )
        metrics = compute_specialization(fp, threshold=0.9)
        np.testing.assert_array_equal(metrics.specialized_count_per_frequency, [0, 1, 0, 0, 0])
        assert metrics.specialized_count_total == 1

    def test_mean_and_median_max(self):
        fp = np.array(
            [
                [0.1, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.7, 0.1, 0.1, 0.1],
            ]
        )
        metrics = compute_specialization(fp, threshold=0.9)
        # max per unit: [0.9, 0.5, 0.7] -> mean=0.7, median=0.7
        assert metrics.mean_max_fractional_power == pytest.approx(0.7, abs=1e-10)
        assert metrics.median_max_fractional_power == pytest.approx(0.7, abs=1e-10)

    def test_frequency_axis_first(self):
        # Shape (n_freq=5, n_units=3). frequency_axis=0.
        fp = np.zeros((5, 3))
        fp[1, 0] = 1.0
        fp[4, 1] = 1.0
        fp[2, 2] = 1.0
        metrics = compute_specialization(fp, threshold=0.9, frequency_axis=0)
        # Per-unit shapes drop the frequency axis: shape (3,)
        assert metrics.max_fractional_power.shape == (3,)
        np.testing.assert_array_equal(metrics.dominant_frequency_idx, [1, 4, 2])
        np.testing.assert_array_equal(metrics.specialized_count_per_frequency, [0, 1, 1, 0, 1])

    def test_no_specialized_units_returns_zero_counts(self):
        # All units distribute power evenly — none crosses threshold
        fp = np.full((4, 5), 0.2)
        metrics = compute_specialization(fp, threshold=0.9)
        assert metrics.specialized_count_total == 0
        np.testing.assert_array_equal(
            metrics.specialized_count_per_frequency, np.zeros(5, dtype=int)
        )

    def test_composes_with_project_onto_fourier_basis(self):
        # End-to-end: pure-frequency signals → project → specialization
        period = 31
        basis = get_fourier_basis(period)
        t = np.arange(period)
        X = np.stack(
            [
                np.cos(2 * np.pi * 4 * t / period),
                np.cos(2 * np.pi * 4 * t / period),
                np.sin(2 * np.pi * 9 * t / period),
            ]
        )
        result = project_onto_fourier_basis(X, basis, period_axis=-1)
        metrics = compute_specialization(result.fractional_power, threshold=0.9)
        # Pure-frequency signals concentrate all power on one bin → all specialized
        assert metrics.specialized_count_total == 3
        # Two units land on bin 3 (freq 4), one lands on bin 8 (freq 9)
        assert metrics.specialized_count_per_frequency[3] == 2
        assert metrics.specialized_count_per_frequency[8] == 1
