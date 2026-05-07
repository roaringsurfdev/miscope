"""Unit tests for trajectory comparison primitives in library/comparison.py
(REQ_109 phase 2d)."""

import numpy as np
import pytest
from scipy.spatial import procrustes as scipy_procrustes

from miscope.analysis.library.comparison import (
    ProcrustesResult,
    compute_procrustes_disparity_matrix,
    procrustes_align,
)


def _arc(n: int = 50, scale: float = 1.0) -> np.ndarray:
    """Quarter-arc trajectory, n points, no shared origin with the unit circle."""
    theta = np.linspace(0.0, np.pi / 2, n)
    return scale * np.column_stack([np.cos(theta), np.sin(theta)])


def _rotation_2d(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


class TestProcrustesAlign:
    def test_returns_typed_result(self):
        a = _arc()
        b = _arc()
        result = procrustes_align(a, b)
        assert isinstance(result, ProcrustesResult)
        assert isinstance(result.disparity, float)
        assert result.standardized_a.shape == a.shape
        assert result.aligned_b.shape == b.shape
        assert result.n_points == a.shape[0]
        assert result.n_features == a.shape[1]

    def test_identical_inputs_zero_disparity(self):
        a = _arc()
        result = procrustes_align(a, a)
        assert result.disparity == pytest.approx(0.0, abs=1e-12)

    def test_translated_copy_zero_disparity(self):
        a = _arc()
        b = a + np.array([5.0, -3.0])
        result = procrustes_align(a, b)
        assert result.disparity == pytest.approx(0.0, abs=1e-12)

    def test_uniformly_scaled_copy_zero_disparity(self):
        a = _arc()
        b = 7.5 * a
        result = procrustes_align(a, b)
        assert result.disparity == pytest.approx(0.0, abs=1e-12)

    def test_rotated_copy_zero_disparity(self):
        a = _arc()
        b = a @ _rotation_2d(0.7).T
        result = procrustes_align(a, b)
        assert result.disparity == pytest.approx(0.0, abs=1e-12)

    def test_reflected_copy_zero_disparity(self):
        a = _arc()
        # Reflect across x-axis
        b = a * np.array([1.0, -1.0])
        result = procrustes_align(a, b)
        assert result.disparity == pytest.approx(0.0, abs=1e-12)

    def test_unrelated_arrays_have_positive_disparity(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(40, 2))
        b = rng.normal(size=(40, 2))
        result = procrustes_align(a, b)
        assert result.disparity > 0.0
        assert result.disparity <= 1.0

    def test_standardized_a_is_zero_mean(self):
        a = _arc() + np.array([10.0, -2.0])
        result = procrustes_align(a, a)
        np.testing.assert_allclose(result.standardized_a.mean(axis=0), [0.0, 0.0], atol=1e-12)

    def test_standardized_a_has_unit_frobenius(self):
        a = _arc(scale=42.0)
        result = procrustes_align(a, a)
        frobenius = np.linalg.norm(result.standardized_a)
        assert frobenius == pytest.approx(1.0, abs=1e-12)

    def test_bit_exact_match_with_scipy(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(30, 3))
        b = rng.normal(size=(30, 3))
        scipy_a, scipy_b, scipy_disp = scipy_procrustes(a, b)
        result = procrustes_align(a, b)
        assert result.disparity == scipy_disp
        np.testing.assert_array_equal(result.standardized_a, scipy_a)
        np.testing.assert_array_equal(result.aligned_b, scipy_b)

    def test_shape_mismatch_raises(self):
        a = np.zeros((10, 2))
        b = np.zeros((10, 3))
        with pytest.raises(ValueError, match="matching shapes"):
            procrustes_align(a, b)

    def test_wrong_ndim_raises(self):
        a = np.zeros(10)
        b = np.zeros(10)
        with pytest.raises(ValueError, match="2D arrays"):
            procrustes_align(a, b)

    def test_accepts_float_coercion(self):
        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
        b = np.array([[2, 4], [6, 8], [10, 12]], dtype=int)
        result = procrustes_align(a, b)
        assert result.disparity == pytest.approx(0.0, abs=1e-12)


class TestProcrustesDisparityMatrix:
    def test_shape_and_diagonal(self):
        rng = np.random.default_rng(2)
        trajectories = [rng.normal(size=(20, 2)) for _ in range(4)]
        matrix = compute_procrustes_disparity_matrix(trajectories)
        assert matrix.shape == (4, 4)
        np.testing.assert_array_equal(np.diag(matrix), np.zeros(4))

    def test_symmetric(self):
        rng = np.random.default_rng(3)
        trajectories = [rng.normal(size=(20, 2)) for _ in range(5)]
        matrix = compute_procrustes_disparity_matrix(trajectories)
        np.testing.assert_array_equal(matrix, matrix.T)

    def test_consistency_with_pairwise_align(self):
        rng = np.random.default_rng(4)
        trajectories = [rng.normal(size=(15, 2)) for _ in range(3)]
        matrix = compute_procrustes_disparity_matrix(trajectories)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                expected = procrustes_align(trajectories[i], trajectories[j]).disparity
                assert matrix[i, j] == pytest.approx(expected, abs=1e-15)

    def test_degenerate_input_writes_nan(self):
        good = _arc()
        # Single repeated row: scipy rejects ("must contain >1 unique points").
        degenerate = np.tile([1.0, 1.0], (good.shape[0], 1))
        matrix = compute_procrustes_disparity_matrix([good, degenerate])
        assert np.isnan(matrix[0, 1])
        assert np.isnan(matrix[1, 0])
        assert matrix[0, 0] == 0.0
        assert matrix[1, 1] == 0.0

    def test_empty_list_returns_zero_zero_matrix(self):
        matrix = compute_procrustes_disparity_matrix([])
        assert matrix.shape == (0, 0)
