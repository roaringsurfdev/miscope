"""Unit tests for PCA primitives (REQ_109 phase 1a)."""

import numpy as np
import pytest

from miscope.analysis.library.pca import pca, pca_rolling, pca_summary
from miscope.core.pca import PCAResult


class TestPCABasic:
    def test_returns_pca_result(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        result = pca(X)
        assert isinstance(result, PCAResult)

    def test_centers_data(self):
        X = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        result = pca(X)
        np.testing.assert_allclose(result.center, [4.0, 5.0])

    def test_first_pc_aligns_with_dominant_axis(self):
        rng = np.random.default_rng(0)
        # Strong x-axis variance, weak y-axis variance
        X = np.column_stack([rng.normal(scale=5.0, size=100), rng.normal(scale=0.1, size=100)])
        result = pca(X)
        # First basis vector should be approximately ±[1, 0]
        assert abs(abs(result.basis_vectors[0, 0]) - 1.0) < 0.05
        assert abs(result.basis_vectors[0, 1]) < 0.05

    def test_eigenvalues_descending(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 5))
        result = pca(X)
        diffs = np.diff(result.eigenvalues)
        assert (diffs <= 1e-12).all()

    def test_explained_variance_ratio_sums_to_one_when_all_kept(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 5))
        result = pca(X)  # all 5 components kept
        np.testing.assert_allclose(result.explained_variance_ratio.sum(), 1.0)

    def test_explained_variance_ratio_full_spectrum_normalization(self):
        # Truncated ratio represents fraction of TOTAL variance (sklearn convention)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 8))
        full = pca(X)
        truncated = pca(X, n_components=3)
        np.testing.assert_allclose(
            truncated.explained_variance_ratio,
            full.explained_variance_ratio[:3],
        )
        # Truncated ratio sums to less than 1
        assert truncated.explained_variance_ratio.sum() < 1.0
        # And matches the cumulative explained variance at k=3
        np.testing.assert_allclose(
            truncated.explained_variance_ratio.sum(),
            full.explained_variance_ratio[:3].sum(),
        )

    def test_explained_variance_alias(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        result = pca(X)
        np.testing.assert_array_equal(result.explained_variance, result.eigenvalues)

    def test_n_components_truncation(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 8))
        result = pca(X, n_components=3)
        assert result.basis_vectors.shape == (3, 8)
        assert result.projections.shape == (20, 3)
        assert result.eigenvalues.shape == (3,)
        assert result.singular_values.shape == (3,)

    def test_n_components_none_takes_min(self):
        # 10 samples × 4 features → max 4 components
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 4))
        result = pca(X)
        assert result.eigenvalues.shape == (4,)

    def test_basis_vectors_orthonormal(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 5))
        result = pca(X)
        gram = result.basis_vectors @ result.basis_vectors.T
        np.testing.assert_allclose(gram, np.eye(5), atol=1e-10)

    def test_projections_match_centered_data_dot_basis(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(15, 4))
        result = pca(X, n_components=4)
        expected = (X - result.center) @ result.basis_vectors.T
        # SVD sign convention may flip individual columns; compare absolute values
        np.testing.assert_allclose(np.abs(result.projections), np.abs(expected), atol=1e-10)


class TestPCAMetrics:
    def test_participation_ratio_one_dominant_direction(self):
        # All variance along x-axis
        X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        result = pca(X)
        assert result.participation_ratio == pytest.approx(1.0, abs=1e-10)

    def test_participation_ratio_isotropic(self):
        # Three orthogonal directions with equal variance → PR = 3
        X = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        result = pca(X)
        assert result.participation_ratio == pytest.approx(3.0, abs=1e-10)

    def test_rank_one_data(self):
        # All points on the line y = 2x — rank 1 after centering
        X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
        result = pca(X)
        assert result.rank == 1

    def test_rank_full(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        result = pca(X)
        assert result.rank == 4

    def test_spread_relates_to_eigenvalues(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 4)) * 2.0
        result = pca(X)
        np.testing.assert_allclose(result.spread, np.sqrt(result.eigenvalues.sum()))

    def test_full_spectrum_scalars_invariant_to_truncation(self):
        # PR, rank, spread describe the data; truncation should not change them
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 6))
        full = pca(X)
        truncated = pca(X, n_components=2)
        assert full.participation_ratio == pytest.approx(truncated.participation_ratio)
        assert full.rank == truncated.rank
        assert full.spread == pytest.approx(truncated.spread)


class TestPCAFloat32Stability:
    """Bit-identical float32 input must produce exact-zero singular values.

    Regression for the dtype bug in REQ_109 phase 1b: float32 ``X.mean``
    + subtraction produced spurious noise that propagated to singular
    values and eigenvalues for structurally-degenerate sites.
    """

    def test_zero_singular_values_for_identical_float32(self):
        rng = np.random.default_rng(0)
        x = (rng.normal(size=8) * 100).astype(np.float32)
        X = np.tile(x, (20, 1))  # all rows identical, float32
        result = pca(X)
        np.testing.assert_array_equal(result.singular_values, np.zeros_like(result.singular_values))
        assert result.spread == 0.0
        assert result.participation_ratio == 0.0


class TestPCAErrors:
    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D input"):
            pca(np.array([1.0, 2.0, 3.0]))

    def test_rejects_excessive_n_components(self):
        with pytest.raises(ValueError, match="exceeds"):
            pca(np.array([[1.0, 2.0]]), n_components=5)


class TestPCASummary:
    def test_3d_input_equivalent_to_flattened(self):
        rng = np.random.default_rng(0)
        sets = rng.normal(size=(3, 10, 4))
        flat = sets.reshape(-1, 4)
        r3d = pca_summary(sets)
        r2d = pca_summary(flat)
        np.testing.assert_allclose(r3d.eigenvalues, r2d.eigenvalues)
        np.testing.assert_allclose(np.abs(r3d.basis_vectors), np.abs(r2d.basis_vectors))

    def test_list_input_concatenates(self):
        rng = np.random.default_rng(0)
        sets = [
            rng.normal(size=(5, 3)),
            rng.normal(size=(7, 3)),
            rng.normal(size=(4, 3)),
        ]
        result = pca_summary(sets)
        assert result.projections.shape == (16, 3)

    def test_shared_basis_across_sets(self):
        # Two sets with identical statistics should produce the same basis
        # whether fed separately or together
        rng = np.random.default_rng(0)
        a = rng.normal(size=(20, 3))
        b = rng.normal(size=(20, 3))
        joint = pca_summary([a, b])
        # Projections of joint correspond to concat([a, b]) projected onto joint basis
        expected = np.concatenate([a, b], axis=0) - joint.center
        expected_proj = expected @ joint.basis_vectors.T
        np.testing.assert_allclose(np.abs(joint.projections), np.abs(expected_proj), atol=1e-10)

    def test_feature_dim_mismatch(self):
        sets = [np.zeros((3, 4)), np.zeros((3, 5))]
        with pytest.raises(ValueError, match="feature dimension"):
            pca_summary(sets)

    def test_empty_input(self):
        with pytest.raises(ValueError, match="at least one"):
            pca_summary([])

    def test_invalid_array_dim(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            pca_summary(np.zeros((3, 4, 5, 6)))


class TestPCARolling:
    def test_window_count_stride_one(self):
        X = np.zeros((10, 3))
        results = pca_rolling(X, window_size=3, stride=1)
        # windows start at 0, 1, ..., 7 → 8 windows
        assert len(results) == 8

    def test_window_count_stride_two(self):
        X = np.zeros((10, 3))
        results = pca_rolling(X, window_size=3, stride=2)
        # windows start at 0, 2, 4, 6 → 4 windows
        assert len(results) == 4

    def test_each_result_is_pca_result(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 5))
        results = pca_rolling(X, window_size=5, stride=2)
        assert all(isinstance(r, PCAResult) for r in results)

    def test_window_size_exceeds_samples(self):
        X = np.zeros((5, 3))
        with pytest.raises(ValueError, match="exceeds"):
            pca_rolling(X, window_size=10)

    def test_invalid_window_size(self):
        X = np.zeros((5, 3))
        with pytest.raises(ValueError, match="positive"):
            pca_rolling(X, window_size=0)

    def test_invalid_stride(self):
        X = np.zeros((5, 3))
        with pytest.raises(ValueError, match="positive"):
            pca_rolling(X, window_size=3, stride=0)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D input"):
            pca_rolling(np.array([1.0, 2.0, 3.0]), window_size=2)
