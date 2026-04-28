"""Unit tests for clustering primitives (REQ_109 phase 1a)."""

import numpy as np
import pytest

from miscope.analysis.library.clustering import (
    FisherDiscriminant,
    compute_center_spread,
    compute_class_centroids,
    compute_class_dimensionality,
    compute_class_radii,
    compute_fisher_discriminant,
)


class TestComputeClassCentroids:
    def test_simple_two_class(self):
        samples = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        labels = np.array([0, 0, 1, 1])
        centroids = compute_class_centroids(samples, labels)
        np.testing.assert_allclose(centroids[0], [1.0, 0.0])
        np.testing.assert_allclose(centroids[1], [1.0, 2.0])

    def test_n_classes_inferred(self):
        samples = np.zeros((6, 3))
        labels = np.array([0, 1, 2, 0, 1, 2])
        centroids = compute_class_centroids(samples, labels)
        assert centroids.shape == (3, 3)

    def test_n_classes_explicit_with_empty_class(self):
        # Class 2 has no samples; centroid should be zero, not error
        samples = np.array([[1.0, 0.0], [3.0, 0.0]])
        labels = np.array([0, 1])
        centroids = compute_class_centroids(samples, labels, n_classes=3)
        assert centroids.shape == (3, 2)
        np.testing.assert_allclose(centroids[2], [0.0, 0.0])

    def test_rejects_1d_samples(self):
        with pytest.raises(ValueError, match="2D"):
            compute_class_centroids(np.zeros(5), np.zeros(5, dtype=int))

    def test_rejects_label_shape_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            compute_class_centroids(np.zeros((5, 3)), np.zeros(4, dtype=int))


class TestComputeClassRadii:
    def test_zero_radii_for_collapsed_classes(self):
        samples = np.array([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]])
        labels = np.array([0, 0, 1, 1])
        centroids = compute_class_centroids(samples, labels)
        radii = compute_class_radii(samples, labels, centroids)
        np.testing.assert_allclose(radii, [0.0, 0.0])

    def test_known_radius(self):
        # Two samples per class, each at distance 1 from the centroid
        samples = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [0.0, 6.0]])
        labels = np.array([0, 0, 1, 1])
        centroids = np.array([[1.0, 0.0], [0.0, 5.0]])
        radii = compute_class_radii(samples, labels, centroids)
        np.testing.assert_allclose(radii, [1.0, 1.0])


class TestComputeFisherDiscriminant:
    def test_returns_fisher_discriminant(self):
        samples = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 0.0], [10.5, 0.0]])
        labels = np.array([0, 0, 1, 1])
        result = compute_fisher_discriminant(samples, labels)
        assert isinstance(result, FisherDiscriminant)
        assert result.mean > 0
        assert result.min > 0

    def test_separable_classes_have_high_fisher(self):
        # Tight clusters at 0 and 100 → high Fisher
        rng = np.random.default_rng(0)
        a = rng.normal(scale=0.1, size=(20, 2))
        b = rng.normal(scale=0.1, size=(20, 2)) + np.array([100.0, 0.0])
        samples = np.vstack([a, b])
        labels = np.array([0] * 20 + [1] * 20)
        result = compute_fisher_discriminant(samples, labels)
        assert result.mean > 1000  # well-separated

    def test_overlapping_classes_have_low_fisher(self):
        rng = np.random.default_rng(0)
        a = rng.normal(scale=1.0, size=(50, 2))
        b = rng.normal(scale=1.0, size=(50, 2))  # same distribution
        samples = np.vstack([a, b])
        labels = np.array([0] * 50 + [1] * 50)
        result = compute_fisher_discriminant(samples, labels)
        assert result.mean < 1.0  # overlapping → small ratio

    def test_centroids_optional(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(size=(20, 3))
        labels = np.array([0] * 10 + [1] * 10)
        r1 = compute_fisher_discriminant(samples, labels)
        centroids = compute_class_centroids(samples, labels)
        r2 = compute_fisher_discriminant(samples, labels, centroids=centroids)
        assert r1.mean == pytest.approx(r2.mean)
        assert r1.min == pytest.approx(r2.min)

    def test_zero_within_class_variance(self):
        # All samples coincide with centroids → within-class variance is 0
        samples = np.array([[0.0, 0.0], [10.0, 0.0]])
        labels = np.array([0, 1])
        centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
        result = compute_fisher_discriminant(samples, labels, centroids=centroids)
        # Within is 0 for the only pair → falls back to (0, 0)
        assert result == FisherDiscriminant(0.0, 0.0)


class TestComputeClassDimensionality:
    def test_one_dimensional_class(self):
        # All samples in class 0 lie on x-axis → PR ≈ 1
        samples = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [10.0, 100.0],  # class 1, irrelevant
            ]
        )
        labels = np.array([0, 0, 0, 1])
        dims = compute_class_dimensionality(samples, labels)
        assert dims[0] == pytest.approx(1.0, abs=1e-10)

    def test_isotropic_class(self):
        # Class 0: equal variance along three axes → PR ≈ 3
        samples = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        labels = np.zeros(6, dtype=int)
        dims = compute_class_dimensionality(samples, labels)
        assert dims[0] == pytest.approx(3.0, abs=1e-10)

    def test_singleton_class_has_zero_dim(self):
        samples = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([0, 1])
        dims = compute_class_dimensionality(samples, labels)
        np.testing.assert_allclose(dims, [0.0, 0.0])


class TestComputeCenterSpread:
    def test_two_centroids_unit_apart(self):
        # Two centroids at ±0.5 from origin → mean(||c-mu||²) = 0.25 → spread = 0.5
        centroids = np.array([[-0.5, 0.0], [0.5, 0.0]])
        spread = compute_center_spread(centroids)
        assert spread == pytest.approx(0.5)

    def test_collapsed_centroids(self):
        centroids = np.zeros((4, 3))
        spread = compute_center_spread(centroids)
        assert spread == 0.0

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2D"):
            compute_center_spread(np.zeros(5))
