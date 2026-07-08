"""Tests for REQ_118 neuron grouping primitives."""

import numpy as np
import pytest

from miscope.analysis.library.grouping import (
    group_neurons,
    group_neurons_summary,
)
from miscope.core.grouping import UNASSIGNED, GroupAssignment, GroupSummary


def _make_well_separated_features(
    n_per_group: int = 30,
    n_features: int = 8,
    n_groups: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate features clustered around `n_groups` well-separated centroids.

    Returns (features, true_labels). Each cluster sits at a distinct corner
    of feature space with small Gaussian noise — well-separated enough that
    any reasonable clustering should recover the structure exactly.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(n_groups, n_features)) * 5.0
    features = []
    labels = []
    for g in range(n_groups):
        cluster = centroids[g] + rng.normal(scale=0.1, size=(n_per_group, n_features))
        features.append(cluster)
        labels.extend([g] * n_per_group)
    return np.vstack(features), np.array(labels, dtype=np.int64)


# ── group_neurons (kmeans) ───────────────────────────────────────────


class TestGroupNeuronsKMeans:
    def test_returns_group_assignment(self):
        features, _ = _make_well_separated_features()
        result = group_neurons(features, n_groups=4, method="kmeans")
        assert isinstance(result, GroupAssignment)

    def test_assignments_shape(self):
        features, _ = _make_well_separated_features(n_per_group=10)
        result = group_neurons(features, n_groups=4, method="kmeans")
        assert result.assignments.shape == (40,)

    def test_assignments_in_range(self):
        features, _ = _make_well_separated_features()
        result = group_neurons(features, n_groups=4, method="kmeans")
        assert (result.assignments >= 0).all()
        assert (result.assignments < 4).all()

    def test_n_groups_propagates(self):
        features, _ = _make_well_separated_features()
        result = group_neurons(features, n_groups=4, method="kmeans")
        assert result.n_groups == 4

    def test_method_propagates(self):
        features, _ = _make_well_separated_features()
        result = group_neurons(features, n_groups=4, method="kmeans")
        assert result.method == "kmeans"

    def test_feature_basis_name_propagates(self):
        features, _ = _make_well_separated_features()
        result = group_neurons(
            features,
            n_groups=4,
            method="kmeans",
            feature_basis_name="weight_signature",
        )
        assert result.feature_basis_name == "weight_signature"

    def test_confidence_is_none(self):
        features, _ = _make_well_separated_features()
        result = group_neurons(features, n_groups=4, method="kmeans")
        assert result.confidence is None

    def test_recovers_well_separated_clusters(self):
        """Well-separated clusters must be recovered up to label permutation."""
        features, true_labels = _make_well_separated_features(
            n_per_group=30, n_features=8, n_groups=4, seed=1
        )
        result = group_neurons(features, n_groups=4, method="kmeans", random_state=1)
        # Build the predicted-vs-true confusion matrix; recovery means each
        # predicted cluster is dominantly one true label.
        confusion = np.zeros((4, 4), dtype=np.int64)
        for true, pred in zip(true_labels, result.assignments):
            confusion[true, int(pred)] += 1
        # Each row should have at least 25/30 mass concentrated in one column
        for row in confusion:
            assert row.max() >= 25

    def test_deterministic_with_seed(self):
        features, _ = _make_well_separated_features()
        a = group_neurons(features, n_groups=4, method="kmeans", random_state=7)
        b = group_neurons(features, n_groups=4, method="kmeans", random_state=7)
        np.testing.assert_array_equal(a.assignments, b.assignments)


# ── group_neurons (argmax_by_basis) ──────────────────────────────────


class TestGroupNeuronsArgmaxByBasis:
    def test_assigns_to_dominant_component(self):
        features = np.array(
            [
                [0.9, 0.05, 0.05],  # dominant 0
                [0.1, 0.8, 0.1],  # dominant 1
                [0.2, 0.2, 0.6],  # dominant 2
            ]
        )
        result = group_neurons(features, n_groups=3, method="argmax_by_basis")
        np.testing.assert_array_equal(result.assignments, [0, 1, 2])

    def test_returns_per_neuron_confidence(self):
        features = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.5, 0.4, 0.1],
            ]
        )
        result = group_neurons(features, n_groups=3, method="argmax_by_basis")
        assert result.confidence is not None
        assert result.confidence.shape == (2,)
        # Variance-fraction convention: max(x^2) / sum(x^2)
        # First neuron: 0.81 / (0.81 + 0.0025 + 0.0025) ≈ 0.9939
        np.testing.assert_allclose(result.confidence[0], 0.81 / 0.815, atol=1e-9)
        # Second neuron: 0.25 / (0.25 + 0.16 + 0.01) = 0.25 / 0.42
        np.testing.assert_allclose(result.confidence[1], 0.25 / 0.42, atol=1e-9)

    def test_threshold_marks_low_confidence_unassigned(self):
        # Variance-fraction confidence semantics:
        # [0.9, 0.05, 0.05]:  0.81 / 0.815  ≈ 0.994 -> assigned
        # [0.4, 0.3, 0.3]:    0.16 / 0.34   ≈ 0.471 -> below 0.5 -> unassigned
        # [0.6, 0.3, 0.1]:    0.36 / 0.46   ≈ 0.783 -> assigned
        features = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.4, 0.3, 0.3],
                [0.6, 0.3, 0.1],
            ]
        )
        result = group_neurons(
            features,
            n_groups=3,
            method="argmax_by_basis",
            confidence_threshold=0.5,
        )
        assert result.assignments[0] == 0
        assert result.assignments[1] == UNASSIGNED
        assert result.assignments[2] == 0

    def test_no_threshold_assigns_all(self):
        features = np.array([[0.4, 0.3, 0.3]])
        result = group_neurons(
            features, n_groups=3, method="argmax_by_basis", confidence_threshold=None
        )
        assert result.assignments[0] == 0  # dominant despite low confidence

    def test_handles_negative_features_via_absolute(self):
        """argmax_by_basis uses |features|; sign should not flip the dominant."""
        features = np.array([[-0.9, 0.05, 0.05]])
        result = group_neurons(features, n_groups=3, method="argmax_by_basis")
        assert result.assignments[0] == 0
        # Variance fraction: 0.81 / (0.81 + 0.0025 + 0.0025) ≈ 0.9939
        np.testing.assert_allclose(result.confidence[0], 0.81 / 0.815, atol=1e-9)  # pyright: ignore[reportOptionalSubscript]

    def test_zero_row_has_zero_confidence(self):
        features = np.array([[0.0, 0.0, 0.0]])
        result = group_neurons(
            features, n_groups=3, method="argmax_by_basis", confidence_threshold=0.1
        )
        assert result.confidence[0] == 0.0  # pyright: ignore[reportOptionalSubscript]
        assert result.assignments[0] == UNASSIGNED


# ── group_neurons (input validation) ─────────────────────────────────


class TestGroupNeuronsValidation:
    def test_rejects_unknown_method(self):
        features = np.zeros((10, 4))
        with pytest.raises(ValueError, match="unknown method"):
            group_neurons(features, n_groups=3, method="not_a_method")

    def test_rejects_1d_features(self):
        with pytest.raises(ValueError, match="features must be 2D"):
            group_neurons(np.array([1.0, 2.0, 3.0]), n_groups=2)

    def test_rejects_n_groups_below_one(self):
        features = np.zeros((10, 4))
        with pytest.raises(ValueError, match="n_groups"):
            group_neurons(features, n_groups=0)


# ── group_neurons_summary ────────────────────────────────────────────


class TestGroupNeuronsSummary:
    def test_returns_group_summary(self):
        features, _ = _make_well_separated_features()
        assignment = group_neurons(features, n_groups=4, method="kmeans")
        summary = group_neurons_summary(features, assignment)
        assert isinstance(summary, GroupSummary)

    def test_centroids_shape(self):
        features, _ = _make_well_separated_features(n_features=8)
        assignment = group_neurons(features, n_groups=4, method="kmeans")
        summary = group_neurons_summary(features, assignment)
        assert summary.centroids.shape == (4, 8)

    def test_n_per_group_sums_to_assigned(self):
        features = np.random.default_rng(0).normal(size=(50, 4))
        assignment = group_neurons(features, n_groups=3, method="kmeans")
        summary = group_neurons_summary(features, assignment)
        assert int(summary.n_per_group.sum()) == 50  # no unassigned for kmeans

    def test_radii_non_negative(self):
        features, _ = _make_well_separated_features()
        assignment = group_neurons(features, n_groups=4, method="kmeans")
        summary = group_neurons_summary(features, assignment)
        assert (summary.radii >= 0).all()

    def test_fisher_min_le_mean(self):
        features, _ = _make_well_separated_features()
        assignment = group_neurons(features, n_groups=4, method="kmeans")
        summary = group_neurons_summary(features, assignment)
        assert summary.fisher_min <= summary.fisher_mean + 1e-9

    def test_well_separated_clusters_have_high_fisher(self):
        """Tight, well-separated clusters should have a large Fisher ratio."""
        features, _ = _make_well_separated_features(seed=2)
        assignment = group_neurons(features, n_groups=4, method="kmeans", random_state=2)
        summary = group_neurons_summary(features, assignment)
        # Well-separated clusters should have very high Fisher discriminant
        assert summary.fisher_min > 100.0

    def test_unassigned_neurons_excluded_from_summary(self):
        # Variance-fraction confidence:
        # [0.9, 0.05, 0.05]:  0.81 / 0.815   ≈ 0.994 -> assigned
        # [0.05, 0.9, 0.05]:  same           ≈ 0.994 -> assigned
        # [0.34, 0.33, 0.33]: 0.1156 / 0.3334 ≈ 0.347 -> below 0.5 -> unassigned
        features = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.34, 0.33, 0.33],
            ]
        )
        assignment = group_neurons(
            features, n_groups=3, method="argmax_by_basis", confidence_threshold=0.5
        )
        summary = group_neurons_summary(features, assignment)
        assert summary.n_unassigned == 1
        assert int(summary.n_per_group.sum()) == 2

    def test_all_unassigned_returns_degenerate_summary(self):
        # Uniform-ish row: variance fraction ≈ 0.347, below 0.5 threshold
        features = np.array([[0.34, 0.33, 0.33]])
        assignment = group_neurons(
            features, n_groups=3, method="argmax_by_basis", confidence_threshold=0.5
        )
        summary = group_neurons_summary(features, assignment)
        assert summary.n_unassigned == 1
        assert summary.centroids.shape == (0, 3)
        assert summary.dispersion == 0.0

    def test_dispersion_equals_mean_radius(self):
        features, _ = _make_well_separated_features()
        assignment = group_neurons(features, n_groups=4, method="kmeans")
        summary = group_neurons_summary(features, assignment)
        np.testing.assert_allclose(summary.dispersion, float(summary.radii.mean()))

    def test_rejects_1d_features(self):
        assignment = GroupAssignment(
            assignments=np.array([0, 1, 0]),
            n_groups=2,
            method="kmeans",
            feature_basis_name="x",
            confidence=None,
        )
        with pytest.raises(ValueError, match="features must be 2D"):
            group_neurons_summary(np.array([1.0, 2.0, 3.0]), assignment)
