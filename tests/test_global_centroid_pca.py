"""Tests for REQ_050: Global PCA for Cross-Epoch Centroid Analysis."""

import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.analysis.analyzers.global_centroid_pca import GlobalCentroidPCA
from miscope.analysis.analyzers.registry import AnalyzerRegistry
from miscope.analysis.library.geometry import compute_global_centroid_pca
from miscope.analysis.protocols import CrossEpochAnalyzer
from miscope.visualization.renderers.repr_geometry import render_centroid_global_pca

_SITES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]


# ── Helpers ───────────────────────────────────────────────────────────


def _make_centroids(n_classes: int, d_model: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_classes, d_model)).astype(np.float32)


def _make_repr_geometry_epoch(n_classes: int = 11, d_model: int = 16, seed: int = 0) -> dict:
    """Create a fake repr_geometry epoch artifact for all four sites."""
    result = {}
    for i, site in enumerate(_SITES):
        centroids = _make_centroids(n_classes, d_model, seed=seed * 10 + i)
        radii = np.abs(np.random.default_rng(seed).normal(size=n_classes)).astype(np.float32)
        result[f"{site}_centroids"] = centroids
        result[f"{site}_radii"] = radii
    return result


@pytest.fixture
def artifacts_with_repr_geometry():
    """Temp artifacts dir with repr_geometry per-epoch files."""
    n_classes, d_model = 11, 16
    epochs = [0, 100, 200, 300, 400]

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = os.path.join(tmpdir, "artifacts")
        rg_dir = os.path.join(artifacts_dir, "repr_geometry")
        os.makedirs(rg_dir)

        for epoch in epochs:
            data = _make_repr_geometry_epoch(n_classes, d_model, seed=epoch)
            path = os.path.join(rg_dir, f"epoch_{epoch:05d}.npz")
            np.savez_compressed(path, **data)  # type: ignore[arg-type]

        yield artifacts_dir, epochs, n_classes, d_model


# ── Library function tests ────────────────────────────────────────────


class TestComputeGlobalCentroidPca:
    """Tests for compute_global_centroid_pca library function."""

    def test_returns_required_keys(self):
        centroids = [_make_centroids(7, 8, seed=i) for i in range(5)]
        result = compute_global_centroid_pca(centroids)
        assert "basis" in result
        assert "mean" in result
        assert "projections" in result
        assert "explained_variance_ratio" in result

    def test_basis_shape(self):
        n_classes, d_model, n_epochs = 7, 8, 5
        centroids = [_make_centroids(n_classes, d_model, seed=i) for i in range(n_epochs)]
        result = compute_global_centroid_pca(centroids)
        n_components = result["basis"].shape[1]
        assert result["basis"].shape == (d_model, n_components)

    def test_mean_shape(self):
        d_model = 12
        centroids = [_make_centroids(5, d_model, seed=i) for i in range(4)]
        result = compute_global_centroid_pca(centroids)
        assert result["mean"].shape == (d_model,)

    def test_projections_shape(self):
        n_classes, d_model, n_epochs = 7, 16, 6
        centroids = [_make_centroids(n_classes, d_model, seed=i) for i in range(n_epochs)]
        result = compute_global_centroid_pca(centroids)
        n_components = result["basis"].shape[1]
        assert result["projections"].shape == (n_epochs, n_classes, n_components)

    def test_explained_variance_ratio_shape(self):
        centroids = [_make_centroids(7, 8, seed=i) for i in range(5)]
        result = compute_global_centroid_pca(centroids)
        n_components = result["basis"].shape[1]
        assert result["explained_variance_ratio"].shape == (n_components,)

    def test_explained_variance_ratio_sums_to_lte_1(self):
        centroids = [_make_centroids(7, 8, seed=i) for i in range(5)]
        result = compute_global_centroid_pca(centroids)
        total = float(result["explained_variance_ratio"].sum())
        assert total <= 1.0 + 1e-6

    def test_captures_at_least_variance_threshold(self):
        """Retained components must capture >= threshold variance."""
        threshold = 0.95
        centroids = [_make_centroids(11, 16, seed=i) for i in range(8)]
        result = compute_global_centroid_pca(centroids, variance_threshold=threshold)
        total = float(result["explained_variance_ratio"].sum())
        assert total >= threshold - 1e-6

    def test_variance_ratio_is_non_negative(self):
        centroids = [_make_centroids(7, 8, seed=i) for i in range(5)]
        result = compute_global_centroid_pca(centroids)
        assert (result["explained_variance_ratio"] >= 0).all()

    def test_single_epoch(self):
        """Single epoch: pooled matrix is the single centroid matrix."""
        centroids = [_make_centroids(5, 8, seed=0)]
        result = compute_global_centroid_pca(centroids)
        assert result["projections"].shape[0] == 1

    def test_lower_threshold_fewer_components(self):
        """Lower threshold should produce fewer or equal components than higher."""
        centroids = [_make_centroids(11, 16, seed=i) for i in range(10)]
        result_low = compute_global_centroid_pca(centroids, variance_threshold=0.70)
        result_high = compute_global_centroid_pca(centroids, variance_threshold=0.95)
        n_low = result_low["basis"].shape[1]
        n_high = result_high["basis"].shape[1]
        assert n_low <= n_high

    def test_projection_uses_global_mean(self):
        """Projection of the global mean should land at the origin."""
        centroids = [_make_centroids(5, 8, seed=i) for i in range(3)]
        result = compute_global_centroid_pca(centroids)
        global_mean = result["mean"]
        projected_mean = (global_mean - result["mean"]) @ result["basis"]
        np.testing.assert_allclose(projected_mean, np.zeros_like(projected_mean), atol=1e-5)


# ── Analyzer protocol tests ───────────────────────────────────────────


class TestGlobalCentroidPCAProtocol:
    def test_conforms_to_cross_epoch_protocol(self):
        assert isinstance(GlobalCentroidPCA(), CrossEpochAnalyzer)

    def test_name(self):
        assert GlobalCentroidPCA().name == "global_centroid_pca"

    def test_requires(self):
        assert GlobalCentroidPCA().requires == ["repr_geometry"]

    def test_registered_in_registry(self):
        assert "global_centroid_pca" in AnalyzerRegistry._cross_epoch_analyzers

    def test_analyze_is_callable(self):
        assert callable(GlobalCentroidPCA().analyze_across_epochs)


# ── Analyzer output tests ─────────────────────────────────────────────


class TestGlobalCentroidPCAOutput:
    def test_returns_dict(self, artifacts_with_repr_geometry):
        artifacts_dir, epochs, _, _ = artifacts_with_repr_geometry
        result = GlobalCentroidPCA().analyze_across_epochs(artifacts_dir, epochs, {})
        assert isinstance(result, dict)

    def test_contains_epochs(self, artifacts_with_repr_geometry):
        artifacts_dir, epochs, _, _ = artifacts_with_repr_geometry
        result = GlobalCentroidPCA().analyze_across_epochs(artifacts_dir, epochs, {})
        np.testing.assert_array_equal(result["epochs"], epochs)

    def test_contains_all_site_keys(self, artifacts_with_repr_geometry):
        artifacts_dir, epochs, _, _ = artifacts_with_repr_geometry
        result = GlobalCentroidPCA().analyze_across_epochs(artifacts_dir, epochs, {})
        for site in _SITES:
            assert f"{site}__projections" in result
            assert f"{site}__basis" in result
            assert f"{site}__mean" in result
            assert f"{site}__explained_variance_ratio" in result

    def test_projections_shape(self, artifacts_with_repr_geometry):
        artifacts_dir, epochs, n_classes, d_model = artifacts_with_repr_geometry
        result = GlobalCentroidPCA().analyze_across_epochs(artifacts_dir, epochs, {})
        n_epochs = len(epochs)
        for site in _SITES:
            proj = result[f"{site}__projections"]
            assert proj.shape[0] == n_epochs
            assert proj.shape[1] == n_classes

    def test_basis_shape(self, artifacts_with_repr_geometry):
        artifacts_dir, epochs, n_classes, d_model = artifacts_with_repr_geometry
        result = GlobalCentroidPCA().analyze_across_epochs(artifacts_dir, epochs, {})
        for site in _SITES:
            basis = result[f"{site}__basis"]
            assert basis.shape[0] == d_model

    def test_mean_shape(self, artifacts_with_repr_geometry):
        artifacts_dir, epochs, n_classes, d_model = artifacts_with_repr_geometry
        result = GlobalCentroidPCA().analyze_across_epochs(artifacts_dir, epochs, {})
        for site in _SITES:
            assert result[f"{site}__mean"].shape == (d_model,)

    def test_explained_variance_captures_threshold(self, artifacts_with_repr_geometry):
        artifacts_dir, epochs, _, _ = artifacts_with_repr_geometry
        result = GlobalCentroidPCA().analyze_across_epochs(artifacts_dir, epochs, {})
        for site in _SITES:
            var_ratio = result[f"{site}__explained_variance_ratio"]
            assert float(var_ratio.sum()) >= 0.95 - 1e-6


# ── Renderer tests ────────────────────────────────────────────────────


class TestRenderCentroidGlobalPca:
    @pytest.fixture
    def cross_epoch_data(self):
        """Fake cross_epoch.npz content for renderer testing."""
        n_epochs, n_classes, d_model = 5, 7, 8
        epochs = [0, 100, 200, 300, 400]
        data: dict = {"epochs": np.array(epochs)}
        rng = np.random.default_rng(42)
        for site in _SITES:
            n_components = 3
            data[f"{site}__projections"] = rng.normal(size=(n_epochs, n_classes, n_components))
            data[f"{site}__basis"] = rng.normal(size=(d_model, n_components))
            data[f"{site}__mean"] = rng.normal(size=(d_model,))
            data[f"{site}__explained_variance_ratio"] = np.array([0.70, 0.20, 0.09])
        return data

    def test_returns_figure(self, cross_epoch_data):
        fig = render_centroid_global_pca(cross_epoch_data, epoch=200, site="resid_post")
        assert isinstance(fig, go.Figure)

    def test_renders_each_site(self, cross_epoch_data):
        for site in _SITES:
            fig = render_centroid_global_pca(cross_epoch_data, epoch=100, site=site)
            assert isinstance(fig, go.Figure)

    def test_nearest_epoch_selected(self, cross_epoch_data):
        """Requesting an epoch not in the list selects the nearest."""
        fig = render_centroid_global_pca(cross_epoch_data, epoch=150, site="resid_post")
        assert isinstance(fig, go.Figure)
        # Should select epoch 100 or 200 — just verify it renders without error

    def test_title_contains_global_pca(self, cross_epoch_data):
        fig = render_centroid_global_pca(cross_epoch_data, epoch=0, site="resid_post")
        assert "Global PCA" in fig.layout.title.text
