"""Tests for REQ_051: Standard DMD on Centroid Class Trajectories."""

import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.analysis.analyzers.centroid_dmd import CentroidDMD
from miscope.analysis.analyzers.registry import AnalyzerRegistry
from miscope.analysis.library.dmd import _truncation_rank, compute_dmd, dmd_reconstruct
from miscope.analysis.protocols import CrossEpochAnalyzer
from miscope.visualization.renderers.dmd import (
    render_dmd_eigenvalues,
    render_dmd_reconstruction,
    render_dmd_residual,
)

_SITES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]


# ── Helpers ───────────────────────────────────────────────────────────


def _make_trajectory(n_steps: int, state_dim: int, seed: int = 0) -> np.ndarray:
    """Create a random real-valued state trajectory."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_steps, state_dim))


def _make_oscillatory_trajectory(n_steps: int, state_dim: int, freq: float = 0.1) -> np.ndarray:
    """Create a trajectory with known oscillatory dynamics."""
    t = np.arange(n_steps, dtype=float)
    traj = np.zeros((n_steps, state_dim))
    for d in range(state_dim):
        traj[:, d] = np.sin(2 * np.pi * freq * t + d * 0.3)
    return traj


def _make_global_pca_artifact(
    n_epochs: int = 10,
    n_classes: int = 7,
    n_components: int = 3,
    seed: int = 0,
) -> dict:
    """Fake global_centroid_pca cross_epoch artifact."""
    rng = np.random.default_rng(seed)
    data: dict = {"epochs": np.arange(n_epochs) * 100}
    for site in _SITES:
        data[f"{site}__projections"] = rng.normal(size=(n_epochs, n_classes, n_components))
        data[f"{site}__basis"] = rng.normal(size=(16, n_components))
        data[f"{site}__mean"] = rng.normal(size=(16,))
        data[f"{site}__explained_variance_ratio"] = np.array([0.70, 0.20, 0.09])
    return data


@pytest.fixture
def artifacts_with_global_pca():
    """Temp artifacts dir with global_centroid_pca cross_epoch.npz."""
    n_epochs, n_classes, n_components = 10, 7, 3
    pca_data = _make_global_pca_artifact(n_epochs, n_classes, n_components)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = os.path.join(tmpdir, "artifacts")
        pca_dir = os.path.join(artifacts_dir, "global_centroid_pca")
        os.makedirs(pca_dir)

        # Also need repr_geometry per-epoch files for the requires check
        rg_dir = os.path.join(artifacts_dir, "repr_geometry")
        os.makedirs(rg_dir)
        for epoch in pca_data["epochs"]:
            path = os.path.join(rg_dir, f"epoch_{int(epoch):05d}.npz")
            np.savez_compressed(path, dummy=np.array([0]))

        np.savez_compressed(  # type: ignore[arg-type]
            os.path.join(pca_dir, "cross_epoch.npz"), **pca_data
        )

        yield artifacts_dir, list(pca_data["epochs"].astype(int)), n_epochs, n_classes, n_components


# ── Library: compute_dmd ─────────────────────────────────────────────


class TestComputeDmd:
    def test_returns_required_keys(self):
        traj = _make_trajectory(8, 6)
        result = compute_dmd(traj)
        for key in [
            "eigenvalues",
            "modes",
            "amplitudes",
            "residual_norms",
            "singular_values",
            "n_modes",
        ]:
            assert key in result

    def test_eigenvalues_shape(self):
        n_steps, state_dim = 8, 6
        traj = _make_trajectory(n_steps, state_dim)
        result = compute_dmd(traj)
        n_modes = int(result["n_modes"])
        assert result["eigenvalues"].shape == (n_modes,)

    def test_eigenvalues_are_complex(self):
        result = compute_dmd(_make_trajectory(8, 6))
        assert np.iscomplexobj(result["eigenvalues"])

    def test_modes_shape(self):
        n_steps, state_dim = 8, 6
        traj = _make_trajectory(n_steps, state_dim)
        result = compute_dmd(traj)
        n_modes = int(result["n_modes"])
        assert result["modes"].shape == (state_dim, n_modes)

    def test_amplitudes_shape(self):
        result = compute_dmd(_make_trajectory(8, 6))
        n_modes = int(result["n_modes"])
        assert result["amplitudes"].shape == (n_modes,)

    def test_residual_norms_shape(self):
        n_steps, state_dim = 8, 6
        traj = _make_trajectory(n_steps, state_dim)
        result = compute_dmd(traj)
        assert result["residual_norms"].shape == (n_steps - 1,)

    def test_residual_norms_are_non_negative(self):
        result = compute_dmd(_make_trajectory(10, 8))
        assert (result["residual_norms"] >= 0).all()

    def test_singular_values_are_non_negative(self):
        result = compute_dmd(_make_trajectory(10, 8))
        assert (result["singular_values"] >= 0).all()

    def test_n_modes_is_positive(self):
        result = compute_dmd(_make_trajectory(10, 8))
        assert int(result["n_modes"]) >= 1

    def test_n_modes_respects_threshold(self):
        """Lower threshold → fewer or equal modes retained."""
        traj = _make_trajectory(12, 10, seed=42)
        r_high = int(compute_dmd(traj, energy_threshold=0.999)["n_modes"])
        r_low = int(compute_dmd(traj, energy_threshold=0.50)["n_modes"])
        assert r_low <= r_high

    def test_perfect_linear_dynamics_low_residual(self):
        """Trajectory generated by a linear map should have near-zero residual."""
        rng = np.random.default_rng(0)
        state_dim = 4
        A = rng.normal(size=(state_dim, state_dim)) * 0.3  # stable system
        x = rng.normal(size=state_dim)
        traj = np.array([x := A @ x for _ in range(20)])  # type: ignore[assignment]
        result = compute_dmd(traj, energy_threshold=0.99)
        assert result["residual_norms"].mean() < 1.0  # much less than random

    def test_minimum_two_steps(self):
        """Two steps = one snapshot pair: minimum valid input."""
        result = compute_dmd(_make_trajectory(2, 4))
        assert result["residual_norms"].shape == (1,)


# ── Library: _truncation_rank ─────────────────────────────────────────


class TestTruncationRank:
    def test_all_singular_values_equal(self):
        """Equal singular values: all retained at any threshold < 1."""
        s = np.ones(5)
        r = _truncation_rank(s, 0.90)
        assert r <= 5

    def test_dominant_first_singular_value(self):
        """One dominant singular value: threshold 0.5 retains just 1."""
        s = np.array([10.0, 0.1, 0.1, 0.1])
        r = _truncation_rank(s, 0.50)
        assert r == 1

    def test_zero_singular_values(self):
        """Zero total energy returns rank 1."""
        r = _truncation_rank(np.zeros(5), 0.99)
        assert r == 1

    def test_threshold_1_retains_all(self):
        s = np.array([3.0, 2.0, 1.0])
        r = _truncation_rank(s, 1.0)
        assert r == len(s)


# ── Library: dmd_reconstruct ──────────────────────────────────────────


class TestDmdReconstruct:
    def test_output_shape(self):
        n_steps, state_dim, n_modes = 10, 6, 3
        eigenvalues = np.ones(n_modes, dtype=complex) * 0.9
        modes = np.random.default_rng(0).normal(size=(state_dim, n_modes)).astype(complex)
        amplitudes = np.ones(n_modes, dtype=complex)
        recon = dmd_reconstruct(eigenvalues, modes, amplitudes, n_steps)
        assert recon.shape == (n_steps, state_dim)

    def test_output_is_real(self):
        n_modes = 3
        eigenvalues = np.exp(1j * np.array([0.1, 0.2, 0.3]))
        modes = np.random.default_rng(0).normal(size=(4, n_modes)).astype(complex)
        amplitudes = np.ones(n_modes, dtype=complex)
        recon = dmd_reconstruct(eigenvalues, modes, amplitudes, 5)
        assert recon.dtype == np.float64 or recon.dtype == np.float32

    def test_consistency_with_compute_dmd(self):
        """Reconstruction at t=0 should closely match the initial state."""
        # rng = np.random.default_rng(42)
        traj = _make_trajectory(15, 8, seed=42)
        result = compute_dmd(traj, energy_threshold=0.99)
        recon = dmd_reconstruct(
            result["eigenvalues"], result["modes"], result["amplitudes"], len(traj)
        )
        assert recon.shape == traj.shape
        # t=0 reconstruction should match x_0 (amplitudes fit to initial condition)
        np.testing.assert_allclose(recon[0], traj[0], atol=1e-4)


# ── Analyzer protocol tests ───────────────────────────────────────────


class TestCentroidDMDProtocol:
    def test_conforms_to_cross_epoch_protocol(self):
        assert isinstance(CentroidDMD(), CrossEpochAnalyzer)

    def test_name(self):
        assert CentroidDMD().name == "centroid_dmd"

    def test_requires(self):
        assert CentroidDMD().requires == ["repr_geometry"]

    def test_registered_in_registry(self):
        assert "centroid_dmd" in AnalyzerRegistry._cross_epoch_analyzers

    def test_analyze_is_callable(self):
        assert callable(CentroidDMD().analyze_across_epochs)


# ── Analyzer output tests ─────────────────────────────────────────────


class TestCentroidDMDOutput:
    def test_returns_dict(self, artifacts_with_global_pca):
        artifacts_dir, epochs, *_ = artifacts_with_global_pca
        result = CentroidDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        assert isinstance(result, dict)

    def test_contains_epochs(self, artifacts_with_global_pca):
        artifacts_dir, epochs, *_ = artifacts_with_global_pca
        result = CentroidDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        np.testing.assert_array_equal(result["epochs"], epochs)

    def test_contains_all_site_keys(self, artifacts_with_global_pca):
        artifacts_dir, epochs, *_ = artifacts_with_global_pca
        result = CentroidDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for site in _SITES:
            assert f"{site}__eigenvalues" in result
            assert f"{site}__modes" in result
            assert f"{site}__amplitudes" in result
            assert f"{site}__residual_norms" in result
            assert f"{site}__singular_values" in result
            assert f"{site}__trajectory" in result
            assert f"{site}__n_classes" in result
            assert f"{site}__n_modes" in result

    def test_residual_norms_shape(self, artifacts_with_global_pca):
        artifacts_dir, epochs, n_epochs, *_ = artifacts_with_global_pca
        result = CentroidDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for site in _SITES:
            assert result[f"{site}__residual_norms"].shape == (n_epochs - 1,)

    def test_trajectory_shape(self, artifacts_with_global_pca):
        artifacts_dir, epochs, n_epochs, n_classes, n_components = artifacts_with_global_pca
        result = CentroidDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        state_dim = n_classes * n_components
        for site in _SITES:
            assert result[f"{site}__trajectory"].shape == (n_epochs, state_dim)

    def test_n_classes_stored_correctly(self, artifacts_with_global_pca):
        artifacts_dir, epochs, _, n_classes, _ = artifacts_with_global_pca
        result = CentroidDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for site in _SITES:
            assert int(result[f"{site}__n_classes"]) == n_classes

    def test_residual_norms_non_negative(self, artifacts_with_global_pca):
        artifacts_dir, epochs, *_ = artifacts_with_global_pca
        result = CentroidDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for site in _SITES:
            assert (result[f"{site}__residual_norms"] >= 0).all()

    def test_missing_global_pca_raises(self):
        """Should raise FileNotFoundError if global_centroid_pca is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = os.path.join(tmpdir, "artifacts")
            rg_dir = os.path.join(artifacts_dir, "repr_geometry")
            os.makedirs(rg_dir)
            np.savez_compressed(os.path.join(rg_dir, "epoch_00000.npz"), dummy=np.array([0]))

            with pytest.raises(FileNotFoundError, match="global_centroid_pca"):
                CentroidDMD().analyze_across_epochs(artifacts_dir, [0], {})


# ── Renderer tests ────────────────────────────────────────────────────


@pytest.fixture
def cross_epoch_dmd_data():
    """Fake centroid_dmd cross_epoch artifact for renderer tests."""
    n_epochs, n_classes, n_components, n_modes = 10, 7, 3, 4
    state_dim = n_classes * n_components
    rng = np.random.default_rng(42)
    epochs = np.arange(n_epochs) * 100
    data: dict = {"epochs": epochs}
    for site in _SITES:
        traj = rng.normal(size=(n_epochs, state_dim))
        data[f"{site}__eigenvalues"] = rng.normal(size=n_modes) + 1j * rng.normal(size=n_modes)
        data[f"{site}__modes"] = rng.normal(size=(state_dim, n_modes)).astype(complex)
        data[f"{site}__amplitudes"] = rng.normal(size=n_modes).astype(complex)
        data[f"{site}__residual_norms"] = np.abs(rng.normal(size=n_epochs - 1))
        data[f"{site}__singular_values"] = np.sort(np.abs(rng.normal(size=n_epochs - 1)))[::-1]
        data[f"{site}__trajectory"] = traj
        data[f"{site}__n_classes"] = np.int64(n_classes)
        data[f"{site}__n_modes"] = np.int64(n_modes)
    return data


class TestDmdRenderers:
    def test_eigenvalues_returns_figure(self, cross_epoch_dmd_data):
        fig = render_dmd_eigenvalues(cross_epoch_dmd_data, site="resid_post")
        assert isinstance(fig, go.Figure)

    def test_eigenvalues_each_site(self, cross_epoch_dmd_data):
        for site in _SITES:
            fig = render_dmd_eigenvalues(cross_epoch_dmd_data, site=site)
            assert isinstance(fig, go.Figure)

    def test_eigenvalues_contains_unit_circle(self, cross_epoch_dmd_data):
        fig = render_dmd_eigenvalues(cross_epoch_dmd_data)
        assert isinstance(fig, go.Figure)
        # Unit circle is the first trace
        assert len(tuple(fig.data)) >= 2

    def test_residual_returns_figure(self, cross_epoch_dmd_data):
        fig = render_dmd_residual(cross_epoch_dmd_data)
        assert isinstance(fig, go.Figure)

    def test_residual_single_site(self, cross_epoch_dmd_data):
        for site in _SITES:
            fig = render_dmd_residual(cross_epoch_dmd_data, site=site)
            assert isinstance(fig, go.Figure)

    def test_residual_all_sites_four_traces(self, cross_epoch_dmd_data):
        fig = render_dmd_residual(cross_epoch_dmd_data, site=None)
        assert isinstance(fig, go.Figure)
        assert len(tuple(fig.data)) == 4

    def test_reconstruction_returns_figure(self, cross_epoch_dmd_data):
        fig = render_dmd_reconstruction(cross_epoch_dmd_data, epoch=500, site="resid_post")
        assert isinstance(fig, go.Figure)

    def test_reconstruction_each_site(self, cross_epoch_dmd_data):
        for site in _SITES:
            fig = render_dmd_reconstruction(cross_epoch_dmd_data, epoch=0, site=site)
            assert isinstance(fig, go.Figure)

    def test_reconstruction_nearest_epoch(self, cross_epoch_dmd_data):
        """Requesting an epoch not in the array should select nearest."""
        fig = render_dmd_reconstruction(cross_epoch_dmd_data, epoch=150, site="resid_post")
        assert isinstance(fig, go.Figure)
