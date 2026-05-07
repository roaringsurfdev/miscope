"""Tests for REQ_042: Neuron Dynamics cross-epoch analyzer and renderers."""

import os
import tempfile
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.analysis.analyzers import AnalyzerRegistry
from miscope.analysis.analyzers.neuron_dynamics import (
    NeuronDynamicsAnalyzer,
    _compute_commitment_epochs,
    _compute_switch_counts,
)
from miscope.visualization.renderers.neuron_freq_clusters import (
    render_commitment_timeline,
    render_neuron_freq_trajectory,
    render_switch_count_distribution,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _make_norm_matrix(n_freq: int = 10, d_mlp: int = 8, seed: int = 0) -> np.ndarray:
    """Create a fake norm_matrix of shape (n_freq, d_mlp)."""
    rng = np.random.default_rng(seed)
    matrix = rng.dirichlet(np.ones(n_freq), size=d_mlp).T.astype(np.float32)
    return matrix


def _make_specialized_norm_matrix(
    n_freq: int = 10,
    d_mlp: int = 8,
    assignments: list[int] | None = None,
) -> np.ndarray:
    """Create a norm_matrix where neurons are strongly specialized."""
    matrix = np.full((n_freq, d_mlp), 0.01, dtype=np.float32)
    if assignments is None:
        assignments = [i % n_freq for i in range(d_mlp)]
    for neuron_idx, freq_idx in enumerate(assignments):
        matrix[freq_idx, neuron_idx] = 0.95
    return matrix


@pytest.fixture
def artifacts_with_neuron_freq_norm():
    """Create temp artifacts dir with neuron_freq_norm epoch files."""
    n_freq = 10
    d_mlp = 8
    # Epochs: neurons start random, then specialize, then some switch
    epoch_assignments = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],  # initial assignment
        100: [0, 1, 2, 3, 4, 5, 6, 7],  # same
        200: [0, 1, 2, 3, 4, 5, 6, 7],  # same
        300: [0, 1, 9, 3, 4, 5, 6, 7],  # neuron 2 switches to freq 9
        400: [0, 1, 9, 3, 8, 5, 6, 7],  # neuron 4 switches to freq 8
    }
    epochs = sorted(epoch_assignments.keys())

    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer_dir = os.path.join(tmpdir, "neuron_freq_norm")
        os.makedirs(analyzer_dir)

        for epoch in epochs:
            assignments = epoch_assignments[epoch]
            norm_matrix = _make_specialized_norm_matrix(n_freq, d_mlp, assignments)
            path = os.path.join(analyzer_dir, f"epoch_{epoch:05d}.npz")
            np.savez_compressed(path, norm_matrix=norm_matrix)

        yield tmpdir, epochs, epoch_assignments


# ── Analyzer unit tests ──────────────────────────────────────────────


class TestSwitchCounts:
    def test_no_switches(self):
        """Neuron that stays on same frequency has 0 switches."""
        dominant_freq = np.array([[0, 1], [0, 1], [0, 1]])
        max_frac = np.array([[0.9, 0.9], [0.9, 0.9], [0.9, 0.9]])
        result = _compute_switch_counts(dominant_freq, max_frac, threshold=0.1)
        np.testing.assert_array_equal(result, [0, 0])

    def test_one_switch(self):
        """Neuron that changes frequency once has 1 switch."""
        dominant_freq = np.array([[0, 1], [0, 1], [0, 2]])
        max_frac = np.array([[0.9, 0.9], [0.9, 0.9], [0.9, 0.9]])
        result = _compute_switch_counts(dominant_freq, max_frac, threshold=0.1)
        np.testing.assert_array_equal(result, [0, 1])

    def test_below_threshold_ignored(self):
        """Switches while below threshold are not counted."""
        dominant_freq = np.array([[0], [5], [3]])
        max_frac = np.array([[0.9], [0.01], [0.9]])
        result = _compute_switch_counts(dominant_freq, max_frac, threshold=0.1)
        # Neuron goes 0 → (below threshold) → 3, that's one switch from 0 to 3
        np.testing.assert_array_equal(result, [1])

    def test_multiple_switches(self):
        """Neuron that changes multiple times."""
        dominant_freq = np.array([[0], [1], [2], [3]])
        max_frac = np.array([[0.9], [0.9], [0.9], [0.9]])
        result = _compute_switch_counts(dominant_freq, max_frac, threshold=0.1)
        np.testing.assert_array_equal(result, [3])


class TestCommitmentEpochs:
    def test_stable_from_start(self):
        """Neuron stable from start commits at first epoch."""
        dominant_freq = np.array([[5], [5], [5]])
        max_frac = np.array([[0.9], [0.9], [0.9]])
        epochs = np.array([0, 100, 200])
        result = _compute_commitment_epochs(dominant_freq, max_frac, epochs, threshold=0.1)
        assert result[0] == 0

    def test_late_commitment(self):
        """Neuron that switches then stabilizes."""
        dominant_freq = np.array([[0], [1], [1], [1]])
        max_frac = np.array([[0.9], [0.9], [0.9], [0.9]])
        epochs = np.array([0, 100, 200, 300])
        result = _compute_commitment_epochs(dominant_freq, max_frac, epochs, threshold=0.1)
        assert result[0] == 100

    def test_uncommitted(self):
        """Neuron below threshold at end has NaN commitment."""
        dominant_freq = np.array([[0], [1], [2]])
        max_frac = np.array([[0.9], [0.9], [0.01]])
        epochs = np.array([0, 100, 200])
        result = _compute_commitment_epochs(dominant_freq, max_frac, epochs, threshold=0.1)
        assert np.isnan(result[0])


# ── Analyzer integration test ────────────────────────────────────────


class TestNeuronDynamicsAnalyzer:
    def test_registration(self):
        """NeuronDynamicsAnalyzer is registered as a cross-epoch analyzer."""
        assert "neuron_dynamics" in AnalyzerRegistry._cross_epoch_analyzers

    def test_analyze_across_epochs(
        self, artifacts_with_neuron_freq_norm: tuple[str, list[int], dict[int, list[int]]]
    ):
        """Analyzer produces expected output fields and shapes."""
        artifacts_dir, epochs, assignments = artifacts_with_neuron_freq_norm
        analyzer = NeuronDynamicsAnalyzer()
        result = analyzer.analyze_across_epochs(artifacts_dir, epochs, context={})

        assert "epochs" in result
        assert "dominant_freq" in result
        assert "max_frac" in result
        assert "switch_counts" in result
        assert "commitment_epochs" in result
        assert "threshold" in result

        n_epochs = len(epochs)
        d_mlp = 8

        assert result["epochs"].shape == (n_epochs,)
        assert result["dominant_freq"].shape == (n_epochs, d_mlp)
        assert result["max_frac"].shape == (n_epochs, d_mlp)
        assert result["switch_counts"].shape == (d_mlp,)
        assert result["commitment_epochs"].shape == (d_mlp,)

    def test_switch_counts_correct(
        self, artifacts_with_neuron_freq_norm: tuple[str, list[int], dict[int, list[int]]]
    ):
        """Switch counts match known assignments."""
        artifacts_dir, epochs, assignments = artifacts_with_neuron_freq_norm
        analyzer = NeuronDynamicsAnalyzer()
        result = analyzer.analyze_across_epochs(artifacts_dir, epochs, context={})

        # Neuron 2 switches once (freq 2 → 9 at epoch 300)
        assert result["switch_counts"][2] == 1
        # Neuron 4 switches once (freq 4 → 8 at epoch 400)
        assert result["switch_counts"][4] == 1
        # Neurons 0, 1, 3, 5, 6, 7 never switch
        for n in [0, 1, 3, 5, 6, 7]:
            assert result["switch_counts"][n] == 0


# ── Renderer tests ───────────────────────────────────────────────────


@pytest.fixture
def sample_cross_epoch_data():
    """Create sample cross-epoch data for renderer tests."""
    n_epochs = 5
    d_mlp = 16
    n_freq = 10
    rng = np.random.default_rng(42)

    dominant_freq = rng.integers(0, n_freq, size=(n_epochs, d_mlp))
    max_frac = rng.uniform(0.0, 1.0, size=(n_epochs, d_mlp)).astype(np.float32)
    switch_counts = rng.integers(0, 5, size=d_mlp).astype(np.int32)
    commitment_epochs = np.array([100 * i if i < d_mlp - 2 else np.nan for i in range(d_mlp)])

    return {
        "epochs": np.array([0, 100, 200, 300, 400]),
        "dominant_freq": dominant_freq,
        "max_frac": max_frac,
        "switch_counts": switch_counts,
        "commitment_epochs": commitment_epochs,
        "threshold": np.array([0.06]),
    }


class TestRenderers:
    def test_trajectory_returns_figure(self, sample_cross_epoch_data: dict[str, Any]):
        fig = render_neuron_freq_trajectory(sample_cross_epoch_data, prime=101)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # type: ignore

    def test_trajectory_sorted(self, sample_cross_epoch_data: dict[str, Any]):
        fig = render_neuron_freq_trajectory(
            sample_cross_epoch_data, prime=101, sorted_by_final=True
        )
        assert isinstance(fig, go.Figure)

    def test_switch_distribution_returns_figure(self, sample_cross_epoch_data: dict[str, Any]):
        fig = render_switch_count_distribution(sample_cross_epoch_data, prime=101, seed=485)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # type: ignore

    def test_commitment_timeline_returns_figure(self, sample_cross_epoch_data: dict[str, Any]):
        fig = render_commitment_timeline(sample_cross_epoch_data, prime=101, seed=485)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # type: ignore


# ── Export registry test ─────────────────────────────────────────────


class TestExportRegistry:
    def test_neuron_dynamics_in_registry(self):
        from miscope.visualization.export import get_available_visualizations

        available = get_available_visualizations()
        assert "activations.mlp.neuron_freq_trajectory" in available
