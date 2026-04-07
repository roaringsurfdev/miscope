"""Tests for FreqGroupWeightGeometryAnalyzer and weight geometry renderers."""

import numpy as np
import pytest

from miscope.analysis.analyzers.freq_group_weight_geometry import (
    FreqGroupWeightGeometryAnalyzer,
    _build_group_labels,
    _compute_group_geometry,
)
from miscope.visualization.renderers.freq_group_weight_geometry import (
    render_weight_geometry_centroid_pca,
    render_weight_geometry_group_snapshot,
    render_weight_geometry_timeseries,
)


# --- _compute_group_geometry unit tests ---


def _two_group_weights(d_mlp=8, d=16, seed=0):
    """4 neurons in group 0, 4 in group 1, cleanly separated."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_mlp, d)).astype(np.float64)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    return W, labels


def test_compute_group_geometry_keys():
    """Returns all expected keys."""
    W, labels = _two_group_weights()
    result = _compute_group_geometry(W, labels, n_groups=2)
    expected_keys = {
        "centroids",
        "radii",
        "dimensionality",
        "center_spread",
        "mean_radius",
        "snr",
        "fisher_mean",
        "fisher_min",
        "circularity",
    }
    assert expected_keys == set(result.keys())


def test_compute_group_geometry_centroid_shape():
    """Centroids have shape (n_groups, d)."""
    W, labels = _two_group_weights(d_mlp=8, d=16)
    result = _compute_group_geometry(W, labels, n_groups=2)
    assert result["centroids"].shape == (2, 16)


def test_compute_group_geometry_radii_shape():
    """Radii have shape (n_groups,)."""
    W, labels = _two_group_weights()
    result = _compute_group_geometry(W, labels, n_groups=2)
    assert result["radii"].shape == (2,)


def test_compute_group_geometry_radii_nonnegative():
    """Radii are non-negative."""
    W, labels = _two_group_weights()
    result = _compute_group_geometry(W, labels, n_groups=2)
    assert np.all(result["radii"] >= 0.0)


def test_compute_group_geometry_dimensionality_positive():
    """Effective dimensionality is positive for groups with variation."""
    W, labels = _two_group_weights(d_mlp=8, d=16)
    result = _compute_group_geometry(W, labels, n_groups=2)
    assert np.all(result["dimensionality"] > 0.0)


def test_compute_group_geometry_circularity_range():
    """Circularity is in [0, 1]."""
    rng = np.random.default_rng(42)
    n_groups = 5
    W = rng.standard_normal((20, 32)).astype(np.float64)
    labels = np.repeat(np.arange(n_groups), 4).astype(np.int32)
    result = _compute_group_geometry(W, labels, n_groups=n_groups)
    assert 0.0 <= result["circularity"] <= 1.0


def test_compute_group_geometry_snr_zero_when_no_spread():
    """SNR=0 when all group centroids are at the same point (collapsed)."""
    d_mlp, d = 6, 4
    # Same weight vector for all neurons
    W = np.ones((d_mlp, d), dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    result = _compute_group_geometry(W, labels, n_groups=2)
    assert result["snr"] == pytest.approx(0.0, abs=1e-6)


def test_compute_group_geometry_excludes_ungrouped():
    """Neurons labeled -1 are excluded from computation."""
    rng = np.random.default_rng(10)
    W = rng.standard_normal((10, 16)).astype(np.float64)
    # 4 grouped, 6 ungrouped (-1)
    labels = np.array([0, 0, 1, 1, -1, -1, -1, -1, -1, -1], dtype=np.int32)
    # Should not raise and should only use the 4 grouped neurons
    result = _compute_group_geometry(W, labels, n_groups=2)
    assert result["centroids"].shape == (2, 16)


# --- _build_group_labels unit tests ---


def _make_loader(norm_matrix):
    """Minimal mock loader for _build_group_labels."""
    from unittest.mock import MagicMock

    loader = MagicMock()
    loader.load_epoch.return_value = {"norm_matrix": norm_matrix}
    return loader


def _make_norm(n_freq, d_mlp, assignments):
    norm = np.full((n_freq, d_mlp), 0.1 / max(n_freq - 1, 1))
    for n, f in enumerate(assignments):
        norm[:, n] = 0.0
        norm[f, n] = 1.0
    return norm.astype(np.float32)


def test_build_group_labels_basic():
    """Two groups of 4 neurons each → 2 groups with contiguous indices."""
    norm = _make_norm(4, 8, [0, 0, 0, 0, 2, 2, 2, 2])
    loader = _make_loader(norm)
    freqs, sizes, labels = _build_group_labels(loader, reference_epoch=0)
    assert len(freqs) == 2
    assert set(freqs) == {0, 2}
    assert sum(sizes) == 8


def test_build_group_labels_singleton_excluded():
    """Groups with only 1 neuron are excluded."""
    norm = _make_norm(4, 6, [0, 0, 0, 1, 2, 3])  # freq 1,2,3 have 1 neuron each
    loader = _make_loader(norm)
    freqs, sizes, labels = _build_group_labels(loader, reference_epoch=0)
    assert freqs == [0]
    assert sizes == [3]


def test_build_group_labels_contiguous_indices():
    """Group labels are contiguous 0..n_groups-1 (not sparse frequency indices)."""
    norm = _make_norm(10, 8, [0, 0, 5, 5, 9, 9, 0, 5])
    loader = _make_loader(norm)
    freqs, sizes, labels = _build_group_labels(loader, reference_epoch=0)
    grouped = labels[labels >= 0]
    assert set(grouped.tolist()) == set(range(len(freqs)))


def test_build_group_labels_ungrouped_are_negative_one():
    """Neurons in excluded singleton groups have label -1."""
    norm = _make_norm(4, 5, [0, 0, 0, 1, 2])  # freqs 1 and 2 singleton
    loader = _make_loader(norm)
    _, _, labels = _build_group_labels(loader, reference_epoch=0)
    assert np.sum(labels == -1) == 2  # neurons 3 and 4


# --- Analyzer integration tests ---


class _MockLoader:
    """Mock ArtifactLoader that serves both neuron_freq_norm and parameter_snapshot."""

    def __init__(self, norm_matrix, W_in_by_epoch, W_out_by_epoch=None):
        self._norm = norm_matrix
        self._W_in = W_in_by_epoch
        self._W_out = W_out_by_epoch

    def load_epoch(self, name: str, epoch: int):
        if name == "neuron_freq_norm":
            return {"norm_matrix": self._norm}
        if name == "parameter_snapshot":
            snap = {"W_in": self._W_in[epoch]}
            if self._W_out is not None:
                snap["W_out"] = self._W_out[epoch]
            return snap
        raise KeyError(name)


def _run_analyzer(loader, epochs):
    from unittest.mock import patch

    analyzer = FreqGroupWeightGeometryAnalyzer()
    with patch(
        "miscope.analysis.analyzers.freq_group_weight_geometry.ArtifactLoader",
        return_value=loader,
    ):
        return analyzer.analyze_across_epochs(artifacts_dir="/fake", epochs=epochs, context={})


def test_analyzer_output_keys_win_only():
    """All Win_* keys present when only W_in is in snapshot."""
    n_freq, d_mlp, d_model = 3, 6, 16
    norm = _make_norm(n_freq, d_mlp, [0, 0, 1, 1, 2, 2])
    epochs = [100, 200]
    rng = np.random.default_rng(0)
    W_in = {e: rng.standard_normal((d_model, d_mlp)).astype(np.float32) for e in epochs}
    result = _run_analyzer(_MockLoader(norm, W_in), epochs)

    win_keys = {
        "Win_centroids",
        "Win_radii",
        "Win_dimensionality",
        "Win_center_spread",
        "Win_mean_radius",
        "Win_snr",
        "Win_fisher_mean",
        "Win_fisher_min",
        "Win_circularity",
    }
    assert win_keys.issubset(result.keys())
    assert "epochs" in result
    assert "group_freqs" in result


def test_analyzer_output_keys_win_and_wout():
    """Both Win_* and Wout_* keys present when both weight matrices available."""
    n_freq, d_mlp, d_model, d_vocab = 3, 6, 16, 113
    norm = _make_norm(n_freq, d_mlp, [0, 0, 1, 1, 2, 2])
    epochs = [100, 200]
    rng = np.random.default_rng(1)
    W_in = {e: rng.standard_normal((d_model, d_mlp)).astype(np.float32) for e in epochs}
    W_out = {e: rng.standard_normal((d_mlp, d_vocab)).astype(np.float32) for e in epochs}
    result = _run_analyzer(_MockLoader(norm, W_in, W_out), epochs)

    wout_keys = {"Wout_centroids", "Wout_radii", "Wout_dimensionality"}
    assert wout_keys.issubset(result.keys())


def test_analyzer_win_centroid_shape():
    """Win_centroids has shape (n_epochs, n_groups, d_model)."""
    n_freq, d_mlp, d_model = 3, 6, 16
    norm = _make_norm(n_freq, d_mlp, [0, 0, 1, 1, 2, 2])
    epochs = [100, 200, 300]
    rng = np.random.default_rng(2)
    W_in = {e: rng.standard_normal((d_model, d_mlp)).astype(np.float32) for e in epochs}
    result = _run_analyzer(_MockLoader(norm, W_in), epochs)

    n_groups = len(result["group_freqs"])
    assert result["Win_centroids"].shape == (3, n_groups, d_model)


def test_analyzer_win_scalar_shapes():
    """Per-epoch global scalars have shape (n_epochs,)."""
    n_freq, d_mlp, d_model = 2, 6, 8
    norm = _make_norm(n_freq, d_mlp, [0, 0, 0, 1, 1, 1])
    epochs = [10, 20, 30, 40]
    rng = np.random.default_rng(3)
    W_in = {e: rng.standard_normal((d_model, d_mlp)).astype(np.float32) for e in epochs}
    result = _run_analyzer(_MockLoader(norm, W_in), epochs)

    for key in [
        "Win_center_spread",
        "Win_mean_radius",
        "Win_snr",
        "Win_fisher_mean",
        "Win_fisher_min",
        "Win_circularity",
    ]:
        assert result[key].shape == (4,), f"{key} shape mismatch"


def test_analyzer_win_snr_nonnegative():
    """SNR is non-negative at all epochs."""
    n_freq, d_mlp, d_model = 2, 6, 8
    norm = _make_norm(n_freq, d_mlp, [0, 0, 0, 1, 1, 1])
    epochs = [10, 20]
    rng = np.random.default_rng(4)
    W_in = {e: rng.standard_normal((d_model, d_mlp)).astype(np.float32) for e in epochs}
    result = _run_analyzer(_MockLoader(norm, W_in), epochs)
    assert np.all(result["Win_snr"] >= 0.0)


def test_analyzer_win_circularity_in_range():
    """Circularity is in [0, 1] at all epochs."""
    n_freq, d_mlp, d_model = 4, 8, 16
    norm = _make_norm(n_freq, d_mlp, [0, 0, 1, 1, 2, 2, 3, 3])
    epochs = [100, 500, 1000]
    rng = np.random.default_rng(5)
    W_in = {e: rng.standard_normal((d_model, d_mlp)).astype(np.float32) for e in epochs}
    result = _run_analyzer(_MockLoader(norm, W_in), epochs)
    circ = result["Win_circularity"]
    assert np.all(circ >= 0.0)
    assert np.all(circ <= 1.0 + 1e-5)


def test_analyzer_epochs_sorted():
    """Output epochs are sorted regardless of input order."""
    n_freq, d_mlp, d_model = 2, 4, 8
    norm = _make_norm(n_freq, d_mlp, [0, 0, 1, 1])
    epochs = [300, 100, 200]
    rng = np.random.default_rng(6)
    W_in = {e: rng.standard_normal((d_model, d_mlp)).astype(np.float32) for e in epochs}
    result = _run_analyzer(_MockLoader(norm, W_in), epochs)
    assert result["epochs"].tolist() == [100, 200, 300]


def test_analyzer_empty_result_when_no_valid_groups():
    """Returns empty arrays when no frequency group has >= 2 neurons."""
    n_freq, d_mlp, d_model = 4, 4, 8
    norm = _make_norm(n_freq, d_mlp, [0, 1, 2, 3])  # all singletons
    epochs = [100]
    W_in = {100: np.random.randn(d_model, d_mlp).astype(np.float32)}
    result = _run_analyzer(_MockLoader(norm, W_in), epochs)
    assert len(result["group_freqs"]) == 0
    assert result["Win_centroids"].shape[1] == 0


def test_analyzer_dtypes():
    """Key arrays have expected dtypes."""
    n_freq, d_mlp, d_model = 2, 6, 8
    norm = _make_norm(n_freq, d_mlp, [0, 0, 0, 1, 1, 1])
    epochs = [10, 20]
    rng = np.random.default_rng(7)
    W_in = {e: rng.standard_normal((d_model, d_mlp)).astype(np.float32) for e in epochs}
    result = _run_analyzer(_MockLoader(norm, W_in), epochs)
    assert result["group_freqs"].dtype == np.int32
    assert result["group_sizes"].dtype == np.int32
    assert result["epochs"].dtype == np.int32
    assert result["Win_centroids"].dtype == np.float32
    assert result["Win_snr"].dtype == np.float32


# --- Renderer smoke tests ---


def _make_artifact(n_epochs=5, n_groups=3, d_model=16, d_vocab=113):
    rng = np.random.default_rng(42)
    epochs = np.linspace(0, 5000, n_epochs, dtype=np.int32)
    group_freqs = np.arange(n_groups, dtype=np.int32) * 7

    def scalar(val=None):
        return (
            rng.uniform(0.1, 2.0, n_epochs).astype(np.float32)
            if val is None
            else np.full(n_epochs, val, dtype=np.float32)
        )

    return {
        "group_freqs": group_freqs,
        "group_sizes": np.full(n_groups, 4, dtype=np.int32),
        "Win_centroids": rng.standard_normal((n_epochs, n_groups, d_model)).astype(np.float32),
        "Win_radii": rng.uniform(0.1, 1.0, (n_epochs, n_groups)).astype(np.float32),
        "Win_dimensionality": rng.uniform(1.0, 8.0, (n_epochs, n_groups)).astype(np.float32),
        "Win_center_spread": scalar(),
        "Win_mean_radius": scalar(),
        "Win_snr": scalar(),
        "Win_fisher_mean": scalar(),
        "Win_fisher_min": scalar(),
        "Win_circularity": rng.uniform(0.0, 1.0, n_epochs).astype(np.float32),
        "Wout_centroids": rng.standard_normal((n_epochs, n_groups, d_vocab)).astype(np.float32),
        "Wout_radii": rng.uniform(0.1, 1.0, (n_epochs, n_groups)).astype(np.float32),
        "Wout_dimensionality": rng.uniform(1.0, 8.0, (n_epochs, n_groups)).astype(np.float32),
        "Wout_center_spread": scalar(),
        "Wout_mean_radius": scalar(),
        "Wout_snr": scalar(),
        "Wout_fisher_mean": scalar(),
        "Wout_fisher_min": scalar(),
        "Wout_circularity": rng.uniform(0.0, 1.0, n_epochs).astype(np.float32),
        "epochs": epochs,
    }


def test_render_timeseries_returns_figure_win():
    import plotly.graph_objects as go

    data = _make_artifact()
    fig = render_weight_geometry_timeseries(data, matrix="Win")
    assert isinstance(fig, go.Figure)


def test_render_timeseries_returns_figure_wout():
    import plotly.graph_objects as go

    data = _make_artifact()
    fig = render_weight_geometry_timeseries(data, matrix="Wout")
    assert isinstance(fig, go.Figure)


def test_render_timeseries_has_four_panels():
    """Four panels: SNR, spread/radius, circularity, Fisher."""
    data = _make_artifact()
    fig = render_weight_geometry_timeseries(data)
    # 4 subplots → 4 y-axes
    assert fig.layout.yaxis4 is not None  # pyright: ignore[reportAttributeAccessIssue]


def test_render_timeseries_with_epoch_cursor():
    import plotly.graph_objects as go

    data = _make_artifact()
    fig = render_weight_geometry_timeseries(data, epoch=2500)
    assert isinstance(fig, go.Figure)


def test_render_group_snapshot_returns_figure():
    import plotly.graph_objects as go

    data = _make_artifact()
    fig = render_weight_geometry_group_snapshot(data, epoch=2500, matrix="Win")
    assert isinstance(fig, go.Figure)


def test_render_group_snapshot_trace_count():
    """One bar per subplot (radii + dimensionality)."""
    data = _make_artifact(n_groups=4)
    fig = render_weight_geometry_group_snapshot(data)
    assert len(fig.data) == 2  # pyright: ignore[reportArgumentType]


def test_render_group_snapshot_wout():
    import plotly.graph_objects as go

    data = _make_artifact()
    fig = render_weight_geometry_group_snapshot(data, matrix="Wout")
    assert isinstance(fig, go.Figure)


def test_render_centroid_pca_returns_figure():
    import plotly.graph_objects as go

    data = _make_artifact()
    fig = render_weight_geometry_centroid_pca(data, matrix="Win")
    assert isinstance(fig, go.Figure)


def test_render_centroid_pca_wout():
    import plotly.graph_objects as go

    data = _make_artifact()
    fig = render_weight_geometry_centroid_pca(data, matrix="Wout")
    assert isinstance(fig, go.Figure)


def test_render_centroid_pca_with_epoch():
    import plotly.graph_objects as go

    data = _make_artifact()
    fig = render_weight_geometry_centroid_pca(data, epoch=2500, matrix="Win")
    assert isinstance(fig, go.Figure)


def test_render_centroid_pca_no_groups():
    """Zero groups renders a fallback figure without error."""
    import plotly.graph_objects as go

    n_epochs = 3
    data = {
        "group_freqs": np.array([], dtype=np.int32),
        "group_sizes": np.array([], dtype=np.int32),
        "Win_centroids": np.empty((n_epochs, 0, 16), dtype=np.float32),
        "epochs": np.array([0, 1000, 5000], dtype=np.int32),
    }
    fig = render_weight_geometry_centroid_pca(data)
    assert isinstance(fig, go.Figure)


def test_render_timeseries_empty_groups():
    """Zero groups renders without error."""
    import plotly.graph_objects as go

    n_epochs = 3
    data = {
        "group_freqs": np.array([], dtype=np.int32),
        "group_sizes": np.array([], dtype=np.int32),
        "Win_centroids": np.empty((n_epochs, 0, 16), dtype=np.float32),
        "Win_radii": np.empty((n_epochs, 0), dtype=np.float32),
        "Win_dimensionality": np.empty((n_epochs, 0), dtype=np.float32),
        "Win_center_spread": np.full(n_epochs, np.nan, dtype=np.float32),
        "Win_mean_radius": np.full(n_epochs, np.nan, dtype=np.float32),
        "Win_snr": np.full(n_epochs, np.nan, dtype=np.float32),
        "Win_fisher_mean": np.full(n_epochs, np.nan, dtype=np.float32),
        "Win_fisher_min": np.full(n_epochs, np.nan, dtype=np.float32),
        "Win_circularity": np.full(n_epochs, np.nan, dtype=np.float32),
        "epochs": np.array([0, 1000, 5000], dtype=np.int32),
    }
    fig = render_weight_geometry_timeseries(data)
    assert isinstance(fig, go.Figure)
