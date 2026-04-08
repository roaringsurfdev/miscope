"""Tests for NeuronGroupPCAAnalyzer and neuron_group_pca renderers."""

import numpy as np
import pytest

from miscope.analysis.analyzers.neuron_group_pca import (
    NeuronGroupPCAAnalyzer,
    _group_pca_stats,
)
from miscope.visualization.renderers.neuron_group_pca import (
    render_neuron_group_pca_cohesion,
    render_neuron_group_scatter,
    render_neuron_group_spread,
)

# --- _group_pca_stats unit tests ---


def test_group_pca_stats_perfectly_aligned():
    """All neurons pointing in same direction → PC1 var explained = 1.0."""
    direction = np.array([1.0, 0.0, 0.0])
    # n_group = 4
    scales = np.array([1.0, 2.0, 1.5, 0.8])
    group_W = np.outer(direction, scales)  # (3, 4)
    pc_var, _ = _group_pca_stats(group_W)
    assert pc_var[0] == pytest.approx(1.0, abs=1e-5)


def test_group_pca_stats_uniform_spread():
    """Neurons spread uniformly → PC1 var explained substantially less than 1."""
    rng = np.random.default_rng(42)
    d_model, n_group = 16, 8
    group_W = rng.standard_normal((d_model, n_group)).astype(np.float64)
    pc_var, _ = _group_pca_stats(group_W)
    assert pc_var[0] < 0.8


def test_group_pca_stats_returns_three_components():
    """Output pc_var has shape (3,)."""
    group_W = np.random.randn(16, 6).astype(np.float32)
    pc_var, _ = _group_pca_stats(group_W)
    assert pc_var.shape == (3,)


def test_group_pca_stats_cumulative_le_one():
    """Cumulative PC1+PC2+PC3 <= 1.0 (fractions of total variance)."""
    group_W = np.random.randn(16, 8).astype(np.float32)
    pc_var, _ = _group_pca_stats(group_W)
    assert float(np.nansum(pc_var)) <= 1.0 + 1e-5


def test_group_pca_stats_two_neuron_group():
    """Group with exactly 2 neurons: PC1 = 1.0 (rank-1 after centering), PC2 = 0, PC3 = NaN."""
    group_W = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    pc_var, _ = _group_pca_stats(group_W)
    assert pc_var[0] == pytest.approx(1.0, abs=1e-5)
    assert pc_var[1] == pytest.approx(0.0, abs=1e-5)  # zero singular value, not NaN
    assert np.isnan(pc_var[2])  # SVD only produces min(d,n)=2 values; PC3 slot stays NaN


def test_group_pca_stats_spread_zero_for_identical_neurons():
    """Identical weight vectors → spread = 0."""
    vec = np.array([1.0, 2.0, 3.0])
    group_W = np.tile(vec[:, None], (1, 5))  # (3, 5) all identical
    _, spread = _group_pca_stats(group_W)
    assert spread == pytest.approx(0.0, abs=1e-6)


def test_group_pca_stats_spread_positive_for_different_neurons():
    """Different weight vectors → positive spread."""
    group_W = np.array([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
    _, spread = _group_pca_stats(group_W)
    assert spread > 0.0


def test_group_pca_stats_output_types():
    """Returns (ndarray, float)."""
    group_W = np.random.randn(8, 4).astype(np.float32)
    pc_var, spread = _group_pca_stats(group_W)
    assert isinstance(pc_var, np.ndarray)
    assert isinstance(spread, float)


def test_group_pca_stats_minimum_group_size():
    """Two neurons: spread is valid, PC2/PC3 are NaN."""
    group_W = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    pc_var, spread = _group_pca_stats(group_W)
    assert 0.0 <= pc_var[0] <= 1.0
    assert spread >= 0.0


# --- Analyzer integration tests (mock artifact loader) ---


class _MockArtifactLoader:
    """Minimal mock for ArtifactLoader that serves synthetic data."""

    def __init__(self, norm_matrix, W_in_by_epoch):
        self._norm = norm_matrix
        self._W_in = W_in_by_epoch  # dict epoch -> W_in array

    def load_epoch(self, name: str, epoch: int):
        if name == "neuron_freq_norm":
            return {"norm_matrix": self._norm}
        if name == "parameter_snapshot":
            # Include a dummy W_E so extract_neuron_weight_matrix recognises
            # the transformer convention: W_in shape is (d_model, d_mlp).
            W_in = self._W_in[epoch]
            dummy_W_E = np.zeros((1, W_in.shape[0]), dtype=np.float32)
            return {"W_in": W_in, "W_E": dummy_W_E}
        raise KeyError(name)


def _make_norm_matrix(n_freq: int, d_mlp: int, assignments: list[int]) -> np.ndarray:
    """Build a norm_matrix where neuron n has dominant frequency assignments[n]."""
    norm = np.full((n_freq, d_mlp), 0.1 / (n_freq - 1))
    for n, f in enumerate(assignments):
        norm[:, n] = 0.1 / (n_freq - 1)
        norm[f, n] = 0.9
    return norm.astype(np.float32)


def _run_analyzer(loader, epochs):
    """Run analyzer using mock loader (bypasses filesystem)."""
    from unittest.mock import patch

    analyzer = NeuronGroupPCAAnalyzer()
    with patch(
        "miscope.analysis.analyzers.neuron_group_pca.ArtifactLoader",
        return_value=loader,
    ):
        return analyzer.analyze_across_epochs(artifacts_dir="/fake", epochs=epochs, context={})


def test_analyzer_output_shapes():
    """Output arrays have correct shapes for 2 groups over 3 epochs."""
    n_freq, d_mlp = 4, 8
    # 4 neurons per freq group → 2 groups with size 4 (freqs 0 and 1)
    assignments = [0, 0, 0, 0, 1, 1, 1, 1]
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [100, 200, 300]
    rng = np.random.default_rng(0)
    W_in_by_epoch = {e: rng.standard_normal((16, d_mlp)).astype(np.float32) for e in epochs}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    n_groups = result["group_freqs"].shape[0]
    assert n_groups == 2
    assert result["pc_var"].shape == (3, 2, 3)
    assert result["mean_spread"].shape == (3, 2)
    assert result["epochs"].shape == (3,)
    assert result["group_sizes"].shape == (2,)


def test_analyzer_output_dtypes():
    """Output arrays have expected dtypes."""
    n_freq, d_mlp = 3, 6
    assignments = [0, 0, 1, 1, 2, 2]
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [10, 20]
    W_in_by_epoch = {e: np.random.randn(8, d_mlp).astype(np.float32) for e in epochs}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    assert result["group_freqs"].dtype == np.int32
    assert result["group_sizes"].dtype == np.int32
    assert result["pc_var"].dtype == np.float32
    assert result["mean_spread"].dtype == np.float32
    assert result["epochs"].dtype == np.int32


def test_analyzer_group_assignment_correctness():
    """Group frequency indices match actual dominant frequencies."""
    n_freq, d_mlp = 4, 8
    # Only freqs 0 and 3 have 2+ neurons; freqs 1 and 2 have 0
    assignments = [0, 0, 0, 3, 3, 3, 0, 3]
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [100]
    W_in_by_epoch = {100: np.random.randn(16, d_mlp).astype(np.float32)}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    assert set(result["group_freqs"].tolist()) == {0, 3}


def test_analyzer_excludes_singleton_groups():
    """Groups with only 1 neuron are excluded (PCA requires >= 2)."""
    n_freq, d_mlp = 4, 5
    # freq 0: 3 neurons, freq 1: 1 neuron (excluded), freq 2: 1 neuron (excluded)
    assignments = [0, 0, 0, 1, 2]
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [100]
    W_in_by_epoch = {100: np.random.randn(16, d_mlp).astype(np.float32)}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    assert result["group_freqs"].tolist() == [0]
    assert result["group_sizes"].tolist() == [3]


def test_analyzer_pc_var_range():
    """All pc_var fractions are in [0, 1] and cumulative sum <= 1."""
    n_freq, d_mlp = 3, 9
    assignments = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [100, 500, 1000]
    rng = np.random.default_rng(7)
    W_in_by_epoch = {e: rng.standard_normal((32, d_mlp)).astype(np.float32) for e in epochs}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    pc_var = result["pc_var"]
    valid = ~np.isnan(pc_var)
    assert np.all(pc_var[valid] >= 0.0)
    assert np.all(pc_var[valid] <= 1.0 + 1e-5)
    cumulative = np.nansum(pc_var, axis=2)  # (n_epochs, n_groups)
    assert np.all(cumulative <= 1.0 + 1e-5)


def test_analyzer_spread_nonnegative():
    """Mean spread is non-negative for all epochs and groups."""
    n_freq, d_mlp = 2, 6
    assignments = [0, 0, 0, 1, 1, 1]
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [100, 200]
    W_in_by_epoch = {e: np.random.randn(16, d_mlp).astype(np.float32) for e in epochs}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    assert np.all(result["mean_spread"] >= 0.0)


def test_analyzer_empty_when_no_valid_groups():
    """Returns empty arrays when no frequency has >= 2 neurons."""
    n_freq, d_mlp = 4, 4
    assignments = [0, 1, 2, 3]  # each freq has exactly 1 neuron
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [100]
    W_in_by_epoch = {100: np.random.randn(8, d_mlp).astype(np.float32)}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    assert len(result["group_freqs"]) == 0
    assert result["pc_var"].shape == (1, 0, 3)
    assert result["mean_spread"].shape == (1, 0)


def test_analyzer_group_bases_shape():
    """group_bases has shape (n_groups, 3, d_model)."""
    n_freq, d_mlp, d_model = 3, 6, 16
    assignments = [0, 0, 1, 1, 2, 2]
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [10, 20]
    W_in_by_epoch = {e: np.random.randn(d_model, d_mlp).astype(np.float32) for e in epochs}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    assert result["group_bases"].shape == (3, 3, d_model)
    assert result["group_bases"].dtype == np.float32


def test_analyzer_group_bases_orthonormal():
    """Each group's basis rows are orthonormal (within float32 tolerance)."""
    n_freq, d_mlp, d_model = 2, 6, 32
    assignments = [0, 0, 0, 1, 1, 1]
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [100]
    rng = np.random.default_rng(5)
    W_in_by_epoch = {100: rng.standard_normal((d_model, d_mlp)).astype(np.float32)}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    bases = result["group_bases"].astype(np.float64)
    for g_idx in range(2):
        gram = bases[g_idx] @ bases[g_idx].T  # (3, 3)
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-4)


def test_analyzer_empty_group_bases_shape():
    """Empty result has group_bases shape (0, 3, 0)."""
    n_freq, d_mlp = 4, 4
    assignments = [0, 1, 2, 3]
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [100]
    W_in_by_epoch = {100: np.random.randn(8, d_mlp).astype(np.float32)}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    assert result["group_bases"].shape == (0, 3, 0)


def test_analyzer_epochs_sorted():
    """Epochs in output are sorted regardless of input order."""
    n_freq, d_mlp = 2, 4
    assignments = [0, 0, 1, 1]
    norm = _make_norm_matrix(n_freq, d_mlp, assignments)
    epochs = [300, 100, 200]  # unsorted
    W_in_by_epoch = {e: np.random.randn(8, d_mlp).astype(np.float32) for e in epochs}

    result = _run_analyzer(_MockArtifactLoader(norm, W_in_by_epoch), epochs)

    assert result["epochs"].tolist() == [100, 200, 300]


# --- Renderer smoke tests ---


def _make_cross_epoch_artifact(n_epochs=5, n_groups=3):
    rng = np.random.default_rng(42)
    # pc_var: 3 components per group, sum <= 1.0 per (epoch, group)
    raw = rng.uniform(0.1, 0.4, (n_epochs, n_groups, 3)).astype(np.float32)
    pc_var = raw / raw.sum(axis=2, keepdims=True)  # normalize so sum == 1
    pc_var *= rng.uniform(0.5, 0.9, (n_epochs, n_groups, 1)).astype(np.float32)
    return {
        "group_freqs": np.arange(n_groups, dtype=np.int32) * 5,
        "group_sizes": np.full(n_groups, 4, dtype=np.int32),
        "pc_var": pc_var,
        "mean_spread": rng.uniform(0.1, 2.0, (n_epochs, n_groups)).astype(np.float32),
        "epochs": np.linspace(0, 5000, n_epochs, dtype=np.int32),
    }


def test_render_cohesion_returns_figure():
    import plotly.graph_objects as go

    data = _make_cross_epoch_artifact()
    fig = render_neuron_group_pca_cohesion(data)
    assert isinstance(fig, go.Figure)


def test_render_cohesion_trace_count():
    """Two traces per group (solid cumulative + dashed PC1)."""
    data = _make_cross_epoch_artifact(n_groups=3)
    fig = render_neuron_group_pca_cohesion(data)
    assert len(fig.data) == 6  # pyright: ignore[reportArgumentType]


def test_render_cohesion_with_epoch_cursor():
    """Epoch cursor adds a shape/vline without error."""
    import plotly.graph_objects as go

    data = _make_cross_epoch_artifact()
    fig = render_neuron_group_pca_cohesion(data, epoch=2500)
    assert isinstance(fig, go.Figure)


def test_render_spread_returns_figure():
    import plotly.graph_objects as go

    data = _make_cross_epoch_artifact()
    fig = render_neuron_group_spread(data)
    assert isinstance(fig, go.Figure)


def test_render_spread_trace_count():
    """One trace per group."""
    data = _make_cross_epoch_artifact(n_groups=4)
    fig = render_neuron_group_spread(data)
    assert len(fig.data) == 4  # pyright: ignore[reportArgumentType]


def test_render_empty_data():
    """Renderers handle zero-group artifact without error."""
    import plotly.graph_objects as go

    data = {
        "group_freqs": np.array([], dtype=np.int32),
        "group_sizes": np.array([], dtype=np.int32),
        "pc_var": np.empty((5, 0, 3), dtype=np.float32),
        "mean_spread": np.empty((5, 0), dtype=np.float32),
        "epochs": np.array([0, 1000, 2000, 3000, 4000], dtype=np.int32),
    }
    fig1 = render_neuron_group_pca_cohesion(data)
    fig2 = render_neuron_group_spread(data)
    assert isinstance(fig1, go.Figure)
    assert isinstance(fig2, go.Figure)
    assert len(fig1.data) == 0  # pyright: ignore[reportArgumentType]
    assert len(fig2.data) == 0  # pyright: ignore[reportArgumentType]


# --- render_neuron_group_scatter tests ---


def _make_scatter_data(n_groups=3, d_model=16, d_mlp=12):
    rng = np.random.default_rng(99)
    n_freq = n_groups * 2
    group_freqs = np.arange(n_groups, dtype=np.int32) * 2
    # Build orthonormal bases via QR
    bases = np.zeros((n_groups, 3, d_model), dtype=np.float32)
    for g in range(n_groups):
        Q, _ = np.linalg.qr(rng.standard_normal((d_model, 3)))
        bases[g] = Q[:, :3].T.astype(np.float32)
    # Assign neurons evenly across groups
    neurons_per_group = d_mlp // n_groups
    assignments = []
    for g in range(n_groups):
        assignments.extend([int(group_freqs[g])] * neurons_per_group)
    assignments.extend([0] * (d_mlp - len(assignments)))
    norm_matrix = np.full((n_freq, d_mlp), 0.1 / (n_freq - 1), dtype=np.float32)
    for n, f in enumerate(assignments):
        norm_matrix[:, n] = 0.1 / (n_freq - 1)
        norm_matrix[f, n] = 0.9
    return {
        "group_bases": bases,
        "group_freqs": group_freqs,
        "W_in": rng.standard_normal((d_model, d_mlp)).astype(np.float32),
        "norm_matrix": norm_matrix,
    }


def test_render_scatter_returns_figure():
    import plotly.graph_objects as go

    data = _make_scatter_data()
    fig = render_neuron_group_scatter(data)
    assert isinstance(fig, go.Figure)


def test_render_scatter_trace_count():
    """One trace per group."""
    data = _make_scatter_data(n_groups=4)
    fig = render_neuron_group_scatter(data)
    assert len(fig.data) == 4  # pyright: ignore[reportArgumentType]


def test_render_scatter_with_epoch():
    """Epoch label included in title without error."""
    import plotly.graph_objects as go

    data = _make_scatter_data()
    fig = render_neuron_group_scatter(data, epoch=5000)
    assert isinstance(fig, go.Figure)
    # assert "5000" in fig.layout.title.text


def test_render_scatter_empty_groups():
    """Zero groups → empty figure."""
    import plotly.graph_objects as go

    data = {
        "group_bases": np.empty((0, 3, 16), dtype=np.float32),
        "group_freqs": np.array([], dtype=np.int32),
        "W_in": np.random.randn(16, 8).astype(np.float32),
        "norm_matrix": np.random.rand(4, 8).astype(np.float32),
    }
    fig = render_neuron_group_scatter(data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0  # pyright: ignore[reportArgumentType]
