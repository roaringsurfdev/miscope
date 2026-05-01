"""Tests for IntraGroupManifoldAnalyzer, characterize_surface, and renderers."""

import numpy as np

from miscope.analysis.analyzers.intragroup_manifold import (
    _build_group_members,
)
from miscope.analysis.library.shape import _SHAPE_TO_INT
from miscope.visualization.renderers.intragroup_manifold import (
    render_intragroup_manifold_summary,
    render_intragroup_manifold_surface_fit,
    render_intragroup_manifold_timeseries,
)

# ---------------------------------------------------------------------------
# Synthetic artifact fixture for analyzer and renderer tests
# ---------------------------------------------------------------------------


def _make_artifact(n_epochs: int = 4, n_groups: int = 3, seed: int = 42) -> dict:
    """Synthetic intragroup_manifold cross-epoch artifact."""
    shape_choices = list(_SHAPE_TO_INT.values())

    rng = np.random.default_rng(seed)
    r2_curvature = rng.uniform(0.0, 1.0, (n_epochs, n_groups)).astype(np.float32)
    a = rng.standard_normal((n_epochs, n_groups)).astype(np.float32)
    b = rng.standard_normal((n_epochs, n_groups)).astype(np.float32)
    c = rng.standard_normal((n_epochs, n_groups)).astype(np.float32)
    shape_int = rng.choice(shape_choices, (n_epochs, n_groups)).astype(np.int32)
    return {
        "group_freqs": np.array([3, 7, 12], dtype=np.int32)[:n_groups],
        "group_sizes": np.array([40, 35, 28], dtype=np.int32)[:n_groups],
        "epochs": np.arange(0, n_epochs * 500, 500, dtype=np.int32),
        "r2_linear": rng.uniform(0.0, 0.1, (n_epochs, n_groups)).astype(np.float32),
        "r2_quadratic": (
            r2_curvature + rng.uniform(0.0, 0.1, (n_epochs, n_groups)).astype(np.float32)
        ),
        "r2_curvature": r2_curvature,
        "a": a,
        "b": b,
        "c": c,
        "shape_int": shape_int,
    }


def _make_legacy_artifact(n_epochs: int = 4, n_groups: int = 3, seed: int = 42) -> dict:
    """Synthetic intragroup_manifold cross-epoch artifact."""
    rng = np.random.default_rng(seed)
    r2_curvature = rng.uniform(0.0, 1.0, (n_epochs, n_groups)).astype(np.float32)
    a = rng.standard_normal((n_epochs, n_groups)).astype(np.float32)
    b = rng.standard_normal((n_epochs, n_groups)).astype(np.float32)
    c = rng.standard_normal((n_epochs, n_groups)).astype(np.float32)
    shape_int = np.array([0, 1, 2], dtype=np.int32)[:n_groups]
    return {
        "group_freqs": np.array([3, 7, 12], dtype=np.int32)[:n_groups],
        "group_sizes": np.array([40, 35, 28], dtype=np.int32)[:n_groups],
        "epochs": np.arange(0, n_epochs * 500, 500, dtype=np.int32),
        "r2_linear": rng.uniform(0.0, 0.1, (n_epochs, n_groups)).astype(np.float32),
        "r2_quadratic": (
            r2_curvature + rng.uniform(0.0, 0.1, (n_epochs, n_groups)).astype(np.float32)
        ),
        "r2_curvature": r2_curvature,
        "a": a,
        "b": b,
        "c": c,
        "shape_int": shape_int,
    }

# ---------------------------------------------------------------------------
# Renderer smoke tests
# ---------------------------------------------------------------------------


def test_summary_renderer_returns_figure():
    """render_intragroup_manifold_summary produces a Figure."""
    import plotly.graph_objects as go

    fig = render_intragroup_manifold_summary(_make_artifact())
    assert isinstance(fig, go.Figure)


def test_timeseries_renderer_returns_figure():
    """render_intragroup_manifold_timeseries produces a Figure."""
    import plotly.graph_objects as go

    fig = render_intragroup_manifold_timeseries(_make_artifact())
    assert isinstance(fig, go.Figure)


def test_timeseries_renderer_with_epoch_cursor():
    """render_intragroup_manifold_timeseries accepts an epoch cursor."""
    import plotly.graph_objects as go

    artifact = _make_artifact()
    epoch = int(artifact["epochs"][1])
    fig = render_intragroup_manifold_timeseries(artifact, epoch=epoch)
    assert isinstance(fig, go.Figure)


def test_surface_fit_renderer_returns_figure():
    """render_intragroup_manifold_surface_fit produces a Figure."""
    import plotly.graph_objects as go

    fig = render_intragroup_manifold_surface_fit(_make_artifact(), group=0)
    assert isinstance(fig, go.Figure)


def test_surface_fit_renderer_group_kwarg():
    """render_intragroup_manifold_surface_fit accepts a group kwarg."""
    import plotly.graph_objects as go

    artifact = _make_artifact(n_groups=3)
    for g in range(3):
        fig = render_intragroup_manifold_surface_fit(artifact, group=g)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# _build_group_members
# ---------------------------------------------------------------------------


def test_build_group_members_correct_assignment():
    """_build_group_members recovers the expected member sets."""
    neuron_group_idx = np.array([0, 1, 0, 2, 1, -1], dtype=np.int32)
    members = _build_group_members(neuron_group_idx, n_groups=3)
    assert set(members[0].tolist()) == {0, 2}
    assert set(members[1].tolist()) == {1, 4}
    assert set(members[2].tolist()) == {3}
