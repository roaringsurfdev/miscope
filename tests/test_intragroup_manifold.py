"""Tests for IntraGroupManifoldAnalyzer, fit_quadratic_surface, and renderers."""

import numpy as np

from miscope.analysis.analyzers.intragroup_manifold import (
    _build_group_members,
    _compute_final_shapes,
    decode_shapes,
)
from miscope.analysis.library.manifold_geometry import (
    _MIN_NEURONS,
    fit_quadratic_surface,
)
from miscope.visualization.renderers.intragroup_manifold import (
    render_intragroup_manifold_summary,
    render_intragroup_manifold_surface_fit,
    render_intragroup_manifold_timeseries,
)

# ---------------------------------------------------------------------------
# fit_quadratic_surface unit tests
# ---------------------------------------------------------------------------


def _make_saddle(n: int = 20, noise: float = 0.05) -> np.ndarray:
    """Synthetic saddle: PC3 = PC1² - PC2² plus small noise."""
    rng = np.random.default_rng(0)
    pc1 = rng.uniform(-2, 2, n)
    pc2 = rng.uniform(-2, 2, n)
    pc3 = pc1**2 - pc2**2 + rng.normal(0, noise, n)
    return np.column_stack([pc1, pc2, pc3])


def _make_bowl(n: int = 20, noise: float = 0.05) -> np.ndarray:
    """Synthetic bowl: PC3 = PC1² + PC2²."""
    rng = np.random.default_rng(1)
    pc1 = rng.uniform(-2, 2, n)
    pc2 = rng.uniform(-2, 2, n)
    pc3 = pc1**2 + pc2**2 + rng.normal(0, noise, n)
    return np.column_stack([pc1, pc2, pc3])


def _make_flat(n: int = 20) -> np.ndarray:
    """Synthetic flat blob: PC3 is constant — zero curvature by construction."""
    rng = np.random.default_rng(2)
    pc1 = rng.uniform(-2, 2, n)
    pc2 = rng.uniform(-2, 2, n)
    pc3 = np.zeros(n)
    return np.column_stack([pc1, pc2, pc3])


def test_fit_returns_expected_keys():
    """fit_quadratic_surface returns all required keys."""
    result = fit_quadratic_surface(_make_saddle())
    expected = {"r2_linear", "r2_quadratic", "r2_curvature", "a", "b", "c", "shape"}
    assert expected == set(result.keys())


def test_saddle_classified_correctly():
    """A clear saddle surface gets classified as 'saddle'."""
    result = fit_quadratic_surface(_make_saddle(n=50, noise=0.01))
    assert result["shape"] == "saddle"


def test_bowl_classified_correctly():
    """A clear bowl surface gets classified as 'bowl'."""
    result = fit_quadratic_surface(_make_bowl(n=50, noise=0.01))
    assert result["shape"] == "bowl"


def test_flat_classified_correctly():
    """Constant PC3 (zero curvature by construction) gets classified as 'flat/blob'."""
    result = fit_quadratic_surface(_make_flat(n=50))
    assert result["shape"] == "flat/blob"


def test_r2_curvature_nonnegative():
    """R²_curvature is non-negative (clipped at 0)."""
    for proj in [_make_saddle(), _make_bowl(), _make_flat()]:
        result = fit_quadratic_surface(proj)
        assert result["r2_curvature"] >= 0.0 # type: ignore


def test_r2_range():
    """R² values are within [0, 1]."""
    result = fit_quadratic_surface(_make_saddle(n=50, noise=0.01))
    for key in ("r2_linear", "r2_quadratic", "r2_curvature"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of [0, 1]: {result[key]}" # type: ignore


def test_r2_quadratic_ge_linear():
    """R²_quadratic >= R²_linear (quadratic fit cannot be worse)."""
    result = fit_quadratic_surface(_make_saddle(n=50, noise=0.01))
    assert result["r2_quadratic"] >= result["r2_linear"] - 1e-9 # type: ignore


def test_saddle_r2_curvature_high():
    """A clean saddle has high R²_curvature (well above the flat threshold)."""
    result = fit_quadratic_surface(_make_saddle(n=50, noise=0.01))
    assert result["r2_curvature"] > 0.5 # type: ignore


def test_too_few_neurons_returns_nan():
    """Groups with fewer than _MIN_NEURONS neurons return NaN values."""
    proj = np.random.default_rng(0).standard_normal((_MIN_NEURONS - 1, 3))
    result = fit_quadratic_surface(proj)
    assert np.isnan(result["r2_linear"])
    assert np.isnan(result["r2_curvature"])
    assert result["shape"] == "flat/blob"


# ---------------------------------------------------------------------------
# decode_shapes and _compute_final_shapes
# ---------------------------------------------------------------------------


def test_decode_shapes_roundtrip():
    """decode_shapes is inverse of the _SHAPE_TO_INT mapping."""
    shape_int = np.array([0, 1, 2, 0, 2], dtype=np.int32)
    shapes = decode_shapes(shape_int)
    expected = ["flat/blob", "bowl", "saddle", "flat/blob", "saddle"]
    assert shapes == expected


def test_compute_final_shapes_flat():
    """Groups with low R²_curvature at final epoch get flat/blob shape."""
    n_epochs, n_groups = 5, 3
    r2_curvature = np.full((n_epochs, n_groups), 0.02, dtype=np.float32)
    a = np.ones((n_epochs, n_groups), dtype=np.float32)
    b = np.ones((n_epochs, n_groups), dtype=np.float32)
    c = np.zeros((n_epochs, n_groups), dtype=np.float32)
    shape_int = _compute_final_shapes(r2_curvature, a, b, c, n_groups)
    shapes = decode_shapes(shape_int)
    assert all(s == "flat/blob" for s in shapes)


def test_compute_final_shapes_saddle():
    """Groups with high R²_curvature and opposite-sign a, b → saddle."""
    n_epochs, n_groups = 5, 2
    r2_curvature = np.full((n_epochs, n_groups), 0.8, dtype=np.float32)
    a = np.full((n_epochs, n_groups), 1.0, dtype=np.float32)
    b = np.full((n_epochs, n_groups), -1.0, dtype=np.float32)
    c = np.zeros((n_epochs, n_groups), dtype=np.float32)
    shape_int = _compute_final_shapes(r2_curvature, a, b, c, n_groups)
    shapes = decode_shapes(shape_int)
    assert all(s == "saddle" for s in shapes)


def test_compute_final_shapes_rotated_saddle():
    """A rotated saddle (a > 0, b > 0, c large) is correctly classified as saddle.

    With a=0.5, b=0.5, c=2.0: det(H) = 4ab - c² = 1.0 - 4.0 = -3.0 < 0 → saddle.
    The old axis-aligned check (a > 0) == (b > 0) would have mis-classified this as bowl.
    """
    n_epochs, n_groups = 5, 1
    r2_curvature = np.full((n_epochs, n_groups), 0.8, dtype=np.float32)
    a = np.full((n_epochs, n_groups), 0.5, dtype=np.float32)
    b = np.full((n_epochs, n_groups), 0.5, dtype=np.float32)
    c = np.full((n_epochs, n_groups), 2.0, dtype=np.float32)
    shape_int = _compute_final_shapes(r2_curvature, a, b, c, n_groups)
    shapes = decode_shapes(shape_int)
    assert shapes == ["saddle"]


# ---------------------------------------------------------------------------
# Synthetic artifact fixture for analyzer and renderer tests
# ---------------------------------------------------------------------------


def _make_artifact(n_epochs: int = 4, n_groups: int = 3, seed: int = 42) -> dict:
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
