"""Intra-Group Manifold Geometry renderers.

Three views into the quadratic surface structure of frequency groups in weight-space
PCA coordinates:

- summary:     bar chart of final-epoch R²_curvature per group, colored by shape label
- timeseries:  R²_curvature trajectory per group over all training epochs
- surface_fit: 3D scatter of group neurons at a selected epoch with the fitted
               quadratic surface overlaid
"""

from __future__ import annotations

import colorsys

import numpy as np
import plotly.graph_objects as go

from miscope.analysis.analyzers.intragroup_manifold import decode_shapes

_SHAPE_COLORS = {
    "saddle": "steelblue",
    "bowl": "darkorange",
    "flat/blob": "lightgray",
}


def _freq_color(freq_idx: int, n_freq: int) -> str:
    """Consistent HSL color for frequency index — matches neuron_group_pca convention."""
    hue = freq_idx / max(n_freq, 1)
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.5)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def render_intragroup_manifold_summary(
    data: dict,
    epoch: int | None = None,
    height: int = 450,
    **kwargs,
) -> go.Figure:
    """Bar chart of final-epoch R²_curvature per frequency group, colored by shape.

    Args:
        data: intragroup_manifold cross-epoch artifact
        epoch: unused (final-epoch summary)
        height: figure height in pixels
    """
    group_freqs = data["group_freqs"]
    r2_curvature = data["r2_curvature"]  # (n_epochs, n_groups)
    shape_int = data["shape_int"]        # (n_groups,)

    shapes = decode_shapes(shape_int)
    final_r2c = r2_curvature[-1, :]     # (n_groups,)

    freq_labels = [str(int(f) + 1) for f in group_freqs]
    colors = [_SHAPE_COLORS.get(s, "lightgray") for s in shapes]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=freq_labels,
            y=final_r2c,
            marker_color=colors,
            text=[f"{v:.3f}" for v in final_r2c],
            textposition="outside",
            customdata=shapes,
            hovertemplate="freq %{x}<br>R²_curvature: %{y:.4f}<br>shape: %{customdata}<extra></extra>",
        )
    )

    _add_shape_legend(fig)

    fig.update_layout(
        title="Intra-Group Manifold Geometry — Final Epoch R²_curvature",
        xaxis_title="Frequency Group",
        yaxis_title="R²_curvature",
        yaxis_range=[0, 1.05],
        height=height,
        showlegend=True,
    )
    return fig


def render_intragroup_manifold_timeseries(
    data: dict,
    epoch: int | None = None,
    height: int = 500,
    **kwargs,
) -> go.Figure:
    """R²_curvature trajectory per frequency group over training epochs.

    Each line is one frequency group.  A vertical cursor marks the selected epoch.
    Reveals whether manifold formation is a gradual ramp or a sharp transition.

    Args:
        data: intragroup_manifold cross-epoch artifact
        epoch: optional epoch cursor (vertical line)
        height: figure height in pixels
    """
    group_freqs = data["group_freqs"]
    epochs = data["epochs"]
    r2_curvature = data["r2_curvature"]  # (n_epochs, n_groups)
    shape_int = data["shape_int"]

    shapes = decode_shapes(shape_int)
    n_groups = len(group_freqs)
    n_freq = int(group_freqs.max()) + 1 if n_groups > 0 else 1

    fig = go.Figure()

    for g_idx in range(n_groups):
        freq = int(group_freqs[g_idx])
        color = _freq_color(freq, n_freq)
        shape = shapes[g_idx]
        dash = "dot" if shape == "flat/blob" else ("dash" if shape == "bowl" else "solid")

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=r2_curvature[:, g_idx],
                mode="lines",
                name=f"freq {freq + 1} ({shape})",
                line=dict(color=color, dash=dash, width=2),
                hovertemplate=f"freq {freq + 1}<br>epoch: %{{x}}<br>R²_curvature: %{{y:.4f}}<extra></extra>",
            )
        )

    if epoch is not None:
        fig.add_vline(x=epoch, line=dict(color="black", width=1, dash="dot"))

    fig.update_layout(
        title="Intra-Group Manifold Geometry — R²_curvature over Training",
        xaxis_title="Epoch",
        yaxis_title="R²_curvature",
        yaxis_range=[-0.02, 1.02],
        height=height,
    )
    return fig


def render_intragroup_manifold_surface_fit(
    data: dict,
    epoch: int | None = None,
    group: int = 0,
    height: int = 600,
    n_surface_pts: int = 30,
    **kwargs,
) -> go.Figure:
    """3D scatter of group neurons with the fitted quadratic surface overlaid.

    The surface is reconstructed from the stored a, b, c, and the linear fit
    re-derived from the stored projections at the selected epoch.

    Args:
        data: intragroup_manifold cross-epoch artifact
        epoch: training epoch to visualise (uses final epoch if None)
        group: group index (0-indexed, refers to groups in group_freqs order)
        height: figure height in pixels
        n_surface_pts: grid resolution for the surface mesh
    """
    group_freqs = data["group_freqs"]
    epochs = data["epochs"]
    r2_curvature = data["r2_curvature"]
    a_arr = data["a"]
    b_arr = data["b"]
    c_arr = data["c"]

    ep_idx = _resolve_epoch_idx(epoch, epochs)
    freq = int(group_freqs[group])

    # projections are not stored in this artifact — surface is reconstructed from coefficients
    r2c = float(r2_curvature[ep_idx, group])
    a = float(a_arr[ep_idx, group])
    b = float(b_arr[ep_idx, group])
    c = float(c_arr[ep_idx, group])

    # Build the surface from coefficients on a symmetric grid
    grid = np.linspace(-2.0, 2.0, n_surface_pts)
    pc1_grid, pc2_grid = np.meshgrid(grid, grid)
    pc3_surface = a * pc1_grid**2 + b * pc2_grid**2 + c * pc1_grid * pc2_grid

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=pc1_grid,
            y=pc2_grid,
            z=pc3_surface,
            colorscale="Blues",
            opacity=0.5,
            showscale=False,
            name="quadratic surface",
        )
    )

    from miscope.analysis.analyzers.intragroup_manifold import _INT_TO_SHAPE
    shape = _INT_TO_SHAPE.get(int(data["shape_int"][group]), "flat/blob")
    actual_epoch = int(epochs[ep_idx])

    fig.update_layout(
        title=(
            f"Manifold Surface Fit — freq {freq + 1} ({shape})<br>"
            f"epoch {actual_epoch}  R²_curvature={r2c:.3f}"
        ),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_shape_legend(fig: go.Figure) -> None:
    """Add invisible scatter traces to provide a shape-color legend."""
    for shape, color in _SHAPE_COLORS.items():
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                marker_color=color,
                name=shape,
                showlegend=True,
            )
        )


def _resolve_epoch_idx(epoch: int | None, epochs: np.ndarray) -> int:
    """Return the index into epochs for the requested epoch, or -1 for final."""
    if epoch is None:
        return len(epochs) - 1
    matches = np.where(epochs == epoch)[0]
    return int(matches[0]) if len(matches) > 0 else len(epochs) - 1
