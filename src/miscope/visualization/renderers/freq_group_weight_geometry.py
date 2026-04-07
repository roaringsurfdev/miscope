"""Frequency Group Weight Geometry renderers.

Two views into geometric separation of frequency groups in weight space:
- timeseries: multi-panel evolution of SNR, spread/radius, circularity,
  and Fisher discriminant for W_in or W_out across training epochs.
- group_snapshot: per-group bar chart of radii and dimensionality at a
  selected epoch, showing which groups are compact vs. diffuse.
"""

import colorsys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_MATRIX_LABELS = {
    "Win": "W_in",
    "Wout": "W_out",
}


def _freq_color(group_idx: int, n_groups: int) -> str:
    """Consistent HSL color for group index."""
    hue = group_idx / max(n_groups, 1)
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.5)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def render_weight_geometry_timeseries(
    data: dict,
    epoch: int | None = None,
    matrix: str = "Win",
    height: int | None = None,
) -> go.Figure:
    """Multi-panel time-series of geometric measures for frequency groups in weight space.

    Panels: SNR, center spread & mean radius, circularity, Fisher discriminant.
    Both W_in and W_out can be selected via the matrix kwarg.

    Args:
        data: cross_epoch artifact from freq_group_weight_geometry
        epoch: optional epoch cursor (vertical line)
        matrix: "Win" or "Wout"
        height: total figure height in pixels; auto-sized if None
    """
    epochs = data["epochs"]
    prefix = matrix
    label = _MATRIX_LABELS.get(matrix, matrix)

    if height is None:
        height = 900

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f"SNR — {label} (group centroid spread² / mean radius²)",
            f"Center Spread & Mean Radius — {label}",
            f"Circularity — {label} (group centroids in top-2 PCA)",
            f"Fisher Discriminant — {label} (between-group separation)",
        ],
    )

    color = "steelblue"

    snr_key = f"{prefix}_snr"
    if snr_key in data:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=data[snr_key],
                mode="lines",
                name="SNR",
                line=dict(color=color, width=2),
                hovertemplate="Epoch %{x}<br>SNR: %{y:.3f}<extra></extra>",
            ),
            row=1, col=1,
        )

    spread_key = f"{prefix}_center_spread"
    radius_key = f"{prefix}_mean_radius"
    if spread_key in data:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=data[spread_key],
                mode="lines",
                name="Center spread",
                line=dict(color=color, width=2),
                hovertemplate="Epoch %{x}<br>Center spread: %{y:.4f}<extra></extra>",
            ),
            row=2, col=1,
        )
    if radius_key in data:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=data[radius_key],
                mode="lines",
                name="Mean radius",
                line=dict(color=color, width=2, dash="dash"),
                hovertemplate="Epoch %{x}<br>Mean radius: %{y:.4f}<extra></extra>",
            ),
            row=2, col=1,
        )

    circ_key = f"{prefix}_circularity"
    if circ_key in data:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=data[circ_key],
                mode="lines",
                name="Circularity",
                line=dict(color=color, width=2),
                hovertemplate="Epoch %{x}<br>Circularity: %{y:.3f}<extra></extra>",
            ),
            row=3, col=1,
        )

    fisher_mean_key = f"{prefix}_fisher_mean"
    fisher_min_key = f"{prefix}_fisher_min"
    if fisher_mean_key in data:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=data[fisher_mean_key],
                mode="lines",
                name="Fisher mean",
                line=dict(color=color, width=2),
                hovertemplate="Epoch %{x}<br>Fisher mean: %{y:.3f}<extra></extra>",
            ),
            row=4, col=1,
        )
    if fisher_min_key in data:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=data[fisher_min_key],
                mode="lines",
                name="Fisher min",
                line=dict(color=color, width=2, dash="dash"),
                hovertemplate="Epoch %{x}<br>Fisher min: %{y:.3f}<extra></extra>",
            ),
            row=4, col=1,
        )

    if epoch is not None:
        for row in range(1, 5):
            fig.add_vline(
                x=epoch,
                line=dict(color="orange", width=1, dash="dot"),
                row=row, col=1,
            )

    fig.update_yaxes(title_text="SNR", row=1, col=1)
    fig.update_yaxes(title_text="Weight norm", row=2, col=1)
    fig.update_yaxes(title_text="Score [0–1]", row=3, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Fisher ratio", row=4, col=1)
    fig.update_xaxes(title_text="Epoch", row=4, col=1)

    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def render_weight_geometry_group_snapshot(
    data: dict,
    epoch: int | None = None,
    matrix: str = "Win",
    height: int | None = None,
) -> go.Figure:
    """Per-group bar chart of radii and effective dimensionality at a selected epoch.

    Shows how compact (low radius) and how low-dimensional each frequency
    group is at the selected epoch. Groups labeled by their frequency index.

    Args:
        data: cross_epoch artifact from freq_group_weight_geometry
        epoch: target epoch; uses final epoch if None or not found
        matrix: "Win" or "Wout"
        height: total figure height in pixels
    """
    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    n_groups = len(group_freqs)
    prefix = matrix
    label = _MATRIX_LABELS.get(matrix, matrix)

    if height is None:
        height = 500

    # Resolve epoch index
    if epoch is not None and epoch in epochs:
        ep_idx = int(np.searchsorted(epochs, epoch))
    else:
        ep_idx = len(epochs) - 1
    actual_epoch = int(epochs[ep_idx])

    radii = data.get(f"{prefix}_radii")
    dims = data.get(f"{prefix}_dimensionality")

    group_labels = [f"f{group_freqs[g]}" for g in range(n_groups)]
    colors = [_freq_color(g, n_groups) for g in range(n_groups)]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Mean Radius per Group — {label}",
            f"Effective Dimensionality per Group — {label}",
        ],
        horizontal_spacing=0.12,
    )

    if radii is not None:
        radii_at_epoch = radii[ep_idx]
        fig.add_trace(
            go.Bar(
                x=group_labels,
                y=radii_at_epoch,
                marker_color=colors,
                name="Radius",
                showlegend=False,
                hovertemplate="Group %{x}<br>Radius: %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )

    if dims is not None:
        dims_at_epoch = dims[ep_idx]
        fig.add_trace(
            go.Bar(
                x=group_labels,
                y=dims_at_epoch,
                marker_color=colors,
                name="Dimensionality",
                showlegend=False,
                hovertemplate="Group %{x}<br>Eff. dim: %{y:.2f}<extra></extra>",
            ),
            row=1, col=2,
        )

    fig.update_yaxes(title_text="RMS radius", row=1, col=1)
    fig.update_yaxes(title_text="Participation ratio", row=1, col=2)
    fig.update_xaxes(title_text="Frequency group", row=1, col=1)
    fig.update_xaxes(title_text="Frequency group", row=1, col=2)

    fig.update_layout(
        height=height,
        title_text=f"Frequency Group Geometry — {label} at epoch {actual_epoch}",
        showlegend=False,
        margin=dict(l=60, r=20, t=80, b=60),
    )

    return fig
