"""Frequency Group Weight Geometry renderers.

Three views into geometric separation of frequency groups in weight space:
- timeseries: multi-panel evolution of SNR, spread/radius, circularity,
  and Fisher discriminant for W_in or W_out across training epochs.
- group_snapshot: per-group bar chart of radii and dimensionality at a
  selected epoch, showing which groups are compact vs. diffuse.
- centroid_pca: 2x2 PCA trajectory plot (PC1vPC2, PC1vPC3, PC2vPC3, 3D)
  showing how group centroids move in weight space over training.
"""

import colorsys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_MATRIX_LABELS = {
    "Win": "W_in",
    "Wout": "W_out",
}


def _freq_color(freq_value: int, n_freq: int) -> str:
    """Consistent HSL color for a frequency value — matches neuron_group_pca convention.

    hue = freq_value / n_freq so the same frequency gets the same color across all views.
    """
    hue = freq_value / max(n_freq, 1)
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
            row=1,
            col=1,
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
            row=2,
            col=1,
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
            row=2,
            col=1,
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
            row=3,
            col=1,
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
            row=4,
            col=1,
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
            row=4,
            col=1,
        )

    if epoch is not None:
        for row in range(1, 5):
            fig.add_vline(
                x=epoch,
                line=dict(color="orange", width=1, dash="dot"),
                row=row,  # type: ignore
                col=1,  # type: ignore
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
    n_freq: int | None = None,
) -> go.Figure:
    """Per-group bar chart of radii and effective dimensionality at a selected epoch.

    Shows how compact (low radius) and how low-dimensional each frequency
    group is at the selected epoch. Groups labeled by their frequency index.

    Args:
        data: cross_epoch artifact from freq_group_weight_geometry
        epoch: target epoch; uses final epoch if None or not found
        matrix: "Win" or "Wout"
        height: total figure height in pixels
        n_freq: total number of frequency slots in the model (used for color wheel
            alignment). Defaults to max(group_freqs) + 1 if not provided.
    """
    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    n_groups = len(group_freqs)
    if n_freq is None:
        n_freq = int(max(group_freqs)) + 1 if n_groups > 0 else 1
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
    colors = [_freq_color(int(group_freqs[g]), n_freq) for g in range(n_groups)]

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
            row=1,
            col=1,
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
            row=1,
            col=2,
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


def render_weight_geometry_centroid_pca(
    data: dict,
    epoch: int | None = None,
    matrix: str = "Win",
    height: int | None = None,
    n_freq: int | None = None,
) -> go.Figure:
    """2x2 PCA trajectory of frequency group centroids in weight space.

    Computes a global PCA basis from all group centroid positions pooled
    across all epochs, then shows each group's trajectory through that
    shared coordinate frame. Groups are colored consistently. Current
    epoch is highlighted with a larger marker on each trajectory.

    Panels: PC1 vs PC2, PC1 vs PC3, PC2 vs PC3, 3D scatter.

    Args:
        data: cross_epoch artifact from freq_group_weight_geometry
        epoch: epoch to highlight as the current position
        matrix: "Win" or "Wout"
        height: total figure height in pixels
        n_freq: total number of frequency slots in the model (used for color wheel
            alignment). Defaults to max(group_freqs) + 1 if not provided.
    """
    from miscope.analysis.library.geometry import compute_global_centroid_pca

    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    n_groups = len(group_freqs)
    if n_freq is None:
        n_freq = int(max(group_freqs)) + 1 if n_groups > 0 else 1
    prefix = matrix
    label = _MATRIX_LABELS.get(matrix, matrix)

    if height is None:
        height = 850

    centroids_key = f"{prefix}_centroids"
    if centroids_key not in data or n_groups == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"No group centroid data for {label}",
            height=height or 400,
        )
        return fig

    centroids_all = data[centroids_key].astype(np.float64)  # (n_epochs, n_groups, d)

    # Build per-group centroid lists and compute global PCA
    # compute_global_centroid_pca expects list of (n_groups, d) arrays
    centroid_list = [centroids_all[i] for i in range(len(epochs))]
    pca = compute_global_centroid_pca(centroid_list)
    projections = pca["projections"]  # (n_epochs, n_groups, n_components)
    var_ratio = pca["explained_variance_ratio"]

    n_components = projections.shape[2]
    if n_components < 3:
        # Pad with zeros if not enough variance for 3 PCs
        pad = np.zeros((projections.shape[0], projections.shape[1], 3 - n_components))
        projections = np.concatenate([projections, pad], axis=2)
        var_ratio = np.concatenate([var_ratio, np.zeros(3 - n_components)])

    colors = [_freq_color(int(group_freqs[g]), n_freq) for g in range(n_groups)]
    group_labels = [f"f{group_freqs[g]}" for g in range(n_groups)]
    epochs_list = epochs.tolist()

    # Resolve current epoch index
    ep_idx = len(epochs_list) - 1
    if epoch is not None:
        closest = int(np.argmin(np.abs(epochs - epoch)))
        ep_idx = closest

    pc_pairs = [(0, 1, "PC1", "PC2"), (0, 2, "PC1", "PC3"), (1, 2, "PC2", "PC3")]
    var_pct = [float(v) * 100 for v in var_ratio[:3]]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "scene"}],
        ],
        subplot_titles=[
            f"PC1 vs PC2 ({var_pct[0]:.1f}% + {var_pct[1]:.1f}%)",
            f"PC1 vs PC3 ({var_pct[0]:.1f}% + {var_pct[2]:.1f}%)",
            f"PC2 vs PC3 ({var_pct[1]:.1f}% + {var_pct[2]:.1f}%)",
            f"3D ({sum(var_pct[:3]):.1f}% total)",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
    )

    positions_2d = [(1, 1), (1, 2), (2, 1)]

    for g in range(n_groups):
        traj = projections[:, g, :]  # (n_epochs, 3)
        color = colors[g]
        glabel = group_labels[g]
        show_legend = True

        for (pc_a, pc_b, xl, yl), (row, col) in zip(pc_pairs, positions_2d):
            fig.add_trace(
                go.Scatter(
                    x=traj[:, pc_a],
                    y=traj[:, pc_b],
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    opacity=0.6,
                    name=glabel,
                    legendgroup=glabel,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=traj[:, pc_a],
                    y=traj[:, pc_b],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=list(range(len(epochs_list))),
                        colorscale=[[0, "rgba(200,200,200,0.5)"], [1, color]],
                        showscale=False,
                    ),
                    name=glabel,
                    legendgroup=glabel,
                    showlegend=show_legend,
                    customdata=epochs_list,
                    hovertemplate=f"{glabel}<br>Epoch %{{customdata}}<br>{xl}: %{{x:.3f}}<br>{yl}: %{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            # Highlight current epoch
            fig.add_trace(
                go.Scatter(
                    x=[traj[ep_idx, pc_a]],
                    y=[traj[ep_idx, pc_b]],
                    mode="markers",
                    marker=dict(
                        size=10, color=color, symbol="circle", line=dict(width=2, color="black")
                    ),
                    name=glabel,
                    legendgroup=glabel,
                    showlegend=False,
                    hovertemplate=f"{glabel} @ epoch {epochs_list[ep_idx]}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            show_legend = False

        # 3D trajectory
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(
                    size=3,
                    color=list(range(len(epochs_list))),
                    colorscale=[[0, "rgba(200,200,200,0.3)"], [1, color]],
                    showscale=False,
                ),
                name=glabel,
                legendgroup=glabel,
                showlegend=False,
                customdata=epochs_list,
                hovertemplate=f"{glabel}<br>Epoch %{{customdata}}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        # Highlight current epoch in 3D
        fig.add_trace(
            go.Scatter3d(
                x=[traj[ep_idx, 0]],
                y=[traj[ep_idx, 1]],
                z=[traj[ep_idx, 2]],
                mode="markers",
                marker=dict(
                    size=8, color=color, symbol="diamond", line=dict(width=1, color="black")
                ),
                name=glabel,
                legendgroup=glabel,
                showlegend=False,
                hovertemplate=f"{glabel} @ epoch {epochs_list[ep_idx]}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    current_epoch_label = epochs_list[ep_idx]
    fig.update_layout(
        height=height,
        title_text=f"Group Centroid Trajectories — {label} (epoch {current_epoch_label})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        scene=dict(
            xaxis_title=f"PC1 ({var_pct[0]:.1f}%)",
            yaxis_title=f"PC2 ({var_pct[1]:.1f}%)",
            zaxis_title=f"PC3 ({var_pct[2]:.1f}%)",
        ),
        margin=dict(l=40, r=20, t=80, b=40),
    )

    return fig
