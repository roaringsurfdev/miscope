"""Neuron Group PCA renderers.

Two views into within-frequency-group coordination in weight space:
- pca_cohesion: cumulative PC1+PC2+PC3 variance explained per group over epochs,
  with PC1 shown as a dashed reference line
- spread: mean L2 distance from group centroid per group over epochs

High cumulative var → top 3 directions capture most group variance (structured group).
Low cumulative var → group variation is spread across many dimensions (diffuse).
"""

import colorsys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _freq_color(freq_idx: int, n_freq: int) -> str:
    """Consistent HSL color for frequency index (0-indexed)."""
    hue = freq_idx / max(n_freq, 1)
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.5)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def render_neuron_group_pca_cohesion(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Line plot of cumulative PC1+PC2+PC3 variance explained per frequency group.

    Solid lines show cumulative variance explained by the top 3 components.
    Dashed lines show PC1 alone as a reference.
    Each color is one frequency group.

    Args:
        data: cross_epoch artifact from neuron_group_pca
        epoch: optional epoch cursor (vertical line)
    """
    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    pc_var = data["pc_var"]  # (n_epochs, n_groups, 3)
    n_freq = int(group_freqs.max()) + 1 if len(group_freqs) > 0 else 1

    fig = go.Figure()

    for g_idx, (freq, size) in enumerate(zip(group_freqs, group_sizes)):
        color = _freq_color(int(freq), n_freq)
        group_pc = pc_var[:, g_idx, :]  # (n_epochs, 3)

        cumulative = np.nansum(group_pc, axis=1).tolist()
        pc1_only = group_pc[:, 0].tolist()

        legend_name = f"freq {freq + 1} (n={size})"

        fig.add_trace(
            go.Scatter(
                x=epochs.tolist(),
                y=cumulative,
                mode="lines",
                name=legend_name,
                line=dict(color=color, width=2),
                legendgroup=str(freq),
                hovertemplate=(
                    f"freq={freq + 1} n={size}<br>epoch=%{{x}}<br>PC1+2+3=%{{y:.3f}}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epochs.tolist(),
                y=pc1_only,
                mode="lines",
                name="PC1 only",
                line=dict(color=color, width=1, dash="dash"),
                legendgroup=str(freq),
                showlegend=False,
                hovertemplate=(
                    f"freq={freq + 1} n={size}<br>epoch=%{{x}}<br>PC1=%{{y:.3f}}<extra></extra>"
                ),
            )
        )

    if epoch is not None:
        fig.add_vline(x=epoch, line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"))

    fig.update_layout(
        title="Within-group variance explained — top 3 PCs (W_in)<br>"
        "<sup>Solid = PC1+PC2+PC3 cumulative &nbsp;|&nbsp; Dashed = PC1 alone</sup>",
        xaxis_title="Epoch",
        yaxis_title="Cumulative variance explained",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        height=440,
        margin=dict(l=60, r=20, t=70, b=60),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=10),
        ),
    )
    return fig


def render_neuron_group_scatter(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Scatter plot of neuron positions in group PCA space (PC1 vs PC2).

    Each neuron is projected onto the final-epoch PCA basis of its frequency
    group. Group membership is re-derived from the current epoch's norm_matrix
    so the scatter reflects the current state of neuron specialization.

    Args:
        data: dict with keys group_bases, group_freqs, W_in, norm_matrix
        epoch: optional epoch label for the title
    """
    group_bases = data["group_bases"]  # (n_groups, 3, d_model)
    group_freqs = data["group_freqs"]
    norm_matrix = data["norm_matrix"]  # (n_freq, d_mlp)
    W_in = data["W_in"]  # (d_model, d_mlp)

    n_groups = len(group_freqs)
    n_freq = int(group_freqs.max()) + 1 if n_groups > 0 else 1
    dominant_freq = np.argmax(norm_matrix, axis=0)  # (d_mlp,)

    fig = go.Figure()

    for g_idx, freq in enumerate(group_freqs):
        members = np.where(dominant_freq == int(freq))[0]
        if len(members) == 0:
            continue

        basis = group_bases[g_idx]  # (3, d_model)
        coords = basis @ W_in[:, members]  # (3, n_members)
        color = _freq_color(int(freq), n_freq)

        fig.add_trace(
            go.Scatter(
                x=coords[0].tolist(),
                y=coords[1].tolist(),
                mode="markers",
                name=f"freq {freq + 1} (n={len(members)})",
                marker=dict(color=color, size=5, opacity=0.7),
                hovertemplate=(
                    f"freq={freq + 1}<br>PC1=%{{x:.3f}}<br>PC2=%{{y:.3f}}<extra></extra>"
                ),
            )
        )

    epoch_label = f" — epoch {epoch}" if epoch is not None else ""
    fig.update_layout(
        title=f"Neuron group W_in scatter (PC1 × PC2){epoch_label}",
        xaxis_title="PC1",
        yaxis_title="PC2",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        height=520,
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=10),
        ),
    )
    return fig


def render_neuron_group_spread(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Line plot of mean within-group L2 spread per frequency group over epochs.

    Each line is one frequency group. Low spread means group neurons have
    converged toward a common weight vector. High and rising spread means
    the group is expanding in weight space.

    Args:
        data: cross_epoch artifact from neuron_group_pca
        epoch: optional epoch cursor (vertical line)
    """
    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    mean_spread = data["mean_spread"]  # (n_epochs, n_groups)
    n_freq = int(group_freqs.max()) + 1 if len(group_freqs) > 0 else 1

    fig = go.Figure()

    for g_idx, (freq, size) in enumerate(zip(group_freqs, group_sizes)):
        color = _freq_color(int(freq), n_freq)
        y = mean_spread[:, g_idx].tolist()
        fig.add_trace(
            go.Scatter(
                x=epochs.tolist(),
                y=y,
                mode="lines",
                name=f"freq {freq + 1} (n={size})",
                line=dict(color=color, width=1.5),
                hovertemplate=f"freq={freq + 1} n={size}<br>epoch=%{{x}}<br>spread=%{{y:.4f}}<extra></extra>",
            )
        )

    if epoch is not None:
        fig.add_vline(x=epoch, line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"))

    fig.update_layout(
        title="Within-group mean L2 spread (W_in)",
        xaxis_title="Epoch",
        yaxis_title="Mean L2 distance from group centroid",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=20, t=50, b=60),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=10),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Projection-based renderers (require 'projections' field in artifact)
# ---------------------------------------------------------------------------


def _needs_rerun_figure(
    msg: str = "Rerun neuron_group_pca analyzer to populate projections",
) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        template="plotly_white", height=400, xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    return fig


def _epoch_idx(epochs: np.ndarray, epoch: int | None) -> int:
    """Return index of the nearest epoch in the epochs array."""
    if epoch is None:
        return len(epochs) - 1
    return int(np.argmin(np.abs(epochs.astype(int) - int(epoch))))


def render_neuron_group_scatter_3d(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """3D scatter of neuron positions in group PCA space (PC1 × PC2 × PC3).

    Neurons are colored by frequency group. Points are pre-computed projections
    centered by the final-epoch group centroid.

    Args:
        data: cross_epoch artifact from neuron_group_pca (requires 'projections')
        epoch: training epoch to visualize (nearest available)
    """
    if "projections" not in data:
        return _needs_rerun_figure()

    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    projections = data["projections"]  # (n_epochs, d_mlp, 3)
    neuron_group_idx = data["neuron_group_idx"]  # (d_mlp,)

    ep_idx = _epoch_idx(epochs, epoch)
    pts = projections[ep_idx]  # (d_mlp, 3)
    actual_epoch = int(epochs[ep_idx])
    n_freq = int(group_freqs.max()) + 1 if len(group_freqs) > 0 else 1

    fig = go.Figure()
    for g_idx, freq in enumerate(group_freqs):
        members = np.where(neuron_group_idx == g_idx)[0]
        if len(members) == 0:
            continue
        g_pts = pts[members]
        color = _freq_color(int(freq), n_freq)
        fig.add_trace(
            go.Scatter3d(
                x=g_pts[:, 0].tolist(),
                y=g_pts[:, 1].tolist(),
                z=g_pts[:, 2].tolist(),
                mode="markers",
                name=f"freq {freq + 1} (n={len(members)})",
                marker=dict(color=color, size=4, opacity=0.85),
                hovertemplate=f"freq={freq + 1}<br>PC1=%{{x:.3f}}<br>PC2=%{{y:.3f}}<br>PC3=%{{z:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Neuron group 3D scatter (PC1×PC2×PC3) — epoch {actual_epoch}",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        template="plotly_white",
        height=560,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10)),
    )
    return fig


def render_neuron_group_scatter_purity(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """PC1 vs PC2 scatter colored by dominant-frequency purity.

    Purity = max(norm_matrix) over all frequencies for each neuron.
    High purity → neuron strongly prefers one frequency.

    Args:
        data: dict with cross_epoch artifact fields + 'norm_matrix' from neuron_freq_norm
        epoch: training epoch (used for title; norm_matrix must already match)
    """
    if "projections" not in data:
        return _needs_rerun_figure()

    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    projections = data["projections"]
    neuron_group_idx = data["neuron_group_idx"]
    norm_matrix = data["norm_matrix"]  # (n_freq, d_mlp)

    ep_idx = _epoch_idx(epochs, epoch)
    pts = projections[ep_idx]  # (d_mlp, 3)
    actual_epoch = int(epochs[ep_idx])

    purity = norm_matrix.max(axis=0)  # (d_mlp,)
    grouped = np.where(neuron_group_idx >= 0)[0]

    group_labels = [
        f"neuron {i}<br>freq {group_freqs[neuron_group_idx[i]] + 1}<br>purity {purity[i]:.2f}"
        for i in grouped
    ]

    fig = go.Figure(
        go.Scatter(
            x=pts[grouped, 0].tolist(),
            y=pts[grouped, 1].tolist(),
            mode="markers",
            text=group_labels,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                color=purity[grouped].tolist(),
                colorscale="RdYlGn",
                cmin=0,
                cmax=1,
                size=7,
                opacity=0.85,
                colorbar=dict(title="Purity", thickness=14),
            ),
        )
    )

    fig.update_layout(
        title=f"Neuron group scatter — purity (epoch {actual_epoch})",
        xaxis_title="PC1",
        yaxis_title="PC2",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        height=520,
        margin=dict(l=60, r=80, t=60, b=60),
    )
    return fig


def render_neuron_group_all_panels(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Multi-panel PC1 vs PC2 scatter — one subplot per frequency group.

    Each panel shows the neurons in that group colored by purity.

    Args:
        data: dict with cross_epoch artifact fields + 'norm_matrix' from neuron_freq_norm
        epoch: training epoch to visualize
    """
    if "projections" not in data:
        return _needs_rerun_figure()

    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    projections = data["projections"]
    neuron_group_idx = data["neuron_group_idx"]
    norm_matrix = data["norm_matrix"]

    n_groups = len(group_freqs)
    if n_groups == 0:
        return go.Figure()

    ep_idx = _epoch_idx(epochs, epoch)
    pts = projections[ep_idx]  # (d_mlp, 3)
    actual_epoch = int(epochs[ep_idx])
    purity = norm_matrix.max(axis=0)  # (d_mlp,)

    n_cols = min(4, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"freq {f + 1} (n={s})" for f, s in zip(group_freqs, group_sizes)],
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for g_idx, freq in enumerate(group_freqs):
        row = g_idx // n_cols + 1
        col = g_idx % n_cols + 1
        members = np.where(neuron_group_idx == g_idx)[0]
        g_pts = pts[members]
        g_purity = purity[members]

        fig.add_trace(
            go.Scatter(
                x=g_pts[:, 0].tolist(),
                y=g_pts[:, 1].tolist(),
                mode="markers",
                showlegend=False,
                hovertemplate=f"freq={freq + 1}<br>PC1=%{{x:.3f}}<br>PC2=%{{y:.3f}}<extra></extra>",
                marker=dict(
                    color=g_purity.tolist(),
                    colorscale="RdYlGn",
                    cmin=0,
                    cmax=1,
                    size=9,
                    opacity=0.85,
                ),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"All frequency groups — PC1×PC2 (epoch {actual_epoch})",
        template="plotly_white",
        height=260 * n_rows + 80,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def render_neuron_group_trajectory(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Neuron paths through PC1 × PC2 space across all training epochs.

    Lines show each neuron's trajectory. Blue circles mark epoch 0;
    red diamonds mark the final epoch. Groups are distinguished by color.

    Args:
        data: cross_epoch artifact from neuron_group_pca (requires 'projections')
        epoch: unused (all epochs shown); kept for API consistency
    """
    if "projections" not in data:
        return _needs_rerun_figure()

    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    projections = data["projections"]  # (n_epochs, d_mlp, 3)
    neuron_group_idx = data["neuron_group_idx"]
    n_freq = int(group_freqs.max()) + 1 if len(group_freqs) > 0 else 1

    fig = go.Figure()

    for g_idx, (freq, size) in enumerate(zip(group_freqs, group_sizes)):
        members = np.where(neuron_group_idx == g_idx)[0]
        if len(members) == 0:
            continue
        color = _freq_color(int(freq), n_freq)

        # Concatenate paths with None separators — one trace per group
        x_lines, y_lines = [], []
        for neuron_idx in members:
            path = projections[:, neuron_idx, :]  # (n_epochs, 3)
            x_lines.extend(path[:, 0].tolist())
            x_lines.append(None)
            y_lines.extend(path[:, 1].tolist())
            y_lines.append(None)

        fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                name=f"freq {freq + 1} (n={size})",
                line=dict(color=color, width=1),
                opacity=0.45,
                legendgroup=str(freq),
            )
        )

        # Start markers (epoch 0)
        fig.add_trace(
            go.Scatter(
                x=[projections[0, i, 0] for i in members],
                y=[projections[0, i, 1] for i in members],
                mode="markers",
                marker=dict(color="steelblue", symbol="circle", size=6, opacity=0.7),
                showlegend=False,
                legendgroup=str(freq),
                hovertemplate=f"freq={freq + 1}<br>epoch 0<extra></extra>",
            )
        )

        # End markers (final epoch)
        fig.add_trace(
            go.Scatter(
                x=[projections[-1, i, 0] for i in members],
                y=[projections[-1, i, 1] for i in members],
                mode="markers",
                marker=dict(color="crimson", symbol="diamond", size=7, opacity=0.9),
                showlegend=False,
                legendgroup=str(freq),
                hovertemplate=f"freq={freq + 1}<br>epoch {int(epochs[-1])}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Neuron group trajectories in PCA space (PC1 × PC2)<br>"
        "<sup>Blue circles = epoch 0 &nbsp;|&nbsp; Red diamonds = final epoch</sup>",
        xaxis_title="PC1",
        yaxis_title="PC2",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        height=580,
        margin=dict(l=60, r=20, t=70, b=60),
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10)),
    )
    return fig


def render_neuron_group_polar_histogram(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Polar histogram of PCA angles per frequency group.

    PCA angle = atan2(PC2, PC1) for each neuron's projection at the
    selected epoch. Uniform distribution supports the phase-tiling
    hypothesis; clustering indicates discrete preferred phases.

    Args:
        data: cross_epoch artifact from neuron_group_pca (requires 'projections')
        epoch: training epoch to visualize
    """
    if "projections" not in data:
        return _needs_rerun_figure()

    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    projections = data["projections"]
    neuron_group_idx = data["neuron_group_idx"]

    n_groups = len(group_freqs)
    if n_groups == 0:
        return go.Figure()

    ep_idx = _epoch_idx(epochs, epoch)
    pts = projections[ep_idx]  # (d_mlp, 3)
    actual_epoch = int(epochs[ep_idx])
    n_freq = int(group_freqs.max()) + 1 if n_groups > 0 else 1

    n_cols = min(4, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    specs = [[{"type": "polar"}] * n_cols for _ in range(n_rows)]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=[f"freq {f + 1} (n={s})" for f, s in zip(group_freqs, group_sizes)],
    )

    n_bins = 16
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers_deg = np.degrees((bin_edges[:-1] + bin_edges[1:]) / 2)
    bin_width_deg = 360.0 / n_bins

    for g_idx, freq in enumerate(group_freqs):
        row = g_idx // n_cols + 1
        col = g_idx % n_cols + 1
        members = np.where(neuron_group_idx == g_idx)[0]
        if len(members) == 0:
            continue

        g_pts = pts[members]
        valid = ~np.isnan(g_pts[:, 0])
        angles = np.arctan2(g_pts[valid, 1], g_pts[valid, 0])
        counts, _ = np.histogram(angles, bins=bin_edges)
        color = _freq_color(int(freq), n_freq)

        fig.add_trace(
            go.Barpolar(
                r=counts.tolist(),
                theta=bin_centers_deg.tolist(),
                width=[bin_width_deg] * n_bins,
                marker_color=color,
                opacity=0.8,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Neuron group PCA angle distribution — epoch {actual_epoch}",
        template="plotly_white",
        height=260 * n_rows + 80,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Group centroid trajectory views
# ---------------------------------------------------------------------------


def render_group_centroid_timeseries(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """PC1, PC2, PC3 of each frequency group's centroid over training.

    The centroid is the mean W_in column vector for the group's neurons.
    Coordinates come from a shared PCA fit jointly on all groups × all epochs,
    so paths are directly comparable across groups.

    Three-row subplot: one row per PC component.
    Variance explained by the shared basis is shown in each row title.
    Epoch cursor drawn as a vertical line when provided.

    Args:
        data: cross_epoch artifact from neuron_group_pca (requires centroid_pca_coords)
        epoch: optional epoch cursor
    """
    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    coords = data["centroid_pca_coords"]   # (n_epochs, n_groups, 3)
    var_exp = data["centroid_pca_var"]     # (3,)
    n_groups = len(group_freqs)
    n_freq = int(group_freqs.max()) + 1 if n_groups > 0 else 1

    row_titles = [
        f"PC1 ({float(var_exp[0]):.1%} var)",
        f"PC2 ({float(var_exp[1]):.1%} var)",
        f"PC3 ({float(var_exp[2]):.1%} var)",
    ]
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=row_titles,
    )

    for g_idx, (freq, size) in enumerate(zip(group_freqs, group_sizes)):
        color = _freq_color(int(freq), n_freq)
        label = f"freq {int(freq) + 1} (n={size})"
        for comp, row in enumerate([1, 2, 3]):
            fig.add_trace(
                go.Scatter(
                    x=epochs.tolist(),
                    y=coords[:, g_idx, comp].tolist(),
                    mode="lines",
                    name=label,
                    legendgroup=str(freq),
                    showlegend=(comp == 0),
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        f"{label}<br>epoch=%{{x}}<br>PC{comp + 1}=%{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=row, col=1,
            )

    if epoch is not None:
        for row in [1, 2, 3]:
            fig.add_vline(
                x=epoch,
                line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"),
                row=row, col=1,
            )

    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    for row in [1, 2, 3]:
        fig.update_yaxes(title_text="Centroid coord", row=row, col=1)

    fig.update_layout(
        title="Group centroid trajectories in shared W_in PCA space<br>"
              "<sup>Shared basis fit jointly on all groups × all epochs</sup>",
        template="plotly_white",
        height=620,
        margin=dict(l=60, r=20, t=80, b=60),
        legend=dict(orientation="v", x=1.02, y=1),
    )
    return fig


def render_group_centroid_paths(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """PC1 vs PC2 centroid paths for each frequency group.

    Each group traces a path through the shared W_in PCA plane as training
    progresses. Marker color encodes epoch (blue=early, red=late).
    Open circle = epoch 0, filled circle = final epoch.

    Args:
        data: cross_epoch artifact from neuron_group_pca (requires centroid_pca_coords)
        epoch: optional epoch cursor (vertical line skipped for 2D scatter)
    """
    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    coords = data["centroid_pca_coords"]   # (n_epochs, n_groups, 3)
    var_exp = data["centroid_pca_var"]     # (3,)
    n_groups = len(group_freqs)
    n_freq = int(group_freqs.max()) + 1 if n_groups > 0 else 1

    # Epoch normalized to [0, 1] for colorscale
    ep_arr = np.asarray(epochs, dtype=float)
    if ep_arr.max() > ep_arr.min():
        col_scale = ((ep_arr - ep_arr.min()) / (ep_arr.max() - ep_arr.min())).tolist()
    else:
        col_scale = [0.5] * len(ep_arr)

    fig = go.Figure()

    for g_idx, (freq, size) in enumerate(zip(group_freqs, group_sizes)):
        color = _freq_color(int(freq), n_freq)
        label = f"freq {int(freq) + 1} (n={size})"
        x = coords[:, g_idx, 0].tolist()
        y = coords[:, g_idx, 1].tolist()

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(
                size=6,
                color=col_scale,
                colorscale="RdYlBu_r",
                cmin=0, cmax=1,
                showscale=False,
            ),
            customdata=epochs.tolist(),
            hovertemplate=(
                f"{label}<br>epoch=%{{customdata}}<br>"
                "PC1=%{x:.3f}  PC2=%{y:.3f}<extra></extra>"
            ),
        ))

        # Start/end markers
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]],
            mode="markers",
            marker=dict(color=color, size=10, symbol="circle-open", line=dict(width=2)),
            showlegend=False,
            hovertemplate=f"{label} start (ep {int(epochs[0])})<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]],
            mode="markers",
            marker=dict(color=color, size=10, symbol="circle"),
            showlegend=False,
            hovertemplate=f"{label} end (ep {int(epochs[-1])})<extra></extra>",
        ))

    # Invisible colorbar trace for epoch scale
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale="RdYlBu_r", cmin=0, cmax=1,
            color=[0], showscale=True,
            colorbar=dict(
                title="epoch",
                tickvals=[0, 0.5, 1],
                ticktext=[str(int(ep_arr.min())), str(int(ep_arr.mean())), str(int(ep_arr.max()))],
                len=0.5, x=1.02,
            ),
        ),
        showlegend=False,
    ))

    fig.update_layout(
        title="Group centroid paths — PC1 vs PC2 (shared W_in PCA)<br>"
              "<sup>Open circle = epoch 0  |  Filled = final epoch  |  "
              f"Marker color = training progress</sup>",
        xaxis_title=f"PC1 ({float(var_exp[0]):.1%} var)",
        yaxis_title=f"PC2 ({float(var_exp[1]):.1%} var)",
        template="plotly_white",
        height=540,
        margin=dict(l=60, r=80, t=80, b=60),
        legend=dict(x=1.08, y=1),
    )
    return fig
