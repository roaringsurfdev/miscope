"""REQ_029, REQ_032, REQ_038: Parameter Space Trajectory Visualizations.

Renders PCA trajectory projections, explained variance, and parameter
velocity from precomputed cross-epoch analysis results.

All renderers accept precomputed PCA results (projections, explained
variance) rather than raw weight snapshots. PCA computation happens
at analysis time via ParameterTrajectoryPCA (REQ_038), not at render time.
"""

import numpy as np
import plotly.graph_objects as go

from miscope.analysis.library.weights import COMPONENT_GROUPS

# Display names for component groups
_GROUP_LABELS = {
    "all": "All Parameters",
    "embedding": "Embedding",
    "attention": "Attention",
    "mlp": "Mlp",
}


def render_parameter_trajectory(
    pca_result: dict[str, np.ndarray],
    epochs: list[int],
    current_epoch: int,
    group_label: str = "All Parameters",
    title: str | None = None,
    height: int = 450,
) -> go.Figure:
    """2D PCA trajectory: PC1 vs PC2.

    Points colored by epoch (blue=early, red=late), connected by path.
    Current epoch highlighted with a larger marker.

    Args:
        pca_result: Dict with 'projections' and 'explained_variance_ratio'.
        epochs: Epoch numbers corresponding to each projection.
        current_epoch: Current epoch for highlight marker.
        group_label: Display name for the component group.
    """
    if title is None:
        title = f"Parameter Trajectory ({group_label})"
    return _render_trajectory_2d(
        pca_result,
        col_x=0,
        col_y=1,
        epochs=epochs,
        current_epoch=current_epoch,
        title=title,
        height=height,
    )


def render_trajectory_pc1_pc3(
    pca_result: dict[str, np.ndarray],
    epochs: list[int],
    current_epoch: int,
    group_label: str = "All Parameters",
    title: str | None = None,
    height: int = 450,
) -> go.Figure:
    """2D PCA trajectory: PC1 vs PC3 (REQ_032)."""
    if title is None:
        title = f"Parameter Trajectory PC1 vs PC3 ({group_label})"
    return _render_trajectory_2d(
        pca_result,
        col_x=0,
        col_y=2,
        epochs=epochs,
        current_epoch=current_epoch,
        title=title,
        height=height,
    )


def render_trajectory_pc2_pc3(
    pca_result: dict[str, np.ndarray],
    epochs: list[int],
    current_epoch: int,
    group_label: str = "All Parameters",
    title: str | None = None,
    height: int = 450,
) -> go.Figure:
    """2D PCA trajectory: PC2 vs PC3 (REQ_032)."""
    if title is None:
        title = f"Parameter Trajectory PC2 vs PC3 ({group_label})"
    return _render_trajectory_2d(
        pca_result,
        col_x=1,
        col_y=2,
        epochs=epochs,
        current_epoch=current_epoch,
        title=title,
        height=height,
    )


def render_trajectory_3d(
    pca_result: dict[str, np.ndarray],
    epochs: list[int],
    current_epoch: int,
    group_label: str = "All Parameters",
    title: str | None = None,
    height: int = 550,
) -> go.Figure:
    """3D interactive PCA trajectory: PC1 vs PC2 vs PC3 (REQ_032).

    Supports rotation, zoom, and pan for exploratory analysis.
    """
    projections = pca_result["projections"]
    var_ratio = pca_result["explained_variance_ratio"]

    fig = go.Figure()

    # Trajectory path
    fig.add_trace(
        go.Scatter3d(
            x=projections[:, 0],
            y=projections[:, 1],
            z=projections[:, 2],
            mode="lines",
            line=dict(color="rgba(150, 150, 150, 0.4)", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Trajectory points colored by epoch
    fig.add_trace(
        go.Scatter3d(
            x=projections[:, 0],
            y=projections[:, 1],
            z=projections[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=epochs,
                colorscale="Bluered",
                showscale=True,
                colorbar=dict(title="Epoch", thickness=15),
            ),
            name="Checkpoints",
            hovertemplate=(
                "Epoch %{customdata}<br>"
                "PC1: %{x:.4f}<br>"
                "PC2: %{y:.4f}<br>"
                "PC3: %{z:.4f}<extra></extra>"
            ),
            customdata=epochs,
        )
    )

    # Highlight current epoch
    if current_epoch in epochs:
        idx = epochs.index(current_epoch)
        fig.add_trace(
            go.Scatter3d(
                x=[projections[idx, 0]],
                y=[projections[idx, 1]],
                z=[projections[idx, 2]],
                mode="markers",
                marker=dict(
                    size=7,
                    color="red",
                    symbol="diamond",
                    line=dict(width=1, color="black"),
                ),
                name=f"Epoch {current_epoch}",
                hovertemplate=f"Epoch {current_epoch}<extra></extra>",
            )
        )

    if title is None:
        title = f"Parameter Trajectory 3D ({group_label})"

    pct = [v * 100 for v in var_ratio[:3]]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"PC1 ({pct[0]:.1f}% var)",
            yaxis_title=f"PC2 ({pct[1]:.1f}% var)",
            zaxis_title=f"PC3 ({pct[2]:.1f}% var)",
        ),
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        legend_orientation="h",
        legend=dict(
            yanchor="top",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def render_explained_variance(
    pca_result: dict[str, np.ndarray],
    group_label: str = "All Parameters",
    n_components: int = 10,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Scree plot of explained variance per principal component.

    Args:
        pca_result: Dict with 'explained_variance_ratio'.
        group_label: Display name for the component group.
        n_components: Max number of PCs to show.
        title: Custom title.
        height: Figure height in pixels.
    """
    var_ratio = pca_result["explained_variance_ratio"][:n_components]
    cumulative = np.cumsum(var_ratio)

    pc_labels = [f"PC{i + 1}" for i in range(len(var_ratio))]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=pc_labels,
            y=var_ratio * 100,
            name="Individual",
            marker_color="rgba(31, 119, 180, 0.7)",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pc_labels,
            y=cumulative * 100,
            mode="lines+markers",
            name="Cumulative",
            line=dict(color="rgba(214, 39, 40, 0.8)", width=2),
            marker=dict(size=5),
            hovertemplate="%{x}: %{y:.1f}% cumulative<extra></extra>",
        )
    )

    if title is None:
        title = f"Explained Variance ({group_label})"

    fig.update_layout(
        title=title,
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_parameter_velocity(
    velocity: np.ndarray,
    epochs: list[int],
    current_epoch: int,
    group_label: str = "All Parameters",
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Parameter velocity (L2 norm of change) over training epochs.

    Args:
        velocity: 1D array of length (n_epochs - 1).
        epochs: Epoch numbers (length n_epochs).
        current_epoch: Current epoch for vertical indicator.
        group_label: Display name for the component group.
        title: Custom title.
        height: Figure height in pixels.
    """
    velocity_epochs = epochs[1:]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=velocity_epochs,
            y=velocity,
            mode="lines",
            name="Velocity",
            line=dict(color="rgba(31, 119, 180, 1.0)", width=2),
            hovertemplate="Epoch %{x}<br>Velocity: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_vline(
        x=current_epoch,
        line_dash="solid",
        line_color="red",
        line_width=2,
        annotation_text=f"Epoch {current_epoch}",
        annotation_position="top right",
        annotation_font_color="red",
    )

    if title is None:
        title = f"Parameter Velocity ({group_label})"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="||delta theta|| / epoch",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_component_velocity(
    cross_epoch_data: dict[str, np.ndarray],
    epochs: list[int],
    current_epoch: int,
    title: str | None = None,
    height: int = 350,
) -> go.Figure:
    """Per-component group velocity over training epochs.

    One line per component group (Embedding, Attention, MLP).

    Args:
        cross_epoch_data: Full cross-epoch data dict with
            {group}__velocity keys.
        epochs: Epoch numbers.
        current_epoch: Current epoch for vertical indicator.
        title: Custom title.
        height: Figure height in pixels.
    """
    velocity_epochs = epochs[1:]
    colors = {
        "embedding": "rgba(31, 119, 180, 1.0)",
        "attention": "rgba(44, 160, 44, 1.0)",
        "mlp": "rgba(214, 39, 40, 1.0)",
    }

    fig = go.Figure()

    for group_name in COMPONENT_GROUPS:
        velocity = cross_epoch_data[f"{group_name}__velocity"]
        fig.add_trace(
            go.Scatter(
                x=velocity_epochs,
                y=velocity,
                mode="lines",
                name=group_name.capitalize(),
                line=dict(color=colors.get(group_name, "gray"), width=2),
                hovertemplate=(
                    f"{group_name.capitalize()}<br>"
                    "Epoch %{x}<br>"
                    "Velocity: %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=current_epoch,
        line_dash="solid",
        line_color="red",
        line_width=2,
        annotation_text=f"Epoch {current_epoch}",
        annotation_position="top right",
        annotation_font_color="red",
    )

    if title is None:
        title = "Component Velocity Comparison"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="||delta theta|| / epoch",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def _render_trajectory_2d(
    pca_result: dict[str, np.ndarray],
    col_x: int,
    col_y: int,
    epochs: list[int],
    current_epoch: int,
    title: str,
    height: int = 450,
) -> go.Figure:
    """Shared 2D trajectory scatter with path and epoch highlight.

    Args:
        pca_result: Output from compute_pca_trajectory.
        col_x: Column index for x-axis (0=PC1, 1=PC2, 2=PC3).
        col_y: Column index for y-axis.
        epochs: Epoch numbers corresponding to each snapshot.
        current_epoch: Current epoch for highlight marker.
        title: Plot title.
        height: Figure height in pixels.
    """
    projections = pca_result["projections"]
    var_ratio = pca_result["explained_variance_ratio"]
    pc_x = col_x + 1
    pc_y = col_y + 1

    fig = go.Figure()

    # Trajectory path
    fig.add_trace(
        go.Scatter(
            x=projections[:, col_x],
            y=projections[:, col_y],
            mode="lines",
            line=dict(color="rgba(150, 150, 150, 0.4)", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Trajectory points colored by epoch
    fig.add_trace(
        go.Scatter(
            x=projections[:, col_x],
            y=projections[:, col_y],
            mode="markers",
            marker=dict(
                size=5,
                color=epochs,
                colorscale="Bluered",
                showscale=True,
                colorbar=dict(title="Epoch", thickness=15),
            ),
            name="Checkpoints",
            hovertemplate=(
                "Epoch %{customdata}<br>"
                f"PC{pc_x}: %{{x:.4f}}<br>"
                f"PC{pc_y}: %{{y:.4f}}<extra></extra>"
            ),
            customdata=epochs,
        )
    )

    # Highlight current epoch
    if current_epoch in epochs:
        idx = epochs.index(current_epoch)
        fig.add_trace(
            go.Scatter(
                x=[projections[idx, col_x]],
                y=[projections[idx, col_y]],
                mode="markers",
                marker=dict(
                    size=14,
                    color="red",
                    symbol="star",
                    line=dict(width=1, color="black"),
                ),
                name=f"Epoch {current_epoch}",
                hovertemplate=f"Epoch {current_epoch}<extra></extra>",
            )
        )

    pct_x = var_ratio[col_x] * 100
    pct_y = var_ratio[col_y] * 100

    fig.update_layout(
        title=title,
        xaxis_title=f"PC{pc_x} ({pct_x:.1f}% var)",
        yaxis_title=f"PC{pc_y} ({pct_y:.1f}% var)",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_trajectory_pca_variance(
    cross_epoch_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    height: int = 600,
) -> go.Figure:
    """Progressive variance explained by PC1/PC2/PC3 of the parameter trajectory.

    At each epoch N, computes what fraction of the accumulated trajectory
    variance (epochs 0..N) is captured by each principal component. Shows
    whether the trajectory expands into higher-dimensional PC space during
    grokking and contracts afterward.

    Three panels (PC1, PC2, PC3), one line per component group
    (all, embedding, attention, mlp).

    Args:
        cross_epoch_data: From ArtifactLoader.load_cross_epoch("parameter_trajectory").
            Must contain "{group}__projections" and "epochs" keys.
        current_epoch: Current epoch for vertical indicator.
        height: Total figure height in pixels.

    Returns:
        Plotly Figure with 3 vertically stacked subplots.
    """
    from plotly.subplots import make_subplots

    epochs = cross_epoch_data["epochs"].tolist()
    n_epochs = len(epochs)

    group_colors = {
        "all": "rgba(100, 100, 100, 1.0)",
        "embedding": "rgba(31, 119, 180, 1.0)",
        "attention": "rgba(44, 160, 44, 1.0)",
        "mlp": "rgba(214, 39, 40, 1.0)",
    }

    group_var_fracs: dict[str, np.ndarray] = {}
    for group_name in group_colors:
        proj_key = f"{group_name}__projections"
        if proj_key not in cross_epoch_data:
            continue
        projections = cross_epoch_data[proj_key]  # (N, n_components)
        var_fracs = np.zeros((n_epochs, 3))
        for i in range(n_epochs):
            chunk = projections[: i + 1]
            variances = np.var(chunk, axis=0)
            total = variances.sum()
            if total > 1e-12:
                var_fracs[i, :3] = variances[:3] / total
            else:
                var_fracs[i, 0] = 1.0
        group_var_fracs[group_name] = var_fracs

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "PC1 Variance Explained",
            "PC2 Variance Explained",
            "PC3 Variance Explained",
        ],
    )

    for group_name, color in group_colors.items():
        if group_name not in group_var_fracs:
            continue
        label = _GROUP_LABELS.get(group_name, group_name.capitalize())
        var_fracs = group_var_fracs[group_name]

        for pc_idx in range(3):
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=var_fracs[:, pc_idx] * 100,
                    mode="lines",
                    name=label,
                    legendgroup=group_name,
                    showlegend=(pc_idx == 0),
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        f"{label}<br>Epoch %{{x}}<br>PC{pc_idx + 1}: %{{y:.1f}}%<extra></extra>"
                    ),
                ),
                row=pc_idx + 1,
                col=1,
            )

    if current_epoch is not None:
        for row in range(1, 4):
            fig.add_vline(
                x=current_epoch,
                line_dash="solid",
                line_color="red",
                line_width=1,
                row=row,  # type: ignore[reportArgumentType]
                col=1,  # type: ignore[reportArgumentType]
            )

    for row in range(1, 4):
        fig.update_yaxes(range=[0, 105], ticksuffix="%", row=row, col=1)

    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.update_layout(
        template="plotly_white",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return fig


def render_trajectory_group_overlay(
    cross_epoch_data: dict[str, np.ndarray],
    epochs: list[int],
    current_epoch: int,
    col_x: int = 0,
    col_y: int = 1,
    title: str | None = None,
    height: int = 500,
) -> go.Figure:
    """Normalized overlay of embedding, attention, and MLP trajectories on shared axes.

    Each group's trajectory is independently centered and scaled so all three
    fit in a comparable space. Shape and loop structure are preserved; absolute
    magnitudes are not (they differ by orders of magnitude across groups).

    Use this to compare trajectory timing and shape — does one group move first?
    Do they trace similar loops or diverge? — without the scale differences
    that make direct overlay of raw projections unreadable.

    Args:
        cross_epoch_data: From ArtifactLoader.load_cross_epoch("parameter_trajectory").
        epochs: Epoch numbers.
        current_epoch: Current epoch for highlight marker.
        col_x: PC column for x-axis (0=PC1, 1=PC2).
        col_y: PC column for y-axis (1=PC2, 2=PC3).
        title: Custom title.
        height: Figure height in pixels.
    """
    pc_x, pc_y = col_x + 1, col_y + 1
    colors = {
        "embedding": "rgba(31, 119, 180, 1.0)",
        "attention": "rgba(44, 160, 44, 1.0)",
        "mlp": "rgba(214, 39, 40, 1.0)",
    }

    fig = go.Figure()

    for group_name, color in colors.items():
        proj_key = f"{group_name}__projections"
        if proj_key not in cross_epoch_data:
            continue
        proj = cross_epoch_data[proj_key]
        nx, ny = _normalize_trajectory(proj[:, col_x], proj[:, col_y])

        fig.add_trace(
            go.Scatter(
                x=nx,
                y=ny,
                mode="lines",
                line=dict(color=color.replace("1.0", "0.35"), width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=nx,
                y=ny,
                mode="markers",
                name=_GROUP_LABELS[group_name],
                marker=dict(size=4, color=epochs, colorscale="Bluered", showscale=False),
                customdata=epochs,
                hovertemplate=(
                    f"{_GROUP_LABELS[group_name]}<br>"
                    "Epoch %{customdata}<br>"
                    f"PC{pc_x}: %{{x:.3f}}<br>"
                    f"PC{pc_y}: %{{y:.3f}}<extra></extra>"
                ),
            )
        )

        if current_epoch in epochs:
            idx = epochs.index(current_epoch)
            fig.add_trace(
                go.Scatter(
                    x=[nx[idx]],
                    y=[ny[idx]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=color,
                        symbol="star",
                        line=dict(width=1, color="black"),
                    ),
                    name=f"{_GROUP_LABELS[group_name]} epoch {current_epoch}",
                    showlegend=False,
                    hovertemplate=f"{_GROUP_LABELS[group_name]} epoch {current_epoch}<extra></extra>",
                )
            )

    if title is None:
        title = f"Group Trajectory Overlay (PC{pc_x} vs PC{pc_y}, normalized)"

    fig.update_layout(
        title=title,
        xaxis_title=f"PC{pc_x} (normalized)",
        yaxis_title=f"PC{pc_y} (normalized)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def _normalize_trajectory(
    pc_x: np.ndarray,
    pc_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Center and scale a trajectory so its widest axis spans [-0.5, 0.5].

    Preserves aspect ratio (shape and loop structure) within each group.
    """
    cx, cy = pc_x.mean(), pc_y.mean()
    nx, ny = pc_x - cx, pc_y - cy
    scale = max(nx.max() - nx.min(), ny.max() - ny.min())
    if scale > 1e-12:
        nx, ny = nx / scale, ny / scale
    return nx, ny


def render_trajectory_proximity(
    cross_epoch_data: dict[str, np.ndarray],
    epochs: list[int],
    current_epoch: int,
    col_x: int = 0,
    col_y: int = 1,
    title: str | None = None,
    height: int = 350,
) -> go.Figure:
    """Pairwise L2 distance between normalized group trajectories over training.

    At each epoch, computes the distance between each pair of groups in
    normalized PC space (same normalization as the group overlay). Distance
    near zero means the two groups are occupying the same region of their
    respective parameter spaces at that moment.

    Args:
        cross_epoch_data: From ArtifactLoader.load_cross_epoch("parameter_trajectory").
        epochs: Epoch numbers.
        current_epoch: Current epoch for vertical cursor.
        col_x: PC column for x-axis (0=PC1, 1=PC2).
        col_y: PC column for y-axis (1=PC2, 2=PC3).
        title: Custom title.
        height: Figure height in pixels.
    """
    pc_x, pc_y = col_x + 1, col_y + 1

    groups: dict[str, np.ndarray] = {}
    for name in ("embedding", "attention", "mlp"):
        proj_key = f"{name}__projections"
        if proj_key not in cross_epoch_data:
            continue
        proj = cross_epoch_data[proj_key]
        nx, ny = _normalize_trajectory(proj[:, col_x], proj[:, col_y])
        groups[name] = np.stack([nx, ny], axis=1)

    pairs = [
        ("emb_attn", "embedding", "attention", "royalblue", "Embedding \u2194 Attention"),
        ("emb_mlp", "embedding", "mlp", "darkorange", "Embedding \u2194 MLP"),
        ("attn_mlp", "attention", "mlp", "seagreen", "Attention \u2194 MLP"),
    ]

    fig = go.Figure()

    for _key, a, b, color, label in pairs:
        if a not in groups or b not in groups:
            continue
        dists = np.minimum(
            np.linalg.norm(groups[a] - groups[b], axis=1),
            np.linalg.norm(groups[a] + groups[b], axis=1),
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=dists.tolist(),
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"{label}<br>Epoch %{{x}}<br>Distance: %{{y:.3f}}<extra></extra>",
            )
        )

    fig.add_vline(
        x=current_epoch,
        line_dash="solid",
        line_color="red",
        line_width=2,
        annotation_text=f"Epoch {current_epoch}",
        annotation_position="top right",
        annotation_font_color="red",
    )

    if title is None:
        title = f"Group Trajectory Proximity (PC{pc_x}/PC{pc_y}, normalized)"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="L2 distance (normalized)",
        yaxis=dict(rangemode="tozero"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def get_group_label(group: str) -> str:
    """Get display label for a component group key."""
    return _GROUP_LABELS.get(group, group.capitalize())
