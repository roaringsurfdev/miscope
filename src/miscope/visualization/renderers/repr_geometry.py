"""REQ_044/REQ_045: Representational Geometry Visualizations.

Renders time-series of geometric measures (SNR, circularity, Fisher, etc.)
from summary data, and centroid PCA snapshots + distance/Fisher heatmaps
from per-epoch data.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from miscope.analysis.library.geometry import _pca_project, compute_fisher_matrix

# Colors for activation sites
_SITE_COLORS = {
    "resid_pre": "rgba(31, 119, 180, 1.0)",
    "attn_out": "rgba(44, 160, 44, 1.0)",
    "mlp_out": "rgba(214, 39, 40, 1.0)",
    "resid_post": "rgba(148, 103, 189, 1.0)",
}

_SITE_LABELS = {
    "resid_pre": "Post-Embed",
    "attn_out": "Attn Out",
    "mlp_out": "MLP Out",
    "resid_post": "Resid Post",
}

_ALL_SITES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]


def render_geometry_timeseries(
    summary_data: dict[str, np.ndarray],
    site: str | None = None,
    current_epoch: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Multi-panel time-series of geometric measures.

    Panels: SNR, center spread + mean radius, circularity + Fourier
    alignment, mean dimensionality, Fisher discriminant (mean + min),
    and optionally Fisher argmin residue difference (if available).

    Args:
        summary_data: From ArtifactLoader.load_summary("repr_geometry").
            Contains "epochs" and site-prefixed scalar keys.
        site: Activation site to display. None = show all sites.
        current_epoch: Current epoch for vertical indicator.
        height: Total figure height in pixels. Auto-sized if None.

    Returns:
        Plotly Figure with 5 or 6 vertically stacked subplots.
    """
    epochs = summary_data["epochs"]
    sites = [site] if site else _ALL_SITES

    # Check if argmin data is available (added by REQ_045)
    has_argmin = any(f"{s}_fisher_argmin_diff" in summary_data for s in sites)
    n_rows = 6 if has_argmin else 5
    if height is None:
        height = 1400 if has_argmin else 1200

    titles = [
        "Signal-to-Noise Ratio (SNR)",
        "Center Spread & Mean Radius",
        "Circularity & Fourier Alignment",
        "Mean Effective Dimensionality",
        "Fisher Discriminant",
    ]
    if has_argmin:
        titles.append("Fisher Argmin Residue Difference")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=titles,
    )

    for s in sites:
        color = _SITE_COLORS.get(s, "gray")
        label = _SITE_LABELS.get(s, s)
        show_legend = True

        # Panel 1: SNR (log scale)
        key = f"{s}_snr"
        if key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[key],
                    mode="lines",
                    name=label,
                    legendgroup=s,
                    showlegend=show_legend,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label}<br>Epoch %{{x}}<br>SNR: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            show_legend = False

        # Panel 2: Center spread + mean radius
        spread_key = f"{s}_center_spread"
        radius_key = f"{s}_mean_radius"
        if spread_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[spread_key],
                    mode="lines",
                    name=f"{label} spread",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label} spread<br>Epoch %{{x}}<br>%{{y:.4f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )
        if radius_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[radius_key],
                    mode="lines",
                    name=f"{label} radius",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dash"),
                    hovertemplate=f"{label} radius<br>Epoch %{{x}}<br>%{{y:.4f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Panel 3: Circularity + Fourier alignment
        circ_key = f"{s}_circularity"
        fourier_key = f"{s}_fourier_alignment"
        if circ_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[circ_key],
                    mode="lines",
                    name=f"{label} circ",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label} circularity<br>Epoch %{{x}}<br>%{{y:.3f}}<extra></extra>",
                ),
                row=3,
                col=1,
            )
        if fourier_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[fourier_key],
                    mode="lines",
                    name=f"{label} Fourier",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dot"),
                    hovertemplate=f"{label} Fourier align<br>Epoch %{{x}}<br>%{{y:.3f}}<extra></extra>",
                ),
                row=3,
                col=1,
            )

        # Panel 4: Mean dimensionality
        dim_key = f"{s}_mean_dim"
        if dim_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[dim_key],
                    mode="lines",
                    name=f"{label} dim",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label} mean dim<br>Epoch %{{x}}<br>%{{y:.1f}}<extra></extra>",
                ),
                row=4,
                col=1,
            )

        # Panel 5: Fisher discriminant (mean solid, min dashed)
        fisher_mean_key = f"{s}_fisher_mean"
        fisher_min_key = f"{s}_fisher_min"
        if fisher_mean_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[fisher_mean_key],
                    mode="lines",
                    name=f"{label} Fisher mean",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label} Fisher mean<br>Epoch %{{x}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=5,
                col=1,
            )
        if fisher_min_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[fisher_min_key],
                    mode="lines",
                    name=f"{label} Fisher min",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dash"),
                    hovertemplate=f"{label} Fisher min<br>Epoch %{{x}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=5,
                col=1,
            )

        # Panel 6: Fisher argmin residue difference (REQ_045)
        argmin_diff_key = f"{s}_fisher_argmin_diff"
        argmin_r_key = f"{s}_fisher_argmin_r"
        argmin_s_key = f"{s}_fisher_argmin_s"
        if has_argmin and argmin_diff_key in summary_data:
            # Build hover text with the actual (r, s) pair
            custom = None
            if argmin_r_key in summary_data and argmin_s_key in summary_data:
                r_vals = summary_data[argmin_r_key].astype(int)
                s_vals = summary_data[argmin_s_key].astype(int)
                custom = np.column_stack([r_vals, s_vals])
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[argmin_diff_key],
                    mode="lines+markers",
                    name=f"{label} argmin |r-s|",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2),
                    marker=dict(size=3),
                    customdata=custom,
                    hovertemplate=(
                        f"{label}<br>Epoch %{{x}}<br>"
                        "|r*-s*| mod p: %{y:.0f}<br>"
                        "Pair: (%{customdata[0]}, %{customdata[1]})"
                        "<extra></extra>"
                    ),
                ),
                row=6,
                col=1,
            )

    # Add epoch indicator
    if current_epoch is not None:
        for row in range(1, n_rows + 1):
            fig.add_vline(
                x=current_epoch,
                line_dash="solid",
                line_color="red",
                line_width=1,
                row=row,  # type: ignore[reportArgumentType]
                col=1,  # type: ignore[reportArgumentType]
            )

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=3, col=1)
    fig.update_xaxes(title_text="Epoch", row=n_rows, col=1)

    fig.update_layout(
        template="plotly_white",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return fig


def render_centroid_pca(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    site: str = "resid_post",
    p: int | None = None,
    height: int = 800,
) -> go.Figure:
    """PCA scatter of class centroids at a single epoch.

    Shows a 2x2 grid: PC1-PC2, PC1-PC3, PC2-PC3, and a 3D scatter.
    Colors by residue class using a cyclic colormap.

    Args:
        epoch_data: From ArtifactLoader.load_epoch("repr_geometry", epoch).
        epoch: Epoch number (for title).
        site: Activation site to display.
        p: Number of classes (inferred from centroid shape if None).
        height: Figure height in pixels.

    Returns:
        Plotly Figure with centroid PCA subplots.
    """
    centroid_key = f"{site}_centroids"
    centroids = epoch_data[centroid_key]
    if p is None:
        p = int(centroids.shape[0])

    projected, var_fracs = _pca_project(centroids, n_components=3)
    residues = np.arange(p)
    labels = [str(r) for r in residues]
    total_var = float(var_fracs.sum())

    # 2D pair panels
    pc_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_labels = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "scene"}],
        ],
        subplot_titles=[
            f"PC1 vs PC2 ({var_fracs[0]:.1%} + {var_fracs[1]:.1%})",
            f"PC1 vs PC3 ({var_fracs[0]:.1%} + {var_fracs[2]:.1%})",
            f"PC2 vs PC3 ({var_fracs[1]:.1%} + {var_fracs[2]:.1%})",
            f"3D ({total_var:.1%} total)",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
    )

    positions = [(1, 1), (1, 2), (2, 1)]
    for (pc_a, pc_b), (xl, yl), (row, col) in zip(pc_pairs, pair_labels, positions):
        fig.add_trace(
            go.Scatter(
                x=projected[:, pc_a],
                y=projected[:, pc_b],
                mode="markers+text",
                marker=dict(
                    size=8,
                    color=residues,
                    colorscale="HSV",
                    showscale=False,
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=7),
                hovertemplate=(
                    f"Residue %{{text}}<br>{xl}: %{{x:.3f}}<br>{yl}: %{{y:.3f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=xl, row=row, col=col)
        fig.update_yaxes(
            title_text=yl,
            scaleanchor=f"x{(row - 1) * 2 + col}" if row == 1 and col == 1 else None,
            row=row,
            col=col,
        )

    # 3D scatter
    fig.add_trace(
        go.Scatter3d(
            x=projected[:, 0],
            y=projected[:, 1],
            z=projected[:, 2],
            mode="markers+text",
            marker=dict(
                size=4,
                color=residues,
                colorscale="HSV",
                showscale=True,
                colorbar=dict(title="Residue", x=1.02, len=0.4, y=0.2),
            ),
            text=labels,
            textfont=dict(size=6),
            hovertemplate=(
                "Residue %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
    )

    site_label = _SITE_LABELS.get(site, site)
    fig.update_layout(
        title=f"Class Centroids PCA — {site_label} — Epoch {epoch}",
        template="plotly_white",
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def render_centroid_distances(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    site: str = "resid_post",
    p: int | None = None,
    height: int = 500,
) -> go.Figure:
    """Pairwise centroid distance heatmap at a single epoch.

    For learned modular structure, this should show a circulant pattern
    where distance depends on |r - s| mod p.

    Args:
        epoch_data: From ArtifactLoader.load_epoch("repr_geometry", epoch).
        epoch: Epoch number (for title).
        site: Activation site to display.
        p: Number of classes (inferred from centroid shape if None).
        height: Figure height in pixels.

    Returns:
        Plotly Figure with p×p distance heatmap.
    """
    centroid_key = f"{site}_centroids"
    centroids = epoch_data[centroid_key]
    if p is None:
        p = int(centroids.shape[0])

    # Compute pairwise distances
    diffs = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs**2, axis=2))

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=distances,
            x=list(range(p)),
            y=list(range(p)),
            colorscale="Viridis",
            colorbar=dict(title="Distance"),
            hovertemplate="Class %{x} ↔ Class %{y}<br>Distance: %{z:.3f}<extra></extra>",
        )
    )

    site_label = _SITE_LABELS.get(site, site)
    fig.update_layout(
        title=f"Centroid Pairwise Distances — {site_label} — Epoch {epoch}",
        xaxis_title="Residue Class",
        yaxis_title="Residue Class",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
        yaxis=dict(autorange="reversed"),
    )

    return fig


def render_centroid_pca_variance_summary(
    summary_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    site: str | None = None,
    height: int = 600,
) -> go.Figure:
    """Time-series of centroid class PCA variance explained per PC over training.

    Summary-based variant — reads pre-computed {site}_pca_var_pc1/2/3 keys
    from summary data. Faster than the stacked-epoch version since no per-epoch
    PCA computation occurs at render time.

    Three panels (PC1, PC2, PC3), one line per activation site.

    Args:
        summary_data: From ArtifactLoader.load_summary("repr_geometry").
            Must contain "{site}_pca_var_pc1/2/3" keys (added after reanalysis).
        current_epoch: Current epoch for vertical indicator.
        site: Single activation site to display. None = show all sites.
        height: Total figure height in pixels.

    Returns:
        Plotly Figure with 3 vertically stacked subplots.
    """
    epochs = summary_data["epochs"]
    sites = [site] if site else _ALL_SITES

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["PC1 Variance Explained", "PC2 Variance Explained", "PC3 Variance Explained"],
    )

    for s in sites:
        color = _SITE_COLORS.get(s, "gray")
        label = _SITE_LABELS.get(s, s)

        for pc_idx in range(3):
            key = f"{s}_pca_var_pc{pc_idx + 1}"
            if key not in summary_data:
                continue
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[key] * 100,
                    mode="lines",
                    name=label,
                    legendgroup=s,
                    showlegend=(pc_idx == 0),
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        f"{label}<br>Epoch %{{x}}<br>"
                        f"PC{pc_idx + 1}: %{{y:.1f}}%<extra></extra>"
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


def render_centroid_pca_variance(
    stacked_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    site: str | None = None,
    height: int = 600,
) -> go.Figure:
    """Time-series of centroid class PCA variance explained per PC over training.

    Three panels (PC1, PC2, PC3), one line per activation site. Shows how the
    model's representational geometry redistributes across principal components
    as training progresses — e.g., the expansion into higher dimensions during
    grokking.

    Args:
        stacked_data: From ArtifactLoader.load_epochs("repr_geometry").
            Contains "epochs" and stacked "{site}_centroids" arrays (N, p, d).
        current_epoch: Current epoch for vertical indicator.
        site: Single activation site to display. None = show all sites.
        height: Total figure height in pixels.

    Returns:
        Plotly Figure with 3 vertically stacked subplots.
    """
    epochs = stacked_data["epochs"]
    sites = [site] if site else _ALL_SITES

    site_var_fracs: dict[str, np.ndarray] = {}
    for s in sites:
        centroid_key = f"{s}_centroids"
        if centroid_key not in stacked_data:
            continue
        all_centroids = stacked_data[centroid_key]  # (n_epochs, p, d)
        n_epochs = len(epochs)
        var_fracs = np.zeros((n_epochs, 3))
        for i in range(n_epochs):
            _, fracs = _pca_project(all_centroids[i], n_components=3)
            var_fracs[i] = fracs
        site_var_fracs[s] = var_fracs

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["PC1 Variance Explained", "PC2 Variance Explained", "PC3 Variance Explained"],
    )

    for s in sites:
        if s not in site_var_fracs:
            continue
        color = _SITE_COLORS.get(s, "gray")
        label = _SITE_LABELS.get(s, s)
        var_fracs = site_var_fracs[s]

        for pc_idx in range(3):
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=var_fracs[:, pc_idx] * 100,
                    mode="lines",
                    name=label,
                    legendgroup=s,
                    showlegend=(pc_idx == 0),
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        f"{label}<br>Epoch %{{x}}<br>"
                        f"PC{pc_idx + 1}: %{{y:.1f}}%<extra></extra>"
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


def render_fisher_heatmap(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    site: str = "resid_post",
    p: int | None = None,
    height: int = 500,
) -> go.Figure:
    """Pairwise Fisher discriminant heatmap at a single epoch.

    Recomputes J(r,s) = ||mu_r - mu_s||^2 / (radius_r^2 + radius_s^2)
    from stored centroids and radii. Low values (cold spots) indicate
    the hardest-to-separate class pairs — the model's vulnerability.

    Args:
        epoch_data: From ArtifactLoader.load_epoch("repr_geometry", epoch).
        epoch: Epoch number (for title).
        site: Activation site to display.
        p: Number of classes (inferred from centroid shape if None).
        height: Figure height in pixels.

    Returns:
        Plotly Figure with p x p Fisher discriminant heatmap.
    """
    centroid_key = f"{site}_centroids"
    radii_key = f"{site}_radii"
    centroids = epoch_data[centroid_key]
    radii = epoch_data[radii_key]
    if p is None:
        p = int(centroids.shape[0])

    fisher_mat = compute_fisher_matrix(centroids, radii)

    # Find argmin pair for annotation
    r_idx, s_idx = np.triu_indices(p, k=1)
    fisher_upper = fisher_mat[r_idx, s_idx]
    argmin_idx = int(np.argmin(fisher_upper))
    argmin_r = int(r_idx[argmin_idx])
    argmin_s = int(s_idx[argmin_idx])
    argmin_val = float(fisher_upper[argmin_idx])
    raw_diff = abs(argmin_s - argmin_r)
    argmin_diff = min(raw_diff, p - raw_diff)

    # Build customdata with |r-s| mod p for hover (vectorized)
    idx = np.arange(p)
    raw_diffs = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
    residue_diffs = np.minimum(raw_diffs, p - raw_diffs)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=fisher_mat,
            x=list(range(p)),
            y=list(range(p)),
            colorscale="Inferno",
            reversescale=True,
            colorbar=dict(title="Fisher J"),
            customdata=residue_diffs,
            hovertemplate=(
                "Class %{x} ↔ Class %{y}<br>"
                "J: %{z:.3f}<br>"
                "|r-s| mod p: %{customdata}"
                "<extra></extra>"
            ),
        )
    )

    # Mark the argmin pair
    fig.add_trace(
        go.Scatter(
            x=[argmin_s, argmin_r],
            y=[argmin_r, argmin_s],
            mode="markers",
            marker=dict(
                size=10,
                color="rgba(0,0,0,0)",
                line=dict(color="lime", width=2),
                symbol="square",
            ),
            name=f"Min pair ({argmin_r},{argmin_s})",
            hovertemplate=(
                f"Argmin pair: ({argmin_r}, {argmin_s})<br>"
                f"J = {argmin_val:.4f}<br>"
                f"|r-s| mod p = {argmin_diff}"
                "<extra></extra>"
            ),
        )
    )

    site_label = _SITE_LABELS.get(site, site)
    fig.update_layout(
        title=(
            f"Fisher Discriminant — {site_label} — Epoch {epoch}"
            f"<br><sub>Min pair: ({argmin_r}, {argmin_s}), "
            f"J={argmin_val:.3f}, |r-s|={argmin_diff}</sub>"
        ),
        xaxis_title="Residue Class",
        yaxis_title="Residue Class",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=70, b=50),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )

    return fig
