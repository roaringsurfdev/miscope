"""REQ_117: Activation DMD visualization renderers.

Four views over the activation_dmd cross_epoch artifact:

- render_activation_dmd_residuals_with_regimes:
    Per-site stacked residual norm trajectories with threshold + regime
    boundaries marked. Primary diagnostic.
- render_activation_dmd_eigenvalue_migration:
    Per-site eigenvalue migration in the complex plane, auto-zoomed to
    each subplot's actual eigenvalue cluster so per-variant signatures
    are visible.
- render_activation_dmd_track_trajectories:
    Per-track |lambda| (top) and arg(lambda) (bottom) for one selected site.
- render_activation_dmd_per_regime_vs_windowed:
    Windowed mean residual vs. per-regime DMD residual on a log-y scale,
    one panel per site.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ALL_SITES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]

_SITE_LABELS = {
    "resid_pre": "Post-Embed",
    "attn_out": "Attn Out",
    "mlp_out": "MLP Out",
    "resid_post": "Resid Post",
}

_SITE_COLORS = {
    "resid_pre": "#1f77b4",
    "attn_out": "#2ca02c",
    "mlp_out": "#d62728",
    "resid_post": "#9467bd",
}


# ── Residuals with regimes ────────────────────────────────────────────


def render_activation_dmd_residuals_with_regimes(
    cross_epoch_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    height: int = 900,
) -> go.Figure:
    """Per-site residual norm + threshold + regime boundaries, stacked.

    The headline diagnostic. Each row is one of the four activation sites;
    the line is the windowed-DMD per-window mean residual; the dotted
    horizontal line is the boundary-detection threshold; the black dashed
    vertical lines are the detected regime boundaries.

    Args:
        cross_epoch_data: From `variant.artifacts.load_cross_epoch("activation_dmd")`.
        current_epoch: If provided, draws a red cursor line on each subplot
            at this epoch.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with 4 stacked rows, shared x-axis.
    """
    epochs = cross_epoch_data["epochs"]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[_SITE_LABELS[s] for s in _ALL_SITES],
    )

    for i, site in enumerate(_ALL_SITES):
        starts = cross_epoch_data[f"{site}__windowed__window_starts"]
        ends = cross_epoch_data[f"{site}__windowed__window_ends"]
        centers = (starts + ends) // 2
        x = epochs[centers]
        y = cross_epoch_data[f"{site}__windowed__residual_norm_mean"]
        threshold = float(cross_epoch_data[f"{site}__regimes__threshold_used"])
        boundaries = cross_epoch_data[f"{site}__regimes__boundary_indices"]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=_SITE_LABELS[site],
                line=dict(color=_SITE_COLORS[site], width=1.5),
                showlegend=False,
            ),
            row=i + 1,
            col=1,
        )
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"threshold={threshold:.2f}",
            annotation_position="top right",
            row=i + 1,  # type: ignore[reportArgumentType]
            col=1,  # type: ignore[reportArgumentType]
        )
        for b in boundaries:
            b_epoch = int(epochs[centers[int(b)]])
            fig.add_vline(
                x=b_epoch,
                line_dash="dash",
                line_color="black",
                opacity=0.5,
                row=i + 1,  # type: ignore[reportArgumentType]
                col=1,  # type: ignore[reportArgumentType]
            )
        if current_epoch is not None:
            fig.add_vline(
                x=current_epoch,
                line_dash="solid",
                line_color="red",
                line_width=1,
                row=i + 1,  # type: ignore[reportArgumentType]
                col=1,  # type: ignore[reportArgumentType]
            )
        fig.update_yaxes(title_text="residual", row=i + 1, col=1)

    fig.update_xaxes(title_text="epoch", row=4, col=1)
    fig.update_layout(
        title="Activation DMD — windowed residual + regime boundaries",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


# ── Eigenvalue migration ──────────────────────────────────────────────


def render_activation_dmd_eigenvalue_migration(
    cross_epoch_data: dict[str, np.ndarray],
    height: int = 850,
) -> go.Figure:
    """Per-site eigenvalue migration on the complex plane.

    Each subplot auto-zooms to its actual eigenvalue cluster (with 10%
    padding), so the small-radius migration patterns near (1, 0) are
    visible without manual zooming. The unit circle is drawn for
    reference; at typical zoom levels it appears as an arc entering the
    visible region from the right side.

    Args:
        cross_epoch_data: From `variant.artifacts.load_cross_epoch("activation_dmd")`.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with 2x2 subplots.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[_SITE_LABELS[s] for s in _ALL_SITES],
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )
    epochs = cross_epoch_data["epochs"]
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = np.cos(theta), np.sin(theta)

    for i, site in enumerate(_ALL_SITES):
        row, col = (i // 2) + 1, (i % 2) + 1
        eigs = cross_epoch_data[f"{site}__windowed__eigenvalues"]
        n_modes_per_window = cross_epoch_data[f"{site}__windowed__n_modes_per_window"]
        starts = cross_epoch_data[f"{site}__windowed__window_starts"]
        ends = cross_epoch_data[f"{site}__windowed__window_ends"]
        center_epochs = epochs[(starts + ends) // 2]

        all_real: list[float] = []
        all_imag: list[float] = []
        all_epoch: list[int] = []
        for w_idx in range(len(starts)):
            k = int(n_modes_per_window[w_idx])
            valid = eigs[w_idx, :k]
            all_real.extend(valid.real.tolist())
            all_imag.extend(valid.imag.tolist())
            all_epoch.extend([int(center_epochs[w_idx])] * k)

        if all_real:
            r_lo, r_hi = min(all_real), max(all_real)
            i_lo, i_hi = min(all_imag), max(all_imag)
            r_pad = max((r_hi - r_lo) * 0.10, 0.05)
            i_pad = max((i_hi - i_lo) * 0.10, 0.05)
            x_range = [r_lo - r_pad, r_hi + r_pad]
            y_range = [i_lo - i_pad, i_hi + i_pad]
        else:
            x_range = [-1.5, 1.5]
            y_range = [-1.5, 1.5]

        fig.add_trace(
            go.Scatter(
                x=cx,
                y=cy,
                mode="lines",
                line=dict(color="lightgray", width=1),
                hoverinfo="skip",
                showlegend=False,
                cliponaxis=True,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=all_real,
                y=all_imag,
                mode="markers",
                marker=dict(
                    size=4,
                    color=all_epoch,
                    colorscale="Viridis",
                    showscale=(i == 0),
                    colorbar=dict(title="epoch", len=0.4) if i == 0 else None,
                ),
                showlegend=False,
                name=site,
                cliponaxis=True,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            title_text="Re(λ)", row=row, col=col, range=x_range, zeroline=True
        )
        fig.update_yaxes(
            title_text="Im(λ)", row=row, col=col, range=y_range, zeroline=True
        )

    fig.update_layout(
        title="Activation DMD — eigenvalue migration across training (auto-zoom)",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


# ── Per-track trajectories ────────────────────────────────────────────


def render_activation_dmd_track_trajectories(
    cross_epoch_data: dict[str, np.ndarray],
    site: str = "mlp_out",
    current_epoch: int | None = None,
    height: int = 600,
) -> go.Figure:
    """|lambda| (top) and arg(lambda) (bottom) per tracked mode.

    Each line is one mode's trajectory across the windowed DMD sequence.
    Modes that drift toward |lambda| ≈ 1 are persistent; those that stay
    smaller are transient. arg(lambda) excursions away from 0 mark
    oscillatory dynamics.

    Args:
        cross_epoch_data: From `variant.artifacts.load_cross_epoch("activation_dmd")`.
        site: Which activation site's tracks to render.
        current_epoch: If provided, draws a red cursor line on each subplot.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with 2 stacked rows.
    """
    eigs = cross_epoch_data[f"{site}__windowed__eigenvalues"]
    track_ids = cross_epoch_data[f"{site}__tracks__track_ids"]
    n_modes_per_window = cross_epoch_data[f"{site}__windowed__n_modes_per_window"]
    starts = cross_epoch_data[f"{site}__windowed__window_starts"]
    ends = cross_epoch_data[f"{site}__windowed__window_ends"]
    epochs = cross_epoch_data["epochs"]
    center_epochs = epochs[(starts + ends) // 2]
    n_tracks = int(cross_epoch_data[f"{site}__tracks__n_tracks"])

    track_data: dict[int, dict[str, list]] = {
        t: {"epoch": [], "eig": []} for t in range(n_tracks)
    }
    for w in range(len(starts)):
        k = int(n_modes_per_window[w])
        for slot in range(k):
            tid = int(track_ids[w, slot])
            if tid >= 0:
                track_data[tid]["epoch"].append(int(center_epochs[w]))
                track_data[tid]["eig"].append(complex(eigs[w, slot]))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["|λ| per track", "arg(λ) per track (radians)"],
    )

    for tid, td in track_data.items():
        if not td["epoch"]:
            continue
        eig_arr = np.array(td["eig"])
        fig.add_trace(
            go.Scatter(
                x=td["epoch"],
                y=np.abs(eig_arr),
                mode="lines",
                name=f"track {tid}",
                legendgroup=f"t{tid}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=td["epoch"],
                y=np.angle(eig_arr),
                mode="lines",
                name=f"track {tid}",
                legendgroup=f"t{tid}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=1, col=1)  # type: ignore[reportArgumentType]
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=2, col=1)  # type: ignore[reportArgumentType]

    if current_epoch is not None:
        fig.add_vline(
            x=current_epoch,
            line_dash="solid",
            line_color="red",
            line_width=1,
            row=1,  # type: ignore[reportArgumentType]
            col=1,  # type: ignore[reportArgumentType]
        )
        fig.add_vline(
            x=current_epoch,
            line_dash="solid",
            line_color="red",
            line_width=1,
            row=2,  # type: ignore[reportArgumentType]
            col=1,  # type: ignore[reportArgumentType]
        )

    fig.update_yaxes(title_text="|λ|", row=1, col=1)
    fig.update_yaxes(title_text="arg(λ)", row=2, col=1, range=[-np.pi, np.pi])
    fig.update_xaxes(title_text="epoch", row=2, col=1)
    fig.update_layout(
        title=f"Activation DMD — eigenvalue track trajectories — {_SITE_LABELS.get(site, site)}",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


# ── Per-regime vs windowed ────────────────────────────────────────────


def render_activation_dmd_per_regime_vs_windowed(
    cross_epoch_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    height: int = 750,
) -> go.Figure:
    """Windowed mean residual (line) vs. per-regime DMD residual (dashed bands).

    Per-regime DMD fits one linear operator inside each detected segment.
    The dashed black horizontal bands span each segment at the per-regime
    mean residual; the colored line is the per-window mean residual across
    the full trajectory. Where the two diverge, the per-window measurement
    captures structure within a segment that the per-regime fit averages out.

    Args:
        cross_epoch_data: From `variant.artifacts.load_cross_epoch("activation_dmd")`.
        current_epoch: If provided, draws a red cursor line on each subplot.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with 2x2 subplots, log-y.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[_SITE_LABELS[s] for s in _ALL_SITES],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )
    epochs = cross_epoch_data["epochs"]

    for i, site in enumerate(_ALL_SITES):
        row, col = (i // 2) + 1, (i % 2) + 1
        w_starts = cross_epoch_data[f"{site}__windowed__window_starts"]
        w_ends = cross_epoch_data[f"{site}__windowed__window_ends"]
        w_residual = cross_epoch_data[f"{site}__windowed__residual_norm_mean"]
        w_x = epochs[(w_starts + w_ends) // 2]
        pr_starts = cross_epoch_data[f"{site}__per_regime__segment_starts"]
        pr_ends = cross_epoch_data[f"{site}__per_regime__segment_ends"]
        pr_residual = cross_epoch_data[f"{site}__per_regime__residual_norm_mean"]

        fig.add_trace(
            go.Scatter(
                x=w_x,
                y=w_residual,
                mode="lines",
                line=dict(color=_SITE_COLORS[site], width=1),
                name=f"{_SITE_LABELS[site]} windowed",
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )
        first_band = True
        for s, e, r in zip(pr_starts, pr_ends, pr_residual):
            if np.isnan(r):
                continue
            x_seg = [int(epochs[s]), int(epochs[min(e - 1, len(epochs) - 1)])]
            fig.add_trace(
                go.Scatter(
                    x=x_seg,
                    y=[r, r],
                    mode="lines",
                    line=dict(color="black", width=2, dash="dash"),
                    name="per-regime" if (i == 0 and first_band) else "",
                    showlegend=bool(i == 0 and first_band),
                ),
                row=row,
                col=col,
            )
            first_band = False

        if current_epoch is not None:
            fig.add_vline(
                x=current_epoch,
                line_dash="solid",
                line_color="red",
                line_width=1,
                row=row,  # type: ignore[reportArgumentType]
                col=col,  # type: ignore[reportArgumentType]
            )

        fig.update_yaxes(title_text="mean residual", row=row, col=col, type="log")
        fig.update_xaxes(title_text="epoch", row=row, col=col)

    fig.update_layout(
        title="Activation DMD — windowed vs. per-regime residual (log y)",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig
