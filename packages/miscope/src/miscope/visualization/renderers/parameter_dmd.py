"""REQ_117 phase 2b: Parameter DMD visualization renderers.

Four views over the parameter_dmd cross_epoch artifact, mirroring the
shape of activation_dmd's renderers but operating on the per-(group,
matrix) axis structure:

- render_parameter_dmd_residuals_with_regimes:
    Per-(group, matrix) residual norm trajectories with threshold +
    regime boundaries. Rows = populated groups, cols = (W_in, W_out).
- render_parameter_dmd_eigenvalue_migration:
    Per-(group, matrix) eigenvalue migration in the complex plane,
    auto-zoomed per panel. Rows × cols layout matches the residuals plot.
- render_parameter_dmd_per_regime_vs_windowed:
    Windowed mean residual vs. per-regime DMD residual on a log-y scale.
    Rows × cols layout matches.
- render_parameter_dmd_track_trajectories:
    Per-track |lambda| (top) and arg(lambda) (bottom) for one selected
    (group, matrix) pair. Single 2-row plot, group + matrix selected
    via dashboard dropdowns.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_MATRICES = ["W_in", "W_out"]

_MATRIX_LABELS = {
    "W_in": "W_in",
    "W_out": "W_out",
}

_MATRIX_COLORS = {
    "W_in": "#1f77b4",
    "W_out": "#d62728",
}


# ── helpers ──────────────────────────────────────────────────────────


def _populated_groups(cross_epoch_data: dict[str, np.ndarray]) -> list[int]:
    """Sorted list of group IDs with ≥1 assigned neuron."""
    return [int(g) for g in cross_epoch_data["populated_groups"]]


def _group_neuron_count(cross_epoch_data: dict[str, np.ndarray], group_id: int) -> int:
    """Lookup n_neurons for a populated group."""
    populated = cross_epoch_data["populated_groups"]
    counts = cross_epoch_data["group_n_neurons"]
    for g, n in zip(populated, counts):
        if int(g) == int(group_id):
            return int(n)
    return 0


def _group_label(group_id: int, n_neurons: int) -> str:
    """Modadd-friendly label: group_id corresponds to dominant-frequency k - 1."""
    return f"group {int(group_id)} (k={int(group_id) + 1}) — {n_neurons} neurons"


# ── Residuals with regimes ────────────────────────────────────────────


def render_parameter_dmd_residuals_with_regimes(
    cross_epoch_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Per-(group, matrix) residual norm + threshold + regime boundaries.

    Rows = populated groups, columns = (W_in, W_out). Each panel:
    line is per-window mean residual; dotted horizontal line is the
    boundary-detection threshold; black dashed vertical lines are
    detected regime boundaries; optional red vertical line marks
    `current_epoch`.

    Args:
        cross_epoch_data: From `variant.artifacts.load_cross_epoch("parameter_dmd")`.
        current_epoch: If provided, draws a red cursor on each subplot.
        height: Figure height. ``None`` auto-scales to ``300 * n_groups``.

    Returns:
        Plotly Figure with ``n_groups × 2`` panels.
    """
    populated = _populated_groups(cross_epoch_data)
    n_groups = len(populated)
    epochs = cross_epoch_data["epochs"]

    titles = []
    for g in populated:
        n = _group_neuron_count(cross_epoch_data, g)
        for matrix in _MATRICES:
            titles.append(f"{_group_label(g, n)} — {_MATRIX_LABELS[matrix]}")

    fig = make_subplots(
        rows=max(n_groups, 1),
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.08,
        subplot_titles=titles,
    )

    for r, g in enumerate(populated, start=1):
        for c, matrix in enumerate(_MATRICES, start=1):
            prefix = f"group_{int(g)}__{matrix}"
            starts = cross_epoch_data[f"{prefix}__windowed__window_starts"]
            ends = cross_epoch_data[f"{prefix}__windowed__window_ends"]
            centers = (starts + ends) // 2
            x = epochs[centers]
            y = cross_epoch_data[f"{prefix}__windowed__residual_norm_mean"]
            threshold = float(cross_epoch_data[f"{prefix}__regimes__threshold_used"])
            boundaries = cross_epoch_data[f"{prefix}__regimes__boundary_indices"]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=_MATRIX_COLORS[matrix], width=1.5),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="gray",
                row=r,  # type: ignore[reportArgumentType]
                col=c,  # type: ignore[reportArgumentType]
            )
            for b in boundaries:
                b_epoch = int(epochs[centers[int(b)]])
                fig.add_vline(
                    x=b_epoch,
                    line_dash="dash",
                    line_color="black",
                    opacity=0.5,
                    row=r,  # type: ignore[reportArgumentType]
                    col=c,  # type: ignore[reportArgumentType]
                )
            if current_epoch is not None:
                fig.add_vline(
                    x=current_epoch,
                    line_dash="solid",
                    line_color="red",
                    line_width=1,
                    row=r,  # type: ignore[reportArgumentType]
                    col=c,  # type: ignore[reportArgumentType]
                )
            fig.update_yaxes(title_text="residual", row=r, col=c)

    for c in (1, 2):
        fig.update_xaxes(title_text="epoch", row=max(n_groups, 1), col=c)

    ref_epoch = int(cross_epoch_data["reference_epoch"])
    fig.update_layout(
        title=(
            f"Parameter DMD — windowed residual + regime boundaries (reference_epoch={ref_epoch})"
        ),
        template="plotly_white",
        height=height if height is not None else max(300 * max(n_groups, 1), 600),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


# ── Eigenvalue migration ──────────────────────────────────────────────


def render_parameter_dmd_eigenvalue_migration(
    cross_epoch_data: dict[str, np.ndarray],
    height: int | None = None,
) -> go.Figure:
    """Per-(group, matrix) eigenvalue migration with per-panel auto-zoom.

    Rows × cols match the residuals plot. The unit circle is drawn as a
    reference; at typical zoom levels it appears as an arc on the right
    side of each panel.
    """
    populated = _populated_groups(cross_epoch_data)
    n_groups = len(populated)
    epochs = cross_epoch_data["epochs"]
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = np.cos(theta), np.sin(theta)

    titles = []
    for g in populated:
        n = _group_neuron_count(cross_epoch_data, g)
        for matrix in _MATRICES:
            titles.append(f"{_group_label(g, n)} — {_MATRIX_LABELS[matrix]}")

    fig = make_subplots(
        rows=max(n_groups, 1),
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
    )

    cell_idx = 0
    for r, g in enumerate(populated, start=1):
        for c, matrix in enumerate(_MATRICES, start=1):
            prefix = f"group_{int(g)}__{matrix}"
            eigs = cross_epoch_data[f"{prefix}__windowed__eigenvalues"]
            n_modes_per_window = cross_epoch_data[f"{prefix}__windowed__n_modes_per_window"]
            starts = cross_epoch_data[f"{prefix}__windowed__window_starts"]
            ends = cross_epoch_data[f"{prefix}__windowed__window_ends"]
            center_epochs = epochs[(starts + ends) // 2]

            all_real: list[float] = []
            all_imag: list[float] = []
            all_epoch: list[int] = []
            for w_idx in range(len(starts)):
                k = int(n_modes_per_window[w_idx])
                valid = eigs[w_idx, :k]
                all_real.extend(valid.real.tolist())  # pyright: ignore[reportAttributeAccessIssue]
                all_imag.extend(valid.imag.tolist())  # pyright: ignore[reportAttributeAccessIssue]
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
                row=r,
                col=c,
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
                        showscale=(cell_idx == 0),
                        colorbar=dict(title="epoch", len=0.4) if cell_idx == 0 else None,
                    ),
                    showlegend=False,
                    cliponaxis=True,
                    customdata=all_epoch,
                    hovertemplate=(
                        "Re(λ)=%{x:.4f}<br>Im(λ)=%{y:.4f}<br>epoch=%{customdata}<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )
            fig.update_xaxes(title_text="Re(λ)", row=r, col=c, range=x_range, zeroline=True)
            fig.update_yaxes(title_text="Im(λ)", row=r, col=c, range=y_range, zeroline=True)
            cell_idx += 1

    ref_epoch = int(cross_epoch_data["reference_epoch"])
    fig.update_layout(
        title=(f"Parameter DMD — eigenvalue migration (auto-zoom) (reference_epoch={ref_epoch})"),
        template="plotly_white",
        height=height if height is not None else max(380 * max(n_groups, 1), 600),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


# ── Per-track trajectories (selected group + matrix) ─────────────────


def render_parameter_dmd_track_trajectories(
    cross_epoch_data: dict[str, np.ndarray],
    group_id: int,
    matrix: str = "W_in",
    current_epoch: int | None = None,
    height: int = 600,
) -> go.Figure:
    """|lambda| (top) and arg(lambda) (bottom) per tracked mode for one
    (group, matrix) pair.

    Args:
        cross_epoch_data: From `variant.artifacts.load_cross_epoch("parameter_dmd")`.
        group_id: Which populated group to render.
        matrix: ``"W_in"`` or ``"W_out"``.
        current_epoch: If provided, red cursor line on each subplot.
        height: Figure height in pixels.
    """
    if matrix not in _MATRICES:
        raise ValueError(f"unknown matrix '{matrix}'; valid: {_MATRICES}")
    prefix = f"group_{int(group_id)}__{matrix}"
    n_neurons = _group_neuron_count(cross_epoch_data, int(group_id))

    eigs = cross_epoch_data[f"{prefix}__windowed__eigenvalues"]
    track_ids = cross_epoch_data[f"{prefix}__tracks__track_ids"]
    n_modes_per_window = cross_epoch_data[f"{prefix}__windowed__n_modes_per_window"]
    starts = cross_epoch_data[f"{prefix}__windowed__window_starts"]
    ends = cross_epoch_data[f"{prefix}__windowed__window_ends"]
    epochs = cross_epoch_data["epochs"]
    center_epochs = epochs[(starts + ends) // 2]
    n_tracks = int(cross_epoch_data[f"{prefix}__tracks__n_tracks"])

    track_data: dict[int, dict[str, list]] = {t: {"epoch": [], "eig": []} for t in range(n_tracks)}
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
        for row in (1, 2):
            fig.add_vline(
                x=current_epoch,
                line_dash="solid",
                line_color="red",
                line_width=1,
                row=row,  # type: ignore[reportArgumentType]
                col=1,  # type: ignore[reportArgumentType]
            )

    fig.update_yaxes(title_text="|λ|", row=1, col=1)
    fig.update_yaxes(title_text="arg(λ)", row=2, col=1, range=[-np.pi, np.pi])
    fig.update_xaxes(title_text="epoch", row=2, col=1)
    fig.update_layout(
        title=(
            f"Parameter DMD — eigenvalue track trajectories — "
            f"{_group_label(int(group_id), n_neurons)} — {_MATRIX_LABELS[matrix]}"
        ),
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


# ── Per-regime vs windowed ────────────────────────────────────────────


def render_parameter_dmd_per_regime_vs_windowed(
    cross_epoch_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Windowed mean residual (line) vs. per-regime DMD residual (dashed bands).

    Rows × cols match the residuals plot. Per-regime residuals appear as
    horizontal dashed black bands spanning each detected segment.
    """
    populated = _populated_groups(cross_epoch_data)
    n_groups = len(populated)
    epochs = cross_epoch_data["epochs"]
    titles = []
    for g in populated:
        n = _group_neuron_count(cross_epoch_data, g)
        for matrix in _MATRICES:
            titles.append(f"{_group_label(g, n)} — {_MATRIX_LABELS[matrix]}")

    fig = make_subplots(
        rows=max(n_groups, 1),
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
    )

    for r, g in enumerate(populated, start=1):
        for c, matrix in enumerate(_MATRICES, start=1):
            prefix = f"group_{int(g)}__{matrix}"
            w_starts = cross_epoch_data[f"{prefix}__windowed__window_starts"]
            w_ends = cross_epoch_data[f"{prefix}__windowed__window_ends"]
            w_residual = cross_epoch_data[f"{prefix}__windowed__residual_norm_mean"]
            w_x = epochs[(w_starts + w_ends) // 2]
            pr_starts = cross_epoch_data[f"{prefix}__per_regime__segment_starts"]
            pr_ends = cross_epoch_data[f"{prefix}__per_regime__segment_ends"]
            pr_residual = cross_epoch_data[f"{prefix}__per_regime__residual_norm_mean"]

            fig.add_trace(
                go.Scatter(
                    x=w_x,
                    y=w_residual,
                    mode="lines",
                    line=dict(color=_MATRIX_COLORS[matrix], width=1),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            for s, e, res in zip(pr_starts, pr_ends, pr_residual):
                if np.isnan(res):
                    continue
                x_seg = [int(epochs[s]), int(epochs[min(e - 1, len(epochs) - 1)])]
                fig.add_trace(
                    go.Scatter(
                        x=x_seg,
                        y=[res, res],
                        mode="lines",
                        line=dict(color="black", width=2, dash="dash"),
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )
            if current_epoch is not None:
                fig.add_vline(
                    x=current_epoch,
                    line_dash="solid",
                    line_color="red",
                    line_width=1,
                    row=r,  # type: ignore[reportArgumentType]
                    col=c,  # type: ignore[reportArgumentType]
                )
            fig.update_yaxes(title_text="mean residual", row=r, col=c, type="log")
            fig.update_xaxes(title_text="epoch", row=r, col=c)

    ref_epoch = int(cross_epoch_data["reference_epoch"])
    fig.update_layout(
        title=(
            "Parameter DMD — windowed vs. per-regime residual (log y) "
            f"(reference_epoch={ref_epoch})"
        ),
        template="plotly_white",
        height=height if height is not None else max(300 * max(n_groups, 1), 600),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig
