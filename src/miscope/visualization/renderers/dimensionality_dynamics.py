"""REQ_095: Dimensionality Dynamics views.

Two views measuring how the (concentration, shape) of learned representations
evolve across training:

- build_dimensionality_timeseries: three-panel epoch timeseries of PR₃ and
  f_top3 across trajectory, class centroid, and within-group weight domains.

- build_dimensionality_state_space: parametric (f_top3, PR₃) plot per
  activation site with epoch encoded as point color. Shows sequencing and
  routing that are invisible in separate timeseries.

PR₃ = (f1+f2+f3)² / (f1²+f2²+f3²), range [1, 3]:
  1 = directed motion (line / spoke)
  2 = planar (ring)
  3 = volumetric (sphere / blob)

f_top3 = (λ1+λ2+λ3) / Σλ — fraction of total variance in top 3 PCs.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Constants ---

_REPR_SITES = ["attn_out", "mlp_out", "resid_post"]
_REPR_SITE_COLORS = {
    "attn_out": "#2ca02c",
    "mlp_out": "#d62728",
    "resid_post": "#9467bd",
}
_REPR_SITE_LABELS = {"attn_out": "Attn", "mlp_out": "MLP", "resid_post": "Resid Post"}

_TRAJ_SITES = ["embedding", "attention", "mlp", "all"]
_TRAJ_SITE_COLORS = {
    "embedding": "#17becf",
    "attention": "#2ca02c",
    "mlp": "#d62728",
    "all": "#7f7f7f",
}

_GROUP_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

_REF_LINES = [1.0, 2.0, 3.0]


# --- Math helpers ---


def _compute_pr3(f1, f2, f3):
    """PR₃ from top-3 fractional eigenvalues. Range [1, 3]."""
    num = (f1 + f2 + f3) ** 2
    den = f1**2 + f2**2 + f3**2
    return np.where(den > 0, num / den, 1.0)


def _compute_rolling_trajectory_metrics(projections, window=10):
    """Rolling PR₃ and f_top3 of trajectory in PC space.

    f_top3 denominator sums ALL available PC variances — detects diffusion
    into higher PCs that PR₃ alone cannot see.

    Returns: (pr3_array, f_top3_array), both (n_epochs,)
    """
    n_epochs, _ = projections.shape
    pr3_arr = np.full(n_epochs, np.nan)
    ft_arr = np.full(n_epochs, np.nan)
    for t in range(n_epochs):
        w = projections[max(0, t - window + 1) : t + 1]
        if w.shape[0] < 2:
            continue
        var = w.var(axis=0)
        total = var.sum()
        if total < 1e-12:
            pr3_arr[t] = 1.0
            ft_arr[t] = 1.0
            continue
        ft_arr[t] = var[:3].sum() / total
        f1, f2, f3 = var[0] / total, var[1] / total, var[2] / total
        pr3_arr[t] = _compute_pr3(f1, f2, f3)
    return pr3_arr, ft_arr


def _extract_repr_dim_metrics(summary, sites):
    """PR₃ and f_top3 per site from repr_geometry summary dict.

    Uses pca_var_pc1/pc2/pc3 fields, which are already fractional eigenvalues,
    so PR₃ is computed directly without additional SVD.
    """
    result = {}
    for site in sites:
        f1 = summary[f"{site}_pca_var_pc1"]
        f2 = summary[f"{site}_pca_var_pc2"]
        f3 = summary[f"{site}_pca_var_pc3"]
        result[site] = {
            "pr3": _compute_pr3(f1, f2, f3),
            "f_top3": f1 + f2 + f3,
        }
    return result


# --- Figure helpers ---


def _add_pr3_f_top3_traces(fig, row, epochs, pr3, ft, color, name, group, show_legend):
    """Solid PR₃ + dashed f_top3 trace pair for one series."""
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=pr3,
            mode="lines",
            name=name,
            legendgroup=group,
            showlegend=show_legend,
            line=dict(color=color, width=2),
            hovertemplate=f"{name}<br>Ep %{{x}}<br>PR₃ %{{y:.3f}}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=ft,
            mode="lines",
            name=name,
            legendgroup=group,
            showlegend=False,
            line=dict(color=color, width=1.5, dash="dash"),
            hovertemplate=f"{name}<br>Ep %{{x}}<br>f_top3 %{{y:.3f}}<extra></extra>",
        ),
        row=row,
        col=1,
    )


def _add_timing_markers(fig, n_rows, onset, fd_end, eff_xover):
    """Vertical timing markers across all subplot rows."""
    for row in range(1, n_rows + 1):
        if fd_end and fd_end > 0:
            fig.add_vline(
                x=fd_end,
                row=row,
                col=1,
                line=dict(color="rgba(210,140,0,0.80)", dash="dot", width=1.5),
            )
        if onset and onset > 0:
            fig.add_vline(
                x=onset,
                row=row,
                col=1,
                line=dict(color="black", dash="dash", width=1.5),
            )
        if eff_xover and eff_xover > 0:
            fig.add_vline(
                x=eff_xover,
                row=row,
                col=1,
                line=dict(color="rgba(60,60,210,0.6)", dash="dot", width=1.5),
            )


def _add_ref_lines(fig, n_rows):
    """Faint horizontal reference lines at PR₃ = 1, 2, 3."""
    for y_val in _REF_LINES:
        for row in range(1, n_rows + 1):
            fig.add_hline(
                y=y_val,
                row=row,
                col=1,
                line=dict(color="rgba(0,0,0,0.10)", dash="dot", width=1),
            )


# --- Panel builders ---


def _add_trajectory_panel(fig, row, pt_data):
    """Panel 1: rolling PR₃ and f_top3 per parameter trajectory site."""
    epochs = pt_data["epochs"].tolist()
    for site in _TRAJ_SITES:
        pr3, ft = _compute_rolling_trajectory_metrics(pt_data[f"{site}__projections"])
        _add_pr3_f_top3_traces(
            fig,
            row,
            epochs,
            pr3.tolist(),
            ft.tolist(),
            _TRAJ_SITE_COLORS[site],
            f"traj {site}",
            f"traj_{site}",
            show_legend=True,
        )


def _add_centroid_panel(fig, row, rg_summary):
    """Panel 2: PR₃ and f_top3 per activation site from repr_geometry."""
    epochs = rg_summary["epochs"].tolist()
    metrics = _extract_repr_dim_metrics(rg_summary, _REPR_SITES)
    for site in _REPR_SITES:
        _add_pr3_f_top3_traces(
            fig,
            row,
            epochs,
            metrics[site]["pr3"].tolist(),
            metrics[site]["f_top3"].tolist(),
            _REPR_SITE_COLORS[site],
            _REPR_SITE_LABELS[site],
            f"repr_{site}",
            show_legend=True,
        )


def _add_weight_group_panel(fig, row, wg_data):
    """Panel 3: PR₃ and f_top3 per frequency group from weight geometry."""
    epochs = wg_data["epochs"].tolist()
    group_freqs = wg_data["group_freqs"]
    group_sizes = wg_data["group_sizes"]
    pr3_all = wg_data["Win_pr3"]  # (n_epochs, n_groups)
    ft_all = wg_data["Win_f_top3"]  # (n_epochs, n_groups)
    for g_idx in range(len(group_freqs)):
        color = _GROUP_PALETTE[g_idx % len(_GROUP_PALETTE)]
        label = f"freq {int(group_freqs[g_idx])} (n={int(group_sizes[g_idx])})"
        _add_pr3_f_top3_traces(
            fig,
            row,
            epochs,
            pr3_all[:, g_idx].tolist(),
            ft_all[:, g_idx].tolist(),
            color,
            label,
            f"wg_{g_idx}",
            show_legend=True,
        )


# --- Public renderers ---


def build_dimensionality_timeseries(
    data: dict,
    epoch: int | None = None,
    height: int = 900,
    width: int = 1000,
) -> go.Figure:
    """Three-panel dimensionality timeseries.

    Panel 1 — Trajectory: rolling PR₃ / f_top3 per parameter site
    Panel 2 — Class Centroids: PR₃ / f_top3 per activation site
    Panel 3 — Within-Group W_in: PR₃ / f_top3 per frequency group

    Solid lines = PR₃.  Dashed lines = f_top3.  Same color per series.
    Y-axis [0, 3.2] on all panels — PR₃ in [1, 3], f_top3 in [0, 1].

    data keys:
        parameter_trajectory  — cross_epoch dict
        repr_geometry_summary — summary dict
        weight_geometry       — cross_epoch dict (must have Win_pr3, Win_f_top3)
        markers               — dict with onset, fd_end, eff_xover (ints or None)
    """
    markers = data.get("markers", {})

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            "Trajectory rolling PR₃ / f_top3  (shape · concentration of learning path)",
            "Class centroid PR₃ / f_top3  (activation-space geometry per site)",
            "Within-group W_in PR₃ / f_top3  (weight-space point cloud per freq group)",
        ],
    )

    _add_trajectory_panel(fig, 1, data["parameter_trajectory"])
    _add_centroid_panel(fig, 2, data["repr_geometry_summary"])
    _add_weight_group_panel(fig, 3, data["weight_geometry"])

    _add_ref_lines(fig, 3)
    _add_timing_markers(
        fig,
        3,
        onset=markers.get("onset"),
        fd_end=markers.get("fd_end"),
        eff_xover=markers.get("eff_xover"),
    )

    for row in range(1, 4):
        fig.update_yaxes(range=[0, 3.2], row=row, col=1)
    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.update_layout(
        title=(
            "Dimensionality Dynamics<br>"
            "<sup>Solid=PR₃ · Dashed=f_top3 · "
            "Orange dotted=first descent end · Black dashed=onset · Blue dotted=eff_xover</sup>"
        ),
        template="plotly_white",
        height=height,
        width=width,
        legend=dict(orientation="v", y=1.04, font=dict(size=10)),
        margin=dict(l=60, r=160, t=80, b=50),
    )
    return fig


def build_dimensionality_state_space(
    data: dict,
    epoch: int | None = None,
    height: int = 520,
    width: int = 1150,
) -> go.Figure:
    """Geometry state space: f_top3 (x) vs PR₃ (y) per activation site.

    Time is encoded as point color (early=blue → late=red) rather than the
    x-axis, making sequencing between sites directly visible: a site that
    reaches (high f_top3, PR₃ ≈ 2) before others is visibly ahead in state
    space.

    Landmarks:
      (high f_top3, PR₃ ≈ 2) — clean concentrated ring: healthy final state
      (high f_top3, PR₃ → 3) — expanding into 3D: working space
      (low f_top3, PR₃ ≈ 2)  — ring-shaped but diffuse
      (low f_top3, PR₃ ≈ 1)  — collapsed and diffuse: pathological

    data keys:
        repr_geometry_summary — summary dict
        markers               — dict with onset, eff_xover (ints or None)
    """
    rg = data["repr_geometry_summary"]
    markers = data.get("markers", {})
    onset = markers.get("onset")
    eff_xover = markers.get("eff_xover")

    epochs_arr = rg["epochs"]
    col_scale = (epochs_arr - epochs_arr.min()) / (epochs_arr.max() - epochs_arr.min() + 1e-9)
    metrics = _extract_repr_dim_metrics(rg, _REPR_SITES)

    fig = make_subplots(
        rows=1,
        cols=len(_REPR_SITES),
        subplot_titles=[_REPR_SITE_LABELS[s] for s in _REPR_SITES],
        horizontal_spacing=0.10,
    )

    for col_idx, site in enumerate(_REPR_SITES, 1):
        _add_state_space_site(
            fig,
            col_idx,
            site,
            epochs_arr,
            col_scale,
            metrics[site],
            onset,
            eff_xover,
        )
        fig.add_hline(
            y=2.0,
            row=1,
            col=col_idx,
            line=dict(color="rgba(0,0,0,0.15)", dash="dot", width=1),
            annotation_text="ring",
            annotation_position="right",
        )
        fig.update_xaxes(title_text="f_top3", range=[0, 1.05], row=1, col=col_idx)
        fig.update_yaxes(
            title_text="PR₃" if col_idx == 1 else "",
            range=[0.9, 3.1],
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=(
            "Geometry State Space  "
            "(open circle=epoch 0 · filled=final · ◆=onset · ▲=eff_xover)<br>"
            "<sup>Color: early=blue → late=red  ·  Target: high f_top3, PR₃ ≈ 2</sup>"
        ),
        template="plotly_white",
        height=height,
        width=width,
        legend=dict(orientation="v", y=1.04, font=dict(size=10)),
        margin=dict(l=60, r=160, t=80, b=50),
    )
    return fig


def _add_state_space_site(fig, col_idx, site, epochs_arr, col_scale, metrics, onset, eff_xover):
    """Render one site's trajectory and event markers in state space."""
    ft = metrics["f_top3"]
    pr3 = metrics["pr3"]
    epochs = epochs_arr.tolist()
    color = _REPR_SITE_COLORS[site]

    # Trajectory line+markers, colored by epoch
    fig.add_trace(
        go.Scatter(
            x=ft.tolist(),
            y=pr3.tolist(),
            mode="lines+markers",
            name=_REPR_SITE_LABELS[site],
            marker=dict(
                size=6,
                color=col_scale.tolist(),
                colorscale="RdYlBu_r",
                cmin=0,
                cmax=1,
                showscale=(col_idx == len(_REPR_SITES)),
                colorbar=dict(title="epoch", len=0.7, x=1.02)
                if col_idx == len(_REPR_SITES)
                else None,
            ),
            line=dict(color="rgba(100,100,100,0.25)", width=1),
            customdata=epochs,
            hovertemplate=(
                f"{_REPR_SITE_LABELS[site]}<br>"
                "epoch %{customdata}<br>"
                "f_top3=%{x:.3f}  PR₃=%{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=col_idx,
    )

    # Epoch 0: open circle
    fig.add_trace(
        go.Scatter(
            x=[ft[0]],
            y=[pr3[0]],
            mode="markers",
            marker=dict(color=color, size=10, symbol="circle-open", line=dict(width=2)),
            showlegend=False,
            hovertemplate=f"{_REPR_SITE_LABELS[site]} epoch 0<br>f_top3=%{{x:.3f}}  PR₃=%{{y:.3f}}<extra></extra>",
        ),
        row=1,
        col=col_idx,
    )

    # Final epoch: filled circle
    fig.add_trace(
        go.Scatter(
            x=[ft[-1]],
            y=[pr3[-1]],
            mode="markers",
            marker=dict(color=color, size=10, symbol="circle"),
            showlegend=False,
            hovertemplate=f"{_REPR_SITE_LABELS[site]} final<br>f_top3=%{{x:.3f}}  PR₃=%{{y:.3f}}<extra></extra>",
        ),
        row=1,
        col=col_idx,
    )

    _add_state_space_event_markers(fig, col_idx, ft, pr3, epochs_arr, onset, eff_xover)


def _add_state_space_event_markers(fig, col_idx, ft, pr3, epochs_arr, onset, eff_xover):
    """Onset diamond and eff_xover triangle markers for one state space panel."""
    if onset and onset > 0:
        idx = int(np.argmin(np.abs(epochs_arr - onset)))
        fig.add_trace(
            go.Scatter(
                x=[ft[idx]],
                y=[pr3[idx]],
                mode="markers+text",
                marker=dict(
                    color="black", size=12, symbol="diamond", line=dict(color="white", width=1)
                ),
                text=["onset"],
                textposition="top center",
                textfont=dict(size=9),
                showlegend=(col_idx == 1),
                name="onset",
                hovertemplate=f"onset (ep {onset})<br>f_top3=%{{x:.3f}}  PR₃=%{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

    if eff_xover and eff_xover > 0:
        idx = int(np.argmin(np.abs(epochs_arr - eff_xover)))
        fig.add_trace(
            go.Scatter(
                x=[ft[idx]],
                y=[pr3[idx]],
                mode="markers+text",
                marker=dict(
                    color="rgba(60,60,210,0.9)",
                    size=12,
                    symbol="triangle-up",
                    line=dict(color="white", width=1),
                ),
                text=["eff_xover"],
                textposition="bottom center",
                textfont=dict(size=9),
                showlegend=(col_idx == 1),
                name="eff_xover",
                hovertemplate=f"eff_xover (ep {eff_xover})<br>f_top3=%{{x:.3f}}  PR₃=%{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
