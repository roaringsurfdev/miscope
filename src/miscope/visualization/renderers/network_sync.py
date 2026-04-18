"""Cross-site synchronization view.

Renders how PC3 budget, circularity, activation scale, and weight-space
compression co-evolve across functional areas of the network, annotated
with standardized grokking epoch markers from variant_summary.json.

The view is designed to reveal *staging* — whether expansion/collapse
events in different functional areas (attention, MLP, residual stream)
happen sequentially or simultaneously.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_SITE_COLORS = {
    "attn_out": "rgba(44, 160, 44, 1.0)",
    "mlp_out": "rgba(214, 39, 40, 1.0)",
    "resid_post": "rgba(148, 103, 189, 1.0)",
}
_SITE_LABELS = {
    "attn_out": "Attn Out",
    "mlp_out": "MLP Out",
    "resid_post": "Resid Post",
}
_ACTIVE_SITES = ["attn_out", "mlp_out", "resid_post"]

# Grokking marker style constants
_ONSET_STYLE = dict(color="rgba(0,0,0,0.55)", width=1.5, dash="dash")
_CROSSOVER_STYLE = dict(color="rgba(60,60,210,0.55)", width=1.5, dash="dot")


def render_network_sync(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Four-row cross-site synchronization view.

    Row 1 — PC3 variance fraction by site (3D budget):
        How much of the class centroid variance lies in the third principal
        direction.  Expansion = model is using the 3rd dimension for
        reorganization.  Collapse = ring has settled back to 2D.

    Row 2 — Circularity by site:
        How close the centroid distribution is to a perfect circle.
        A dip during the second descent is expected as PC3 expands and
        the ring reorganizes.

    Row 3 — Mean ring radius by site (activation-space L2 scale):
        The mean distance of class centroids from their centroid.
        Compression here reflects weight-decay-driven collapse of the
        representational scale in each activation site.

    Row 4 — W_in within-group L2 spread, mean across groups:
        Mean L2 dispersion of neuron weight vectors within frequency groups.
        Tracks weight-space compression independent of activation scale.

    Vertical markers:
        Dashed black  — second_descent_onset_epoch (grokking onset)
        Dotted blue   — effective_dimensionality_cross_over_epoch
                        (W_out PR crosses below W_in PR; midpoint of descent)

    Args:
        data: dict containing:
            repr_summary  — repr_geometry summary dict
            group_spread  — (n_epochs, n_groups) float32 (optional)
            spread_epochs — (n_epochs,) int32 matching group_spread
            markers       — dict with grokking epoch keys (optional)
        epoch: optional epoch cursor (vertical line)
    """
    summary = data["repr_summary"]
    group_spread = data.get("group_spread")  # (n_epochs, n_groups) or None
    spread_epochs = data.get("spread_epochs")
    markers = data.get("markers") or {}

    epochs = summary["epochs"]
    n_rows = 4 if (group_spread is not None and len(group_spread) > 0) else 3

    row_titles = [
        "PC3 variance fraction  (3D budget)",
        "Circularity",
        "Mean ring radius  (activation-space scale)",
        "W_in group spread  (weight-space compression)",
    ][:n_rows]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=row_titles,
    )

    _add_site_traces(fig, summary, epochs)
    if n_rows == 4:
        _add_spread_trace(fig, group_spread, spread_epochs, n_rows) # type: ignore

    _add_grokking_markers(fig, markers, n_rows)

    if epoch is not None:
        for row in range(1, n_rows + 1):
            fig.add_vline(
                x=epoch,
                line=dict(color="rgba(0,0,0,0.2)", width=1, dash="dash"),
                row=row, # type: ignore
                col=1, # type: ignore
            )

    fig.update_xaxes(title_text="Epoch", row=n_rows, col=1)
    y_labels = ["PC3 frac", "Circularity", "Radius", "L2 spread"]
    for i, y_label in enumerate(y_labels[:n_rows]):
        fig.update_yaxes(title_text=y_label, row=i + 1, col=1)

    fig.update_layout(
        title=(
            "Cross-site synchronization: PC3 · circularity · scale · weight compression<br>"
            "<sup>Dashed: grokking onset  ·  Dotted: W_out PR crosses W_in PR</sup>"
        ),
        template="plotly_white",
        height=200 * n_rows + 100,
        margin=dict(l=70, r=20, t=90, b=60),
        legend=dict(orientation="v", x=1.02, y=1),
    )
    return fig


def _add_site_traces(
    fig: go.Figure,
    summary: dict,
    epochs: np.ndarray,
) -> None:
    """Add rows 1-3: PC3, circularity, mean_radius per active site."""
    metric_keys = [
        ("{site}_pca_var_pc3", 1),
        ("{site}_circularity", 2),
        ("{site}_mean_radius", 3),
    ]
    for site in _ACTIVE_SITES:
        first_key = metric_keys[0][0].format(site=site)
        if first_key not in summary:
            continue
        color = _SITE_COLORS[site]
        label = _SITE_LABELS[site]
        for key_template, row in metric_keys:
            key = key_template.format(site=site)
            if key not in summary:
                continue
            fig.add_trace(
                go.Scatter(
                    x=epochs.tolist(),
                    y=summary[key].tolist(),
                    mode="lines",
                    name=label,
                    legendgroup=site,
                    showlegend=(row == 1),
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label}<br>epoch=%{{x}}<br>%{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )


def _add_spread_trace(
    fig: go.Figure,
    group_spread: np.ndarray,
    spread_epochs: np.ndarray,
    row: int,
) -> None:
    """Add row 4: mean W_in within-group L2 spread across all groups."""
    mean_spread = group_spread.mean(axis=1)  # (n_epochs,)
    fig.add_trace(
        go.Scatter(
            x=spread_epochs.tolist(),
            y=mean_spread.tolist(),
            mode="lines",
            name="W_in spread",
            legendgroup="w_in",
            showlegend=True,
            line=dict(color="rgba(31, 119, 180, 1.0)", width=2),
            hovertemplate="W_in spread<br>epoch=%{x}<br>%{y:.4f}<extra></extra>",
        ),
        row=row,
        col=1,
    )


def _add_grokking_markers(
    fig: go.Figure,
    markers: dict,
    n_rows: int,
) -> None:
    """Add standardized grokking epoch annotations to all rows."""
    onset = markers.get("second_descent_onset_epoch")
    crossover = markers.get("effective_dimensionality_cross_over_epoch")

    for row in range(1, n_rows + 1):
        if onset is not None:
            fig.add_vline(
                x=onset,
                line=_ONSET_STYLE,
                row=row, # type: ignore
                col=1, # type: ignore
            )
        if crossover is not None:
            fig.add_vline(
                x=crossover,
                line=_CROSSOVER_STYLE,
                row=row, # type: ignore
                col=1, # type: ignore
            )
