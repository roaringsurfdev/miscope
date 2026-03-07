"""REQ_058: Band concentration visualization renderers.

Renderers for neuron band concentration health metrics:
    render_concentration_trajectory  — HHI + active band count over epochs
    render_rank_alignment_trajectory — embedding-neuron Spearman correlation over epochs
    render_concentration_scatter     — cross-variant midpoint HHI vs grokking onset
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    import pandas as pd

_DEFAULT_HEIGHT = 400
_DEFAULT_WIDTH = None

_COLOR_HHI = "#4C72B0"
_COLOR_BANDS = "#DD8452"
_COLOR_CORR = "#55A868"
_COLOR_GROKKING = "rgba(200, 50, 50, 0.6)"

_FAILURE_MODE_COLORS = {
    "healthy": "#2ecc71",
    "late_grokker": "#f39c12",
    "degraded": "#e74c3c",
    "no_grokking": "#95a5a6",
}


def render_concentration_trajectory(
    data: dict,
    grokking_onset_epoch: int | None = None,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> go.Figure:
    """Line chart of HHI and active band count over epochs.

    Args:
        data: Output of compute_band_concentration_trajectory, with keys:
              epochs, hhi, active_band_count, max_band_share.
        grokking_onset_epoch: If provided, adds a vertical marker.
        title: Optional title override.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        go.Figure with two subplots: HHI (top), active band count (bottom).
    """
    epochs = data["epochs"]
    hhi = data["hhi"]
    active_band_count = data["active_band_count"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Band Concentration (HHI)", "Active Band Count"),
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=hhi,
            mode="lines",
            name="HHI",
            line=dict(color=_COLOR_HHI, width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=active_band_count,
            mode="lines",
            name="Active bands",
            line=dict(color=_COLOR_BANDS, width=2),
        ),
        row=2,
        col=1,
    )

    if grokking_onset_epoch is not None:
        for row in (1, 2):
            fig.add_vline(
                x=grokking_onset_epoch,
                line=dict(color=_COLOR_GROKKING, width=2, dash="dash"),
                row=row,
                col=1,
            )

    fig.update_yaxes(title_text="HHI", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Bands", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)

    fig.update_layout(
        title=title or "Neuron Band Concentration Trajectory",
        height=height or _DEFAULT_HEIGHT,
        width=width or _DEFAULT_WIDTH,
        showlegend=False,
        margin=dict(t=60, b=40, l=60, r=20),
    )

    return fig


def render_rank_alignment_trajectory(
    data: dict,
    grokking_onset_epoch: int | None = None,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> go.Figure:
    """Line chart of embedding-neuron Spearman rank correlation over epochs.

    Args:
        data: Output of compute_rank_alignment_trajectory, with keys:
              epochs, rank_correlation.
        grokking_onset_epoch: If provided, adds a vertical marker.
        title: Optional title override.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        go.Figure.
    """
    epochs = data["epochs"]
    rank_corr = data["rank_correlation"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=rank_corr,
            mode="lines",
            name="Rank correlation",
            line=dict(color=_COLOR_CORR, width=2),
        )
    )

    # Reference line at 0
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"))

    if grokking_onset_epoch is not None:
        fig.add_vline(
            x=grokking_onset_epoch,
            line=dict(color=_COLOR_GROKKING, width=2, dash="dash"),
        )

    fig.update_layout(
        title=title or "Embedding–Neuron Rank Alignment Trajectory",
        xaxis_title="Epoch",
        yaxis_title="Spearman ρ",
        yaxis=dict(range=[-1.1, 1.1]),
        height=height or _DEFAULT_HEIGHT,
        width=width or _DEFAULT_WIDTH,
        showlegend=False,
        margin=dict(t=60, b=40, l=60, r=20),
    )

    return fig


def render_concentration_scatter(
    df: pd.DataFrame,
    hhi_col: str = "midpoint_hhi",
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> go.Figure:
    """Cross-variant scatter: midpoint HHI vs grokking onset, colored by failure mode.

    Args:
        df: DataFrame as returned by load_family_comparison() (with band concentration
            columns added). Must contain: hhi_col, grokking_onset_epoch,
            failure_mode, variant_name.
        hhi_col: Column name for the HHI metric to plot on x-axis.
        title: Optional title override.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        go.Figure with one scatter point per variant.
    """
    fig = go.Figure()

    failure_modes = df["failure_mode"].unique() if "failure_mode" in df.columns else []

    for mode in failure_modes:
        subset = df[df["failure_mode"] == mode]
        color = _FAILURE_MODE_COLORS.get(str(mode), "#888")

        x_vals = subset[hhi_col].values if hhi_col in subset.columns else np.full(len(subset), np.nan)
        y_vals = subset["grokking_onset_epoch"].values if "grokking_onset_epoch" in subset.columns else np.full(len(subset), np.nan)
        names = subset["variant_name"].values if "variant_name" in subset.columns else [""] * len(subset)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=str(mode),
                text=names,
                hovertemplate="%{text}<br>HHI: %{x:.3f}<br>Grokking onset: %{y}<extra></extra>",
                marker=dict(color=color, size=10, line=dict(color="white", width=1)),
            )
        )

    fig.update_layout(
        title=title or "Cross-Variant: Midpoint HHI vs Grokking Onset",
        xaxis_title="Midpoint HHI (band concentration)",
        yaxis_title="Grokking onset epoch",
        height=height or _DEFAULT_HEIGHT,
        width=width or _DEFAULT_WIDTH,
        margin=dict(t=60, b=40, l=60, r=20),
    )

    return fig
