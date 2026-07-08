"""REQ_055: Attention Head Fourier Decomposition renderers.

Renders per-head QK^T Fourier spectrum heatmaps and temporal head
alignment trajectories. Both views surface which Fourier frequency
each head's similarity computation emphasizes.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_qk_freq_heatmap(
    epoch_data: dict,
    epoch: int,
    title: str | None = None,
    height: int = 350,
    width: int = 700,
) -> go.Figure:
    """Render QK^T Fourier spectrum as a head × frequency heatmap.

    Args:
        epoch_data: Single-epoch dict with 'qk_freq_norms' (n_heads, n_freq).
        epoch: Epoch number for the title.
        title: Custom title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure with a heatmap trace.
    """
    qk_norms = epoch_data["qk_freq_norms"]  # (n_heads, n_freq)
    n_heads, n_freq = qk_norms.shape
    freq_labels = [str(k) for k in range(1, n_freq + 1)]
    head_labels = [f"H{h}" for h in range(n_heads)]

    fig = go.Figure(
        go.Heatmap(
            z=qk_norms,
            x=freq_labels,
            y=head_labels,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Freq fraction", thickness=15),
            hovertemplate="Head %{y} | Freq %{x}<br>Fraction: %{z:.3f}<extra></extra>",
        )
    )

    if title is None:
        title = f"QK\u1d40 Fourier Spectrum — Epoch {epoch}"

    fig.update_layout(
        title=title,
        xaxis_title="Frequency k",
        yaxis_title="Attention Head",
        height=height,
        width=width,
        margin=dict(l=60, r=60, t=50, b=50),
        template="plotly_white",
    )

    return fig


def render_v_freq_heatmap(
    epoch_data: dict,
    epoch: int,
    title: str | None = None,
    height: int = 350,
    width: int = 700,
) -> go.Figure:
    """Render V Fourier spectrum as a head × frequency heatmap.

    Args:
        epoch_data: Single-epoch dict with 'v_freq_norms' (n_heads, n_freq).
        epoch: Epoch number for the title.
        title: Custom title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure with a heatmap trace.
    """
    v_norms = epoch_data["v_freq_norms"]  # (n_heads, n_freq)
    n_heads, n_freq = v_norms.shape
    freq_labels = [str(k) for k in range(1, n_freq + 1)]
    head_labels = [f"H{h}" for h in range(n_heads)]

    fig = go.Figure(
        go.Heatmap(
            z=v_norms,
            x=freq_labels,
            y=head_labels,
            colorscale="Plasma",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Freq fraction", thickness=15),
            hovertemplate="Head %{y} | Freq %{x}<br>Fraction: %{z:.3f}<extra></extra>",
        )
    )

    if title is None:
        title = f"V Fourier Spectrum — Epoch {epoch}"

    fig.update_layout(
        title=title,
        xaxis_title="Frequency k",
        yaxis_title="Attention Head",
        height=height,
        width=width,
        margin=dict(l=60, r=60, t=50, b=50),
        template="plotly_white",
    )

    return fig


def render_head_alignment_trajectory(
    stacked_data: dict,
    title: str | None = None,
    height: int = 400,
    width: int = 900,
) -> go.Figure:
    """Render dominant frequency alignment per head across epochs.

    Shows the dominant QK^T frequency fraction for each head over time,
    revealing when each head locks onto a specific frequency.

    Args:
        stacked_data: Dict with 'epochs' (n_epochs,) and
            'qk_freq_norms' (n_epochs, n_heads, n_freq).
        title: Custom title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure with one line trace per head.
    """
    epochs = stacked_data["epochs"]  # (n_epochs,)
    qk_norms = stacked_data["qk_freq_norms"]  # (n_epochs, n_heads, n_freq)
    n_epochs, n_heads, n_freq = qk_norms.shape

    dominant_freq = np.argmax(qk_norms, axis=2) + 1  # (n_epochs, n_heads), 1-indexed
    max_frac = np.max(qk_norms, axis=2)  # (n_epochs, n_heads)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
    )

    colors = _head_colors(n_heads)

    for h in range(n_heads):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=dominant_freq[:, h],
                mode="lines",
                name=f"H{h} dominant freq",
                line=dict(color=colors[h], width=1.5),
                hovertemplate=f"H{h} | Epoch %{{x}}<br>Dominant freq: %{{y}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=max_frac[:, h],
                mode="lines",
                name=f"H{h} max fraction",
                line=dict(color=colors[h], width=1.5, dash="dot"),
                showlegend=False,
                hovertemplate=f"H{h} | Epoch %{{x}}<br>Max fraction: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    if title is None:
        title = "Attention Head QK\u1d40 Frequency Alignment Trajectory"

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.01, font=dict(size=10)
        ),
        margin=dict(l=60, r=150, t=50, b=50),
    )
    fig.update_yaxes(title_text="Dominant Freq k", row=1, col=1)
    fig.update_yaxes(title_text="Max Fraction", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)

    return fig


def _head_colors(n_heads: int) -> list[str]:
    """Generate distinct colors for each attention head."""
    palette = [
        "rgba(31, 119, 180, 0.9)",
        "rgba(255, 127, 14, 0.9)",
        "rgba(44, 160, 44, 0.9)",
        "rgba(214, 39, 40, 0.9)",
        "rgba(148, 103, 189, 0.9)",
        "rgba(140, 86, 75, 0.9)",
        "rgba(227, 119, 194, 0.9)",
        "rgba(127, 127, 127, 0.9)",
    ]
    return [palette[h % len(palette)] for h in range(n_heads)]
