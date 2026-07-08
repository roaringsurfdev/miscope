"""REQ_026: Attention Head Frequency Specialization Renderers.

Renders frequency decomposition of attention patterns per head:
- Per-epoch heatmap (frequencies × heads)
- Cross-epoch specialization trajectory (one line per head)
- Cross-epoch dominant frequency per head
"""

import numpy as np
import plotly.graph_objects as go

# Plotly default color cycle for consistent head colors
HEAD_COLORS = [
    "rgba(31, 119, 180, 1.0)",
    "rgba(255, 127, 14, 1.0)",
    "rgba(44, 160, 44, 1.0)",
    "rgba(214, 39, 40, 1.0)",
    "rgba(148, 103, 189, 1.0)",
    "rgba(140, 86, 75, 1.0)",
    "rgba(227, 119, 194, 1.0)",
    "rgba(127, 127, 127, 1.0)",
]


def render_attention_freq_heatmap(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    title: str | None = None,
    colorscale: str = "Blues",
    height: int = 400,
    width: int = 400,
) -> go.Figure:
    """Render frequency decomposition heatmap for attention heads.

    Shows what fraction of each head's attention pattern variance is
    explained by each frequency component.

    Args:
        epoch_data: Dict with 'freq_matrix' of shape (n_freq, n_heads).
        epoch: Epoch number (used for title).
        title: Custom title.
        colorscale: Plotly colorscale name.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure with heatmap.
    """
    data = epoch_data["freq_matrix"]  # (n_freq, n_heads)
    n_freq, n_heads = data.shape

    freq_labels = [str(i + 1) for i in range(n_freq)]
    head_labels = [f"Head {i}" for i in range(n_heads)]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=data,
            x=list(range(n_heads)),
            y=list(range(n_freq)),
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            hovertemplate=(
                "%{customdata[0]}<br>"
                "Freq %{customdata[1]}<br>"
                "Frac Explained: %{z:.4f}<extra></extra>"
            ),
            customdata=[
                [(head_labels[h], freq_labels[f]) for h in range(n_heads)] for f in range(n_freq)
            ],
            colorbar=dict(
                title=dict(text="Frac<br>Explained", side="right"),
                thickness=15,
                len=0.9,
            ),
        )
    )

    if title is None:
        title = f"Attention Head Frequency Specialization \u2014 Epoch {epoch}"

    fig.update_layout(
        title=title,
        xaxis_title="Head",
        yaxis_title="Frequency",
        xaxis=dict(
            tickvals=list(range(n_heads)),
            ticktext=head_labels,
        ),
        yaxis=dict(
            tickvals=list(range(n_freq)),
            ticktext=freq_labels,
        ),
        height=height,
        width=width,
        template="plotly_white",
        margin=dict(l=60, r=80, t=50, b=50),
    )

    return fig


def render_attention_specialization_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Render per-head specialization strength over training.

    Shows how strongly each attention head specializes in a single
    frequency over the course of training (max variance fraction).

    Args:
        summary_data: Dict from ArtifactLoader.load_summary("attention_freq"),
            containing 'epochs' and 'max_frac_per_head' arrays.
        current_epoch: Current epoch for vertical indicator line.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with one line per head and epoch indicator.
    """
    epochs = summary_data["epochs"]
    max_frac = summary_data["max_frac_per_head"]  # (n_epochs, n_heads)
    n_heads = max_frac.shape[1]

    fig = go.Figure()

    for h in range(n_heads):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=max_frac[:, h],
                mode="lines",
                name=f"Head {h}",
                line=dict(color=HEAD_COLORS[h % len(HEAD_COLORS)], width=2),
                hovertemplate=(f"Head {h}<br>Epoch %{{x}}<br>Max Frac: %{{y:.4f}}<extra></extra>"),
            )
        )

    # Vertical epoch indicator
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
        title = "Attention Head Specialization Over Training"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Max Variance Fraction",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_attention_dominant_frequencies(
    summary_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Render dominant frequency per head over training.

    Shows which frequency each head is tuned to at each checkpoint,
    revealing when heads "lock in" to their frequencies.

    Args:
        summary_data: Dict from ArtifactLoader.load_summary("attention_freq"),
            containing 'epochs' and 'dominant_freq_per_head' arrays.
        current_epoch: Optional current epoch for vertical indicator.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with one trace per head showing dominant frequency index.
    """
    epochs = summary_data["epochs"]
    dominant = summary_data["dominant_freq_per_head"]  # (n_epochs, n_heads)
    n_heads = dominant.shape[1]

    fig = go.Figure()

    for h in range(n_heads):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=dominant[:, h],
                mode="lines+markers",
                name=f"Head {h}",
                line=dict(
                    color=HEAD_COLORS[h % len(HEAD_COLORS)],
                    width=2,
                    shape="hv",
                ),
                marker=dict(size=4),
                hovertemplate=(
                    f"Head {h}<br>Epoch %{{x}}<br>Dominant Freq: %{{y:.0f}}<extra></extra>"
                ),
            )
        )

    if current_epoch is not None:
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
        title = "Dominant Frequency per Attention Head"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Frequency Index",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig
