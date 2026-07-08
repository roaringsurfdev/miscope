"""REQ_025: Attention Head Pattern Visualization.

Renders attention head patterns as heatmaps with two viewing modes:
1. All heads view - faceted heatmap showing all heads for a position pair
2. Single head view - detailed heatmap for one head
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_POSITION_LABELS = ["a", "b", "="]


def render_attention_heads(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    to_position: int = 2,
    from_position: int = 0,
    position_labels: list[str] | None = None,
    title: str | None = None,
    colorscale: str = "Blues",
    subplot_size: int = 250,
) -> go.Figure:
    """Render attention patterns for all heads as a faceted heatmap.

    Args:
        epoch_data: Dict containing 'patterns' array of shape
            (n_heads, n_positions, n_positions, p, p).
        epoch: Epoch number (used for title).
        to_position: Destination token position index (default: 2, the = token).
        from_position: Source token position index (default: 0, the a token).
        position_labels: Human-readable labels for positions
            (default: ["a", "b", "="]).
        title: Custom title (default: auto-generated).
        colorscale: Plotly colorscale name.
        subplot_size: Size of each subplot in pixels.

    Returns:
        Plotly Figure with faceted heatmaps.
    """
    patterns = epoch_data["patterns"]
    n_heads = patterns.shape[0]
    labels = position_labels or DEFAULT_POSITION_LABELS

    to_label = labels[to_position]
    from_label = labels[from_position]

    cols = min(n_heads, 4)
    rows = (n_heads + cols - 1) // cols

    subplot_titles = [f"Head {i}" for i in range(n_heads)]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Find global color range for consistent scaling
    all_data = patterns[:, to_position, from_position]  # (n_heads, p, p)
    zmin = float(all_data.min())
    zmax = float(all_data.max())

    for i in range(n_heads):
        row = i // cols + 1
        col = i % cols + 1

        data = patterns[i, to_position, from_position]  # (p, p)
        p = data.shape[0]

        fig.add_trace(
            go.Heatmap(
                z=data,
                x=list(range(p)),
                y=list(range(p)),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=(i == 0),
                hovertemplate=(
                    f"Head {i}<br>a=%{{y}}, b=%{{x}}<br>Attention: %{{z:.4f}}<extra></extra>"
                ),
                colorbar=dict(title="Attention") if i == 0 else None,
            ),
            row=row,
            col=col,
        )

    if title is None:
        title = f"Attention Heads ({to_label} \u2190 {from_label}) \u2014 Epoch {epoch}"

    fig.update_layout(
        title=title,
        height=rows * subplot_size + 100,
        width=cols * subplot_size + 100,
        template="plotly_white",
    )

    # Label axes on edge subplots
    for i in range(n_heads):
        row = i // cols + 1
        col = i % cols + 1
        fig.update_xaxes(title_text="b" if row == rows else None, row=row, col=col)
        fig.update_yaxes(title_text="a" if col == 1 else None, row=row, col=col)

    return fig


def render_attention_single_head(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    head_idx: int = 0,
    to_position: int = 2,
    from_position: int = 0,
    position_labels: list[str] | None = None,
    title: str | None = None,
    colorscale: str = "Blues",
) -> go.Figure:
    """Render attention pattern for a single head.

    Args:
        epoch_data: Dict containing 'patterns' array of shape
            (n_heads, n_positions, n_positions, p, p).
        epoch: Epoch number (used for title).
        head_idx: Which head to display (0 to n_heads-1).
        to_position: Destination token position index (default: 2, the = token).
        from_position: Source token position index (default: 0, the a token).
        position_labels: Human-readable labels for positions
            (default: ["a", "b", "="]).
        title: Custom title (default: auto-generated).
        colorscale: Plotly colorscale name.

    Returns:
        Plotly Figure with single heatmap.
    """
    patterns = epoch_data["patterns"]
    n_heads = patterns.shape[0]
    labels = position_labels or DEFAULT_POSITION_LABELS

    if head_idx < 0 or head_idx >= n_heads:
        raise IndexError(f"head_idx {head_idx} out of range [0, {n_heads})")

    to_label = labels[to_position]
    from_label = labels[from_position]

    data = patterns[head_idx, to_position, from_position]  # (p, p)
    p = data.shape[0]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=data,
            x=list(range(p)),
            y=list(range(p)),
            colorscale=colorscale,
            hovertemplate=(
                f"Head {head_idx}<br>a=%{{y}}, b=%{{x}}<br>Attention: %{{z:.4f}}<extra></extra>"
            ),
            colorbar=dict(title="Attention"),
        )
    )

    if title is None:
        title = f"Head {head_idx} Attention ({to_label} \u2190 {from_label}) \u2014 Epoch {epoch}"

    fig.update_layout(
        title=title,
        xaxis_title="b",
        yaxis_title="a",
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x", constrain="domain"),
        template="plotly_white",
    )

    return fig
