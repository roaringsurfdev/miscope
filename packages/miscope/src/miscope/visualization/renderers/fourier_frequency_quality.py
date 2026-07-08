"""REQ_052: Fourier Frequency Quality Trajectory Renderer.

Renders the per-epoch R² quality score and dominant frequency count
produced by the FourierFrequencyQualityAnalyzer.
"""

import numpy as np
import plotly.graph_objects as go


def render_fourier_quality_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int,
    title: str | None = None,
    height: int = 400,
) -> go.Figure:
    """Quality score and dominant frequency count over training.

    Primary y-axis: R² quality score (0–1).
    Secondary y-axis: dominant basis vector count k (dashed).
    Vertical indicator at current epoch.

    Args:
        summary_data: From ArtifactLoader.load_summary('fourier_frequency_quality').
            Contains 'epochs', 'quality_score', 'reconstruction_error', 'k'.
        current_epoch: Current epoch for vertical indicator.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with dual-axis trajectory plot.
    """
    epochs = summary_data["epochs"]
    quality = summary_data["quality_score"]
    k_vals = summary_data["k"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=quality,
            mode="lines",
            name="Frequency Quality (R²)",
            line=dict(color="rgba(31, 119, 180, 1.0)", width=2),
            hovertemplate="Epoch %{x}<br>Quality: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=k_vals,
            mode="lines",
            name="Dominant Freq. Count (k)",
            line=dict(color="rgba(255, 127, 14, 0.6)", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="Epoch %{x}<br>k: %{y}<extra></extra>",
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
        title = "Fourier Frequency Quality"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis=dict(title="Quality (R²)", range=[0, 1]),
        yaxis2=dict(
            title="Dominant Freq. Count (k)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=height,
        margin=dict(l=60, r=60, t=50, b=50),
    )

    return fig
