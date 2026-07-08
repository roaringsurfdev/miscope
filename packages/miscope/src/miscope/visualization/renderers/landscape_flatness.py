"""REQ_031: Loss Landscape Flatness Visualizations.

Renders flatness metric trajectories over training and per-epoch
perturbation distributions from landscape flatness analysis.
"""

import numpy as np
import plotly.graph_objects as go

FLATNESS_METRICS = {
    "mean_delta_loss": "Mean Delta Loss",
    "median_delta_loss": "Median Delta Loss",
    "max_delta_loss": "Max Delta Loss",
    "flatness_ratio": "Flatness Ratio",
    "p90_delta_loss": "P90 Delta Loss",
    "std_delta_loss": "Std Delta Loss",
}


def render_flatness_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int,
    metric: str = "mean_delta_loss",
    title: str | None = None,
    height: int = 400,
) -> go.Figure:
    """Flatness metric over epochs with baseline loss on secondary axis.

    Primary y-axis: selected flatness metric line.
    Secondary y-axis: baseline_loss line (dashed gray).
    Vertical indicator at current epoch.

    Args:
        summary_data: From ArtifactLoader.load_summary(). Contains
            "epochs" array and metric arrays.
        current_epoch: Current epoch for vertical indicator.
        metric: Summary key to plot (default "mean_delta_loss").
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with dual-axis trajectory plot.
    """
    epochs = summary_data["epochs"]
    fig = go.Figure()

    metric_label = FLATNESS_METRICS.get(metric, metric)
    if metric in summary_data:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=summary_data[metric],
                mode="lines",
                name=metric_label,
                line=dict(color="rgba(31, 119, 180, 1.0)", width=2),
                hovertemplate=(
                    f"{metric_label}<br>Epoch %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    if "baseline_loss" in summary_data:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=summary_data["baseline_loss"],
                mode="lines",
                name="Baseline Loss",
                line=dict(
                    color="rgba(150, 150, 150, 0.6)",
                    width=1.5,
                    dash="dash",
                ),
                yaxis="y2",
                hovertemplate=("Baseline Loss<br>Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>"),
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
        title = f"Landscape Flatness — {metric_label}"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=metric_label,
        yaxis2=dict(
            title="Baseline Loss",
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


def render_perturbation_distribution(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    title: str | None = None,
    height: int = 350,
) -> go.Figure:
    """Histogram of loss changes from random perturbations at one epoch.

    Shows the distribution of delta_loss values, annotated with
    mean, median, and flatness_ratio.

    Args:
        epoch_data: From ArtifactLoader.load_epoch(). Contains
            "delta_losses", "baseline_loss", "epsilon".
        epoch: Epoch number for title display.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with histogram and annotation lines.
    """
    delta = epoch_data["delta_losses"]
    baseline = float(epoch_data["baseline_loss"])

    mean_val = float(np.mean(delta))
    median_val = float(np.median(delta))

    threshold = 0.1 * baseline if baseline > 0 else 0.1
    flat_count = np.sum(delta < threshold)
    flatness_ratio = float(flat_count / len(delta))

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=delta,
            nbinsx=min(30, len(delta)),
            marker_color="rgba(31, 119, 180, 0.7)",
            name="Delta Loss",
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.add_vline(
        x=mean_val,
        line_dash="solid",
        line_color="red",
        line_width=2,
        annotation_text=f"Mean: {mean_val:.4f}",
        annotation_position="top right",
        annotation_font_color="red",
    )

    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text=f"Median: {median_val:.4f}",
        annotation_position="top left",
        annotation_font_color="green",
    )

    if title is None:
        title = f"Perturbation Distribution — Epoch {epoch} (Flatness Ratio: {flatness_ratio:.2f})"

    fig.update_layout(
        title=title,
        xaxis_title="Delta Loss",
        yaxis_title="Count",
        template="plotly_white",
        showlegend=False,
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig
