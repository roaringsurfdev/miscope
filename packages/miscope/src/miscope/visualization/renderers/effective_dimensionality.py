"""REQ_030: Effective Dimensionality Visualizations.

Renders participation ratio trajectories and singular value spectra
from weight matrix SVD data across training checkpoints.
"""

import numpy as np
import plotly.graph_objects as go

from miscope.analysis.library.weights import (
    ATTENTION_MATRICES,
    WEIGHT_MATRIX_NAMES,
)

# Colors for consistent matrix identification
_MATRIX_COLORS = {
    "W_E": "rgba(31, 119, 180, 1.0)",
    "W_pos": "rgba(174, 199, 232, 1.0)",
    "W_Q": "rgba(44, 160, 44, 1.0)",
    "W_K": "rgba(152, 223, 138, 1.0)",
    "W_V": "rgba(23, 190, 207, 1.0)",
    "W_O": "rgba(158, 218, 229, 1.0)",
    "W_in": "rgba(214, 39, 40, 1.0)",
    "W_out": "rgba(255, 127, 14, 1.0)",
    "W_U": "rgba(148, 103, 189, 1.0)",
}


def render_dimensionality_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int,
    matrices: list[str] | None = None,
    title: str | None = None,
    height: int = 400,
) -> go.Figure:
    """Participation ratio over epochs for selected weight matrices.

    One line per matrix. For attention matrices with per-head PRs,
    plots the mean across heads.

    Args:
        summary_data: From ArtifactLoader.load_summary(). Contains
            "epochs" array and "pr_{name}" arrays.
        current_epoch: Current epoch for vertical indicator.
        matrices: Weight matrix names to display. None = all non-attention.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with multi-line trajectory plot.
    """
    epochs = summary_data["epochs"]

    if matrices is None:
        matrices = [n for n in WEIGHT_MATRIX_NAMES if n not in ATTENTION_MATRICES]

    fig = go.Figure()

    for name in matrices:
        pr_key = f"pr_{name}"
        if pr_key not in summary_data:
            continue

        pr_data = summary_data[pr_key]

        if pr_data.ndim == 2:
            # Per-head attention: plot mean across heads
            pr_values = pr_data.mean(axis=1)
            display_name = f"{name} (mean)"
        else:
            pr_values = pr_data
            display_name = name

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=pr_values,
                mode="lines",
                name=display_name,
                line=dict(
                    color=_MATRIX_COLORS.get(name, "gray"),
                    width=2,
                ),
                hovertemplate=(f"{display_name}<br>Epoch %{{x}}<br>PR: %{{y:.1f}}<extra></extra>"),
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
        title = "Effective Dimensionality (Participation Ratio)"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Participation Ratio",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_singular_value_spectrum(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    matrix_name: str = "W_in",
    head_idx: int | None = None,
    title: str | None = None,
    height: int = 350,
) -> go.Figure:
    """Bar chart of singular values for a weight matrix at one epoch.

    Shows the distribution of singular values sorted descending,
    annotated with the participation ratio.

    Args:
        epoch_data: From the singular-value-spectrum loader. Contains
            "sv_{name}" singular-value arrays and matching "pr_{name}"
            participation ratios (pre-computed by the loader; scalar for
            non-attention matrices, per-head 1D array for attention).
        epoch: Epoch number (for title display).
        matrix_name: Weight matrix to display (e.g., "W_in").
        head_idx: For attention matrices, which head to show.
            None defaults to head 0.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with singular value bar chart.
    """
    sv_key = f"sv_{matrix_name}"
    sv = epoch_data[sv_key]
    pr_all = epoch_data[f"pr_{matrix_name}"]

    # Handle per-head attention matrices
    head_label = ""
    if sv.ndim == 2:
        idx = head_idx if head_idx is not None else 0
        sv = sv[idx]
        pr = float(pr_all[idx])
        head_label = f" Head {idx}"
    else:
        pr = float(pr_all)
    indices = list(range(len(sv)))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=indices,
            y=sv,
            marker_color=_MATRIX_COLORS.get(matrix_name, "rgba(31, 119, 180, 0.7)"),
            hovertemplate="SV %{x}: %{y:.4f}<extra></extra>",
        )
    )

    if title is None:
        title = f"{matrix_name}{head_label} Singular Values — Epoch {epoch} (PR = {pr:.1f})"

    fig.update_layout(
        title=title,
        xaxis_title="Singular Value Index",
        yaxis_title="Singular Value",
        template="plotly_white",
        showlegend=False,
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig
