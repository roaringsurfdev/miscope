"""REQ_057: Cross-variant comparison renderers.

Renders family-level comparison views: metrics table and loss curve overlay.
These renderers operate on data from multiple variants simultaneously, so
they do not fit the per-variant ViewCatalog pattern — call them directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from miscope.families.variant import Variant


_FAILURE_MODE_COLORS = {
    "healthy": "rgba(44, 160, 44, 0.85)",
    "late_grokker": "rgba(255, 127, 14, 0.85)",
    "degraded": "rgba(214, 39, 40, 0.85)",
    "no_grokking": "rgba(148, 103, 189, 0.85)",
    "unknown": "rgba(150, 150, 150, 0.85)",
}


def render_metrics_table(
    df: pd.DataFrame,
    sort_by: str | None = None,
    ascending: bool = True,
    title: str | None = None,
    height: int = 500,
) -> go.Figure:
    """Render cross-variant summary metrics as a Plotly table.

    Rows are colored by failure mode category. Pass a pre-sorted DataFrame
    or use sort_by to control display order.

    Args:
        df: DataFrame as returned by load_family_comparison.
        sort_by: Column name to sort by before rendering. If None, preserves
            DataFrame order (load_family_comparison sorts by grokking onset).
        ascending: Sort direction when sort_by is specified.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with a styled Table trace.
    """
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending, na_position="last")

    display_cols = [
        "variant_name",
        "prime",
        "seed",
        "grokking_onset_epoch",
        "final_test_loss",
        "frequency_band_count",
        "competition_window_duration",
        "final_circularity",
        "final_fisher_discriminant",
        "failure_mode",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    col_labels = {
        "variant_name": "Variant",
        "prime": "p",
        "seed": "Seed",
        "grokking_onset_epoch": "Grokking Onset",
        "final_test_loss": "Final Test Loss",
        "frequency_band_count": "Freq Bands",
        "competition_window_duration": "Competition Window",
        "final_circularity": "Final Circularity",
        "final_fisher_discriminant": "Fisher Discriminant",
        "failure_mode": "Failure Mode",
    }

    headers = [col_labels.get(c, c) for c in display_cols]
    cell_values = []
    for col in display_cols:
        col_data = df[col]
        if col == "final_test_loss":
            formatted = [f"{v:.5f}" if pd.notna(v) else "—" for v in col_data]
        elif col in ("final_circularity", "final_fisher_discriminant"):
            formatted = [f"{v:.2f}" if pd.notna(v) else "—" for v in col_data]
        elif col in ("grokking_onset_epoch", "frequency_band_count", "competition_window_duration"):
            formatted = [str(int(v)) if pd.notna(v) else "—" for v in col_data]
        else:
            formatted = [str(v) if pd.notna(v) else "—" for v in col_data]
        cell_values.append(formatted)

    # Row colors by failure mode
    failure_modes = df["failure_mode"].tolist() if "failure_mode" in df.columns else []
    row_colors = [
        _FAILURE_MODE_COLORS.get(fm, _FAILURE_MODE_COLORS["unknown"])
        for fm in failure_modes
    ]

    fig = go.Figure(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in headers],
                fill_color="rgb(40, 40, 40)",
                font=dict(color="white", size=12),
                align="left",
                height=30,
            ),
            cells=dict(
                values=cell_values,
                fill_color=[row_colors] * len(display_cols),
                font=dict(color="black", size=11),
                align="left",
                height=26,
            ),
        )
    )

    if title is None:
        title = "Cross-Variant Grokking Health Summary"

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def render_loss_curve_overlay(
    variants: list[Variant],
    align_by_grokking: bool = False,
    grokking_threshold: float = 0.1,
    show_train: bool = False,
    failure_modes: dict[str, str] | None = None,
    title: str | None = None,
    height: int = 450,
    width: int = 1000,
) -> go.Figure:
    """Render test loss curves for all variants on a single chart.

    Args:
        variants: List of Variant objects to include.
        align_by_grokking: If True, shift epoch axis so grokking onset = 0
            for each variant (variants that never grok are excluded).
        grokking_threshold: Loss threshold defining grokking onset.
        show_train: If True, also plot train loss (dashed).
        failure_modes: Optional dict mapping variant_name -> failure_mode for
            coloring. If None, all curves use auto-color.
        title: Custom title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure with one trace per variant.
    """
    fig = go.Figure()

    for variant in variants:
        meta = variant.metadata
        test_losses = np.array(meta["test_losses"], dtype=np.float32)
        train_losses = np.array(meta["train_losses"], dtype=np.float32) if show_train else None
        epochs = np.arange(len(test_losses))

        if align_by_grokking:
            onset = next(
                (i for i, loss in enumerate(test_losses) if loss < grokking_threshold), None
            )
            if onset is None:
                continue
            epochs = epochs - onset

        fm = (failure_modes or {}).get(variant.name, "unknown")
        color = _FAILURE_MODE_COLORS.get(fm, _FAILURE_MODE_COLORS["unknown"])
        short_name = variant.name.split("_")[-2] + "/" + variant.name.split("_")[-1]

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=test_losses,
                mode="lines",
                name=short_name,
                line=dict(color=color, width=1.5),
                hovertemplate=f"{variant.name}<br>Epoch %{{x}}<br>Test Loss: %{{y:.5f}}<extra></extra>",
            )
        )

        if show_train and train_losses is not None:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=train_losses,
                    mode="lines",
                    name=f"{short_name} (train)",
                    line=dict(color=color, width=1.0, dash="dash"),
                    showlegend=False,
                    hovertemplate=f"{variant.name}<br>Train Loss: %{{y:.5f}}<extra></extra>",
                )
            )

    x_label = "Epochs from grokking onset" if align_by_grokking else "Epoch"
    if title is None:
        title = "Loss Curves — All Variants"
        if align_by_grokking:
            title += " (aligned by grokking onset)"

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Test Loss",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=10),
        ),
        height=height,
        width=width,
        margin=dict(l=60, r=150, t=50, b=50),
    )

    return fig
