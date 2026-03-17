"""Renderer for the intervention hook verification chart.

Visualizes per-frequency amplitude of hook_attn_out — baseline vs.
hook-modified — so that the user can confirm the gain function is
producing the intended signal change at each checkpoint epoch.

Used by the Intervention Check dashboard page (REQ_070).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go


def render_hook_verification_chart(result: dict[str, Any]) -> go.Figure:
    """Render per-frequency amplitude: baseline vs. hook-modified.

    Two bar traces are drawn side by side for each frequency.  Modified
    bars for target frequencies are coloured to indicate gain direction
    (green = boost, orange = dampen); all other bars are neutral blue.

    Args:
        result: Dict returned by compute_hook_verification().

    Returns:
        Plotly Figure with grouped bar chart.
    """
    baseline: np.ndarray = result["baseline_power"]
    modified: np.ndarray = result["modified_power"]
    freq_labels: list[int] = result["freq_labels"]
    target_freqs: set[int] = set(result["target_frequencies"])
    gain: dict[int, float] = result["gain"]
    ramp_factor: float = result["ramp_factor"]
    epoch: int = result["epoch"]
    epoch_start: int = result["epoch_start"]
    epoch_end: int = result["epoch_end"]
    prime: int = result["prime"]

    x_labels = [str(f) for f in freq_labels]

    # --- Colour each modified bar by its role ---
    modified_colors = []
    for f in freq_labels:
        if f not in target_freqs:
            modified_colors.append("#4e8cc2")  # neutral blue (unchanged)
        elif gain.get(f, 1.0) >= 1.0:
            modified_colors.append("#22c55e")  # green = boost
        else:
            modified_colors.append("#ef4444")  # red = dampen

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=x_labels,
            y=baseline.tolist(),
            marker_color="#94a3b8",
            opacity=0.85,
        )
    )

    fig.add_trace(
        go.Bar(
            name="Hook-Modified",
            x=x_labels,
            y=modified.tolist(),
            marker_color=modified_colors,
            opacity=0.9,
        )
    )

    # Window status annotation
    if ramp_factor == 0.0:
        if epoch < epoch_start:
            window_status = f"epoch {epoch} — before window (window: {epoch_start}–{epoch_end})"
        else:
            window_status = f"epoch {epoch} — after window (window: {epoch_start}–{epoch_end})"
    else:
        window_status = (
            f"epoch {epoch} — ramp factor {ramp_factor:.3f} (window: {epoch_start}–{epoch_end})"
        )

    # Target frequency summary
    boost_freqs = sorted(f for f in target_freqs if gain.get(f, 1.0) >= 1.0)
    dampen_freqs = sorted(f for f in target_freqs if gain.get(f, 1.0) < 1.0)
    target_summary_parts = []
    if boost_freqs:
        target_summary_parts.append(f"boost: {boost_freqs}")
    if dampen_freqs:
        target_summary_parts.append(f"dampen: {dampen_freqs}")
    target_summary = " | ".join(target_summary_parts) if target_summary_parts else "no targets"

    fig.update_layout(
        title=dict(
            text=(
                f"hook_attn_out Frequency Amplitudes (p={prime})<br>"
                f"<sup>{window_status} | {target_summary}</sup>"
            ),
            font=dict(size=14),
        ),
        barmode="group",
        xaxis=dict(
            title="Frequency (1-based)",
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(title="RMS Amplitude"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )

    return fig
