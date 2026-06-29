"""REQ_155: Circuit-spectra trajectory renderer (plot-only).

Thin renderer: it receives already-prepared per-head metric trajectories from the
view's ``load_data`` and assembles a Plotly figure. No data processing,
aggregation, joins, or metric derivation happen here — site/metric selection is a
lookup into prepared data, and the grok-marker epoch is computed in the view
(data) layer. (REQ_099 plot-only line; the no-processing-in-renderer rule.)
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

_METRIC_LABELS = {
    "copying_score": "Copying score  Σ max(Re λ,0) / Σ|λ|",
    "effective_rank": "Effective rank (participation ratio)",
    "operator_norm": "Operator norm (top singular value)",
}


def render_circuit_spectra_trajectory(
    data: dict[str, Any],
    epoch: int | None = None,
    *,
    site: str = "full_ov",
    metric: str = "copying_score",
    title: str | None = None,
) -> go.Figure:
    """Per-head trajectory of one spectral metric for one circuit site over training.

    Args:
        data: Prepared payload from the view's ``load_data`` — ``epochs`` (n,),
            ``series`` mapping ``(site, metric) -> (n_epochs, n_heads)``, the
            available ``sites``/``metrics`` lists, and an optional ``grok_epoch``.
        epoch: Current cursor epoch (drawn as a dotted line); None to omit.
        site: Which circuit to plot (``full_ov``, ``full_qk``, ``ov``, ``qk``,
            ``direct_path``).
        metric: Which spectral metric (``copying_score``, ``effective_rank``,
            ``operator_norm``).
        title: Optional override; a sensible default is derived from site/metric.
    """
    series = data["series"]
    key = (site, metric)
    if key not in series:
        raise KeyError(
            f"No trajectory for site={site!r}, metric={metric!r}. "
            f"Available sites: {data['sites']}; metrics: {data['metrics']}."
        )
    values = series[key]  # (n_epochs, n_heads)
    epochs = data["epochs"]

    fig = go.Figure()
    n_heads = values.shape[1]
    for head in range(n_heads):
        name = f"head {head}" if n_heads > 1 else "circuit"
        fig.add_trace(go.Scatter(x=epochs, y=values[:, head], mode="lines", name=name))

    grok = data.get("grok_epoch")
    if grok is not None:
        fig.add_vline(
            x=grok,
            line_dash="dash",
            line_color="green",
            annotation_text="grok",
            annotation_position="top",
        )
    if epoch is not None:
        fig.add_vline(x=epoch, line_dash="dot", line_color="gray")

    fig.update_layout(
        title=title or f"{site} · {metric} over training",
        xaxis_title="epoch",
        yaxis_title=_METRIC_LABELS.get(metric, metric),
        template="plotly_white",
        legend_title="head",
    )
    return fig
