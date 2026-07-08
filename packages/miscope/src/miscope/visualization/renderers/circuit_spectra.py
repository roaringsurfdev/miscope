"""REQ_155: Circuit-spectra renderers (plot-only).

Thin renderers: each receives already-prepared per-head data from a view's
``load_data`` and assembles a Plotly figure. No data processing, aggregation,
joins, or metric derivation happen here — site/metric/head selection is a lookup
into prepared data; epoch slicing, grok-marker resolution, tensor resolution, and
cross-variant event-relative alignment all live in the view (data) layer. (REQ_099
plot-only line; the no-processing-in-renderer rule, load-bearing for REQ_155.)

Three renderers surface the now-built Layer 4 circuit data (REQ_152/154):
``trajectory`` (per-head metric over training), ``ranking`` (per-head metric at one
epoch, sorted — the copy/transform head split), and ``matrix`` (the composed
``(p, p)`` circuit as a heatmap, resolved from the tensor catalog). A fourth,
``cross_variant``, overlays one metric's trajectory across variants on
event-relative time.
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
        # connectgaps=False: NaN values (e.g. a metric undefined for this circuit)
        # render as gaps rather than a misleading interpolated line.
        fig.add_trace(
            go.Scatter(x=epochs, y=values[:, head], mode="lines", name=name, connectgaps=False)
        )

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


def render_circuit_spectra_ranking(
    data: dict[str, Any],
    epoch: int | None = None,
    *,
    site: str = "full_ov",
    metric: str = "copying_score",
    title: str | None = None,
) -> go.Figure:
    """Per-head spectral metric at one epoch as a sorted bar (copy/transform split).

    The headline use is ``copying_score(full_ov)`` at grok: copy-like heads (high)
    separate from transform-like heads (low). Universal over site/metric — a lookup
    into prepared per-head values, sorted for legibility. The true head index is
    kept on each bar (sorting reorders bars, never relabels them), so a head's
    copy-vs-transform identity stays readable.

    Args:
        data: Prepared payload from the view's ``load_data`` — ``values`` mapping
            ``(site, metric) -> (n_heads,)`` at the bound epoch, the available
            ``sites``/``metrics`` lists, and the resolved ``epoch``.
        epoch: Unused for slicing (the data layer already sliced); accepted for the
            renderer signature and folded into the default title.
        site: Which circuit to rank.
        metric: Which spectral metric to rank by.
        title: Optional override.
    """
    key = (site, metric)
    values_by_key = data["values"]
    if key not in values_by_key:
        raise KeyError(
            f"No values for site={site!r}, metric={metric!r}. "
            f"Available sites: {data['sites']}; metrics: {data['metrics']}."
        )
    values = values_by_key[key]  # (n_heads,)
    order = sorted(range(len(values)), key=lambda h: values[h], reverse=True)
    labels = [f"head {h}" for h in order]
    ranked = [values[h] for h in order]

    fig = go.Figure(go.Bar(x=labels, y=ranked, marker_color="indianred"))
    shown_epoch = data.get("epoch", epoch)
    epoch_str = f" @ epoch {shown_epoch}" if shown_epoch is not None else ""
    fig.update_layout(
        title=title or f"{site} · {metric} per head{epoch_str}",
        xaxis_title="head (sorted)",
        yaxis_title=_METRIC_LABELS.get(metric, metric),
        template="plotly_white",
    )
    return fig


def render_circuit_matrix_heatmap(
    data: dict[str, Any],
    epoch: int | None = None,
    *,
    site: str = "full_ov",
    head: int = 0,
    title: str | None = None,
) -> go.Figure:
    """The composed ``(p, p)`` circuit matrix at one epoch/head as a heatmap.

    Demonstrates the columnar→blob hop: the view's ``load_data`` resolves the dense
    ``circuit_matrix`` tensor through the tensor catalog; this renderer only draws
    it. For modular addition the ``full_ov`` matrix is the source-token → output-logit
    map, so its banded structure is the additive algorithm made visible.

    Args:
        data: Prepared payload — ``matrices_by_site`` mapping ``site -> (n_heads,
            p, p)`` resolved at the bound epoch, the available ``sites`` list, and
            ``epoch``.
        epoch: Unused for slicing; folded into the default title.
        site: Which circuit's matrix to show.
        head: Which head's matrix to show (head-less circuits have ``n_heads=1``).
        title: Optional override.
    """
    by_site = data["matrices_by_site"]
    if site not in by_site:
        raise KeyError(f"No circuit matrix for site={site!r}. Available sites: {data['sites']}.")
    matrices = by_site[site]  # (n_heads, p, p)
    n_heads = matrices.shape[0]
    head = max(0, min(head, n_heads - 1))
    fig = go.Figure(go.Heatmap(z=matrices[head], colorscale="RdBu", zmid=0))
    shown_epoch = data.get("epoch", epoch)
    epoch_str = f" @ epoch {shown_epoch}" if shown_epoch is not None else ""
    head_str = f" · head {head}" if n_heads > 1 else ""
    fig.update_layout(
        title=title or f"{site} circuit matrix{head_str}{epoch_str}",
        xaxis_title="input token",
        yaxis_title="output token",
        template="plotly_white",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig


def render_circuit_spectra_cross_variant(
    data: dict[str, Any],
    epoch: int | None = None,
    *,
    title: str | None = None,
) -> go.Figure:
    """One metric's head-mean trajectory overlaid across variants, event-relative.

    Shows whether a circuit's dynamics (e.g. the OV copy/transform split, or QK
    rank collapse) land at a shared *event-relative* time across variants. All
    alignment (epoch − grok), the head-mean reduction, and per-variant colouring
    happen in the data layer; this renderer overlays the prepared series.

    Args:
        data: Prepared payload — ``series`` (list of ``{label, x, y, color}``), the
            ``site``/``metric`` plotted, and an ``x_title`` (e.g. "epochs since grok").
        epoch: Unused (cross-variant view).
        title: Optional override.
    """
    fig = go.Figure()
    for s in data["series"]:
        fig.add_trace(
            go.Scatter(
                x=s["x"],
                y=s["y"],
                mode="lines",
                name=s["label"],
                line=dict(color=s.get("color")),
                connectgaps=False,
            )
        )
    if any(s["x"] is not None for s in data["series"]):
        fig.add_vline(x=0, line_dash="dash", line_color="green", annotation_text="event")
    site = data.get("site") or ""
    metric = data.get("metric") or ""
    fig.update_layout(
        title=title or f"{site} · {metric} across variants",
        xaxis_title=data.get("x_title", "epoch"),
        yaxis_title=_METRIC_LABELS.get(metric, metric),
        template="plotly_white",
        legend_title="variant",
    )
    return fig
