"""REQ_077: Site Gradient Convergence renderers.

Two views over the gradient_site cross-epoch artifact:

  render_site_gradient_convergence — cosine similarity trajectory (primary panel)
      plus raw gradient magnitude trajectory (secondary panel).

  render_site_gradient_heatmap — three-panel heatmap (embedding / attention / MLP)
      showing direction-normalized energy per (epoch, frequency).
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PAIR_LABELS = {
    "similarity_emb_attn": "embedding \u2194 attention",
    "similarity_emb_mlp": "embedding \u2194 MLP",
    "similarity_attn_mlp": "attention \u2194 MLP",
}
_PAIR_COLORS = {
    "similarity_emb_attn": "royalblue",
    "similarity_emb_mlp": "darkorange",
    "similarity_attn_mlp": "seagreen",
}

_SITE_LABELS = {
    "magnitude_embedding": "Embedding",
    "magnitude_attention": "Attention",
    "magnitude_mlp": "MLP",
}
_SITE_COLORS = {
    "magnitude_embedding": "royalblue",
    "magnitude_attention": "darkorange",
    "magnitude_mlp": "seagreen",
}

_ENERGY_SITES = ("embedding", "attention", "mlp")
_ENERGY_SITE_TITLES = {
    "embedding": "Embedding",
    "attention": "Attention (Q+K+V)",
    "mlp": "MLP",
}


def render_site_gradient_convergence(
    data: dict[str, Any],
    epoch: int | None,
    **kwargs: Any,
) -> go.Figure:
    """Render site gradient convergence — similarity trajectory + magnitude panel.

    Args:
        data: From ArtifactLoader.load_cross_epoch("gradient_site"). Contains
            epochs, similarity_emb_attn/emb_mlp/attn_mlp, magnitude_embedding/
            attention/mlp, and window_epochs.
        epoch: Unused (full training arc shown); present for catalog interface.

    Returns:
        Two-panel Plotly figure: top = cosine similarity [0, 1],
        bottom = gradient magnitude (log scale).
    """
    epochs = data["epochs"].tolist()
    window_epochs = data.get("window_epochs", np.array([])).tolist()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Site Convergence (cosine similarity)", "Gradient Magnitude"],
        vertical_spacing=0.12,
    )

    for key, label in _PAIR_LABELS.items():
        raw = data[key].tolist()
        sims = [s if not np.isnan(s) else None for s in raw]
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=sims,
                mode="lines+markers",
                name=label,
                line=dict(color=_PAIR_COLORS[key], width=2),
                marker=dict(size=5),
                connectgaps=False,
            ),
            row=1,
            col=1,
        )

    for key, label in _SITE_LABELS.items():
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=data[key].tolist(),
                mode="lines+markers",
                name=label,
                line=dict(color=_SITE_COLORS[key], width=1.5),
                marker=dict(size=4),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    for ep in window_epochs:
        fig.add_vline(
            x=ep,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
        )

    fig.update_yaxes(range=[0, 1], title_text="Cosine similarity", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Gradient magnitude (L2)", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_layout(height=500, legend=dict(orientation="h", y=1.12, x=0))
    return fig


def render_site_gradient_heatmap(
    data: dict[str, Any],
    epoch: int | None,
    **kwargs: Any,
) -> go.Figure:
    """Render site gradient heatmap — three panels sharing the epoch axis.

    Each panel shows direction-normalized gradient energy per (epoch, frequency)
    for one computational site. Key frequencies are marked as vertical dashed
    lines; window boundaries as horizontal dotted lines.

    Args:
        data: From ArtifactLoader.load_cross_epoch("gradient_site"). Contains
            energy_embedding/attention/mlp (n_sampled × n_freqs), epochs,
            key_frequencies, and window_epochs.
        epoch: Unused; present for catalog interface.

    Returns:
        One-row three-panel Plotly figure with shared y-axis.
    """
    epochs = data["epochs"].tolist()
    window_epochs = data.get("window_epochs", np.array([])).tolist()
    key_freqs = data.get("key_frequencies", np.array([])).tolist()
    prime = int(data.get("prime", np.array([113]))[0])
    n_freqs = prime // 2
    freq_labels = list(range(1, n_freqs + 1))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[_ENERGY_SITE_TITLES[s] for s in _ENERGY_SITES],
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )

    for col_idx, site in enumerate(_ENERGY_SITES):
        z = data[f"energy_{site}"].tolist()
        show_scale = col_idx == 2
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=freq_labels,
                y=epochs,
                colorscale="Viridis",
                showscale=show_scale,
                colorbar=dict(title="Energy", len=0.75) if show_scale else None,
            ),
            row=1,
            col=col_idx + 1,
        )
        for kf in key_freqs:
            fig.add_vline(
                x=kf,
                line_dash="dash",
                line_color="white",
                opacity=0.6,
                row=1,
                col=col_idx + 1,
            )

    for ep in window_epochs:
        fig.add_hline(y=ep, line_dash="dot", line_color="red", opacity=0.4)

    fig.update_yaxes(title_text="Epoch", col=1)
    fig.update_xaxes(title_text="Frequency k")
    fig.update_layout(height=500, width=1100)
    return fig
