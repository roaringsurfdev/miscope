"""Neuron Group PCA renderers.

Two views into within-frequency-group coordination in weight space:
- pca_cohesion: cumulative PC1+PC2+PC3 variance explained per group over epochs,
  with PC1 shown as a dashed reference line
- spread: mean L2 distance from group centroid per group over epochs

High cumulative var → top 3 directions capture most group variance (structured group).
Low cumulative var → group variation is spread across many dimensions (diffuse).
"""

import colorsys

import numpy as np
import plotly.graph_objects as go


def _freq_color(freq_idx: int, n_freq: int) -> str:
    """Consistent HSL color for frequency index (0-indexed)."""
    hue = freq_idx / max(n_freq, 1)
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.5)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def render_neuron_group_pca_cohesion(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Line plot of cumulative PC1+PC2+PC3 variance explained per frequency group.

    Solid lines show cumulative variance explained by the top 3 components.
    Dashed lines show PC1 alone as a reference.
    Each color is one frequency group.

    Args:
        data: cross_epoch artifact from neuron_group_pca
        epoch: optional epoch cursor (vertical line)
    """
    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    pc_var = data["pc_var"]  # (n_epochs, n_groups, 3)
    n_freq = int(group_freqs.max()) + 1 if len(group_freqs) > 0 else 1

    fig = go.Figure()

    for g_idx, (freq, size) in enumerate(zip(group_freqs, group_sizes)):
        color = _freq_color(int(freq), n_freq)
        group_pc = pc_var[:, g_idx, :]  # (n_epochs, 3)

        cumulative = np.nansum(group_pc, axis=1).tolist()
        pc1_only = group_pc[:, 0].tolist()

        legend_name = f"freq {freq} (n={size})"

        fig.add_trace(
            go.Scatter(
                x=epochs.tolist(),
                y=cumulative,
                mode="lines",
                name=legend_name,
                line=dict(color=color, width=2),
                legendgroup=str(freq),
                hovertemplate=(
                    f"freq={freq} n={size}<br>"
                    "epoch=%{x}<br>"
                    "PC1+2+3=%{y:.3f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epochs.tolist(),
                y=pc1_only,
                mode="lines",
                name=f"PC1 only",
                line=dict(color=color, width=1, dash="dash"),
                legendgroup=str(freq),
                showlegend=False,
                hovertemplate=(
                    f"freq={freq} n={size}<br>"
                    "epoch=%{x}<br>"
                    "PC1=%{y:.3f}<extra></extra>"
                ),
            )
        )

    if epoch is not None:
        fig.add_vline(x=epoch, line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"))

    fig.update_layout(
        title="Within-group variance explained — top 3 PCs (W_in)<br>"
              "<sup>Solid = PC1+PC2+PC3 cumulative &nbsp;|&nbsp; Dashed = PC1 alone</sup>",
        xaxis_title="Epoch",
        yaxis_title="Cumulative variance explained",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        height=440,
        margin=dict(l=60, r=20, t=70, b=60),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=10),
        ),
    )
    return fig


def render_neuron_group_spread(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Line plot of mean within-group L2 spread per frequency group over epochs.

    Each line is one frequency group. Low spread means group neurons have
    converged toward a common weight vector. High and rising spread means
    the group is expanding in weight space.

    Args:
        data: cross_epoch artifact from neuron_group_pca
        epoch: optional epoch cursor (vertical line)
    """
    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    mean_spread = data["mean_spread"]  # (n_epochs, n_groups)
    n_freq = int(group_freqs.max()) + 1 if len(group_freqs) > 0 else 1

    fig = go.Figure()

    for g_idx, (freq, size) in enumerate(zip(group_freqs, group_sizes)):
        color = _freq_color(int(freq), n_freq)
        y = mean_spread[:, g_idx].tolist()
        fig.add_trace(
            go.Scatter(
                x=epochs.tolist(),
                y=y,
                mode="lines",
                name=f"freq {freq} (n={size})",
                line=dict(color=color, width=1.5),
                hovertemplate=f"freq={freq} n={size}<br>epoch=%{{x}}<br>spread=%{{y:.4f}}<extra></extra>",
            )
        )

    if epoch is not None:
        fig.add_vline(x=epoch, line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"))

    fig.update_layout(
        title="Within-group mean L2 spread (W_in)",
        xaxis_title="Epoch",
        yaxis_title="Mean L2 distance from group centroid",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=20, t=50, b=60),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=10),
        ),
    )
    return fig
