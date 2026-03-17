"""REQ_063: Fourier Nucleation Predictor Visualizations.

Two renderers for surfacing latent frequency bias from MLP initialization:

- render_nucleation_heatmap: Iteration × frequency heatmap of aggregate spectral
  energy, with neuron peak histogram and convergence traces. Primary diagnostic.

- render_nucleation_frequency_gains: Bar chart of which frequencies gained the
  most energy from raw projection (iteration 0) to final sharpened result.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_TOP_N_CONVERGENCE = 8  # Frequencies to track in convergence traces
_TOP_N_GAINS = 12  # Frequencies to show in gains chart


def render_nucleation_heatmap(
    epoch_data: dict,
    epoch: int | None,
    **kwargs,
) -> go.Figure:
    """Render iteration × frequency heatmap of aggregate spectral energy.

    Shows how the Fourier energy distribution sharpens across iterations.
    Iteration 0 is the raw projection; later iterations amplify dominant
    features. Includes a neuron peak histogram and convergence traces for
    the top-N frequencies.

    Args:
        epoch_data: Single-epoch dict from load_epoch("fourier_nucleation", 0).
            Keys: aggregate_energy (n_iters+1, n_freqs), neuron_peak_freq
            (n_iters+1, d_mlp), frequencies (n_freqs,), prime, iterations.
        epoch: Epoch number for display in title (epoch 0 = initialization).
        **kwargs: Unused.

    Returns:
        Plotly Figure with three panels: heatmap, neuron histogram, convergence.
    """
    agg_energy = epoch_data["aggregate_energy"]  # (n_iters+1, n_freqs)
    peak_freq = epoch_data["neuron_peak_freq"]  # (n_iters+1, d_mlp)
    frequencies = epoch_data["frequencies"]  # (n_freqs,)
    prime = int(epoch_data["prime"])
    n_iters = int(epoch_data["iterations"])

    n_iters_stored = agg_energy.shape[0]
    iter_labels = list(range(n_iters_stored))
    freq_labels = frequencies.tolist()

    # Top frequencies at final iteration
    final_energy = agg_energy[-1]
    top_idx = np.argsort(final_energy)[::-1][:_TOP_N_CONVERGENCE]
    top_freqs = frequencies[top_idx].tolist()

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.6, 0.4],
        subplot_titles=[
            f"Spectral Energy — All Iterations (p={prime})",
            "Convergence Traces (Top Frequencies)",
            "Neuron Peak Frequency Distribution (Final Iteration)",
            "",
        ],
        specs=[[{"colspan": 2}, None], [{}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # --- Row 1: Iteration × frequency heatmap ---
    fig.add_trace(
        go.Heatmap(
            z=agg_energy,
            x=freq_labels,
            y=iter_labels,
            colorscale="Teal",
            showscale=True,
            colorbar=dict(title="Norm. Energy", len=0.55, y=0.78),
            hovertemplate="Iter %{y}, k=%{x}<br>Energy: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Highlight top frequencies with vertical annotations
    for k in top_freqs:
        fig.add_vline(
            x=k,
            line_width=1,
            line_dash="dot",
            line_color="rgba(255, 200, 80, 0.5)",
            row=1,
            col=1,  # type: ignore[arg-type]
        )

    # --- Row 2 left: Neuron peak histogram (final iteration) ---
    final_peaks = peak_freq[-1]  # (d_mlp,)
    peak_counts = np.bincount(
        np.searchsorted(frequencies, final_peaks),
        minlength=len(frequencies),
    )
    bar_colors = [
        "rgba(255, 200, 80, 0.85)" if f in top_freqs else "rgba(30, 160, 160, 0.5)"
        for f in freq_labels
    ]
    fig.add_trace(
        go.Bar(
            x=freq_labels,
            y=peak_counts.tolist(),
            marker_color=bar_colors,
            hovertemplate="k=%{x}<br>Neurons: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # --- Row 2 right: Convergence traces ---
    colors = _make_frequency_colors(top_freqs, top_idx)
    for k, color in zip(top_freqs, colors):
        ki = int(np.searchsorted(frequencies, k))
        trace_energy = agg_energy[:, ki]
        fig.add_trace(
            go.Scatter(
                x=iter_labels,
                y=trace_energy.tolist(),
                mode="lines",
                name=f"k={k}",
                line=dict(color=color, width=2),
                hovertemplate=f"k={k}<br>Iter %{{x}}<br>Energy: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    title = (
        f"Fourier Nucleation — Epoch {epoch}"
        if epoch is not None
        else "Fourier Nucleation — Epoch 0"
    )
    sharpness = float(epoch_data.get("sharpness", 0.7))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=650,
        template="plotly_dark",
        plot_bgcolor="#0a1010",
        paper_bgcolor="#060a0a",
        font=dict(family="IBM Plex Mono, monospace", size=11),
        showlegend=True,
        legend=dict(x=1.0, y=0.3, xanchor="left"),
        annotations=[
            *fig.layout.annotations,  # type: ignore[attr-defined]
            dict(
                text=f"Sharpness: {int(sharpness * 100)}% | Iterations: {n_iters}",
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.01,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=9, color="#3a5858"),
            ),
        ],
    )

    fig.update_xaxes(title_text="Frequency k", row=1, col=1)
    fig.update_yaxes(title_text="Iteration", row=1, col=1, autorange="reversed")
    fig.update_xaxes(title_text="Frequency k", row=2, col=1)
    fig.update_yaxes(title_text="Neurons", row=2, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=2)
    fig.update_yaxes(title_text="Norm. Energy", row=2, col=2)

    return fig


def render_nucleation_frequency_gains(
    epoch_data: dict,
    epoch: int | None,
    **kwargs,
) -> go.Figure:
    """Render which frequencies gained the most energy through sharpening.

    Compares iteration 0 (raw projection) to the final iteration.
    Frequencies with large positive gains are those the initialization is
    most predisposed toward — the latent nucleation seeds.

    Args:
        epoch_data: Single-epoch dict from load_epoch("fourier_nucleation", 0).
        epoch: Epoch number for title (epoch 0 = initialization).
        **kwargs: Unused.

    Returns:
        Plotly Figure with a bar chart of energy gains per frequency.
    """
    agg_energy = epoch_data["aggregate_energy"]  # (n_iters+1, n_freqs)
    frequencies = epoch_data["frequencies"]  # (n_freqs,)
    prime = int(epoch_data["prime"])

    gains = agg_energy[-1] - agg_energy[0]  # (n_freqs,)

    # Sort descending, show top N
    sorted_idx = np.argsort(gains)[::-1][:_TOP_N_GAINS]
    top_freqs = frequencies[sorted_idx]
    top_gains = gains[sorted_idx]

    bar_colors = [
        "rgba(255, 200, 80, 0.9)" if g > 0.05 else "rgba(30, 160, 160, 0.5)" for g in top_gains
    ]

    fig = go.Figure(
        go.Bar(
            x=[f"k={k}" for k in top_freqs],
            y=top_gains.tolist(),
            marker_color=bar_colors,
            hovertemplate="k=%{x}<br>Gain: %{y:.3f}<extra></extra>",
        )
    )

    title = (
        f"Emerging Frequencies — Epoch {epoch} (p={prime})"
        if epoch is not None
        else f"Emerging Frequencies — Epoch 0 (p={prime})"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Frequency k",
        yaxis_title="Energy Gain (final − initial)",
        height=350,
        template="plotly_dark",
        plot_bgcolor="#0a1010",
        paper_bgcolor="#060a0a",
        font=dict(family="IBM Plex Mono, monospace", size=11),
        showlegend=False,
    )

    return fig


def _make_frequency_colors(top_freqs: list[int], top_idx: np.ndarray) -> list[str]:
    """Assign distinguishable colors to top frequencies."""
    palette = [
        "#1ea0a0",
        "#ffc850",
        "#e06080",
        "#80c050",
        "#a060e0",
        "#50b0e0",
        "#e09020",
        "#60e0a0",
    ]
    return [palette[i % len(palette)] for i in range(len(top_freqs))]
