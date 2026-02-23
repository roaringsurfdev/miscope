"""REQ_049: Neuron Fourier Decomposition Visualizations.

Renders per-neuron Fourier magnitude heatmaps for both MLP weight layers,
showing how individual neurons specialize into distinct frequency components
over training. Corresponds to He et al. (2026) Figure 2.
"""

import numpy as np
import plotly.graph_objects as go


def render_neuron_fourier_heatmap(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    layer: str = "input",
    title: str | None = None,
) -> go.Figure:
    """Render per-neuron Fourier magnitude heatmap for one epoch.

    Each row is a neuron; each column is a frequency k. Cell value is the
    Fourier magnitude for that (neuron, frequency) pair. A fully specialized
    neuron shows one bright cell per row.

    Args:
        epoch_data: neuron_fourier artifact containing alpha_mk, beta_mk,
                    freq_indices arrays.
        epoch: Epoch number (used in title).
        layer: "input" (alpha_mk) or "output" (beta_mk).
        title: Custom title (default: auto-generated).

    Returns:
        Plotly Figure with a heatmap of shape (n_neurons, n_frequencies).
    """
    if layer == "input":
        magnitudes = epoch_data["alpha_mk"]
        layer_label = "Input Layer (α)"
    else:
        magnitudes = epoch_data["beta_mk"]
        layer_label = "Output Layer (β)"

    freq_indices = epoch_data["freq_indices"]
    n_neurons, n_freqs = magnitudes.shape

    x_labels = [f"k={k}" for k in freq_indices]
    y_labels = [str(m) for m in range(n_neurons)]

    if title is None:
        title = f"Neuron Fourier Magnitudes — {layer_label} — Epoch {epoch}"

    fig = go.Figure(
        go.Heatmap(
            z=magnitudes,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            colorbar=dict(title="Magnitude"),
            hovertemplate="Neuron %{y}<br>Frequency %{x}<br>Magnitude: %{z:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Frequency k",
        yaxis_title="Neuron",
        yaxis=dict(showticklabels=n_neurons <= 64),
        template="plotly_white",
        height=max(300, min(800, n_neurons * 2)),
    )

    return fig


def render_neuron_fourier_heatmap_output(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    title: str | None = None,
) -> go.Figure:
    """Render per-neuron Fourier magnitude heatmap for the output layer.

    Convenience wrapper around render_neuron_fourier_heatmap for the output
    (β_mk) layer. Intended for side-by-side comparison with the input heatmap.

    Args:
        epoch_data: neuron_fourier artifact.
        epoch: Epoch number.
        title: Custom title.

    Returns:
        Plotly Figure showing output layer magnitudes.
    """
    return render_neuron_fourier_heatmap(
        epoch_data,
        epoch,
        layer="output",
        title=title,
    )
