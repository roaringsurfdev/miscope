"""REQ_005: Neuron Activation Heatmaps Visualization.

Renders neuron activations as heatmaps with two viewing modes:
1. Single neuron view - detailed heatmap for one neuron
2. Grid view - multiple neurons displayed together
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_neuron_heatmap(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    neuron_idx: int = 0,
    title: str | None = None,
    colorscale: str = "RdBu",
) -> go.Figure:
    """Render activation heatmap for a single neuron.

    Args:
        epoch_data: Dict containing 'activations' array of shape (d_mlp, p, p).
        epoch: Epoch number (used for title).
        neuron_idx: Which neuron to display (0 to d_mlp-1).
        title: Custom title (default: auto-generated).
        colorscale: Plotly colorscale name.

    Returns:
        Plotly Figure with heatmap.
    """
    activations = epoch_data["activations"]

    n_neurons = activations.shape[0]
    if neuron_idx < 0 or neuron_idx >= n_neurons:
        raise IndexError(f"neuron_idx {neuron_idx} out of range [0, {n_neurons})")

    data = activations[neuron_idx]  # Shape: (p, p)
    p = data.shape[0]

    # Center colorscale at 0
    zmax = max(abs(data.min()), abs(data.max()))
    zmin = -zmax

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=data,
            x=list(range(p)),
            y=list(range(p)),
            colorscale=colorscale,
            zmid=0,
            zmin=zmin,
            zmax=zmax,
            hovertemplate="a=%{y}<br>b=%{x}<br>Activation: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Activation"),
        )
    )

    if title is None:
        title = f"Neuron {neuron_idx} Activations - Epoch {epoch}"

    fig.update_layout(
        title=title,
        xaxis_title="b",
        yaxis_title="a",
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x", constrain="domain"),
        template="plotly_white",
    )

    return fig


def render_neuron_grid(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    neuron_indices: list[int],
    cols: int = 5,
    title: str | None = None,
    colorscale: str = "RdBu",
    subplot_size: int = 150,
) -> go.Figure:
    """Render grid of activation heatmaps for multiple neurons.

    Args:
        epoch_data: Dict containing 'activations' array of shape (d_mlp, p, p).
        epoch: Epoch number (used for title).
        neuron_indices: List of neuron indices to display.
        cols: Number of columns in grid.
        title: Main title for the figure.
        colorscale: Plotly colorscale name.
        subplot_size: Size of each subplot in pixels.

    Returns:
        Plotly Figure with grid of heatmaps.
    """
    activations = epoch_data["activations"]

    n_neurons = len(neuron_indices)
    rows = (n_neurons + cols - 1) // cols

    # Create subplot titles
    subplot_titles = [f"Neuron {idx}" for idx in neuron_indices]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    # Find global color range for consistent scaling
    all_data = activations[neuron_indices]
    zmax = max(abs(all_data.min()), abs(all_data.max()))
    zmin = -zmax

    for i, neuron_idx in enumerate(neuron_indices):
        row = i // cols + 1
        col = i % cols + 1

        data = activations[neuron_idx]

        fig.add_trace(
            go.Heatmap(
                z=data,
                colorscale=colorscale,
                zmid=0,
                zmin=zmin,
                zmax=zmax,
                showscale=(i == 0),  # Only show colorbar for first
                hovertemplate=f"Neuron {neuron_idx}<br>"
                + "a=%{y}, b=%{x}<br>Activation: %{z:.4f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    if title is None:
        title = f"Neuron Activations - Epoch {epoch}"

    fig.update_layout(
        title=title,
        height=rows * subplot_size + 100,
        width=cols * subplot_size + 100,
        template="plotly_white",
    )

    return fig


def render_neuron_across_epochs(
    artifact: dict[str, np.ndarray],
    neuron_idx: int,
    epoch_indices: list[int] | None = None,
    cols: int = 4,
    title: str | None = None,
    colorscale: str = "RdBu",
    subplot_size: int = 150,
) -> go.Figure:
    """Render single neuron's activation across multiple epochs.

    Useful for visualizing how a neuron's activation pattern evolves.

    Args:
        artifact: Dict containing 'epochs' and 'activations' arrays
            (stacked format from ArtifactLoader.load_epochs()).
        neuron_idx: Which neuron to display.
        epoch_indices: Which epoch indices to show (default: all).
        cols: Number of columns in grid.
        title: Main title.
        colorscale: Plotly colorscale name.
        subplot_size: Size of each subplot in pixels.

    Returns:
        Plotly Figure with grid showing neuron across epochs.
    """
    epochs = artifact["epochs"]
    activations = artifact["activations"]

    if epoch_indices is None:
        epoch_indices = list(range(len(epochs)))

    n_epochs_to_show = len(epoch_indices)
    rows = (n_epochs_to_show + cols - 1) // cols

    # Create subplot titles with epoch numbers
    subplot_titles = [f"Epoch {int(epochs[idx])}" for idx in epoch_indices]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    # Find global color range across all selected epochs
    all_data = activations[epoch_indices, neuron_idx]
    zmax = max(abs(all_data.min()), abs(all_data.max()))
    zmin = -zmax

    for i, epoch_idx in enumerate(epoch_indices):
        row = i // cols + 1
        col = i % cols + 1

        data = activations[epoch_idx, neuron_idx]
        epoch = int(epochs[epoch_idx])

        fig.add_trace(
            go.Heatmap(
                z=data,
                colorscale=colorscale,
                zmid=0,
                zmin=zmin,
                zmax=zmax,
                showscale=(i == 0),
                hovertemplate=f"Epoch {epoch}<br>"
                + "a=%{y}, b=%{x}<br>Activation: %{z:.4f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    if title is None:
        title = f"Neuron {neuron_idx} Activation Over Training"

    fig.update_layout(
        title=title,
        height=rows * subplot_size + 100,
        width=cols * subplot_size + 100,
        template="plotly_white",
    )

    return fig


def get_most_active_neurons(
    epoch_data: dict[str, np.ndarray],
    top_k: int = 10,
) -> list[int]:
    """Get indices of neurons with highest activation variance.

    Useful for identifying "interesting" neurons to display.

    Args:
        epoch_data: Dict containing 'activations' array of shape (d_mlp, p, p).
        top_k: Number of top neurons to return.

    Returns:
        List of neuron indices sorted by activation variance.
    """
    activations = epoch_data["activations"]  # (d_mlp, p, p)

    # Compute variance across (a, b) for each neuron
    variances = activations.var(axis=(1, 2))

    # Get top k indices
    top_indices = np.argsort(variances)[-top_k:][::-1]

    return [int(i) for i in top_indices]
