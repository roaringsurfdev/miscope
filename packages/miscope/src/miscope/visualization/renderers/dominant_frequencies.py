"""REQ_004: Dominant Embedding Frequencies Visualization.

Renders Fourier coefficient norms as a bar plot with threshold highlighting.
"""

import numpy as np
import plotly.graph_objects as go


def get_fourier_basis_names(n_components: int) -> list[str]:
    """Generate Fourier basis names for labeling.

    The Fourier basis has structure:
    - Index 0: Constant
    - Odd indices (1, 3, 5, ...): Cos terms (cos 1, cos 2, cos 3, ...)
    - Even indices (2, 4, 6, ...): Sin terms (sin 1, sin 2, sin 3, ...)

    Args:
        n_components: Number of Fourier components.

    Returns:
        List of component names.
    """
    names = ["Const"]
    for freq in range(1, (n_components + 1) // 2):
        names.append(f"cos {freq}")
        if len(names) < n_components:
            names.append(f"sin {freq}")
    return names[:n_components]


def get_dominant_indices(coefficients: np.ndarray, threshold: float = 1.0) -> list[int]:
    """Get indices of dominant Fourier components above threshold.

    Args:
        coefficients: 1D array of coefficient norms for a single epoch.
        threshold: Minimum norm to be considered dominant.

    Returns:
        List of indices where coefficient norm exceeds threshold.
    """
    return [int(i) for i in np.where(coefficients > threshold)[0]]


def render_dominant_frequencies(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    threshold: float = 1.0,
    highlight_dominant: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Render dominant frequencies as a bar plot for a single epoch.

    Args:
        epoch_data: Dict containing 'coefficients' array of shape (n_fourier,).
        epoch: Epoch number (used for title).
        threshold: Threshold line for dominance indication.
        highlight_dominant: Whether to highlight bars above threshold.
        title: Custom title (default: auto-generated with epoch number).

    Returns:
        Plotly Figure object ready for display.
    """
    data = epoch_data["coefficients"]
    n_components = len(data)

    # Generate labels
    x_labels = get_fourier_basis_names(n_components)

    # Determine colors based on threshold
    if highlight_dominant:
        colors = [
            "rgba(31, 119, 180, 1.0)" if val > threshold else "rgba(31, 119, 180, 0.4)"
            for val in data
        ]
    else:
        colors = "rgba(31, 119, 180, 0.8)"

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=data,
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Norm: %{y:.4f}<extra></extra>",
        )
    )

    # Add threshold line
    if threshold > 0:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold}",
            annotation_position="top right",
        )

    # Title
    if title is None:
        title = f"Embedding Fourier Coefficients - Epoch {epoch}"

    fig.update_layout(
        title=title,
        xaxis_title="Fourier Component",
        yaxis_title="Norm",
        xaxis_tickangle=-45,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def render_dominant_frequencies_over_time(
    artifact: dict[str, np.ndarray],
    component_indices: list[int] | None = None,
    top_k: int = 5,
    title: str | None = None,
) -> go.Figure:
    """Render selected Fourier components over epochs as line plot.

    Useful for visualizing how frequency dominance evolves during training.

    Args:
        artifact: Dict containing 'epochs' and 'coefficients' arrays
            (stacked format from ArtifactLoader.load_epochs()).
        component_indices: Specific component indices to plot.
            If None, uses top_k components from final epoch.
        top_k: Number of top components to show if component_indices is None.
        title: Custom title.

    Returns:
        Plotly Figure showing component norms over training epochs.
    """
    epochs = artifact["epochs"]
    coefficients = artifact["coefficients"]
    n_components = coefficients.shape[1]

    # Get component labels
    labels = get_fourier_basis_names(n_components)

    # Determine which components to plot
    if component_indices is None:
        # Use top k from final epoch
        final_coeffs = coefficients[-1]
        component_indices = list(np.argsort(final_coeffs)[-top_k:][::-1])

    fig = go.Figure()

    for idx in component_indices:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=coefficients[:, idx],
                mode="lines+markers",
                name=labels[idx],
                hovertemplate=f"<b>{labels[idx]}</b><br>"
                + "Epoch: %{x}<br>Norm: %{y:.4f}<extra></extra>",
            )
        )

    if title is None:
        title = "Dominant Fourier Components Over Training"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Norm",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
