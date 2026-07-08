"""Line plot utility replacing neel-plotly functionality.

This module provides a line plot function with features similar to neel_plotly.plot.line,
including multi-line support, axis labels, logarithmic scaling, and interactive toggles.
"""

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go


def to_numpy(data) -> np.ndarray:
    """Convert tensor or array-like to numpy array."""
    if hasattr(data, "detach"):
        # PyTorch tensor
        return data.detach().cpu().numpy()
    return np.asarray(data)


def line(
    y,
    x=None,
    *,
    xaxis: str = "",
    yaxis: str = "",
    title: str = "",
    line_labels: Sequence[str | int] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    toggle_x: bool = False,
    toggle_y: bool = False,
    renderer: str | None = None,
) -> go.Figure:
    """Create a line plot with optional multi-line support and axis toggles.

    This function provides similar functionality to neel_plotly.plot.line,
    supporting both single-line and multi-line plots with interactive features.

    Args:
        y: Data to plot. Can be:
            - 1D array for single line
            - 2D array where each row is a separate line
            - List of 1D arrays for multiple lines
        x: X-axis values. If None, uses integer indices.
            Can be array of values or list of labels.
        xaxis: Label for x-axis.
        yaxis: Label for y-axis.
        title: Plot title.
        line_labels: Names for each line (for legend). If None, uses "Line 0", etc.
        log_x: Use logarithmic x-axis.
        log_y: Use logarithmic y-axis.
        toggle_x: Add button to toggle x-axis between linear and log scale.
        toggle_y: Add button to toggle y-axis between linear and log scale.
        renderer: Plotly renderer to use when showing.

    Returns:
        Plotly Figure object (also displayed via .show()).
    """
    # Normalize y to list of 1D arrays
    if isinstance(y, list) and len(y) > 0 and hasattr(y[0], "__len__"):
        # List of arrays
        y_arrays = [to_numpy(arr) for arr in y]
    else:
        y_arr = to_numpy(y)
        if y_arr.ndim == 1:
            y_arrays = [y_arr]
        else:
            # 2D array: each row is a line
            y_arrays = [y_arr[i] for i in range(y_arr.shape[0])]

    n_lines = len(y_arrays)

    # Handle x values
    if x is None:
        x_values = np.arange(len(y_arrays[0]))
        x_labels = None
    else:
        x_arr = to_numpy(x) if not isinstance(x, list) else x
        if isinstance(x_arr, np.ndarray) and x_arr.dtype.kind in ("U", "S", "O"):
            # String labels
            x_labels = list(x_arr)
            x_values = np.arange(len(x_labels))
        elif isinstance(x_arr, list) and len(x_arr) > 0 and isinstance(x_arr[0], str):
            x_labels = x_arr
            x_values = np.arange(len(x_labels))
        else:
            x_values = np.asarray(x_arr)
            x_labels = None

    # Generate line labels if not provided
    if line_labels is None:
        labels = [f"Line {i}" for i in range(n_lines)]
    else:
        labels = [str(lbl) for lbl in line_labels]

    # Create figure
    fig = go.Figure()

    for i, y_data in enumerate(y_arrays):
        name = labels[i] if i < len(labels) else f"Line {i}"
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_data,
                mode="lines",
                name=name,
            )
        )

    # Apply x-axis labels if string
    if x_labels is not None:
        fig.update_xaxes(
            tickmode="array",
            tickvals=x_values,
            ticktext=x_labels,
        )

    # Layout
    layout_updates = {
        "title": title if title else None,
        "xaxis_title": xaxis if xaxis else None,
        "yaxis_title": yaxis if yaxis else None,
        "template": "plotly_white",
        "hovermode": "x unified",
    }

    # Apply log scale
    if log_x:
        layout_updates["xaxis_type"] = "log"
    if log_y:
        layout_updates["yaxis_type"] = "log"

    fig.update_layout(**{k: v for k, v in layout_updates.items() if v is not None})

    # Add toggle buttons if requested
    buttons = []
    if toggle_y:
        buttons.extend(
            [
                {
                    "args": [{"yaxis.type": "linear"}],
                    "label": "Linear Y",
                    "method": "relayout",
                },
                {
                    "args": [{"yaxis.type": "log"}],
                    "label": "Log Y",
                    "method": "relayout",
                },
            ]
        )
    if toggle_x:
        buttons.extend(
            [
                {
                    "args": [{"xaxis.type": "linear"}],
                    "label": "Linear X",
                    "method": "relayout",
                },
                {
                    "args": [{"xaxis.type": "log"}],
                    "label": "Log X",
                    "method": "relayout",
                },
            ]
        )

    if buttons:
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "buttons": buttons,
                    "showactive": True,
                    "x": 0.0,
                    "xanchor": "left",
                    "y": 1.15,
                    "yanchor": "top",
                }
            ]
        )

    # Show the figure
    fig.show(renderer=renderer)

    return fig
