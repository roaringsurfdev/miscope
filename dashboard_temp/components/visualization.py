import plotly.graph_objects as go
from dash import dcc


def create_empty_figure(message: str = "No data") -> go.Figure:
    """Create a placeholder figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        template="plotly_white",
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

def create_empty_graph(graph_id: str, height: str = "400px", message="No Data") -> dcc.Graph:
    """Create a dcc.Graph with consistent config."""
    return dcc.Graph(
        id=graph_id,
        config={"displayModeBar": True},
        style={"height": height},
        figure=create_empty_figure(message)
    )

def create_graph_from_figure(graph_id: str, figure: go.Figure, height: str = "400px") -> dcc.Graph:
    """Create a dcc.Graph with consistent config."""
    return dcc.Graph(
        id=graph_id,
        config={"displayModeBar": True},
        style={"height": height},
        figure=figure
    )

def create_graph(graph_id: str, height: str = "400px") -> dcc.Graph:
    """Create a dcc.Graph with consistent config."""
    return dcc.Graph(
        id=graph_id,
        config={"displayModeBar": True},
        style={"height": height},
    )
