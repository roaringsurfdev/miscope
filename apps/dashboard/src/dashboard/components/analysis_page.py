"""
Contains helper functions shared across pages that render Variant analysis plots
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from dashboard.state import variant_server_state

# ---------------------------------------------------------------------------
# Global graph registry
# ---------------------------------------------------------------------------
# Populated by AnalysisPageGraphManager.__init__ at import time.
# Maps prefixed graph_id (str) → view_name (str).

_GRAPH_REGISTRY: dict[str, str] = {}


def get_view_name(graph_id: str) -> str | None:
    """Return the view_name registered for a graph ID, or None."""
    return _GRAPH_REGISTRY.get(graph_id)


_SITE_OPTIONS = [
    {"label": "All Sites", "value": "all"},
    {"label": "Post-Embed", "value": "resid_pre"},
    {"label": "Attn Out", "value": "attn_out"},
    {"label": "MLP Out", "value": "mlp_out"},
    {"label": "Resid Post", "value": "resid_post"},
]


class AnalysisPageGraphManager:
    def __init__(self, view_list, page_prefix=None):
        self.view_list = view_list
        self.page_prefix = page_prefix
        # Register all views so the global export callback can resolve graph → view.
        for graph_id, entry in view_list.items():
            view_name = entry.get("view_name")
            if view_name:
                prefixed = f"{page_prefix}-{graph_id}" if page_prefix else graph_id
                _GRAPH_REGISTRY[prefixed] = view_name

    def create_empty_figure(self, message: str = "No data") -> go.Figure:
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

    def create_graph(
        self, graph_id: str, height: str = "400px", view_type: str = "default_graph"
    ) -> html.Div:
        """Create a dcc.Graph wrapped with a per-graph export button."""
        view_type = self.view_list[graph_id].get("view_type")
        if not view_type:
            view_type = "default_graph"

        prefixed_id = f"{self.page_prefix}-{graph_id}" if self.page_prefix else graph_id
        component_id = {"view_type": view_type, "index": prefixed_id}

        return html.Div(
            [
                html.Div(
                    dbc.Button(
                        "⬇ Export",
                        id={"type": "export-btn", "index": prefixed_id},
                        size="sm",
                        color="light",
                        className="ms-auto",
                        title="Export this plot as PNG",
                    ),
                    style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "2px"},
                ),
                dcc.Graph(
                    id=component_id,
                    config={"displayModeBar": True},
                    style={"height": height},
                ),
            ]
        )

    def get_graph_output_list(self, view_filter_set: str | None = None):
        graph_list = []
        views = [
            key
            for key in self.view_list.keys()
            if self.view_list[key].get("view_filter_set") == view_filter_set
        ]
        for view_item in views:
            view_type = self.view_list[view_item].get("view_type")
            graph_id = view_item
            if self.page_prefix:
                graph_id = f"{self.page_prefix}-{graph_id}"

            graph_list.append({"view_type": view_type, "index": graph_id})

        return graph_list

    def update_graphs(
        self,
        variant_data: dict | None,
        view_filter_set: str | None = None,
        view_kwargs: dict | None = None,
    ) -> list[go.Figure]:
        stored = variant_data or {}
        variant_name = stored.get("variant_name")
        last_field_updated = stored.get("last_field_updated")
        figures = []

        # Clear graphs if variant_name is None
        if variant_name is None:
            no_data = self.create_empty_figure("Select a variant")
            figures = [no_data for pid in self.view_list.keys()]

        if last_field_updated in ["variant_name", "intervention_name", "epoch"]:
            # Update graphs
            views = [
                key
                for key in self.view_list.keys()
                if self.view_list[key].get("view_filter_set") == view_filter_set
            ]
            for view_item in views:
                view_name = self.view_list[view_item].get("view_name")
                if view_name in variant_server_state.available_views:
                    if view_kwargs:
                        figures.append(
                            variant_server_state.context.view(view_name).figure(**view_kwargs)
                        )
                    else:
                        figures.append(variant_server_state.context.view(view_name).figure())
                else:
                    figures.append(self.create_empty_figure("No view found"))
        else:
            raise PreventUpdate

        return figures
