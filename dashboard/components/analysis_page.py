"""
Contains helper functions shared across pages that render Variant analysis plots
"""
import plotly.graph_objects as go
from dash import dcc
from dash.exceptions import PreventUpdate

from dashboard.state import variant_state


class AnalysisPageGraphManager:

    def __init__(self, view_list):
        self.view_list = view_list

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

    def create_graph(self, graph_id: str, height: str = "400px", view_type: str = "default_graph") -> dcc.Graph:
        """Create a dcc.Graph with consistent config."""
        view_type = self.view_list[graph_id].get("view_type")
        if not view_type:
            view_type = "default_graph"

        component_id={'view_type': view_type, 'index': graph_id}
        return dcc.Graph(
            id=component_id,
            config={"displayModeBar": True},
            style={"height": height},
        )

    def get_graph_output_list(self, view_parameter: str | None = None):
        graph_list = []
        views = [key for key in self.view_list.keys() if self.view_list[key].get("view_parameter") == view_parameter]
        for view_item in views:
            view_type = self.view_list[view_item].get("view_type")
            graph_list.append({'view_type': view_type, 'index': view_item})

        return graph_list

    def update_graphs(self, variant_data: dict | None, view_parameter: str | None = None, view_parameter_value: dict | None = None) -> list[go.Figure]:
        stored = variant_data or {}
        variant_name = stored.get("variant_name")
        last_field_updated = stored.get("last_field_updated")
        figures = []

        # Clear graphs if variant_name is None
        if variant_name is None:
            no_data = self.create_empty_figure("Select a variant")
            figures = [no_data for pid in self.view_list.keys()]

        if last_field_updated in ["variant_name", "epoch"]:
            #Update graphs
            views = [key for key in self.view_list.keys() if self.view_list[key].get("view_parameter") == view_parameter]
            print(views)
            for view_item in views:
                view_name = self.view_list[view_item].get("view_name")
                if view_name in variant_state.available_views:
                    if view_parameter_value:
                        figures.append(variant_state.context.view(view_name).figure(**view_parameter_value))
                    else:
                        figures.append(variant_state.context.view(view_name).figure())                    
                else:
                    figures.append(self.create_empty_figure("No view found"))
        else:
            raise PreventUpdate
        
        return figures
