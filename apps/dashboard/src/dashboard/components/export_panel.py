"""Export panel and callbacks (REQ_079).

Provides:
- create_export_notification(): global toast for result feedback (placed in root layout)
- create_export_panel(): collapsible batch export section for the left nav sidebar
- register_export_callbacks(app): single-graph and batch export callbacks
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from dashboard.components.analysis_page import _GRAPH_REGISTRY, get_view_name
from dashboard.state import variant_server_state

# ---------------------------------------------------------------------------
# Root-layout components
# ---------------------------------------------------------------------------


def create_export_notification() -> html.Div:
    """Global export result toast + store — must be in the root layout."""
    return html.Div(
        [
            dcc.Store(id="export-result-store", data=None),
            dbc.Toast(
                id="export-toast",
                header="Export",
                is_open=False,
                dismissable=True,
                duration=8000,
                style={
                    "position": "fixed",
                    "bottom": "20px",
                    "right": "20px",
                    "width": "420px",
                    "zIndex": 9999,
                },
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Sidebar batch export panel
# ---------------------------------------------------------------------------


def create_export_panel() -> html.Div:
    """Collapsible batch export section for the left nav sidebar."""
    return html.Div(
        [
            html.Hr(),
            dbc.Button(
                "⬇ Batch Export",
                id="export-panel-toggle",
                color="secondary",
                outline=True,
                size="sm",
                className="w-100 mb-2",
            ),
            dbc.Collapse(
                id="export-panel-collapse",
                is_open=False,
                children=[
                    dbc.Label("Select views to export:", className="small fw-bold"),
                    dcc.Checklist(
                        id="export-batch-checklist",
                        options=[],
                        value=[],
                        labelStyle={"display": "block", "fontSize": "12px", "marginBottom": "3px"},
                        className="mb-2",
                    ),
                    dbc.Button(
                        "Export Selected",
                        id="export-batch-btn",
                        color="primary",
                        size="sm",
                        className="w-100",
                    ),
                ],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_export_callbacks(app: Dash) -> None:
    @app.callback(
        Output("export-panel-collapse", "is_open"),
        Input("export-panel-toggle", "n_clicks"),
        State("export-panel-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_export_panel(n_clicks: int | None, is_open: bool) -> bool:
        return not is_open

    @app.callback(
        Output("export-batch-checklist", "options"),
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def update_batch_options(_ts: str | None, store_data: dict | None) -> list[dict]:
        """Populate checklist with all views available for the current variant."""
        stored = store_data or {}
        available: set[str] = set(getattr(variant_server_state, "available_views", []))

        seen: set[str] = set()
        options: list[dict] = []
        for view_name in _GRAPH_REGISTRY.values():
            if view_name in seen:
                continue
            seen.add(view_name)
            disabled = bool(stored.get("variant_name")) and view_name not in available
            options.append({"label": view_name, "value": view_name, "disabled": disabled})

        return sorted(options, key=lambda o: o["value"])

    @app.callback(
        Output("export-result-store", "data"),
        Input({"type": "export-btn", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_single_export(n_clicks_list: list[int | None]) -> dict:
        """Export a single graph when its Export button is clicked."""
        triggered = ctx.triggered_id
        if triggered is None or not any(n for n in n_clicks_list if n):
            raise PreventUpdate

        graph_id = triggered["index"]
        view_name = get_view_name(graph_id)
        if not view_name:
            return {"status": "error", "message": f"No registered view for graph: {graph_id}"}

        return _run_export([view_name])

    @app.callback(
        Output("export-result-store", "data", allow_duplicate=True),
        Input("export-batch-btn", "n_clicks"),
        State("export-batch-checklist", "value"),
        prevent_initial_call=True,
    )
    def on_batch_export(n_clicks: int | None, selected_views: list[str] | None) -> dict:
        """Export all selected views sequentially."""
        if not n_clicks or not selected_views:
            raise PreventUpdate
        return _run_export(selected_views)

    @app.callback(
        Output("export-toast", "children"),
        Output("export-toast", "is_open"),
        Output("export-toast", "header"),
        Input("export-result-store", "data"),
        prevent_initial_call=True,
    )
    def show_export_notification(result: dict | None) -> tuple[str, bool, str]:
        if not result:
            raise PreventUpdate
        if result.get("status") == "error":
            return result.get("message", "Export failed"), True, "Export Error"
        paths = result.get("paths", [])
        body = "\n".join(paths) if paths else "No files written"
        return body, True, f"Exported {len(paths)} plot(s)"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _run_export(view_names: list[str]) -> dict:
    """Export a list of view_names using the current server state context."""
    if not hasattr(variant_server_state, "context"):
        return {"status": "error", "message": "No variant loaded — select a variant first."}

    paths: list[str] = []
    errors: list[str] = []

    for view_name in view_names:
        try:
            path = variant_server_state.context.view(view_name).export("png")
            paths.append(str(path))
        except Exception as exc:
            errors.append(f"{view_name}: {exc}")

    if errors and not paths:
        return {"status": "error", "message": "\n".join(errors)}
    return {"status": "ok", "paths": paths, "errors": errors}
