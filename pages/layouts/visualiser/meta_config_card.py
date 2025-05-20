from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


# --- Metainformation selection card ---
layout = dbc.Row(
    [
        dbc.Card(
            [
                dbc.CardHeader(html.H4("Select meta-information variables")),
                dbc.CardBody(
                    [
                        html.Div(
                            id="columns-checklist-container",
                            children=[
                                # dummay checklist
                                dbc.Checklist(
                                    id="meta-columns-checklist",
                                    value=[],
                                    # hidden by default
                                    style={"display": "none"},
                                )
                            ],
                        ),
                        dbc.Button(
                            "Confirm selection",
                            id="confirmed-selection-btn",
                            color="primary",
                            className="w-100 mb-3",
                            n_clicks=0,
                        ),
                        html.Div(
                            id="meta-selection-output",
                            className="w-100 mt-3",
                        ),
                    ]
                ),
            ],
            id="meta-config-card",
            className="mb-4",
            color="secondary",
            inverse=True,
            style={"display": "none"},
        ),
    ]
)


# --- Callback 1: display checklist for meta-information specification ---
@callback(
    Output("columns-checklist-container", "children"),
    Input("stored-table", "data"),
)
def display_column_checklist(data_table):
    if data_table is not None:
        # get column names
        col_names = list(data_table[0].keys())

        # create checklist
        checklist = dbc.Checklist(
            id="meta-columns-checklist",
            options=[
                {"label": col, "value": col} for col in col_names if col != "index"
            ],
            value=[],
            style={"display": "block"},
        )

        return html.Div(
            [
                html.P(
                    "Select the variables that represent meta-information (categorical/text):"
                ),
                checklist,
            ]
        )
    else:
        raise PreventUpdate


# --- Callback 2: store meta-information when confirmed
@callback(
    Output("stored-metainformation", "data"),
    Output("meta-selection-output", "children"),
    Input("confirmed-selection-btn", "n_clicks"),
    State("meta-columns-checklist", "value"),
    prevent_initial_call=True,
)
def store_meta_info(n_clicks, meta_columns):
    if n_clicks > 0:
        return (
            meta_columns,
            [dbc.Alert("Selection confirmed", color="success")],
        )
    else:
        PreventUpdate
