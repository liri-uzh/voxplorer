from dash import html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# --- Metainformation config card ---
layout = dbc.Row(
    [
        dbc.Card(
            [
                dbc.CardHeader(html.H4("Select meta-information variables")),
                dbc.CardBody(
                    [
                        html.Div(
                            id="columns-checklist-container",
                            children=[dbc.Checklist(id="meta-columns-checklist")],
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


# --- Callback 1: display checklist ---
@callback(
    [
        Output("meta-config-card", "style"),
        Output("columns-checklist-container", "children"),
    ],
    [
        Input("stored-data-table", "data"),
    ],
)
def display_columns_checklist(data_table):
    if data_table is None:
        return (
            {"display": "none"},
            [],
        )

    # get column names
    colnames = list(data_table[0].keys())

    # create checklist
    checklist = dbc.Checklist(
        id="meta-columns-checklist",
        options=[{"label": col, "value": col} for col in colnames if col != "index"],
        value=[],
        style={"display": "block"},
    )

    return (
        {"display": "block"},
        [
            html.P(
                "Select the variables that represent meta-information (categorical/text):"
            ),
            checklist,
        ],
    )
