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
                            children=[],
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
    if data_table is not None:
        # get column names
        colnames = list(data_table[0].keys())

        # create checklist
        checklist = dbc.Checklist(
            id="meta-columns-checklist",
            options=[
                {"label": col, "value": col} for col in colnames if col != "index"
            ],
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
    else:
        raise PreventUpdate


# --- Callback 2: store selected metavariables ---
@callback(
    [
        Output("stored-metainformation-table", "data"),
        Output("meta-selection-output", "children"),
    ],
    [
        Input("confirmed-selection-btn", "n_clicks"),
    ],
    [
        State("meta-columns-checklist", "value"),
    ],
    prevent_initial_call=True,
)
def store_metavariables(
    n_clicks,
    meta_columns,
):
    if n_clicks > 0:
        return (
            meta_columns,
            [
                dbc.Alert(
                    "Selection confirmed",
                    "success",
                    dismissable=True,
                )
            ],
        )

    else:
        raise PreventUpdate


# TODO: Make metavars checklist disappear after confirmation
# NOTE:
# app.layout = dbc.Container([
#     # a Store to hold the hidden/showing state
#     dcc.Store(id="meta-card-hidden-store", data=False),
#
#     dbc.Row([
#         dbc.Card(
#             [
#                 dbc.CardHeader(html.H4("Select meta-information variables")),
#                 dbc.CardBody(
#                     [
#                         html.Div(id="columns-checklist-container"),
#                         dbc.Button(
#                             "Confirm selection",
#                             id="confirmed-selection-btn",
#                             color="primary",
#                             className="w-100 mb-3",
#                             n_clicks=0,
#                         ),
#                         # This button only shows after confirm
#                         dbc.Button(
#                             "Change selection",
#                             id="change-selection-btn",
#                             color="secondary",
#                             className="w-100 mb-3",
#                             n_clicks=0,
#                             style={"display": "none"},
#                         ),
#                         html.Div(id="meta-selection-output", className="w-100 mt-3"),
#                     ]
#                 ),
#             ],
#             id="meta-config-card",
#             className="mb-4",
#             color="secondary",
#             inverse=True,
#             style={"display": "block"},
#         ),
#     ]),
#
#     # simulate a stored-data-table for demo
#     dcc.Store(id="stored-data-table", data=[{"index": 0, "foo": 1, "bar": 2}]),
#     dcc.Store(id="stored-metainformation-table", data=[]),
# ])
#
# # --- Callback 1: populate checklist when data arrives -----------------------
# @callback(
#     Output("columns-checklist-container", "children"),
#     Input("stored-data-table", "data"),
# )
# def display_columns_checklist(data_table):
#     if not data_table:
#         raise PreventUpdate
#
#     colnames = [c for c in data_table[0].keys() if c != "index"]
#     checklist = dbc.Checklist(
#         id="meta-columns-checklist",
#         options=[{"label": c, "value": c} for c in colnames],
#         value=[],
#     )
#
#     return [
#         html.P("Select the variables that represent meta-information:"),
#         checklist,
#     ]
#
#
# # --- Callback 2: store the selection & show success alert -------------------
# @callback(
#     [
#         Output("stored-metainformation-table", "data"),
#         Output("meta-selection-output", "children"),
#         Output("meta-card-hidden-store", "data"),
#     ],
#     Input("confirmed-selection-btn", "n_clicks"),
#     State("meta-columns-checklist", "value"),
#     prevent_initial_call=True,
# )
# def store_metavariables(n_clicks, meta_columns):
#     # when confirm is clicked, store the picked columns,
#     # show success, and set hidden‐flag = True
#     if n_clicks:
#         alert = dbc.Alert("Selection confirmed!", color="success", dismissable=True)
#         return meta_columns, alert, True
#     raise PreventUpdate
#
#
# # --- Callback 3: watch `hidden-store` and toggle the Card visible/hidden ---
# @callback(
#     [
#         Output("meta-config-card", "style"),
#         Output("confirmed-selection-btn", "style"),
#         Output("change-selection-btn", "style"),
#     ],
#     Input("meta-card-hidden-store", "data"),
# )
# def toggle_card(hidden):
#     if hidden:
#         # hide the card, hide the "confirm" btn, show the "change selection" btn
#         return {"display": "none"}, {"display": "none"}, {"display": "block"}
#     else:
#         # show the card and the confirm button, hide the "change selection" btn
#         return {"display": "block"}, {"display": "block"}, {"display": "none"}
#
#
# # --- Callback 4: clicking “Change selection” pops card back up -------------
# @callback(
#     Output("meta-card-hidden-store", "data"),
#     Input("change-selection-btn", "n_clicks"),
#     prevent_initial_call=True,
# )
# def change_selection(n):
#     # simply set hidden=False
#     if n:
#         return False
#     raise PreventUpdate
#
#
# if __name__ == "__main__":
#     app.run_server(debug=True)
