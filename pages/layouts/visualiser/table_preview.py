import dash
from dash import html, callback, Input, Output, State, dash_table, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Select All (filtered)",
                        id="select-all-btn",
                        color="secondary",
                        className="mt-3",
                        style={"display": "none"},
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Button(
                        "De-select All",
                        id="deselect-all-btn",
                        color="secondary",
                        className="mt-3",
                        style={"display": "none"},
                    ),
                    width=2,
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Button(
                                "Downlaod All",
                                id="download-all-btn",
                                style={"display": "none"},
                            ),
                            dcc.Download(id="download-all-csv"),
                            html.Div(id="download-all-output"),
                        ],
                    ),
                    width=2,
                    className="justify-content-end",
                    style={"margin-left": "auto"},
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Button(
                                "Downlaod Selected",
                                id="download-selected-btn",
                                style={"display": "none"},
                            ),
                            dcc.Download(id="download-selected-csv"),
                            html.Div(id="download-selected-output"),
                        ],
                    ),
                    width=2,
                    className="justify-content-end",
                    style={"margin-left": "auto"},
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    id="interactive-data-container",
                    children=[dash_table.DataTable(id="interactive-table")],
                ),
                width=12,
            )
        ),
    ]
)


# --- Callback 1: build interactive display for uploaded data ---
@callback(
    [
        Output("interactive-data-container", "children"),
        Output("select-all-btn", "style"),
        Output("deselect-all-btn", "style"),
        Output("download-all-btn", "style"),
        Output("download-selected-btn", "style"),
    ],
    [
        Input("stored-table", "data"),
        Input("stored-metainformation", "data"),
    ],
)
def build_interactive_table(data_table, meta_columns):
    if data_table is None and meta_columns is None:
        return (
            [],
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
        )

    # parse meta columns
    meta_columns = meta_columns if meta_columns is not None else []

    # build table config
    sample_row = data_table[0]
    columns_config = []
    for col in sample_row.keys():
        # Use type 'text' if variable is meta-information; else numeric
        col_type = "text" if col in meta_columns else "numeric"
        columns_config.append(
            {
                "name": col,
                "id": col,
                "type": col_type,
                "selectable": True,
                "editable": True,
            }
        )

    # build interactive table
    try:
        interactive_table = dash_table.DataTable(
            id="interactive-table",
            data=data_table,
            columns=columns_config,
            editable=True,
            sort_action="native",
            filter_action="native",
            row_selectable="multi",
            row_deletable=False,
            page_size=15,
            style_cell={
                "fontFamily": "Arial",
                "padding": "5px 10px",
            },
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            style_table={
                "maxWidth": "100%",
                "overflowX": "auto",
            },
        )

        return (
            dbc.Card(
                dbc.CardBody(interactive_table),
                className="mt-4",
            ),
            {"display": "block"},
            {"display": "block"},
            {"display": "block"},
            {"display": "block"},
        )
    except Exception as e:
        return (
            dbc.Alert(
                f"Error creating interactive table: {e}",
                color="danger",
                dismissable=True,
            ),
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
        )


# --- Callback 2: select all (filtered) data ---
@callback(
    [
        Output("interactive-table", "selected_rows"),
    ],
    [
        Input("select-all-btn", "n_clicks"),
        Input("deselect-all-btn", "n_clicks"),
        # Input("plot", "selectedData"),
        Input("selected-observations", "data"),
    ],
    [
        State("interactive-table", "data"),
        State("interactive-table", "derived_virtual_data"),
        State("interactive-table", "selected_rows"),
    ],
    prevent_initial_call=True,
)
def select_deselect_all(
    select_n_clicks,
    deselect_nclicks,
    selected_data,
    original_rows,
    filtered_rows,
    cur_selected_rows,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # solution from: https://github.com/plotly/dash-table/issues/249#issuecomment-693131768
    # get trigger ID
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print("from select_deselect_all - trigger:", trigger_id)

    if trigger_id == "selected-observations":
        if selected_data is not None:
            if cur_selected_rows and set(selected_data) == set(cur_selected_rows):
                raise PreventUpdate
            else:
                return (selected_data,)
    elif trigger_id == "select-all-btn":
        to_select = [i for i, row in enumerate(original_rows) if row in filtered_rows]
        return (to_select,)
    elif trigger_id == "deselect-all-btn":
        return ([],)
    else:
        raise PreventUpdate


# --- Callback 3a: downlaod all data ---
@callback(
    [
        Output("download-all-csv", "data"),
        Output("download-all-output", "children"),
    ],
    [
        Input("download-all-btn", "n_clicks"),
    ],
    [
        State("stored-table", "data"),
    ],
    prevent_initial_call=True,
)
def download_all(n_clicks, data_table):
    try:
        df = pd.DataFrame(data_table)
        to_download = dcc.send_data_frame(df.to_csv, "data_table.csv")
    except Exception as e:
        return (
            None,
            dbc.Alert(
                f"Error downloading data: {e}",
                color="danger",
                dismissable=True,
            ),
        )

    return (
        to_download,
        None,
    )


# --- Callback 3b: download selected data ---
@callback(
    [
        Output("download-selected-csv", "data"),
        Output("download-selected-output", "children"),
    ],
    [
        Input("download-selected-btn", "n_clicks"),
    ],
    [
        State("stored-table", "data"),
        State("se#lected-observations", "data"),
    ],
    prevent_initial_call=True,
)
def download_selected(n_clicks, data_table, selectedobservations):
    try:
        df = pd.DataFrame(data_table)

        if selectedobservations is None or len(selectedobservations) == 0:
            return (
                None,
                dbc.Alert(
                    "No observations selected",
                    color="warning",
                    dismissable=True,
                ),
            )
        else:
            df = df.iloc[selectedobservations]
            to_download = dcc.send_data_frame(df.to_csv, "selected_data_table.csv")
    except Exception as e:
        return (
            None,
            dbc.Alert(
                f"Error downloading data: {e}",
                color="danger",
                dismissable=True,
            ),
        )

    return (
        to_download,
        None,
    )
