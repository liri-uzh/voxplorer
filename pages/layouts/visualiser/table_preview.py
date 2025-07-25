import json
from dash import html, callback, Input, Output, State, dash_table, dcc
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
                    dbc.Button(
                        "Reload table",
                        id="reload-table-btn",
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
                            dcc.Download(id="download-reduced-all-csv"),
                            dcc.Download(id="download-logs"),
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
                            dcc.Download(id="download-reduced-selected-csv"),
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
            html.Div(
                html.A(
                    "Filtering syntax help",
                    href="https://dash.plotly.com/datatable/filtering",
                    target="_blank",
                    className="mt-2 d-block",
                ),
                id="filtering-docs-link",
                style={"display": "none"},
            ),
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
        Output("reload-table-btn", "style"),
        Output("download-all-btn", "style"),
        Output("download-selected-btn", "style"),
        Output("filtering-docs-link", "style"),
    ],
    [
        Input("stored-table", "data"),
        Input("stored-metainformation", "data"),
        Input("reload-table-btn", "n_clicks"),
    ],
)
def build_interactive_table(
    data_table,
    meta_columns,
    n_clicks,
):
    if data_table is None and meta_columns is None:
        return (
            [],
            {"display": "none"},
            {"display": "none"},
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
        col_type = "text" if col != "row_index" and col in meta_columns else "numeric"
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
        # TODO: set cell sizes
        # FIXME: when no pagination, if only one row -> row squashed
        interactive_table = dash_table.DataTable(
            id="interactive-table",
            data=data_table,
            columns=columns_config,
            editable=True,
            sort_action="native",
            filter_action="native",
            page_action="none",
            row_selectable="multi",
            row_deletable=False,
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
            virtualization=True,
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
            {"display": "none"},
            {"display": "none"},
        )


# --- Callback 2a: downlaod all data ---
@callback(
    [
        Output("download-all-csv", "data"),
        Output("download-reduced-all-csv", "data"),
        Output("download-logs", "data", allow_duplicate=True),
        Output("download-all-output", "children"),
    ],
    [
        Input("download-all-btn", "n_clicks"),
    ],
    [
        State("stored-table", "data"),
        State("stored-reduced-data", "data"),
        State("processing-logs", "data"),
    ],
    prevent_initial_call=True,
)
def download_all(
    n_clicks,
    data_table,
    reduced_data_table,
    logs,
):
    alert = None
    # Prep data
    to_download_full = None
    try:
        df = pd.DataFrame(data_table)
        to_download_full = dcc.send_data_frame(
            df.to_csv,
            "data_table.csv",
        )
    except Exception as e:
        alert = dbc.Alert(
            f"Error downloading full data: {e}",
            color="danger",
            dismissable=True,
        )

    # Prep reduced data
    if reduced_data_table:
        try:
            df = pd.DataFrame(reduced_data_table)
            to_download_reduced = dcc.send_data_frame(
                df.to_csv,
                "data_table_reduced.csv",
            )
        except Exception as e:
            alert = dbc.Alert(
                f"Error downloading reduced data: {e}",
                color="danger",
                dismissable=True,
            )
    else:
        to_download_reduced = None

    # Prep logs
    if logs:
        try:
            json_logs = json.dumps(logs, indent=2)
            to_download_logs = dict(
                content=json_logs, filename="processing_settings_logs.json"
            )
        except Exception as e:
            print(f"Error converting logs to json: {e}")
            to_download_logs = None
    else:
        to_download_logs = None

    return (
        to_download_full,
        to_download_reduced,
        to_download_logs,
        alert,
    )


# --- Callback 2b: download selected data ---
@callback(
    [
        Output("download-selected-csv", "data"),
        Output("download-reduced-selected-csv", "data"),
        Output("download-logs", "data", allow_duplicate=True),
        Output("download-selected-output", "children"),
    ],
    [
        Input("download-selected-btn", "n_clicks"),
    ],
    [
        State("stored-table", "data"),
        State("stored-reduced-data", "data"),
        State("processing-logs", "data"),
        State("selected-observations", "data"),
    ],
    prevent_initial_call=True,
)
def download_selected(
    n_clicks,
    data_table,
    reduced_data_table,
    logs,
    selectedobservations,
):
    alert = None
    # Prep data
    to_download_full = None
    try:
        df = pd.DataFrame(data_table)

        if selectedobservations is None or len(selectedobservations) == 0:
            alert = dbc.Alert(
                "No observations selected",
                color="warning",
                dismissable=True,
            )
        else:
            df = df.iloc[selectedobservations]
            to_download_full = dcc.send_data_frame(
                df.to_csv,
                "selected_data_table.csv",
            )
    except Exception as e:
        alert = dbc.Alert(
            f"Error downloading data: {e}",
            color="danger",
            dismissable=True,
        )

    # Prep reduced data
    if reduced_data_table:
        try:
            df = pd.DataFrame(reduced_data_table)

            if selectedobservations is None or len(selectedobservations) == 0:
                to_download_reduced = None
            else:
                df = df.iloc[selectedobservations]
                to_download_reduced = dcc.send_data_frame(
                    df.to_csv,
                    "selected_data_table_reduced.csv",
                )
        except Exception as e:
            alert = dbc.Alert(
                f"Error downloading reduced data: {e}",
                color="danger",
                dismissable=True,
            )
    else:
        to_download_reduced = None

    # Prep logs
    if logs:
        try:
            json_logs = json.dumps(logs, indent=2)
            to_download_logs = dict(
                content=json_logs, filename="processing_settings_logs.json"
            )
        except Exception as e:
            print(f"Error converting logs to json: {e}")
            to_download_logs = None
    else:
        to_download_logs = None

    return (
        to_download_full,
        to_download_reduced,
        to_download_logs,
        alert,
    )
