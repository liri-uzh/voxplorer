###################
# Visualiser mode #
###################

import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from plotly import graph_objects as go

# local imports
from pages.layouts.visualiser.table_upload import table_upload
from pages.layouts.visualiser.audio_upload import audio_upload
from pages.layouts.visualiser import (
    table_preview,
    dimensionality_reduction_opts,
    plot_layout,
)


# init
dash.register_page(__name__, path="/visualiser")

# --- Define default page layout ---
# data storing
layout_storage = html.Div(
    [
        # data storage for session
        dcc.Store(id="stored-table", storage_type="memory"),
        # table-spec storage for session
        dcc.Store(id="stored-metainformation", storage_type="memory"),
        # reduced dimensions table
        dcc.Store(id="stored-reduced-data", storage_type="memory"),
        # selected observations
        dcc.Store(id="selected-observations", storage_type="memory"),
        # Temporary tables based on upload
        # table storage
        dcc.Store(id="stored-data-table", storage_type="memory"),
        # audio storage
        dcc.Store(id="stored-data-audio", storage_type="memory"),
        dcc.Store(id="stored-metainformation-audio", storage_type="memory"),
    ]
)


# --- Upload options buttons ---
upload_sel_layout = dbc.Row(
    dbc.Card(
        [
            dbc.CardHeader(html.H4("Upload a table or upload audio files")),
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Upload table",
                                id="upload-table-btn",
                                color="primary",
                                className="w-100",
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Upload audio",
                                id="upload-audio-btn",
                                color="primary",
                                className="w-100",
                            ),
                            width=6,
                        ),
                    ]
                ),
            ),
        ]
    )
)


# --- Main layout ---
layout = dbc.Container(
    [
        layout_storage,
        upload_sel_layout,
        audio_upload.layout,
        table_upload.layout,
        dimensionality_reduction_opts.layout,
        table_preview.layout,
        plot_layout.layout,
    ],
    fluid=True,
)


# --- Callback 1: choice of data mode ---
@callback(
    [
        Output("upload-table-layout", "style"),
        Output("upload-audio-layout", "style"),
        Output("upload-table-component", "children", allow_duplicate=True),
        Output("upload-audio-component", "children", allow_duplicate=True),
        Output("stored-table", "clear_data"),
        Output("stored-metainformation", "clear_data"),
    ],
    [
        Input("upload-table-btn", "n_clicks"),
        Input("upload-audio-btn", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def upload_choice(n_clicks_table, n_clicks_audio):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Identify button clicked
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "upload-table-btn":
        return (
            {"display": "block"},
            {"display": "none"},
            table_upload.upload_table_component(),
            audio_upload.upload_audio_component(),
            True,
            True,
        )
    elif trigger_id == "upload-audio-btn":
        return (
            {"display": "none"},
            {"display": "block"},
            table_upload.upload_table_component(),
            audio_upload.upload_audio_component(),
            True,
            True,
        )
    else:
        raise PreventUpdate


# --- Callback 2: update store components when data is uploaded ---
@callback(
    [
        Output("stored-table", "data"),
        Output("stored-metainformation", "data"),
        Output("stored-data-table", "clear_data"),
        Output("stored-data-audio", "clear_data"),
        Output("stored-metainformation-audio", "clear_data"),
    ],
    [
        Input("confirmed-selection-btn", "n_clicks"),
        Input("stored-data-audio", "data"),
    ],
    [
        State("stored-data-table", "data"),
        State("meta-columns-checklist", "value"),
        State("stored-metainformation-audio", "data"),
    ],
    prevent_initial_call=True,
)
def promote_and_clear_temp_store(
    n_clicks_table,
    data_audio,
    data_table,
    metainformation_table,
    metainformation_audio,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "confirmed-selection-btn":
        if data_table is None and metainformation_table is None:
            raise PreventUpdate

        new_table = data_table or dash.no_update
        new_meta = metainformation_table or dash.no_update

    elif trigger_id == "stored-data-audio":
        if data_audio is None and metainformation_audio is None:
            raise PreventUpdate

        new_table = data_audio or dash.no_update
        new_meta = metainformation_audio or dash.no_update

    else:
        print("Problem storing data: triggered by none")
        raise PreventUpdate
    return (
        new_table,
        new_meta,
        True,
        True,
        True,
    )


# # --- Callback 3: Synchronise table and plot selections ---
# @callback(
#     [
#         Output("selected-observations", "data"),
#     ],
#     [
#         Input("interactive-table", "selected_rows"),
#         Input("plot", "selectedData"),
#     ],
#     [
#         State("selected-observations", "data"),
#     ],
#     prevent_initial_call=True,
# )
# def sync_table_and_plot(
#     table_selected_rows,
#     plot_selected_data,
#     selected_data,
# ):
#     print("\n")
#     print("-----")
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         raise PreventUpdate
#
#     # Get ID of trigger
#     trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
#     print("Triggered by:", trigger_id)
#
#     # Get current table data and selected rows
#     if trigger_id == "interactive-table":
#         if selected_data and set(selected_data) == set(table_selected_rows):
#             raise PreventUpdate
#         else:
#             print("new table selections:", table_selected_rows)
#             return (table_selected_rows,)
#     elif trigger_id == "plot":
#         if plot_selected_data:
#             print(plot_selected_data)
#             plot_selected_rows = [p["pointIndex"] for p in plot_selected_data["points"]]
#             if selected_data and set(selected_data) == set(plot_selected_rows):
#                 raise PreventUpdate
#             else:
#                 print("new plot selections:", plot_selected_rows)
#                 return (plot_selected_rows,)
#         else:
#             return ([],)


# --- Callback 3: sync selected data ---
# FIXME: I am broken!
@callback(
    [
        Output("selected-observations", "data"),
        Output("plot", "figure", allow_duplicate=True),
        Output("interactive-table", "selected_rows", allow_duplicate=True),
    ],
    [
        Input("plot", "selectedData"),
        Input("interactive-table", "selected_rows"),
    ],
    [
        State("plot", "figure"),
        State("stored-table", "data"),
    ],
    prevent_initial_call=True,
)
def sync_selected_data(
    plot_selected,
    table_selected,
    fig_dict,
    data_table,
):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    if data_table is None:
        raise PreventUpdate

    if fig_dict is None:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(f"triggered by: {trigger_id}")

    selected = []

    if trigger_id == "plot":
        if not plot_selected or "points" not in plot_selected:
            raise PreventUpdate
        # Loop over selected points
        try:
            for pt in plot_selected["points"]:
                selected.append(pt["customdata"][0])
        except Exception as e:
            print(
                f"Error getting selected points from plot: {e}"
                + f"\nplot_selected={plot_selected}"
            )
            raise e

    elif trigger_id == "interactive-table":
        try:
            selected = table_selected or []
        except Exception as e:
            print(
                f"Error getting selected points from table: {e}"
                + f"\ntable_selected={table_selected}"
            )
    else:
        print(f"Error syncing selection: trigger_id is {trigger_id}")
        raise PreventUpdate

    print(f"fig_dict? {bool(fig_dict)}")
    fig = go.Figure(fig_dict) if fig_dict else {}
    print(f"fig: {type(fig)}")
    if fig:
        try:
            for trace in fig.data:
                cd = list(trace.customdata or [])
                print(f"cd: {cd}")
                sel_pts = [i for i, cd_pt in enumerate(cd) if cd_pt[0] in selected]
                print(f"sel_pts: {sel_pts}")
                trace.selectedpoints = sel_pts or None
        except Exception as e:
            print(f"Error when updating fig: {e}")
        fig.update_layout(dragmode="select")

    return (
        selected,
        fig,
        selected,
    )  # TODO: finish this callback and clean up other callbacks
