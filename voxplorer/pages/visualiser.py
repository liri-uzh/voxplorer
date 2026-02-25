###################
# Visualiser mode #
###################

import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# local imports
from pages.layouts.visualiser.table_upload import table_upload
from pages.layouts.visualiser.audio_upload import audio_upload
from pages.layouts.visualiser import (
    table_preview,
    dimensionality_reduction_opts,
    plot_layout,
)
from lib.plotting import scatter_2d, scatter_3d


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
        # logs
        dcc.Store(id="processing-logs", storage_type="memory"),
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
        Output("stored-reduced-data", "clear_data"),
        Output("selected-observations", "clear_data"),
        Output("plot-layout", "style", allow_duplicate=True),
        Output("dim-red-output", "children", allow_duplicate=True),
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
            True,
            True,
            {"display": "none"},
            None,
        )
    elif trigger_id == "upload-audio-btn":
        return (
            {"display": "none"},
            {"display": "block"},
            table_upload.upload_table_component(),
            audio_upload.upload_audio_component(),
            True,
            True,
            True,
            True,
            {"display": "none"},
            None,
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

    new_meta.insert(0, "row_index")

    return (
        new_table,
        new_meta,
        True,
        True,
        True,
    )


# Callback 3: get selections if only table is present
@callback(
    [
        Output("selected-observations", "data", allow_duplicate=True),
        Output("interactive-table", "selected_rows", allow_duplicate=True),
    ],
    [
        Input("select-all-btn", "n_clicks"),
        Input("deselect-all-btn", "n_clicks"),
        Input("interactive-table", "selected_rows"),
    ],
    [
        State("stored-reduced-data", "data"),
        State("interactive-table", "data"),
        State("interactive-table", "derived_virtual_data"),
    ],
    prevent_initial_call=True,
)
def table_selected_data(
    select_all_n_clicks,
    deselect_all_n_clicks,
    selected_rows,
    reduced_data,
    original_rows,
    filtered_rows,
):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    if reduced_data is not None:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "select-all-btn":
        selected = [i for i, row in enumerate(original_rows) if row in filtered_rows]
    elif trigger_id == "deselect-all-btn":
        selected = []
    else:
        selected = selected_rows or []

    return selected, selected


# --- Callback 4: sync selected data between plot and figure---
@callback(
    [
        Output("selected-observations", "data", allow_duplicate=True),
        Output("plot", "figure", allow_duplicate=True),
        Output("interactive-table", "selected_rows"),
    ],
    [
        Input("select-all-btn", "n_clicks"),
        Input("deselect-all-btn", "n_clicks"),
        Input("plot", "selectedData"),
        Input("interactive-table", "selected_rows"),
    ],
    [
        State("plot", "figure"),
        State("stored-reduced-data", "data"),
        State("stored-metainformation", "data"),
        State("interactive-table", "data"),
        State("interactive-table", "derived_virtual_data"),
        State("num-dimensions", "value"),
        State("dim-reduction-algorithm", "value"),
        State("color-by-dropdown", "value"),
        State("shape-by-dropdown", "value"),
        State("cmap-dropdown", "value"),
        State("theme-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def sync_selected_data(
    select_all_n_clicks,
    deselect_all_n_clicks,
    plot_selected,
    table_selected,
    fig_dict,
    reduced_data,
    metavars,
    original_rows,
    filtered_rows,
    n_components,
    algorithm,
    color,
    symbol,
    color_map,
    template,
):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    if reduced_data is None:
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
        if fig_dict:
            fig = go.Figure(fig_dict)
        else:
            fig = None

    elif trigger_id in {
        "interactive-table",
        "select-all-btn",
        "deselect-all-btn",
    }:
        if trigger_id == "select-all-btn":
            selected = [
                i for i, row in enumerate(original_rows) if row in filtered_rows
            ]
        elif trigger_id == "deselect-all-btn":
            selected = []
        else:
            try:
                selected = table_selected or []
            except Exception as e:
                print(
                    f"Error getting selected points from table: {e}"
                    + f"\ntable_selected={table_selected}"
                )
        try:
            title = f"{algorithm.upper()} {n_components}D embedding"
            if n_components == 2:
                fig = scatter_2d(
                    data=reduced_data,
                    x="DIM1",
                    y="DIM2",
                    color=color,
                    symbol=symbol,
                    selections=selected,
                    hover_data=metavars if metavars else None,
                    color_discrete_sequence=plot_layout.color_opts[color_map],
                    template=template,
                    title=title,
                )
            elif n_components == 3:
                fig = scatter_3d(
                    data=reduced_data,
                    x="DIM1",
                    y="DIM2",
                    z="DIM3",
                    color=color,
                    symbol=symbol,
                    selections=selected,
                    hover_data=metavars if metavars else None,
                    color_discrete_sequence=plot_layout.color_opts[color_map],
                    template=template,
                    title=title,
                )
            else:
                raise ValueError(f"Invalid number of dimensions: {n_components}")
        except Exception as e:
            print(f"Error updating plot selections from table: {e}")
    else:
        print(f"Error syncing selection: trigger_id is {trigger_id}")
        raise PreventUpdate

    if fig:
        fig.update_layout(dragmode="select")

    return (
        selected,
        fig,
        selected,
    )
