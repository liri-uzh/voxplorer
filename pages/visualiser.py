###################
# Visualiser mode #
###################

import os
import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# local imports
from lib.data_loader import (
    ALLOWED_EXTENSIONS_AUDIO,
)
from pages.layouts.visualiser.table_upload import table_upload
from pages.layouts.visualiser import (
    feature_extraction_opts,
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
        dcc.Store(id="stored-table", storage_type="session"),
        # table-spec storage for session
        dcc.Store(id="stored-metainformation", storage_type="session"),
        # FIXME: new layouts structure -> audio_upload
        # feature extraction options
        dcc.Store(id="stored-feature-extraction-opts", storage_type="session"),
        # reduced dimensions table
        dcc.Store(id="stored-reduced-data", storage_type="session"),
        # selected observations
        dcc.Store(id="selected-observations", storage_type="session"),
        # FIXME: new layouts structure -> audio_upload
        # temporary table for feature extraction to avoid duplicate outputs in callbacks
        dcc.Store(id="tmp-features-table", storage_type="session"),
        # FIXME: new layouts structure -> audio_upload
        # temporary metavars for feature extraction to avoid duplicate outputs in callbacks
        dcc.Store(id="tmp-features-metainformation", storage_type="session"),
    ]
)


# --- Upload options buttons ---
upload_sel_layout = dbc.Row(
    dbc.Card(
        [
            dbc.CardHeader(html.H4("Upload pre-computed table or feature extraction?")),
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


# --- Layout upload audio ---
layout_upload_audio = dbc.Row(
    [
        html.Div(id="upload-table"),
        dbc.Card(
            [
                dbc.CardHeader(html.H4("Upload audio files (.wav)")),
                dbc.CardBody(
                    [
                        dcc.Upload(
                            id="upload-audio",
                            children=html.Div(
                                [
                                    "Drag and drop or ",
                                    html.A("Select audio files"),
                                ]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                            },
                            multiple=True,
                        ),
                        html.Div(id="audio-output", className="mt-3"),
                    ]
                ),
            ]
        ),
    ],
)


# --- Layout audio ---
layout_audio = [
    layout_upload_audio,
    feature_extraction_opts.layout,
    table_preview.layout,
    dimensionality_reduction_opts.layout,
    plot_layout.layout,
]


# --- Main layout ---
layout = dbc.Container(
    [
        layout_storage,
        upload_sel_layout,
        html.Div(id="data-mode-div"),
        dimensionality_reduction_opts.layout,
        table_preview.layout,
        plot_layout.layout,
    ],
    fluid=True,
)


# --- Callback 1: choice of data mode ---
@callback(
    [
        Output("data-mode-div", "children"),
    ],
    [
        Input("upload-table-btn", "n_clicks"),
        Input("upload-audio-btn", "n_clicks"),
    ],
)
def upload_choice(n_clicks_table, n_clicks_audio):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Identify button clicked
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "upload-table-btn":
        return (table_upload.layout,)
    elif trigger_id == "upload-audio-btn":
        return (layout_audio,)
    else:
        raise PreventUpdate


# TODO: Add callback for when data_table and metainformation
# are stored; interactive (disappearing) metaconfig card in case
# user wants to update matavars. (This can be shared by both table and audio uplaod)
# --- Callback 2: update store components when data is uploaded ---
@callback(
    [
        Output("stored-table", "data"),
        Output("stored-metainformation", "data"),
        Output("stored-data-table", "data", allow_duplicate=True),
        Output("stored-metainformation-table", "data", allow_duplicate=True),
    ],
    [
        Input("confirmed-selection-btn", "n_clicks"),
    ],
    [
        State("stored-data-table", "data"),
        State("stored-metainformation-table", "data"),
    ],
    prevent_initial_call=True,
)
def promote_and_clear_temp_store(
    n_clicks_table,
    data_table,
    metainformation_table,
):
    # TODO: add pipeline for audio upload
    if data_table is None and metainformation_table is None:
        raise PreventUpdate

    new_table = data_table or dash.no_update
    new_meta = metainformation_table or dash.no_update

    # Clear memory used by temporary store components
    cleared_table = None
    cleared_meta = None

    return new_table, new_meta, cleared_table, cleared_meta


# TODO: this goes to the audio_upload layout
# --- Callback 2b: display options ---
@callback(
    [
        Output("audio-output", "children"),
        Output("feature-extraction-opts-card", "style"),
        Output("example-file-label", "children"),
    ],
    [
        Input("upload-audio", "filename"),
    ],
)
def display_feature_extraction_opts(filenames):
    if not filenames:
        raise PreventUpdate

    # Check that all .wav files
    for fl in filenames:
        if not os.path.splitext(fl)[-1].lower() in ALLOWED_EXTENSIONS_AUDIO:
            return (
                dbc.Alert(
                    f"{fl} filetype is not supported."
                    "\nSupported filetypes are: "
                    + f"{', '.join(feature_extraction_opts.supported_filetypes)}",
                    color="danger",
                    dismissable=True,
                ),
                {"display": "none"},
                "",
            )

    return (
        dbc.Alert(
            f"{len(filenames)} files uploaded",
            color="success",
            dismissable=True,
        ),
        {"display": "block"},
        f"{filenames[0]}",
    )


# --- Callback 3: Synchronise table and plot selections ---
@callback(
    [
        Output("selected-observations", "data"),
    ],
    [
        Input("interactive-table", "selected_rows"),
        Input("plot", "selectedData"),
    ],
    [
        State("selected-observations", "data"),
    ],
)
def sync_table_and_plot(
    table_selected_rows,
    plot_selected_data,
    selected_data,
):
    print("\n")
    print("-----")
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Get ID of trigger
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print("Triggered by:", trigger_id)

    # Get current table data and selected rows
    if trigger_id == "interactive-table":
        if selected_data and set(selected_data) == set(table_selected_rows):
            raise PreventUpdate
        else:
            print("new table selections:", table_selected_rows)
            return (table_selected_rows,)
    elif trigger_id == "plot":
        if plot_selected_data:
            print(plot_selected_data)
            plot_selected_rows = [p["pointIndex"] for p in plot_selected_data["points"]]
            if selected_data and set(selected_data) == set(plot_selected_rows):
                raise PreventUpdate
            else:
                print("new plot selections:", plot_selected_rows)
                return (plot_selected_rows,)
        else:
            return ([],)
