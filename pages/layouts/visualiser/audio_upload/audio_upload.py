import os
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Local
from lib.data_loader import parse_audio_contents, ALLOWED_EXTENSIONS_AUDIO
from pages.layouts.visualiser.audio_upload import feature_extraction_opts

# --- Setup audio-upload storage ---
storage_component = html.Div(
    [
        # data
        dcc.Store(id="stored-data-audio", storage_type="memory"),
        # metadata variables
        dcc.Store(id="stored-metainformation-audio", storage_type="memory"),
    ]
)

# --- Upload ---
upload_component = html.Div(
    [
        dbc.Card(
            [
                dbc.CardHeader(html.H4("Upload audio files. (.wav, .flac, .mp3)")),
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
        )
    ]
)


# --- Main layout ---
layout = dbc.Row(
    [
        storage_component,
        upload_component,
        feature_extraction_opts.layout,
    ]
)
# ]


# --- Callback 1: display feature extraction options ---
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
                    + "\nSupported filetypes are: "
                    + f"{', '.join(ALLOWED_EXTENSIONS_AUDIO)}",
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
