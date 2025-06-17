import dash
from dash import dcc, html, Input, Output, State, callback, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Local
from lib.data_loader import parse_audio_contents
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
layout = [
    dbc.Row(
        [
            storage_component,
            upload_component,
            feature_extraction_opts.layout,
        ]
    )
]


# helper for callback 1: process feature methods and metavars
def _process_opts(
    feeature_extraction_method: str,
    feature_opts: list,
    feature_opts_values: list,
    metavar_opts: list,
    metavar_opts_values: list,
    separator: str,
):
    # NOTE: check lib.feature_extraction.FeatureExtractor for info on format
    parsed_features_opts = {feeature_extraction_method: {}}
    parsed_metavars_opts = {
        "variables": [None] * len(metavar_opts),
        "split_char": separator,
    }

    # Parse feature opts
    i = 0
    while i < len(feature_opts):
        param_name = feature_opts[i]["id"]["id"]
        parsed_features_opts[feeature_extraction_method][param_name] = (
            feature_opts_values[i]
        )
        i += 1

    # Parse metavar opts
    i = 0
    while i < len(metavar_opts):
        idx_var = int(metavar_opts[i]["id"]["id"])
        varname = metavar_opts_values[i]

        if varname == "" or varname is None:
            varname = "-"  # Leave empty == ignore

        parsed_metavars_opts["variables"][idx_var] = varname
        i += 1

    return parsed_features_opts, parsed_metavars_opts


# --- Callback 1: feature extraction ---
@callback(
    [
        Output("stored-data-audio", "data"),
        Output("stored-metainformation-audio", "data"),
        Output("audio-output", "children", allow_duplicate=True),
    ],
    [
        Input("extract-features-btn", "n_clicks"),
    ],
    [
        State("upload-audio", "contents"),
        State("upload-audio", "filename"),
        State("feature-extraction-algorithm", "value"),
        State({"type": "feat-extr-param", "id": ALL}, "value"),
        State("metavars-separator", "value"),
        State({"type": "metavars-param", "id": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def extract_features(
    n_clicks,
    contents,
    filenames,
    feat_extr_algo,
    feat_extr_opts_vals,
    separator,
    metav_opts_vals,
):
    if n_clicks < 1:
        raise PreventUpdate

    # Get the full states
    feat_extr_states = dash.callback_context.states_list[-3]
    metav_states = dash.callback_context.states_list[-1]

    # Parse selected options
    parsed_features, parsed_metav = _process_opts(
        feeature_extraction_method=feat_extr_algo,
        feature_opts=feat_extr_states,
        feature_opts_values=feat_extr_opts_vals,
        metavar_opts=metav_states,
        metavar_opts_values=metav_opts_vals,
        separator=separator,
    )

    # Extract features
    data_table, metacols, alert = parse_audio_contents(
        contents=contents,
        filenames=filenames,
        feature_extraction_args=parsed_features,
        metavars=parsed_metav,
    )

    return data_table, metacols, alert
