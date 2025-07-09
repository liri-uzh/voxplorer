import os
import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from lib.data_loader import ALLOWED_EXTENSIONS_AUDIO


# --- Helper functions ---
def _split_example_file(sep: str, example_str: str):
    example_str = example_str.rsplit(".", 1)[0]
    return example_str.split(sep)


def _populate_metavars_form(tokens: list | tuple):
    form_elements = []
    for idx, tok in enumerate(tokens):
        form_elements.append(
            dbc.Row(
                [
                    dbc.Label(tok),
                    dbc.Input(
                        id={"type": "metavars-param", "id": f"{idx}"},
                        type="text",
                        placeholder="variable name or ('-' to ignore)",
                    ),
                ]
            )
        )

    return form_elements


# --- Mel-features opts ---
mel_opts = [
    html.A(
        "Documentation",
        href="https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html",
        target="_blank",
        className="mt-2 d-block",
        style={"color": "blue"},
    ),
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Num. MFCCs"),
                    dbc.Input(
                        id={
                            "type": "feat-extr-param",
                            "id": "n_mfccs",
                        },
                        type="number",
                        min=4,
                        max=42,
                        step=1,
                        value=13,
                        required=True,
                    ),
                ]
            ),
            dbc.Col(
                [
                    dbc.Label("Num. Mel filters"),
                    dbc.Input(
                        id={
                            "type": "feat-extr-param",
                            "id": "n_mels",
                        },
                        type="number",
                        min=4,
                        max=128,
                        step=1,
                        value=40,
                        required=True,
                    ),
                ]
            ),
        ],
    ),
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Window length (ms)"),
                    dbc.Input(
                        id={
                            "type": "feat-extr-param",
                            "id": "win_length",
                        },
                        type="number",
                        min=5.0,
                        step=0.1,
                        value=25.0,
                        required=True,
                    ),
                ]
            ),
            dbc.Col(
                [
                    dbc.Label("Overlap (ms)"),
                    dbc.Input(
                        id={
                            "type": "feat-extr-param",
                            "id": "overlap",
                        },
                        type="number",
                        min=2.0,
                        step=0.1,
                        value=10.0,
                        required=True,
                    ),
                ]
            ),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Min. frequency (Hz)"),
                    dbc.Input(
                        id={
                            "type": "feat-extr-param",
                            "id": "fmin",
                        },
                        type="number",
                        min=0.0,
                        max=12_000.0,
                        step=0.1,
                        value=60.0,
                        required=True,
                    ),
                ]
            ),
            dbc.Col(
                [
                    dbc.Label("Max. frequency (Hz)"),
                    dbc.Input(
                        id={
                            "type": "feat-extr-param",
                            "id": "fmax",
                        },
                        type="number",
                        min=4000.0,
                        max=22_000.0,
                        step=0.1,
                        value=10_000.0,
                        required=True,
                    ),
                ]
            ),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Pre-emphasis"),
                    dbc.Input(
                        id={
                            "type": "feat-extr-param",
                            "id": "preemphasis",
                        },
                        type="number",
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.95,
                        required=True,
                    ),
                ]
            ),
            dbc.Col(
                [
                    dbc.Label("Lifter"),
                    dbc.Input(
                        id={
                            "type": "feat-extr-param",
                            "id": "lifter",
                        },
                        type="number",
                        min=0.0,
                        max=128.0,
                        step=0.1,
                        value=22.0,
                        required=True,
                    ),
                ]
            ),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(
                dbc.Switch(
                    id={
                        "type": "feat-extr-param",
                        "id": "delta",
                    },
                    label="delta-features",
                    value=True,
                )
            ),
            dbc.Col(
                dbc.Switch(
                    id={
                        "type": "feat-extr-param",
                        "id": "delta_delta",
                    },
                    label="delta-delta-features",
                    value=True,
                )
            ),
            dbc.Col(
                [
                    dbc.Switch(
                        id={
                            "type": "feat-extr-param",
                            "id": "summarise",
                        },
                        label="summarise",
                        value=False,
                    ),
                    dbc.FormText("Summarise to mean and std. all frames of a file"),
                ]
            ),
        ]
    ),
]


# --- Sp. Embeddings opts ---
sp_emb_opts = [
    html.A(
        "Documentation",
        href="https://huggingface.co/speechbrain",
        target="_blank",
        className="mt-2 d-block",
        style={"color": "blue"},
    ),
    dbc.Label("Pre-trained model"),
    dbc.Input(
        id={
            "type": "feat-extr-param",
            "id": "model",
        },
        type="text",
        value="speechbrain/spkrec-ecapa-voxceleb",
        required=True,
    ),
]


# --- Feature extraction card ---
feature_extraction_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Specify feature extraction")),
        dbc.CardBody(
            [
                dbc.Select(
                    id="feature-extraction-algorithm",
                    options=[
                        {
                            "label": "Mel features",
                            "value": "mel_features",
                        },
                        {
                            "label": "DNN speaker embeddings",
                            "value": "speaker_embeddings",
                        },
                    ],
                    value="mel_features",
                ),
                html.Div(
                    id="feature-extraction-parameters",
                    children=mel_opts,
                ),
                dcc.Interval(
                    id="feature-extraction-params-delayed-update",
                    interval=50,
                    n_intervals=0,
                    disabled=True,
                ),
            ]
        ),
    ]
)


# --- Metavars card ---
metavars_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Specify how to extract metadata")),
        dbc.CardBody(
            [
                dbc.FormText(
                    "Specify the separator character and then name"
                    + " the metadata variables.\n"
                    + "Use '-' to specify variables which should be ignored."
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Example file: ",
                                ),
                                html.P(
                                    "",
                                    id="example-file-label",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Separator", width=2),
                                dbc.Input(
                                    type="text",
                                    value="_",
                                    id="metavars-separator",
                                ),
                            ],
                            width=6,
                        ),
                    ],
                ),
                dbc.Row(
                    html.Div(
                        id="metavars-specification",
                        children=[],
                    ),
                ),
                dcc.Interval(
                    id="metavars-specification-delayed-update",
                    interval=50,
                    n_intervals=0,
                    disabled=True,
                ),
            ],
        ),
    ],
    className="mb-3",
)


# --- Main layout ---
layout = dbc.Row(
    [
        feature_extraction_card,
        metavars_card,
        dbc.Button(
            "Extract features",
            id="extract-features-btn",
            color="primary",
            className="w-100 mb-3",
        ),
        html.Div(
            id="feature-extraction-output",
            className="w-100 mt-3",
        ),
    ],
    style={"display": "none"},
    id="feature-extraction-opts-card",
)


# --- Callback 1: display feature extraction options ---
@callback(
    [
        Output("audio-output", "children"),
        Output("feature-extraction-opts-card", "style", allow_duplicate=True),
        Output("example-file-label", "children"),
    ],
    [
        Input("upload-audio", "filename"),
        Input("stored-table", "data"),
    ],
    prevent_initial_call=True,
)
def display_feature_extraction_opts(
    filenames,
    data_table,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "upload-audio":
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

    else:
        if data_table is None:
            raise PreventUpdate

        return (
            [],
            {"display": "none"},
            "",
        )


# --- Callback 2: update opts layout ---
@callback(
    [
        Output("feature-extraction-parameters", "children"),
        Output("feature-extraction-params-delayed-update", "disabled"),
    ],
    [
        Input("feature-extraction-algorithm", "value"),
        Input("feature-extraction-params-delayed-update", "n_intervals"),
    ],
)
def update_feature_extraction_parameters(algorithm, n_intervals):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Identify trigger input
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # When algo change, clear container and enable interval
    if trigger_id == "feature-extraction-algorithm":
        return [html.Div("Loading...")], False

    # else if interval trigger, create new container
    elif trigger_id == "feature-extraction-params-delayed-update":
        if n_intervals < 1:
            raise PreventUpdate

        if algorithm == "mel_features":
            form_elements = mel_opts
        elif algorithm == "speaker_embeddings":
            form_elements = sp_emb_opts
        else:
            return (
                [
                    dbc.Alert(
                        "Error loading feature extraction options",
                        color="danger",
                        dismissable=True,
                    )
                ],
                True,
            )

    return form_elements, True


# --- Callback 3: metavars interactive specification ---
@callback(
    [
        Output("metavars-specification", "children"),
        Output("metavars-specification-delayed-update", "disabled"),
    ],
    [
        Input("example-file-label", "children"),
        Input("metavars-separator", "value"),
        Input("metavars-specification-delayed-update", "n_intervals"),
    ],
)
def update_metavars_params(example_filename, separator, n_intervals):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Get trigger id
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # When sep changes, clear container and enable interval
    if trigger_id == "metavars-separator":
        return (
            [html.Div("Loading...")],
            False,
        )

    # else create new container
    elif trigger_id in {
        "metavars-specification-delayed-update",
        "example-file-label",
    }:
        try:
            tokens = _split_example_file(
                sep=separator,
                example_str=example_filename,
            )
            form_elements = _populate_metavars_form(tokens)
        except Exception as e:
            return (
                [
                    dbc.Alert(
                        f"Error while updating metavariables specification: {e}",
                        color="danger",
                        dismissable=True,
                    )
                ],
                False,
            )

        return (
            form_elements,
            True,
        )

    else:
        print("update metavars params called but did nothing")
        print(f"Trigger ID: {trigger_id}")
        raise PreventUpdate
