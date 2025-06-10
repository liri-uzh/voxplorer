import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


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
                    dbc.FormText(
                        "Summarise to mean and std. all frames of a file"),
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


# --- Main layout ---
layout = dbc.Row(
    dbc.Card(
        [
            dbc.CardHeader(html.H4("Specify feature extraction")),
            dbc.CardBody(
                [
                    dbc.Select(
                        id="feature-extraction-algorithm",
                        options=[
                            {
                                "label": "Mel features",
                                "value": "mel",
                            },
                            {
                                "label": "DNN speaker embeddings",
                                "value": "sp_emb",
                            },
                        ],
                        value="mel",
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
                ]
            ),
        ]
    ),
    style={"display": "none"},
    id="feature-extraction-opts-card",
)


# --- Callback 1: update opts layout ---
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

        if algorithm == "mel":
            form_elements = mel_opts
        elif algorithm == "sp_emb":
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
