import dash
from dash import dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


# --- Parameter options for each algorithm ---
arg_opts = {
    "pca": {
        "docs-href": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html",
        "params": {
            "svd_solver": {
                "type": "dropdown",
                "options": [
                    {"label": "auto", "value": "auto"},
                    {"label": "full", "value": "full"},
                    {
                        "label": "covariance_eigh",
                        "value": "covariance_eigh",
                    },
                    {"label": "arpack", "value": "arpack"},
                    {"label": "randomized", "value": "randomized"},
                ],
                "default": "auto",
            },
            "tol": {
                "type": "input-numeric",
                "min": 0.0,
                "max": None,
                "step": "any",
                "default": 0.0,
            },
            "random_state": {
                "type": "input-numeric",
                "min": 0,
                "max": None,
                "step": 1,
                "default": 42,
            },
        },
    },
    "umap": {
        "docs-href": "https://umap-learn.readthedocs.io/en/latest/parameters.html",
        "params": {
            "n_neighbors": {
                "type": "input-numeric",
                "min": 2,
                "max": None,
                "step": 1,
                "default": 15,
            },
            "min_dist": {
                "type": "slider",
                "min": 0.0,
                "max": 0.99,
                "step": 0.1,
                "default": 0.1,
            },
            "metric": {
                "type": "dropdown",
                "options": [
                    {"label": "euclidean", "value": "euclidean"},
                    {"label": "manhattan", "value": "manhattan"},
                    {"label": "minkowski", "value": "minkowski"},
                    {"label": "mahalanobis", "value": "mahalanobis"},
                    {"label": "seuclidean", "value": "seuclidean"},
                    {"label": "cosine", "value": "cosine"},
                ],
                "default": "euclidean",
            },
            "random_state": {
                "type": "input-numeric",
                "min": 0,
                "max": None,
                "step": 1,
                "default": 42,
            },
        },
    },
    "tsne": {
        "docs-href": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html",
        "params": {
            "perplexity": {
                "type": "input-numeric",
                "min": 5.0,
                "max": 50.0,
                "step": 0.1,
                "default": 30.0,
            },
            "learning_rate": {
                "type": "dropdown",
                "options": [
                    {"label": "auto", "value": "auto"},
                    {"label": "custom", "value": "custom"},
                ],
                "default": "auto",
                "custom-options": {
                    "type": "input-numeric",
                    "min": 10.0,
                    "max": 1500.0,
                    "step": 0.1,
                    "default": 300.0,
                },
            },
            "max_iter": {
                "type": "input-numeric",
                "min": 250,
                "max": 10000,
                "step": 1,
                "default": 1000,
            },
            "init": {
                "type": "dropdown",
                "options": [
                    {"label": "pca", "value": "pca"},
                    {"label": "random", "value": "random"},
                ],
                "default": "pca",
            },
            "random_state": {
                "type": "input-numeric",
                "min": 0,
                "max": None,
                "step": 1,
                "default": 42,
            },
        },
    },
    "mds": {
        "docs-href": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html",
        "params": {
            "metric": {
                "type": "dropdown",
                "options": [
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},
                ],
                "default": True,
            },
            "n_init": {
                "type": "input-numeric",
                "max": 20,
                "min": 1,
                "step": 1,
                "default": 4,
            },
            "max_iter": {
                "type": "input-numeric",
                "min": 100,
                "max": 10000,
                "step": 1,
                "default": 300,
            },
            "random_state": {
                "type": "input-numeric",
                "min": 0,
                "max": None,
                "step": 1,
                "default": 42,
            },
        },
    },
}


# --- Helper to create forms based on selected algorithm ---
def create_algorithm_params_form(algorithm):
    # Create options dynamically for specific algo
    form_elements = []
    i = 0
    for param, specs in arg_opts[algorithm]["params"].items():
        i += 1
        if specs["type"] == "dropdown":
            # dropdown spec param
            if "custom-options" in specs:
                # dropdown + custom inputs
                form_elements.append(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(param),
                                    dbc.Select(
                                        id={
                                            "type": "dim-red-param",
                                            "id": f"{algorithm}-{param}",
                                        },
                                        options=specs["options"],
                                        value=specs["default"],
                                    ),
                                ]
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("If 'Custom' is selected"),
                                    dbc.Input(
                                        id={
                                            "type": "dim-red-param",
                                            "id": f"{algorithm}-{param}-custom",
                                        },
                                        type="number",
                                        min=specs["custom-options"]["min"],
                                        max=specs["custom-options"]["max"],
                                        step=specs["custom-options"]["step"],
                                        value=specs["custom-options"]["default"],
                                        required=True,
                                        # style={"display": "none"},    # would need callback to show up when custom selected
                                    ),
                                    dbc.FormText("Custom option"),
                                ]
                            ),
                        ],
                        key=f"{algorithm}–{param}-row-{i}",
                    )
                )

            else:
                # only dropdown
                form_elements.append(
                    dbc.Row(
                        [
                            dbc.Label(param),
                            dbc.Select(
                                id={
                                    "type": "dim-red-param",
                                    "id": f"{algorithm}-{param}",
                                },
                                options=specs["options"],
                                value=specs["default"],
                            ),
                        ],
                        key=f"{algorithm}–{param}-row-{i}",
                    )
                )

        elif specs["type"] == "input-numeric":
            # numeric input
            form_elements.append(
                dbc.Row(
                    [
                        dbc.Label(param),
                        dbc.Input(
                            id={
                                "type": "dim-red-param",
                                "id": f"{algorithm}-{param}",
                            },
                            type="number",
                            min=specs["min"],
                            max=specs["max"],
                            step=specs["step"],
                            value=specs["default"],
                            required=True,
                        ),
                    ],
                    id="styled-numeric-input",
                    key=f"{algorithm}–{param}-row-{i}",
                )
            )

        elif specs["type"] == "input-text":
            # text input
            form_elements.append(
                dbc.Row(
                    [
                        dbc.Label(param),
                        dbc.Input(
                            id={
                                "type": "dim-red-param",
                                "id": f"{algorithm}-{param}",
                            },
                            type="text",
                            value=specs["default"],
                            required=True,
                        ),
                    ],
                    key=f"{algorithm}–{param}-row-{i}",
                )
            )

        elif specs["type"] == "slider":
            # slider
            form_elements.append(
                dbc.Row(
                    [
                        dbc.Label(param),
                        dcc.Slider(
                            id={
                                "type": "dim-red-param",
                                "id": f"{algorithm}-{param}",
                            },
                            min=specs["min"],
                            max=specs["max"],
                            step=specs["step"],
                            value=specs["default"],
                        ),
                    ],
                    key=f"{algorithm}–{param}-row-{i}",
                )
            )

    return form_elements


# --- Dim. reduction options card layout ---
layout = dbc.Row(
    dbc.Card(
        [
            dbc.CardHeader(html.H4("Specify options for dimensionality reduction")),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Dimensionality Reduction Algorithm"),
                                    dbc.Select(
                                        id="dim-reduction-algorithm",
                                        options=[
                                            {"label": k.upper(), "value": k}
                                            for k in arg_opts.keys()
                                        ],
                                        value="pca",
                                    ),
                                    html.A(
                                        "Documentation",
                                        id="docs-link",
                                        href=arg_opts["pca"]["docs-href"],
                                        target="_blank",
                                        className="mt-2 d-block",
                                        style={"color": "blue"},
                                    ),
                                ]
                            ),
                            dbc.Col(
                                [
                                    html.Label("Number of dimensions"),
                                    dbc.RadioItems(
                                        id="num-dimensions",
                                        options=[
                                            {"label": str(n), "value": n}
                                            for n in (2, 3)
                                        ],
                                        value=2,
                                        inline=True,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    # algorithm parameter options
                    dbc.Row(
                        [
                            html.Div(
                                id="algorithm-parameters",
                                children=create_algorithm_params_form("pca"),
                            ),
                        ]
                    ),
                    # interval to delay populating container
                    dcc.Interval(
                        id="algorithm-parameters-delayed-update",
                        interval=50,
                        n_intervals=0,
                        disabled=True,
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Run dimensionality reduction",
                                        id="run-dim-red-btn",
                                        color="primary",
                                        className="w-100 mb-3",
                                    ),
                                    html.Div(
                                        id="dim-red-output",
                                        className="w-100 mt-3",
                                    ),
                                ]
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Plot reduced data",
                                        id="plot-data-btn",
                                        color="primary",
                                        className="w-100 mb-3",
                                    ),
                                    html.Div(
                                        id="plot-output",
                                        className="w-100 mt-3",
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ],
        className="mb-4",
        color="secondary",
        inverse=True,
    ),
    style={"display": "none"},
    id="dim-reduction-options",
)


# --- Callback 1: display options ---
@callback(
    [
        Output("dim-reduction-options", "style"),
    ],
    [
        Input("stored-table", "data"),
        Input("confirmed-selection-btn", "n_clicks"),
    ],
)
def display_dimreduction_opts(data_table, n_clicks):
    if data_table is None or n_clicks < 1:
        return ({"display": "none"},)

    return ({"display": "block"},)


# --- Callback 2: update algorithm parameters ---
@callback(
    [
        Output("algorithm-parameters", "children"),
        Output("docs-link", "href"),
        Output("algorithm-parameters-delayed-update", "disabled"),
    ],
    [
        Input("dim-reduction-algorithm", "value"),
        Input("algorithm-parameters-delayed-update", "n_intervals"),
    ],
)
def update_algorithm_parameters(algorithm, n_intervals):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # get documentation link
    docs_link = arg_opts[algorithm]["docs-href"]

    # Identify which input triggered the callback
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # When algo change, clear container and enable interval
    if trigger_id == "dim-reduction-algorithm":
        return [html.Div("Loading...")], docs_link, False

    # else if interval trigger, create new container
    elif trigger_id == "algorithm-parameters-delayed-update":
        if n_intervals < 1:
            raise PreventUpdate

        form_elements = create_algorithm_params_form(algorithm)

    # print(algorithm)
    # print(form_elements)
    # print("\n")

    return form_elements, docs_link, True
