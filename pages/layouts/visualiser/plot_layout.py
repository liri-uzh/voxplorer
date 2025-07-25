"""Create a plot view of the reduced data dimesnions.

Give option for which meatdata variable to use for colouring and which for shape using dbc.Select inputs.

Create interactive selection function where selections in the plot match selections in the table preview (if no rows are selected than all data is "selected" for plotting).
"""

import dash
from dash import dcc, html, callback, Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import polars as pl

# local imports
from lib.plotting import scatter_2d, scatter_3d
from lib.dimensionality_reduction import (
    pca_reduction,
    mds_reduction,
    umap_reduction,
    tsne_reduction,
)
from pages.layouts.visualiser.dimensionality_reduction_opts import arg_opts


# color options
color_opts = {
    "Plotly": px.colors.qualitative.Plotly,
    "Vivid": px.colors.qualitative.Vivid,
    "Dark24": px.colors.qualitative.Dark24,
    "Light24": px.colors.qualitative.Light24,
    "Bold": px.colors.qualitative.Bold,
    "Pastel": px.colors.qualitative.Pastel,
    "Safe": px.colors.qualitative.Safe,
}


# --- Helper functions ---
def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }


def _prep_data_dim_red(
    data: list[dict],
    metavars: list[str],
) -> np.ndarray:
    """Prepares the data for dimensionality reduction"""
    # convert data to DataFrame
    df = pl.DataFrame(data)

    # Split into np.ndarrays
    if metavars is None or len(metavars) == 0:
        # no metavars -> treat all numeric vars as features
        numeric_cols = [
            col
            for col, dt in df.schema.items()
            if _is_numeric_dtype(dt) and not col == "index"
        ]
        X = df.select(numeric_cols).to_numpy()
    else:
        # split features and metavars
        feature_cols = [
            col
            for col, dt in df.schema.items()
            if _is_numeric_dtype(dt) and not (col in metavars or col == "index")
        ]
        X = df.select(feature_cols).to_numpy()

    return X


def _reconstruct_reduced_data(
    reduced_X: np.ndarray,
    data: list[dict],
    metavars: list[str],
) -> list[dict]:
    """Concatenates the reduced features back with the metadata from
    the original data.
    """
    # original df
    df_original = pl.DataFrame(data)

    # reduceed df
    reduced_cols = [f"DIM{i + 1}" for i in range(reduced_X.shape[1])]
    df_reduced = pl.DataFrame(reduced_X, schema=reduced_cols)

    # concat with metavars
    if metavars and any(col in df_original.columns for col in metavars):
        # if metavars provided and exists in the original data -> extract
        meta_cols = [col for col in metavars if col in df_original.columns]

    # get meta columns
    df_meta = df_original.select(meta_cols)

    # horizontal concatenation
    df_out = df_meta.hstack(df_reduced)

    return df_out.to_dicts()


def _make_styling_dropdowns(
    id: str,
    metavars: list[str] = [],
) -> dbc.Select:
    """Creates dropdown for plot styling.
    If `metavars` is None or [] then only "" option given (i.e no styling).
    """
    options = [{"label": "", "value": None}]
    if metavars:
        options += [{"label": metavar, "value": metavar} for metavar in metavars]

    return dbc.Select(
        id=id,
        options=options,
        value=None,
    )


# --- Layout: plot ---
layout = html.Div(
    [
        dbc.Row(
            [
                html.Div(
                    dcc.Graph(
                        id="plot",
                        responsive=True,
                        style={
                            "height": "100%",
                        },
                    ),
                    style={
                        "width": "100%",  # Fluid width
                        "height": "800px",
                        "border": "2px solid #2196F3",
                        "padding": "10px",
                        "margin": "10px 0",
                    },
                ),
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Color by:"),
                                        html.Div(
                                            id="color-by-div",
                                            children=_make_styling_dropdowns(
                                                id="color-by-dropdown",
                                                metavars=None,
                                            ),
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Shape by:"),
                                        html.Div(
                                            id="shape-by-div",
                                            children=_make_styling_dropdowns(
                                                id="shape-by-dropdown",
                                                metavars=None,
                                            ),
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Color map"),
                                        html.A(
                                            "Color swatches",
                                            href="https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express",
                                            target="_blank",
                                            className="mt-2 d-block",
                                            style={"color": "blue"},
                                        ),
                                        dbc.Select(
                                            id="cmap-dropdown",
                                            options=[
                                                {
                                                    "label": "Plotly",
                                                    "value": "Plotly",
                                                },
                                                {
                                                    "label": "Vivid",
                                                    "value": "Vivid",
                                                },
                                                {
                                                    "label": "Dark24",
                                                    "value": "Dark24",
                                                },
                                                {
                                                    "label": "Light24",
                                                    "value": "Light24",
                                                },
                                                {
                                                    "label": "Bold",
                                                    "value": "Bold",
                                                },
                                                {
                                                    "label": "Pastel",
                                                    "value": "Pastel",
                                                },
                                                {
                                                    "label": "Safe",
                                                    "value": "Safe",
                                                },
                                            ],
                                            value="Bold",
                                        ),
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Theme"),
                                        html.A(
                                            "Templates",
                                            href="https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express",
                                            target="_blank",
                                            className="mt-2 d-block",
                                            style={"color": "blue"},
                                        ),
                                        dbc.Select(
                                            id="theme-dropdown",
                                            options=[
                                                {
                                                    "label": "plotly_white",
                                                    "value": "plotly_white",
                                                },
                                                {
                                                    "label": "plotly_dark",
                                                    "value": "plotly_dark",
                                                },
                                                {
                                                    "label": "simple_white",
                                                    "value": "simple_white",
                                                },
                                                {
                                                    "label": "seaborn",
                                                    "value": "seaborn",
                                                },
                                                {
                                                    "label": "ggplot2",
                                                    "value": "ggplot2",
                                                },
                                            ],
                                            value="plotly_white",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    className="mb-4",
                    color="secondary",
                    inverse=True,
                ),
            ],
            id="plot-layout",
            style={"display": "none"},
        ),
    ]
)


# --- Callback 1: dimension reduction ---
# TODO: add visual loading while computing
@callback(
    [
        Output("stored-reduced-data", "data"),
        Output("dim-red-output", "children", allow_duplicate=True),
        Output("processing-logs", "data", allow_duplicate=True),
    ],
    [
        Input("run-dim-red-btn", "n_clicks"),
    ],
    [
        State("stored-table", "data"),
        State("stored-metainformation", "data"),
        State("processing-logs", "data"),
        State("dim-reduction-algorithm", "value"),
        State("num-dimensions", "value"),
        State({"type": "dim-red-param", "id": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def run_dim_reduction(
    n_clicks,
    data_table,
    metavars,
    logs,
    algorithm,
    n_components,
    values,
):
    if (not n_clicks) or (data_table is None) or (metavars is None):
        raise PreventUpdate

    try:
        # Get algorithm specific params
        algo_opts = arg_opts[algorithm]["params"]

        # Extract input params
        input_params = dash.callback_context.states_list[-1]

        # Init parsed kwargs dict
        parsed_kwargs = {}

        # add n_components to kwargs
        parsed_kwargs["n_components"] = n_components

        # Parse the parameters and values
        i = 0
        while i < len(input_params):
            # Get parameter name
            _, param_name = input_params[i]["id"]["id"].split("-")

            # Check if there is a custom option
            if "custom-options" in algo_opts[param_name]:
                # If value is custom get next value else current one
                if values[i] == "custom":
                    parsed_kwargs[param_name] = values[i + 1]
                else:
                    parsed_kwargs[param_name] = values[i]
                i += 1

            else:
                # Just store param
                parsed_kwargs[param_name] = values[i]

            # Always increse i (if custom we want to skip next)
            i += 1

    except Exception as e:
        return (
            None,
            dbc.Alert(
                f"Error while parsing algorithm parameters: {e}",
                color="danger",
                dismissable=True,
            ),
            None,
        )
    try:
        # Prepare data for dimensionality reduction
        X = _prep_data_dim_red(data=data_table, metavars=metavars)

        # Run dimensionality reduction
        if algorithm == "pca":
            (
                reduced_X,
                _,
                _,
            ) = pca_reduction(
                X=X,
                **parsed_kwargs,
            )
        elif algorithm == "umap":
            reduced_X = umap_reduction(
                X=X,
                **parsed_kwargs,
            )
        elif algorithm == "tsne":
            reduced_X = tsne_reduction(
                X=X,
                **parsed_kwargs,
            )
        elif algorithm == "mds":
            reduced_X = mds_reduction(
                X=X,
                **parsed_kwargs,
            )
        else:
            raise ValueError("Unknown dimensionality reduction algorithm specified.")

    except Exception as e:
        return (
            None,
            dbc.Alert(
                f"Error during dimensionality reduction: {e}",
                color="danger",
                dismissable=True,
            ),
            None,
        )

    try:
        # Reconcatenate reducded data and metavars
        new_data = _reconstruct_reduced_data(reduced_X, data_table, metavars)
    except Exception as e:
        return (
            None,
            dbc.Alert(
                f"Error combining metavars to reduced features: {e}",
                color="danger",
                dismissable=True,
            ),
            None,
        )

    # Write logs
    if logs is not None:
        if not isinstance(logs, dict):
            logs_out = dict(logs)
        else:
            logs_out = logs
    else:
        logs_out = {}

    logs_out["dimensionality_reduction"] = {algorithm: parsed_kwargs}

    return (
        new_data,
        dbc.Alert(
            "Data successfully reduced from "
            + f"{X.shape[1]} to {n_components} dimensions"
            + f" using {algorithm.upper()}",
            color="success",
            dismissable=True,
        ),
        logs_out,
    )


# --- Callback 2: styling dropdowns init ---
@callback(
    [
        Output("color-by-div", "children"),
        Output("shape-by-div", "children"),
        Output("plot-layout", "style"),
    ],
    [
        Input("plot-data-btn", "n_clicks"),
    ],
    [
        State("stored-metainformation", "data"),
    ],
)
def create_style_dropdowns(
    n_clicks,
    metavars,
):
    if not n_clicks:
        raise PreventUpdate

    # Populate color-by-div and shape-by-div based on metavars
    try:
        color_by_children = _make_styling_dropdowns(
            id="color-by-dropdown",
            metavars=metavars[1:] or None,
        )
    except Exception as e:
        print(f"Error creating 'color-by-dropdown: {e}")
        return (
            None,
            None,
            {"display": "none"},
        )
    try:
        shape_by_children = _make_styling_dropdowns(
            id="shape-by-dropdown",
            metavars=metavars[1:] or None,
        )
    except Exception as e:
        print(f"Error creating 'shape-by-dropdown': {e}")
        return (
            None,
            None,
            {"display": "none"},
        )

    return (
        color_by_children,
        shape_by_children,
        {
            "display": "flex",
            "flexDirection": "row",
        },
    )


# TODO: add exception when number of vars is <=3 --> plot anyways and use the correct column names?
# --- Callback 3: Plot data ---
@callback(
    [
        Output("plot", "figure", allow_duplicate=True),
        Output("plot-output", "children"),
    ],
    [
        Input("plot-data-btn", "n_clicks"),
        Input("color-by-dropdown", "value"),
        Input("shape-by-dropdown", "value"),
        Input("cmap-dropdown", "value"),
        Input("theme-dropdown", "value"),
    ],
    [
        State("stored-reduced-data", "data"),
        State("stored-metainformation", "data"),
        State("num-dimensions", "value"),
        State("dim-reduction-algorithm", "value"),
        State("selected-observations", "data"),
    ],
    prevent_initial_call=True,
)
def plot_update(
    n_clicks,
    color,
    symbol,
    color_map,
    template,
    reduced_data,
    metavars,
    n_components,
    algorithm,
    selected_obs,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    alert = None
    stop = False

    if reduced_data is None:
        fig = None
        alert = dbc.Alert(
            "Data must be first reduced before plotting!",
            color="danger",
        )
        stop = True

    # Create figure
    if not stop:
        try:
            title = f"{algorithm.upper()} {n_components}D embedding"
            df = pl.DataFrame(reduced_data)

            # Make sure metavars are categorical
            if color and color in df.columns:
                df = df.cast({color: pl.String})
            if symbol and symbol in df.columns:
                df = df.cast({symbol: pl.String})

            if n_components == 2:
                fig = scatter_2d(
                    data=df,
                    x="DIM1",
                    y="DIM2",
                    color=color,
                    symbol=symbol,
                    selections=selected_obs,
                    hover_data=metavars if metavars else None,
                    color_discrete_sequence=color_opts[color_map],
                    template=template,
                    title=title,
                )
            elif n_components == 3:
                fig = scatter_3d(
                    data=df,
                    x="DIM1",
                    y="DIM2",
                    z="DIM3",
                    color=color,
                    symbol=symbol,
                    selections=selected_obs,
                    hover_data=metavars if metavars else None,
                    color_discrete_sequence=color_opts[color_map],
                    template=template,
                    title=title,
                )
                alert = dbc.Alert(
                    "Interactive selection in the plot is not available for 3D plots.",
                    color="info",
                    dismissable=True,
                )
            else:
                fig = None
                alert = dbc.Alert(
                    f"Invalid number of dimensions: {n_components}",
                    color="danger",
                )
        except Exception as e:
            print(e)
            fig = None
            alert = dbc.Alert(
                f"Error creating figure: {e}",
                color="danger",
                dismissable=True,
            )

        # Set default dragmode
        fig.update_layout(
            dragmode="select",
        )

    return (
        fig,
        alert,
    )
