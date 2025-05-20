"""Plotting module.
This module contains functions to plot different types of plots using Plotly Express.
It requires the `plotly` package to be installed.
Almost all functions take in a feature matrix `X` and a target vector `y` as input.
The `X` matrix is assumed to contain all features, while the target vecor `y` is
assumed to contain categorical information (or colouring) information about the data.

Functions
---------
scatter_2d(X: np.ndarray, y: np.ndarray = None, title: str = None, **kwargs)
    Plot 2D scatter plot.
scatter_3d(X: np.ndarray, y: np.ndarray = None, title: str = None, **kwargs)
    Plot 3D scatter plot.
line_plot(X: np.ndarray, y: np.ndarray = None, title: str = None, **kwargs)
    Plot line plot.
bar_plot(X: np.ndarray, y: np.ndarray = None, title: str = None, **kwargs)
    Plot bar plot.
histogram(X: np.ndarray, y: np.ndarray = None, title: str = None, **kwargs)
    Plot histogram.
box_plot(X: np.ndarray, y: np.ndarray = None, title: str = None, **kwargs)
    Plot box plot.
heatmap(X: np.ndarray, z: np.ndarray, title: str = None, **kwargs)
    Plot heatmap.

Raises
------
ImportError if the `plotly` package is not installed.
ValueError if the dimensions of the input data are too small to plot.
ValueWarning if the input data is not in the correct format.
"""

# Imports
from typing_extensions import Union
import numpy as np
import plotly.express as px
import polars as pl
import warnings


def scatter_2d(
    df: Union[dict, pl.DataFrame],
    x: str,
    y: str,
    width=1080,
    height=1080,
    **kwargs,
):
    """Plot 2D scatter plot.

    Parameters
    -----------
    df: dict|polars.DataFrame
        Dataframe-like dictionary or polars DataFrame.
    x: str
        Column name for x-axis.
    y: str
        Column name for y-axis.
    width: int
        The figure's width in pixels. Default=1080.
    height: int
        The figure's height in pixels. Default=1080.
    **kwargs: dict
        Additional keyword arguments.

    Returns
    --------
    fig: plotly.graph_objs.Figure
        Plotly figure object.
    """
    if df is None:
        raise ValueError("No data!")

    fig = px.scatter(
        data_frame=df,
        x=x,
        y=y,
        height=height,
        width=width,
        **kwargs,
    )

    return fig


def scatter_3d(
    df: Union[dict, pl.DataFrame],
    x: str,
    y: str,
    z: str,
    width=1080,
    height=1080,
    **kwargs,
):
    """Plot 3D scatter plot.

    Parameters
    -----------
    df: dict|polars.DataFrame
        Dataframe-like dictionary or polars DataFrame.
    x: str
        Column name for x-axis.
    y: str
        Column name for y-axis.
    z: str
        Column name for z-axis.
    width: int
        The figure's width in pixels. Default=1080.
    height: int
        The figure's height in pixels. Default=1080.
    **kwargs: dict
        Additional keyword arguments.
    """
    if df is None:
        raise ValueError("No data!")

    fig = px.scatter_3d(
        data_frame=df,
        x=x,
        y=y,
        z=z,
        height=height,
        width=width,
        **kwargs,
    )

    return fig


def line_plot(X: np.ndarray, y: np.ndarray = None, title: str = None, **kwargs):
    """Plot line plot.

    Parameters
    -----------
    X: np.ndarray
        Feature matrix (2D array).
    y: np.ndarray
        Target vector.
    title: str
        Plot title.
    **kwargs: dict
        Additional keyword arguments.

    Returns
    --------
    fig: plotly.graph_objs.Figure
        Plotly figure object.
    """
    if X.shape[1] != 2:
        if X.shape[1] < 2:
            warnings.warn(
                "ValueWarning: Input data is not 2D. Plotting first two dimensions."
            )
            X = X[:, :2]
        else:
            raise ValueError("Input data is not 2D. Unable to plot.")
    fig = px.line(x=X[:, 0], y=X[:, 1], color=y, title=title, **kwargs)
    return fig


def bar_plot(X: np.ndarray, y: np.ndarray = None, title: str = None, **kwargs):
    """Plot bar plot.

    Parameters
    -----------
    X: np.ndarray
        Feature matrix (2D array).
    y: np.ndarray
        Target vector.
    title: str
        Plot title.
    **kwargs: dict
        Additional keyword arguments.

    Returns
    --------
    fig: plotly.graph_objs.Figure
        Plotly figure object.
    """
    if X.shape[1] != 2:
        if X.shape[1] < 2:
            warnings.warn(
                "ValueWarning: Input data is not 2D. Plotting first two dimensions."
            )
            X = X[:, :2]
        else:
            raise ValueError("Input data is not 2D. Unable to plot.")
    fig = px.bar(x=X[:, 0], y=X[:, 1], color=y, title=title, **kwargs)
    return fig


def histogram(X: np.ndarray, y: np.ndarray = None, title: str = None, **kwargs):
    """Plot histogram.

    Parameters
    -----------
    X: np.ndarray
        Feature matrix (1D array).
    y: np.ndarray
        Target vector.
    title: str
        Plot title.
    **kwargs: dict
        Additional keyword arguments.
    """
    if X.shape[1] != 1:
        if X.shape[1] < 1:
            warnings.warn(
                "ValueWarning: Input data is not 1D. Plotting first dimension."
            )
            X = X[:, :1]
        else:
            raise ValueError("Input data is not 1D. Unable to plot.")
    fig = px.histogram(x=X, color=y, title=title, **kwargs)
    return fig


def box_plot(X: np.ndarray, y: np.ndarray = None, title: str = None, **kwargs):
    """Plot box plot.

    Parameters
    -----------
    X: np.ndarray
        Feature matrix (2D array).
    y: np.ndarray
        Target vector.
    title: str
        Plot title.
    **kwargs: dict
        Additional keyword arguments.

    Returns
    --------
    fig: plotly.graph_objs.Figure
        Plotly figure object.
    """
    if X.shape[1] != 2:
        if X.shape[1] < 2:
            warnings.warn(
                "ValueWarning: Input data is not 2D. Plotting first two dimensions."
            )
            X = X[:, :2]
        else:
            raise ValueError("Input data is not 2D. Unable to plot.")
    fig = px.box(x=X[:, 0], y=X[:, 1], color=y, title=title, **kwargs)
    return fig


# def choropleth_map():
#     """Plot choropleth map."""
#     pass


def heatmap(X: np.ndarray, z: np.ndarray, title: str = None, **kwargs):
    """Plot heatmap.

    Parameters
    -----------
    X: np.ndarray
        Feature matrix (2D array). Co-ordinates.
    z: np.ndarray
        Target vector. Values.
    title: str
        Plot title.
    **kwargs: dict
        Additional keyword arguments.

    Returns
    --------
    fig: plotly.graph_objs.Figure
        Plotly figure object.
    """
    if X.shape[1] != 2:
        if X.shape[1] < 2:
            warnings.warn(
                "ValueWarning: Input data is not 2D. Plotting first two dimensions."
            )
            X = X[:, :2]
        else:
            raise ValueError("Input data is not 2D. Unable to plot.")
    if z.shape[0] != X.shape[0]:
        raise ValueError("Input data dimensions do not match. Unable to plot.")
    if len(z.shape) != 1 and not z.shape[1] == 1:
        if z.shape[1] > 1:
            warnings.warn(
                "ValueWarning: Input data (z) is not 1D. Plotting first dimension."
            )
            z = z[:, 0]
        else:
            raise ValueError("Input data (z) is not 1D. Unable to plot.")
    fig = px.density_heatmap(x=X[:, 0], y=X[:, 1], z=z, title=title, **kwargs)
    return fig
