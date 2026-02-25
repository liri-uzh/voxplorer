"""Plotting module for dash app."""

from typing import Sequence, Union

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import polars as pl


def scatter_2d(
    data: Union[list, pl.DataFrame, pd.DataFrame],
    x: str,
    y: str,
    width: int = 1080,
    height: int = 1080,
    color: str = None,
    symbol: str = None,
    selections: Sequence[int] = None,
    color_discrete_sequence: Sequence[str] = px.colors.qualitative.Plotly,
    hover_data: Sequence[str] = None,
    title: str = None,
    template: str = None,
) -> go.Figure:
    if isinstance(data, pl.DataFrame):
        df = data.to_pandas()
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(
            "Invalid data parsed. Must be dict, polars.DataFrame, or pandas.DataFrame."
        )

    # Get grouping keys
    group_cols = []
    if color:
        group_cols.append(color)
    if symbol:
        group_cols.append(symbol)

    # Build figure
    fig = go.Figure()
    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        grouped = [(None, df)]

    # Prep maps
    palette = color_discrete_sequence
    symbols = [
        "circle",
        "square",
        "cross",
        "diamond",
        "x",
        "circle-open",
        "square-open",
        "diamond-open",
    ]
    unique_colors = df[color].dropna().unique() if color else []
    unique_symbols = df[symbol].dropna().unique() if symbol else []

    # Trace
    for grp, subdf in grouped:
        if grp is None:
            name = ""
        elif isinstance(grp, tuple):
            name = " | ".join(str(g) for g in grp)
        else:
            name = str(grp)

        # Markers
        mk = dict(size=8, opacity=0.8)
        if color and color in subdf.columns:
            i = list(unique_colors).index(grp if not isinstance(grp, tuple) else grp[0])
            mk["color"] = palette[i % len(palette)]

        if symbol and symbol in subdf.columns:
            j = list(unique_symbols).index(
                grp if not isinstance(grp, tuple) else grp[1]
            )
            mk["symbol"] = symbols[j % len(symbols)]

        # Hover template
        idx = np.asarray(subdf.index).reshape(-1, 1)
        if hover_data:
            cd = subdf[hover_data].to_numpy()
            cd = np.hstack([idx, cd])
            hint = (
                "<br>".join(
                    [f"{c}: %{{customdata[{k + 1}]}}" for k, c in enumerate(hover_data)]
                )
                + "<extra></extra>"
            )

        else:
            cd = idx
            hint = f"row_index: %{{customdata[{0}]}}"

        fig.add_trace(
            go.Scatter(
                x=subdf[x],
                y=subdf[y],
                mode="markers",
                name=name,
                marker=mk,
                customdata=cd,
                hovertemplate=hint,
                selected=dict(marker=dict(opacity=0.8)),
                unselected=dict(marker=dict(opacity=0.2)),
            )
        )
        if selections:
            trace = fig.data[-1]
            cd = list(trace.customdata)
            sel_pts = [i for i, cd_pt in enumerate(cd) if cd_pt[0] in selections]
            trace.selectedpoints = sel_pts

    # Final touches
    fig.update_layout(
        title=title,
        template=template,
        width=width,
        height=height,
    )

    return fig


def scatter_3d(
    data: Union[list, pd.DataFrame, pl.DataFrame],
    x: str,
    y: str,
    z: str,
    width: int = 1080,
    height: int = 1080,
    color: str = None,
    symbol: str = None,
    selections: Sequence[int] = None,
    color_discrete_sequence: Sequence[str] = px.colors.qualitative.Plotly,
    hover_data: Sequence[str] = None,
    title: str = None,
    template: str = None,
):
    if isinstance(data, pl.DataFrame):
        df = data.to_pandas()
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(
            "Invalid data parsed. Must be dict, polars.DataFrame, or pandas.DataFrame."
        )

    # Build figure
    fig = go.Figure()
    traced_groups = set()

    # Prep maps
    palette = color_discrete_sequence
    symbols = [
        "circle",
        "square",
        "cross",
        "diamond",
        "x",
        "circle-open",
        "square-open",
        "diamond-open",
    ]
    unique_colors = df[color].dropna().unique() if color else []
    unique_symbols = df[symbol].dropna().unique() if symbol else []

    # Traces
    for idx, row in df.iterrows():
        grp = None
        if color is not None and symbol is not None:
            grp = (row[color], row[symbol])
        elif color is not None:
            grp = row[color]
        elif symbol is not None:
            grp = row[symbol]

        if grp is None:
            name = ""
        elif isinstance(grp, tuple):
            name = " | ".join(str(g) for g in grp)
        else:
            name = str(grp)

        mk = dict(size=8)
        if color:
            i = list(unique_colors).index(grp if not isinstance(grp, tuple) else grp[0])
            mk["color"] = palette[i % len(palette)]
        else:
            mk["color"] = "rgba(0, 0, 0, 0.5)"

        if symbol:
            j = list(unique_symbols).index(
                grp if not isinstance(grp, tuple) else grp[1]
            )
            mk["symbol"] = symbols[j % len(symbols)]

        mk["opacity"] = 0.8 if not selections or idx in selections else 0.2

        # Hover template
        if hover_data:
            cd = row[hover_data].to_numpy()
            cd = np.hstack([idx, cd])
            hint = (
                "<br>".join([f"{h}: {row[h]}" for h in hover_data]) + "<extra></extra>"
            )

        else:
            cd = idx
            hint = f"id: {idx}"

        fig.add_trace(
            go.Scatter3d(
                x=[row[x]],
                y=[row[y]],
                z=[row[z]],
                mode="markers",
                marker=mk,
                customdata=cd,
                name=name,
                hovertemplate=hint,
                showlegend=True if name != "" and not name in traced_groups else False,
            )
        )

        if name:
            traced_groups.add(name)

    # Final touches
    fig.update_layout(
        title=title,
        template=template,
        width=width,
        height=height,
    )

    return fig
