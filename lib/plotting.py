"""Plotting module for dash app."""

from typing import Sequence, Union

import numpy as np
import plotly.graph_objects as go
import polars as pl


def scatter_2d(
    data: Union[dict, pl.DataFrame],
    x: str,
    y: str,
    width: int = 1080,
    height: int = 1080,
    color: str = None,
    symbol: str = None,
    color_discrete_sequence: Sequence[str] = None,
    hover_data: Sequence[str] = None,
    title: str = None,
    template: str = None,
) -> go.Figure:
    if isinstance(data, pl.DataFrame):
        df = data.to_pandas()
    else:
        import pandas as pd

        df = pd.DataFrame(data)

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
    palette = color_discrete_sequence or ["#636efa", "#EF553B", "#00cc96", "#ab63fa"]
    symbols = ["circle", "square", "diamond", "cross", "triangle-up", "x"]
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
            hint = None

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

    # Final touches
    fig.update_layout(
        title=title,
        template=template,
        width=width,
        height=height,
    )

    return fig
