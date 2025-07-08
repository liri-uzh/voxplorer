from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Local
from lib.data_loader import parse_table_contents
from pages.layouts.visualiser.table_upload import metaconfig


def upload_table_component():
    return [
        dcc.Upload(
            id="upload-table",
            children=html.Div(
                [
                    "Drag and drop or ",
                    html.A("Select a file"),
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
            multiple=False,
        )
    ]


# --- Upload ---
upload_component = html.Div(
    [
        dbc.Card(
            [
                dbc.CardHeader(html.H4("Upload a table (.csv, .tsv, .xlxs)")),
                dbc.CardBody(
                    [
                        html.Div(
                            id="upload-table-component",
                            children=upload_table_component(),
                        ),
                        html.Div(id="table-output", className="mt-3"),
                    ]
                ),
            ]
        ),
    ],
)

# --- Main layout ---
layout = dbc.Row(
    [
        upload_component,
        metaconfig.layout,
    ],
    id="upload-table-layout",
    style={"display": "none"},
)


# --- Callback 1: table upload ---
@callback(
    [
        Output("table-output", "children"),
        Output("stored-data-table", "data"),
        Output("upload-table-component", "children", allow_duplicate=True),
    ],
    [
        Input("upload-table", "contents"),
    ],
    [
        State("upload-table", "filename"),
    ],
    prevent_initial_call=True,
)
def upload_table(
    contents,
    filename,
):
    # Parse the file
    if contents is not None and filename is not None:
        data_table, alert = parse_table_contents(
            contents,
            filename,
        )

        return (
            alert,
            data_table,
            upload_table_component(),
        )

    else:
        raise PreventUpdate
