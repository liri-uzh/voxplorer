import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Local
from lib.data_loader import parse_table_contents
from pages.layouts.visualiser.table_upload import metaconfig


# --- Setup table-upload storage ---
storage_components = html.Div(
    [
        # table storage
        dcc.Store(id="stored-data-table", storage_type="memory"),
        # metadata variables
        dcc.Store(id="stored-metainformation-table", storage_type="memory"),
    ]
)

# --- Upload ---
upload_component = html.Div(
    [
        dbc.Card(
            [
                dbc.CardHeader(html.H4("Upload a table (.csv, .tsv, .xlxs)")),
                dbc.CardBody(
                    [
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
                        ),
                        html.Div(id="table-output", className="mt-3"),
                    ]
                ),
            ]
        ),
    ],
)

# --- Main layout ---
layout = [
    dbc.Row(
        [
            storage_components,
            upload_component,
            metaconfig.layout,
        ]
    )
]


# --- Callback 1: table upload ---
@callback(
    [
        Output("table-output", "children"),
        Output("stored-data-table", "data"),
    ],
    [
        Input("upload-table", "contents"),
    ],
    [
        State("upload-table", "filename"),
    ],
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

        # Manage returns
        if data_table is not None:
            return (
                dbc.Alert(
                    f"{filename} uploaded successfully",
                    color="success",
                    dismissable=True,
                ),
                data_table,
            )

        else:
            return (
                alert,
                None,
            )

    return (
        [],
        None,
    )


# --- Callback 2: update store components when data is uploaded ---
@callback(
    [
        # TODO: might have to add allow_duplicate=True
        Output("stored-table", "data"),
        Output("stored-metainformation", "data"),
        Output("stored-data-table", "clear_data"),
        Output("stored-metainformation-table", "clear_data"),
    ],
    [
        Input("confirmed-selection-btn", "n_clicks"),
    ],
    [
        State("stored-data-table", "data"),
        State("stored-metainformation-table", "data"),
    ],
    prevent_initial_call=True,
)
def promote_and_clear_temp_store(
    n_clicks_table,
    data_table,
    metainformation_table,
):
    if data_table is None and metainformation_table is None:
        raise PreventUpdate

    new_table = data_table or dash.no_update
    new_meta = metainformation_table or dash.no_update

    return new_table, new_meta, True, True
