import dash
from dash import html
import dash_bootstrap_components as dbc

##############################################
# Load data                                  #
# link Markdown documentation (brief) to use #
# LiRI — DORA                                #
##############################################

# init
dash.register_page(__name__, path="/")

# --- Define page layout ---
layout = dbc.Container(
    [
        # Page title
        dbc.Row(
            dbc.Col(
                html.H1("Welcome to LiRI — voxplorer"), className="text-center my-4"
            ),
        ),
        # Buttons for mode selection
        dbc.Row(
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.H4(
                            "Choose the mode in which you would like to use voxplorer"
                        )
                    ),
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Go to Visualiser",
                                        id="go-to-visualiser-btn",
                                        href="/visualiser",
                                        color="primary",
                                        style={
                                            "margin-top": "10px",
                                            "display": "block",
                                        },
                                    ),
                                    width=6,
                                ),
                                # dbc.Col(
                                #     dbc.Button(
                                #         "Go to Recogniser",
                                #         id="go-to-recogniser-btn",
                                #         href="/recogniser",
                                #         color="primary",
                                #         style={
                                #             "margin-top": "10px",
                                #             "display": "block",
                                #         },
                                #     ),
                                #     width=6,
                                # ),
                                # NOTE: placeholder
                                dbc.Col(
                                    dbc.Alert(
                                        "Recogniser is coming soon",
                                        color="info",
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                    ),
                ]
            )
        ),
    ],
    fluid=True,
)
# TODO: add some documentation and CITATION REFERENCE at the bottom of this page
