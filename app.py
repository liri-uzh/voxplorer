import os
import signal
import dash
from dash import dcc, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Create the dash app
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    prevent_initial_callbacks=True,
)

# Navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.Nav(
            [
                dbc.NavItem(
                    dbc.NavLink(
                        "Visualiser",
                        href="/visualiser",
                        id="visualiser-link",
                        active="partial",
                    )
                ),
                # TODO: Uncomment this when Recogniser is ready
                # dbc.NavItem(
                #     dbc.NavLink(
                #         "Recogniser",
                #         href="/recogniser",
                #         id="recogniser-link",
                #         active="partial",
                #     )
                # ),
                dbc.NavItem(
                    dbc.NavLink(
                        "Exit",
                        id="exit-link",
                        href="#",
                        active="partial",
                        className="text-danger",
                    )
                ),
            ],
            pills=True,
            navbar=True,
        )
    ],
    brand="LiRI — voxplorer",
    brand_href="/",
    color="dark",
    dark=True,
)

# Overall layout
app.layout = dbc.Container(
    [
        # navigation bar
        navbar,
        dash.page_container,
        dcc.ConfirmDialog(
            id="confirm-exit",
            message="Are you sure you want to exit?",
        ),
    ],
    id="app-container",
)


@app.callback(
    Output("confirm-exit", "displayed"),
    Input("exit-link", "n_clicks"),
)
def display_confirm_dialog(n_clicks):
    if n_clicks:
        return True
    return False


@app.callback(
    [
        Output("exit-link", "children"),
        Output("app-container", "children"),
    ],
    [
        Input("confirm-exit", "submit_n_clicks"),
        Input("confirm-exit", "cancel_n_clicks"),
    ],
)
def handle_exit(submit_clicks, cancel_clicks):
    if submit_clicks:
        os.kill(os.getpid(), signal.SIGINT)
        return (
            "Exit",
            [
                dbc.Alert(
                    "App not running!",
                    color="danger",
                )
            ],
        )
    raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)
