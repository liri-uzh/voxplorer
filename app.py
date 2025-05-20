import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

# Create the dash app
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
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
                dbc.NavItem(
                    dbc.NavLink(
                        "Recogniser",
                        href="/recogniser",
                        id="recogniser-link",
                        active="partial",
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
    ]
)


# def print_layout(component, indent=0):
#     space = "\t" * indent
#     comp_type = type(component).__name__
#     comp_id = getattr(component, "id", None)
#     print(f"{space}{comp_type}: {comp_id}")
#
#     children = getattr(component, "children", None)
#     if children:
#         if isinstance(children, list):
#             for child in children:
#                 print_layout(child, indent + 1)
#         else:
#             print_layout(children, indent + 1)


if __name__ == "__main__":
    app.run_server(
        debug=True,
    )
