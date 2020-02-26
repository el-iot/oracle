import numpy as np
import pandas as pd
from scipy.stats import poisson

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dash.dependencies import Input, Output

external_stylesheets = ["style.css"]
colors = {"background": "#ffffff", "text": "#000000"}

betting_odds = {key: pd.read_csv(f"data/test{key}.csv").iloc[:, 1:] for key in ["usa", "uk", "au"]}

TITLE = "Oracle20"

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = TITLE

with open("description.txt", "r") as file:
    description = file.read()


def generate_dashtable(country_code):
    """
    Generate dash-table for app-layout
    """
    return html.Div(
        [
            dash_table.DataTable(
                id=f"{country_code}_results",
                columns=[{"name": i, "id": i} for i in betting_odds[country_code]],
                data=betting_odds[country_code].to_dict("rows"),
                style_cell={
                    "font_size": "10px",
                    "text_align": "center",
                    "textOverflow": "ellipsis",
                },
            )
        ],
        className="betting-odds-table-container",
    )


def build_model():
    """
    Builds poisson model from existing game data.
    To use multiple game years of data - use the following:
    df1 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1819/E0.csv")
    dflist = [df1, df2]
    df = pd.concat(dflist, ignore_index=True, sort = False)
    """

    columns = {
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
    }

    df = pd.read_csv("http://www.football-data.co.uk/mmz4281/1920/E0.csv", usecols=[*columns])
    df = df.rename(columns=columns)

    goal_model_data = pd.concat(
        [
            df[["home_team", "away_team", "home_goals"]]
            .assign(home=1)
            .rename(columns={"home_team": "team", "away_team": "opponent", "home_goals": "goals"}),
            df[["away_team", "home_team", "away_goals"]]
            .assign(home=0)
            .rename(columns={"away_team": "team", "home_team": "opponent", "away_goals": "goals"}),
        ]
    )

    poisson_model = smf.glm(
        formula="goals ~ home + team + opponent", data=goal_model_data, family=sm.families.Poisson()
    ).fit()

    return poisson_model


def _predict(model, home_team, away_team, home):
    """
    Helper method for simulate_match
    """

    return model.predict(
        pd.DataFrame(
            data={
                "team": home_team if home else away_team,
                "opponent": away_team if home else home_team,
                "home": home,
            },
            index=[1],
        )
    ).values[0]


def simulate_match(model, home_team, away_team, max_goals=10):
    """
    Simulate a match between two teams
    """

    home_goals_avg = _predict(model, home_team, away_team, 1)
    away_goals_avg = _predict(model, home_team, away_team, 0)

    team_pred = [
        [poisson.pmf(i, team_avg) for i in range(max_goals + 1)]
        for team_avg in [home_goals_avg, away_goals_avg]
    ]

    return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))


centre_style = {"textAlign": "center", "color": colors["text"]}

app.title = "Oracle"

app.layout = html.Div(
    [
        html.Div([html.Img(src="static/logo.png")], style={"textAlign": "center"}),
        html.Div(
            [html.H3(children="ORACLE", style={"textAlign": "center", "marginBottom": "0em"})],
            style={"textAlign": "center", "width": "100%", "color": colors["text"]},
        ),
        html.Div(
            [
                html.H6(
                    children="FOOTBALL MACHINE LEARNING ENGINE",
                    style={"display": "inline-block", "marginBottom": "0em"},
                ),
            ],
            style={"textAlign": "center", "width": "100%", "color": colors["text"]},
        ),
        html.Label(
            [
                "created by ",
                html.A(
                    "Rory(c) 2020",
                    href="https://github.com/rorcores",
                    style={
                        "textAlign": "center",
                        "width": "100%",
                        "color": colors["text"],
                        "margin-top:": 150,
                    },
                ),
            ],
            style={"textAlign": "center", "width": "100%", "color": colors["text"]},
        ),
        html.Div(
            children="________________________________________________",
            style={"textAlign": "center", "width": "100%", "color": colors["text"]},
        ),
        html.Br(),
        html.Div(
            [dcc.Markdown(description)],
            style={
                "width": "100%",
                "text-align": "center",
                "font-size": "12px",
                "color": colors["text"],
            },
        ),
        html.Br(),
        html.Div(
            [
                html.H4(
                    children="HEAD TO HEAD",
                    style={"display": "inline-block", "marginBottom": "0em"},
                ),
            ],
            style={"textAlign": "center", "width": "100%", "color": colors["text"]},
        ),
        html.Br(),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            children="SELECT HOME TEAM:",
                            style={"textAlign": "center", "color": colors["text"]},
                        ),
                        dcc.Dropdown(
                            id="home_team",
                            options=[
                                {"label": "Arsenal", "value": "Arsenal"},
                                {"label": "Aston Villa", "value": "Aston Villa"},
                                {"label": "Bournemouth", "value": "Bournemouth"},
                                {"label": "Brighton & Hove Albion", "value": "Brighton"},
                                {"label": "Burnley", "value": "Burnley"},
                                {"label": "Chelsea", "value": "Chelsea"},
                                {"label": "Crystal Palace", "value": "Crystal Palace"},
                                {"label": "Everton", "value": "Everton"},
                                {"label": "Leicester City", "value": "Leicester"},
                                {"label": "Liverpool", "value": "Liverpool"},
                                {"label": "Manchester City", "value": "Man City"},
                                {"label": "Manchester United", "value": "Man United"},
                                {"label": "Newcastle United", "value": "Newcastle"},
                                {"label": "Norwich City", "value": "Norwich City"},
                                {"label": "Sheffield United", "value": "Sheffield United"},
                                {"label": "Southampton", "value": "Southampton"},
                                {"label": "Tottenham Hotspur", "value": "Tottenham"},
                                {"label": "Watford", "value": "Watford"},
                                {"label": "West Ham United", "value": "West Ham"},
                                {"label": "Wolverhampton Wanderers", "value": "Wolves"},
                            ],
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block", "text-align": "center"},
                ),
                html.Div(
                    [
                        html.Div(
                            children="SELECT AWAY TEAM:",
                            style={"textAlign": "center", "color": colors["text"]},
                        ),
                        dcc.Dropdown(
                            id="away_team",
                            options=[
                                {"label": "Arsenal", "value": "Arsenal"},
                                {"label": "Aston Villa", "value": "Aston Villa"},
                                {"label": "Bournemouth", "value": "Bournemouth"},
                                {"label": "Brighton & Hove Albion", "value": "Brighton"},
                                {"label": "Burnley", "value": "Burnley"},
                                {"label": "Chelsea", "value": "Chelsea"},
                                {"label": "Crystal Palace", "value": "Crystal Palace"},
                                {"label": "Everton", "value": "Everton"},
                                {"label": "Leicester City", "value": "Leicester"},
                                {"label": "Liverpool", "value": "Liverpool"},
                                {"label": "Manchester City", "value": "Man City"},
                                {"label": "Manchester United", "value": "Man United"},
                                {"label": "Newcastle United", "value": "Newcastle"},
                                {"label": "Norwich City", "value": "Norwich City"},
                                {"label": "Sheffield United", "value": "Sheffield United"},
                                {"label": "Southampton", "value": "Southampton"},
                                {"label": "Tottenham Hotspur", "value": "Tottenham"},
                                {"label": "Watford", "value": "Watford"},
                                {"label": "West Ham United", "value": "West Ham"},
                                {"label": "Wolverhampton Wanderers", "value": "Wolves"},
                            ],
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block", "text-align": "center"},
                ),
            ],
            style={"width": "75%", "margin-left": "auto", "margin-right": "auto"},
        ),
        html.Br(),
        html.Div(
            id="result_output_home",
            children="x",
            style={
                "textAlign": "center",
                "width": "100%",
                "color": colors["text"],
                "margin-top:": 150,
            },
        ),
        html.Div(
            children="________________________________________________",
            style={"textAlign": "center", "width": "100%", "color": colors["text"]},
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(
            [
                html.H4(
                    children="BEST UPCOMING BETS USA",
                    style={"display": "inline-block", "marginBottom": "0em"},
                ),
            ],
            style={"textAlign": "center", "width": "100%", "color": colors["text"]},
        ),
        html.Div(children="", style={"textAlign": "center", "color": colors["text"]}),
        generate_dashtable("usa"),
        html.Br(),
        html.Br(),
        html.Div(
            [
                html.H4(
                    children="BEST UPCOMING BETS UK",
                    style={"display": "inline-block", "marginBottom": "0em"},
                ),
            ],
            style={"textAlign": "center", "width": "100%", "color": colors["text"]},
        ),
        html.Div(children="", style={"textAlign": "center", "color": colors["text"]}),
        generate_dashtable("uk"),
        html.Br(),
        html.Br(),
        html.Div(
            [
                html.H4(
                    children="BEST UPCOMING BETS AUS",
                    style={"display": "inline-block", "marginBottom": "0em"},
                ),
            ],
            style={"textAlign": "center", "width": "100%", "color": colors["text"]},
        ),
        html.Div(children="", style={"textAlign": "center", "color": colors["text"]}),
        generate_dashtable("au"),
        html.Br(),
        html.Br(),
        html.Label(
            [
                "Oracle. Created by ",
                html.A(
                    "Rory(c) 2020",
                    href="https://github.com/rorcores",
                    style={
                        "textAlign": "center",
                        "width": "100%",
                        "color": colors["text"],
                        "margin-top:": 150,
                    },
                ),
            ],
            style={"textAlign": "center", "width": "100%", "color": colors["text"]},
        ),
    ]
)


@app.callback(
    Output(component_id="result_output_home", component_property="children"),
    [
        Input(component_id="home_team", component_property="value"),
        Input(component_id="away_team", component_property="value"),
    ],
)
def crunchOdds(home_team, away_team):
    poisson_model = build_model()

    if home_team is None:
        return ""

    if away_team is None:
        return ""

    if home_team and away_team is not None:

        game_result = simulate_match(poisson_model, home_team, away_team, max_goals=10)

        home_prob = np.sum(np.tril(game_result, -1)) * 100
        draw_prob = np.sum(np.diag(game_result)) * 100
        away_prob = np.sum(np.triu(game_result, 1)) * 100

        home_prob = str(round(home_prob, 2))
        draw_prob = str(round(draw_prob, 2))
        away_prob = str(round(away_prob, 2))

        home_prob = home_prob + "%"
        draw_prob = draw_prob + "%"
        away_prob = away_prob + "%"

        home_team_string = "Probability of ", home_team, " winning:"

        away_team_string = "Probability of ", away_team, " winning:"

        full_result = (
            home_team_string,
            home_prob,
            away_team_string,
            away_prob,
            "\nProbability of draw:",
            draw_prob,
        )

        if home_team == away_team:
            return html.Div(
                [html.Label("Teams cannot play against themselves, please check your selection")]
            )

        return html.Div([html.H6(string) for string in full_result])


if __name__ == "__main__":

    poisson_model = build_model()
    app.run_server(8080)
