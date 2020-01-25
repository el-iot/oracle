import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

colors = {
    'background': '#ffffff',
    'text': '#000000'
}



usa_region_best = pd.read_csv('testusa.csv')
usa_region_best = usa_region_best.iloc[:, 1:]

uk_region_best = pd.read_csv('testuk.csv')
uk_region_best = uk_region_best.iloc[:, 1:]

au_region_best = pd.read_csv('testau.csv')
au_region_best = au_region_best.iloc[:, 1:]



# build model
def build_model():
    """  Builds poisson model from existing game data.
         To use multiple game years of data - use the following:
         df1 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1819/E0.csv")
         dflist = [df1, df2]
         df = pd.concat(dflist, ignore_index=True, sort = False)"""

    df = pd.read_csv("e0.csv")

    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
    df = df.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})

    goal_model_data = pd.concat([df[['HomeTeam', 'AwayTeam', 'HomeGoals']].assign(home=1).rename(
        columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'HomeGoals': 'goals'}),
        df[['AwayTeam', 'HomeTeam', 'AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'AwayGoals': 'goals'})])

    poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                            family=sm.families.Poisson()).fit()

    return (poisson_model)




def simulate_match(foot_model, home_team, away_team, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': home_team,
                                                           'opponent': away_team, 'home': 1},
                                                     index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': away_team,
                                                           'opponent': home_team, 'home': 0},
                                                     index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                 [home_goals_avg, away_goals_avg]]
    return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))




centre_style = {'textAlign': 'center',
                'color': colors['text']}

app.title = "Oracle"

app.layout = html.Div([

    html.Div([

        html.H1(children='ORACLE', style={'textAlign': 'center', 'marginBottom': '0em'}),

    ],
        style={'textAlign': 'center', 'width': '100%', 'color': colors['text']}),

    html.Div([

        html.H6(children='PREMIER LEAGUE PREDICTION ENGINE', style={'display': 'inline-block', 'marginBottom': '0em'}),

    ],
        style={'textAlign': 'center', 'width': '100%', 'color': colors['text']}),

    html.Label(['created by ', html.A('Rory(c) 2020', href='https://github.com/rorcores', style={
        'textAlign': 'center', 'width': '100%', 'color': colors['text'], 'margin-top:': 150}),

                ],

               style={'textAlign': 'center', 'width': '100%', 'color': colors['text']}),

    html.Div(children='________________________________________________', style={
        'textAlign': 'center', 'width': '100%', 'color': colors['text']
    }),

    html.Br(),

    html.Div([
        dcc.Markdown('''

    *Oracle* uses a form of [machine learning](https://en.wikipedia.org/wiki/Machine_learning) known as [poisson regression](https://en.wikipedia.org/wiki/Poisson_regression) to predict 
    the outcome of upcoming [Premier League](https://en.wikipedia.org/wiki/Premier_League) football games based upon past event data.

    Bookmaker data adjusts in real time as bets are placed in order to hedge the bookmaker's position on either side of the event. 
    Popular bets where the published odds have become skewed can be identified and ranked by significance using machine learning.

    *Oracle* hunts the internet for bookmaker data in the 3 major betting regions (*USA, UK and Australia*). It compares the current published odds data with the outcome of the *Oracle* machine learning engine and suggests the bets with the highest [expected value](https://en.wikipedia.org/wiki/Expected_value). The strongest bets are those with the highest integer values in the significance column and the most stars in the strength column.

    *Oracle* also provides a more straightforward **head to head** demonstration mode with win/draw probabilities updated in real time.

    *Oracle* is a demonstration of machine learning only and *Oracle* assumes no responsibility for the outcome of any games or bets made as a result of this engine.
    *Oracle* was created by [Rory](https://www.linkedin.com/in/rory-garton-smith-5b991659/). Poisson regression for this style of problem was first suggested in papers by [David Sheehan](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling/) and [Erlandson F. Saraiva](https://www.researchgate.net/publication/305801126_Predicting_football_scores_via_Poisson_regression_model_applications_to_the_National_Football_League).
    ''')],
        style={'width': '100%', 'text-align': 'center', 'font-size': '12px', 'color': colors['text']}),

    html.Br(),

    html.Div([

        html.H4(children='HEAD TO HEAD', style={'display': 'inline-block', 'marginBottom': '0em'}),

    ],
        style={'textAlign': 'center', 'width': '100%', 'color': colors['text']}),

    html.Br(),

    html.Div([

        html.Div([

            html.Div(children='SELECT HOME TEAM:', style={
                'textAlign': 'center', 'color': colors['text']
            }),

            dcc.Dropdown(id="home_team",
                         options=[
                             {'label': 'Arsenal', 'value': 'Arsenal'},
                             {'label': 'Aston Villa', 'value': 'Aston Villa'},
                             {'label': 'Bournemouth', 'value': 'Bournemouth'},
                             {'label': 'Brighton & Hove Albion', 'value': 'Brighton'},
                             {'label': 'Burnley', 'value': 'Burnley'},
                             {'label': 'Chelsea', 'value': 'Chelsea'},
                             {'label': 'Crystal Palace', 'value': 'Crystal Palace'},
                             {'label': 'Everton', 'value': 'Everton'},
                             {'label': 'Leicester City', 'value': 'Leicester'},
                             {'label': 'Liverpool', 'value': 'Liverpool'},
                             {'label': 'Manchester City', 'value': 'Man City'},
                             {'label': 'Manchester United', 'value': 'Man United'},
                             {'label': 'Newcastle United', 'value': 'Newcastle'},
                             {'label': 'Norwich City', 'value': 'Norwich City'},
                             {'label': 'Sheffield United', 'value': 'Sheffield United'},
                             {'label': 'Southampton', 'value': 'Southampton'},
                             {'label': 'Tottenham Hotspur', 'value': 'Tottenham'},
                             {'label': 'Watford', 'value': 'Watford'},
                             {'label': 'West Ham United', 'value': 'West Ham'},
                             {'label': 'Wolverhampton Wanderers', 'value': 'Wolves'},
                         ]
                         )
        ],
            style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'}),

        html.Div([

            html.Div(children='SELECT AWAY TEAM:', style={
                'textAlign': 'center', 'color': colors['text']
            }),

            dcc.Dropdown(id="away_team",
                         options=[
                             {'label': 'Arsenal', 'value': 'Arsenal'},
                             {'label': 'Aston Villa', 'value': 'Aston Villa'},
                             {'label': 'Bournemouth', 'value': 'Bournemouth'},
                             {'label': 'Brighton & Hove Albion', 'value': 'Brighton'},
                             {'label': 'Burnley', 'value': 'Burnley'},
                             {'label': 'Chelsea', 'value': 'Chelsea'},
                             {'label': 'Crystal Palace', 'value': 'Crystal Palace'},
                             {'label': 'Everton', 'value': 'Everton'},
                             {'label': 'Leicester City', 'value': 'Leicester'},
                             {'label': 'Liverpool', 'value': 'Liverpool'},
                             {'label': 'Manchester City', 'value': 'Man City'},
                             {'label': 'Manchester United', 'value': 'Man United'},
                             {'label': 'Newcastle United', 'value': 'Newcastle'},
                             {'label': 'Norwich City', 'value': 'Norwich City'},
                             {'label': 'Sheffield United', 'value': 'Sheffield United'},
                             {'label': 'Southampton', 'value': 'Southampton'},
                             {'label': 'Tottenham Hotspur', 'value': 'Tottenham'},
                             {'label': 'Watford', 'value': 'Watford'},
                             {'label': 'West Ham United', 'value': 'West Ham'},
                             {'label': 'Wolverhampton Wanderers', 'value': 'Wolves'},
                         ]
                         )

        ], style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'})
    ], style={'width': '75%', 'margin-left': 'auto', 'margin-right': 'auto'}),

    html.Br(),

    html.Div(id="result_output_home", children='x', style={
        'textAlign': 'center', 'width': '100%', 'color': colors['text'], 'margin-top:': 150
    }),

    html.Div(children='________________________________________________', style={
        'textAlign': 'center', 'width': '100%', 'color': colors['text']
    }),

    html.Br(),

    html.Br(),

    html.Br(),

    html.Div([

        html.H4(children='BEST UPCOMING BETS USA', style={'display': 'inline-block', 'marginBottom': '0em'}),

    ],
        style={'textAlign': 'center', 'width': '100%', 'color': colors['text']}),

    html.Div(children='', style={
        'textAlign': 'center', 'color': colors['text']
    }),

    dash_table.DataTable(
        id='usa_results',
        columns=[{"name": i, "id": i} for i in usa_region_best.columns],
        data=usa_region_best.to_dict("rows"),
        style_cell={
            'font_size': '10px',
            'text_align': 'center',
            'textOverflow': 'ellipsis'
        },
    ),

    html.Br(),

    html.Br(),

    html.Div([

        html.H4(children='BEST UPCOMING BETS UK', style={'display': 'inline-block', 'marginBottom': '0em'}),

    ],
        style={'textAlign': 'center', 'width': '100%', 'color': colors['text']}),

    html.Div(children='', style={
        'textAlign': 'center', 'color': colors['text']
    }),

    dash_table.DataTable(
        id='uk_results',
        columns=[{"name": i, "id": i} for i in uk_region_best.columns],
        data=uk_region_best.to_dict("rows"),
        style_cell={
            'font_size': '10px',
            'text_align': 'center',
            'textOverflow': 'ellipsis'
        },
    ),

    html.Br(),

    html.Br(),

    html.Div([

        html.H4(children='BEST UPCOMING BETS AUS', style={'display': 'inline-block', 'marginBottom': '0em'}),

    ],
        style={'textAlign': 'center', 'width': '100%', 'color': colors['text']}),

    html.Div(children='', style={
        'textAlign': 'center', 'color': colors['text']
    }),

    dash_table.DataTable(
        id='au_results',
        columns=[{"name": i, "id": i} for i in au_region_best.columns],
        data=au_region_best.to_dict("rows"),
        style_cell={
            'font_size': '10px',
            'text_align': 'center',
            'textOverflow': 'ellipsis'
        },
    ),

    html.Br(),

    html.Br(),

    html.Label(['Oracle. Created by ', html.A('Rory(c) 2020', href='https://github.com/rorcores', style={
        'textAlign': 'center', 'width': '100%', 'color': colors['text'], 'margin-top:': 150}),

                ],

               style={'textAlign': 'center', 'width': '100%', 'color': colors['text']})

])


@app.callback(
    Output(component_id='result_output_home', component_property='children'),
    [Input(component_id='home_team', component_property='value'),
     Input(component_id='away_team', component_property='value')])
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

        home_prob = (home_prob + "%")
        draw_prob = (draw_prob + "%")
        away_prob = (away_prob + "%")

        home_team_string = "Probability of ", home_team, " winning:"

        away_team_string = "Probability of ", away_team, " winning:"

        full_result = (home_team_string, home_prob, away_team_string, away_prob, "\nProbability of draw:", draw_prob)

        if home_team == away_team:
            return html.Div([html.Label("Teams cannot play against themselves, please check your selection")])

        return html.Div([html.Label(string) for string in full_result])


if __name__ == '__main__':

    poisson_model = build_model()

    app.run_server()
