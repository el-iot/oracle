from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import requests


class Oracle:
    def __init__(self, region):
        self.api_key = "insert_api_key" 

        self.team_dict = {'Brighton and Hove Albion': 'Brighton',
                          'Newcastle United': 'Newcastle',
                          'Leicester City': 'Leicester',
                          'Manchester City': 'Man City',
                          'Manchester United': 'Man United',
                          'Norwich City': 'Norwich',
                          'Tottenham Hotspur': 'Tottenham',
                          'West Ham United': 'West Ham',
                          'Wolverhampton Wanderers': 'Wolves'}

        self.fixtures_list = []

        self.region = region

        self.poisson_model = ""

    @staticmethod
    def _strength_rater(prob, significance):
        strength = ""
        if prob > 50:
            strength += "*"
        if significance > 10:
            strength += "*"
        if significance > 20:
            strength += "*"
        return strength

    def build_model(self):
        """  Builds poisson model from existing game data.
             To use multiple game years of data - use the following:
             df1 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1819/E0.csv")
             dflist = [df1, df2]
             df = pd.concat(dflist, ignore_index=True, sort = False)"""

        df = pd.read_csv("http://www.football-data.co.uk/mmz4281/1920/E0.csv")

        df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
        df = df.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})

        goal_model_data = pd.concat([df[['HomeTeam', 'AwayTeam', 'HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'HomeGoals': 'goals'}),
            df[['AwayTeam', 'HomeTeam', 'AwayGoals']].assign(home=0).rename(
                columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'AwayGoals': 'goals'})])

        poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                                family=sm.families.Poisson()).fit()

        self.poisson_model = poisson_model

    @staticmethod
    def _simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
        home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam,
                                                               'opponent': awayTeam, 'home': 1},
                                                         index=[1])).values[0]
        away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam,
                                                               'opponent': homeTeam, 'home': 0},
                                                         index=[1])).values[0]
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                     [home_goals_avg, away_goals_avg]]
        return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

    def get_odds(self, api_verbose=True):

        odds_response = requests.get('https://api.the-odds-api.com/v3/odds', params={
            'api_key': self.api_key,
            'sport': 'soccer_epl',
            'region': self.region,  # uk | us | au are the options
            'mkt': 'h2h'
        })

        if api_verbose:
            print('Remaining requests', odds_response.headers['x-requests-remaining'])
            print('Used requests', odds_response.headers['x-requests-used'])

        return json.loads(odds_response.text)

    def build_fixture_lists(self):

        odds_json = self.get_odds()

        dfcols = ["home_team", "away_team", "kick_off_time", "bookmaker", "home_win_prob", 'draw_prob', 'away_win_prob',
                  "oracle_home_win_prob", 'oracle_draw_prob', 'oracle_away_prob']

        for i in odds_json['data']:
            df = pd.DataFrame(columns=dfcols)

            home_team = i['home_team']
            if home_team == i['teams'][0]:
                away_team = i['teams'][1]
            else:
                away_team = i['teams'][0]

            assert home_team != away_team
            kick_off_time = datetime.fromtimestamp(i['commence_time'])

            # Quick Name Conversions for model
            for k, v in self.team_dict.items():
                home_team = home_team.replace(k, v)

            for k, v in self.team_dict.items():
                away_team = away_team.replace(k, v)

            # Calculate Oracle Odds
            game_result = self._simulate_match(self.poisson_model, home_team, away_team, max_goals=10)

            oracle_home_prob = np.sum(np.tril(game_result, -1)) * 100
            oracle_draw_prob = np.sum(np.diag(game_result)) * 100
            oracle_away_prob = np.sum(np.triu(game_result, 1)) * 100

            # Grab bookie odds
            for site in i['sites']:

                bookie = site['site_nice']

                for index, team in enumerate(i['teams']):

                    for k, v in self.team_dict.items():
                        i['teams'][index] = i['teams'][index].replace(k, v)

                if home_team == i['teams'][0]:
                    home_win_prob = (1 / site['odds']['h2h'][0]) * 100
                    draw_prob = (1 / site['odds']['h2h'][2]) * 100
                    away_win_prob = (1 / site['odds']['h2h'][1]) * 100

                if home_team == i['teams'][1]:
                    home_win_prob = (1 / site['odds']['h2h'][1]) * 100
                    draw_prob = (1 / site['odds']['h2h'][2]) * 100
                    away_win_prob = (1 / site['odds']['h2h'][0]) * 100

                coolList = [
                    [home_team, away_team, kick_off_time, bookie, home_win_prob, draw_prob, away_win_prob,
                     oracle_home_prob,
                     oracle_draw_prob, oracle_away_prob]]

                df = df.append(pd.DataFrame(coolList, columns=dfcols), ignore_index=True)
                df = df.round(2)

                df['homeWinBetOpportunity'] = (((df['home_win_prob'] + 5) < df['oracle_home_win_prob']) * 1).astype(str)

                df['drawBetOpportunity'] = (((df['draw_prob'] + 5) < df['oracle_draw_prob']) * 1).astype(str)

                df['awayWinBetOpportunity'] = (((df['away_win_prob'] + 5) < df['oracle_away_prob']) * 1).astype(str)

                df = df[['home_team',
                         'away_team',
                         'kick_off_time',
                         'bookmaker',
                         'home_win_prob',
                         'draw_prob',
                         'away_win_prob',
                         'oracle_home_win_prob',
                         'oracle_draw_prob',
                         'oracle_away_prob',
                         'homeWinBetOpportunity',
                         'drawBetOpportunity',
                         'awayWinBetOpportunity']]

            self.fixtures_list.append(df)

    def build_significant_bets_table(self):
        significance_df_cols = ["Home Team", "Away Team", "Game Time", "Bookmaker", "Suggested Bet", "Bookie Probability",
                       "Oracle Probability"]

        significance_df = pd.DataFrame(columns=significance_df_cols)

        # Find best opportunity for each game
        for i in self.fixtures_list:

            # Get Rid of betfair - it messes everything up
            index_names = i[i['bookmaker'] == "Betfair"].index
            i.drop(index_names, inplace=True)

            if int(i["awayWinBetOpportunity"].sum()) >= 1:
                tempdf = i.sort_values('away_win_prob', ascending=True).head(1)
                for index, row in tempdf.iterrows():
                    append_list = [
                        [row.home_team, row.away_team, row.kick_off_time, row.bookmaker, "Away Win", row.away_win_prob,
                         row.oracle_away_prob]]
                significance_df = significance_df.append(pd.DataFrame(append_list, columns=significance_df_cols), ignore_index=True)

            if int(i["drawBetOpportunity"].sum()) >= 1:
                tempdf = i.sort_values('draw_prob', ascending=True).head(1)
                for index, row in tempdf.iterrows():
                    append_list = [
                        [row.home_team, row.away_team, row.kick_off_time, row.bookmaker, "Draw", row.draw_prob,
                         row.oracle_draw_prob]]
                significance_df = significance_df.append(pd.DataFrame(append_list, columns=significance_df_cols), ignore_index=True)

            if int(i["homeWinBetOpportunity"].sum()) >= 1:
                tempdf = i.sort_values('home_win_prob', ascending=True).head(1)
                for index, row in tempdf.iterrows():
                    append_list = [
                        [row.home_team, row.away_team, row.kick_off_time, row.bookmaker, "Home Win", row.home_win_prob,
                         row.oracle_home_win_prob]]
                significance_df = significance_df.append(pd.DataFrame(append_list, columns=significance_df_cols), ignore_index=True)

        significance_df['Significance'] = abs(significance_df["Bookie Probability"] - significance_df["Oracle Probability"])

        significance_df = significance_df.sort_values('Game Time', ascending=True)

        significance_df = significance_df[significance_df.Significance > 10]

        significance_df["Strength"] = significance_df.apply(lambda x: self._strength_rater(x['Oracle Probability'], x['Significance']),
                                            axis=1)

        significance_df['Significance'] = significance_df['Significance'].astype(int)

        return significance_df


if __name__ == '__main__':
    print('Quitting')
    pass
