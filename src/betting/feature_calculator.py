"""
Feature calculator for live betting predictions
Calculates rolling features from recent goalie games using boxscore data
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path


class BettingFeatureCalculator:
    """Calculate features for betting predictions"""

    def __init__(self):
        # Load feature names to ensure correct order
        # Use the tuned model's feature names (114 features including engineered)
        feature_file = Path('models/trained/tuned_v1_20260201_155204/classifier_feature_names.json')
        if not feature_file.exists():
            # Fallback to generic location
            feature_file = Path('models/classifier_feature_names.json')
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = None

    def _compute_rolling(self, values, windows):
        """
        Compute rolling mean and std for multiple windows.
        Values should be in most-recent-first order.

        Returns dict of feature_name -> value for each window.
        """
        results = {}
        for window in windows:
            if len(values) >= window:
                subset = values[:window]
            elif len(values) > 0:
                subset = values
            else:
                subset = None

            if subset is not None:
                results[f'mean_{window}'] = float(np.mean(subset))
                results[f'std_{window}'] = float(np.std(subset)) if len(subset) > 1 else 0.0
            else:
                results[f'mean_{window}'] = 0.0
                results[f'std_{window}'] = 0.0

        return results

    def calculate_goalie_features(self, recent_games, game_date, nhl_fetcher=None, goalie_id=None):
        """
        Calculate goalie rolling features from recent games.

        Uses game log for basic stats (saves, shots_against, goals_against, save_pct)
        and boxscores for situation-specific stats (even_strength, power_play, short_handed).

        Args:
            recent_games: List of recent game dicts from NHL API (most recent first)
            game_date: Date of prediction (to calculate rest days)
            nhl_fetcher: NHLBettingData instance for fetching boxscores
            goalie_id: NHL goalie ID (needed for boxscore lookups)

        Returns:
            dict: Feature name -> value mapping
        """
        features = {}

        if not recent_games:
            return self._get_default_features()

        # Filter out games from the current date to prevent data leakage
        filtered_games = [g for g in recent_games if g.get('gameDate', '') != game_date]

        if not filtered_games:
            return self._get_default_features()

        windows = [3, 5, 10]

        # --- Basic stats from game log ---
        # Game log fields: shotsAgainst, goalsAgainst, savePctg (NO saves field)
        saves_values = []
        shots_against_values = []
        goals_against_values = []
        save_pct_values = []

        for g in filtered_games:
            sa = g.get('shotsAgainst', 0)
            ga = g.get('goalsAgainst', 0)
            saves_values.append(sa - ga)  # saves = shotsAgainst - goalsAgainst
            shots_against_values.append(sa)
            goals_against_values.append(ga)
            save_pct_values.append(g.get('savePctg', 0.0))

        basic_stats = {
            'saves': saves_values,
            'shots_against': shots_against_values,
            'goals_against': goals_against_values,
            'save_percentage': save_pct_values,
        }

        for stat_name, values in basic_stats.items():
            rolling = self._compute_rolling(values, windows)
            for window in windows:
                features[f'{stat_name}_rolling_{window}'] = rolling[f'mean_{window}']
                features[f'{stat_name}_rolling_std_{window}'] = rolling[f'std_{window}']

        # --- Situation-specific stats from boxscores ---
        situation_stats = {
            'even_strength_saves': [],
            'even_strength_shots_against': [],
            'even_strength_goals_against': [],
            'power_play_saves': [],
            'power_play_shots_against': [],
            'power_play_goals_against': [],
            'short_handed_saves': [],
            'short_handed_shots_against': [],
            'short_handed_goals_against': [],
        }

        if nhl_fetcher and goalie_id:
            for g in filtered_games:
                gid = g.get('gameId')
                if not gid:
                    # Append zeros if no game ID
                    for key in situation_stats:
                        situation_stats[key].append(0)
                    continue

                box_stats = nhl_fetcher.get_goalie_boxscore_stats(gid, goalie_id)
                if box_stats:
                    for key in situation_stats:
                        situation_stats[key].append(box_stats.get(key, 0))
                else:
                    for key in situation_stats:
                        situation_stats[key].append(0)
        else:
            # No fetcher available - use zeros (will match default behavior)
            for g in filtered_games:
                for key in situation_stats:
                    situation_stats[key].append(0)

        for stat_name, values in situation_stats.items():
            rolling = self._compute_rolling(values, windows)
            for window in windows:
                features[f'{stat_name}_rolling_{window}'] = rolling[f'mean_{window}']
                features[f'{stat_name}_rolling_std_{window}'] = rolling[f'std_{window}']

        # --- Rest days ---
        if filtered_games:
            last_game_date = filtered_games[0].get('gameDate', '')
            if last_game_date:
                last_date = datetime.strptime(last_game_date, '%Y-%m-%d')
                current_date = datetime.strptime(game_date, '%Y-%m-%d')
                features['goalie_days_rest'] = (current_date - last_date).days
            else:
                features['goalie_days_rest'] = 3
            features['goalie_is_back_to_back'] = 1 if features.get('goalie_days_rest', 3) == 1 else 0
        else:
            features['goalie_days_rest'] = 3
            features['goalie_is_back_to_back'] = 0

        return features

    def calculate_team_features(self, team, opponent, game_date, nhl_fetcher=None, recent_games=None):
        """
        Calculate team defensive and opponent offensive rolling features.

        Args:
            team: Team abbreviation
            opponent: Opponent team abbreviation
            game_date: Current game date string
            nhl_fetcher: NHLBettingData instance for fetching data
            recent_games: Goalie's recent games (for team defensive stats from same boxscores)

        Returns:
            dict: Team/opponent feature mappings
        """
        features = {}

        # --- Team defensive stats (from goalie's recent games boxscores) ---
        team_ga_values = []
        team_sa_values = []

        if nhl_fetcher and recent_games:
            filtered = [g for g in recent_games if g.get('gameDate', '') != game_date]
            for g in filtered:
                gid = g.get('gameId')
                if not gid:
                    continue
                # The goalie's team abbrev may vary if traded, but use team param
                team_stats = nhl_fetcher.get_team_boxscore_stats(gid, team)
                if team_stats:
                    team_ga_values.append(team_stats['opp_goals'])
                    team_sa_values.append(team_stats['opp_shots'])

        if team_ga_values:
            for window in [5, 10]:
                if len(team_ga_values) >= window:
                    features[f'team_goals_against_rolling_{window}'] = float(np.mean(team_ga_values[:window]))
                    features[f'team_shots_against_rolling_{window}'] = float(np.mean(team_sa_values[:window]))
                else:
                    features[f'team_goals_against_rolling_{window}'] = float(np.mean(team_ga_values))
                    features[f'team_shots_against_rolling_{window}'] = float(np.mean(team_sa_values))
        else:
            for window in [5, 10]:
                features[f'team_goals_against_rolling_{window}'] = 3.0
                features[f'team_shots_against_rolling_{window}'] = 30.0

        # --- Opponent offensive stats (from opponent's own recent games) ---
        if nhl_fetcher:
            opp_stats = nhl_fetcher.get_opponent_recent_stats(opponent, game_date)
            if opp_stats:
                opp_goals_values = [s['opp_goals'] for s in opp_stats]
                opp_shots_values = [s['opp_shots'] for s in opp_stats]

                for window in [5, 10]:
                    if len(opp_goals_values) >= window:
                        features[f'opp_goals_rolling_{window}'] = float(np.mean(opp_goals_values[:window]))
                        features[f'opp_shots_rolling_{window}'] = float(np.mean(opp_shots_values[:window]))
                    elif opp_goals_values:
                        features[f'opp_goals_rolling_{window}'] = float(np.mean(opp_goals_values))
                        features[f'opp_shots_rolling_{window}'] = float(np.mean(opp_shots_values))
                    else:
                        features[f'opp_goals_rolling_{window}'] = 3.0
                        features[f'opp_shots_rolling_{window}'] = 30.0
            else:
                for window in [5, 10]:
                    features[f'opp_goals_rolling_{window}'] = 3.0
                    features[f'opp_shots_rolling_{window}'] = 30.0
        else:
            for window in [5, 10]:
                features[f'opp_goals_rolling_{window}'] = 3.0
                features[f'opp_shots_rolling_{window}'] = 30.0

        return features

    def prepare_prediction_features(self, goalie_id, team, opponent, is_home, game_date,
                                     recent_games, betting_line=None, nhl_fetcher=None):
        """
        Combine all features into model input format (114 features in correct order).

        Args:
            goalie_id: NHL goalie ID
            team: Team abbreviation
            opponent: Opponent abbreviation
            is_home: 1 if home, 0 if away
            game_date: Date of game
            recent_games: List of recent game dicts
            betting_line: Betting line for saves over/under (REQUIRED)
            nhl_fetcher: NHLBettingData instance for fetching boxscores

        Returns:
            pd.DataFrame: Single row with 96 features in correct order
        """
        features = {}

        # Home/away indicator
        features['is_home'] = is_home

        # Goalie rolling features (basic + situation-specific)
        goalie_features = self.calculate_goalie_features(
            recent_games, game_date, nhl_fetcher=nhl_fetcher, goalie_id=goalie_id
        )
        features.update(goalie_features)

        # Team/opponent features
        team_features = self.calculate_team_features(
            team, opponent, game_date, nhl_fetcher=nhl_fetcher, recent_games=recent_games
        )
        features.update(team_features)

        # Betting line
        if betting_line is not None:
            features['betting_line'] = betting_line
        else:
            features['betting_line'] = 25.0

        # Line-relative features
        bl = features['betting_line']
        for window in [3, 5, 10]:
            rolling_key = f'saves_rolling_{window}'
            std_key = f'saves_rolling_std_{window}'
            rolling_val = features.get(rolling_key, 25.0)
            std_val = features.get(std_key, 5.0)

            features[f'line_vs_rolling_{window}'] = bl - rolling_val
            features[f'line_z_score_{window}'] = (
                (bl - rolling_val) / std_val if std_val > 0.01 else 0.0
            )

        # --- Engineered features (18 total) ---
        self._add_engineered_features(features)

        # Ensure correct feature order
        if self.feature_names:
            ordered_features = {}
            for feat_name in self.feature_names:
                ordered_features[feat_name] = features.get(feat_name, 0.0)
            return pd.DataFrame([ordered_features])
        else:
            return pd.DataFrame([features])

    def _add_engineered_features(self, features):
        """
        Add the 18 engineered features to the feature dict.
        Must be called after all base features (goalie, team, line-relative) are computed.
        Matches the exact logic used in training (scripts/optimize_features.py).
        """
        # --- Interaction features ---
        # Save efficiency: saves / shots_against
        for w in [3, 5, 10]:
            sr = features.get(f'saves_rolling_{w}', 25.0)
            sar = features.get(f'shots_against_rolling_{w}', 28.0)
            features[f'save_efficiency_{w}'] = sr / max(sar, 1.0)

        # Even strength saves proportion of total saves
        for w in [5, 10]:
            es = features.get(f'even_strength_saves_rolling_{w}', 20.0)
            sr = features.get(f'saves_rolling_{w}', 25.0)
            features[f'es_saves_proportion_{w}'] = es / max(sr, 1.0)

        # Opponent shots vs team shots against
        for w in [5, 10]:
            opp_shots = features.get(f'opp_shots_rolling_{w}', 30.0)
            team_sa = features.get(f'team_shots_against_rolling_{w}', 30.0)
            features[f'opp_vs_team_shots_{w}'] = opp_shots - team_sa

        # --- Volatility features ---
        # Coefficient of variation for saves
        for w in [5, 10]:
            mean_val = features.get(f'saves_rolling_{w}', 25.0)
            std_val = features.get(f'saves_rolling_std_{w}', 5.0)
            features[f'saves_cv_{w}'] = std_val / max(mean_val, 1.0)

        # Volatility relative to line
        bl = features.get('betting_line', 25.0)
        for w in [5, 10]:
            std_val = features.get(f'saves_rolling_std_{w}', 5.0)
            features[f'volatility_vs_line_{w}'] = std_val / max(bl, 1.0)

        # --- Trend / momentum features ---
        # Short-term (3-game) vs long-term (10-game) momentum
        for stat in ['saves', 'shots_against', 'goals_against']:
            short_val = features.get(f'{stat}_rolling_3', 0.0)
            long_val = features.get(f'{stat}_rolling_10', 0.0)
            features[f'{stat}_momentum'] = short_val - long_val

        # Save percentage momentum
        sp_short = features.get('save_percentage_rolling_3', 0.905)
        sp_long = features.get('save_percentage_rolling_10', 0.905)
        features['save_pct_momentum'] = sp_short - sp_long

        # --- Matchup context features ---
        # Expected workload diff: opponent shots tendency vs goalie's recent workload
        opp_shots_5 = features.get('opp_shots_rolling_5', 30.0)
        sa_5 = features.get('shots_against_rolling_5', 28.0)
        features['expected_workload_diff'] = opp_shots_5 - sa_5

        # Line vs opponent-implied saves
        opp_goals_5 = features.get('opp_goals_rolling_5', 3.0)
        opp_saves_implied = opp_shots_5 - opp_goals_5
        features['line_vs_opp_implied_saves'] = bl - opp_saves_implied

        # Rest * recent performance interaction
        rest = min(features.get('goalie_days_rest', 3), 7)
        saves_5 = features.get('saves_rolling_5', 25.0)
        features['rest_x_performance'] = rest * saves_5

    def _get_default_features(self):
        """Return default feature values when no history available"""
        defaults = {
            'is_home': 0,
            'goalie_days_rest': 3,
            'goalie_is_back_to_back': 0,
        }

        windows = [3, 5, 10]
        for window in windows:
            defaults[f'saves_rolling_{window}'] = 25.0
            defaults[f'saves_rolling_std_{window}'] = 5.0
            defaults[f'shots_against_rolling_{window}'] = 28.0
            defaults[f'shots_against_rolling_std_{window}'] = 5.0
            defaults[f'goals_against_rolling_{window}'] = 3.0
            defaults[f'goals_against_rolling_std_{window}'] = 1.5
            defaults[f'save_percentage_rolling_{window}'] = 0.905
            defaults[f'save_percentage_rolling_std_{window}'] = 0.05

            # Situation-specific defaults
            defaults[f'even_strength_saves_rolling_{window}'] = 20.0
            defaults[f'even_strength_saves_rolling_std_{window}'] = 4.0
            defaults[f'even_strength_shots_against_rolling_{window}'] = 22.0
            defaults[f'even_strength_shots_against_rolling_std_{window}'] = 4.0
            defaults[f'even_strength_goals_against_rolling_{window}'] = 2.5
            defaults[f'even_strength_goals_against_rolling_std_{window}'] = 1.5

            defaults[f'power_play_saves_rolling_{window}'] = 3.5
            defaults[f'power_play_saves_rolling_std_{window}'] = 1.5
            defaults[f'power_play_shots_against_rolling_{window}'] = 4.0
            defaults[f'power_play_shots_against_rolling_std_{window}'] = 1.5
            defaults[f'power_play_goals_against_rolling_{window}'] = 0.5
            defaults[f'power_play_goals_against_rolling_std_{window}'] = 0.7

            defaults[f'short_handed_saves_rolling_{window}'] = 0.6
            defaults[f'short_handed_saves_rolling_std_{window}'] = 0.5
            defaults[f'short_handed_shots_against_rolling_{window}'] = 0.7
            defaults[f'short_handed_shots_against_rolling_std_{window}'] = 0.5
            defaults[f'short_handed_goals_against_rolling_{window}'] = 0.1
            defaults[f'short_handed_goals_against_rolling_std_{window}'] = 0.3

        # Team/opponent defaults
        defaults['opp_goals_rolling_5'] = 3.0
        defaults['opp_goals_rolling_10'] = 3.0
        defaults['opp_shots_rolling_5'] = 30.0
        defaults['opp_shots_rolling_10'] = 30.0
        defaults['team_goals_against_rolling_5'] = 3.0
        defaults['team_goals_against_rolling_10'] = 3.0
        defaults['team_shots_against_rolling_5'] = 30.0
        defaults['team_shots_against_rolling_10'] = 30.0

        # Line-relative feature defaults
        for window in windows:
            defaults[f'line_vs_rolling_{window}'] = 0.0
            defaults[f'line_z_score_{window}'] = 0.0

        # Engineered feature defaults
        for w in [3, 5, 10]:
            defaults[f'save_efficiency_{w}'] = 25.0 / 28.0  # ~0.893
        for w in [5, 10]:
            defaults[f'es_saves_proportion_{w}'] = 20.0 / 25.0  # ~0.80
            defaults[f'opp_vs_team_shots_{w}'] = 0.0
            defaults[f'saves_cv_{w}'] = 5.0 / 25.0  # ~0.20
            defaults[f'volatility_vs_line_{w}'] = 5.0 / 25.0  # ~0.20
        defaults['saves_momentum'] = 0.0
        defaults['shots_against_momentum'] = 0.0
        defaults['goals_against_momentum'] = 0.0
        defaults['save_pct_momentum'] = 0.0
        defaults['expected_workload_diff'] = 0.0
        defaults['line_vs_opp_implied_saves'] = 0.0
        defaults['rest_x_performance'] = 3 * 25.0  # rest=3 * saves=25

        return defaults
