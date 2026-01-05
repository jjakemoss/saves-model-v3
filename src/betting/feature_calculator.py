"""
Feature calculator for live betting predictions
Calculates rolling features from recent goalie games
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
        feature_file = Path('models/classifier_feature_names.json')
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = None

    def calculate_goalie_features(self, recent_games, game_date):
        """
        Calculate goalie rolling features from recent games

        Args:
            recent_games: List of recent game dicts from NHL API
            game_date: Date of prediction (to calculate rest days)

        Returns:
            dict: Feature name -> value mapping
        """
        features = {}

        if not recent_games:
            # Return defaults if no history
            return self._get_default_features()

        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(recent_games)

        # CRITICAL: Filter out any games from the current date to prevent data leakage
        # This allows re-running predictions for late games after early games complete
        if 'gameDate' in df.columns:
            df = df[df['gameDate'] != game_date].reset_index(drop=True)

            if len(df) == 0:
                # No historical games available (only same-day game in log)
                return self._get_default_features()

        # Calculate rolling averages for different windows
        windows = [3, 5, 10]

        # Basic stats to track
        stats = {
            'saves': 'saves',
            'shots_against': 'shots',
            'goals_against': 'goalsAgainst',
            'save_percentage': 'savePctg',
        }

        # Situation-specific stats
        situation_stats = {
            'even_strength': 'evenStrength',
            'power_play': 'powerPlay',
            'short_handed': 'shortHanded',
        }

        # Calculate basic rolling features
        for stat_name, api_field in stats.items():
            if api_field in df.columns:
                values = df[api_field].values

                for window in windows:
                    # Mean
                    if len(values) >= window:
                        features[f'{stat_name}_rolling_{window}'] = np.mean(values[:window])
                    else:
                        features[f'{stat_name}_rolling_{window}'] = np.mean(values) if len(values) > 0 else 0

                    # Std
                    if len(values) >= window:
                        features[f'{stat_name}_rolling_std_{window}'] = np.std(values[:window])
                    else:
                        features[f'{stat_name}_rolling_std_{window}'] = np.std(values) if len(values) > 1 else 0

        # Calculate situation-specific features
        for situation, api_prefix in situation_stats.items():
            saves_field = f'{api_prefix}SavePctg'  # They provide save % directly

            if saves_field in df.columns:
                values = df[saves_field].values * 100  # Convert to percentage

                for window in windows:
                    stat_name = f'{situation}_save_pct'

                    if len(values) >= window:
                        features[f'{stat_name}_rolling_{window}'] = np.mean(values[:window])
                    else:
                        features[f'{stat_name}_rolling_{window}'] = np.mean(values) if len(values) > 0 else 0

                    if len(values) >= window:
                        features[f'{stat_name}_rolling_std_{window}'] = np.std(values[:window])
                    else:
                        features[f'{stat_name}_rolling_std_{window}'] = np.std(values) if len(values) > 1 else 0

        # Calculate rest days
        if len(recent_games) > 0:
            last_game_date = recent_games[0].get('gameDate', '')
            if last_game_date:
                last_date = datetime.strptime(last_game_date, '%Y-%m-%d')
                current_date = datetime.strptime(game_date, '%Y-%m-%d')
                features['goalie_days_rest'] = (current_date - last_date).days
            else:
                features['goalie_days_rest'] = 3  # Default

            # Check if back-to-back
            features['goalie_is_back_to_back'] = 1 if features.get('goalie_days_rest', 3) == 1 else 0
        else:
            features['goalie_days_rest'] = 3
            features['goalie_is_back_to_back'] = 0

        return features

    def calculate_team_features(self, team, opponent, season='20252026'):
        """
        Calculate team defensive and opponent offensive features

        Args:
            team: Team abbreviation
            opponent: Opponent team abbreviation
            season: Season in YYYYYYYY format

        Returns:
            dict: Team/opponent feature mappings
        """
        features = {}

        # For now, use defaults
        # In production, would fetch team stats from NHL API
        # These would be rolling averages of team defensive performance

        features['opp_goals_rolling_5'] = 3.0
        features['opp_goals_rolling_10'] = 3.0
        features['opp_shots_rolling_5'] = 30.0
        features['opp_shots_rolling_10'] = 30.0

        features['team_goals_against_rolling_5'] = 3.0
        features['team_goals_against_rolling_10'] = 3.0
        features['team_shots_against_rolling_5'] = 30.0
        features['team_shots_against_rolling_10'] = 30.0

        return features

    def prepare_prediction_features(self, goalie_id, team, opponent, is_home, game_date, recent_games):
        """
        Combine all features into model input format (89 features)

        Args:
            goalie_id: NHL goalie ID
            team: Team abbreviation
            opponent: Opponent abbreviation
            is_home: 1 if home, 0 if away
            game_date: Date of game
            recent_games: List of recent game dicts

        Returns:
            pd.DataFrame: Single row with 89 features in correct order
        """
        features = {}

        # Home/away indicator
        features['is_home'] = is_home

        # Goalie rolling features
        goalie_features = self.calculate_goalie_features(recent_games, game_date)
        features.update(goalie_features)

        # Team/opponent features
        team_features = self.calculate_team_features(team, opponent)
        features.update(team_features)

        # If we have feature names, ensure correct order
        if self.feature_names:
            # Create ordered dict matching feature names
            ordered_features = {}
            for feat_name in self.feature_names:
                ordered_features[feat_name] = features.get(feat_name, 0.0)

            return pd.DataFrame([ordered_features])
        else:
            return pd.DataFrame([features])

    def _get_default_features(self):
        """Return default feature values when no history available"""
        defaults = {
            'is_home': 0,
            'goalie_days_rest': 3,
            'goalie_is_back_to_back': 0,
        }

        # Default rolling stats (league average-ish)
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

        # Team/opponent defaults
        defaults['opp_goals_rolling_5'] = 3.0
        defaults['opp_goals_rolling_10'] = 3.0
        defaults['opp_shots_rolling_5'] = 30.0
        defaults['opp_shots_rolling_10'] = 30.0
        defaults['team_goals_against_rolling_5'] = 3.0
        defaults['team_goals_against_rolling_10'] = 3.0
        defaults['team_shots_against_rolling_5'] = 30.0
        defaults['team_shots_against_rolling_10'] = 30.0

        return defaults
