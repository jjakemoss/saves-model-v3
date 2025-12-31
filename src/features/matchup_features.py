"""Calculate matchup and contextual features

These features capture game context that affects save totals:
- Home/Away splits
- Rest and fatigue (days between games, back-to-backs)
- Travel distance
- Historical head-to-head performance
- Betting line context
- Volatility metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MatchupFeatureCalculator:
    """
    Calculate matchup-specific and contextual features

    Includes:
    - Home/away performance splits
    - Rest/fatigue metrics
    - Travel impact
    - Historical matchups (H2H)
    - Betting line comparisons
    - Volatility and consistency metrics
    """

    # NHL team locations (for travel distance calculation)
    TEAM_LOCATIONS = {
        'ANA': (33.8, -117.9),  # Anaheim
        'BOS': (42.4, -71.1),   # Boston
        'BUF': (42.9, -78.9),   # Buffalo
        'CGY': (51.0, -114.1),  # Calgary
        'CAR': (35.8, -78.6),   # Carolina
        'CHI': (41.9, -87.7),   # Chicago
        'COL': (39.7, -105.0),  # Colorado
        'CBJ': (40.0, -83.0),   # Columbus
        'DAL': (32.8, -96.8),   # Dallas
        'DET': (42.3, -83.0),   # Detroit
        'EDM': (53.5, -113.5),  # Edmonton
        'FLA': (26.2, -80.3),   # Florida
        'LAK': (34.0, -118.3),  # LA Kings
        'MIN': (44.9, -93.1),   # Minnesota
        'MTL': (45.5, -73.6),   # Montreal
        'NSH': (36.2, -86.8),   # Nashville
        'NJD': (40.7, -74.2),   # New Jersey
        'NYI': (40.7, -73.6),   # NY Islanders
        'NYR': (40.8, -73.9),   # NY Rangers
        'OTT': (45.3, -75.9),   # Ottawa
        'PHI': (39.9, -75.2),   # Philadelphia
        'PIT': (40.4, -80.0),   # Pittsburgh
        'SEA': (47.6, -122.3),  # Seattle
        'SJS': (37.3, -121.9),  # San Jose
        'STL': (38.6, -90.2),   # St. Louis
        'TBL': (27.9, -82.5),   # Tampa Bay
        'TOR': (43.6, -79.4),   # Toronto
        'VAN': (49.3, -123.1),  # Vancouver
        'VGK': (36.1, -115.2),  # Vegas
        'WSH': (38.9, -77.0),   # Washington
        'WPG': (49.9, -97.1),   # Winnipeg
        'ARI': (33.4, -112.1),  # Arizona (historical)
        'UTH': (40.8, -111.9),  # Utah
    }

    def calculate_rest_features(
        self,
        current_game_date: str,
        player_games: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate rest and fatigue features

        Args:
            current_game_date: Date of current game (YYYY-MM-DD)
            player_games: Player's game log (sorted chronologically, EXCLUDING current game)

        Returns:
            Dictionary of rest/fatigue features
        """
        if len(player_games) == 0:
            return {
                'days_since_last_start': 7,  # Assume well-rested
                'consecutive_starts_streak': 0,
                'starts_in_last_7_days': 0,
                'is_back_to_back': 0,
                'is_back_to_back_second_game': 0,
            }

        # Parse current game date
        current_date = pd.to_datetime(current_game_date)

        # Get last game date
        last_game = player_games.iloc[-1]
        last_game_date = pd.to_datetime(last_game['game_date'])

        days_since_last = (current_date - last_game_date).days

        # Count starts in last 7 days
        seven_days_ago = current_date - timedelta(days=7)
        recent_games = player_games[pd.to_datetime(player_games['game_date']) >= seven_days_ago]
        starts_last_7 = len(recent_games)

        # Check for back-to-back
        is_b2b = days_since_last == 1

        # Check if this is second game of back-to-back
        # (last game was also after only 1 day rest)
        is_b2b_second = False
        if len(player_games) >= 2:
            second_last_game = player_games.iloc[-2]
            second_last_date = pd.to_datetime(second_last_game['game_date'])
            days_between_last_two = (last_game_date - second_last_date).days
            is_b2b_second = (days_since_last == 1 and days_between_last_two == 1)

        # Calculate consecutive starts streak
        consecutive_starts = 1  # Current game will be at least 1
        for i in range(len(player_games) - 1, 0, -1):
            current_date_i = pd.to_datetime(player_games.iloc[i]['game_date'])
            prev_date = pd.to_datetime(player_games.iloc[i-1]['game_date'])
            days_diff = (current_date_i - prev_date).days

            # If games are consecutive (1-2 days apart), increment streak
            if days_diff <= 2:
                consecutive_starts += 1
            else:
                break

        return {
            'days_since_last_start': days_since_last,
            'consecutive_starts_streak': consecutive_starts,
            'starts_in_last_7_days': starts_last_7,
            'is_back_to_back': int(is_b2b),
            'is_back_to_back_second_game': int(is_b2b_second),
        }

    def calculate_home_away_splits(
        self,
        player_games: pd.DataFrame,
        is_home_game: bool
    ) -> Dict[str, float]:
        """
        Calculate home/away performance splits for goalie

        Args:
            player_games: Player's game log (excluding current game)
            is_home_game: Whether current game is at home

        Returns:
            Dictionary with home/away split features
        """
        if len(player_games) == 0:
            return {
                'is_home': int(is_home_game),
                'goalie_home_save_pct_season': 0.0,
                'goalie_away_save_pct_season': 0.0,
                'goalie_home_advantage': 0.0,
                'goalie_home_saves_avg': 0.0,
                'goalie_away_saves_avg': 0.0,
            }

        # Filter home and away games
        home_games = player_games[player_games['is_home'] == True]
        away_games = player_games[player_games['is_home'] == False]

        # Calculate save percentages
        home_save_pct = home_games['save_percentage'].mean() if len(home_games) > 0 else 0.0
        away_save_pct = away_games['save_percentage'].mean() if len(away_games) > 0 else 0.0

        # Home advantage = difference in save %
        home_advantage = home_save_pct - away_save_pct

        # Average saves by location
        home_saves_avg = home_games['saves'].mean() if len(home_games) > 0 else 0.0
        away_saves_avg = away_games['saves'].mean() if len(away_games) > 0 else 0.0

        return {
            'is_home': int(is_home_game),
            'goalie_home_save_pct_season': home_save_pct,
            'goalie_away_save_pct_season': away_save_pct,
            'goalie_home_advantage': home_advantage,
            'goalie_home_saves_avg': home_saves_avg,
            'goalie_away_saves_avg': away_saves_avg,
        }

    def calculate_travel_features(
        self,
        current_team: str,
        opponent_team: str,
        is_home_game: bool,
        player_games: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate travel distance and road trip fatigue

        Args:
            current_team: Team abbreviation
            opponent_team: Opponent abbreviation
            is_home_game: Whether game is at home
            player_games: Recent games (for road trip tracking)

        Returns:
            Dictionary with travel features
        """
        # Calculate distance between teams
        distance = self._calculate_distance_between_teams(current_team, opponent_team)

        # For home games, no travel for current team
        if is_home_game:
            travel_miles = 0
            games_on_road_trip = 0
        else:
            travel_miles = distance

            # Count consecutive away games (road trip length)
            games_on_road_trip = 1  # Current game
            for i in range(len(player_games) - 1, -1, -1):
                if not player_games.iloc[i]['is_home']:
                    games_on_road_trip += 1
                else:
                    break

        return {
            'travel_miles_since_last_game': travel_miles,
            'games_on_current_road_trip': games_on_road_trip,
        }

    def _calculate_distance_between_teams(self, team1: str, team2: str) -> float:
        """Calculate great circle distance between two teams in miles"""
        if team1 not in self.TEAM_LOCATIONS or team2 not in self.TEAM_LOCATIONS:
            return 0.0

        lat1, lon1 = self.TEAM_LOCATIONS[team1]
        lat2, lon2 = self.TEAM_LOCATIONS[team2]

        # Haversine formula
        R = 3959  # Earth radius in miles

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        distance = R * c
        return distance

    def calculate_h2h_features(
        self,
        player_games: pd.DataFrame,
        opponent_team: str,
        window: int = 5
    ) -> Dict[str, float]:
        """
        Calculate head-to-head historical performance vs opponent

        Args:
            player_games: Player's game log (excluding current game)
            opponent_team: Opponent abbreviation
            window: Number of recent H2H games to consider

        Returns:
            Dictionary with H2H features
        """
        # Filter games vs this opponent
        h2h_games = player_games[player_games['opponent_team'] == opponent_team]

        if len(h2h_games) == 0:
            return {
                'h2h_saves_avg_last_5': 0.0,
                'goalie_vs_opponent_save_pct_career': 0.0,
                'h2h_games_played': 0,
            }

        # Get most recent H2H games
        recent_h2h = h2h_games.tail(window)

        return {
            'h2h_saves_avg_last_5': recent_h2h['saves'].mean(),
            'goalie_vs_opponent_save_pct_career': h2h_games['save_percentage'].mean(),
            'h2h_games_played': len(h2h_games),
        }

    def calculate_volatility_metrics(
        self,
        player_games: pd.DataFrame,
        stat_column: str = 'saves',
        window: int = 10
    ) -> Dict[str, float]:
        """
        Calculate performance volatility metrics

        Args:
            player_games: Player's game log
            stat_column: Column to analyze
            window: Window for volatility calculation

        Returns:
            Dictionary with volatility features
        """
        if len(player_games) < 3:
            return {
                f'{stat_column}_volatility_last_{window}': 0.0,
                f'{stat_column}_std_last_{window}': 0.0,
            }

        # Get recent games
        recent_games = player_games.tail(window)

        if stat_column not in recent_games.columns:
            return {
                f'{stat_column}_volatility_last_{window}': 0.0,
                f'{stat_column}_std_last_{window}': 0.0,
            }

        values = recent_games[stat_column]

        # Standard deviation
        std = values.std()

        # Coefficient of variation (volatility)
        mean_val = values.mean()
        volatility = std / mean_val if mean_val > 0 else 0.0

        return {
            f'{stat_column}_volatility_last_{window}': volatility,
            f'{stat_column}_std_last_{window}': std,
        }

    def calculate_betting_line_context(
        self,
        betting_line: float,
        player_games: pd.DataFrame,
        stat_column: str = 'saves'
    ) -> Dict[str, float]:
        """
        Calculate features relative to betting line

        Args:
            betting_line: Current betting line (e.g., 27.5)
            player_games: Player's game log
            stat_column: Column to compare to line

        Returns:
            Dictionary with betting line context features
        """
        if len(player_games) == 0:
            return {
                'betting_line': betting_line,
                'line_vs_goalie_season_avg': 0.0,
                'line_vs_goalie_last_5_avg': 0.0,
                'overline_frequency_last_10': 0.0,
            }

        # Season average
        season_avg = player_games[stat_column].mean()
        line_vs_season = betting_line - season_avg

        # Last 5 games average
        last_5_avg = player_games.tail(5)[stat_column].mean()
        line_vs_last_5 = betting_line - last_5_avg

        # Over line frequency in last 10 games
        last_10 = player_games.tail(10)
        over_count = (last_10[stat_column] > betting_line).sum()
        over_freq = over_count / len(last_10) if len(last_10) > 0 else 0.0

        return {
            'betting_line': betting_line,
            'line_vs_goalie_season_avg': line_vs_season,
            'line_vs_goalie_last_5_avg': line_vs_last_5,
            'overline_frequency_last_10': over_freq,
        }


def calculate_all_matchup_features(
    current_game: Dict[str, Any],
    player_games: pd.DataFrame,
    betting_line: float
) -> Dict[str, Any]:
    """
    Calculate all matchup features for a single game prediction

    Args:
        current_game: Dictionary with current game info (date, home/away, teams)
        player_games: Player's prior game log (EXCLUDING current game)
        betting_line: Betting line for saves

    Returns:
        Dictionary with all matchup features
    """
    calculator = MatchupFeatureCalculator()

    features = {}

    # Rest/fatigue
    features.update(calculator.calculate_rest_features(
        current_game['game_date'],
        player_games
    ))

    # Home/away splits
    features.update(calculator.calculate_home_away_splits(
        player_games,
        current_game['is_home']
    ))

    # Travel
    features.update(calculator.calculate_travel_features(
        current_game['team'],
        current_game['opponent'],
        current_game['is_home'],
        player_games
    ))

    # Head-to-head
    features.update(calculator.calculate_h2h_features(
        player_games,
        current_game['opponent']
    ))

    # Volatility
    features.update(calculator.calculate_volatility_metrics(
        player_games,
        stat_column='saves',
        window=10
    ))

    # Betting line context
    features.update(calculator.calculate_betting_line_context(
        betting_line,
        player_games,
        stat_column='saves'
    ))

    return features
