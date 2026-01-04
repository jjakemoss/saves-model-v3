"""Calculate rolling and exponential weighted average features

This module implements exponential weighted averages (EWA) which give more weight
to recent games compared to older games. This is superior to arithmetic rolling
averages for predicting future performance.

CRITICAL: All rolling features EXCLUDE the current game to prevent data leakage.
For game at index i, we only use games 0 to i-1.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RollingFeatureCalculator:
    """
    Calculate exponential weighted and rolling average features

    Exponential weights example for 5-game window:
    - Most recent game: 50%
    - 2nd most recent: 30%
    - 3rd: 15%
    - 4th: 4%
    - 5th: 1%
    (Weights sum to 100%)
    """

    def __init__(self, windows: List[int] = [3, 5, 10], min_games: int = 3):
        """
        Initialize rolling feature calculator

        Args:
            windows: List of window sizes for rolling averages (e.g., [3, 5, 10])
            min_games: Minimum games required before calculating rolling stats
        """
        self.windows = windows
        self.min_games = min_games

    def calculate_ewa(self, values: pd.Series, window: int) -> pd.Series:
        """
        Calculate exponential weighted average

        Args:
            values: Series of values (already sorted chronologically)
            window: Window size

        Returns:
            Series of exponential weighted averages (same length as input)
        """
        # Use pandas ewm with span parameter
        # span roughly corresponds to window size
        # adjust=False means we use recursive formula
        # min_periods ensures we have enough data
        return values.ewm(span=window, adjust=False, min_periods=self.min_games).mean()

    def calculate_rolling_features_for_player(
        self,
        player_games: pd.DataFrame,
        stat_columns: List[str],
        player_id_col: str = 'goalie_id'
    ) -> pd.DataFrame:
        """
        Calculate rolling features for a single player's game log

        CRITICAL: Features for game i only use games 0 to i-1 (exclude current game)

        Args:
            player_games: DataFrame with player's games (sorted chronologically)
            stat_columns: List of column names to calculate rolling stats for
            player_id_col: Column name for player identifier

        Returns:
            DataFrame with original data plus rolling feature columns
        """
        if len(player_games) == 0:
            return player_games

        # Ensure sorted by date
        if 'game_date' in player_games.columns:
            player_games = player_games.sort_values('game_date').reset_index(drop=True)

        result_df = player_games.copy()

        # Collect all new columns in a dict to add at once (more efficient)
        new_columns = {}

        for stat in stat_columns:
            if stat not in player_games.columns:
                logger.warning(f"Column {stat} not found in player games")
                continue

            for window in self.windows:
                # CRITICAL: Use shift(1) to exclude current game
                # For game i, we calculate stats using games 0 to i-1
                shifted_values = player_games[stat].shift(1)

                # Exponential weighted average
                ewa_col = f'{stat}_ewa_{window}'
                new_columns[ewa_col] = shifted_values.ewm(
                    span=window,
                    adjust=False,
                    min_periods=min(self.min_games, window)
                ).mean()

                # Arithmetic rolling average (for comparison)
                rolling_col = f'{stat}_rolling_{window}'
                new_columns[rolling_col] = shifted_values.rolling(
                    window=window,
                    min_periods=min(self.min_games, window)
                ).mean()

                # Rolling standard deviation (for volatility)
                std_col = f'{stat}_rolling_std_{window}'
                new_columns[std_col] = shifted_values.rolling(
                    window=window,
                    min_periods=min(self.min_games, window)
                ).std()

        # Add all new columns at once using concat (much more efficient)
        if new_columns:
            new_cols_df = pd.DataFrame(new_columns, index=result_df.index)
            result_df = pd.concat([result_df, new_cols_df], axis=1)

        return result_df

    def calculate_seasonal_aggregates(
        self,
        player_games: pd.DataFrame,
        stat_columns: List[str],
        current_game_idx: int
    ) -> Dict[str, float]:
        """
        Calculate season-to-date aggregates for early-season games

        Args:
            player_games: DataFrame with player's games
            stat_columns: Columns to aggregate
            current_game_idx: Index of current game (we exclude this and use only prior games)

        Returns:
            Dictionary of seasonal averages
        """
        # Use only games before current game
        prior_games = player_games.iloc[:current_game_idx]

        if len(prior_games) == 0:
            return {f'{stat}_season_avg': 0.0 for stat in stat_columns}

        aggregates = {}
        for stat in stat_columns:
            if stat in prior_games.columns:
                aggregates[f'{stat}_season_avg'] = prior_games[stat].mean()
            else:
                aggregates[f'{stat}_season_avg'] = 0.0

        return aggregates

    def calculate_form_metrics(
        self,
        player_games: pd.DataFrame,
        stat_column: str = 'saves',
        window: int = 5
    ) -> pd.DataFrame:
        """
        Calculate form metrics (trend, consistency)

        Args:
            player_games: Player's game log
            stat_column: Column to analyze
            window: Window for trend calculation

        Returns:
            DataFrame with form metric columns added
        """
        result_df = player_games.copy()

        if stat_column not in player_games.columns:
            return result_df

        # Shift to exclude current game
        shifted_values = player_games[stat_column].shift(1)

        # Trend: difference between recent average and older average
        # Positive trend = improving
        recent_avg = shifted_values.rolling(window=window//2, min_periods=1).mean()
        older_avg = shifted_values.rolling(window=window, min_periods=1).mean()
        result_df[f'{stat_column}_trend_{window}'] = recent_avg - older_avg

        # Volatility: coefficient of variation (std / mean)
        rolling_mean = shifted_values.rolling(window=window, min_periods=self.min_games).mean()
        rolling_std = shifted_values.rolling(window=window, min_periods=self.min_games).std()
        result_df[f'{stat_column}_volatility_{window}'] = rolling_std / (rolling_mean + 1e-6)

        return result_df

    def calculate_over_line_frequency(
        self,
        player_games: pd.DataFrame,
        betting_lines: pd.Series,
        stat_column: str = 'saves',
        window: int = 10
    ) -> pd.Series:
        """
        Calculate frequency of going over betting line in recent games

        Args:
            player_games: Player's game log
            betting_lines: Series of betting lines for each game
            stat_column: Column with actual values
            window: Rolling window

        Returns:
            Series with over-line frequency
        """
        if stat_column not in player_games.columns:
            return pd.Series(0.0, index=player_games.index)

        # Binary indicator: did player go over line?
        over_line = (player_games[stat_column] > betting_lines).astype(int)

        # Shift to exclude current game
        shifted_over = over_line.shift(1)

        # Rolling frequency
        over_freq = shifted_over.rolling(window=window, min_periods=self.min_games).mean()

        return over_freq


def calculate_all_rolling_features(
    games_df: pd.DataFrame,
    goalie_stats: List[str] = ['saves', 'save_percentage', 'shots_against', 'goals_against'],
    team_stats: List[str] = ['team_shots', 'opp_shots', 'team_goals', 'opp_goals'],
    windows: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Calculate rolling features for all players in a dataset

    Args:
        games_df: DataFrame with all games
        goalie_stats: List of goalie stat columns
        team_stats: List of team stat columns
        windows: Rolling window sizes

    Returns:
        DataFrame with rolling features added
    """
    calculator = RollingFeatureCalculator(windows=windows)

    # Group by goalie and calculate rolling features
    result_dfs = []

    for goalie_id, goalie_games in games_df.groupby('goalie_id'):
        # Sort by date within each goalie's games
        goalie_games = goalie_games.sort_values('game_date').reset_index(drop=True)

        # Calculate rolling features for goalie stats
        goalie_games = calculator.calculate_rolling_features_for_player(
            goalie_games,
            stat_columns=goalie_stats,
            player_id_col='goalie_id'
        )

        # Calculate form metrics
        goalie_games = calculator.calculate_form_metrics(
            goalie_games,
            stat_column='saves',
            window=10
        )

        result_dfs.append(goalie_games)

    # Concatenate all goalie DataFrames
    if result_dfs:
        result = pd.concat(result_dfs, ignore_index=True)
        return result
    else:
        return games_df
