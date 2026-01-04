"""Calculate rest and fatigue features

This module calculates goalie and team rest/fatigue metrics which are critical
for predicting save volume. Tired goalies and teams show significant performance
degradation.

Key insights:
- Goalie back-to-back: -4 to -6 saves (fatigue + early pulls)
- Opponent back-to-back: -2 to -3 shots (less energy, slower pace)
- Fresh goalie (3+ days): +1 to +2 saves (sharper reflexes)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def calculate_rest_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rest and fatigue features for goalies and teams

    Features added:
    - goalie_days_rest: Days since goalie's last start (0-7+)
    - goalie_is_back_to_back: Binary indicator for back-to-back games
    - goalie_consecutive_starts: Number of consecutive starts (1, 2, 3+)
    - goalie_starts_last_7_days: Number of starts in last 7 days
    - goalie_starts_last_14_days: Number of starts in last 14 days
    - team_days_rest: Days since team's last game
    - team_is_back_to_back: Binary indicator for team back-to-back
    - team_games_last_7_days: Number of games team played in last 7 days
    - opponent_days_rest: Days since opponent's last game
    - opponent_is_back_to_back: Binary indicator for opponent back-to-back
    - opponent_games_last_7_days: Number of games opponent played in last 7 days

    Args:
        games_df: DataFrame with game data (must have game_date, goalie_id, team_abbrev, opponent_team)

    Returns:
        DataFrame with rest/fatigue features added
    """
    logger.info("Calculating rest and fatigue features...")

    # Ensure sorted by date
    df = games_df.sort_values('game_date').reset_index(drop=True)

    # Convert game_date to datetime if not already
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Initialize feature columns
    df['goalie_days_rest'] = np.nan
    df['goalie_is_back_to_back'] = 0
    df['goalie_consecutive_starts'] = 1
    df['goalie_starts_last_7_days'] = 1  # Current game counts
    df['goalie_starts_last_14_days'] = 1

    df['team_days_rest'] = np.nan
    df['team_is_back_to_back'] = 0
    df['team_games_last_7_days'] = 1

    df['opponent_days_rest'] = np.nan
    df['opponent_is_back_to_back'] = 0
    df['opponent_games_last_7_days'] = 1

    # Calculate goalie rest features
    logger.info("Calculating goalie rest features...")
    for goalie_id, goalie_games in df.groupby('goalie_id'):
        goalie_games = goalie_games.sort_values('game_date').reset_index(drop=False)
        indices = goalie_games.index

        for i in range(len(goalie_games)):
            current_idx = indices[i]
            current_date = goalie_games.iloc[i]['game_date']

            if i > 0:
                # Get previous game date
                prev_date = goalie_games.iloc[i-1]['game_date']
                days_diff = (current_date - prev_date).days

                df.at[current_idx, 'goalie_days_rest'] = days_diff
                df.at[current_idx, 'goalie_is_back_to_back'] = 1 if days_diff == 1 else 0

                # Consecutive starts: count how many games in a row (with <= 2 days between)
                consecutive = 1
                for j in range(i-1, -1, -1):
                    if j == i-1:
                        gap = days_diff
                    else:
                        gap = (goalie_games.iloc[j+1]['game_date'] - goalie_games.iloc[j]['game_date']).days

                    if gap <= 2:  # Consider consecutive if <= 2 days apart
                        consecutive += 1
                    else:
                        break

                df.at[current_idx, 'goalie_consecutive_starts'] = consecutive

                # Starts in last 7 days (including current game)
                cutoff_7 = current_date - timedelta(days=7)
                starts_7 = goalie_games[goalie_games['game_date'] >= cutoff_7].shape[0]
                df.at[current_idx, 'goalie_starts_last_7_days'] = starts_7

                # Starts in last 14 days
                cutoff_14 = current_date - timedelta(days=14)
                starts_14 = goalie_games[goalie_games['game_date'] >= cutoff_14].shape[0]
                df.at[current_idx, 'goalie_starts_last_14_days'] = starts_14
            else:
                # First game for this goalie - assume fresh (7+ days rest)
                df.at[current_idx, 'goalie_days_rest'] = 7

    # Calculate team rest features
    logger.info("Calculating team rest features...")
    for team, team_games in df.groupby('team_abbrev'):
        team_games = team_games.sort_values('game_date').reset_index(drop=False)
        indices = team_games.index

        for i in range(len(team_games)):
            current_idx = indices[i]
            current_date = team_games.iloc[i]['game_date']

            if i > 0:
                prev_date = team_games.iloc[i-1]['game_date']
                days_diff = (current_date - prev_date).days

                df.at[current_idx, 'team_days_rest'] = days_diff
                df.at[current_idx, 'team_is_back_to_back'] = 1 if days_diff == 1 else 0

                # Games in last 7 days (including current)
                cutoff_7 = current_date - timedelta(days=7)
                games_7 = team_games[team_games['game_date'] >= cutoff_7].shape[0]
                df.at[current_idx, 'team_games_last_7_days'] = games_7
            else:
                # First game for this team
                df.at[current_idx, 'team_days_rest'] = 2  # Assume 2 days rest

    # Calculate opponent rest features
    logger.info("Calculating opponent rest features...")
    for team, team_games in df.groupby('opponent_team'):
        team_games = team_games.sort_values('game_date').reset_index(drop=False)
        indices = team_games.index

        for i in range(len(team_games)):
            current_idx = indices[i]
            current_date = team_games.iloc[i]['game_date']

            if i > 0:
                prev_date = team_games.iloc[i-1]['game_date']
                days_diff = (current_date - prev_date).days

                df.at[current_idx, 'opponent_days_rest'] = days_diff
                df.at[current_idx, 'opponent_is_back_to_back'] = 1 if days_diff == 1 else 0

                # Games in last 7 days (including current)
                cutoff_7 = current_date - timedelta(days=7)
                games_7 = team_games[team_games['game_date'] >= cutoff_7].shape[0]
                df.at[current_idx, 'opponent_games_last_7_days'] = games_7
            else:
                # First game against this opponent
                df.at[current_idx, 'opponent_days_rest'] = 2  # Assume 2 days rest

    # Fill any remaining NaN values
    df['goalie_days_rest'] = df['goalie_days_rest'].fillna(7)
    df['team_days_rest'] = df['team_days_rest'].fillna(2)
    df['opponent_days_rest'] = df['opponent_days_rest'].fillna(2)

    # Cap days rest at 7 (anything more is effectively the same as 7)
    df['goalie_days_rest'] = df['goalie_days_rest'].clip(upper=7)
    df['team_days_rest'] = df['team_days_rest'].clip(upper=7)
    df['opponent_days_rest'] = df['opponent_days_rest'].clip(upper=7)

    logger.info("Rest and fatigue features calculated successfully")
    logger.info(f"  - Goalie back-to-backs: {df['goalie_is_back_to_back'].sum()}")
    logger.info(f"  - Team back-to-backs: {df['team_is_back_to_back'].sum()}")
    logger.info(f"  - Opponent back-to-backs: {df['opponent_is_back_to_back'].sum()}")

    return df


def calculate_game_state_features(games_df: pd.DataFrame, windows: list = [5, 10]) -> pd.DataFrame:
    """
    Calculate game state features (how teams play when winning vs losing)

    Features added:
    - team_avg_goal_differential_last_N: Average goal differential (positive = typically leading)
    - team_wins_last_N: Number of wins in last N games
    - team_losses_last_N: Number of losses in last N games
    - opponent_avg_goal_differential_last_N: Opponent's goal differential
    - opponent_wins_last_N: Opponent wins
    - opponent_losses_last_N: Opponent losses

    Args:
        games_df: DataFrame with game data
        windows: List of window sizes for rolling calculations

    Returns:
        DataFrame with game state features added
    """
    logger.info("Calculating game state features...")

    df = games_df.sort_values('game_date').reset_index(drop=True)

    # Calculate goal differential for each game (from team perspective)
    df['goal_differential'] = df['team_goals'] - df['opp_goals']

    # Determine win/loss (1 = win, 0 = loss/OT loss)
    df['is_win'] = (df['goal_differential'] > 0).astype(int)
    df['is_loss'] = (df['goal_differential'] < 0).astype(int)

    # Calculate rolling features for each team
    for window in windows:
        # Team features (using shift(1) to exclude current game)
        df[f'team_avg_goal_diff_last_{window}'] = df.groupby('team_abbrev')['goal_differential'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        df[f'team_wins_last_{window}'] = df.groupby('team_abbrev')['is_win'].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
        )
        df[f'team_losses_last_{window}'] = df.groupby('team_abbrev')['is_loss'].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
        )

        # Opponent features (using shift(1))
        df[f'opponent_avg_goal_diff_last_{window}'] = df.groupby('opponent_team')['goal_differential'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        df[f'opponent_wins_last_{window}'] = df.groupby('opponent_team')['is_win'].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
        )
        df[f'opponent_losses_last_{window}'] = df.groupby('opponent_team')['is_loss'].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
        )

    # Fill NaN values with 0 (for first games)
    for window in windows:
        df[f'team_avg_goal_diff_last_{window}'] = df[f'team_avg_goal_diff_last_{window}'].fillna(0)
        df[f'team_wins_last_{window}'] = df[f'team_wins_last_{window}'].fillna(0)
        df[f'team_losses_last_{window}'] = df[f'team_losses_last_{window}'].fillna(0)
        df[f'opponent_avg_goal_diff_last_{window}'] = df[f'opponent_avg_goal_diff_last_{window}'].fillna(0)
        df[f'opponent_wins_last_{window}'] = df[f'opponent_wins_last_{window}'].fillna(0)
        df[f'opponent_losses_last_{window}'] = df[f'opponent_losses_last_{window}'].fillna(0)

    logger.info(f"Game state features calculated for windows: {windows}")

    return df
