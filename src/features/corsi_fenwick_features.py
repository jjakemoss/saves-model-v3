"""Calculate Corsi and Fenwick shot suppression metrics

Corsi and Fenwick are advanced hockey analytics metrics that measure shot attempts,
which are more predictive of future performance than just shots on goal.

Definitions:
- Corsi For (CF): All shot attempts FOR (shots + blocked + missed)
- Corsi Against (CA): All shot attempts AGAINST
- Fenwick For (FF): Unblocked shot attempts FOR (shots + missed, excludes blocked)
- Fenwick Against (FA): Unblocked shot attempts AGAINST
- Corsi For % (CF%): CF / (CF + CA) - possession metric

For save prediction:
- Team Corsi Against predicts shot volume better than shots on goal
- High Corsi Against = team allows more shot attempts = more saves for goalie
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


def calculate_corsi_fenwick_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Corsi and Fenwick metrics from game data

    Approximation from available data:
    - Corsi For = team_shots + opp_blocked_shots
    - Corsi Against = opp_shots + team_blocked_shots
    - Fenwick For ≈ team_shots (missed shots not available)
    - Fenwick Against ≈ opp_shots

    Args:
        games_df: DataFrame with team_shots, opp_shots, team_blocked_shots

    Returns:
        DataFrame with Corsi/Fenwick features added
    """
    logger.info("Calculating Corsi and Fenwick features...")

    df = games_df.copy()

    # Check required columns exist
    required_cols = ['team_shots', 'opp_shots', 'team_blocked_shots']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.warning(f"Missing columns for Corsi calculation: {missing}")
        logger.warning("Skipping Corsi/Fenwick features")
        return df

    # Note: We need opponent's blocked shots to calculate team's Corsi For
    # Since we don't have this directly, we'll create a mapping

    # Create a temporary opponent blocked shots column by looking up the opponent's game
    # For each game, find the opponent's team_blocked_shots in that same game
    df['opp_blocked_shots'] = 0  # Initialize

    # Group by game_id and fill in opponent blocked shots
    for game_id in df['game_id'].unique():
        game_rows = df[df['game_id'] == game_id]

        if len(game_rows) == 2:  # Should have both goalies in the same game
            # Get both teams' blocked shots
            team_1_idx = game_rows.index[0]
            team_2_idx = game_rows.index[1]

            team_1_blocks = game_rows.loc[team_1_idx, 'team_blocked_shots']
            team_2_blocks = game_rows.loc[team_2_idx, 'team_blocked_shots']

            # Team 1's opponent blocked shots = Team 2's blocked shots
            df.at[team_1_idx, 'opp_blocked_shots'] = team_2_blocks
            df.at[team_2_idx, 'opp_blocked_shots'] = team_1_blocks

    # Calculate Corsi metrics
    # OFFENSIVE (Corsi For) - how many shot attempts team generates
    df['team_corsi_for'] = df['team_shots'] + df['opp_blocked_shots']

    # DEFENSIVE (Corsi Against) - how many shot attempts team allows
    df['team_corsi_against'] = df['opp_shots'] + df['team_blocked_shots']

    # Fenwick (unblocked shot attempts)
    df['team_fenwick_for'] = df['team_shots']  # Approximation (no missed shot data)
    df['team_fenwick_against'] = df['opp_shots']

    # Corsi For Percentage (possession metric)
    total_corsi = df['team_corsi_for'] + df['team_corsi_against']
    df['team_corsi_for_pct'] = df['team_corsi_for'] / total_corsi.replace(0, 1)

    # Fenwick For Percentage
    total_fenwick = df['team_fenwick_for'] + df['team_fenwick_against']
    df['team_fenwick_for_pct'] = df['team_fenwick_for'] / total_fenwick.replace(0, 1)

    logger.info("Corsi and Fenwick features calculated")
    logger.info(f"  - Average Corsi For: {df['team_corsi_for'].mean():.1f}")
    logger.info(f"  - Average Corsi Against: {df['team_corsi_against'].mean():.1f}")
    logger.info(f"  - Average Corsi For %: {df['team_corsi_for_pct'].mean():.3f}")

    return df


def add_corsi_rolling_features(
    games_df: pd.DataFrame,
    windows: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Add rolling averages for Corsi/Fenwick metrics

    This is similar to team_rolling_features but specifically for Corsi/Fenwick.
    We calculate these separately to allow for different window sizes.

    Args:
        games_df: DataFrame with Corsi/Fenwick features
        windows: Rolling window sizes

    Returns:
        DataFrame with Corsi/Fenwick rolling features
    """
    logger.info("Calculating Corsi/Fenwick rolling features...")

    df = games_df.sort_values('game_date').reset_index(drop=True)

    corsi_stats = [
        'team_corsi_for',
        'team_corsi_against',
        'team_corsi_for_pct',
        'team_fenwick_for',
        'team_fenwick_against',
        'team_fenwick_for_pct'
    ]

    # Only use stats that exist
    corsi_stats = [stat for stat in corsi_stats if stat in df.columns]

    if not corsi_stats:
        logger.warning("No Corsi/Fenwick stats found, skipping rolling features")
        return df

    # Calculate rolling features for each team
    for stat in corsi_stats:
        for window in windows:
            # Team rolling average (using shift(1) to exclude current game)
            col_name = f'{stat}_rolling_{window}'
            df[col_name] = df.groupby('team_abbrev')[stat].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )

    # Also calculate opponent Corsi For (which becomes our Corsi Against perspective)
    for stat in corsi_stats:
        for window in windows:
            col_name = f'opponent_{stat}_rolling_{window}'
            df[col_name] = df.groupby('opponent_team')[stat].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )

    logger.info(f"Corsi/Fenwick rolling features calculated for windows: {windows}")
    logger.info(f"  - Added {len(corsi_stats) * len(windows) * 2} features")

    return df
