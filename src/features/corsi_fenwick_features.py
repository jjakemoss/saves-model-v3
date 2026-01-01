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

    CRITICAL: Calculate rolling features separately for each team to prevent data leakage.
    Similar approach to team_rolling_features.py - create team game logs, calculate rolling
    features with proper chronological ordering, then merge back.

    Args:
        games_df: DataFrame with Corsi/Fenwick features
        windows: Rolling window sizes

    Returns:
        DataFrame with Corsi/Fenwick rolling features
    """
    logger.info("Calculating Corsi/Fenwick rolling features...")

    # Ensure sorted by date
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

    # Get unique teams
    unique_teams = df['team_abbrev'].unique()

    # Store rolling features for each team
    team_corsi_features = {}

    logger.info(f"Calculating Corsi/Fenwick rolling features for {len(unique_teams)} teams...")

    for team in unique_teams:
        # Get this team's games in chronological order
        team_games = df[df['team_abbrev'] == team].copy()
        team_games = team_games.sort_values('game_date').reset_index(drop=True)

        # Calculate rolling features for each stat
        rolling_data = {
            'game_id': team_games['game_id'].values,
            'team_abbrev': [team] * len(team_games)  # Include team for merge key
        }

        for stat in corsi_stats:
            for window in windows:
                # CRITICAL: Use shift(1) to exclude current game
                shifted_values = team_games[stat].shift(1)
                rolling_avg = shifted_values.rolling(window=window, min_periods=1).mean()

                col_name = f'{stat}_rolling_{window}'
                rolling_data[col_name] = rolling_avg.values

        # Store as DataFrame
        team_corsi_features[team] = pd.DataFrame(rolling_data)

    # Combine all team rolling features
    all_team_features = []
    for team, features in team_corsi_features.items():
        all_team_features.append(features)

    team_features_combined = pd.concat(all_team_features, ignore_index=True)

    # Merge back into main dataset - use game_id AND team_abbrev to avoid duplicates
    df = df.merge(
        team_features_combined,
        on=['game_id', 'team_abbrev'],
        how='left',
        suffixes=('', '_drop')
    )

    # Drop any duplicate columns from merge
    df = df[[col for col in df.columns if not col.endswith('_drop')]]

    # Also calculate opponent Corsi/Fenwick rolling features
    opponent_corsi_features = {}

    for team in unique_teams:
        # Get games where this team was the OPPONENT
        opp_games = df[df['opponent_team'] == team].copy()
        opp_games = opp_games.sort_values('game_date').reset_index(drop=True)

        # Calculate rolling features
        rolling_data = {
            'game_id': opp_games['game_id'].values,
            'opponent_team': [team] * len(opp_games)  # Include opponent_team for merge key
        }

        for stat in corsi_stats:
            for window in windows:
                # Use shift(1) to exclude current game
                shifted_values = opp_games[stat].shift(1)
                rolling_avg = shifted_values.rolling(window=window, min_periods=1).mean()

                col_name = f'opponent_{stat}_rolling_{window}'
                rolling_data[col_name] = rolling_avg.values

        opponent_corsi_features[team] = pd.DataFrame(rolling_data)

    # Combine opponent features
    all_opp_features = []
    for team, features in opponent_corsi_features.items():
        all_opp_features.append(features)

    opp_features_combined = pd.concat(all_opp_features, ignore_index=True)

    # Merge back - use game_id AND opponent_team to avoid duplicates
    df = df.merge(
        opp_features_combined,
        on=['game_id', 'opponent_team'],
        how='left',
        suffixes=('', '_drop')
    )

    # Drop duplicates
    df = df[[col for col in df.columns if not col.endswith('_drop')]]

    logger.info(f"Corsi/Fenwick rolling features calculated for windows: {windows}")
    logger.info(f"  - Added {len(corsi_stats) * len(windows) * 2} features")

    return df
