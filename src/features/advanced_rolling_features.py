"""Calculate advanced rolling features for shot quality and opponent offense

This module adds:
1. Shot quality by danger level rolling features (goalie & team)
2. Opponent shooting percentage rolling features
3. Expected Goals (xG) rolling features

These are high-value features that better predict save totals than basic metrics.
"""

import numpy as np
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)


def add_shot_quality_rolling_features(
    games_df: pd.DataFrame,
    windows: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Add rolling averages for shot quality metrics by danger level

    CRITICAL: Uses .shift(1) to exclude current game from rolling calculations

    Args:
        games_df: DataFrame with shot quality features (already extracted from play-by-play)
        windows: Rolling window sizes

    Returns:
        DataFrame with shot quality rolling features added
    """
    logger.info("Calculating shot quality rolling features...")

    df = games_df.copy()

    # Shot quality stats to track (for goalies)
    goalie_shot_quality_stats = [
        'high_danger_save_pct',
        'mid_danger_save_pct',
        'low_danger_save_pct',
        'total_xg_against',
        'high_danger_xg_against',
        'rebound_rate',
        'dangerous_rebound_pct',
    ]

    # Only use stats that exist
    goalie_shot_quality_stats = [s for s in goalie_shot_quality_stats if s in df.columns]

    if not goalie_shot_quality_stats:
        logger.warning("No shot quality stats found, skipping shot quality rolling features")
        return df

    # Calculate rolling features for each goalie
    logger.info(f"Calculating goalie shot quality rolling features for {len(goalie_shot_quality_stats)} stats...")

    for goalie_id, goalie_games in df.groupby('goalie_id'):
        # Sort by date
        goalie_games = goalie_games.sort_values('game_date').reset_index(drop=True)

        for stat in goalie_shot_quality_stats:
            for window in windows:
                # CRITICAL: Use shift(1) to exclude current game
                shifted_values = goalie_games[stat].shift(1)
                rolling_avg = shifted_values.rolling(window=window, min_periods=1).mean()

                col_name = f'{stat}_rolling_{window}'
                df.loc[goalie_games.index, col_name] = rolling_avg.values

    logger.info(f"Added {len(goalie_shot_quality_stats) * len(windows)} goalie shot quality rolling features")

    return df


def add_team_shot_quality_rolling_features(
    games_df: pd.DataFrame,
    windows: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Add team-level shot quality rolling features

    These track how the TEAM's defense performs at preventing high-danger chances

    CRITICAL: Calculate separately for each team, then merge back

    Args:
        games_df: DataFrame with team-level shot quality aggregates
        windows: Rolling window sizes

    Returns:
        DataFrame with team shot quality rolling features
    """
    logger.info("Calculating team shot quality rolling features...")

    df = games_df.sort_values('game_date').reset_index(drop=True)

    # Team defensive shot quality stats (how well team defense limits danger)
    team_shot_quality_stats = [
        'high_danger_shots_against',
        'mid_danger_shots_against',
        'low_danger_shots_against',
        'total_xg_against',
        'high_danger_xg_against',
    ]

    # Only use stats that exist
    team_shot_quality_stats = [s for s in team_shot_quality_stats if s in df.columns]

    if not team_shot_quality_stats:
        logger.warning("No team shot quality stats found, skipping")
        return df

    # Get unique teams
    unique_teams = df['team_abbrev'].unique()

    # Store rolling features for each team
    team_shot_quality_features = {}

    logger.info(f"Calculating team shot quality rolling features for {len(unique_teams)} teams...")

    for team in unique_teams:
        # Get this team's games in chronological order
        team_games = df[df['team_abbrev'] == team].copy()
        team_games = team_games.sort_values('game_date').reset_index(drop=True)

        # Calculate rolling features for each stat
        rolling_data = {
            'game_id': team_games['game_id'].values,
            'team_abbrev': [team] * len(team_games)
        }

        for stat in team_shot_quality_stats:
            for window in windows:
                # CRITICAL: Use shift(1) to exclude current game
                shifted_values = team_games[stat].shift(1)
                rolling_avg = shifted_values.rolling(window=window, min_periods=1).mean()

                col_name = f'team_defense_{stat}_rolling_{window}'
                rolling_data[col_name] = rolling_avg.values

        # Store as DataFrame
        team_shot_quality_features[team] = pd.DataFrame(rolling_data)

    # Combine all team rolling features
    all_team_features = []
    for team, features in team_shot_quality_features.items():
        all_team_features.append(features)

    team_features_combined = pd.concat(all_team_features, ignore_index=True)

    # Merge back into main dataset
    # CRITICAL: Use suffixes to prevent _x/_y column creation
    df = df.merge(
        team_features_combined,
        on=['game_id', 'team_abbrev'],
        how='left',
        suffixes=('_DROP', '')  # Keep new features, drop old duplicates
    )

    # Drop duplicate columns
    drop_cols = [col for col in df.columns if col.endswith('_DROP')]
    if drop_cols:
        logger.warning(f"Dropping {len(drop_cols)} duplicate columns from team shot quality merge")
        df = df.drop(columns=drop_cols)

    logger.info(f"Added {len(team_shot_quality_stats) * len(windows)} team shot quality rolling features")

    return df


def add_opponent_shooting_pct_features(
    games_df: pd.DataFrame,
    windows: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Add opponent shooting percentage rolling features

    CRITICAL FEATURE: Opponent shooting skill directly affects saves
    - High shooting %: More shots convert to goals → Fewer saves
    - Low shooting %: Shots miss net → More saves

    Example: 30 shots at 10% = 3 goals, 27 saves
             30 shots at 15% = 4.5 goals, 25.5 saves

    Args:
        games_df: DataFrame with team goals and shots
        windows: Rolling window sizes

    Returns:
        DataFrame with opponent shooting % rolling features
    """
    logger.info("Calculating opponent shooting percentage features...")

    df = games_df.sort_values('game_date').reset_index(drop=True)

    # First, calculate shooting % for each team as the OFFENSIVE team
    # (This will become the opponent's shooting % when they play against you)

    unique_teams = df['team_abbrev'].unique()

    # Store features for each team
    team_shooting_features = {}

    logger.info(f"Calculating shooting % for {len(unique_teams)} teams...")

    for team in unique_teams:
        # Get games where this team was the OFFENSIVE team
        team_games = df[df['team_abbrev'] == team].copy()
        team_games = team_games.sort_values('game_date').reset_index(drop=True)

        # Calculate shooting percentage (goals / shots)
        # Use team_goals and team_shots (offensive stats)
        if 'team_goals' in df.columns and 'team_shots' in df.columns:
            team_games['shooting_pct'] = team_games['team_goals'] / team_games['team_shots'].replace(0, 1)
        else:
            logger.warning(f"Missing team_goals or team_shots columns")
            continue

        # Calculate 5v5 shooting % if available
        if 'even_strength_goals' in df.columns and 'even_strength_shots' in df.columns:
            team_games['shooting_pct_5v5'] = (
                team_games['even_strength_goals'] / team_games['even_strength_shots'].replace(0, 1)
            )

        # Calculate rolling averages
        rolling_data = {
            'game_id': team_games['game_id'].values,
            'team_abbrev': [team] * len(team_games)
        }

        for window in windows:
            # Overall shooting %
            shifted_values = team_games['shooting_pct'].shift(1)
            rolling_avg = shifted_values.rolling(window=window, min_periods=1).mean()
            rolling_data[f'shooting_pct_overall_rolling_{window}'] = rolling_avg.values

            # 5v5 shooting % (if available)
            if 'shooting_pct_5v5' in team_games.columns:
                shifted_values_5v5 = team_games['shooting_pct_5v5'].shift(1)
                rolling_avg_5v5 = shifted_values_5v5.rolling(window=window, min_periods=1).mean()
                rolling_data[f'shooting_pct_5v5_rolling_{window}'] = rolling_avg_5v5.values

        team_shooting_features[team] = pd.DataFrame(rolling_data)

    # Now merge as OPPONENT features
    # When team A plays team B, we want team B's shooting % as a feature for team A's goalie

    all_opp_features = []
    for team, features in team_shooting_features.items():
        # Rename team_abbrev to opponent_team for joining
        features_renamed = features.rename(columns={'team_abbrev': 'opponent_team'})

        # Rename columns to indicate these are opponent stats
        rename_map = {col: f'opponent_{col}' for col in features_renamed.columns
                     if 'rolling' in col}
        features_renamed = features_renamed.rename(columns=rename_map)

        all_opp_features.append(features_renamed)

    # Combine all opponent features
    opp_features_combined = pd.concat(all_opp_features, ignore_index=True)

    # Merge back into main dataset
    # CRITICAL: Use suffixes to prevent _x/_y column creation
    df = df.merge(
        opp_features_combined,
        on=['game_id', 'opponent_team'],
        how='left',
        suffixes=('_DROP', '')  # Keep new features, drop old duplicates
    )

    # Drop duplicates
    drop_cols = [col for col in df.columns if col.endswith('_DROP')]
    if drop_cols:
        logger.warning(f"Dropping {len(drop_cols)} duplicate columns from opponent shooting % merge")
        df = df.drop(columns=drop_cols)

    logger.info(f"Added {len(windows) * 2} opponent shooting percentage features")

    return df


def add_all_advanced_rolling_features(
    games_df: pd.DataFrame,
    windows: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Orchestrate all advanced rolling feature calculations

    Args:
        games_df: DataFrame with base features
        windows: Rolling window sizes

    Returns:
        DataFrame with all advanced rolling features added
    """
    logger.info("=== Adding Advanced Rolling Features ===")

    df = games_df.copy()

    # 1. Shot quality rolling features (goalie-level)
    df = add_shot_quality_rolling_features(df, windows=windows)

    # 2. Team shot quality rolling features
    df = add_team_shot_quality_rolling_features(df, windows=windows)

    # 3. Opponent shooting percentage features
    df = add_opponent_shooting_pct_features(df, windows=[5, 10])

    logger.info("=== Advanced Rolling Features Complete ===")

    return df
