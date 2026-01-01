"""Calculate team-level rolling average features

This module focuses on TEAM defensive stats and OPPONENT offensive stats,
which are more predictive of goalie save totals than individual goalie performance.

Key insight: A goalie's saves depend more on:
1. How many shots their team allows (team defense quality)
2. How many shots the opponent generates (opponent offense quality)
3. Game situation (home/away, rest, back-to-back)

CRITICAL: All rolling features EXCLUDE the current game to prevent data leakage.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TeamRollingFeatureCalculator:
    """
    Calculate team-level rolling averages for defensive and offensive stats

    Focus on metrics that predict shot volume and quality:
    - Shots allowed/generated
    - Corsi (all shot attempts: shots + blocked + missed)
    - Fenwick (unblocked shot attempts: shots + missed)
    - Blocked shots
    - Hits, takeaways, giveaways
    - Goals for/against
    - Power play/penalty kill performance
    """

    def __init__(self, windows: List[int] = [3, 5, 10], min_games: int = 3):
        """
        Initialize team rolling feature calculator

        Args:
            windows: List of window sizes for rolling averages (e.g., [3, 5, 10])
            min_games: Minimum games required before calculating rolling stats
        """
        self.windows = windows
        self.min_games = min_games

    def calculate_team_rolling_features(
        self,
        team_games: pd.DataFrame,
        stat_columns: List[str],
        team_id_col: str = 'team_abbrev'
    ) -> pd.DataFrame:
        """
        Calculate rolling features for a team's defensive stats

        CRITICAL: Features for game i only use games 0 to i-1 (exclude current game)

        Args:
            team_games: DataFrame with team's games (sorted chronologically)
            stat_columns: List of column names to calculate rolling stats for
            team_id_col: Column name for team identifier

        Returns:
            DataFrame with original data plus rolling feature columns
        """
        if len(team_games) == 0:
            return team_games

        # Ensure sorted by date
        if 'game_date' in team_games.columns:
            team_games = team_games.sort_values('game_date').reset_index(drop=True)

        result_df = team_games.copy()

        # Collect all new columns
        new_columns = {}

        for stat in stat_columns:
            if stat not in team_games.columns:
                logger.warning(f"Column {stat} not found in team games")
                continue

            for window in self.windows:
                # CRITICAL: Use shift(1) to exclude current game
                # For game i, we calculate stats using games 0 to i-1
                shifted_values = team_games[stat].shift(1)

                # Arithmetic rolling average (most interpretable for team stats)
                rolling_col = f'{stat}_rolling_{window}'
                new_columns[rolling_col] = shifted_values.rolling(
                    window=window,
                    min_periods=min(self.min_games, window)
                ).mean()

        # Add all new columns at once
        if new_columns:
            new_cols_df = pd.DataFrame(new_columns, index=result_df.index)
            result_df = pd.concat([result_df, new_cols_df], axis=1)

        return result_df

    def prepare_team_game_log(
        self,
        games_df: pd.DataFrame,
        team_abbrev: str
    ) -> pd.DataFrame:
        """
        Extract a team's game log with both offensive and defensive stats

        Args:
            games_df: All games in the dataset
            team_abbrev: Team abbreviation (e.g., 'MIN', 'TOR')

        Returns:
            Team's game log sorted by date
        """
        # Filter to games where this team played
        team_games = games_df[games_df['team_abbrev'] == team_abbrev].copy()

        if len(team_games) == 0:
            logger.warning(f"No games found for team {team_abbrev}")
            return pd.DataFrame()

        # Sort by date
        team_games = team_games.sort_values('game_date').reset_index(drop=True)

        return team_games

    def merge_team_features_to_goalie_games(
        self,
        goalie_games: pd.DataFrame,
        team_rolling_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge team rolling features into goalie game log

        Args:
            goalie_games: Goalie's game-by-game stats
            team_rolling_features: Team's rolling averages

        Returns:
            Goalie games with team features merged in
        """
        # Merge on game_id and team_abbrev
        merged = goalie_games.merge(
            team_rolling_features,
            on=['game_id', 'team_abbrev', 'game_date'],
            how='left',
            suffixes=('', '_team_dup')
        )

        # Drop duplicate columns from merge
        dup_cols = [col for col in merged.columns if col.endswith('_team_dup')]
        merged = merged.drop(columns=dup_cols)

        return merged


def calculate_corsi_fenwick_from_boxscore(boxscore_data: Dict[str, Any], team_abbrev: str) -> Dict[str, float]:
    """
    Calculate Corsi and Fenwick metrics from play-by-play data

    Corsi = Shots + Blocked Shots + Missed Shots (all shot attempts)
    Fenwick = Shots + Missed Shots (unblocked shot attempts)

    Note: This requires play-by-play data. If not available, we'll approximate
    using available stats from boxscore.

    Args:
        boxscore_data: Parsed boxscore JSON
        team_abbrev: Team abbreviation

    Returns:
        Dictionary with Corsi/Fenwick metrics
    """
    # TODO: If play-by-play data is available, parse it here
    # For now, we'll use approximations from boxscore

    is_home = boxscore_data.get('homeTeam', {}).get('abbrev') == team_abbrev
    team_key = 'homeTeam' if is_home else 'awayTeam'
    opp_key = 'awayTeam' if is_home else 'homeTeam'

    team_data = boxscore_data.get(team_key, {})
    opp_data = boxscore_data.get(opp_key, {})

    # Approximate Corsi/Fenwick using available stats
    # Corsi For (CF) = team's shot attempts
    # Corsi Against (CA) = opponent's shot attempts

    team_shots = team_data.get('sog', 0)
    team_blocked = team_data.get('blocks', 0)  # Blocks BY this team (opponent's blocked shots)

    opp_shots = opp_data.get('sog', 0)
    opp_blocked = opp_data.get('blocks', 0)  # Blocks BY opponent (team's blocked shots)

    # Estimate missed shots as ~15% of total attempts (league average)
    # Corsi = Shots + Blocked + Missed
    # Since we don't have missed shots, approximate:
    # Corsi ≈ Shots + Blocked / 0.85 (assuming 15% are missed)

    # For the team (offensive):
    team_corsi_for = team_shots + opp_blocked  # Team's shots + shots opponent blocked
    team_fenwick_for = team_shots  # Just shots (missed not available)

    # Against the team (defensive):
    team_corsi_against = opp_shots + team_blocked  # Opponent's shots + shots team blocked
    team_fenwick_against = opp_shots  # Opponent's shots (unblocked)

    return {
        'team_corsi_for': team_corsi_for,
        'team_corsi_against': team_corsi_against,
        'team_fenwick_for': team_fenwick_for,
        'team_fenwick_against': team_fenwick_against,

        # Derived metrics
        'team_corsi_for_pct': team_corsi_for / max(team_corsi_for + team_corsi_against, 1),
        'team_shot_attempts_against': team_corsi_against,  # All attempts opponent made
        'team_unblocked_attempts_against': team_fenwick_against,  # Shots that got through
    }


def add_team_rolling_features_to_dataset(
    games_df: pd.DataFrame,
    windows: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Add team-level rolling features to the full dataset

    CRITICAL: This calculates rolling features within each team's game log,
    then merges back using game_id. The .shift(1) ensures that for each team's
    game i, we only use stats from games 0 to i-1 in THAT TEAM'S timeline.

    Process:
    1. For each unique team, create chronologically sorted game log
    2. Calculate rolling averages using .shift(1) (excludes current game)
    3. Merge back on game_id (safe because rolling features already shifted)
    4. Calculate opponent offensive stats the same way

    Args:
        games_df: Full dataset with all goalie games (must be sorted by game_date)
        windows: Rolling window sizes

    Returns:
        Dataset with team rolling features added
    """
    logger.info("Calculating team-level rolling features...")

    # CRITICAL: Ensure input is sorted by date globally
    games_df = games_df.sort_values('game_date').reset_index(drop=True)

    calculator = TeamRollingFeatureCalculator(windows=windows)

    # Team defensive stats to track (from goalie's team perspective)
    team_defensive_stats = [
        'opp_shots',  # Shots allowed (defensive metric)
        'opp_goals',  # Goals allowed
        'team_blocked_shots',  # Shots blocked by team
        'team_hits',  # Defensive aggression
        'opp_powerplay_opportunities',  # Penalties taken (defensive weakness)
    ]

    # Team offensive stats to track (for opponent analysis)
    team_offensive_stats = [
        'team_shots',  # Shots generated
        'team_goals',  # Goals scored
        'team_powerplay_goals',  # PP success
        'team_powerplay_opportunities',  # PP chances drawn
    ]

    # Get unique teams
    unique_teams = games_df['team_abbrev'].unique()

    # Store team rolling features for each team
    team_features_dict = {}

    logger.info(f"Calculating rolling features for {len(unique_teams)} teams...")

    for team in unique_teams:
        # Get this team's game log (sorted chronologically)
        # CRITICAL: We sort by game_date to ensure games are in temporal order
        team_games = games_df[games_df['team_abbrev'] == team].copy()
        team_games = team_games.sort_values('game_date').reset_index(drop=True)

        # Calculate rolling averages for DEFENSIVE stats (how many shots/goals team allows)
        # The .shift(1) inside calculate_team_rolling_features ensures:
        # - Team's game 1: rolling avg = NaN (no prior games)
        # - Team's game 2: rolling avg uses only game 1
        # - Team's game 6: rolling_5 uses games 1-5 (NOT including game 6)
        team_games = calculator.calculate_team_rolling_features(
            team_games,
            stat_columns=team_defensive_stats,
            team_id_col='team_abbrev'
        )

        # Calculate rolling averages for OFFENSIVE stats (for when this team is the opponent)
        team_games = calculator.calculate_team_rolling_features(
            team_games,
            stat_columns=team_offensive_stats,
            team_id_col='team_abbrev'
        )

        # Store the features we need (game_id + rolling columns)
        rolling_cols = [col for col in team_games.columns if '_rolling_' in col]
        keep_cols = ['game_id', 'team_abbrev', 'game_date'] + rolling_cols

        team_features_dict[team] = team_games[keep_cols]

    # Now merge team features back into main dataset
    # SAFE because rolling features already have .shift(1) applied
    result_df = games_df.copy()

    # Merge team's defensive features (goalie's team)
    logger.info("Merging team defensive features...")

    # Concatenate all team features into a single DataFrame
    all_team_features = []
    for team, features in team_features_dict.items():
        # Rename columns to make it clear these are TEAM defensive stats
        rename_map = {col: f'team_defense_{col}' for col in features.columns if '_rolling_' in col}
        features_renamed = features.rename(columns=rename_map)
        all_team_features.append(features_renamed)

    # Combine all teams into one DataFrame
    team_features_combined = pd.concat(all_team_features, ignore_index=True)

    # Single merge operation for all teams
    # This is SAFE because:
    # 1. Each team's game log was sorted chronologically
    # 2. Rolling features used .shift(1) to exclude current game
    # 3. game_id uniquely identifies a specific game
    # 4. The rolling feature for game_id X only uses data from games before X
    result_df = result_df.merge(
        team_features_combined,
        on=['game_id', 'team_abbrev', 'game_date'],
        how='left'
    )

    # Merge opponent's offensive features
    logger.info("Merging opponent offensive features...")

    # Concatenate all opponent features
    all_opp_features = []
    for team, features in team_features_dict.items():
        # Rename columns to make it clear these are OPPONENT offensive stats
        rename_map = {col: f'opp_offense_{col}' for col in features.columns if '_rolling_' in col and 'team_' in col}
        features_renamed = features.rename(columns=rename_map)

        # Rename team_abbrev to opponent_team for joining
        features_renamed = features_renamed.rename(columns={'team_abbrev': 'opponent_team'})
        all_opp_features.append(features_renamed)

    # Combine all opponent features into one DataFrame
    opp_features_combined = pd.concat(all_opp_features, ignore_index=True)

    # Single merge operation for all opponents
    result_df = result_df.merge(
        opp_features_combined,
        on=['game_id', 'opponent_team', 'game_date'],
        how='left'
    )

    logger.info(f"Added {len([c for c in result_df.columns if 'team_defense_' in c or 'opp_offense_' in c])} team/opponent features")

    return result_df
