"""
Clean feature engineering pipeline for classification model

This script creates a SIMPLE, CLEAN feature set with NO data leakage.

Key principles:
1. Only calculate features ONCE (no duplicates)
2. All rolling features use .shift(1) to exclude current game
3. Sort by date BEFORE calculating rolling features
4. No complex merge operations that create _x/_y columns
5. Easy to verify correctness

Output: data/processed/clean_training_data.parquet
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_boxscores(boxscores_dir):
    """Load all boxscore data"""
    logger.info(f"Loading boxscores from {boxscores_dir}")

    all_games = []

    for file_path in tqdm(sorted(boxscores_dir.glob("*.json")), desc="Loading games"):
        try:
            with open(file_path, 'r') as f:
                game_data = json.load(f)

            game_id = game_data.get('id')
            game_date = game_data.get('gameDate', '')[:10]  # YYYY-MM-DD
            season = game_data.get('season')

            # Process each team (home and away)
            for team_key in ['homeTeam', 'awayTeam']:
                is_home = (team_key == 'homeTeam')
                team_data = game_data.get(team_key, {})
                team_abbrev = team_data.get('abbrev', '')

                opp_key = 'awayTeam' if is_home else 'homeTeam'
                opp_data = game_data.get(opp_key, {})
                opponent_team = opp_data.get('abbrev', '')

                # Get goalies for this team
                goalies = game_data.get('playerByGameStats', {}).get(team_key, {}).get('goalies', [])

                for goalie in goalies:
                    # Only process starters
                    if goalie.get('starter') != 1:
                        continue

                    goalie_id = goalie.get('playerId')
                    if not goalie_id:
                        continue

                    # Parse situation-specific stats
                    def parse_fraction(frac_str):
                        """Parse '27/29' format to (saves, shots)"""
                        if not frac_str or frac_str == '0/0':
                            return 0, 0
                        try:
                            saves, shots = frac_str.split('/')
                            return int(saves), int(shots)
                        except:
                            return 0, 0

                    es_saves, es_shots = parse_fraction(goalie.get('evenStrengthShotsAgainst', '0/0'))
                    pp_saves, pp_shots = parse_fraction(goalie.get('powerPlayShotsAgainst', '0/0'))
                    sh_saves, sh_shots = parse_fraction(goalie.get('shorthandedShotsAgainst', '0/0'))

                    # Extract goalie stats
                    game_record = {
                        'game_id': game_id,
                        'game_date': pd.to_datetime(game_date),
                        'season': season,
                        'goalie_id': goalie_id,
                        'team_abbrev': team_abbrev,
                        'opponent_team': opponent_team,
                        'is_home': int(is_home),

                        # Target variable (what we're predicting)
                        'saves': goalie.get('saves', 0),

                        # Goalie performance (CURRENT GAME - will be excluded from features)
                        'shots_against': goalie.get('shotsAgainst', 0),
                        'goals_against': goalie.get('goalsAgainst', 0),
                        'save_percentage': goalie.get('savePctg', 0.0) if goalie.get('savePctg') else 0.0,
                        'toi': goalie.get('toi', '0:00'),

                        # Situation-specific goalie stats (CURRENT GAME - will be excluded)
                        'even_strength_saves': es_saves,
                        'even_strength_shots_against': es_shots,
                        'even_strength_goals_against': goalie.get('evenStrengthGoalsAgainst', 0),
                        'power_play_saves': pp_saves,
                        'power_play_shots_against': pp_shots,
                        'power_play_goals_against': goalie.get('powerPlayGoalsAgainst', 0),
                        'short_handed_saves': sh_saves,
                        'short_handed_shots_against': sh_shots,
                        'short_handed_goals_against': goalie.get('shorthandedGoalsAgainst', 0),

                        # Team stats (CURRENT GAME - will be excluded)
                        'team_goals': team_data.get('score', 0),
                        'team_shots': team_data.get('sog', 0),
                        'opp_goals': opp_data.get('score', 0),
                        'opp_shots': opp_data.get('sog', 0),
                    }

                    all_games.append(game_record)

        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            continue

    df = pd.DataFrame(all_games)
    logger.info(f"Loaded {len(df)} goalie-game records from {df['game_id'].nunique()} games")

    return df


def calculate_rolling_features(df):
    """
    Calculate rolling features with NO data leakage

    CRITICAL: Uses .shift(1) so game N only sees games 1 to N-1
    """
    logger.info("Calculating rolling features (with shift to prevent leakage)...")

    # Stats to calculate rolling features for
    stats = [
        'saves', 'shots_against', 'goals_against', 'save_percentage',
        'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
        'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
        'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against'
    ]
    windows = [3, 5, 10]

    # CRITICAL: Sort by goalie and date FIRST
    df = df.sort_values(['goalie_id', 'game_date']).reset_index(drop=True)

    # Calculate for each stat and window
    for stat in stats:
        for window in windows:
            # Rolling MEAN
            df[f'{stat}_rolling_{window}'] = df.groupby('goalie_id')[stat].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )

            # Rolling STD (volatility)
            df[f'{stat}_rolling_std_{window}'] = df.groupby('goalie_id')[stat].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
            )

    logger.info(f"Created {len(stats) * len(windows) * 2} rolling features")

    return df


def calculate_opponent_rolling_features(df):
    """
    Calculate rolling features for opponent teams

    How often does this opponent score? How many shots do they take?
    """
    logger.info("Calculating opponent offensive rolling features...")

    windows = [5, 10]

    # For each team, calculate their OFFENSIVE stats (when they're the opponent)
    opponent_features = {}

    for team in df['opponent_team'].unique():
        # Get games where this team played (as either home or away)
        team_games = df[df['team_abbrev'] == team].sort_values('game_date').copy()

        if len(team_games) == 0:
            continue

        # Calculate rolling averages of their offensive output
        team_features = pd.DataFrame({
            'game_id': team_games['game_id'],
            'opponent_team': team
        })

        for window in windows:
            # Goals scored (as offense)
            team_features[f'opp_goals_rolling_{window}'] = team_games.groupby('team_abbrev')['team_goals'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )

            # Shots taken (as offense)
            team_features[f'opp_shots_rolling_{window}'] = team_games.groupby('team_abbrev')['team_shots'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )

        opponent_features[team] = team_features

    # Combine all opponent features
    all_opp = pd.concat(opponent_features.values(), ignore_index=True)

    # Merge back to main dataset
    df = df.merge(all_opp, on=['game_id', 'opponent_team'], how='left')

    logger.info(f"Added {len(windows) * 2} opponent rolling features")

    return df


def calculate_team_defensive_features(df):
    """
    Calculate rolling features for team defensive performance

    How well does the goalie's team defend? (shots against, goals against allowed)
    This is different from goalie stats - it's about the team's overall defensive ability
    """
    logger.info("Calculating team defensive rolling features...")

    windows = [5, 10]

    # For each team, calculate their DEFENSIVE stats
    team_features = {}

    for team in df['team_abbrev'].unique():
        # Get games where this team played (to calculate their defensive performance)
        team_games = df[df['team_abbrev'] == team].sort_values('game_date').copy()

        if len(team_games) == 0:
            continue

        # Calculate rolling averages of their defensive performance
        features = pd.DataFrame({
            'game_id': team_games['game_id'],
            'team_abbrev': team
        })

        for window in windows:
            # Goals against (team defense allows)
            features[f'team_goals_against_rolling_{window}'] = team_games.groupby('team_abbrev')['opp_goals'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )

            # Shots against (team defense allows)
            features[f'team_shots_against_rolling_{window}'] = team_games.groupby('team_abbrev')['opp_shots'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )

        team_features[team] = features

    # Combine all team features
    all_team = pd.concat(team_features.values(), ignore_index=True)

    # Merge back to main dataset
    df = df.merge(all_team, on=['game_id', 'team_abbrev'], how='left')

    logger.info(f"Added {len(windows) * 2} team defensive rolling features")

    return df


def calculate_rest_features(df):
    """Calculate days rest for goalie"""
    logger.info("Calculating rest/fatigue features...")

    df = df.sort_values(['goalie_id', 'game_date']).reset_index(drop=True)

    # Days since last game
    df['goalie_days_rest'] = df.groupby('goalie_id')['game_date'].diff().dt.days
    df['goalie_days_rest'] = df['goalie_days_rest'].fillna(7)  # First game = assume well rested

    # Back to back indicator
    df['goalie_is_back_to_back'] = (df['goalie_days_rest'] <= 1).astype(int)

    logger.info("Added 2 rest/fatigue features")

    return df


def exclude_current_game_features(df):
    """
    Drop all features that contain information about the CURRENT game

    These are only useful for training the target, not for prediction
    """
    logger.info("Identifying current-game features to exclude from training...")

    # Features that are ONLY known AFTER the game is played
    current_game_cols = [
        'saves',  # This is our TARGET
        'shots_against',
        'goals_against',
        'save_percentage',
        'toi',
        'team_goals',
        'team_shots',
        'opp_goals',
        'opp_shots',
    ]

    # Verify these exist
    exclude_cols = [c for c in current_game_cols if c in df.columns]

    logger.info(f"Marked {len(exclude_cols)} current-game columns for exclusion")
    logger.info(f"Columns: {exclude_cols}")

    return exclude_cols


def create_clean_features():
    """Main pipeline"""
    logger.info("=" * 70)
    logger.info("CLEAN FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 70)

    # 1. Load data
    boxscores_dir = Path('data/raw/boxscores')
    df = load_boxscores(boxscores_dir)

    # 2. Calculate rolling features for goalies
    df = calculate_rolling_features(df)

    # 3. Calculate opponent offensive patterns
    df = calculate_opponent_rolling_features(df)

    # 4. Calculate team defensive patterns
    df = calculate_team_defensive_features(df)

    # 5. Calculate rest/fatigue
    df = calculate_rest_features(df)

    # 6. Fill NaN values (from first few games with no rolling history)
    logger.info("Filling NaN values in rolling features...")
    rolling_cols = [c for c in df.columns if 'rolling' in c]
    df[rolling_cols] = df[rolling_cols].fillna(0)

    # 6. Save
    output_path = Path('data/processed/clean_training_data.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving to {output_path}")
    df.to_parquet(output_path, index=False)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    logger.info(f"Unique goalies: {df['goalie_id'].nunique()}")
    logger.info(f"Unique games: {df['game_id'].nunique()}")

    # Identify current-game columns
    exclude_cols = exclude_current_game_features(df)
    feature_cols = [c for c in df.columns if c not in exclude_cols and c not in ['game_id', 'game_date', 'season', 'goalie_id', 'team_abbrev', 'opponent_team']]

    logger.info(f"\nUsable features: {len(feature_cols)}")
    logger.info(f"Target variable: saves")

    # Save metadata
    metadata = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'feature_columns': feature_cols,
        'exclude_columns': exclude_cols,
        'date_range': {
            'min': str(df['game_date'].min()),
            'max': str(df['game_date'].max())
        }
    }

    metadata_path = Path('data/processed/clean_features_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to {metadata_path}")

    return df


if __name__ == "__main__":
    create_clean_features()
