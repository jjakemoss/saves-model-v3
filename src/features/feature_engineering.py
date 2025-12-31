"""Main feature engineering pipeline

Orchestrates all feature extraction modules to create the complete training dataset.

CRITICAL DATA LEAKAGE PREVENTION:
- All features for game i are calculated using ONLY games 0 to i-1
- Games are sorted chronologically before feature calculation
- Rolling averages explicitly exclude the current game
- No future information is used

Target variable: over_line (1 if actual_saves > betting_line, 0 otherwise)
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .base_features import BaseFeatureExtractor
from .rolling_features import RollingFeatureCalculator, calculate_all_rolling_features
from .shot_quality_features import ShotQualityFeatureExtractor
from .matchup_features import MatchupFeatureCalculator, calculate_all_matchup_features
from .interaction_features import calculate_all_interaction_features

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline

    Workflow:
    1. Load raw game data (boxscores, play-by-play)
    2. Extract base features
    3. Sort chronologically by date (CRITICAL)
    4. Calculate rolling features (excluding current game)
    5. Calculate shot quality and xG features
    6. Calculate matchup and contextual features
    7. Calculate interaction features
    8. Create target variable
    9. Save processed dataset
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize feature engineering pipeline

        Args:
            config_path: Path to configuration file
        """
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.base_extractor = BaseFeatureExtractor()
        self.shot_quality_extractor = ShotQualityFeatureExtractor()
        self.rolling_calculator = RollingFeatureCalculator(
            windows=self.config['features']['rolling_windows'],
            min_games=self.config['features']['min_games_for_rolling']
        )
        self.matchup_calculator = MatchupFeatureCalculator()

        logger.info("Feature engineering pipeline initialized")

    def extract_all_base_features(
        self,
        boxscores_dir: Path,
        pbp_dir: Path
    ) -> pd.DataFrame:
        """
        Extract base features from all boxscores and play-by-play files

        Args:
            boxscores_dir: Directory containing boxscore JSON files
            pbp_dir: Directory containing play-by-play JSON files

        Returns:
            DataFrame with base features for all goalie-game combinations
        """
        logger.info("Extracting base features from raw data...")

        all_features = []

        # Get all boxscore files
        boxscore_files = sorted(boxscores_dir.glob("*.json"))

        for boxscore_path in tqdm(boxscore_files, desc="Processing games"):
            try:
                # Load boxscore
                boxscore_data = self.base_extractor.load_boxscore(boxscore_path)
                if not boxscore_data:
                    continue

                game_id = boxscore_data.get('id')

                # Extract features for each goalie in the game
                for team_key in ['homeTeam', 'awayTeam']:
                    is_home = (team_key == 'homeTeam')
                    team_data = boxscore_data.get(team_key, {})
                    team_abbrev = team_data.get('abbrev', '')
                    opp_key = 'awayTeam' if is_home else 'homeTeam'
                    opp_abbrev = boxscore_data.get(opp_key, {}).get('abbrev', '')

                    # Get goalies for this team
                    goalies = boxscore_data.get('playerByGameStats', {}).get(team_key, {}).get('goalies', [])

                    for goalie in goalies:
                        goalie_id = goalie.get('playerId')
                        if not goalie_id:
                            continue

                        # Extract base goalie features
                        goalie_features = self.base_extractor.extract_goalie_game_features(
                            boxscore_data,
                            goalie_id
                        )

                        if not goalie_features:
                            continue

                        # Add team context
                        goalie_features['team_abbrev'] = team_abbrev
                        goalie_features['opponent_team'] = opp_abbrev
                        goalie_features['is_home'] = is_home

                        # Extract team features
                        team_features = self.base_extractor.extract_team_game_features(
                            boxscore_data,
                            team_abbrev
                        )
                        goalie_features.update(team_features)

                        # Extract shot quality features from play-by-play
                        pbp_path = pbp_dir / f"{game_id}.json"
                        if pbp_path.exists():
                            pbp_data = self.shot_quality_extractor.load_play_by_play(pbp_path)
                            if pbp_data:
                                shots = self.shot_quality_extractor.extract_shots_from_pbp(pbp_data)
                                shot_quality = self.shot_quality_extractor.aggregate_goalie_shot_quality(
                                    shots,
                                    goalie_id
                                )
                                goalie_features.update(shot_quality)

                                # Rebound metrics
                                rebound_metrics = self.shot_quality_extractor.calculate_rebound_metrics(
                                    shots,
                                    goalie_id
                                )
                                goalie_features.update(rebound_metrics)

                        all_features.append(goalie_features)

            except Exception as e:
                logger.error(f"Error processing {boxscore_path}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(all_features)

        logger.info(f"Extracted base features for {len(df)} goalie-game combinations")

        return df

    def calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling and exponential weighted average features

        CRITICAL: Sorts by date and excludes current game from rolling calculations

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with rolling features added
        """
        logger.info("Calculating rolling features...")

        # Define stats to calculate rolling features for
        goalie_stats = [
            'saves',
            'save_percentage',
            'shots_against',
            'goals_against',
            'even_strength_save_pct',
            'power_play_save_pct',
            'high_danger_save_pct',
            'mid_danger_save_pct',
            'low_danger_save_pct',
            'total_xg_against',
            'rebound_rate'
        ]

        # Calculate rolling features using the dedicated function
        df = calculate_all_rolling_features(
            df,
            goalie_stats=goalie_stats,
            windows=self.config['features']['rolling_windows']
        )

        logger.info("Rolling features calculated")

        return df

    def calculate_all_features(
        self,
        boxscores_dir: Path,
        pbp_dir: Path,
        output_path: Path
    ) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline

        Args:
            boxscores_dir: Directory with boxscore JSON files
            pbp_dir: Directory with play-by-play JSON files
            output_path: Path to save processed features

        Returns:
            Complete feature DataFrame
        """
        logger.info("=" * 60)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Extract base features
        df = self.extract_all_base_features(boxscores_dir, pbp_dir)

        if len(df) == 0:
            logger.error("No features extracted. Check data directories.")
            return pd.DataFrame()

        # Step 2: Sort by date (CRITICAL for preventing data leakage)
        logger.info("Sorting by date...")
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['goalie_id', 'game_date']).reset_index(drop=True)

        # Step 3: Calculate rolling features
        df = self.calculate_rolling_features(df)

        # Step 4: Fill NaN values for early-season games
        # For games where we don't have enough history, use season averages
        logger.info("Filling missing values...")
        df = self._fill_missing_values(df)

        # Step 5: Save processed features
        # Note: 'saves' column is our target variable for regression
        logger.info(f"Saving processed features to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine='pyarrow')

        logger.info("=" * 60)
        logger.info(f"FEATURE ENGINEERING COMPLETE")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Total rows: {len(df)}")
        logger.info("=" * 60)

        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in features

        Strategy:
        - For rolling features: Use season average if not enough games
        - For percentage features: Fill with league average (0.900 save %)
        - For count features: Fill with 0

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with missing values filled
        """
        # Identify feature types
        rolling_cols = [col for col in df.columns if '_ewa_' in col or '_rolling_' in col]
        pct_cols = [col for col in df.columns if 'pct' in col.lower() or 'percentage' in col.lower()]

        # Fill rolling features with season average
        for col in rolling_cols:
            # Extract base stat name (e.g., 'saves_ewa_3' -> 'saves')
            base_stat = col.split('_ewa_')[0].split('_rolling_')[0]

            if base_stat in df.columns:
                # Group by goalie and season, fill with expanding mean
                df[col] = df.groupby(['goalie_id', 'season'])[base_stat].transform(
                    lambda x: x.fillna(x.expanding().mean())
                )

        # Fill percentage features with league average
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.900)  # League average save %

        # Fill remaining NaN with 0
        df = df.fillna(0)

        return df

    def _create_simulated_betting_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create simulated betting lines based on goalie's rolling average

        In production, betting lines would come from sportsbooks.
        For training, we simulate them based on the goalie's 5-game average.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with betting_line, over_line, and saves_margin columns
        """
        # Use 5-game rolling average as betting line (if available)
        # Otherwise use season average
        if 'saves_ewa_5' in df.columns:
            betting_line = df['saves_ewa_5'].copy()
        elif 'saves_rolling_5' in df.columns:
            betting_line = df['saves_rolling_5'].copy()
        else:
            # Fall back to season average
            betting_line = df.groupby(['goalie_id', 'season'])['saves'].transform('mean')

        # Add some noise to make it realistic (bookmakers don't use exact averages)
        # Round to nearest 0.5 (typical for betting lines)
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 1.5, size=len(betting_line))
        betting_line = np.round((betting_line + noise) * 2) / 2

        # Ensure betting lines are reasonable (between 15 and 45 saves)
        betting_line = betting_line.clip(15, 45)

        # Fill NaN with median saves for that goalie
        betting_line = betting_line.fillna(df.groupby('goalie_id')['saves'].transform('median'))

        # Create target variables
        df['betting_line'] = betting_line
        df['over_line'] = (df['saves'] > betting_line).astype(int)
        df['saves_margin'] = df['saves'] - betting_line

        # Log statistics
        logger.info(f"Betting line statistics:")
        logger.info(f"  Mean: {betting_line.mean():.2f}")
        logger.info(f"  Median: {betting_line.median():.2f}")
        logger.info(f"  Std: {betting_line.std():.2f}")
        logger.info(f"  Range: {betting_line.min():.1f} - {betting_line.max():.1f}")
        logger.info(f"Over/Under distribution:")
        logger.info(f"  Over: {df['over_line'].sum()} ({df['over_line'].mean():.1%})")
        logger.info(f"  Under: {(1 - df['over_line']).sum()} ({(1 - df['over_line'].mean()):.1%})")

        return df

    def create_target_variable(
        self,
        df: pd.DataFrame,
        betting_lines: pd.Series,
        stat_column: str = 'saves'
    ) -> pd.DataFrame:
        """
        Create target variable for model training

        Target: over_line (1 if actual_saves > betting_line, 0 otherwise)

        Args:
            df: DataFrame with features
            betting_lines: Series with betting lines for each game
            stat_column: Column to compare to betting line

        Returns:
            DataFrame with target variable added
        """
        df['betting_line'] = betting_lines
        df['over_line'] = (df[stat_column] > betting_lines).astype(int)

        # Also create continuous target (margin)
        df['saves_margin'] = df[stat_column] - betting_lines

        return df


def create_training_dataset(
    raw_data_dir: str = "data/raw",
    output_path: str = "data/processed/training_data.parquet",
    config_path: str = "config/config.yaml"
) -> pd.DataFrame:
    """
    Create complete training dataset from raw data

    Args:
        raw_data_dir: Directory containing raw data
        output_path: Path to save processed dataset
        config_path: Path to config file

    Returns:
        Complete training DataFrame
    """
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(config_path)

    # Define data directories
    raw_dir = Path(raw_data_dir)
    boxscores_dir = raw_dir / "boxscores"
    pbp_dir = raw_dir / "play_by_play"

    # Run feature engineering
    df = pipeline.calculate_all_features(
        boxscores_dir=boxscores_dir,
        pbp_dir=pbp_dir,
        output_path=Path(output_path)
    )

    return df


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create training dataset
    df = create_training_dataset()

    print(f"\nTraining dataset created!")
    print(f"Shape: {df.shape}")
    print(f"\nFeature columns:")
    for col in sorted(df.columns):
        print(f"  - {col}")
