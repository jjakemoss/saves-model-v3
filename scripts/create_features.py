#!/usr/bin/env python
"""
Script to create training features from collected raw data

Usage:
    python scripts/create_features.py
    python scripts/create_features.py --output data/processed/features_v2.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.feature_engineering import create_training_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Create training features from raw NHL data"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data (default: data/raw)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/training_data.parquet",
        help="Output path for processed features (default: data/processed/training_data.parquet)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/feature_engineering.log'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # Validate inputs
    raw_dir = Path(args.raw_data_dir)
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        sys.exit(1)

    boxscores_dir = raw_dir / "boxscores"
    if not boxscores_dir.exists():
        logger.error(f"Boxscores directory not found: {boxscores_dir}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run feature engineering
    logger.info("Starting feature engineering pipeline...")

    try:
        df = create_training_dataset(
            raw_data_dir=str(raw_dir),
            output_path=str(output_path),
            config_path=args.config
        )

        logger.info(f"\n{'=' * 60}")
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Unique goalies: {df['goalie_id'].nunique()}")
        logger.info(f"Unique games: {df['game_id'].nunique()}")
        logger.info(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"{'=' * 60}\n")

        # Display feature summary
        logger.info("Feature categories:")
        feature_categories = {
            'Base features': [col for col in df.columns if not any(x in col for x in ['_ewa_', '_rolling_', 'interaction', 'high_danger', 'mid_danger', 'low_danger'])],
            'Rolling features': [col for col in df.columns if '_ewa_' in col or '_rolling_' in col],
            'Shot quality features': [col for col in df.columns if any(x in col for x in ['high_danger', 'mid_danger', 'low_danger', 'xg', 'rebound'])],
        }

        for category, features in feature_categories.items():
            logger.info(f"  {category}: {len(features)} features")

        logger.info(f"\nTotal features: {len(df.columns)}")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
