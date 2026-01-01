"""Daily data update script

Fetches completed games from yesterday, collects stats, and appends to training dataset.
Run daily via cron/scheduler to keep training data current.

Usage:
    python scripts/update_daily_data.py
    python scripts/update_daily_data.py --date 2025-01-15  # Specific date
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.api_client import NHLAPIClient
from data.collectors import GameCollector
from features.feature_engineering import FeatureEngineeringPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daily_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_completed_games_for_date(api_client: NHLAPIClient, target_date: datetime) -> list:
    """
    Get all completed games for a specific date

    Args:
        api_client: NHL API client
        target_date: Date to fetch games for

    Returns:
        List of completed game IDs
    """
    logger.info(f"Fetching games for {target_date.strftime('%Y-%m-%d')}")

    # Get schedule for the date
    date_str = target_date.strftime('%Y-%m-%d')
    endpoint = f"/v1/schedule/{date_str}"

    try:
        response = api_client.get(endpoint)

        if not response or 'gameWeek' not in response:
            logger.warning(f"No games found for {date_str}")
            return []

        # Extract completed games
        completed_games = []

        for day in response.get('gameWeek', []):
            for game in day.get('games', []):
                game_id = game.get('id')
                game_state = game.get('gameState')
                game_type = game.get('gameType')

                # Only include completed regular season games
                if game_state in ['OFF', 'FINAL'] and game_type == 2:
                    completed_games.append(game_id)

        logger.info(f"Found {len(completed_games)} completed games")
        return completed_games

    except Exception as e:
        logger.error(f"Error fetching schedule for {date_str}: {e}")
        return []


def collect_game_data(game_collector: GameCollector, game_ids: list) -> pd.DataFrame:
    """
    Collect boxscore and play-by-play data for games

    Args:
        game_collector: Game data collector
        game_ids: List of game IDs to collect

    Returns:
        DataFrame with collected game data
    """
    logger.info(f"Collecting data for {len(game_ids)} games")

    all_games = []

    for game_id in game_ids:
        try:
            # Collect game data
            game_data = game_collector.collect_game(game_id)

            if game_data:
                all_games.append(game_data)
                logger.info(f"Collected data for game {game_id}")
            else:
                logger.warning(f"No data returned for game {game_id}")

        except Exception as e:
            logger.error(f"Error collecting game {game_id}: {e}")
            continue

    if not all_games:
        logger.warning("No game data collected")
        return pd.DataFrame()

    # Combine into DataFrame
    df = pd.DataFrame(all_games)
    logger.info(f"Collected {len(df)} goalie performances")

    return df


def append_to_training_data(new_data: pd.DataFrame, output_path: Path):
    """
    Append new data to existing training dataset

    Args:
        new_data: New game data to append
        output_path: Path to training data file
    """
    logger.info(f"Appending {len(new_data)} rows to training data")

    if output_path.exists():
        # Load existing data
        existing_data = pd.read_parquet(output_path)
        logger.info(f"Loaded {len(existing_data)} existing rows")

        # Combine and remove duplicates
        combined = pd.concat([existing_data, new_data], ignore_index=True)

        # Remove duplicates based on goalie_id + game_id
        combined = combined.drop_duplicates(subset=['goalie_id', 'game_id'], keep='last')

        # Sort by date
        combined = combined.sort_values('game_date').reset_index(drop=True)

        logger.info(f"Combined dataset has {len(combined)} rows ({len(combined) - len(existing_data)} new)")

        # Save
        combined.to_parquet(output_path)
        logger.info(f"Saved updated data to {output_path}")

    else:
        # No existing data, just save new data
        new_data.to_parquet(output_path)
        logger.info(f"Created new training data file at {output_path}")


def main():
    """Main daily update workflow"""
    parser = argparse.ArgumentParser(description='Update training data with completed games')
    parser.add_argument('--date', type=str, help='Date to fetch (YYYY-MM-DD). Default: yesterday')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')

    args = parser.parse_args()

    print("=" * 70)
    print("NHL GOALIE SAVES MODEL - DAILY DATA UPDATE")
    print("=" * 70)
    print()

    # Load config
    config = load_config(args.config)

    # Determine target date
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        # Default: yesterday
        target_date = datetime.now() - timedelta(days=1)

    logger.info(f"Updating data for {target_date.strftime('%Y-%m-%d')}")

    # Initialize collectors
    api_client = NHLAPIClient()
    game_collector = GameCollector(api_client)

    # Get completed games for date
    completed_games = get_completed_games_for_date(api_client, target_date)

    if not completed_games:
        logger.info("No completed games to process")
        print("\nNo completed games found for this date.")
        return

    # Collect raw game data
    raw_data = collect_game_data(game_collector, completed_games)

    if raw_data.empty:
        logger.warning("No data collected from games")
        print("\nFailed to collect data from games.")
        return

    # Save raw data
    raw_dir = Path(config['paths']['raw_data'])
    date_str = target_date.strftime('%Y%m%d')
    raw_output = raw_dir / f"games_{date_str}.parquet"
    raw_data.to_parquet(raw_output)
    logger.info(f"Saved raw data to {raw_output}")

    # Run feature engineering on new data
    logger.info("Running feature engineering on new data")
    feature_pipeline = FeatureEngineeringPipeline(config)

    # Load existing processed data to get context for rolling features
    processed_path = Path(config['paths']['processed_data']) / 'training_data.parquet'

    if processed_path.exists():
        # Combine with existing data for proper rolling feature calculation
        existing_data = pd.read_parquet(processed_path)

        # Combine old + new
        combined_raw = pd.concat([existing_data, raw_data], ignore_index=True)
        combined_raw = combined_raw.drop_duplicates(subset=['goalie_id', 'game_id'], keep='last')
        combined_raw = combined_raw.sort_values('game_date').reset_index(drop=True)

        logger.info(f"Combined {len(existing_data)} existing + {len(raw_data)} new = {len(combined_raw)} total rows")

        # Run feature engineering on combined dataset
        processed_data = feature_pipeline.run(combined_raw)

        # Save complete updated dataset
        processed_data.to_parquet(processed_path)
        logger.info(f"Updated training data saved to {processed_path}")

    else:
        # No existing data, process new data only
        logger.info("No existing processed data found. Processing new data only.")
        processed_data = feature_pipeline.run(raw_data)
        processed_data.to_parquet(processed_path)
        logger.info(f"Created new training data at {processed_path}")

    # Summary
    print()
    print("=" * 70)
    print("UPDATE COMPLETE")
    print("=" * 70)
    print(f"Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Games processed: {len(completed_games)}")
    print(f"Goalie performances added: {len(raw_data)}")
    print(f"Total dataset size: {len(processed_data)} rows")
    print()
    logger.info("Daily update completed successfully")


if __name__ == '__main__':
    main()
