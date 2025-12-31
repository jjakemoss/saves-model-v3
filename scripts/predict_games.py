"""Interactive prediction script for NHL goalie saves

Prompts user for game details and generates predictions using trained model.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.predictor import GoaliePredictor
from data.api_client import NHLAPIClient
from pipeline.realtime_features import RealtimeFeatureCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_user_input(prompt: str, input_type: type = str, required: bool = True):
    """
    Get input from user with validation

    Args:
        prompt: Prompt message
        input_type: Expected type (str, int, float)
        required: Whether input is required

    Returns:
        User input converted to input_type
    """
    while True:
        try:
            value = input(f"{prompt}: ").strip()

            if not value and not required:
                return None

            if not value and required:
                print("This field is required. Please enter a value.")
                continue

            if input_type == float:
                return float(value)
            elif input_type == int:
                return int(value)
            else:
                return value

        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")


def fetch_goalie_recent_stats(
    feature_collector: RealtimeFeatureCollector,
    goalie_name: str,
    team_abbrev: str,
    opponent_abbrev: str,
    is_home: bool,
    season: str = "20242025"
) -> dict:
    """
    Fetch recent stats for a goalie using real-time API

    Args:
        feature_collector: Feature collector instance
        goalie_name: Name of goalie
        team_abbrev: Team abbreviation
        opponent_abbrev: Opponent abbreviation
        is_home: Whether playing at home
        season: Season string

    Returns:
        Dictionary of feature values
    """
    logger.info(f"Fetching features for {goalie_name}...")

    # Collect features from NHL API
    features = feature_collector.collect_goalie_features(
        goalie_name=goalie_name,
        team_abbrev=team_abbrev,
        opponent_abbrev=opponent_abbrev,
        is_home=is_home,
        season=season
    )

    return features


def main():
    """Main interactive prediction workflow"""
    print("=" * 70)
    print("NHL GOALIE SAVES PREDICTION TOOL")
    print("=" * 70)
    print()

    # Load config
    config = load_config()

    # Load model
    model_dir = Path(config['paths'].get('models', 'models/trained'))
    model_name = 'xgboost_goalie_model'

    model_path = model_dir / f"{model_name}.pkl"
    features_path = model_dir / f"{model_name}_features.json"

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train the model first: python scripts/train_model.py")
        sys.exit(1)

    logger.info("Loading trained model...")
    predictor = GoaliePredictor(model_path, features_path)
    print()

    # Initialize feature collector
    api_client = NHLAPIClient()
    feature_collector = RealtimeFeatureCollector(api_client)

    # Interactive input
    print("Enter game details:")
    print("-" * 70)
    print()

    # Teams
    home_team = get_user_input("Home team (e.g., MIN, TOR, BOS)")
    away_team = get_user_input("Away team (e.g., SJ, CHI, MTL)")
    print()

    # Goalies
    home_goalie = get_user_input("Home goalie name (e.g., Filip Gustavsson)")
    away_goalie = get_user_input("Away goalie name (e.g., Mackenzie Blackwood)")
    print()

    # Betting lines (optional)
    print("Betting lines (press Enter to skip):")
    home_line = get_user_input(f"{home_goalie} saves line", float, required=False)
    away_line = get_user_input(f"{away_goalie} saves line", float, required=False)
    print()

    # Fetch recent stats using real-time feature collector
    logger.info(f"Fetching recent stats for {home_goalie}...")
    home_features = fetch_goalie_recent_stats(
        feature_collector,
        home_goalie,
        home_team,
        away_team,
        is_home=True
    )

    logger.info(f"Fetching recent stats for {away_goalie}...")
    away_features = fetch_goalie_recent_stats(
        feature_collector,
        away_goalie,
        away_team,
        home_team,
        is_home=False
    )
    print()

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = predictor.predict_game(
        home_features,
        away_features,
        home_line,
        away_line
    )

    # Display results
    print()
    print("=" * 70)
    print(f"{home_team} vs {away_team} - GOALIE SAVES PREDICTION")
    print("=" * 70)
    print()

    # Home goalie
    print(f"{home_goalie} ({home_team} - Home)")
    print(f"  Predicted Saves: {predictions['home']['predicted_saves']:.1f}")
    if home_line is not None:
        print(f"  Betting Line: {home_line}")
        print(f"  Difference: {predictions['home']['difference']:+.1f} saves")
        print(f"  Recommendation: {predictions['home']['recommendation']}")
    print()

    # Away goalie
    print(f"{away_goalie} ({away_team} - Away)")
    print(f"  Predicted Saves: {predictions['away']['predicted_saves']:.1f}")
    if away_line is not None:
        print(f"  Betting Line: {away_line}")
        print(f"  Difference: {predictions['away']['difference']:+.1f} saves")
        print(f"  Recommendation: {predictions['away']['recommendation']}")
    print()

    print("=" * 70)
    print()
    print("NOTE: Goalie lookup by name is not yet implemented.")
    print("Team/opponent stats are fetched from live NHL API.")
    print("Rolling averages use default values until goalie ID lookup is added.")
    print()


if __name__ == '__main__':
    main()
