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
    api_client: NHLAPIClient,
    goalie_name: str,
    num_games: int = 10
) -> dict:
    """
    Fetch recent stats for a goalie

    For now, returns placeholder. Will be fully implemented once we build
    the real-time data fetching pipeline.

    Args:
        api_client: NHL API client
        goalie_name: Name of goalie
        num_games: Number of recent games to fetch

    Returns:
        Dictionary of feature values
    """
    logger.warning(f"Real-time stat fetching not yet implemented for {goalie_name}")
    logger.warning("Using placeholder values for demonstration")

    # Placeholder features (will be replaced with real API calls)
    placeholder_features = {
        'is_starter': 1.0,
        'is_home': 1.0,
        'opp_shots': 32.0,
        'opp_goals': 3.0,
        'pim': 2.0,
        'team_shots': 28.0,
        'team_goals': 2.5,
        'team_shooting_pct': 0.095,
        'team_powerplay_pct': 0.20,
        'team_faceoff_win_pct': 0.51,
        'saves_rolling_3': 25.0,
        'saves_rolling_5': 26.0,
        'saves_rolling_10': 27.0,
        'save_percentage_rolling_3': 0.910,
        'save_percentage_rolling_5': 0.912,
        'save_percentage_rolling_10': 0.908,
        'shots_against_rolling_3': 28.0,
        'shots_against_rolling_5': 29.0,
        'shots_against_rolling_10': 30.0,
        'goals_against_rolling_3': 2.5,
        'goals_against_rolling_5': 2.6,
        'goals_against_rolling_10': 2.8,
        'even_strength_save_pct_rolling_3': 0.920,
        'even_strength_save_pct_rolling_5': 0.918,
        'even_strength_save_pct_rolling_10': 0.915,
        'power_play_save_pct_rolling_3': 0.850,
        'power_play_save_pct_rolling_5': 0.860,
        'power_play_save_pct_rolling_10': 0.855,
        'saves_trend_10': 0.5,
        'opp_powerplay_goals': 1.2,
        'opp_powerplay_opportunities': 3.5,
        'team_powerplay_goals': 1.0,
        'team_powerplay_opportunities': 3.0,
        'team_blocked_shots': 15.0,
        'team_hits': 20.0,
        'team_pim': 8.0
    }

    return placeholder_features


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

    # Initialize API client (for future use)
    api_client = NHLAPIClient()

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

    # Fetch recent stats (placeholder for now)
    logger.info(f"Fetching recent stats for {home_goalie}...")
    home_features = fetch_goalie_recent_stats(api_client, home_goalie)

    logger.info(f"Fetching recent stats for {away_goalie}...")
    away_features = fetch_goalie_recent_stats(api_client, away_goalie)
    away_features['is_home'] = 0.0  # Away goalie
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
    print("NOTE: This tool currently uses placeholder stats for demonstration.")
    print("Real-time stat fetching will be implemented in the next phase.")
    print()


if __name__ == '__main__':
    main()
