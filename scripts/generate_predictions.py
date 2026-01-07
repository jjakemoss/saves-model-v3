"""
Generate predictions for games with betting lines entered

This script:
1. Reads betting_tracker.xlsx
2. Finds games with betting lines but no predictions
3. Fetches recent goalie stats
4. Calculates 89 features
5. Generates predictions using trained model
6. Updates tracker with predictions

Usage:
    python scripts/generate_predictions.py [--date YYYY-MM-DD]
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting import (
    NHLBettingData,
    BettingTracker,
    BettingFeatureCalculator,
    BettingPredictor
)


def generate_predictions(date=None, tracker_file='betting_tracker.xlsx'):
    """
    Generate predictions for games with betting lines

    Args:
        date: Date string (YYYY-MM-DD). If None, uses today
        tracker_file: Path to betting tracker Excel file

    Returns:
        int: Number of predictions generated
    """
    # Initialize clients
    nhl_data = NHLBettingData()
    tracker = BettingTracker(tracker_file)
    feature_calc = BettingFeatureCalculator()
    predictor = BettingPredictor()

    # Get date
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    print(f"\nGenerating predictions for {date}...")

    # Get games needing predictions
    pending = tracker.get_pending_predictions(date)

    if len(pending) == 0:
        print(f"[OK] No games need predictions for {date}")
        print("\nPossible reasons:")
        print("  - No betting lines entered yet")
        print("  - Predictions already generated")
        print("  - No games scheduled")
        return 0

    print(f"Found {len(pending)} games needing predictions")

    predictions_list = []

    for idx, row in pending.iterrows():
        game_id = row['game_id']
        goalie_id = row['goalie_id']
        goalie_name = row['goalie_name']
        team = row['team_abbrev']
        opponent = row['opponent_team']
        is_home = row['is_home']
        betting_line = row['betting_line']
        line_over_odds = row.get('line_over')
        line_under_odds = row.get('line_under')

        print(f"\n  {goalie_name} ({team} vs {opponent}) - Line: {betting_line}")

        # Skip if goalie name not provided
        if not goalie_name or goalie_name == '' or pd.isna(goalie_name):
            print("    [SKIP] No goalie name entered")
            continue

        # Look up goalie_id if not already set
        if pd.isna(goalie_id) or goalie_id is None:
            print(f"    Looking up goalie ID for '{goalie_name}'...")
            goalie_id = nhl_data.get_goalie_id_from_game(game_id, goalie_name)

            if goalie_id is None:
                print(f"    [ERROR] Could not find goalie '{goalie_name}' in game roster")
                print(f"    Check spelling or wait for lineups to be announced")
                continue

            print(f"    Found goalie ID: {goalie_id}")

        try:
            # Fetch recent games for goalie
            print("    Fetching recent games...")
            recent_games = nhl_data.get_goalie_recent_games(
                goalie_id,
                season='20252026',
                n_games=15
            )

            if len(recent_games) < 3:
                print(f"    [WARNING] Only {len(recent_games)} recent games found")

            # Calculate features
            print("    Calculating features...")
            features_df = feature_calc.prepare_prediction_features(
                goalie_id=goalie_id,
                team=team,
                opponent=opponent,
                is_home=is_home,
                game_date=date,
                recent_games=recent_games
            )

            # Generate prediction (pass betting_line and odds)
            print("    Running model prediction...")
            prediction = predictor.predict(
                features_df,
                betting_line=betting_line,
                line_over_odds=line_over_odds if pd.notna(line_over_odds) else None,
                line_under_odds=line_under_odds if pd.notna(line_under_odds) else None
            )

            # Add game_id, goalie info, and date for tracking
            prediction['game_id'] = game_id
            prediction['goalie_id'] = goalie_id
            prediction['goalie_name'] = goalie_name  # Include for Excel matching
            prediction['game_date'] = date  # Include for date sheet routing

            predictions_list.append(prediction)

            # Display result
            print(f"    [OK] Prediction: {prediction['recommendation']}")
            print(f"      Prob Over: {prediction['prob_over']:.1%}")
            print(f"      Confidence: {prediction['confidence_bucket']}")
            if prediction.get('recommended_ev') is not None:
                print(f"      Expected EV: {prediction['recommended_ev']:+.1%}")

        except Exception as e:
            print(f"    [ERROR] Error generating prediction: {e}")
            continue

    if len(predictions_list) == 0:
        print("\n[WARNING] No predictions generated")
        return 0

    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions_list)

    # Update tracker
    tracker.update_predictions(predictions_df)

    print(f"\n{'='*60}")
    print(f"[OK] Generated {len(predictions_df)} predictions")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Open betting_tracker.xlsx")
    print("  2. Review recommendations and confidence levels")
    print("  3. Enter bet_amount for games you want to bet")
    print("  4. Enter bet_selection (OVER/UNDER) for those games")
    print("  5. Place bets at sportsbook")

    # Show summary of recommendations
    print("\nRecommendation Summary:")
    rec_counts = predictions_df['recommendation'].value_counts()
    for rec, count in rec_counts.items():
        print(f"  {rec}: {count}")

    return len(predictions_df)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate predictions for games with betting lines'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Date to generate predictions for (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--tracker',
        type=str,
        default='betting_tracker.xlsx',
        help='Path to betting tracker file. Default: betting_tracker.xlsx'
    )

    args = parser.parse_args()

    try:
        generate_predictions(date=args.date, tracker_file=args.tracker)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
