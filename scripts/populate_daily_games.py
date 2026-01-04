"""
Populate betting tracker with today's NHL games

This script:
1. Fetches today's NHL schedule
2. Identifies starting goalies (or marks TBD if unknown)
3. Appends game rows to betting_tracker.xlsx
4. Does NOT generate predictions (user enters betting lines first)

Usage:
    python scripts/populate_daily_games.py [--date YYYY-MM-DD]
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting import NHLBettingData, BettingTracker


def populate_daily_games(date=None, tracker_file='betting_tracker.xlsx'):
    """
    Fetch today's games and add to tracker

    Args:
        date: Date string (YYYY-MM-DD). If None, uses today
        tracker_file: Path to betting tracker Excel file

    Returns:
        int: Number of games added
    """
    # Initialize clients
    nhl_data = NHLBettingData()
    tracker = BettingTracker(tracker_file)

    # Get date
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    print(f"\nFetching NHL games for {date}...")

    # Fetch today's games
    games = nhl_data.get_todays_games(date)

    if not games:
        print(f"[OK] No games scheduled for {date}")
        return 0

    print(f"Found {len(games)} games")

    # Process each game to get goalies
    games_to_add = []

    for game in games:
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']

        print(f"\n  Game {game_id}: {away_team} @ {home_team}")

        # Add rows for both goalies (user will enter goalie last name later)
        games_to_add.append({
            'game_date': date,
            'game_id': game_id,
            'goalie_name': '',  # User fills in last name
            'goalie_id': None,  # Will be looked up in prediction script
            'team_abbrev': home_team,
            'opponent_team': away_team,
            'is_home': 1
        })

        games_to_add.append({
            'game_date': date,
            'game_id': game_id,
            'goalie_name': '',  # User fills in last name
            'goalie_id': None,  # Will be looked up in prediction script
            'team_abbrev': away_team,
            'opponent_team': home_team,
            'is_home': 0
        })

    # Convert to DataFrame
    games_df = pd.DataFrame(games_to_add)

    # Append to tracker
    tracker.append_games(games_df)

    print(f"\n{'='*60}")
    print(f"[OK] Added {len(games_df)} game rows to {tracker_file}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Open betting_tracker.xlsx")
    print("  2. For games you want to bet, enter:")
    print("     - goalie_name: Goalie's LAST NAME (e.g., 'Shesterkin')")
    print("     - betting_line: Saves line from sportsbook (e.g., 24.5)")
    print("  3. Save file")
    print("  4. Run: python scripts/generate_predictions.py")
    print("     (Script will look up goalie_id automatically)")

    return len(games_df)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Populate betting tracker with today\'s NHL games'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Date to fetch games for (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--tracker',
        type=str,
        default='betting_tracker.xlsx',
        help='Path to betting tracker file. Default: betting_tracker.xlsx'
    )

    args = parser.parse_args()

    try:
        populate_daily_games(date=args.date, tracker_file=args.tracker)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nDid you forget to initialize the tracker?")
        print("Run: python scripts/init_betting_tracker.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
