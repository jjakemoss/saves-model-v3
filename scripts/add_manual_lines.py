"""
Add blank rows for manual betting line entry

This script:
1. Reads the current day's sheet from betting_tracker.xlsx
2. Gets unique goalies already in the tracker
3. Adds a new row for each goalie with a specified book name
4. User then fills in betting_line, line_over, line_under manually

Usage:
    python scripts/add_manual_lines.py [--date YYYY-MM-DD] [--book BOOK_NAME]
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting import BettingTracker


def add_manual_lines(date=None, book_name='Manual', tracker_file='betting_tracker.xlsx'):
    """
    Add blank rows for each goalie to enter manual betting lines.

    Args:
        date: Date string (YYYY-MM-DD). If None, uses today
        book_name: Name of the sportsbook (default: 'Manual')
        tracker_file: Path to betting tracker Excel file

    Returns:
        int: Number of rows added
    """
    tracker = BettingTracker(tracker_file)

    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    print(f"\nAdding manual line rows for {date}...")

    # Check if sheet exists
    try:
        existing_df = pd.read_excel(tracker_file, sheet_name=date)
    except Exception:
        print(f"[ERROR] No sheet found for {date}")
        print("Run fetch_and_predict.py first to populate games.")
        return 0

    if existing_df.empty:
        print(f"[ERROR] No data found for {date}")
        return 0

    # Get unique goalies from existing data
    goalies = existing_df[existing_df['goalie_name'].notna() & (existing_df['goalie_name'] != '')]
    unique_goalies = goalies.drop_duplicates(subset=['game_id', 'goalie_id'])[
        ['game_date', 'game_id', 'goalie_name', 'goalie_id', 'team_abbrev', 'opponent_team', 'is_home']
    ]

    if unique_goalies.empty:
        print("[ERROR] No goalies found in tracker for this date")
        return 0

    # Check which goalies already have rows for this book
    existing_book_rows = existing_df[existing_df['book'] == book_name]
    existing_goalie_ids = set(existing_book_rows['goalie_id'].dropna().astype(int).tolist())

    # Create new rows for goalies that don't have this book yet
    rows_to_add = []
    for _, goalie in unique_goalies.iterrows():
        goalie_id = goalie['goalie_id']
        if pd.notna(goalie_id) and int(goalie_id) in existing_goalie_ids:
            continue

        rows_to_add.append({
            'game_date': goalie['game_date'],
            'game_id': goalie['game_id'],
            'book': book_name,
            'goalie_name': goalie['goalie_name'],
            'betting_line': None,
            'line_over': None,
            'line_under': None,
            'goalie_id': goalie['goalie_id'],
            'team_abbrev': goalie['team_abbrev'],
            'opponent_team': goalie['opponent_team'],
            'is_home': goalie['is_home'],
        })

    if not rows_to_add:
        print(f"[OK] All goalies already have rows for '{book_name}'")
        return 0

    # Append to tracker
    new_df = pd.DataFrame(rows_to_add)
    tracker.append_games(new_df)

    print(f"\n{'='*60}")
    print(f"[OK] Added {len(rows_to_add)} rows for '{book_name}'")
    print(f"{'='*60}")

    for row in rows_to_add:
        print(f"  {row['goalie_name']} ({row['team_abbrev']})")

    print("\nNext steps:")
    print("  1. Open betting_tracker.xlsx")
    print(f"  2. Fill in betting_line, line_over, line_under for '{book_name}' rows")
    print("  3. Save file")
    print("  4. Run: python scripts/generate_predictions.py")

    return len(rows_to_add)


def main():
    parser = argparse.ArgumentParser(
        description='Add blank rows for manual betting line entry'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Date to add rows for (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--book',
        type=str,
        default='Manual',
        help='Sportsbook name for the new rows. Default: Manual'
    )
    parser.add_argument(
        '--tracker',
        type=str,
        default='betting_tracker.xlsx',
        help='Path to betting tracker file. Default: betting_tracker.xlsx'
    )

    args = parser.parse_args()

    try:
        add_manual_lines(date=args.date, book_name=args.book, tracker_file=args.tracker)
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
