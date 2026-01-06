"""
Update betting tracker with yesterday's game results

This script:
1. Reads betting_tracker.xlsx
2. Finds games needing results (yesterday's games with empty actual_saves)
3. Fetches completed game boxscores
4. Updates actual_saves, result (WIN/LOSS/PUSH/NO BET), profit_loss
5. Creates daily backup to data/betting_history/

Usage:
    python scripts/update_betting_results.py [--date YYYY-MM-DD]
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting import NHLBettingData, BettingTracker


def calculate_profit_loss(row):
    """
    Calculate profit/loss for a bet

    Args:
        row: DataFrame row with bet info

    Returns:
        float: Profit/loss in units
    """
    bet_selection = row['bet_selection']
    bet_amount = row['bet_amount']
    actual_saves = row['actual_saves']
    betting_line = row['betting_line']

    # No bet placed
    if pd.isna(bet_selection) or bet_selection == 'NONE' or pd.isna(bet_amount) or bet_amount == 0:
        return 0.0

    # Calculate win payout (assuming -110 odds)
    win_payout = bet_amount * (100 / 110)

    # Determine result
    if actual_saves > betting_line:
        # Goalie went OVER the line
        if bet_selection == 'OVER':
            return win_payout  # WIN
        else:
            return -bet_amount  # LOSS
    elif actual_saves < betting_line:
        # Goalie went UNDER the line
        if bet_selection == 'UNDER':
            return win_payout  # WIN
        else:
            return -bet_amount  # LOSS
    else:
        # PUSH (actual saves exactly on the line)
        return 0.0


def determine_result(row):
    """
    Determine result string for a bet

    Args:
        row: DataFrame row with bet info

    Returns:
        str: WIN/LOSS/PUSH/NO BET
    """
    bet_selection = row['bet_selection']
    bet_amount = row['bet_amount']
    actual_saves = row['actual_saves']
    betting_line = row['betting_line']

    # No bet placed
    if pd.isna(bet_selection) or bet_selection == 'NONE' or pd.isna(bet_amount) or bet_amount == 0:
        return 'NO BET'

    # Calculate result
    if actual_saves > betting_line:
        return 'WIN' if bet_selection == 'OVER' else 'LOSS'
    elif actual_saves < betting_line:
        return 'WIN' if bet_selection == 'UNDER' else 'LOSS'
    else:
        return 'PUSH'


def update_betting_results(date=None, tracker_file='betting_tracker.xlsx'):
    """
    Update results for yesterday's games

    Args:
        date: Date string (YYYY-MM-DD). If None, uses yesterday
        tracker_file: Path to betting tracker Excel file

    Returns:
        int: Number of results updated
    """
    # Initialize clients
    nhl_data = NHLBettingData()
    tracker = BettingTracker(tracker_file)

    # Get date (default to yesterday)
    if date is None:
        yesterday = datetime.now() - timedelta(days=1)
        date = yesterday.strftime('%Y-%m-%d')

    print(f"\nUpdating results for {date}...")

    # Get games needing results
    pending = tracker.get_pending_results(date)

    if len(pending) == 0:
        print(f"[OK] No games need results for {date}")
        print("\nPossible reasons:")
        print("  - Results already updated")
        print("  - No games scheduled")
        print("  - Games not completed yet")
        return 0

    print(f"Found {len(pending)} games needing results")

    results_list = []

    for idx, row in pending.iterrows():
        game_id = row['game_id']
        goalie_id = row['goalie_id']
        goalie_name = row['goalie_name']
        team = row['team_abbrev']
        opponent = row['opponent_team']

        print(f"\n  {goalie_name} ({team} vs {opponent})")

        # Skip if goalie unknown
        if goalie_name == "TBD" or pd.isna(goalie_id):
            print("    [SKIP] Goalie TBD")
            continue

        try:
            # Fetch game result
            print("    Fetching game result...")
            game_saves = nhl_data.get_game_result(game_id)

            if goalie_id not in game_saves:
                print(f"    [WARNING] Goalie {goalie_id} not found in game results")
                print(f"    Available goalies: {list(game_saves.keys())}")
                continue

            actual_saves = game_saves[goalie_id]
            print(f"    Actual saves: {actual_saves}")

            # Calculate result and profit/loss
            row_copy = row.copy()
            row_copy['actual_saves'] = actual_saves

            result = determine_result(row_copy)
            profit_loss = calculate_profit_loss(row_copy)

            results_list.append({
                'game_id': game_id,
                'goalie_id': goalie_id,
                'game_date': date,  # Include for date sheet routing
                'actual_saves': actual_saves,
                'result': result,
                'profit_loss': profit_loss
            })

            # Display result
            if result == 'NO BET':
                print(f"    [OK] Result: {result}")
            else:
                print(f"    [OK] Result: {result} ({profit_loss:+.2f} units)")

        except Exception as e:
            print(f"    [ERROR] Error fetching result: {e}")
            continue

    if len(results_list) == 0:
        print("\n[WARNING] No results updated")
        return 0

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Update tracker
    tracker.update_results(results_df)

    # Create backup
    backup_file = tracker.backup_to_csv()

    print(f"\n{'='*60}")
    print(f"[OK] Updated {len(results_df)} game results")
    print(f"{'='*60}")

    # Show summary of results
    print("\nResults Summary:")
    result_counts = results_df['result'].value_counts()
    for result, count in result_counts.items():
        print(f"  {result}: {count}")

    # Show profit/loss
    total_pl = results_df['profit_loss'].sum()
    if total_pl != 0:
        print(f"\nProfit/Loss: {total_pl:+.2f} units")

    print(f"\nNext step:")
    print(f"  Run: python scripts/betting_dashboard.py")

    return len(results_df)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Update betting tracker with yesterday\'s results'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Date to update results for (YYYY-MM-DD). Default: yesterday'
    )
    parser.add_argument(
        '--tracker',
        type=str,
        default='betting_tracker.xlsx',
        help='Path to betting tracker file. Default: betting_tracker.xlsx'
    )

    args = parser.parse_args()

    try:
        update_betting_results(date=args.date, tracker_file=args.tracker)
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
