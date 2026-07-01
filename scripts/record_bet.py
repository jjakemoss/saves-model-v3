"""
Record a bet you actually placed, independent of the model's recommendation.

Run this right after placing a bet on your sportsbook app -- from your
phone via the "Record Bet" GitHub Action, or locally. Works the same way
whether or not the model recommended the bet, so betting below the EV
threshold (or on the side the model didn't pick) is just as easy to log.

Matches the most recently fetched line for the given date/goalie/book, so
it still works if the line moved and got re-fetched during the day.

Usage:
    python scripts/record_bet.py --goalie_name Shesterkin --book Underdog \
        --bet_selection OVER --bet_amount 2
    python scripts/record_bet.py --date 2026-01-15 --goalie_name Hellebuyck \
        --book BetMGM --bet_selection UNDER --bet_amount 1.5 --notes "line moved late"
"""
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting.db_manager import BettingDB, DEFAULT_DB_PATH
from betting.excel_export import export_to_excel, DEFAULT_XLSX_PATH


def record_bet(goalie_name, book, bet_selection, bet_amount, date=None, notes=None,
                db_path=DEFAULT_DB_PATH, xlsx_path=DEFAULT_XLSX_PATH):
    """
    Record a bet and regenerate the Excel snapshot.

    Returns:
        dict: The updated database row
    """
    db = BettingDB(db_path)
    updated = db.record_bet(
        goalie_name=goalie_name,
        book=book,
        bet_selection=bet_selection,
        bet_amount=bet_amount,
        date=date,
        notes=notes,
    )

    export_to_excel(db_path, xlsx_path)

    print("[OK] Recorded bet:")
    print(f"  {updated['goalie_name']} ({updated['team_abbrev']} vs {updated['opponent_team']}) @ {updated['book']}")
    print(f"  {updated['bet_selection']} {updated['betting_line']} for {updated['bet_amount']} unit(s)")
    print(f"  Model recommendation was: {updated['recommendation']} (EV: {updated['ev']})")

    return updated


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Record a bet you placed, independent of the model's recommendation"
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Game date (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--goalie_name',
        type=str,
        required=True,
        help="Goalie's last name"
    )
    parser.add_argument(
        '--book',
        type=str,
        required=True,
        help='Sportsbook (e.g. Underdog, BetOnline, BetMGM, Caesars)'
    )
    parser.add_argument(
        '--bet_selection',
        type=str,
        required=True,
        choices=['OVER', 'UNDER'],
        help='Which side you bet'
    )
    parser.add_argument(
        '--bet_amount',
        type=float,
        required=True,
        help='Units wagered'
    )
    parser.add_argument(
        '--notes',
        type=str,
        default=None,
        help='Optional notes'
    )
    parser.add_argument(
        '--db',
        type=str,
        default=DEFAULT_DB_PATH,
        help='Path to betting database. Default: data/betting.db'
    )
    parser.add_argument(
        '--tracker',
        type=str,
        default=DEFAULT_XLSX_PATH,
        help='Path to betting tracker Excel snapshot. Default: betting_tracker.xlsx'
    )

    args = parser.parse_args()

    try:
        record_bet(
            goalie_name=args.goalie_name,
            book=args.book,
            bet_selection=args.bet_selection,
            bet_amount=args.bet_amount,
            date=args.date,
            notes=args.notes,
            db_path=args.db,
            xlsx_path=args.tracker,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
