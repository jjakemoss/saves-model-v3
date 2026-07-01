"""
Initialize the betting tracker database and its Excel snapshot.

Creates data/betting.db (SQLite -- the source of truth) and
betting_tracker.xlsx (a read-only rendering of it, regenerated after
every write). The xlsx file should never be hand-edited.

Usage:
    python scripts/init_betting_tracker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting.db_manager import init_db, DEFAULT_DB_PATH
from betting.excel_export import export_to_excel, DEFAULT_XLSX_PATH


def create_betting_tracker(db_path=DEFAULT_DB_PATH, xlsx_path=DEFAULT_XLSX_PATH):
    """Create a new betting database and its Excel snapshot."""
    db_file = init_db(db_path)
    print(f"[OK] Created betting database: {db_file.absolute()}")

    xlsx_file = export_to_excel(db_path, xlsx_path)
    print(f"[OK] Created Excel snapshot: {xlsx_file.absolute()}")

    print('\nNext steps:')
    print('  1. Run: python scripts/fetch_and_predict.py --verbose')
    print('  2. Place bets on your sportsbook app(s)')
    print('  3. Run: python scripts/record_bet.py --goalie_name <name> --book <book> '
          '--bet_selection OVER --bet_amount 1')


if __name__ == '__main__':
    create_betting_tracker()
