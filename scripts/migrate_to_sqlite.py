"""
One-off migration: fold the existing betting_tracker.xlsx and the
consolidated season archive (data/betting_history/season_20252026.csv)
into data/betting.db.

Why two sources: betting_tracker.xlsx only goes back to 2026-02-01 (its
January sheets were lost when the workbook was recreated mid-season), but
the archived CSV covers the full season from 2026-01-04 onward. So:

  - For dates before 2026-02-01, the CSV archive is the only surviving
    record and is used as-is.
  - For 2026-02-01 onward, the live xlsx is authoritative (it reflects
    the final state of every row, whereas the CSV archive is a set of
    daily snapshots that may be stale for rows updated after the last
    backup ran).

Rows with no goalie_name or no betting_line are dropped -- they're blank
template rows from the old manual-entry workflow (populate_daily_games.py
/ add_manual_lines.py, both since removed) and carry no prediction or bet
information.

Usage:
    python scripts/migrate_to_sqlite.py
    python scripts/migrate_to_sqlite.py --xlsx betting_tracker.xlsx \
        --csv data/betting_history/season_20252026.csv --db data/betting.db
"""
import sys
from pathlib import Path
import argparse
import sqlite3
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting.db_manager import init_db, DEFAULT_DB_PATH

KEY_COLUMNS = ['game_id', 'goalie_name', 'book', 'betting_line', 'line_over', 'line_under']

DB_COLUMNS = [
    'game_date', 'game_id', 'book', 'goalie_name', 'goalie_id', 'team_abbrev',
    'opponent_team', 'is_home', 'betting_line', 'line_over', 'line_under',
    'predicted_saves', 'prob_over', 'confidence_pct', 'confidence_bucket',
    'recommendation', 'ev', 'bet_amount', 'bet_selection', 'actual_saves',
    'result', 'profit_loss', 'notes',
]

INT_COLUMNS = ['game_id', 'goalie_id', 'line_over', 'line_under', 'is_home']
FLOAT_COLUMNS = ['betting_line', 'predicted_saves', 'prob_over', 'confidence_pct',
                  'ev', 'bet_amount', 'actual_saves', 'profit_loss']


def _clean_rows(df):
    """Drop blank template rows and normalize types for insertion."""
    df = df[df['goalie_name'].notna() & df['betting_line'].notna()].copy()
    df['book'] = df['book'].fillna('Unknown')
    if 'bet_selection' in df.columns:
        df['bet_selection'] = df['bet_selection'].fillna('NONE')
    return df


def load_csv_archive(csv_path, before_date):
    """Load the consolidated CSV archive, restricted to dates before before_date."""
    df = pd.read_csv(csv_path)
    df = _clean_rows(df)
    df = df[df['game_date'] < before_date]
    return df


def load_xlsx(xlsx_path):
    """Load every date sheet from the live Excel tracker."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    date_sheets = [s for s in wb.sheetnames if s not in ('Summary', 'Settings')]
    wb.close()

    frames = []
    for sheet_name in date_sheets:
        frames.append(pd.read_excel(xlsx_path, sheet_name=sheet_name))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = _clean_rows(df)
    # The live workbook can contain literal duplicate rows (a known bug in
    # the old NaN-vs-NaN dedup check) -- collapse to the most complete copy.
    df = df.sort_values('game_date').drop_duplicates(subset=KEY_COLUMNS, keep='last')
    return df


def _row_value(row, col):
    value = row.get(col)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if col in INT_COLUMNS:
        return int(value)
    if col in FLOAT_COLUMNS:
        return float(value)
    return value


def migrate(xlsx_path='betting_tracker.xlsx',
            csv_path='data/betting_history/season_20252026.csv',
            db_path=DEFAULT_DB_PATH):
    xlsx_path = Path(xlsx_path)
    csv_path = Path(csv_path)
    db_path = Path(db_path)

    if db_path.exists():
        print(f"[ERROR] {db_path} already exists. Delete it first if you want to re-run the migration.")
        sys.exit(1)

    print("Loading live Excel tracker...")
    xlsx_rows = load_xlsx(xlsx_path)
    print(f"  {len(xlsx_rows)} usable rows ({xlsx_rows['game_date'].min()} to {xlsx_rows['game_date'].max()})")

    earliest_xlsx_date = xlsx_rows['game_date'].min()

    print("\nLoading CSV archive for dates before the live tracker...")
    csv_rows = load_csv_archive(csv_path, before_date=earliest_xlsx_date)
    if len(csv_rows):
        print(f"  {len(csv_rows)} usable rows ({csv_rows['game_date'].min()} to {csv_rows['game_date'].max()})")
    else:
        print(f"  {len(csv_rows)} usable rows")

    combined = pd.concat([csv_rows, xlsx_rows], ignore_index=True)
    combined = combined.sort_values('game_date').drop_duplicates(subset=KEY_COLUMNS, keep='last')
    print(f"\nCombined: {len(combined)} unique rows "
          f"({combined['game_date'].min()} to {combined['game_date'].max()})")

    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        inserted = 0
        for _, row in combined.iterrows():
            values = [_row_value(row, col) for col in DB_COLUMNS]
            cur.execute(
                f"""
                INSERT INTO bets ({', '.join(DB_COLUMNS)})
                VALUES ({', '.join(['?'] * len(DB_COLUMNS))})
                """,
                values,
            )
            inserted += 1
        conn.commit()
    finally:
        conn.close()

    print(f"\n[OK] Inserted {inserted} rows into {db_path}")

    # Verification
    conn = sqlite3.connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
        by_result = conn.execute(
            "SELECT result, COUNT(*) FROM bets GROUP BY result ORDER BY COUNT(*) DESC"
        ).fetchall()
        date_range = conn.execute("SELECT MIN(game_date), MAX(game_date) FROM bets").fetchone()
    finally:
        conn.close()

    print(f"\nVerification:")
    print(f"  Total rows in database: {total}")
    print(f"  Date range: {date_range[0]} to {date_range[1]}")
    print(f"  By result:")
    for result, count in by_result:
        print(f"    {result or '(none)'}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Migrate betting_tracker.xlsx + CSV archive into SQLite')
    parser.add_argument('--xlsx', type=str, default='betting_tracker.xlsx')
    parser.add_argument('--csv', type=str, default='data/betting_history/season_20252026.csv')
    parser.add_argument('--db', type=str, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    migrate(xlsx_path=args.xlsx, csv_path=args.csv, db_path=args.db)


if __name__ == '__main__':
    main()
