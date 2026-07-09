"""
Migration: add the ticket/CLV tracking tables to data/betting.db.

Strictly additive -- creates line_snapshots, tickets, and ticket_legs
with CREATE TABLE IF NOT EXISTS, and adds bets.model_version with
ALTER TABLE ... ADD COLUMN guarded by a PRAGMA table_info check. Never
drops, alters, or rewrites any existing table or row. Safe to run
repeatedly (idempotent) and safe to run against a database that already
has some or all of the new tables.

Run this once before the season's first fetch_and_predict.py run --
fetch_and_predict.py writes line_snapshots rows and a model_version
value on every prediction, so both need to exist first.

Usage:
    python scripts/add_tracking_tables.py
    python scripts/add_tracking_tables.py --db data/betting.db
"""
import sys
from pathlib import Path
import argparse
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting.tracking_db import init_tracking_schema, DEFAULT_DB_PATH


def _table_names(db_path):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def _has_model_version(db_path):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("PRAGMA table_info(bets)")
        return any(row[1] == 'model_version' for row in cur.fetchall())
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Add ticket/CLV tracking tables to the betting database (idempotent, additive-only)'
    )
    parser.add_argument(
        '--db',
        type=str,
        default=DEFAULT_DB_PATH,
        help='Path to betting database. Default: data/betting.db'
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        print("Run: python scripts/init_betting_tracker.py first")
        sys.exit(1)

    before = _table_names(db_path)
    had_model_version = _has_model_version(db_path)

    init_tracking_schema(db_path)

    after = _table_names(db_path)
    new_tables = sorted(after - before)
    now_has_model_version = _has_model_version(db_path)

    print(f"[OK] Tracking schema applied to {db_path}")
    if new_tables:
        print(f"  Created tables: {', '.join(new_tables)}")
    else:
        print("  All tracking tables already existed")

    if now_has_model_version and not had_model_version:
        print("  Added column: bets.model_version")
    elif now_has_model_version:
        print("  Column bets.model_version already existed")


if __name__ == '__main__':
    main()
