"""
SQLite-backed betting tracker storage.

This is the system of record for all predictions, bets, and results.
betting_tracker.xlsx is a generated, read-only view of this database --
see excel_export.py. Nothing should ever write to the xlsx file directly;
all writes go through BettingDB.
"""
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DEFAULT_DB_PATH = 'data/betting.db'

SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date         TEXT NOT NULL,
    game_id           INTEGER NOT NULL,
    book              TEXT NOT NULL,
    goalie_name       TEXT NOT NULL,
    goalie_id         INTEGER,
    team_abbrev       TEXT,
    opponent_team     TEXT,
    is_home           INTEGER,
    betting_line      REAL,
    line_over         INTEGER,
    line_under        INTEGER,
    predicted_saves   REAL,
    prob_over         REAL,
    confidence_pct    REAL,
    confidence_bucket TEXT,
    recommendation    TEXT,
    ev                REAL,
    bet_amount        REAL,
    bet_selection     TEXT DEFAULT 'NONE',
    bet_placed_at     TEXT,
    actual_saves      REAL,
    result            TEXT,
    profit_loss       REAL,
    notes             TEXT,
    UNIQUE(game_id, goalie_name, book, betting_line, line_over, line_under)
);
CREATE INDEX IF NOT EXISTS idx_bets_game_date ON bets(game_date);
CREATE INDEX IF NOT EXISTS idx_bets_game_id ON bets(game_id);
"""


def init_db(db_path=DEFAULT_DB_PATH):
    """Create the database file and schema if they don't already exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()

    return path


def _int_or_none(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


def _float_or_none(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def _clean_or_none(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


class BettingDB:
    """Read/write interface to the betting tracker SQLite database."""

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Betting database not found: {db_path}\n"
                f"Run: python scripts/init_betting_tracker.py"
            )

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def append_games(self, games_df):
        """
        Insert new rows for games/lines that don't already exist.

        Args:
            games_df: pd.DataFrame with columns matching _APPEND_COLUMNS
                      (extra columns are ignored)
        """
        if len(games_df) == 0:
            return

        conn = self._connect()
        inserted = 0
        skipped = 0
        try:
            cur = conn.cursor()
            for _, game in games_df.iterrows():
                try:
                    cur.execute(
                        """
                        INSERT INTO bets (
                            game_date, game_id, book, goalie_name, betting_line,
                            line_over, line_under, goalie_id, team_abbrev,
                            opponent_team, is_home, bet_selection
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'NONE')
                        """,
                        (
                            game.get('game_date'),
                            _int_or_none(game.get('game_id')),
                            game.get('book', ''),
                            game.get('goalie_name'),
                            _float_or_none(game.get('betting_line')),
                            _int_or_none(game.get('line_over')),
                            _int_or_none(game.get('line_under')),
                            _int_or_none(game.get('goalie_id')),
                            game.get('team_abbrev'),
                            game.get('opponent_team'),
                            _int_or_none(game.get('is_home')),
                        )
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    # Already exists (same game/goalie/book/line/odds) -- skip.
                    # Callers are expected to dedupe before calling this, so
                    # this is a safety net, not the primary dedup path.
                    skipped += 1
                    continue
            conn.commit()
        finally:
            conn.close()

        msg = f"[OK] Appended {inserted} games to {self.db_path}"
        if skipped:
            msg += f" (skipped {skipped} duplicates)"
        print(msg)

    def update_predictions(self, predictions_df):
        """
        Fill in prediction columns for rows that don't have one yet.

        Args:
            predictions_df: pd.DataFrame with game_id, goalie_name, book,
                             betting_line, line_over, line_under (match key)
                             plus predicted_saves, prob_over, confidence_pct,
                             confidence_bucket, recommendation, recommended_ev,
                             and optionally model_version (requires
                             scripts/add_tracking_tables.py to have been run;
                             silently ignored on older databases)
        """
        if len(predictions_df) == 0:
            return

        conn = self._connect()
        total_updated = 0
        has_model_version = 'model_version' in predictions_df.columns and \
            any(row[1] == 'model_version' for row in conn.execute("PRAGMA table_info(bets)").fetchall())

        set_clause = "predicted_saves = ?, prob_over = ?, confidence_pct = ?, confidence_bucket = ?, recommendation = ?, ev = ?"
        if has_model_version:
            set_clause += ", model_version = ?"

        query = f"""
            UPDATE bets
            SET {set_clause}
            WHERE game_id = ?
              AND lower(goalie_name) = lower(?)
              AND book = ?
              AND betting_line IS ?
              AND line_over IS ?
              AND line_under IS ?
              AND predicted_saves IS NULL
        """

        try:
            cur = conn.cursor()
            for _, pred in predictions_df.iterrows():
                params = [
                    _float_or_none(pred.get('predicted_saves')),
                    _float_or_none(pred.get('prob_over')),
                    _float_or_none(pred.get('confidence_pct')),
                    _clean_or_none(pred.get('confidence_bucket')),
                    _clean_or_none(pred.get('recommendation')),
                    _float_or_none(pred.get('recommended_ev')),
                ]
                if has_model_version:
                    params.append(_clean_or_none(pred.get('model_version')))
                params.extend([
                    _int_or_none(pred.get('game_id')),
                    pred.get('goalie_name', ''),
                    pred.get('book', ''),
                    _float_or_none(pred.get('betting_line')),
                    _int_or_none(pred.get('line_over')),
                    _int_or_none(pred.get('line_under')),
                ])
                cur.execute(query, params)
                total_updated += cur.rowcount
            conn.commit()
        finally:
            conn.close()

        print(f"[OK] Updated {total_updated} predictions")

    def update_results(self, results_df):
        """
        Fill in actual_saves/result/profit_loss for completed games.

        Args:
            results_df: pd.DataFrame with game_id, goalie_id, book,
                        betting_line, line_over, line_under (match key)
                        plus actual_saves, result, profit_loss
        """
        if len(results_df) == 0:
            return

        conn = self._connect()
        total_updated = 0
        try:
            cur = conn.cursor()
            for _, result in results_df.iterrows():
                cur.execute(
                    """
                    UPDATE bets
                    SET actual_saves = ?,
                        result = ?,
                        profit_loss = ?
                    WHERE game_id = ?
                      AND goalie_id IS ?
                      AND book = ?
                      AND betting_line IS ?
                      AND line_over IS ?
                      AND line_under IS ?
                    """,
                    (
                        _float_or_none(result.get('actual_saves')),
                        _clean_or_none(result.get('result')),
                        _float_or_none(result.get('profit_loss')),
                        _int_or_none(result.get('game_id')),
                        _int_or_none(result.get('goalie_id')),
                        result.get('book', ''),
                        _float_or_none(result.get('betting_line')),
                        _int_or_none(result.get('line_over')),
                        _int_or_none(result.get('line_under')),
                    )
                )
                total_updated += cur.rowcount
            conn.commit()
        finally:
            conn.close()

        print(f"[OK] Updated results for {total_updated} games")

    def record_bet(self, goalie_name, book, bet_selection, bet_amount, date=None, notes=None):
        """
        Record a bet you actually placed, independent of the model's
        recommendation. Matches the most recently fetched line for the
        given date/goalie/book so it still works if the line moved and
        was re-fetched during the day.

        Args:
            goalie_name: Goalie's last name (case-insensitive match)
            book: Sportsbook name (case-insensitive match)
            bet_selection: 'OVER' or 'UNDER'
            bet_amount: Units wagered
            date: Game date (YYYY-MM-DD). Defaults to today.
            notes: Optional notes (replaces any existing notes if provided)

        Returns:
            dict: The updated row

        Raises:
            ValueError: If bet_selection is invalid or no matching row is found
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        if bet_selection not in ('OVER', 'UNDER'):
            raise ValueError(f"bet_selection must be OVER or UNDER, got: {bet_selection!r}")

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id FROM bets
                WHERE game_date = ?
                  AND lower(goalie_name) = lower(?)
                  AND lower(book) = lower(?)
                ORDER BY id DESC
                LIMIT 1
                """,
                (date, goalie_name, book)
            )
            row = cur.fetchone()

            if row is None:
                raise ValueError(
                    f"No line found for date={date}, goalie_name={goalie_name!r}, book={book!r}. "
                    f"Make sure fetch_and_predict.py has already fetched this line for today."
                )

            bet_id = row['id']
            placed_at = datetime.now().isoformat(timespec='seconds')

            if notes:
                cur.execute(
                    """
                    UPDATE bets
                    SET bet_amount = ?, bet_selection = ?, bet_placed_at = ?, notes = ?
                    WHERE id = ?
                    """,
                    (bet_amount, bet_selection, placed_at, notes, bet_id)
                )
            else:
                cur.execute(
                    """
                    UPDATE bets
                    SET bet_amount = ?, bet_selection = ?, bet_placed_at = ?
                    WHERE id = ?
                    """,
                    (bet_amount, bet_selection, placed_at, bet_id)
                )
            conn.commit()

            cur.execute("SELECT * FROM bets WHERE id = ?", (bet_id,))
            updated = dict(cur.fetchone())
        finally:
            conn.close()

        return updated

    def get_todays_games(self, date=None):
        """
        Get all rows for a specific date.

        Args:
            date: Date string (YYYY-MM-DD). If None, uses today

        Returns:
            pd.DataFrame: Rows for that date
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        conn = self._connect()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM bets WHERE game_date = ? ORDER BY id", conn, params=(date,)
            )
        finally:
            conn.close()

        return df

    def get_pending_results(self, date=None):
        """
        Get rows for a date that don't have a result yet.

        Args:
            date: Date string. If None, uses yesterday

        Returns:
            pd.DataFrame: Rows needing results
        """
        if date is None:
            yesterday = datetime.now() - timedelta(days=1)
            date = yesterday.strftime('%Y-%m-%d')

        conn = self._connect()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM bets WHERE game_date = ? AND actual_saves IS NULL ORDER BY id",
                conn, params=(date,)
            )
        finally:
            conn.close()

        return df

    def get_all_bets(self):
        """
        Get every row in the database.

        Returns:
            pd.DataFrame: All rows, ordered by date
        """
        conn = self._connect()
        try:
            df = pd.read_sql_query("SELECT * FROM bets ORDER BY game_date, id", conn)
        finally:
            conn.close()

        return df
