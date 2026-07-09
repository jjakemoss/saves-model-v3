"""
Schema and storage layer for ticket/CLV tracking.

Adds three tables on top of the existing `bets` schema in db_manager.py,
plus one additive column on `bets` itself:

  - line_snapshots: one row per (fetch run, book, goalie, market line) --
    the CLV backbone. Every fetch_and_predict.py run appends one row per
    matched line here (regardless of whether that line is new or a
    duplicate of the last-known bets row), so line-move history and
    closing lines can be reconstructed after the fact.
  - tickets / ticket_legs: actual ticket economics (stake, payout,
    reason code, per-leg closing line and CLV) independent of the `bets`
    table, which only tracks a single implicit bet per row and has no
    concept of parlay/ticket structure.
  - bets.model_version: additive column so recommendation rows (the
    "shadow run" log -- every prediction, staked or not) are attributable
    to the model that produced them once model versions change.

scripts/add_tracking_tables.py applies this schema. It is idempotent and
never touches existing bets rows/columns beyond adding model_version.

This module also owns the CLV math: de-vigged (no-vig) implied
probability from an American odds pair, and closing-line lookup. Odds
are never arithmetically averaged -- see the odds-averaging bug in
docs/HISTORICAL_DATA_ANALYSIS.md section 1. Any aggregation of odds
first converts to implied probability (via american_to_implied_prob) and
de-vigs by normalizing the over/under pair (devig_prob below).
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from .odds_utils import american_to_implied_prob

DEFAULT_DB_PATH = 'data/betting.db'

SCHEMA_TRACKING = """
CREATE TABLE IF NOT EXISTS line_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at_utc      TEXT NOT NULL,
    game_date           TEXT NOT NULL,
    game_id             INTEGER,
    book                TEXT NOT NULL,
    goalie_name         TEXT NOT NULL,
    goalie_id           INTEGER,
    team_abbrev         TEXT,
    opponent_team       TEXT,
    is_home             INTEGER,
    betting_line        REAL,
    odds_over           INTEGER,
    odds_under          INTEGER,
    scheduled_start_utc TEXT,
    fetch_source        TEXT
);
CREATE INDEX IF NOT EXISTS idx_line_snapshots_game_date ON line_snapshots(game_date);
CREATE INDEX IF NOT EXISTS idx_line_snapshots_game_id ON line_snapshots(game_id);
CREATE INDEX IF NOT EXISTS idx_line_snapshots_goalie_book ON line_snapshots(goalie_name, book);
CREATE INDEX IF NOT EXISTS idx_line_snapshots_fetched_at ON line_snapshots(fetched_at_utc);

CREATE TABLE IF NOT EXISTS tickets (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at_utc       TEXT NOT NULL,
    book                TEXT NOT NULL,
    ticket_type         TEXT NOT NULL,
    stake               REAL NOT NULL,
    payout_multiplier   REAL,
    potential_payout    REAL,
    status              TEXT NOT NULL DEFAULT 'pending',
    actual_payout       REAL,
    reason_code         TEXT NOT NULL,
    model_version       TEXT,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_tickets_placed_at ON tickets(placed_at_utc);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);

CREATE TABLE IF NOT EXISTS ticket_legs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id            INTEGER NOT NULL REFERENCES tickets(id),
    leg_number           INTEGER NOT NULL,
    game_date            TEXT NOT NULL,
    game_id              INTEGER,
    goalie_name          TEXT NOT NULL,
    goalie_id            INTEGER,
    team_abbrev          TEXT,
    book                 TEXT NOT NULL,
    side                 TEXT NOT NULL,
    line_at_bet          REAL,
    odds_at_bet          INTEGER,
    line_snapshot_id     INTEGER REFERENCES line_snapshots(id),
    result               TEXT,
    actual_saves         REAL,
    closing_line         REAL,
    closing_odds         INTEGER,
    closing_snapshot_id  INTEGER REFERENCES line_snapshots(id),
    clv_saves            REAL,
    clv_prob_novig       REAL,
    UNIQUE(ticket_id, leg_number)
);
CREATE INDEX IF NOT EXISTS idx_ticket_legs_ticket_id ON ticket_legs(ticket_id);
CREATE INDEX IF NOT EXISTS idx_ticket_legs_game_date ON ticket_legs(game_date);
"""


def _has_column(conn, table, column):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def init_tracking_schema(db_path=DEFAULT_DB_PATH):
    """
    Create the tracking tables and the additive bets.model_version column
    if they don't already exist. Idempotent -- safe to run repeatedly.
    Never touches existing bets rows or columns beyond adding the one
    new column.
    """
    path = Path(db_path)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA_TRACKING)
        if not _has_column(conn, 'bets', 'model_version'):
            conn.execute("ALTER TABLE bets ADD COLUMN model_version TEXT")
        conn.commit()
    finally:
        conn.close()
    return path


def tracking_tables_exist(db_path):
    """
    True if all three tracking tables (line_snapshots, tickets,
    ticket_legs) exist in the database. Scripts that depend on the
    tracking schema check this up front so an un-migrated database
    (scripts/add_tracking_tables.py not yet run) produces a clear
    remedy message instead of a raw sqlite3.OperationalError.
    """
    conn = sqlite3.connect(Path(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('line_snapshots', 'tickets', 'ticket_legs')"
        )
        return {row[0] for row in cur.fetchall()} == {'line_snapshots', 'tickets', 'ticket_legs'}
    finally:
        conn.close()


def utc_now_iso():
    """Current UTC time as a normalized 'YYYY-MM-DDTHH:MM:SSZ' string."""
    return normalize_utc_iso(datetime.now(timezone.utc))


def normalize_utc_iso(ts):
    """
    Normalize a variety of ISO8601 UTC timestamp representations (trailing
    'Z', trailing '+00:00', with/without fractional seconds, a datetime
    object) to a single consistent 'YYYY-MM-DDTHH:MM:SSZ' string.

    This matters because line_snapshots.fetched_at_utc/scheduled_start_utc
    and tickets.placed_at_utc are compared with plain string
    ORDER BY / <= in the closing-line lookup below. Underdog and
    The-Odds-API both emit 'Z'-suffixed timestamps, while Python's
    datetime.isoformat() on a tz-aware datetime emits '+00:00' -- those
    two suffix styles do NOT sort consistently against each other as
    strings ('Z' > '+00:00' lexicographically for every timestamp, not
    just some), which would silently corrupt "last snapshot before puck
    drop" lookups. Normalizing everything to the same 'Z' form on the way
    into the database avoids that.

    Returns None if ts is None or unparseable.
    """
    if ts is None:
        return None

    if isinstance(ts, datetime):
        dt = ts
    else:
        s = str(ts).strip()
        if not s:
            return None
        s = s.replace('Z', '+00:00')
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def insert_line_snapshots(db_path, rows, fetched_at_utc=None, fetch_source='fetch_and_predict'):
    """
    Append one line_snapshots row per entry in `rows`. This is the CLV
    backbone: called once per fetch run with every matched line (not just
    newly-added ones), so line-move history exists even for lines that
    didn't change between fetches.

    Args:
        db_path: path to betting.db
        rows: list of dicts with game_date, game_id, book, goalie_name,
              goalie_id, team_abbrev, opponent_team, is_home,
              betting_line, and odds for both sides under either
              odds_over/odds_under or line_over/line_under (the bets-table
              names -- accepted so callers can pass the same dicts they
              already build for tracker.append_games without renaming
              keys). scheduled_start_utc is optional.
        fetched_at_utc: ISO8601 UTC timestamp shared by every row in this
                        batch. Defaults to now.
        fetch_source: free-text label for which script/run inserted this batch.

    Returns:
        int: number of rows inserted
    """
    if not rows:
        return 0

    fetched_at_utc = normalize_utc_iso(fetched_at_utc) or utc_now_iso()

    conn = sqlite3.connect(Path(db_path))
    inserted = 0
    try:
        cur = conn.cursor()
        for row in rows:
            odds_over = row.get('odds_over', row.get('line_over'))
            odds_under = row.get('odds_under', row.get('line_under'))
            cur.execute(
                """
                INSERT INTO line_snapshots (
                    fetched_at_utc, game_date, game_id, book, goalie_name,
                    goalie_id, team_abbrev, opponent_team, is_home,
                    betting_line, odds_over, odds_under,
                    scheduled_start_utc, fetch_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fetched_at_utc,
                    row.get('game_date'),
                    row.get('game_id'),
                    row.get('book'),
                    row.get('goalie_name'),
                    row.get('goalie_id'),
                    row.get('team_abbrev'),
                    row.get('opponent_team'),
                    row.get('is_home'),
                    row.get('betting_line'),
                    odds_over,
                    odds_under,
                    normalize_utc_iso(row.get('scheduled_start_utc')),
                    fetch_source,
                )
            )
            inserted += 1
        conn.commit()
    finally:
        conn.close()

    return inserted


def devig_prob(odds_side, odds_other):
    """
    Convert a pair of American odds (opposite sides of the same market)
    to no-vig (de-vigged) implied probabilities via additive
    normalization. Never averages American odds directly.

    Returns:
        (novig_prob_side, novig_prob_other), or (None, None) if either
        odds value is missing or the pair is degenerate.
    """
    if odds_side is None or odds_other is None:
        return None, None

    p_side = american_to_implied_prob(odds_side)
    p_other = american_to_implied_prob(odds_other)
    total = p_side + p_other
    if total <= 0:
        return None, None

    return p_side / total, p_other / total


def find_closing_snapshot(conn, game_date, goalie_name, book):
    """
    Return the sqlite3.Row of the closing snapshot for a goalie/book/night:
    the last snapshot fetched at or before scheduled_start_utc, or -- if
    no snapshot in the group carries a scheduled_start_utc -- the last
    snapshot of the day. Returns None if there are no snapshots at all
    for this goalie/book/date.
    """
    cur = conn.execute(
        """
        SELECT * FROM line_snapshots
        WHERE game_date = ? AND lower(goalie_name) = lower(?) AND lower(book) = lower(?)
        ORDER BY fetched_at_utc ASC
        """,
        (game_date, goalie_name, book)
    )
    rows = cur.fetchall()
    if not rows:
        return None

    scheduled_start = next((r['scheduled_start_utc'] for r in rows if r['scheduled_start_utc']), None)

    if scheduled_start:
        before_start = [r for r in rows if r['fetched_at_utc'] and r['fetched_at_utc'] <= scheduled_start]
        candidates = before_start if before_start else rows
    else:
        candidates = rows

    return candidates[-1]


def find_latest_snapshot_at_or_before(conn, game_date, goalie_name, book, at_utc):
    """Latest snapshot for this goalie/book/date at or before `at_utc`."""
    at_utc = normalize_utc_iso(at_utc)
    cur = conn.execute(
        """
        SELECT * FROM line_snapshots
        WHERE game_date = ? AND lower(goalie_name) = lower(?) AND lower(book) = lower(?)
          AND fetched_at_utc <= ?
        ORDER BY fetched_at_utc DESC
        LIMIT 1
        """,
        (game_date, goalie_name, book, at_utc)
    )
    return cur.fetchone()


def find_latest_snapshot(conn, game_date, goalie_name, book):
    """Latest snapshot for this goalie/book/date, regardless of time."""
    cur = conn.execute(
        """
        SELECT * FROM line_snapshots
        WHERE game_date = ? AND lower(goalie_name) = lower(?) AND lower(book) = lower(?)
        ORDER BY fetched_at_utc DESC
        LIMIT 1
        """,
        (game_date, goalie_name, book)
    )
    return cur.fetchone()


def compute_leg_clv(conn, leg_row):
    """
    Given a ticket_legs row (sqlite3.Row), find the closing snapshot and
    return a dict of closing_line, closing_odds, closing_snapshot_id,
    clv_saves, clv_prob_novig -- ready to UPDATE onto the leg.

    Sign convention: positive clv_saves/clv_prob_novig always means the
    bettor got the better number/price for THEIR side, regardless of
    whether they bet OVER or UNDER.

      clv_saves = closing_line - line_at_bet   if side == OVER
                = line_at_bet - closing_line   if side == UNDER

    (an OVER bettor wants the market to have moved the total UP after
    they locked in a lower number; an UNDER bettor wants the opposite.)

    clv_prob_novig compares the de-vigged probability of the bettor's
    side at bet time (using the bet-time snapshot's odds pair, if the leg
    is linked to one) against the de-vigged probability of that same
    side at the closing snapshot. Positive means the market moved toward
    the bettor's side after they bet -- they got in before the market
    priced their side higher. This mixes line movement and price
    movement into one probability signal (unlike clv_saves, which
    isolates pure line movement) since totals lines move the number
    itself rather than holding it fixed and moving only the price.

    Returns None if there is no snapshot at all for this goalie/book/date
    (nothing to close against yet).
    """
    closing = find_closing_snapshot(conn, leg_row['game_date'], leg_row['goalie_name'], leg_row['book'])
    if closing is None:
        return None

    side = leg_row['side']
    closing_line = closing['betting_line']
    closing_odds = closing['odds_over'] if side == 'OVER' else closing['odds_under']

    line_at_bet = leg_row['line_at_bet']
    clv_saves = None
    if closing_line is not None and line_at_bet is not None:
        if side == 'OVER':
            clv_saves = closing_line - line_at_bet
        else:
            clv_saves = line_at_bet - closing_line

    clv_prob_novig = None
    bet_snapshot_id = leg_row['line_snapshot_id']
    bet_snap = None
    if bet_snapshot_id is not None:
        bet_snap = conn.execute(
            "SELECT * FROM line_snapshots WHERE id = ?", (bet_snapshot_id,)
        ).fetchone()

    if bet_snap is not None:
        bet_odds_side = bet_snap['odds_over'] if side == 'OVER' else bet_snap['odds_under']
        bet_odds_other = bet_snap['odds_under'] if side == 'OVER' else bet_snap['odds_over']
        close_odds_side = closing['odds_over'] if side == 'OVER' else closing['odds_under']
        close_odds_other = closing['odds_under'] if side == 'OVER' else closing['odds_over']

        bet_novig_side, _ = devig_prob(bet_odds_side, bet_odds_other)
        close_novig_side, _ = devig_prob(close_odds_side, close_odds_other)

        if bet_novig_side is not None and close_novig_side is not None:
            clv_prob_novig = close_novig_side - bet_novig_side

    return {
        'closing_line': closing_line,
        'closing_odds': closing_odds,
        'closing_snapshot_id': closing['id'],
        'clv_saves': clv_saves,
        'clv_prob_novig': clv_prob_novig,
    }


def leg_result(side, line_at_bet, actual_saves):
    """WIN/LOSS/PUSH for a single leg, or None if inputs are missing."""
    if line_at_bet is None or actual_saves is None:
        return None
    if actual_saves == line_at_bet:
        return 'PUSH'
    over_hit = actual_saves > line_at_bet
    if side == 'OVER':
        return 'WIN' if over_hit else 'LOSS'
    return 'LOSS' if over_hit else 'WIN'


def settle_ticket_if_complete(conn, ticket_id):
    """
    If every leg on this ticket has a result, compute and store the
    ticket's status/actual_payout.

    Simplifying assumption: this does not model reduced/flex payout
    tables for parlays where some legs push -- books each have their own
    reduced-payout schedule for N-1 leg parlays, which isn't stored
    anywhere in this schema. Those cases are marked 'partial' with
    actual_payout left NULL for manual entry.

    Returns the new status, or None if the ticket isn't fully graded yet.
    """
    legs = conn.execute(
        "SELECT * FROM ticket_legs WHERE ticket_id = ?", (ticket_id,)
    ).fetchall()
    if not legs or any(leg['result'] is None for leg in legs):
        return None

    results = [leg['result'] for leg in legs]
    ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()

    if any(r == 'LOSS' for r in results):
        status, actual_payout = 'lost', 0.0
    elif all(r == 'WIN' for r in results):
        status, actual_payout = 'won', ticket['potential_payout']
    elif all(r == 'PUSH' for r in results):
        status, actual_payout = 'push', ticket['stake']
    else:
        # Mix of WIN and PUSH, no LOSS -- a reduced-leg parlay payout.
        status, actual_payout = 'partial', None

    conn.execute(
        "UPDATE tickets SET status = ?, actual_payout = ? WHERE id = ?",
        (status, actual_payout, ticket_id)
    )
    return status


def insert_ticket(db_path, placed_at_utc, book, ticket_type, stake, reason_code,
                   payout_multiplier=None, potential_payout=None, model_version=None,
                   notes=None, status='pending'):
    """
    Create a ticket row. reason_code is required -- every ticket must be
    attributable to a specific rationale (see docs/OFFSEASON_OPTIMIZATION_PLAN.md
    section 4.6, rule 4). Exactly one of payout_multiplier/potential_payout
    is required; the other is derived from stake.

    Returns the new ticket id.
    """
    if not reason_code or not str(reason_code).strip():
        raise ValueError("reason_code is required for every ticket")
    if payout_multiplier is None and potential_payout is None:
        raise ValueError("must supply payout_multiplier or potential_payout")
    if payout_multiplier is None:
        payout_multiplier = potential_payout / stake if stake else None
    if potential_payout is None:
        potential_payout = stake * payout_multiplier

    placed_at_utc = normalize_utc_iso(placed_at_utc) or utc_now_iso()

    conn = sqlite3.connect(Path(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tickets (
                placed_at_utc, book, ticket_type, stake, payout_multiplier,
                potential_payout, status, actual_payout, reason_code,
                model_version, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (placed_at_utc, book, ticket_type, stake, payout_multiplier,
             potential_payout, status, reason_code, model_version, notes)
        )
        ticket_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    return ticket_id


def insert_ticket_leg(db_path, ticket_id, leg_number, placed_at_utc, game_date,
                       goalie_name, team_abbrev, book, side,
                       line_at_bet=None, odds_at_bet=None):
    """
    Insert one ticket_legs row. Auto-matches the most recent
    line_snapshots row at or before placed_at_utc for
    (game_date, goalie_name, book) to fill game_id/goalie_id and,
    if not given explicitly, line_at_bet/odds_at_bet. Falls back to the
    latest snapshot regardless of time if none exists before
    placed_at_utc (best effort -- still better than no link at all).

    Returns the new ticket_legs id.
    """
    if side not in ('OVER', 'UNDER'):
        raise ValueError(f"side must be OVER or UNDER, got {side!r}")

    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    try:
        snap = find_latest_snapshot_at_or_before(conn, game_date, goalie_name, book, placed_at_utc)
        if snap is None:
            snap = find_latest_snapshot(conn, game_date, goalie_name, book)

        line_snapshot_id = snap['id'] if snap is not None else None
        game_id = snap['game_id'] if snap is not None else None
        goalie_id = snap['goalie_id'] if snap is not None else None

        if line_at_bet is None and snap is not None:
            line_at_bet = snap['betting_line']
        if odds_at_bet is None and snap is not None:
            odds_at_bet = snap['odds_over'] if side == 'OVER' else snap['odds_under']
        if not team_abbrev and snap is not None:
            team_abbrev = snap['team_abbrev']

        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ticket_legs (
                ticket_id, leg_number, game_date, game_id, goalie_name,
                goalie_id, team_abbrev, book, side, line_at_bet,
                odds_at_bet, line_snapshot_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, leg_number, game_date, game_id, goalie_name,
             goalie_id, team_abbrev, book, side, line_at_bet,
             odds_at_bet, line_snapshot_id)
        )
        leg_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    return leg_id, line_snapshot_id


def get_ticket_leg(db_path, leg_id):
    """Fetch a single ticket_legs row by id (used to display the actual
    persisted, auto-matched values after insert_ticket_leg)."""
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM ticket_legs WHERE id = ?", (leg_id,)).fetchone()
    finally:
        conn.close()


def compute_rec_level_clv(conn, date=None):
    """
    Shadow-run CLV: for every `bets` row with an OVER/UNDER recommendation
    (whether or not it was actually staked), look up the closing snapshot
    for that goalie/book/night and compute the CLV the recommendation
    would have realized. Independent of whether a ticket was ever placed
    -- this is what settles whether the live (model + feature pipeline)
    combination beats the close, per
    docs/OFFSEASON_OPTIMIZATION_PLAN.md section 3.11.

    Returns a list of dicts (not persisted anywhere -- recomputed on
    demand since it's a cheap join over bets + line_snapshots).
    """
    query = """
        SELECT id, game_date, goalie_name, book, recommendation,
               betting_line, line_over, line_under, model_version
        FROM bets
        WHERE recommendation IS NOT NULL AND recommendation != 'NO BET'
    """
    params = ()
    if date:
        query += " AND game_date = ?"
        params = (date,)

    rows = conn.execute(query, params).fetchall()
    results = []
    for row in rows:
        side = row['recommendation']
        if side not in ('OVER', 'UNDER'):
            continue

        closing = find_closing_snapshot(conn, row['game_date'], row['goalie_name'], row['book'])
        if closing is None:
            continue

        line_at_rec = row['betting_line']
        closing_line = closing['betting_line']
        clv_saves = None
        if line_at_rec is not None and closing_line is not None:
            clv_saves = (closing_line - line_at_rec) if side == 'OVER' else (line_at_rec - closing_line)

        odds_side = row['line_over'] if side == 'OVER' else row['line_under']
        odds_other = row['line_under'] if side == 'OVER' else row['line_over']
        close_odds_side = closing['odds_over'] if side == 'OVER' else closing['odds_under']
        close_odds_other = closing['odds_under'] if side == 'OVER' else closing['odds_over']

        rec_novig, _ = devig_prob(odds_side, odds_other)
        close_novig, _ = devig_prob(close_odds_side, close_odds_other)
        clv_prob_novig = None
        if rec_novig is not None and close_novig is not None:
            clv_prob_novig = close_novig - rec_novig

        results.append({
            'bet_id': row['id'],
            'game_date': row['game_date'],
            'goalie_name': row['goalie_name'],
            'book': row['book'],
            'recommendation': side,
            'model_version': row['model_version'],
            'betting_line': line_at_rec,
            'closing_line': closing_line,
            'clv_saves': clv_saves,
            'clv_prob_novig': clv_prob_novig,
        })

    return results
