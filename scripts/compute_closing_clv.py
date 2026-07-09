"""
Compute closing lines and CLV for ticket legs, settle completed tickets,
and summarize shadow-run (recommendation-level) CLV.

Run this after games complete -- the natural place is right after
update_betting_results.py in the "Update Betting Results" workflow, since
both need the same completed-game boxscore data. For every ticket leg
without a result yet (defaulting to yesterday, same convention as
update_betting_results.py), this:

  1. Finds the closing line_snapshots row for that goalie/book/night:
     the last snapshot fetched before the scheduled start time, or the
     last snapshot of the day if no start time was ever captured.
  2. Fills actual_saves/result on the leg from the NHL boxscore (same
     lookup update_betting_results.py already uses).
  3. Fills closing_line/closing_odds/clv_saves/clv_prob_novig.
  4. Settles the parent ticket's status/actual_payout once every leg on
     it has a result.

It also fills closing/CLV for legs that already have a result but are
still missing closing/CLV (e.g. a snapshot landed late), and prints a
rec-level (shadow-run) CLV summary: the same closing-line lookup applied
to every `bets` recommendation, whether or not it was actually staked.
That summary is not persisted -- it's a cheap join over bets +
line_snapshots, recomputed here and again (in more detail) by
scripts/clv_report.py.

This script never touches the model, predictor, or bets recommendation
columns -- it only reads bets/ticket_legs/line_snapshots and writes to
the new tracking tables.

Usage:
    python scripts/compute_closing_clv.py [--date YYYY-MM-DD] [--db data/betting.db]
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting import NHLBettingData
from betting.tracking_db import (
    DEFAULT_DB_PATH, compute_leg_clv, leg_result, settle_ticket_if_complete,
    compute_rec_level_clv, tracking_tables_exist,
)
from data.api_client import RateLimitError


def _fetch_actual_saves(nhl_data, game_result_cache, game_id, goalie_id):
    if game_id is None or goalie_id is None:
        return None
    if game_id not in game_result_cache:
        game_result_cache[game_id] = nhl_data.get_game_result(game_id)
    return game_result_cache[game_id].get(goalie_id)


def compute_closing_and_clv(date=None, db_path=DEFAULT_DB_PATH):
    """
    Returns (legs_updated, tickets_settled) counts.
    """
    if date is None:
        yesterday = datetime.now() - timedelta(days=1)
        date = yesterday.strftime('%Y-%m-%d')

    print(f"\nComputing closing lines / CLV for {date}...")

    # An un-migrated database is not an error for this script: it runs
    # unconditionally in the update_results workflow, and failing here
    # would block the results commit that step exists to make. Warn,
    # point at the migration, and succeed with nothing to do.
    if not tracking_tables_exist(db_path):
        print(f"  [WARNING] Tracking tables not found in {db_path} -- nothing to compute.")
        print(f"  [WARNING] Run: python scripts/add_tracking_tables.py --db {db_path} to enable CLV tracking")
        return 0, 0

    nhl_data = NHLBettingData()
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row

    game_result_cache = {}
    legs_updated = 0
    touched_tickets = set()

    try:
        pending_legs = conn.execute(
            "SELECT * FROM ticket_legs WHERE game_date = ? AND result IS NULL",
            (date,)
        ).fetchall()
        print(f"  {len(pending_legs)} ticket leg(s) need a result")

        for leg in pending_legs:
            goalie_id = leg['goalie_id']
            game_id = leg['game_id']

            if goalie_id is None:
                goalie_id = nhl_data.get_goalie_id_by_name(leg['goalie_name'])

            if game_id is None or goalie_id is None:
                print(f"  [SKIP] leg {leg['id']} ({leg['goalie_name']}) -- "
                      f"missing game_id/goalie_id (no matching line_snapshot at bet time)")
                continue

            try:
                saves = _fetch_actual_saves(nhl_data, game_result_cache, game_id, goalie_id)
            except RateLimitError as e:
                print(f"\n[FATAL] NHL API rate limit exhausted: {e}")
                print("[FATAL] Aborting to avoid partial CLV writes.")
                sys.exit(1)

            if saves is None:
                print(f"  [SKIP] leg {leg['id']} ({leg['goalie_name']}) -- goalie not in boxscore yet")
                continue

            result = leg_result(leg['side'], leg['line_at_bet'], saves)
            clv = compute_leg_clv(conn, leg)

            conn.execute(
                """
                UPDATE ticket_legs
                SET actual_saves = ?, result = ?, closing_line = ?, closing_odds = ?,
                    closing_snapshot_id = ?, clv_saves = ?, clv_prob_novig = ?
                WHERE id = ?
                """,
                (
                    saves, result,
                    clv['closing_line'] if clv else None,
                    clv['closing_odds'] if clv else None,
                    clv['closing_snapshot_id'] if clv else None,
                    clv['clv_saves'] if clv else None,
                    clv['clv_prob_novig'] if clv else None,
                    leg['id'],
                )
            )
            conn.commit()
            legs_updated += 1
            touched_tickets.add(leg['ticket_id'])

            clv_str = f"{clv['clv_saves']:+.1f} saves" if clv and clv['clv_saves'] is not None else "N/A"
            print(f"  [OK] leg {leg['id']} ({leg['goalie_name']} {leg['side']} {leg['line_at_bet']}) "
                  f"-> {result}, saves={saves}, CLV={clv_str}")

        # Backfill closing/CLV for legs that already have a result but are
        # still missing it (e.g. this script ran before a closing snapshot
        # existed and is being re-run).
        stale_closing = conn.execute(
            "SELECT * FROM ticket_legs WHERE game_date = ? AND result IS NOT NULL AND closing_line IS NULL",
            (date,)
        ).fetchall()
        for leg in stale_closing:
            clv = compute_leg_clv(conn, leg)
            if clv is None:
                continue
            conn.execute(
                """
                UPDATE ticket_legs
                SET closing_line = ?, closing_odds = ?, closing_snapshot_id = ?,
                    clv_saves = ?, clv_prob_novig = ?
                WHERE id = ?
                """,
                (clv['closing_line'], clv['closing_odds'], clv['closing_snapshot_id'],
                 clv['clv_saves'], clv['clv_prob_novig'], leg['id'])
            )
            conn.commit()
            touched_tickets.add(leg['ticket_id'])
            print(f"  [OK] backfilled closing/CLV for leg {leg['id']} ({leg['goalie_name']})")

        tickets_settled = 0
        for ticket_id in touched_tickets:
            status = settle_ticket_if_complete(conn, ticket_id)
            if status:
                conn.commit()
                tickets_settled += 1
                print(f"  Ticket {ticket_id} settled: {status}")

        # Rec-level (shadow-run) CLV summary -- every recommendation,
        # staked or not, whether it beat the close.
        rec_clv = compute_rec_level_clv(conn, date=date)
        saves_vals = [r['clv_saves'] for r in rec_clv if r['clv_saves'] is not None]
        prob_vals = [r['clv_prob_novig'] for r in rec_clv if r['clv_prob_novig'] is not None]

        print(f"\n  Shadow-run rec-level CLV for {date}: {len(rec_clv)} recommendation(s) with a closing snapshot")
        if saves_vals:
            print(f"    Mean saves-line CLV:   {sum(saves_vals) / len(saves_vals):+.3f} (n={len(saves_vals)})")
        if prob_vals:
            print(f"    Mean no-vig prob CLV:  {sum(prob_vals) / len(prob_vals):+.4f} (n={len(prob_vals)})")
        if not saves_vals and not prob_vals:
            print(f"    No closing snapshots found yet for this date's recommendations")

    finally:
        conn.close()

    print(f"\n[OK] Updated {legs_updated} ticket leg(s), settled {len(touched_tickets)} ticket(s) touched")
    return legs_updated, len(touched_tickets)


def main():
    parser = argparse.ArgumentParser(
        description='Compute closing lines / CLV for ticket legs and settle completed tickets'
    )
    parser.add_argument('--date', type=str, help="Date to compute for (YYYY-MM-DD). Default: yesterday")
    parser.add_argument('--db', type=str, default=DEFAULT_DB_PATH, help='Path to betting database. Default: data/betting.db')

    args = parser.parse_args()

    try:
        compute_closing_and_clv(date=args.date, db_path=args.db)
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
