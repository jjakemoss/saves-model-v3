"""
Print a closing-line-value (CLV) report: per-leg and aggregate CLV, split
by straight vs parlay legs and by reason_code, plus rec-level shadow CLV
across every model recommendation (staked or not).

This is the weekly review artifact for the "review by CLV first, P/L
second" rule in docs/OFFSEASON_OPTIMIZATION_PLAN.md section 4.6: CLV is
meant to converge to a real signal in weeks, well before P/L does, so
this report is the thing to actually watch during the measurement
program.

Two CLV metrics throughout, never conflated:
  - saves-line CLV: pure line movement, signed so positive = the bettor
    got the better number for their side.
  - no-vig prob CLV: de-vigged implied probability of the bettor's side
    at bet time vs at close. American odds are never averaged directly
    anywhere in this report -- every probability is derived via
    american_to_implied_prob and de-vigged (normalized against the
    opposite side) before any aggregation.

Plain text output, no emojis.

Usage:
    python scripts/clv_report.py [--db data/betting.db] [--start_date YYYY-MM-DD] [--end_date YYYY-MM-DD]
"""
import sys
from pathlib import Path
import sqlite3
import argparse
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting.tracking_db import DEFAULT_DB_PATH, compute_rec_level_clv, tracking_tables_exist


def _mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def _median_or_none(vals):
    vals = [v for v in vals if v is not None]
    return median(vals) if vals else None


def _fmt(v, fmt):
    return format(v, fmt) if v is not None else 'N/A'


def load_ticket_legs(conn, start_date=None, end_date=None):
    query = """
        SELECT tl.*, t.ticket_type, t.reason_code, t.model_version AS ticket_model_version,
               t.book AS ticket_book, t.status AS ticket_status
        FROM ticket_legs tl
        JOIN tickets t ON t.id = tl.ticket_id
        WHERE 1=1
    """
    params = []
    if start_date:
        query += " AND tl.game_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND tl.game_date <= ?"
        params.append(end_date)
    query += " ORDER BY tl.game_date, tl.ticket_id, tl.leg_number"

    return conn.execute(query, params).fetchall()


def _leg_group(ticket_type):
    return 'straight' if (ticket_type or '').strip().lower() == 'straight' else 'parlay'


def print_per_leg(legs):
    print("PER-LEG CLV")
    print("-" * 100)
    header = f"{'Date':<11} {'Goalie':<14} {'Book':<10} {'Side':<6} {'Bet':<6} {'Close':<6} {'CLV(sv)':<9} {'CLV(pr)':<9} {'Reason':<20}"
    print(header)
    print("-" * 100)
    for leg in legs:
        clv_saves = _fmt(leg['clv_saves'], '+.1f')
        clv_prob = _fmt(leg['clv_prob_novig'], '+.4f')
        bet_line = _fmt(leg['line_at_bet'], '.1f')
        close_line = _fmt(leg['closing_line'], '.1f')
        reason = (leg['reason_code'] or '')[:20]
        print(f"{leg['game_date']:<11} {leg['goalie_name']:<14} {leg['book']:<10} {leg['side']:<6} "
              f"{bet_line:<6} {close_line:<6} {clv_saves:<9} {clv_prob:<9} {reason:<20}")
    print("")


def print_aggregate_by_group(legs):
    print("AGGREGATE CLV BY TICKET TYPE (straight vs parlay)")
    print("-" * 70)
    print(f"{'Group':<12} {'Legs':<6} {'Graded':<8} {'Mean CLV(sv)':<14} {'Median CLV(sv)':<16} {'Mean CLV(pr)':<14}")
    print("-" * 70)
    for group in ('straight', 'parlay'):
        group_legs = [leg for leg in legs if _leg_group(leg['ticket_type']) == group]
        graded = [leg for leg in group_legs if leg['clv_saves'] is not None]
        saves_vals = [leg['clv_saves'] for leg in graded]
        prob_vals = [leg['clv_prob_novig'] for leg in group_legs if leg['clv_prob_novig'] is not None]
        print(f"{group:<12} {len(group_legs):<6} {len(graded):<8} "
              f"{_fmt(_mean_or_none(saves_vals), '+.3f'):<14} "
              f"{_fmt(_median_or_none(saves_vals), '+.3f'):<16} "
              f"{_fmt(_mean_or_none(prob_vals), '+.4f'):<14}")
    print("")


def print_aggregate_by_reason(legs):
    print("AGGREGATE CLV BY REASON CODE")
    print("-" * 80)
    print(f"{'Reason code':<30} {'Legs':<6} {'Graded':<8} {'Mean CLV(sv)':<14} {'Mean CLV(pr)':<14}")
    print("-" * 80)

    reasons = sorted({leg['reason_code'] for leg in legs if leg['reason_code']})
    for reason in reasons:
        reason_legs = [leg for leg in legs if leg['reason_code'] == reason]
        graded = [leg for leg in reason_legs if leg['clv_saves'] is not None]
        saves_vals = [leg['clv_saves'] for leg in graded]
        prob_vals = [leg['clv_prob_novig'] for leg in reason_legs if leg['clv_prob_novig'] is not None]
        print(f"{reason[:30]:<30} {len(reason_legs):<6} {len(graded):<8} "
              f"{_fmt(_mean_or_none(saves_vals), '+.3f'):<14} "
              f"{_fmt(_mean_or_none(prob_vals), '+.4f'):<14}")
    print("")


def print_rec_level_clv(conn, start_date=None, end_date=None):
    print("REC-LEVEL SHADOW CLV (every model recommendation, staked or not)")
    print("-" * 80)

    # compute_rec_level_clv takes a single date; sweep the range if given,
    # otherwise pull the full history.
    if start_date or end_date:
        dates = [row[0] for row in conn.execute(
            "SELECT DISTINCT game_date FROM bets WHERE game_date >= ? AND game_date <= ? ORDER BY game_date",
            (start_date or '0000-00-00', end_date or '9999-99-99')
        ).fetchall()]
        rec_clv = []
        for d in dates:
            rec_clv.extend(compute_rec_level_clv(conn, date=d))
    else:
        rec_clv = compute_rec_level_clv(conn)

    if not rec_clv:
        print("  No recommendations with a matching closing snapshot found")
        print("")
        return

    print(f"{'Model version':<32} {'Recs':<8} {'Mean CLV(sv)':<14} {'Mean CLV(pr)':<14}")
    print("-" * 80)

    versions = sorted({r['model_version'] or 'unknown' for r in rec_clv})
    for version in versions:
        v_recs = [r for r in rec_clv if (r['model_version'] or 'unknown') == version]
        saves_vals = [r['clv_saves'] for r in v_recs if r['clv_saves'] is not None]
        prob_vals = [r['clv_prob_novig'] for r in v_recs if r['clv_prob_novig'] is not None]
        print(f"{version:<32} {len(v_recs):<8} "
              f"{_fmt(_mean_or_none(saves_vals), '+.3f'):<14} "
              f"{_fmt(_mean_or_none(prob_vals), '+.4f'):<14}")

    print("-" * 80)
    all_saves = [r['clv_saves'] for r in rec_clv if r['clv_saves'] is not None]
    all_prob = [r['clv_prob_novig'] for r in rec_clv if r['clv_prob_novig'] is not None]
    print(f"{'ALL':<32} {len(rec_clv):<8} "
          f"{_fmt(_mean_or_none(all_saves), '+.3f'):<14} "
          f"{_fmt(_mean_or_none(all_prob), '+.4f'):<14}")

    print("")
    print("  By side:")
    for side in ('OVER', 'UNDER'):
        side_recs = [r for r in rec_clv if r['recommendation'] == side]
        saves_vals = [r['clv_saves'] for r in side_recs if r['clv_saves'] is not None]
        prob_vals = [r['clv_prob_novig'] for r in side_recs if r['clv_prob_novig'] is not None]
        print(f"    {side:<8} n={len(side_recs):<6} "
              f"mean CLV(sv)={_fmt(_mean_or_none(saves_vals), '+.3f'):<10} "
              f"mean CLV(pr)={_fmt(_mean_or_none(prob_vals), '+.4f')}")
    print("")


def run_report(db_path=DEFAULT_DB_PATH, start_date=None, end_date=None):
    if not tracking_tables_exist(db_path):
        print(f"[ERROR] Tracking tables not found in {db_path}")
        print(f"[ERROR] Run: python scripts/add_tracking_tables.py --db {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row

    try:
        print("=" * 100)
        print("CLV REPORT")
        if start_date or end_date:
            print(f"Range: {start_date or '(start)'} to {end_date or '(end)'}")
        print("=" * 100)
        print("")

        legs = load_ticket_legs(conn, start_date, end_date)
        graded_legs = [leg for leg in legs if leg['result'] is not None]

        print(f"Total ticket legs: {len(legs)} ({len(graded_legs)} graded, {len(legs) - len(graded_legs)} pending)")
        print("")

        if legs:
            print_per_leg(legs)
            print_aggregate_by_group(legs)
            print_aggregate_by_reason(legs)
        else:
            print("No ticket legs recorded yet -- nothing to report at the ticket level.")
            print("")

        print_rec_level_clv(conn, start_date, end_date)

        print("=" * 100)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Print the CLV report (per-leg, aggregate, and shadow-run)')
    parser.add_argument('--db', type=str, default=DEFAULT_DB_PATH, help='Path to betting database. Default: data/betting.db')
    parser.add_argument('--start_date', type=str, default=None, help='Filter: game_date >= this (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='Filter: game_date <= this (YYYY-MM-DD)')

    args = parser.parse_args()

    try:
        run_report(db_path=args.db, start_date=args.start_date, end_date=args.end_date)
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
