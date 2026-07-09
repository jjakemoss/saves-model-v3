"""
Record a ticket (1-3 legs) you actually placed, with full ticket
economics -- stake, payout, and a required reason code.

This is the tickets/CLV sibling of record_bet.py. record_bet.py logs a
single side against the existing `bets` row it matches, with no concept
of stake structure beyond units wagered; this script creates a new
`tickets` row plus 1-3 `ticket_legs` rows so parlay economics (stake,
payout multiplier, ticket status) and per-leg closing-line value are
actually representable. Both scripts keep working independently --
record_bet.py is still the quick path for a single line you want noted
against the model's recommendation without full ticket accounting.

Each leg is auto-matched against the most recent line_snapshots row for
(game_date, goalie_name, book) at or before the ticket's placed time, to
fill in game_id/goalie_id and, if not given explicitly, line_at_bet/
odds_at_bet. Run scripts/fetch_and_predict.py earlier in the day first so
a snapshot exists to match against.

Compact leg syntax -- one leg per ';'-separated entry:

    GoalieLastName:TEAM:SIDE:LINE[:ODDS[:BOOK]]

    SIDE is OVER or UNDER. LINE is the saves total (e.g. 24.5). ODDS and
    BOOK are optional: ODDS is looked up from the matched line_snapshots
    row if omitted, and BOOK defaults to the ticket's --book if omitted.

Usage:
    python scripts/record_ticket.py --book Underdog --ticket_type straight \\
        --stake 2 --payout_multiplier 1.91 --reason_code "market-anchor model edge" \\
        --legs "Shesterkin:NYR:OVER:24.5"

    python scripts/record_ticket.py --book Underdog --ticket_type parlay_2 \\
        --stake 1 --potential_payout 3 --reason_code "line-shop gap" \\
        --legs "Shesterkin:NYR:OVER:24.5;Hellebuyck:WPG:UNDER:26.5:-110"
"""
import sys
from pathlib import Path
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting.tracking_db import (
    DEFAULT_DB_PATH, insert_ticket, insert_ticket_leg, utc_now_iso, get_ticket_leg,
    tracking_tables_exist,
)

MAX_LEGS = 3


def parse_legs(legs_str, default_book, default_date):
    """
    Parse the compact 'GoalieLastName:TEAM:SIDE:LINE[:ODDS[:BOOK]]'
    syntax, ';'-separated, into a list of leg dicts.
    """
    legs = []
    for raw in legs_str.split(';'):
        raw = raw.strip()
        if not raw:
            continue

        fields = raw.split(':')
        if len(fields) < 4 or len(fields) > 6:
            raise ValueError(
                f"Bad leg syntax: {raw!r}. Expected "
                f"GoalieLastName:TEAM:SIDE:LINE[:ODDS[:BOOK]]"
            )

        goalie_name, team_abbrev, side, line_str = fields[:4]
        odds_str = fields[4] if len(fields) >= 5 else ''
        book = fields[5] if len(fields) == 6 else ''

        side = side.strip().upper()
        if side not in ('OVER', 'UNDER'):
            raise ValueError(f"Bad leg syntax: {raw!r} -- side must be OVER or UNDER, got {side!r}")

        try:
            line_at_bet = float(line_str)
        except ValueError:
            raise ValueError(f"Bad leg syntax: {raw!r} -- line must be numeric, got {line_str!r}")

        odds_at_bet = None
        if odds_str.strip():
            try:
                odds_at_bet = int(odds_str.strip())
            except ValueError:
                raise ValueError(f"Bad leg syntax: {raw!r} -- odds must be an integer, got {odds_str!r}")

        legs.append({
            'game_date': default_date,
            'goalie_name': goalie_name.strip(),
            'team_abbrev': team_abbrev.strip().upper() or None,
            'book': book.strip() or default_book,
            'side': side,
            'line_at_bet': line_at_bet,
            'odds_at_bet': odds_at_bet,
        })

    if not legs:
        raise ValueError("No legs parsed from --legs")
    if len(legs) > MAX_LEGS:
        raise ValueError(f"Ticket has {len(legs)} legs; max is {MAX_LEGS}")

    return legs


def record_ticket(book, ticket_type, stake, reason_code, legs_str, date=None,
                   payout_multiplier=None, potential_payout=None,
                   model_version=None, notes=None, db_path=DEFAULT_DB_PATH):
    """
    Record a ticket and its legs. Returns (ticket_id, list of leg ids).
    """
    if not tracking_tables_exist(db_path):
        raise ValueError(
            f"Tracking tables not found in {db_path}. "
            f"Run: python scripts/add_tracking_tables.py --db {db_path}"
        )

    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    legs = parse_legs(legs_str, default_book=book, default_date=date)

    # Soft check for the "one goalie per game per ticket" discipline rule
    # (docs/OFFSEASON_OPTIMIZATION_PLAN.md section 4.6, rule 5) -- warn,
    # don't block, since this is a measurement program and we'd rather
    # record an override than lose the data.
    seen = set()
    for leg in legs:
        key = (leg['game_date'], leg['goalie_name'].lower())
        if key in seen:
            print(f"  [WARNING] Multiple legs on {leg['goalie_name']} ({leg['game_date']}) in one ticket "
                  f"-- correlated same-game exposure (see plan doc section 4.6, rule 5)")
        seen.add(key)

    placed_at_utc = utc_now_iso()

    ticket_id = insert_ticket(
        db_path=db_path,
        placed_at_utc=placed_at_utc,
        book=book,
        ticket_type=ticket_type,
        stake=stake,
        reason_code=reason_code,
        payout_multiplier=payout_multiplier,
        potential_payout=potential_payout,
        model_version=model_version,
        notes=notes,
    )

    leg_ids = []
    unmatched_legs = []
    persisted_legs = []
    for i, leg in enumerate(legs, start=1):
        leg_id, line_snapshot_id = insert_ticket_leg(
            db_path=db_path,
            ticket_id=ticket_id,
            leg_number=i,
            placed_at_utc=placed_at_utc,
            game_date=leg['game_date'],
            goalie_name=leg['goalie_name'],
            team_abbrev=leg['team_abbrev'],
            book=leg['book'],
            side=leg['side'],
            line_at_bet=leg['line_at_bet'],
            odds_at_bet=leg['odds_at_bet'],
        )
        leg_ids.append(leg_id)
        # Re-read the persisted row so the printed summary reflects what was
        # actually auto-matched/stored, not just what was typed on the phone.
        persisted_legs.append(get_ticket_leg(db_path, leg_id))
        if line_snapshot_id is None:
            unmatched_legs.append(leg['goalie_name'])

    print(f"[OK] Recorded ticket {ticket_id}: {book} {ticket_type}, stake={stake}, reason={reason_code!r}")
    for leg in persisted_legs:
        odds_str = f"{leg['odds_at_bet']:+d}" if leg['odds_at_bet'] is not None else "N/A"
        line_str = leg['line_at_bet'] if leg['line_at_bet'] is not None else "N/A"
        print(f"  Leg: {leg['goalie_name']} ({leg['team_abbrev']}) @ {leg['book']} -- "
              f"{leg['side']} {line_str} ({odds_str})")

    if unmatched_legs:
        print(f"  [WARNING] No line_snapshots match found for: {', '.join(unmatched_legs)} -- "
              f"line/odds used as given (or blank if not provided), CLV columns will need a "
              f"matching snapshot to backfill later")

    return ticket_id, leg_ids


def main():
    parser = argparse.ArgumentParser(
        description='Record a ticket (1-3 legs) you placed, with stake/payout/reason'
    )
    parser.add_argument('--date', type=str, default=None, help='Game date (YYYY-MM-DD). Default: today')
    parser.add_argument('--book', type=str, required=True, help='Sportsbook/venue the ticket was placed at')
    parser.add_argument(
        '--ticket_type', type=str, required=True,
        help="Free text: straight, parlay_2, parlay_3, flex_2, flex_3, etc."
    )
    parser.add_argument('--stake', type=float, required=True, help='Units/dollars wagered')
    parser.add_argument(
        '--payout_multiplier', type=float, default=None,
        help='Total payout multiple if won (e.g. 3 for a 3x parlay). One of '
             '--payout_multiplier/--potential_payout is required.'
    )
    parser.add_argument(
        '--potential_payout', type=float, default=None,
        help='Total payout amount if won (stake included). Alternative to --payout_multiplier.'
    )
    parser.add_argument(
        '--reason_code', type=str, required=True,
        help='Required. e.g. "market-anchor model edge", "stale starter news", '
             '"line-shop gap", "closing-line move still pending", "manual hockey-context override"'
    )
    parser.add_argument(
        '--legs', type=str, required=True,
        help="Legs separated by ';'. Each leg: GoalieLastName:TEAM:SIDE:LINE[:ODDS[:BOOK]]. "
             "SIDE=OVER/UNDER. ODDS/BOOK optional -- auto-matched from the most recent "
             "line_snapshots row if omitted. Max 3 legs. "
             "Example: Shesterkin:NYR:OVER:24.5;Hellebuyck:WPG:UNDER:26.5:-110"
    )
    parser.add_argument('--model_version', type=str, default=None, help='Optional model version attribution')
    parser.add_argument('--notes', type=str, default=None, help='Optional notes')
    parser.add_argument('--db', type=str, default=DEFAULT_DB_PATH, help='Path to betting database. Default: data/betting.db')

    args = parser.parse_args()

    try:
        record_ticket(
            book=args.book,
            ticket_type=args.ticket_type,
            stake=args.stake,
            reason_code=args.reason_code,
            legs_str=args.legs,
            date=args.date,
            payout_multiplier=args.payout_multiplier,
            potential_payout=args.potential_payout,
            model_version=args.model_version,
            notes=args.notes,
            db_path=args.db,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
