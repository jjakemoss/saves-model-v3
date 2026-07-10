"""
Build timing-safe game-market features (moneyline and total-line prices,
de-vigged implied probabilities) from the 22:30Z bulk h2h/totals snapshots.

Ground truth for the source archive format and this artifact's contract
lives in docs/OFFSEASON_OPTIMIZATION_PLAN.md section 3.15 ("What each
purchase feeds (derived artifacts, pre-registered uses)") -- read that
before changing this script.

Scans data/raw/betting_lines/cache/bulk_date=*_markets=h2h,totals_regions=us.json
(one API response envelope per requested 22:30Z snapshot; the bulk endpoint
returns every game commencing in roughly the next 1-3 days, so the same
game shows up in several daily snapshot files before it is played).
odds_*.json and events_*.json files are ignored -- those feed a different
pre-registered artifact (saves_lines_snapshots.parquet,
scripts/build_odds_snapshots.py). Goalie/saves markets are out of scope
here entirely.

Tidy long output: one row per (event, requested snapshot, book, market
outcome). h2h and totals are de-vigged independently within the same
book + market (+ point, for totals) two-way group using proportional
(multiplicative) normalization -- each side's raw implied probability
(1 / decimal price) divided by the pair's raw-probability sum. Never
averaged across books or across snapshots (the odds-averaging bug,
docs/HISTORICAL_DATA_ANALYSIS.md section 1).

Commence-time drift: a game's reported commence_time can shift by a few
minutes to a few hours between snapshot files taken days apart (schedule
firming up, or -- rarely -- a real time change). Using a stale early
snapshot's commence_time to decide "is this snapshot pregame" per row
would let some genuinely-pregame snapshots get miscounted. Instead, each
event's canonical commence_time (and home/away team strings) is taken from
the appearance with the highest requested_ts -- the freshest information
this archive holds about that game's schedule -- and applied to every row
of that event. is_latest_pregame_snapshot then flags, per event, the
single requested_ts that is the last 22:30Z snapshot at or before that
canonical commence_time: the timing-safe "bet-time" view.

Usage:
    python scripts/build_market_game_features.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from betting.odds_archive import DEFAULT_CACHE_DIR, scan_archive  # noqa: E402
from build_multibook_training_data import TEAM_NAME_TO_ABBREV  # noqa: E402
from fetch_historical_odds_snapshots import commence_to_eastern_date  # noqa: E402

OUTPUT_PATH = Path("data/processed/market_game_features.parquet")

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Eastern-date-based season windows (inclusive), matching the convention in
# scripts/build_odds_snapshots.py.
SEASON_BOUNDS = [
    ("2023-24", "2023-08-01", "2024-07-31"),
    ("2024-25", "2024-08-01", "2025-07-31"),
    ("2025-26", "2025-08-01", "2026-07-31"),
]

OUTPUT_COLUMNS = [
    "event_id", "commence_time", "game_date_eastern", "home_team", "away_team",
    "home_abbrev", "away_abbrev", "requested_ts", "resolved_ts", "book",
    "market", "outcome_label", "point", "price_decimal", "implied_prob_devig",
    "is_latest_pregame_snapshot",
]


def season_from_eastern_date(date_str: str) -> str:
    """Map a US/Eastern game date (YYYY-MM-DD) to a season label using the
    fixed boundaries in SEASON_BOUNDS."""
    for label, start, end in SEASON_BOUNDS:
        if start <= date_str <= end:
            return label
    return "other"


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, TS_FORMAT).replace(tzinfo=timezone.utc)


def devig_pair(outcomes: list) -> list:
    """Proportional (multiplicative) de-vig for a same book+market(+point)
    two-way outcome group: raw implied probability (1 / decimal price) for
    each side, normalized so the pair sums to 1.0. Returns a list of
    (outcome, devig_prob_or_None) parallel to `outcomes`; devig_prob is
    None when the group isn't a clean priced two-way market (missing
    price, non-positive price, or fewer/more than 2 outcomes) -- there is
    no principled way to de-vig a one-sided quote."""
    prices = [o.get("price") for o in outcomes]
    if len(outcomes) != 2 or any(p is None or p <= 0 for p in prices):
        return [(o, None) for o in outcomes]
    raw_probs = [1.0 / p for p in prices]
    total = sum(raw_probs)
    if total <= 0:
        return [(o, None) for o in outcomes]
    return [(o, rp / total) for o, rp in zip(outcomes, raw_probs)]


def build_rows(cache_dir: Path = DEFAULT_CACHE_DIR):
    """Scan every h2h,totals bulk file in the cache and flatten it into one
    row per (event, requested snapshot, book, market outcome). Returns
    (DataFrame, stats dict)."""
    manifest = scan_archive(cache_dir)
    bulk_manifest = manifest[
        (manifest["kind"] == "bulk")
        & (manifest["markets"] == "h2h,totals")
        & (manifest["regions"] == "us")
    ].sort_values("requested_ts").reset_index(drop=True)

    print(f"  Cache manifest: {len(manifest)} parsed files, {len(bulk_manifest)} are h2h,totals/us bulk files")
    unparsable = manifest.attrs.get("unparsable", [])
    if unparsable:
        print(f"  [WARNING] {len(unparsable)} filenames in cache_dir did not match any known shape")

    # Pass 1: canonical per-event schedule facts (commence_time, home_team,
    # away_team), taken from the appearance with the highest requested_ts.
    # bulk_manifest is sorted ascending by requested_ts, so a plain
    # dict-overwrite loop leaves each event_id pointing at its freshest
    # appearance once the loop finishes.
    canonical = {}
    appearances = []  # (event_id, requested_ts, resolved_ts, bookmakers)
    n_bad_envelope = 0
    n_zero_bookmakers = 0
    n_games_seen = 0

    for row in bulk_manifest.itertuples(index=False):
        path = Path(row.path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                envelope = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            n_bad_envelope += 1
            print(f"  [WARNING] failed to parse {path.name}: {exc}")
            continue

        resolved_ts = envelope.get("timestamp") or row.resolved_ts
        games = envelope.get("data") or []
        for g in games:
            n_games_seen += 1
            event_id = g.get("id")
            commence_time = g.get("commence_time")
            home_team = g.get("home_team")
            away_team = g.get("away_team")
            if not event_id or not commence_time or not home_team or not away_team:
                n_bad_envelope += 1
                continue

            canonical[event_id] = {
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
            }

            bookmakers = g.get("bookmakers") or []
            if not bookmakers:
                n_zero_bookmakers += 1
            appearances.append({
                "event_id": event_id,
                "requested_ts": row.requested_ts,
                "resolved_ts": resolved_ts,
                "bookmakers": bookmakers,
            })

    print(f"  {n_games_seen} game-appearances across {len(bulk_manifest)} files, {len(canonical)} distinct events")

    # Pass 2: expand into outcome-level rows using each event's canonical
    # schedule facts (game-level) and each appearance's own bookmaker
    # quotes (row-level, genuinely time-varying).
    rows = []
    unmapped_teams = set()

    for appearance in appearances:
        event_id = appearance["event_id"]
        game = canonical[event_id]
        commence_time = game["commence_time"]
        home_team = game["home_team"]
        away_team = game["away_team"]
        home_abbrev = TEAM_NAME_TO_ABBREV.get(home_team)
        away_abbrev = TEAM_NAME_TO_ABBREV.get(away_team)
        if home_abbrev is None:
            unmapped_teams.add(home_team)
        if away_abbrev is None:
            unmapped_teams.add(away_team)
        game_date_eastern = commence_to_eastern_date(commence_time)

        for bookmaker in appearance["bookmakers"]:
            book = bookmaker.get("key")
            for market in bookmaker.get("markets") or []:
                mkey = market.get("key")
                if mkey not in ("h2h", "totals"):
                    continue
                outcomes = market.get("outcomes") or []

                if mkey == "h2h":
                    groups = [outcomes]
                else:
                    by_point = {}
                    for o in outcomes:
                        by_point.setdefault(o.get("point"), []).append(o)
                    groups = list(by_point.values())

                for group in groups:
                    devigged = devig_pair(group)
                    for outcome, prob in devigged:
                        if mkey == "h2h":
                            team_name = outcome.get("name")
                            label = TEAM_NAME_TO_ABBREV.get(team_name)
                            if label is None:
                                unmapped_teams.add(team_name)
                                label = team_name
                            point = None
                        else:
                            label = outcome.get("name")
                            point = outcome.get("point")

                        rows.append({
                            "event_id": event_id,
                            "commence_time": commence_time,
                            "game_date_eastern": game_date_eastern,
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_abbrev": home_abbrev,
                            "away_abbrev": away_abbrev,
                            "requested_ts": appearance["requested_ts"],
                            "resolved_ts": appearance["resolved_ts"],
                            "book": book,
                            "market": mkey,
                            "outcome_label": label,
                            "point": point,
                            "price_decimal": outcome.get("price"),
                            "implied_prob_devig": prob,
                        })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS[:-1])

    # is_latest_pregame_snapshot: per event_id, the single requested_ts
    # that is the last 22:30Z snapshot at or before that event's canonical
    # commence_time. ISO 'Z' timestamps are fixed-width and zero-padded, so
    # plain string comparison sorts them correctly.
    n_events_no_pregame_snapshot = 0
    if len(df) > 0:
        eligible = df[df["requested_ts"] <= df["commence_time"]]
        latest_ts = eligible.groupby("event_id")["requested_ts"].max().rename("_latest_requested_ts")
        df = df.merge(latest_ts, on="event_id", how="left")
        df["is_latest_pregame_snapshot"] = df["requested_ts"] == df["_latest_requested_ts"]
        n_events_no_pregame_snapshot = int(df["_latest_requested_ts"].isna().groupby(df["event_id"]).first().sum())
        df = df.drop(columns=["_latest_requested_ts"])
    else:
        df["is_latest_pregame_snapshot"] = pd.Series(dtype=bool)

    df = df[OUTPUT_COLUMNS]

    stats = {
        "n_bulk_files": len(bulk_manifest),
        "n_bad_envelope": n_bad_envelope,
        "n_zero_bookmakers": n_zero_bookmakers,
        "n_games_seen": n_games_seen,
        "n_events": len(canonical),
        "n_rows": len(df),
        "n_events_no_pregame_snapshot": n_events_no_pregame_snapshot,
        "unmapped_teams": unmapped_teams,
    }
    return df, stats


def run_validations(df: pd.DataFrame, stats: dict) -> bool:
    """Print every validation the task requires. Returns True iff nothing
    that should hard-fail the build was found."""
    ok = True
    df = df.copy()
    df["season"] = df["game_date_eastern"].map(season_from_eastern_date)

    print("\n[1] File, game, and row counts")
    print(f"  Bulk files scanned: {stats['n_bulk_files']}")
    print(f"  Files that failed to parse: {stats['n_bad_envelope']}")
    print(f"  Game-appearances (event x snapshot file): {stats['n_games_seen']}")
    print(f"  Distinct games (event_id): {stats['n_events']}")
    print(f"  Rows produced: {stats['n_rows']}")
    print("  Rows per season:")
    print(df.groupby("season").size().to_string())

    print("\n[2] Distinct games per season with >=1 totals quote and >=1 h2h quote at the latest pregame snapshot")
    if stats["n_events_no_pregame_snapshot"]:
        print(f"  [NOTE] {stats['n_events_no_pregame_snapshot']} event(s) have no snapshot at or before their "
              f"(latest-known) commence_time -- every bulk file we hold for them was requested after "
              f"commence, so they carry no is_latest_pregame_snapshot=True row and are excluded below.")
    latest = df[df["is_latest_pregame_snapshot"]]
    has_totals = latest.loc[latest["market"] == "totals"].groupby("season")["event_id"].apply(set)
    has_h2h = latest.loc[latest["market"] == "h2h"].groupby("season")["event_id"].apply(set)
    for season in sorted(set(has_totals.index) | set(has_h2h.index)):
        both = has_totals.get(season, set()) & has_h2h.get(season, set())
        print(f"  {season}: {len(both)} games with both markets at latest pregame snapshot")

    print("\n[3] De-vig sanity: implied_prob_devig sums to 1.0 within a (event, requested_ts, book, market, point) group")
    devig_rows = df.dropna(subset=["implied_prob_devig"])
    group_cols = ["event_id", "requested_ts", "book", "market", "point"]
    sums = devig_rows.groupby(group_cols, dropna=False)["implied_prob_devig"].sum()
    deviation = (sums - 1.0).abs()
    max_dev = float(deviation.max()) if len(deviation) else float("nan")
    n_null_devig = int(df["implied_prob_devig"].isna().sum())
    print(f"  Groups checked: {len(sums)}; max |sum - 1.0| = {max_dev:.3e}")
    print(f"  Rows with no de-vig probability (unpaired/bad-price outcome, excluded above): {n_null_devig}")
    if len(sums) and max_dev > 1e-9:
        ok = False
        print(f"  [FAIL] max deviation {max_dev:.3e} exceeds 1e-9")
    else:
        print("  [OK]")

    print("\n[4] Spot-check: bulk_date=2024-01-18T22_30_00Z, Colorado Avalanche @ Boston Bruins, betrivers")
    spot = df[
        (df["home_abbrev"] == "BOS") & (df["away_abbrev"] == "COL")
        & (df["requested_ts"] == "2024-01-18T22:30:00Z")
        & (df["book"] == "betrivers")
    ]
    if spot.empty:
        ok = False
        print("  [FAIL] no rows found for this event/snapshot/book")
    else:
        expected_h2h = {"BOS": 1.67, "COL": 2.23}
        expected_totals = {("Over", 6.0): 1.80, ("Under", 6.0): 2.02}

        h2h_rows = spot[spot["market"] == "h2h"]
        for abbrev, exp_price in expected_h2h.items():
            match = h2h_rows[h2h_rows["outcome_label"] == abbrev]
            if match.empty:
                ok = False
                print(f"  [FAIL] no h2h row for {abbrev}")
                continue
            actual_price = float(match.iloc[0]["price_decimal"])
            actual_prob = float(match.iloc[0]["implied_prob_devig"])
            passed = abs(actual_price - exp_price) < 1e-9
            ok = ok and passed
            print(f"  [{'OK' if passed else 'FAIL'}] h2h {abbrev}: price={actual_price} (expected {exp_price}), "
                  f"devig_prob={actual_prob:.6f}")

        totals_rows = spot[spot["market"] == "totals"]
        for (label, point), exp_price in expected_totals.items():
            match = totals_rows[(totals_rows["outcome_label"] == label) & (totals_rows["point"] == point)]
            if match.empty:
                ok = False
                print(f"  [FAIL] no totals row for {label} {point}")
                continue
            actual_price = float(match.iloc[0]["price_decimal"])
            actual_prob = float(match.iloc[0]["implied_prob_devig"])
            passed = abs(actual_price - exp_price) < 1e-9
            ok = ok and passed
            print(f"  [{'OK' if passed else 'FAIL'}] totals {label} {point}: price={actual_price} "
                  f"(expected {exp_price}), devig_prob={actual_prob:.6f}")

    print("\n[5] No snapshot leakage: requested_ts must be <= commence_time on every is_latest_pregame_snapshot row")
    latest = df[df["is_latest_pregame_snapshot"]]
    if len(latest):
        leaked = latest.apply(lambda r: _parse_ts(r["requested_ts"]) > _parse_ts(r["commence_time"]), axis=1)
        n_leaked = int(leaked.sum())
    else:
        n_leaked = 0
    print(f"  is_latest_pregame_snapshot rows checked: {len(latest)}; leaked rows: {n_leaked}")
    if n_leaked:
        ok = False
        print("  [FAIL] snapshot leakage detected")
    else:
        print("  [OK]")

    print("\n[6] Team abbreviation mapping")
    unmapped = stats["unmapped_teams"]
    if unmapped:
        ok = False
        print(f"  [FAIL] {len(unmapped)} team name(s) failed to map: {sorted(unmapped)}")
    else:
        print("  [OK] every team name mapped via TEAM_NAME_TO_ABBREV")

    return ok


def main():
    print("[1/3] Scanning h2h,totals bulk snapshots and flattening...")
    df, stats = build_rows()

    print("\n[2/3] Running validations...")
    ok = run_validations(df, stats)

    print(f"\n[3/3] Saving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved {len(df)} rows, {len(df.columns)} columns")

    if not ok:
        print("\n[FAIL] one or more validations failed -- see output above")
        sys.exit(1)
    print("\n[OK] all validations passed")


if __name__ == "__main__":
    main()
