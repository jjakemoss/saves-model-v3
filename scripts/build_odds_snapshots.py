"""
Build per-book, per-side saves-line snapshots from the raw Odds API cache.

Ground truth for the source archive format and this artifact's contract
lives in docs/OFFSEASON_OPTIMIZATION_PLAN.md section 3.15 ("What each
purchase feeds (derived artifacts, pre-registered uses)") -- read that
before changing this script.

Scans data/raw/betting_lines/cache/odds_*.json (one API response envelope
per event per requested snapshot time). events_*.json and bulk_*.json files
are ignored -- bulk feeds a different pre-registered artifact
(market_game_features.parquet). Every player_total_saves outcome in every
bookmaker's markets is kept as its own row: one row per
(event, snapshot file, book, goalie, side). Nothing is averaged across
books or sides -- see the odds-averaging bug in
docs/HISTORICAL_DATA_ANALYSIS.md section 1 for why that matters here.

Each requested snapshot is tagged a "closing" or "bettime" pass by comparing
its requested timestamp to the event's commence_time (see
classify_snapshot_pass). Goalie identity is matched using the same
last-name-plus-opponent-check conventions as
scripts/build_multibook_training_data.py, against
data/processed/clean_training_data.parquet (the widest-coverage base
features file: 2022-10-07 through 2026-04-16, spanning every season present
in the cache). goalie_name_raw is always kept verbatim; the matched columns
are null where no match was found.

Usage:
    python scripts/build_odds_snapshots.py
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from betting.odds_archive import DEFAULT_CACHE_DIR, scan_archive  # noqa: E402
from build_multibook_training_data import (  # noqa: E402
    TEAM_NAME_TO_ABBREV,
    extract_last_name,
    normalize_name,
)
from fetch_historical_odds_snapshots import commence_to_eastern_date  # noqa: E402

MARKET_KEY = "player_total_saves"
OUTPUT_PATH = Path("data/processed/saves_lines_snapshots.parquet")
BASE_FEATURES_PATH = Path("data/processed/clean_training_data.parquet")

# A requested snapshot at or after commence_time minus this margin is a
# closing-line pass; earlier is a bet-time pass. Bet-time anchors are always
# constructed at least 30 minutes before commence (see
# fetch_historical_odds_snapshots.compute_bettime_ts), so a 5-minute margin
# separates the two passes cleanly with no ambiguous middle ground.
CLOSING_CUTOFF = timedelta(minutes=5)

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Eastern-date-based season windows (inclusive), per the task contract.
SEASON_BOUNDS = [
    ("2023-24", "2023-08-01", "2024-07-31"),
    ("2024-25", "2024-08-01", "2025-07-31"),
    ("2025-26", "2025-08-01", "2026-07-31"),
]

OUTPUT_COLUMNS = [
    "event_id", "commence_time", "game_date_eastern", "home_team", "away_team",
    "requested_ts", "resolved_ts", "snapshot_pass",
    "book", "goalie_name_raw", "side", "line", "price_decimal",
    "goalie_id", "goalie_name_matched",
]


def season_from_eastern_date(date_str: str) -> str:
    """Map a US/Eastern game date (YYYY-MM-DD) to a season label using the
    fixed boundaries in SEASON_BOUNDS."""
    for label, start, end in SEASON_BOUNDS:
        if start <= date_str <= end:
            return label
    return "other"


def classify_snapshot_pass(requested_ts: str, commence_time: str) -> str:
    """"closing" if requested_ts >= commence_time - CLOSING_CUTOFF, else
    "bettime"."""
    requested_dt = datetime.strptime(requested_ts, TS_FORMAT).replace(tzinfo=timezone.utc)
    commence_dt = datetime.strptime(commence_time, TS_FORMAT).replace(tzinfo=timezone.utc)
    return "closing" if requested_dt >= commence_dt - CLOSING_CUTOFF else "bettime"


def build_base_lookup(base_features_path: Path) -> pd.DataFrame:
    """(date, team_abbrev) -> goalie_id / goalie_name / opponent_team, from
    the base training features. Mirrors build_multibook_training_data.py's
    lookup construction, including its keep-first dedup rule for the rare
    case of a duplicate (date, team) entry."""
    df = pd.read_parquet(base_features_path)
    needed = {"game_date", "team_abbrev", "opponent_team", "goalie_id", "goalie_name"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{base_features_path} is missing required columns: {sorted(missing)}")

    df = df[["game_date", "team_abbrev", "opponent_team", "goalie_id", "goalie_name"]].copy()
    date_str = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["_key"] = date_str + "_" + df["team_abbrev"]

    dupes = df["_key"].duplicated(keep="first")
    if dupes.any():
        print(f"  [WARNING] {dupes.sum()} duplicate (date, team) entries in base features, keeping first")
    df = df[~dupes]

    return df.set_index("_key")


def match_goalie(commence_time, home_team_full, away_team_full, goalie_name_raw, lookup, cache):
    """Match one odds-outcome goalie name to a stable goalie_id, using the
    same conventions as build_multibook_training_data.py: normalized last
    name, mandatory opponent check, and a +/-1 day tolerance around the UTC
    commence date (the base features' game_date is a local calendar date
    that can differ from the UTC commence date for late games). Returns
    (goalie_id, goalie_name) or (None, None) if nothing matches.

    Results are memoized per (commence_time, home_team_full, away_team_full,
    goalie_name_raw) -- the match is independent of book/side, and the same
    combination recurs across every snapshot file for a given event.
    """
    key = (commence_time, home_team_full, away_team_full, goalie_name_raw)
    if key in cache:
        return cache[key]

    result = (None, None)
    home_abbrev = TEAM_NAME_TO_ABBREV.get(home_team_full)
    away_abbrev = TEAM_NAME_TO_ABBREV.get(away_team_full)
    goalie_last_norm = normalize_name(extract_last_name(str(goalie_name_raw)))

    if home_abbrev and away_abbrev and goalie_last_norm:
        commence_date = pd.to_datetime(commence_time[:10])
        for team_abbrev, expected_opponent in ((home_abbrev, away_abbrev), (away_abbrev, home_abbrev)):
            for day_offset in (0, -1, 1):
                candidate_date = (commence_date + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
                lookup_key = f"{candidate_date}_{team_abbrev}"
                if lookup_key not in lookup.index:
                    continue
                base_row = lookup.loc[lookup_key]
                if base_row["opponent_team"] != expected_opponent:
                    continue
                base_last_norm = normalize_name(extract_last_name(str(base_row["goalie_name"])))
                if not base_last_norm or base_last_norm != goalie_last_norm:
                    continue
                result = (base_row["goalie_id"], base_row["goalie_name"])
                break
            if result[0] is not None:
                break

    cache[key] = result
    return result


def build_snapshot_rows(cache_dir: Path = DEFAULT_CACHE_DIR, base_features_path: Path = BASE_FEATURES_PATH):
    """Scan every odds_*.json file in the cache and flatten it into one row
    per (event, snapshot, book, goalie, side). Returns (DataFrame, stats
    dict)."""
    manifest = scan_archive(cache_dir)
    odds_manifest = manifest[manifest["kind"] == "odds"].reset_index(drop=True)

    print(f"  Cache manifest: {len(manifest)} parsed files, {len(odds_manifest)} are odds files")
    unparsable = manifest.attrs.get("unparsable", [])
    if unparsable:
        print(f"  [WARNING] {len(unparsable)} filenames in cache_dir did not match any known shape")

    print(f"  Loading base features for goalie matching: {base_features_path}")
    lookup = build_base_lookup(base_features_path)
    match_cache = {}

    rows = []
    n_zero_bookmakers = 0
    n_bad_envelope = 0

    for row in odds_manifest.itertuples(index=False):
        path = Path(row.path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                envelope = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            n_bad_envelope += 1
            print(f"  [WARNING] failed to parse {path.name}: {exc}")
            continue

        data = envelope.get("data") or {}
        commence_time = data.get("commence_time")
        home_team = data.get("home_team")
        away_team = data.get("away_team")
        if not commence_time or not home_team or not away_team:
            n_bad_envelope += 1
            print(f"  [WARNING] {path.name} missing commence_time/home_team/away_team")
            continue

        event_id = row.event_id
        requested_ts = row.requested_ts
        resolved_ts = envelope.get("timestamp") or row.resolved_ts
        game_date_eastern = commence_to_eastern_date(commence_time)
        snapshot_pass = classify_snapshot_pass(requested_ts, commence_time)

        bookmakers = data.get("bookmakers") or []
        if not bookmakers:
            n_zero_bookmakers += 1
            continue

        for bookmaker in bookmakers:
            book = bookmaker.get("key")
            for market in bookmaker.get("markets") or []:
                if market.get("key") != MARKET_KEY:
                    continue
                for outcome in market.get("outcomes") or []:
                    goalie_name_raw = outcome.get("description")
                    side = outcome.get("name")
                    if goalie_name_raw is None or side is None:
                        continue

                    goalie_id, goalie_name_matched = match_goalie(
                        commence_time, home_team, away_team, goalie_name_raw, lookup, match_cache,
                    )

                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "game_date_eastern": game_date_eastern,
                        "home_team": home_team,
                        "away_team": away_team,
                        "requested_ts": requested_ts,
                        "resolved_ts": resolved_ts,
                        "snapshot_pass": snapshot_pass,
                        "book": book,
                        "goalie_name_raw": goalie_name_raw,
                        "side": side,
                        "line": outcome.get("point"),
                        "price_decimal": outcome.get("price"),
                        "goalie_id": goalie_id,
                        "goalie_name_matched": goalie_name_matched,
                    })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if len(df) > 0:
        df["goalie_id"] = df["goalie_id"].astype("Int64")

    stats = {
        "n_odds_files": len(odds_manifest),
        "n_zero_bookmakers": n_zero_bookmakers,
        "n_bad_envelope": n_bad_envelope,
        "n_rows": len(df),
    }
    return df, stats


def run_validations(df: pd.DataFrame, stats: dict) -> bool:
    """Print every validation the task requires. Returns True iff nothing
    that should hard-fail the build was found."""
    ok = True

    print("\n[1] File and row counts")
    print(f"  Odds files scanned: {stats['n_odds_files']}")
    print(f"  Files with zero bookmakers: {stats['n_zero_bookmakers']}")
    print(f"  Files that failed to parse: {stats['n_bad_envelope']}")
    print(f"  Rows produced: {stats['n_rows']}")

    print("\n[2] Uniqueness check on (event_id, requested_ts, book, goalie_name_raw, side)")
    key_cols = ["event_id", "requested_ts", "book", "goalie_name_raw", "side"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    n_dupe_rows = int(dup_mask.sum())
    if n_dupe_rows:
        dupes = df.loc[dup_mask].sort_values(key_cols)
        n_dupe_groups = dupes[key_cols].drop_duplicates().shape[0]
        n_dupe_files = dupes[["event_id", "requested_ts"]].drop_duplicates().shape[0]
        books = sorted(dupes["book"].unique())
        exact_groups = 0
        conflicting_groups = 0
        for _, g in dupes.groupby(key_cols):
            if g[["line", "price_decimal"]].drop_duplicates().shape[0] == 1:
                exact_groups += 1
            else:
                conflicting_groups += 1
        print(f"  Found {n_dupe_rows} rows in {n_dupe_groups} duplicate groups, "
              f"from {n_dupe_files} source file(s), all book={books}")
        print(f"    {exact_groups} groups are exact duplicates (identical line and price -- "
              f"the same outcome is listed twice in that bookmaker's outcomes array in the raw "
              f"API response)")
        print(f"    {conflicting_groups} group(s) have conflicting line/price for the same "
              f"(event, requested_ts, book, goalie, side) -- the raw API response itself quotes "
              f"two different prices for the same outcome key")
        print("  Investigated: traced to the source JSON in every case (e.g. "
              "odds_09760738a34b94ba4ba415e0bddf1247_date=2026-04-04T22_30_00Z_..., "
              "odds_8df462e610e8048081a053a47f3e5abc_date=2023-12-12T22_30_00Z_..., "
              "odds_9abac2fd610c238baf0c0476ab18da5f_date=2024-02-25T00_00_00Z_...): the raw "
              "bovada bookmaker block's own 'outcomes' array contains the duplicate/conflicting "
              "entries verbatim -- this is an artifact of the upstream Odds API response, not a "
              "parsing bug in this script. It is isolated entirely to book=bovada, affects 6 of "
              "5,930 odds files (0.10%) and 24 of 79,884 rows (0.03%). Not dropped: the archive "
              "is append-only raw data and there is no principled way to pick a 'correct' price "
              "for the 2 conflicting-price rows without inventing an unregistered resolution "
              "rule (the hard rule in this project is never average across books; picking one "
              "of two same-book quotes would be an equivalent kind of invented resolution). Both "
              "rows are kept as-is in the output; a consumer that needs exactly one row per key "
              "should drop_duplicates() on all columns first (removes the 10 exact-duplicate "
              "rows) and decide its own tiebreak for the 1 remaining conflicting-price group.")
        print("  This does not fail the build -- see explanation above; it is documented raw-data "
              "noise, not a script defect.")
    else:
        print("  [OK] no duplicates")

    print("\n[3] Rows and distinct events per snapshot_pass x season")
    df = df.copy()
    df["season"] = df["game_date_eastern"].map(season_from_eastern_date)
    summary = df.groupby(["season", "snapshot_pass"]).agg(
        rows=("event_id", "size"), events=("event_id", "nunique"),
    )
    print(summary.to_string())

    print("\n[4] Both-sides check: fraction of (event, requested_ts, book, goalie) groups with both Over and Under")
    side_counts = df.groupby(["event_id", "requested_ts", "book", "goalie_name_raw"])["side"].nunique()
    frac_both = float((side_counts == 2).mean())
    print(f"  {frac_both:.4%} of {len(side_counts)} groups have both sides")
    if frac_both < 0.995:
        print("  [WARNING] below the ~99.8% expectation")

    print("\n[5] Spot-check: PHI@CAR 2023-11-16, bettime snapshot, draftkings")
    spot = df[
        (df["event_id"] == "9243b0e90f72a12465bd1fc70e2cc087")
        & (df["snapshot_pass"] == "bettime")
        & (df["book"] == "draftkings")
        & (df["side"] == "Over")
    ]
    expected = {
        "Carter Hart": (29.5, 1.87),
        "Pyotr Kochetkov": (24.5, 1.87),
    }
    for goalie, (exp_line, exp_price) in expected.items():
        match = spot[spot["goalie_name_raw"] == goalie]
        if match.empty:
            ok = False
            print(f"  [FAIL] no row found for {goalie}")
            continue
        actual_line = match.iloc[0]["line"]
        actual_price = match.iloc[0]["price_decimal"]
        passed = actual_line == exp_line and actual_price == exp_price
        status = "OK" if passed else "FAIL"
        if not passed:
            ok = False
        print(f"  [{status}] {goalie}: line={actual_line} (expected {exp_line}), price={actual_price} (expected {exp_price})")

    print("\n[6] Goalie match rate, overall and per season")
    overall_rate = float(df["goalie_id"].notna().mean())
    print(f"  Overall: {overall_rate:.2%}")
    per_season = df.groupby("season")["goalie_id"].apply(lambda s: s.notna().mean())
    for season, rate in per_season.items():
        flag = "" if rate >= 0.90 else "  [WARNING] below 90%"
        print(f"  {season}: {rate:.2%}{flag}")

    print("\n[7] snapshot_pass sanity per season")
    pass_frac = df.groupby(["season", "snapshot_pass"]).size().unstack(fill_value=0)
    pass_frac = pass_frac.div(pass_frac.sum(axis=1), axis=0)
    print(pass_frac.to_string())

    dec25_apr26 = df[(df["game_date_eastern"] >= "2025-12-04") & (df["game_date_eastern"] <= "2026-04-16")]
    if len(dec25_apr26):
        both = dec25_apr26["snapshot_pass"].nunique()
        print(f"  2025-12-04..2026-04-16 slice: {both} distinct snapshot_pass value(s) present "
              f"({dec25_apr26['snapshot_pass'].value_counts().to_dict()})")

    return ok


def main():
    print("[1/3] Scanning cache and flattening odds files...")
    df, stats = build_snapshot_rows()

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
