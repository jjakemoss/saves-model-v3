"""
Build row-level snapshots from the purchased core_bettime_202607 pass.

Binding contract: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 14.5
("Binding ingestion rules for core_bettime_202607") -- fixed BEFORE any
price-level record in data/raw/betting_lines/passes/core_bettime_202607/
was opened, and binding here by reference (also binds Experiments 12 and
13, section 14.5's own preamble). Do not add, relax, or reorder an
exclusion rule without re-reading that section first.

This is a pure parser with validation counts. Per section 15.8 (which this
script's output must also satisfy, since Experiments 12/13 read this
artifact), it performs NO analysis, NO grading, and NO price-level
distributional summaries of the parsed data -- only coverage counts, match
rates, and the exclusion accounting section 14.5 requires. Anything beyond
that (means/medians of lines or prices, outcome joins, EV, ROI) belongs in
a later, separately preregistered experiment script, not here.

Source records are `core_event={event_id}_signature={signature}.json`
envelopes written by scripts/purchase_core_bettime_passes.py (schema
documented in that script's `execute()`): schema_version, signature,
request, event (season/event_id/commence_time/bettime_ts/home_team/
away_team), pass_name, fetched_at, status_code, quota_headers, raw_body
(verbatim The Odds API historical event-odds response text). This is a
different envelope shape than scripts/build_odds_snapshots.py's
odds_*.json cache files, so it cannot reuse that script's file-scanning
directly -- but per section 14.5 rule 1 it MUST reuse that script's row-
shape and matching conventions. Concretely, this script imports and calls
build_odds_snapshots.build_base_lookup / .match_goalie / .commence_to_
eastern_date / .season_from_eastern_date unchanged (same sys.path pattern
scripts/experiment_rolling_origin.py uses to import clv_audit_pace_policy)
rather than re-deriving goalie-matching logic here.

Markets kept: player_total_saves and player_shots_on_goal only (the two
markets this pass purchased). Any other market key is a schema surprise
and stops the run -- see MARKET_KEYS below.

Exclusion steps, applied in this exact order, each counted in the JSON
summary and the printed report. "Rule N" always means section 14.5's OWN
numbering (rule 1 = parsing/pairing conventions, rule 2 = duplicates,
rule 3 = conflicting prices, rule 4 = commence drift, rule 5 = fanatics,
rule 6 = pushes) -- which is NOT this script's application order:
  a. Non-200 records (expected: 2, both EVENT_NOT_FOUND 404s). Not a
     numbered 14.5 rule -- these records carry no odds payload to parse.
  b. Commence-drift rule (14.5 rule 4): exclude any event whose
     effective_gap_minutes = (true_commence_time - requested_ts), in
     minutes, is < 10. The exact count is NOT assumed in advance -- it is
     computed here and reported, whatever it turns out to be.
  c. Byte-identical duplicate outcomes (14.5 rule 2): within (event, book,
     market), drop extra copies identical across every outcome field.
  d. Conflicting-price groups (14.5 rule 3): within (event, book, market,
     player_name_raw, line, side), any group with more than one distinct
     price is excluded ENTIRELY (not tie-broken). Expected exactly 3, all
     fanduel / 2023-24 / player_shots_on_goal; any other count, book,
     market, or season is a fail-closed stop, not a guess.
  e. Fanatics (14.5 rule 5): any fanatics row found anywhere is a schema
     surprise -- stop and report rather than silently include or drop it.
     (Checked up front as a hard stop, before b-d, since its presence
     anywhere invalidates the whole parse.)

Applying drift (rule 4) before duplicates (rule 2) and conflicts (rule 3)
is order-independent for the final row set -- the operations touch
disjoint row populations here -- and the two-stage duplicate accounting
(post-drift vs all-200-records) handles the one bookkeeping interaction,
so the audit's duplicate counts reconcile on the audit's own scope.

Pushes are NOT excluded here (14.5 rule 6 is a grading-time rule, out of
scope for this ingestion script).

Usage:
    python scripts/build_core_bettime_pass_snapshots.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import build_odds_snapshots as bos  # noqa: E402 -- canonical parser, section 14.5 rule 1
from betting.odds_utils import decimal_to_american  # noqa: E402

PASS_DIR = Path("data/raw/betting_lines/passes/core_bettime_202607")
AUDIT_PATH = PASS_DIR / "audit_summary.json"
EXISTING_SNAPSHOTS_PATH = Path("data/processed/saves_lines_snapshots.parquet")  # read-only

OUTPUT_PATH = Path("data/processed/core_bettime_202607_snapshots.parquet")
SUMMARY_PATH = Path("data/processed/core_bettime_202607_snapshots_summary.json")

MARKET_KEYS = {"player_total_saves", "player_shots_on_goal"}
SNAPSHOT_PASS = "bettime"
DRIFT_FLOOR_MINUTES = 10.0
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

OUTPUT_COLUMNS = [
    "pass_name", "season", "event_id",
    "requested_ts", "fetched_at", "cached_commence_time", "true_commence_time",
    "effective_gap_minutes", "game_date_eastern", "resolved_ts", "snapshot_pass",
    "home_team", "away_team",
    "book_key", "market_key",
    "player_name_raw", "side", "line", "price", "price_decimal", "multiplier",
    "goalie_id", "goalie_name_matched",
]


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, TS_FORMAT).replace(tzinfo=timezone.utc)


def iter_pass_records(pass_dir: Path):
    """Yield (path, record dict) for every core_event=*.json record, sorted
    for determinism. run_log.jsonl and audit_summary.json are excluded by
    the glob pattern itself, not by a separate skip check."""
    for path in sorted(pass_dir.glob("core_event=*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            yield path, json.load(fh)


def parse_records(pass_dir: Path, base_features_path: Path = bos.BASE_FEATURES_PATH):
    """Parse every record into pre-exclusion rows plus the bookkeeping
    needed for every exclusion rule. Returns a dict with:
      rows: list of row dicts (one per outcome, saves+SOG only)
      non200: list of (event_id, pass_name, status_code) for skipped records
      event_gaps: list of (pass_name, event_id, season, effective_gap_minutes,
                  requested_ts, true_commence_time, home_team, away_team)
                  for EVERY 200-status event, whether or not it produced any
                  row (an event with zero bookmakers still needs a gap value
                  for the drift-exclusion accounting)
      fanatics_rows: list of (event_id, book_key) if any fanatics bookmaker
                     block was seen (should be empty)
      unexpected_market_keys: set of any market key outside MARKET_KEYS
      n_records_total: total core_event files scanned
    """
    print(f"  Loading base features for goalie matching: {base_features_path}")
    lookup = bos.build_base_lookup(base_features_path)
    match_cache = {}

    rows = []
    non200 = []
    event_gaps = []
    fanatics_rows = []
    unexpected_market_keys = set()
    n_records_total = 0

    for path, record in iter_pass_records(pass_dir):
        n_records_total += 1
        pass_name = record["pass_name"]
        event_meta = record["event"]
        event_id = event_meta["event_id"]
        status_code = record["status_code"]

        if status_code != 200:
            non200.append((event_id, pass_name, status_code))
            continue

        body = json.loads(record["raw_body"])
        data = body.get("data") or {}
        true_commence_time = data.get("commence_time")
        if not true_commence_time:
            raise SystemExit(
                f"[SCHEMA SURPRISE] 200 record with no data.commence_time: {path}"
            )

        requested_ts = event_meta["bettime_ts"]
        cached_commence_time = event_meta["commence_time"]
        fetched_at = record["fetched_at"]
        resolved_ts = body.get("timestamp")
        home_team = data.get("home_team") or event_meta.get("home_team")
        away_team = data.get("away_team") or event_meta.get("away_team")

        gap_minutes = (
            _parse_utc(true_commence_time) - _parse_utc(requested_ts)
        ).total_seconds() / 60.0
        game_date_eastern = bos.commence_to_eastern_date(true_commence_time)
        season = bos.season_from_eastern_date(game_date_eastern)

        event_gaps.append({
            "pass_name": pass_name,
            "event_id": event_id,
            "season": season,
            "effective_gap_minutes": gap_minutes,
            "requested_ts": requested_ts,
            "true_commence_time": true_commence_time,
            "home_team": home_team,
            "away_team": away_team,
        })

        for bookmaker in data.get("bookmakers") or []:
            book_key = bookmaker.get("key")
            if book_key == "fanatics":
                fanatics_rows.append((event_id, book_key))

            for market in bookmaker.get("markets") or []:
                market_key = market.get("key")
                if market_key not in MARKET_KEYS:
                    unexpected_market_keys.add(market_key)
                    continue

                is_saves = market_key == "player_total_saves"
                for outcome in market.get("outcomes") or []:
                    player_name_raw = outcome.get("description")
                    side = outcome.get("name")
                    line = outcome.get("point")
                    price_decimal = outcome.get("price")
                    multiplier = outcome.get("multiplier")

                    goalie_id, goalie_name_matched = (None, None)
                    if is_saves:
                        goalie_id, goalie_name_matched = bos.match_goalie(
                            true_commence_time, home_team, away_team,
                            player_name_raw, lookup, match_cache,
                        )

                    rows.append({
                        "pass_name": pass_name,
                        "season": season,
                        "event_id": event_id,
                        "requested_ts": requested_ts,
                        "fetched_at": fetched_at,
                        "cached_commence_time": cached_commence_time,
                        "true_commence_time": true_commence_time,
                        "effective_gap_minutes": gap_minutes,
                        "game_date_eastern": game_date_eastern,
                        "resolved_ts": resolved_ts,
                        "snapshot_pass": SNAPSHOT_PASS,
                        "home_team": home_team,
                        "away_team": away_team,
                        "book_key": book_key,
                        "market_key": market_key,
                        "player_name_raw": player_name_raw,
                        "side": side,
                        "line": line,
                        "price": decimal_to_american(price_decimal),
                        "price_decimal": price_decimal,
                        "multiplier": multiplier,
                        "goalie_id": goalie_id,
                        "goalie_name_matched": goalie_name_matched,
                    })

    return {
        "rows": rows,
        "non200": non200,
        "event_gaps": event_gaps,
        "fanatics_rows": fanatics_rows,
        "unexpected_market_keys": unexpected_market_keys,
        "n_records_total": n_records_total,
    }


def apply_exclusions(parsed: dict) -> dict:
    """Apply section 14.5's exclusion rules on top of the already-excluded
    non-200 records. ACTUAL application order (rule numbers are 14.5's own,
    not positional): fanatics (rule 5) as an up-front hard stop, then
    commence-drift (rule 4), then byte-identical duplicates (rule 2), then
    conflicting-price groups (rule 3). The operations are order-independent
    for the final row set; the two-stage duplicate accounting (post-drift
    vs all-200-records) handles the bookkeeping interaction between the
    drift exclusion and the audit's duplicate counts. Returns a dict with
    the final DataFrame plus every count needed for the summary/report."""
    df = pd.DataFrame(parsed["rows"], columns=OUTPUT_COLUMNS)
    if len(df) > 0:
        # Nullable integer dtype, matching build_odds_snapshots.py's own
        # goalie_id cast convention (section 14.5 rule 1).
        df["goalie_id"] = df["goalie_id"].astype("Int64")
    summary = {}

    # --- 14.5 rule 5 (fanatics) is a hard stop, checked first regardless
    # of ordering, since its presence anywhere invalidates the whole
    # parse. ---
    if parsed["fanatics_rows"]:
        raise SystemExit(
            f"[SCHEMA SURPRISE] fanatics rows found: {parsed['fanatics_rows'][:10]} "
            f"(total {len(parsed['fanatics_rows'])}) -- section 14.5 rule 5 requires a stop."
        )
    summary["fanatics_rows_found"] = 0

    if parsed["unexpected_market_keys"]:
        raise SystemExit(
            f"[SCHEMA SURPRISE] unexpected market key(s): {sorted(parsed['unexpected_market_keys'])} "
            "-- stopping per task spec (only player_total_saves / player_shots_on_goal expected)."
        )

    # --- 14.5 rule 4: commence-drift exclusion, at EVENT granularity
    # (covers events with zero rows, e.g. a 200 response with zero
    # bookmakers). ---
    gaps_df = pd.DataFrame(parsed["event_gaps"])
    drift_excluded = gaps_df[gaps_df["effective_gap_minutes"] < DRIFT_FLOOR_MINUTES].copy()
    drift_excluded_ids = set(drift_excluded["event_id"])
    summary["drift_excluded_events"] = drift_excluded.sort_values("effective_gap_minutes").to_dict("records")
    summary["n_drift_excluded_events"] = len(drift_excluded_ids)

    n_rows_pre_drift = len(df)
    df_post_drift = df[~df["event_id"].isin(drift_excluded_ids)].copy()
    summary["n_rows_dropped_by_drift_exclusion"] = n_rows_pre_drift - len(df_post_drift)

    # --- 14.5 rule 2: byte-identical duplicate outcomes within (event,
    # book, market). Computed on POST-drift rows for the kept output, but
    # also independently on ALL pre-drift rows so the audit's
    # exact_duplicate_extra_copies_per_season_book_market (which counted
    # duplicates on every 200-status record, before any drift exclusion)
    # can be reconciled honestly rather than compared apples-to-oranges. ---
    dup_key_cols = ["event_id", "book_key", "market_key", "player_name_raw", "side", "line", "price_decimal", "multiplier"]

    dup_mask_post_drift = df_post_drift.duplicated(subset=dup_key_cols, keep="first")
    dup_extra_post_drift = df_post_drift[dup_mask_post_drift]
    summary["n_duplicate_extra_copies_dropped"] = int(dup_mask_post_drift.sum())
    summary["duplicate_extra_copies_per_season_book_market"] = (
        dup_extra_post_drift.groupby(["season", "book_key", "market_key"]).size()
        .reset_index(name="n_extra_copies").to_dict("records")
    )

    dup_mask_all = df.duplicated(subset=dup_key_cols, keep="first")
    summary["n_duplicate_extra_copies_all_200_records_no_drift_exclusion"] = int(dup_mask_all.sum())
    summary["duplicate_extra_copies_per_season_book_market_all_200_records"] = (
        df[dup_mask_all].groupby(["season", "book_key", "market_key"]).size()
        .reset_index(name="n_extra_copies").to_dict("records")
    )

    df_dedup = df_post_drift[~dup_mask_post_drift].copy()

    # --- 14.5 rule 3: conflicting-price groups. Fail-closed if the finding
    # differs from the preregistered expectation (exactly 3, all fanduel /
    # 2023-24 / player_shots_on_goal). ---
    conflict_key_cols = ["event_id", "book_key", "market_key", "player_name_raw", "line", "side"]
    price_nunique = df_dedup.groupby(conflict_key_cols)["price_decimal"].nunique()
    conflict_keys = price_nunique[price_nunique > 1]
    conflict_key_tuples = set(conflict_keys.index)

    conflict_detail = []
    if len(conflict_key_tuples):
        idx = df_dedup.set_index(conflict_key_cols, drop=False).index
        conflict_mask = idx.isin(conflict_key_tuples)
        conflict_rows_df = df_dedup[conflict_mask]
        conflict_detail = (
            conflict_rows_df[conflict_key_cols + ["season", "price_decimal", "price"]]
            .sort_values(conflict_key_cols).to_dict("records")
        )
    else:
        conflict_mask = pd.Series(False, index=df_dedup.index)

    summary["n_conflicting_price_groups"] = len(conflict_key_tuples)
    summary["conflicting_price_group_rows"] = conflict_detail

    expected_books = {row["book_key"] for row in conflict_detail}
    expected_markets = {row["market_key"] for row in conflict_detail}
    expected_seasons = {row["season"] for row in conflict_detail}
    fail_closed = (
        len(conflict_key_tuples) != 3
        or expected_books - {"fanduel"}
        or expected_markets - {"player_shots_on_goal"}
        or expected_seasons - {"2023-24"}
    )
    summary["conflicting_price_matches_expectation"] = not fail_closed
    if fail_closed:
        raise SystemExit(
            "[FAIL-CLOSED] conflicting-price groups do not match the preregistered "
            f"expectation (exactly 3, all fanduel/2023-24/player_shots_on_goal). Found "
            f"{len(conflict_key_tuples)} group(s): books={sorted(expected_books)}, "
            f"markets={sorted(expected_markets)}, seasons={sorted(expected_seasons)}. "
            f"Detail: {conflict_detail}"
        )

    # price_decimal is kept in the final output (not dropped) -- it is the
    # raw, untouched API value, preserved verbatim alongside the derived
    # American-odds "price" column, matching this repo's raw-value-
    # preservation convention (goalie_name_raw etc.).
    df_final = df_dedup[~conflict_mask].reset_index(drop=True)

    summary["n_rows_final"] = len(df_final)
    return {"df_final": df_final, "summary": summary, "df_pre_exclusion": df}


def reconcile_pre_exclusion_coverage(df_pre_exclusion: pd.DataFrame, audit: dict) -> dict:
    """Coverage on ALL parsed 200-records, BEFORE any of the section 14.5
    exclusions, cross-checked against audit_summary.json's independently
    audited counts (task requirement: these must match exactly)."""
    df = df_pre_exclusion
    saves_2425 = df[(df["season"] == "2024-25") & (df["market_key"] == "player_total_saves")]
    sog_2425 = df[(df["season"] == "2024-25") & (df["market_key"] == "player_shots_on_goal")]
    sog_2324 = df[(df["season"] == "2023-24") & (df["market_key"] == "player_shots_on_goal")]

    found = {
        "2024-25_saves_events": int(saves_2425["event_id"].nunique()),
        "2024-25_sog_events": int(sog_2425["event_id"].nunique()),
        "2024-25_betonlineag_saves_events": int(saves_2425[saves_2425["book_key"] == "betonlineag"]["event_id"].nunique()),
        "2024-25_prizepicks_saves_events": int(saves_2425[saves_2425["book_key"] == "prizepicks"]["event_id"].nunique()),
        "2024-25_underdog_saves_events": int(saves_2425[saves_2425["book_key"] == "underdog"]["event_id"].nunique()),
        "2023-24_sog_events": int(sog_2324["event_id"].nunique()),
        "fanatics_rows_anywhere": int((df["book_key"] == "fanatics").sum()),
    }
    events_per_season_market = {
        (row["season"], row["market_key"]): row["n_events"]
        for row in audit["coverage"]["events_per_season_market"]
    }
    expected = {
        "2024-25_saves_events": events_per_season_market.get(("2024-25", "player_total_saves")),
        "2024-25_sog_events": events_per_season_market.get(("2024-25", "player_shots_on_goal")),
        "2024-25_betonlineag_saves_events": audit["coverage"]["key_questions"]["2024-25_events_with_betonlineag_saves"],
        "2024-25_prizepicks_saves_events": audit["coverage"]["key_questions"]["2024-25_events_with_prizepicks_saves"],
        "2024-25_underdog_saves_events": 0,
        "2023-24_sog_events": events_per_season_market.get(("2023-24", "player_shots_on_goal")),
        "fanatics_rows_anywhere": 0,
    }
    matches = {key: found[key] == expected[key] for key in found}
    return {"found": found, "expected": expected, "matches": matches, "all_match": all(matches.values())}


def overlap_with_existing_archive(df_final: pd.DataFrame, existing_path: Path) -> dict:
    """Read-only check (14.3): how many of the existing 2024-25 bettime
    archive's 21 events share an event_id with the new pass's included
    events. Never writes to existing_path."""
    existing = pd.read_parquet(existing_path)
    game_date = existing["game_date_eastern"]
    existing_2425 = existing[(game_date >= "2024-08-01") & (game_date <= "2025-07-31") & (existing["snapshot_pass"] == "bettime")]
    existing_events = set(existing_2425["event_id"].unique())

    new_2425_events = set(df_final[df_final["season"] == "2024-25"]["event_id"].unique())
    overlap = existing_events & new_2425_events
    return {
        "existing_2024_25_bettime_rows": int(len(existing_2425)),
        "existing_2024_25_bettime_events": len(existing_events),
        "new_pass_2024_25_included_events": len(new_2425_events),
        "n_overlapping_events": len(overlap),
        "overlapping_event_ids": sorted(overlap),
    }


def run_validations(df_final: pd.DataFrame) -> dict:
    """Sanity checks required by the task spec. Returns a dict of results;
    raises SystemExit on a hard violation (fail-closed, matching the rest
    of this script's discipline)."""
    checks = {}

    bad_side = df_final[~df_final["side"].isin(["Over", "Under"])]
    checks["n_rows_bad_side"] = int(len(bad_side))
    if len(bad_side):
        raise SystemExit(f"[FAIL] {len(bad_side)} row(s) with side outside {{Over, Under}}")

    bad_pass = df_final[df_final["snapshot_pass"] != SNAPSHOT_PASS]
    checks["n_rows_bad_snapshot_pass"] = int(len(bad_pass))
    if len(bad_pass):
        raise SystemExit(f"[FAIL] {len(bad_pass)} row(s) with snapshot_pass != '{SNAPSHOT_PASS}'")

    bad_gap = df_final[df_final["effective_gap_minutes"] < DRIFT_FLOOR_MINUTES]
    checks["n_rows_bad_gap"] = int(len(bad_gap))
    if len(bad_gap):
        raise SystemExit(f"[FAIL] {len(bad_gap)} row(s) with effective_gap_minutes < {DRIFT_FLOOR_MINUTES} survived exclusion")

    checks["n_rows_null_price"] = int(df_final["price"].isna().sum())
    checks["n_rows_null_line"] = int(df_final["line"].isna().sum())

    saves_rows = df_final[df_final["market_key"] == "player_total_saves"]
    sog_rows = df_final[df_final["market_key"] == "player_shots_on_goal"]
    checks["saves_goalie_match_rate"] = float(saves_rows["goalie_id"].notna().mean()) if len(saves_rows) else None
    checks["sog_rows_with_non_null_goalie_id"] = int(sog_rows["goalie_id"].notna().sum())

    return checks


def print_report(parsed: dict, excl: dict, pre_reconcile: dict, overlap: dict, checks: dict) -> None:
    df_final = excl["df_final"]
    summary = excl["summary"]

    print("\n[1] Record-level counts")
    print(f"  Total core_event=*.json records scanned: {parsed['n_records_total']}")
    print(f"  Non-200 records (excluded; not a numbered 14.5 rule): {len(parsed['non200'])}")
    for event_id, pass_name, status_code in parsed["non200"]:
        print(f"    event_id={event_id} pass={pass_name} status={status_code}")
    print(f"  Rows parsed pre-exclusion (saves + SOG outcomes): {len(parsed['rows'])}")

    print("\n[2] Pre-exclusion coverage reconciliation vs audit_summary.json")
    for key in pre_reconcile["found"]:
        found = pre_reconcile["found"][key]
        expected = pre_reconcile["expected"][key]
        status = "OK" if pre_reconcile["matches"][key] else "MISMATCH"
        print(f"    [{status}] {key}: found={found} expected={expected}")
    print(f"  All pre-exclusion coverage checks match: {pre_reconcile['all_match']}")

    print("\n[3] Rule 4 (14.5) -- commence-drift exclusion (effective_gap_minutes < 10)")
    print(f"  Events excluded: {summary['n_drift_excluded_events']}")
    for row in summary["drift_excluded_events"]:
        print(f"    event_id={row['event_id']} pass={row['pass_name']} season={row['season']} "
              f"{row['away_team']} @ {row['home_team']} requested_ts={row['requested_ts']} "
              f"true_commence_time={row['true_commence_time']} gap_min={row['effective_gap_minutes']:.3f}")
    print(f"  Rows dropped by drift exclusion: {summary['n_rows_dropped_by_drift_exclusion']}")

    print("\n[4] Rule 2 (14.5) -- byte-identical duplicate outcomes")
    print(f"  Extra copies dropped (post-drift-exclusion, what is actually removed from output): "
          f"{summary['n_duplicate_extra_copies_dropped']}")
    for row in summary["duplicate_extra_copies_per_season_book_market"]:
        print(f"    {row['season']} / {row['book_key']} / {row['market_key']}: {row['n_extra_copies']}")
    print(f"  Extra copies on ALL 200-status records, no drift exclusion (audit's own counting scope): "
          f"{summary['n_duplicate_extra_copies_all_200_records_no_drift_exclusion']}")
    for row in summary["duplicate_extra_copies_per_season_book_market_all_200_records"]:
        print(f"    {row['season']} / {row['book_key']} / {row['market_key']}: {row['n_extra_copies']}")

    print("\n[5] Rule 3 (14.5) -- conflicting-price groups")
    print(f"  Groups found: {summary['n_conflicting_price_groups']} "
          f"(matches expectation of exactly 3, all fanduel/2023-24/player_shots_on_goal: "
          f"{summary['conflicting_price_matches_expectation']})")
    for row in summary["conflicting_price_group_rows"]:
        print(f"    {row}")

    print("\n[6] Rule 5 (14.5) -- fanatics")
    print(f"  Fanatics rows found anywhere: {summary['fanatics_rows_found']}")

    print("\n[7] Final post-exclusion counts")
    print(f"  Final row count: {summary['n_rows_final']}")
    by_season_market = df_final.groupby(["season", "market_key"]).agg(
        rows=("event_id", "size"), events=("event_id", "nunique"),
    )
    print(by_season_market.to_string())
    print("\n  By season / book / market:")
    by_book = df_final.groupby(["season", "book_key", "market_key"]).agg(
        rows=("event_id", "size"), events=("event_id", "nunique"),
    )
    print(by_book.to_string())

    print("\n[8] Goalie match rate (player_total_saves rows only)")
    print(f"  Match rate: {checks['saves_goalie_match_rate']}")
    print(f"  player_shots_on_goal rows with non-null goalie_id (should be 0, no matching attempted): "
          f"{checks['sog_rows_with_non_null_goalie_id']}")

    print("\n[9] Null/missing price or point checks")
    print(f"  Rows with null price: {checks['n_rows_null_price']}")
    print(f"  Rows with null line/point: {checks['n_rows_null_line']}")

    print("\n[10] Sanity checks")
    print(f"  Rows with side outside {{Over, Under}}: {checks['n_rows_bad_side']}")
    print(f"  Rows with snapshot_pass != 'bettime': {checks['n_rows_bad_snapshot_pass']}")
    print(f"  Rows with effective_gap_minutes < 10 surviving exclusion: {checks['n_rows_bad_gap']}")

    print("\n[11] Overlap with existing 2024-25 bettime archive (data/processed/saves_lines_snapshots.parquet, read-only)")
    print(f"  Existing 2024-25 bettime rows / events: {overlap['existing_2024_25_bettime_rows']} / "
          f"{overlap['existing_2024_25_bettime_events']}")
    print(f"  New pass 2024-25 included events: {overlap['new_pass_2024_25_included_events']}")
    print(f"  Overlapping event_ids: {overlap['n_overlapping_events']}")
    if overlap["overlapping_event_ids"]:
        print(f"    {overlap['overlapping_event_ids']}")


def build_summary_json(parsed, excl, pre_reconcile, overlap, checks) -> dict:
    df_final = excl["df_final"]
    by_season_market = (
        df_final.groupby(["season", "market_key"]).agg(rows=("event_id", "size"), events=("event_id", "nunique"))
        .reset_index().to_dict("records")
    )
    by_season_book_market = (
        df_final.groupby(["season", "book_key", "market_key"]).agg(rows=("event_id", "size"), events=("event_id", "nunique"))
        .reset_index().to_dict("records")
    )
    return {
        "generated_at": datetime.now(timezone.utc).strftime(TS_FORMAT),
        "source_pass_dir": str(PASS_DIR),
        "n_records_total": parsed["n_records_total"],
        "non200_records": [
            {"event_id": eid, "pass_name": pn, "status_code": sc} for eid, pn, sc in parsed["non200"]
        ],
        "n_rows_pre_exclusion": len(parsed["rows"]),
        "pre_exclusion_coverage_reconciliation": pre_reconcile,
        "exclusion_rule4_drift": {
            "floor_minutes": DRIFT_FLOOR_MINUTES,
            "n_events_excluded": excl["summary"]["n_drift_excluded_events"],
            "events_excluded": excl["summary"]["drift_excluded_events"],
            "n_rows_dropped": excl["summary"]["n_rows_dropped_by_drift_exclusion"],
        },
        "exclusion_rule2_duplicates": {
            "n_extra_copies_dropped_post_drift_exclusion": excl["summary"]["n_duplicate_extra_copies_dropped"],
            "per_season_book_market_post_drift_exclusion": excl["summary"]["duplicate_extra_copies_per_season_book_market"],
            "n_extra_copies_all_200_records_no_drift_exclusion": excl["summary"]["n_duplicate_extra_copies_all_200_records_no_drift_exclusion"],
            "per_season_book_market_all_200_records": excl["summary"]["duplicate_extra_copies_per_season_book_market_all_200_records"],
            "audit_expected_total": 5296,
            "audit_expected_per_season_book_market": [
                {"season": "2023-24", "book_key": "betmgm", "market_key": "player_shots_on_goal", "n_extra_copies": 4},
                {"season": "2023-24", "book_key": "fanduel", "market_key": "player_shots_on_goal", "n_extra_copies": 5280},
                {"season": "2024-25", "book_key": "bovada", "market_key": "player_shots_on_goal", "n_extra_copies": 2},
                {"season": "2024-25", "book_key": "prizepicks", "market_key": "player_shots_on_goal", "n_extra_copies": 10},
            ],
        },
        "exclusion_rule3_conflicting_price": {
            "n_groups": excl["summary"]["n_conflicting_price_groups"],
            "matches_expectation": excl["summary"]["conflicting_price_matches_expectation"],
            "groups": excl["summary"]["conflicting_price_group_rows"],
        },
        "exclusion_rule5_fanatics": {
            "n_rows_found": excl["summary"]["fanatics_rows_found"],
        },
        "n_rows_final": excl["summary"]["n_rows_final"],
        "final_counts_by_season_market": by_season_market,
        "final_counts_by_season_book_market": by_season_book_market,
        "goalie_match_rate_saves_rows": checks["saves_goalie_match_rate"],
        "sog_rows_with_non_null_goalie_id": checks["sog_rows_with_non_null_goalie_id"],
        "n_rows_null_price": checks["n_rows_null_price"],
        "n_rows_null_line": checks["n_rows_null_line"],
        "sanity_checks": {
            "n_rows_bad_side": checks["n_rows_bad_side"],
            "n_rows_bad_snapshot_pass": checks["n_rows_bad_snapshot_pass"],
            "n_rows_bad_gap": checks["n_rows_bad_gap"],
        },
        "overlap_with_existing_2024_25_bettime_archive": overlap,
        "price_format_note": (
            "Raw API 'price' field is decimal odds (The Odds API default; no oddsFormat "
            "param was requested by purchase_core_bettime_passes.py, same convention as "
            "build_odds_snapshots.py's price_decimal column). Output column 'price' is "
            "converted to American odds via src/betting/odds_utils.decimal_to_american "
            "(the existing repo utility) to match the column's requested semantics; the "
            "untouched raw decimal value is preserved verbatim in 'price_decimal'."
        ),
    }


def main():
    print("[1/4] Parsing raw pass records...")
    parsed = parse_records(PASS_DIR)

    print("\n[2/4] Applying section 14.5 exclusion rules...")
    excl = apply_exclusions(parsed)

    print("\n[3/4] Running reconciliation and validation checks...")
    with open(AUDIT_PATH, "r", encoding="utf-8") as fh:
        audit = json.load(fh)
    pre_reconcile = reconcile_pre_exclusion_coverage(excl["df_pre_exclusion"], audit)
    overlap = overlap_with_existing_archive(excl["df_final"], EXISTING_SNAPSHOTS_PATH)
    checks = run_validations(excl["df_final"])

    print_report(parsed, excl, pre_reconcile, overlap, checks)

    print(f"\n[4/4] Saving outputs...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    excl["df_final"].to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved {len(excl['df_final'])} rows, {len(excl['df_final'].columns)} columns -> {OUTPUT_PATH}")

    summary_json = build_summary_json(parsed, excl, pre_reconcile, overlap, checks)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2, sort_keys=True, default=str)
    print(f"  Saved summary -> {SUMMARY_PATH}")

    if not pre_reconcile["all_match"]:
        print("\n[FAIL] pre-exclusion coverage reconciliation did not match audit_summary.json exactly")
        sys.exit(1)
    print("\n[OK] build complete")


if __name__ == "__main__":
    main()
