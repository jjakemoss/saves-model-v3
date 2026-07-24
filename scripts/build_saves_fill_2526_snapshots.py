"""
Build row-level snapshots from the purchased saves_fill_2526_202607 pass.

Binding contract: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 20
("2025-26 bet-time saves archive completion purchase"), section 20.9 in
particular ("Implemented result") -- the purchase and its independent audit
(scripts/audit_2526_bettime_saves_fill.py) are already done and VERDICT
CLEAN. This script is the free, zero-credit ingestion follow-on 20.9
explicitly flags as "not yet done at the time of writing."

This is a PURE PARSER with validation counts -- the direct analogue of
scripts/build_core_bettime_pass_snapshots.py, but for this pass. It performs
NO analysis, NO grading, NO EV, NO outcome joins, and NO model anything.
Anything beyond coverage/match-rate/uniqueness counts belongs in a later,
separately preregistered experiment script, not here (20.1 item 1: this
purchase pre-authorizes archive completion only, not any downstream modeling
or ingestion decision).

Source records are `savesfill_event={event_id}_signature={signature}.json`
envelopes written by scripts/purchase_2526_bettime_saves_fill.py. This is a
DIFFERENT envelope shape than both scripts/build_odds_snapshots.py's
odds_*.json cache files and scripts/build_core_bettime_pass_snapshots.py's
core_event=*.json records -- notably there is no `pass_name` key here. Per
the discipline established by build_core_bettime_pass_snapshots.py this
script imports and reuses build_odds_snapshots.py's row-shape and matching
conventions unchanged (build_base_lookup, match_goalie,
classify_snapshot_pass, season_from_eastern_date, commence_to_eastern_date)
rather than re-deriving goalie-matching logic here. The output schema is
build_odds_snapshots.py's own canonical 15-column OUTPUT_COLUMNS, reused
directly (not hand-copied) so the two column lists can never drift apart --
this guarantees the new rows are a drop-in pd.concat with the existing
data/processed/saves_lines_snapshots.parquet archive. This script never
writes to that archive; it produces a NEW SIBLING parquet only.

Market kept: player_total_saves only (the only market this pass purchased --
any other market key found is a schema surprise and stops the run).

Only ONE exclusion happens implicitly: HTTP non-200 records (expected: 1,
free EVENT_NOT_FOUND 404) and 200-status records with zero bookmakers
(expected: 41) simply produce no rows, since there is nothing to parse --
this is not a numbered exclusion rule the way
build_core_bettime_pass_snapshots.py's section 14.5 rules are; there is no
equivalent commence-drift, duplicate, conflicting-price, or fanatics rule
registered for this pass. Duplicate/conflicting-price rows found in the raw
data are documented, not dropped (the archive is append-only raw data,
matching build_odds_snapshots.py's own keep-and-document convention).

This script also reproduces, read-only, the registered coverage-jump
measurement from section 20.2/20.9: the BEFORE (751/1,232, ~60.9%) and AFTER
(existing archive's bettime rows unioned with this pass's new rows)
correctly-anchored event counts, using the exact cache-anchored,
min-gap-over-all-owned-snapshots, 300-second-tolerance test 20.2 defines.
Nothing here mutates data/processed/saves_lines_snapshots.parquet or
data/betting.db (never opened at all).

Usage:
    python scripts/build_saves_fill_2526_snapshots.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import build_odds_snapshots as bos  # noqa: E402 -- canonical parser/schema, reused unchanged
from fetch_historical_odds_snapshots import compute_bettime_ts  # noqa: E402
from purchase_core_bettime_passes import (  # noqa: E402
    CANONICAL_EVENTS_CACHE,
    _season_events,
    load_cached_events,
)

PASS_DIR = Path("data/raw/betting_lines/passes/saves_fill_2526_202607")
EXISTING_SNAPSHOTS_PATH = Path("data/processed/saves_lines_snapshots.parquet")  # read-only

OUTPUT_PATH = Path("data/processed/saves_fill_2526_202607_snapshots.parquet")
SUMMARY_PATH = Path("data/processed/saves_fill_2526_202607_snapshots_summary.json")

MARKET_KEY = "player_total_saves"
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Section 20.2's registered universe window (US/Eastern game dates).
UNIVERSE_START = "2025-10-07"
UNIVERSE_END = "2026-04-19"
# Section 20.2's registered min-gap tolerance for "correctly anchored".
ANCHOR_TOLERANCE_SECONDS = 300

# The canonical 15-column schema, reused directly (not hand-copied) from
# build_odds_snapshots.py so the two column lists can never drift apart.
OUTPUT_COLUMNS = bos.OUTPUT_COLUMNS


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, TS_FORMAT).replace(tzinfo=timezone.utc)


def iter_pass_records(pass_dir: Path):
    """Yield (path, record dict) for every savesfill_event=*.json record,
    sorted for determinism. plan_saves_fill_2526.json, run_log.jsonl,
    audit_summary.json, and execute_stdout.log are excluded by the glob
    pattern itself, not by a separate skip check."""
    for path in sorted(pass_dir.glob("savesfill_event=*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            yield path, json.load(fh)


def parse_records(pass_dir: Path, base_features_path: Path = bos.BASE_FEATURES_PATH):
    """Parse every record into output rows plus the bookkeeping needed for
    every validation. Returns a dict with:
      rows: list of row dicts (one per saves outcome), matching OUTPUT_COLUMNS
      n_records_total: total savesfill_event=*.json files scanned
      non200: list of (event_id, requested_ts, status_code) for skipped records
      n_zero_bookmakers: count of 200-status records with an empty bookmakers list
      n_with_saves_market: count of 200-status records that produced >=1 row
      n_other_market_no_saves: count of 200-status records with >=1 bookmaker
                                but zero player_total_saves markets (schema
                                surprise if nonzero)
      unexpected_market_keys: set of any market key seen other than
                               player_total_saves
      anchor_mismatches: list of (event_id, requested_ts, request_date,
                          event_bettime_ts) where the three don't agree
      alignment_gaps: list of (event_id, gap_seconds) for every 200-status
                       (resolved) event -- recomputed compute_bettime_ts(cache
                       commence_time) vs requested_ts
      closing_violations: list of (event_id, requested_ts, true_commence_time)
                           for any row that classifies as "closing" instead
                           of "bettime"
      n_multiplier_non_null: count of outcomes carrying a non-null
                              'multiplier' field (dropped from the output
                              schema for drop-in parity with
                              saves_lines_snapshots.parquet, but counted here
                              for honesty)
    """
    print(f"  Loading base features for goalie matching: {base_features_path}")
    lookup = bos.build_base_lookup(base_features_path)
    match_cache = {}

    rows = []
    non200 = []
    anchor_mismatches = []
    alignment_gaps = []
    closing_violations = []
    n_records_total = 0
    n_zero_bookmakers = 0
    n_with_saves_market = 0
    n_other_market_no_saves = 0
    n_multiplier_non_null = 0
    unexpected_market_keys = set()

    for path, record in iter_pass_records(pass_dir):
        n_records_total += 1
        event_meta = record["event"]
        event_id = event_meta["event_id"]
        status_code = record["status_code"]
        requested_ts = event_meta["bettime_ts"]
        request_date = record["request"]["params"]["date"]

        # Anchor consistency: requested_ts == request.params.date ==
        # event.bettime_ts for every record, 200 or not.
        if not (requested_ts == request_date == event_meta["bettime_ts"]):
            anchor_mismatches.append((event_id, requested_ts, request_date, event_meta["bettime_ts"]))

        if status_code != 200:
            non200.append((event_id, requested_ts, status_code))
            continue

        cache_commence_time = event_meta["commence_time"]
        registered_anchor = compute_bettime_ts(cache_commence_time)
        gap_seconds = abs((_parse_utc(requested_ts) - _parse_utc(registered_anchor)).total_seconds())
        alignment_gaps.append((event_id, gap_seconds))

        body = json.loads(record["raw_body"])
        data = body.get("data") or {}
        true_commence_time = data.get("commence_time")
        if not true_commence_time:
            raise SystemExit(f"[SCHEMA SURPRISE] 200 record with no data.commence_time: {path}")

        home_team = data.get("home_team") or event_meta.get("home_team")
        away_team = data.get("away_team") or event_meta.get("away_team")
        resolved_ts = body.get("timestamp")
        game_date_eastern = bos.commence_to_eastern_date(true_commence_time)
        snapshot_pass = bos.classify_snapshot_pass(requested_ts, true_commence_time)
        if snapshot_pass != "bettime":
            closing_violations.append((event_id, requested_ts, true_commence_time))

        bookmakers = data.get("bookmakers") or []
        if not bookmakers:
            n_zero_bookmakers += 1
            continue

        record_has_saves_market = False
        for bookmaker in bookmakers:
            book = bookmaker.get("key")
            for market in bookmaker.get("markets") or []:
                market_key = market.get("key")
                if market_key != MARKET_KEY:
                    unexpected_market_keys.add(market_key)
                    continue
                record_has_saves_market = True
                for outcome in market.get("outcomes") or []:
                    goalie_name_raw = outcome.get("description")
                    side = outcome.get("name")
                    if goalie_name_raw is None or side is None:
                        continue

                    if outcome.get("multiplier") is not None:
                        n_multiplier_non_null += 1

                    goalie_id, goalie_name_matched = bos.match_goalie(
                        true_commence_time, home_team, away_team, goalie_name_raw, lookup, match_cache,
                    )

                    rows.append({
                        "event_id": event_id,
                        "commence_time": true_commence_time,
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

        if record_has_saves_market:
            n_with_saves_market += 1
        else:
            n_other_market_no_saves += 1

    return {
        "rows": rows,
        "n_records_total": n_records_total,
        "non200": non200,
        "n_zero_bookmakers": n_zero_bookmakers,
        "n_with_saves_market": n_with_saves_market,
        "n_other_market_no_saves": n_other_market_no_saves,
        "unexpected_market_keys": unexpected_market_keys,
        "anchor_mismatches": anchor_mismatches,
        "alignment_gaps": alignment_gaps,
        "closing_violations": closing_violations,
        "n_multiplier_non_null": n_multiplier_non_null,
    }


def build_dataframe(parsed: dict) -> pd.DataFrame:
    """Fail-closed schema checks, then materialize the final DataFrame.
    Raises SystemExit on any hard violation -- never silently proceeds."""
    if parsed["anchor_mismatches"]:
        raise SystemExit(
            "[FAIL-CLOSED] requested_ts / request.params.date / event.bettime_ts disagree for "
            f"{len(parsed['anchor_mismatches'])} record(s): {parsed['anchor_mismatches'][:10]}"
        )

    if parsed["unexpected_market_keys"]:
        raise SystemExit(
            f"[SCHEMA SURPRISE] unexpected market key(s) found: {sorted(parsed['unexpected_market_keys'])} "
            "-- only player_total_saves is expected for this pass."
        )

    if parsed["n_other_market_no_saves"]:
        raise SystemExit(
            f"[SCHEMA SURPRISE] {parsed['n_other_market_no_saves']} 200-status record(s) had "
            ">=1 bookmaker but zero player_total_saves markets -- expected every 200-status "
            "record with bookmakers to carry the saves market (41 zero-bookmaker + 439 "
            "saves-market should equal all 480 200-status records)."
        )

    if parsed["closing_violations"]:
        raise SystemExit(
            "[FAIL-CLOSED] one or more records classified as 'closing' instead of the expected "
            f"'bettime' (they are bettime-anchored by construction, so this should never happen): "
            f"{parsed['closing_violations']}"
        )

    df = pd.DataFrame(parsed["rows"], columns=OUTPUT_COLUMNS)
    if len(df) > 0:
        df["goalie_id"] = df["goalie_id"].astype("Int64")
    return df


def run_validations(df: pd.DataFrame, parsed: dict) -> dict:
    """Print every validation the task requires. Raises SystemExit on a hard
    violation. Returns a dict of findings for the summary JSON."""
    findings = {}

    print("\n[1] Record and row counts")
    print(f"  Total records scanned: {parsed['n_records_total']}")
    print(f"  Non-200 records (excluded, no rows): {len(parsed['non200'])}")
    for event_id, requested_ts, status_code in parsed["non200"]:
        print(f"    event_id={event_id} requested_ts={requested_ts} status={status_code}")
    print(f"  200-status, zero bookmakers (excluded, no rows): {parsed['n_zero_bookmakers']}")
    print(f"  200-status, carrying player_total_saves market: {parsed['n_with_saves_market']}")
    print(f"  Rows produced: {len(df)}")
    findings["n_records_total"] = parsed["n_records_total"]
    findings["n_non200"] = len(parsed["non200"])
    findings["n_zero_bookmakers"] = parsed["n_zero_bookmakers"]
    findings["n_with_saves_market"] = parsed["n_with_saves_market"]
    findings["n_rows"] = len(df)

    print("\n[2] Market key check")
    print(f"  Unexpected market keys found: {sorted(parsed['unexpected_market_keys'])} (must be empty)")
    findings["unexpected_market_keys"] = sorted(parsed["unexpected_market_keys"])

    print("\n[3] Anchor consistency (requested_ts == request.params.date == event.bettime_ts)")
    print(f"  Mismatches: {len(parsed['anchor_mismatches'])} (must be 0)")
    findings["n_anchor_mismatches"] = len(parsed["anchor_mismatches"])

    print("\n[4] Anchor alignment: |requested_ts - compute_bettime_ts(cache commence_time)|")
    gaps = [g for _, g in parsed["alignment_gaps"]]
    gaps_series = pd.Series(gaps, dtype="float64")
    max_gap = float(gaps_series.max()) if len(gaps_series) else None
    print(f"  n resolved (200-status) events: {len(gaps_series)}")
    print(f"  gap_seconds distribution: min={gaps_series.min()} max={max_gap} "
          f"mean={gaps_series.mean()} median={gaps_series.median()}")
    findings["n_resolved_events"] = len(gaps_series)
    findings["alignment_gap_seconds_max"] = max_gap
    findings["alignment_gap_seconds_mean"] = float(gaps_series.mean()) if len(gaps_series) else None

    print("\n[5] snapshot_pass distribution")
    pass_counts = df["snapshot_pass"].value_counts().to_dict()
    print(f"  {pass_counts} (expect 100% bettime; closing rows would have hard-stopped above)")
    findings["snapshot_pass_counts"] = pass_counts

    print("\n[6] Uniqueness check on (event_id, requested_ts, book, goalie_name_raw, side)")
    key_cols = ["event_id", "requested_ts", "book", "goalie_name_raw", "side"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    n_dupe_rows = int(dup_mask.sum())
    exact_groups = 0
    conflicting_groups = 0
    conflicting_detail = []
    if n_dupe_rows:
        dupes = df.loc[dup_mask].sort_values(key_cols)
        n_dupe_groups = dupes[key_cols].drop_duplicates().shape[0]
        for key, g in dupes.groupby(key_cols):
            if g[["line", "price_decimal"]].drop_duplicates().shape[0] == 1:
                exact_groups += 1
            else:
                conflicting_groups += 1
                conflicting_detail.append(dict(zip(key_cols, key)))
        print(f"  Found {n_dupe_rows} rows in {n_dupe_groups} duplicate group(s)")
        print(f"    {exact_groups} exact-duplicate group(s) (identical line and price)")
        print(f"    {conflicting_groups} conflicting-price group(s) (same key, different line/price)")
        print("  Not dropped -- documented per build_odds_snapshots.py's keep-and-document "
              "convention (the archive is append-only raw data).")
    else:
        print("  [OK] no duplicates")
    findings["n_duplicate_rows"] = n_dupe_rows
    findings["n_exact_duplicate_groups"] = exact_groups
    findings["n_conflicting_price_groups"] = conflicting_groups
    findings["conflicting_price_groups"] = conflicting_detail

    print("\n[7] Both-sides check: fraction of (event, requested_ts, book, goalie) groups with both Over and Under")
    side_counts = df.groupby(["event_id", "requested_ts", "book", "goalie_name_raw"])["side"].nunique()
    frac_both = float((side_counts == 2).mean()) if len(side_counts) else None
    print(f"  {frac_both:.4%} of {len(side_counts)} groups have both sides" if frac_both is not None else "  no groups")
    findings["frac_groups_both_sides"] = frac_both
    findings["n_goalie_groups"] = int(len(side_counts))

    print("\n[8] Goalie match rate")
    overall_rate = float(df["goalie_id"].notna().mean()) if len(df) else None
    print(f"  Overall: {overall_rate:.2%}" if overall_rate is not None else "  no rows")
    all_2025_26 = bool((df["game_date_eastern"].map(bos.season_from_eastern_date) == "2025-26").all())
    print(f"  All rows are 2025-26: {all_2025_26}")
    findings["goalie_match_rate"] = overall_rate
    findings["all_rows_2025_26"] = all_2025_26

    print("\n[9] Null line / null price counts")
    n_null_line = int(df["line"].isna().sum())
    n_null_price = int(df["price_decimal"].isna().sum())
    print(f"  Null line: {n_null_line}")
    print(f"  Null price_decimal: {n_null_price}")
    findings["n_null_line"] = n_null_line
    findings["n_null_price_decimal"] = n_null_price

    bad_side = df[~df["side"].isin(["Over", "Under"])]
    print(f"  Rows with side outside {{Over, Under}}: {len(bad_side)}")
    findings["n_bad_side"] = int(len(bad_side))
    if len(bad_side):
        raise SystemExit(f"[FAIL] {len(bad_side)} row(s) with side outside {{Over, Under}}: "
                          f"{bad_side[['event_id', 'book', 'side']].to_dict('records')}")

    print("\n[10] Distinct events producing rows")
    n_distinct_events = int(df["event_id"].nunique())
    print(f"  {n_distinct_events} (expect 439)")
    findings["n_distinct_events"] = n_distinct_events

    print("\n[11] Multiplier field (dropped from output schema for parity; counted for honesty)")
    print(f"  Outcomes with a non-null multiplier: {parsed['n_multiplier_non_null']}")
    findings["n_multiplier_non_null"] = parsed["n_multiplier_non_null"]

    return findings


def _min_gap_seconds(requested_ts_list, anchor_dt):
    return min(abs((_parse_utc(ts) - anchor_dt).total_seconds()) for ts in requested_ts_list)


def measure_coverage_jump(df_new: pd.DataFrame) -> dict:
    """Read-only reproduction of section 20.2's registered coverage-jump
    measurement. Never writes to EXISTING_SNAPSHOTS_PATH."""
    events_by_id = load_cached_events(CANONICAL_EVENTS_CACHE)
    universe = _season_events(events_by_id, UNIVERSE_START, UNIVERSE_END)
    universe_ids = {event["id"] for event in universe}
    anchors = {event["id"]: _parse_utc(compute_bettime_ts(event["commence_time"])) for event in universe}

    existing = pd.read_parquet(EXISTING_SNAPSHOTS_PATH)
    existing_bettime = existing[
        (existing["snapshot_pass"] == "bettime") & (existing["event_id"].isin(universe_ids))
    ]
    before_req_by_event = existing_bettime.groupby("event_id")["requested_ts"].apply(list).to_dict()

    before_correct = set()
    for event_id in universe_ids:
        owned = before_req_by_event.get(event_id)
        if owned and _min_gap_seconds(owned, anchors[event_id]) <= ANCHOR_TOLERANCE_SECONDS:
            before_correct.add(event_id)

    new_bettime = df_new[df_new["event_id"].isin(universe_ids)]
    combined = pd.concat(
        [existing_bettime[["event_id", "requested_ts"]], new_bettime[["event_id", "requested_ts"]]],
        ignore_index=True,
    )
    after_req_by_event = combined.groupby("event_id")["requested_ts"].apply(list).to_dict()

    after_correct = set()
    for event_id in universe_ids:
        owned = after_req_by_event.get(event_id)
        if owned and _min_gap_seconds(owned, anchors[event_id]) <= ANCHOR_TOLERANCE_SECONDS:
            after_correct.add(event_id)

    mis_anchored_before = (set(before_req_by_event.keys()) & universe_ids) - before_correct
    # "Attempted" = in the 481-event buy set (20.2's 30 mis-anchored events
    # are, by construction, all part of the buy set). "Yielded new rows" is a
    # strictly smaller, honest subset: some buy-set events came back with
    # zero bookmakers this time (the same 41-event outcome that hit the buy
    # set generally) and so produced no new bettime row at all -- those
    # events remain mis-anchored after this pass, not "silently fixed."
    new_event_ids = set(new_bettime["event_id"].unique())
    re_bought_with_new_rows = mis_anchored_before & new_event_ids
    re_bought_now_correct = re_bought_with_new_rows & after_correct
    still_mis_anchored_after = mis_anchored_before - after_correct

    return {
        "universe_size": len(universe_ids),
        "before_correctly_anchored": len(before_correct),
        "before_pct": len(before_correct) / len(universe_ids),
        "after_correctly_anchored": len(after_correct),
        "after_pct": len(after_correct) / len(universe_ids),
        "delta": len(after_correct) - len(before_correct),
        "n_previously_mis_anchored_total": len(mis_anchored_before),
        "n_previously_mis_anchored_with_new_rows_this_pass": len(re_bought_with_new_rows),
        "n_previously_mis_anchored_now_correct": len(re_bought_now_correct),
        "n_previously_mis_anchored_still_mis_anchored": len(still_mis_anchored_after),
        "previously_mis_anchored_still_mis_anchored_event_ids": sorted(still_mis_anchored_after),
        "all_with_new_rows_now_correct": re_bought_with_new_rows == re_bought_now_correct,
    }


def print_coverage_report(coverage: dict) -> None:
    print("\n[12] Coverage-jump measurement (section 20.2 registered definitions, read-only)")
    print(f"  Universe (in-window 2025-26 cached events): {coverage['universe_size']}")
    print(f"  BEFORE correctly-anchored: {coverage['before_correctly_anchored']} "
          f"({coverage['before_pct']:.1%})")
    print(f"  AFTER correctly-anchored:  {coverage['after_correctly_anchored']} "
          f"({coverage['after_pct']:.1%})")
    print(f"  Delta: +{coverage['delta']}")
    print(f"  Previously mis-anchored events (owned but gap > 300s on every snapshot): "
          f"{coverage['n_previously_mis_anchored_total']} -- all 30 of these are, by "
          "construction, part of this pass's 481-event buy set (re-buy was attempted for all).")
    print(f"  ...of those, {coverage['n_previously_mis_anchored_with_new_rows_this_pass']} "
          "actually got a new bettime row from this pass (the rest fell into the same "
          "zero-bookmakers outcome the buy set saw generally)")
    print(f"  ...of THOSE, {coverage['n_previously_mis_anchored_now_correct']} are now "
          f"correctly anchored (all that got new rows: {coverage['all_with_new_rows_now_correct']})")
    if coverage["n_previously_mis_anchored_still_mis_anchored"]:
        print(f"  [NOTE] {coverage['n_previously_mis_anchored_still_mis_anchored']} previously "
              "mis-anchored event(s) remain mis-anchored after this pass (re-buy attempted, "
              "returned zero bookmakers, so no new bettime row landed for them): "
              f"{coverage['previously_mis_anchored_still_mis_anchored_event_ids']}")


def build_summary_json(parsed: dict, findings: dict, coverage: dict) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).strftime(TS_FORMAT),
        "source_pass_dir": str(PASS_DIR),
        "record_counts": {
            "n_records_total": findings["n_records_total"],
            "n_non200": findings["n_non200"],
            "non200_records": [
                {"event_id": eid, "requested_ts": ts, "status_code": sc}
                for eid, ts, sc in parsed["non200"]
            ],
            "n_zero_bookmakers": findings["n_zero_bookmakers"],
            "n_with_saves_market": findings["n_with_saves_market"],
            "n_rows": findings["n_rows"],
        },
        "unexpected_market_keys": findings["unexpected_market_keys"],
        "n_anchor_mismatches": findings["n_anchor_mismatches"],
        "alignment": {
            "n_resolved_events": findings["n_resolved_events"],
            "gap_seconds_max": findings["alignment_gap_seconds_max"],
            "gap_seconds_mean": findings["alignment_gap_seconds_mean"],
        },
        "snapshot_pass_counts": findings["snapshot_pass_counts"],
        "duplicates": {
            "n_duplicate_rows": findings["n_duplicate_rows"],
            "n_exact_duplicate_groups": findings["n_exact_duplicate_groups"],
            "n_conflicting_price_groups": findings["n_conflicting_price_groups"],
            "conflicting_price_groups": findings["conflicting_price_groups"],
        },
        "both_sides": {
            "n_goalie_groups": findings["n_goalie_groups"],
            "frac_groups_both_sides": findings["frac_groups_both_sides"],
        },
        "goalie_match_rate": findings["goalie_match_rate"],
        "all_rows_2025_26": findings["all_rows_2025_26"],
        "n_null_line": findings["n_null_line"],
        "n_null_price_decimal": findings["n_null_price_decimal"],
        "n_bad_side": findings["n_bad_side"],
        "n_distinct_events": findings["n_distinct_events"],
        "n_multiplier_non_null_dropped_from_schema": findings["n_multiplier_non_null"],
        "coverage_jump": coverage,
        "schema_note": (
            "15-column schema matches data/processed/saves_lines_snapshots.parquet exactly "
            "(OUTPUT_COLUMNS reused directly from build_odds_snapshots.py) so this parquet is "
            "a drop-in pd.concat with that archive. No 'multiplier' column is present -- "
            "outcome.multiplier is dropped for schema parity but its non-null count is reported "
            "above for honesty; market_key/season are implicit (player_total_saves only / "
            "always 2025-26) and are not stored as columns, matching the target schema."
        ),
    }


def main():
    print("[1/4] Parsing raw pass records...")
    parsed = parse_records(PASS_DIR)

    print("\n[2/4] Building and schema-checking the output DataFrame...")
    df = build_dataframe(parsed)

    print("\n[3/4] Running validations...")
    findings = run_validations(df, parsed)
    coverage = measure_coverage_jump(df)
    print_coverage_report(coverage)

    print(f"\n[4/4] Saving outputs...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved {len(df)} rows, {len(df.columns)} columns -> {OUTPUT_PATH}")

    summary_json = build_summary_json(parsed, findings, coverage)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2, sort_keys=True, default=str)
    print(f"  Saved summary -> {SUMMARY_PATH}")

    print("\n[OK] build complete")


if __name__ == "__main__":
    main()
