#!/usr/bin/env python3
"""Independent, read-only audit of the core_bettime_202607 season-scale purchase.

This script is the auditor, not the purchaser: it trusts nothing from
scripts/purchase_core_bettime_passes.py's own run-log summaries and
recomputes every claim directly from the raw cached response records under
data/raw/betting_lines/passes/core_bettime_202607/ (one
core_event={event_id}_signature={signature}.json file per historical
event-odds call, plus run_log.jsonl).

It checks, independently:
    1. record integrity      -- every record parses; signature ==
                                 sha256(canonical json of its request field);
                                 request params match the registered spec;
                                 no record contains an API key anywhere in
                                 its raw text.
    2. billing arithmetic     -- x-requests-last == 10 * (distinct market
                                 keys actually returned); grand total billed
                                 across all records; balance reconciliation
                                 (weakly-decreasing remaining, final value).
    3. 404s                   -- enumerate non-200 records, identify the
                                 events, and check whether a rescheduled
                                 replacement event id exists in the events
                                 cache and was itself purchased.
    4. anchor integrity       -- returned envelope timestamp must be at or
                                 before the requested bettime anchor; compare
                                 cached vs. returned commence_time; flag any
                                 event where the anchor landed after the true
                                 puck drop.
    5. coverage               -- per season/market/book event counts, unique
                                 player counts, paired Over/Under
                                 completeness, exact-duplicate outcome counts.
    6. pairing potential       -- read-only intersection against
                                 data/processed/saves_lines_snapshots.parquet
                                 to size the CLV/W6-usable overlap.
    7. duplicate/schema traps -- exact duplicate outcomes, one-sided
                                 (unpaired) outcomes, null points/descriptions,
                                 unexpected market keys, non-null multipliers.

Read-only on everything except its own two deliverables: this script file
and data/raw/betting_lines/passes/core_bettime_202607/audit_summary.json
(written once; refuses to overwrite an existing summary). Never touches
data/betting.db, never reads .env, never makes a network call. Deterministic:
given the same raw records, always produces the same report.

Usage:
    python scripts/audit_core_bettime_passes.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

# ---------------------------------------------------------------------------
# Fixed locations (all read-only except SUMMARY_PATH, written once).
# ---------------------------------------------------------------------------

PASS_CACHE_DIR = Path("data/raw/betting_lines/passes/core_bettime_202607")
EVENTS_CACHE_DIR = Path("data/raw/betting_lines/cache")
PARQUET_PATH = Path("data/processed/saves_lines_snapshots.parquet")
RUN_LOG_PATH = PASS_CACHE_DIR / "run_log.jsonl"
SUMMARY_PATH = PASS_CACHE_DIR / "audit_summary.json"

EASTERN = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# The registered spec this purchase was supposed to follow (independently
# re-stated here from the acquisition task description, not imported from
# purchase_core_bettime_passes.py -- an audit must not trust the purchaser's
# own constants to check the purchaser's own output).
# ---------------------------------------------------------------------------

BOOKMAKERS = (
    "draftkings",
    "fanduel",
    "betmgm",
    "williamhill_us",
    "fanatics",
    "bovada",
    "betonlineag",
    "underdog",
    "prizepicks",
)
EXPECTED_BOOKMAKERS_PARAM = ",".join(BOOKMAKERS)
EXPECTED_SPORT_PATH_TEMPLATE = "/sports/icehockey_nhl/events/{event_id}/odds"

PASS_SPECS: dict[str, dict[str, Any]] = {
    "combined-2024-25": {
        "season": "2024-25",
        "window": (date(2024, 10, 4), date(2025, 4, 17)),
        "markets": ("player_total_saves", "player_shots_on_goal"),
        "expected_events": 1313,
    },
    "sog-2023-24": {
        "season": "2023-24",
        "window": (date(2023, 10, 10), date(2024, 4, 18)),
        "markets": ("player_shots_on_goal",),
        "expected_events": 1313,
    },
}

TWO_SIDED_MARKETS = {"player_total_saves", "player_shots_on_goal"}

# Claims made by the acquisition run summaries -- reported here ONLY as
# comparison targets; every value on the left is recomputed independently
# from raw records elsewhere in this script.
CLAIMED_TOTAL_BILLED = 38570
CLAIMED_STARTING_BALANCE = 51465
CLAIMED_FINAL_REMAINING = 12895
CLAIMED_404_COUNT = 2


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _fmt_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_bettime_ts(commence_time: str) -> str:
    """Independent re-derivation of the archive's bet-time anchor convention:
    min(22:30Z game date, commence minus 30 minutes)."""
    commence_dt = _parse_utc(commence_time)
    game_date = commence_dt.astimezone(EASTERN).date()
    anchor = datetime(game_date.year, game_date.month, game_date.day, 22, 30, tzinfo=timezone.utc)
    return min(anchor, commence_dt - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")


def commence_to_eastern_date(commence_time: str) -> date:
    return _parse_utc(commence_time).astimezone(EASTERN).date()


def canonical_signature(request: dict[str, Any]) -> str:
    encoded = json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


# ---------------------------------------------------------------------------
# 1. Load raw records
# ---------------------------------------------------------------------------

def load_raw_records(cache_dir: Path) -> list[dict[str, Any]]:
    """Load every core_event=*.json record, keeping the raw text alongside
    the parsed JSON so later checks (apiKey scan, signature recompute) work
    off the exact bytes on disk, not a re-serialized copy."""
    entries = []
    for path in sorted(cache_dir.glob("core_event=*.json")):
        raw_text = path.read_text(encoding="utf-8")
        try:
            record = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            entries.append({"path": path, "raw_text": raw_text, "record": None, "parse_error": str(exc)})
            continue
        entries.append({"path": path, "raw_text": raw_text, "record": record, "parse_error": None})
    return entries


# ---------------------------------------------------------------------------
# 2. Record integrity: parsing, signatures, params, secrets
# ---------------------------------------------------------------------------

def check_record_integrity(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(entries)
    parse_failures = [str(e["path"]) for e in entries if e["record"] is None]

    signature_mismatches: list[str] = []
    filename_mismatches: list[str] = []
    param_violations: list[dict[str, Any]] = []
    apikey_hits: list[str] = []
    unknown_pass_names: list[str] = []
    out_of_window: list[dict[str, Any]] = []
    duplicate_event_within_pass: list[dict[str, Any]] = []

    seen_event_per_pass: dict[str, set[str]] = defaultdict(set)

    for entry in entries:
        rec = entry["record"]
        path = entry["path"]
        if rec is None:
            continue

        # apiKey leak scan over the exact raw bytes on disk.
        if "apikey" in entry["raw_text"].lower():
            apikey_hits.append(str(path))

        request = rec.get("request", {})
        signature_claimed = rec.get("signature")
        signature_recomputed = canonical_signature(request)
        if signature_claimed != signature_recomputed:
            signature_mismatches.append(str(path))

        # filename must embed event_id + signature exactly.
        event = rec.get("event", {})
        event_id = event.get("event_id")
        expected_name = f"core_event={event_id}_signature={signature_claimed}.json"
        if path.name != expected_name:
            filename_mismatches.append(str(path))

        pass_name = rec.get("pass_name")
        spec = PASS_SPECS.get(pass_name)
        if spec is None:
            unknown_pass_names.append(str(path))
            continue

        if event_id in seen_event_per_pass[pass_name]:
            duplicate_event_within_pass.append({"path": str(path), "pass_name": pass_name, "event_id": event_id})
        seen_event_per_pass[pass_name].add(event_id)

        params = request.get("params", {})
        bettime_ts = event.get("bettime_ts")
        expected_markets_param = ",".join(spec["markets"])
        reasons = []
        if params.get("bookmakers") != EXPECTED_BOOKMAKERS_PARAM:
            reasons.append(f"bookmakers={params.get('bookmakers')!r}")
        if params.get("markets") != expected_markets_param:
            reasons.append(f"markets={params.get('markets')!r} expected {expected_markets_param!r}")
        if params.get("includeMultipliers") != "true":
            reasons.append(f"includeMultipliers={params.get('includeMultipliers')!r}")
        if params.get("date") != bettime_ts:
            reasons.append(f"params.date={params.get('date')!r} != event.bettime_ts={bettime_ts!r}")
        if request.get("method") != "GET":
            reasons.append(f"method={request.get('method')!r}")
        expected_path = EXPECTED_SPORT_PATH_TEMPLATE.format(event_id=event_id)
        if request.get("path") != expected_path:
            reasons.append(f"path={request.get('path')!r} expected {expected_path!r}")
        # independent re-derivation of the anchor formula from commence_time
        commence_time = event.get("commence_time")
        if commence_time:
            recomputed_bettime = compute_bettime_ts(commence_time)
            if recomputed_bettime != bettime_ts:
                reasons.append(
                    f"bettime_ts={bettime_ts!r} != recomputed {recomputed_bettime!r} from commence_time"
                )
            game_date = commence_to_eastern_date(commence_time)
            start, end = spec["window"]
            if not (start <= game_date <= end):
                out_of_window.append(
                    {"path": str(path), "pass_name": pass_name, "event_id": event_id, "game_date": game_date.isoformat()}
                )
        if reasons:
            param_violations.append({"path": str(path), "pass_name": pass_name, "reasons": reasons})

    event_counts_per_pass = {pn: len(ids) for pn, ids in seen_event_per_pass.items()}

    return {
        "total_records_on_disk": total,
        "parse_failures": parse_failures,
        "signature_mismatches": signature_mismatches,
        "filename_mismatches": filename_mismatches,
        "unknown_pass_names": unknown_pass_names,
        "param_violations": param_violations,
        "out_of_season_window": out_of_window,
        "duplicate_event_within_pass": duplicate_event_within_pass,
        "apikey_hits": apikey_hits,
        "unique_event_count_per_pass": event_counts_per_pass,
        "expected_event_count_per_pass": {pn: spec["expected_events"] for pn, spec in PASS_SPECS.items()},
        "all_clean": not (
            parse_failures
            or signature_mismatches
            or filename_mismatches
            or unknown_pass_names
            or param_violations
            or out_of_window
            or duplicate_event_within_pass
            or apikey_hits
        ),
    }


# ---------------------------------------------------------------------------
# 3. Billing arithmetic
# ---------------------------------------------------------------------------

def distinct_market_keys(raw_body_parsed: dict[str, Any]) -> set[str]:
    data = raw_body_parsed.get("data") or {}
    keys: set[str] = set()
    for bookmaker in data.get("bookmakers") or []:
        for market in bookmaker.get("markets") or []:
            key = market.get("key")
            if key:
                keys.add(key)
    return keys


def check_billing(entries: list[dict[str, Any]]) -> dict[str, Any]:
    billing_rows = []
    violations = []
    for entry in entries:
        rec = entry["record"]
        if rec is None:
            continue
        status_code = rec.get("status_code")
        quota = rec.get("quota_headers", {})
        x_last = _as_int(quota.get("x-requests-last"))
        x_remaining = _as_int(quota.get("x-requests-remaining"))
        x_used = _as_int(quota.get("x-requests-used"))

        expected_last = None
        if status_code == 200:
            try:
                body = json.loads(rec.get("raw_body", ""))
            except json.JSONDecodeError:
                body = {}
            n_distinct = len(distinct_market_keys(body))
            expected_last = 10 * n_distinct
        elif status_code == 404:
            expected_last = 0

        if expected_last is not None and x_last != expected_last:
            violations.append(
                {
                    "path": str(entry["path"]),
                    "status_code": status_code,
                    "x_requests_last": x_last,
                    "expected": expected_last,
                }
            )

        billing_rows.append(
            {
                "path": str(entry["path"]),
                "pass_name": rec.get("pass_name"),
                "fetched_at": rec.get("fetched_at"),
                "status_code": status_code,
                "x_last": x_last,
                "x_remaining": x_remaining,
                "x_used": x_used,
            }
        )

    df = pd.DataFrame(billing_rows)
    sum_x_last = int(df["x_last"].fillna(0).sum())

    # Balance reconciliation, sorted by fetched_at (raw evidence order, not
    # the run log's claimed order).
    df_sorted = df.sort_values("fetched_at", kind="stable").reset_index(drop=True)
    remaining_series = df_sorted["x_remaining"]
    non_monotonic = []
    prev = None
    for i, row in df_sorted.iterrows():
        cur = row["x_remaining"]
        if prev is not None and cur is not None and prev is not None and cur > prev:
            non_monotonic.append(
                {"index": int(i), "path": row["path"], "prev_remaining": int(prev), "cur_remaining": int(cur)}
            )
        if cur is not None:
            prev = cur

    first_row = df_sorted.iloc[0]
    last_row = df_sorted.iloc[-1]
    implied_starting_balance = None
    if first_row["x_remaining"] is not None and first_row["x_last"] is not None:
        implied_starting_balance = int(first_row["x_remaining"]) + int(first_row["x_last"])

    final_remaining = int(last_row["x_remaining"]) if last_row["x_remaining"] is not None else None

    # Cross-check against run_log.jsonl's own cumulative figures (evidence,
    # not trusted claims -- reported for comparison only).
    run_log_entries = []
    if RUN_LOG_PATH.exists():
        for line in RUN_LOG_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                run_log_entries.append(json.loads(line))
    run_log_sum = sum(e.get("cumulative_header_last_credits", 0) or 0 for e in run_log_entries)

    return {
        "n_records_considered": len(billing_rows),
        "n_billing_violations": len(violations),
        "billing_violations": violations[:50],
        "sum_x_requests_last_across_all_records": sum_x_last,
        "claimed_total_billed": CLAIMED_TOTAL_BILLED,
        "matches_claimed_total_billed": sum_x_last == CLAIMED_TOTAL_BILLED,
        "run_log_sum_cumulative_header_last_credits": run_log_sum,
        "run_log_sum_matches_record_sum": run_log_sum == sum_x_last,
        "n_non_monotonic_remaining_transitions": len(non_monotonic),
        "non_monotonic_remaining_transitions": non_monotonic[:20],
        "first_record_by_fetched_at": {
            "path": first_row["path"],
            "fetched_at": first_row["fetched_at"],
            "x_last": _as_int(first_row["x_last"]),
            "x_remaining": _as_int(first_row["x_remaining"]),
        },
        "implied_starting_balance_before_this_purchase": implied_starting_balance,
        "claimed_starting_balance": CLAIMED_STARTING_BALANCE,
        "matches_claimed_starting_balance": implied_starting_balance == CLAIMED_STARTING_BALANCE,
        "last_record_by_fetched_at": {
            "path": last_row["path"],
            "fetched_at": last_row["fetched_at"],
            "x_remaining": final_remaining,
        },
        "claimed_final_remaining": CLAIMED_FINAL_REMAINING,
        "matches_claimed_final_remaining": final_remaining == CLAIMED_FINAL_REMAINING,
    }


# ---------------------------------------------------------------------------
# 4. 404s
# ---------------------------------------------------------------------------

def load_events_cache(cache_dir: Path) -> dict[str, dict[str, Any]]:
    events_by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(cache_dir.glob("events_date=*.json")):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for event in envelope.get("data") or []:
            if event.get("id") and event.get("commence_time"):
                events_by_id[event["id"]] = event
    return events_by_id


def find_team_pair_matches(
    events_by_id: dict[str, dict[str, Any]], team_a: str, team_b: str, around: date, radius_days: int = 30
) -> list[dict[str, Any]]:
    lo = around - timedelta(days=radius_days)
    hi = around + timedelta(days=radius_days)
    matches = []
    for eid, ev in events_by_id.items():
        commence = ev.get("commence_time")
        if not commence:
            continue
        d = _parse_utc(commence).date()
        if not (lo <= d <= hi):
            continue
        teams = {ev.get("home_team"), ev.get("away_team")}
        if team_a in teams and team_b in teams:
            matches.append({"event_id": eid, "commence_time": commence, "home_team": ev.get("home_team"), "away_team": ev.get("away_team")})
    return sorted(matches, key=lambda m: m["commence_time"])


def check_404s(entries: list[dict[str, Any]], events_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    non200 = []
    for entry in entries:
        rec = entry["record"]
        if rec is None:
            continue
        if rec.get("status_code") != 200:
            non200.append(entry)

    purchased_event_ids = {
        entry["record"]["event"]["event_id"] for entry in entries if entry["record"] is not None
    }

    details = []
    for entry in non200:
        rec = entry["record"]
        event = rec.get("event", {})
        quota = rec.get("quota_headers", {})
        try:
            body = json.loads(rec.get("raw_body", ""))
        except json.JSONDecodeError:
            body = {}
        commence = event.get("commence_time")
        replacement_candidates = []
        if commence and event.get("home_team") and event.get("away_team"):
            replacement_candidates = [
                m
                for m in find_team_pair_matches(
                    events_by_id, event["home_team"], event["away_team"], _parse_utc(commence).date()
                )
                if m["event_id"] != event.get("event_id")
            ]
        replacement_purchased = [
            {**m, "purchased_in_this_pass": m["event_id"] in purchased_event_ids} for m in replacement_candidates
        ]
        details.append(
            {
                "path": str(entry["path"]),
                "pass_name": rec.get("pass_name"),
                "status_code": rec.get("status_code"),
                "x_requests_last": quota.get("x-requests-last"),
                "error_code": body.get("error_code"),
                "event": event,
                "fetched_at": rec.get("fetched_at"),
                "candidate_replacement_events": replacement_purchased,
            }
        )

    all_zero_cost = all(_as_int(d["x_requests_last"]) == 0 for d in details)
    all_event_not_found = all(d["error_code"] == "EVENT_NOT_FOUND" for d in details)

    return {
        "n_non_200": len(non200),
        "claimed_n_404": CLAIMED_404_COUNT,
        "matches_claimed_count": len(non200) == CLAIMED_404_COUNT,
        "all_zero_cost": all_zero_cost,
        "all_error_code_event_not_found": all_event_not_found,
        "details": details,
    }


# ---------------------------------------------------------------------------
# 5. Anchor integrity
# ---------------------------------------------------------------------------

def check_anchor_integrity(entries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    anchor_after_commence: list[dict[str, Any]] = []
    envelope_after_requested: list[dict[str, Any]] = []
    for entry in entries:
        rec = entry["record"]
        if rec is None or rec.get("status_code") != 200:
            continue
        request_date = rec.get("request", {}).get("params", {}).get("date")
        try:
            body = json.loads(rec.get("raw_body", ""))
        except json.JSONDecodeError:
            continue
        envelope_ts = body.get("timestamp")
        data_commence = (body.get("data") or {}).get("commence_time")
        cached_commence = rec.get("event", {}).get("commence_time")

        gap_seconds = None
        if request_date and envelope_ts:
            gap_seconds = (_parse_utc(request_date) - _parse_utc(envelope_ts)).total_seconds()
            if gap_seconds < 0:
                envelope_after_requested.append(
                    {
                        "path": str(entry["path"]),
                        "event_id": rec["event"]["event_id"],
                        "requested": request_date,
                        "envelope_timestamp": envelope_ts,
                        "gap_seconds": gap_seconds,
                    }
                )

        commence_drift_seconds = None
        if data_commence and cached_commence:
            commence_drift_seconds = (_parse_utc(data_commence) - _parse_utc(cached_commence)).total_seconds()

        anchor_after_true_commence = False
        if request_date and data_commence:
            anchor_after_true_commence = _parse_utc(request_date) >= _parse_utc(data_commence)
            if anchor_after_true_commence:
                anchor_after_commence.append(
                    {
                        "path": str(entry["path"]),
                        "event_id": rec["event"]["event_id"],
                        "pass_name": rec.get("pass_name"),
                        "requested_bettime": request_date,
                        "true_commence_time": data_commence,
                        "cached_commence_time": cached_commence,
                        "home_team": rec["event"].get("home_team"),
                        "away_team": rec["event"].get("away_team"),
                    }
                )

        rows.append(
            {
                "gap_seconds": gap_seconds,
                "commence_drift_seconds": commence_drift_seconds,
            }
        )

    df = pd.DataFrame(rows)
    gaps = df["gap_seconds"].dropna()
    drifts = df["commence_drift_seconds"].dropna().abs()

    gap_stats = {
        "n": int(gaps.shape[0]),
        "median_seconds": float(gaps.median()) if not gaps.empty else None,
        "p95_seconds": float(gaps.quantile(0.95)) if not gaps.empty else None,
        "max_seconds": float(gaps.max()) if not gaps.empty else None,
        "min_seconds": float(gaps.min()) if not gaps.empty else None,
    }
    drift_stats = {
        "n": int(drifts.shape[0]),
        "n_drift_gt_5min": int((drifts > 300).sum()),
        "n_drift_gt_30min": int((drifts > 1800).sum()),
        "median_seconds": float(drifts.median()) if not drifts.empty else None,
        "max_seconds": float(drifts.max()) if not drifts.empty else None,
    }

    return {
        "gap_requested_minus_envelope_seconds": gap_stats,
        "n_envelope_after_requested_ts_violations": len(envelope_after_requested),
        "envelope_after_requested_ts_violations": envelope_after_requested[:20],
        "commence_time_drift_abs_seconds": drift_stats,
        "n_anchor_at_or_after_true_commence": len(anchor_after_commence),
        "anchor_at_or_after_true_commence_events": anchor_after_commence,
    }


# ---------------------------------------------------------------------------
# 6. Coverage + 7. duplicates/schema traps (share the flattened outcomes df)
# ---------------------------------------------------------------------------

def flatten_outcomes(entries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        rec = entry["record"]
        if rec is None or rec.get("status_code") != 200:
            continue
        event = rec.get("event", {})
        event_id = event.get("event_id")
        season = event.get("season")
        pass_name = rec.get("pass_name")
        try:
            body = json.loads(rec.get("raw_body", ""))
        except json.JSONDecodeError:
            continue
        data = body.get("data") or {}
        for bookmaker in data.get("bookmakers") or []:
            book = bookmaker.get("key")
            for market in bookmaker.get("markets") or []:
                market_key = market.get("key")
                for outcome in market.get("outcomes") or []:
                    rows.append(
                        {
                            "event_id": event_id,
                            "season": season,
                            "pass_name": pass_name,
                            "book": book,
                            "market_key": market_key,
                            "side": outcome.get("name"),
                            "description": outcome.get("description"),
                            "point": outcome.get("point"),
                            "price": outcome.get("price"),
                            "multiplier": outcome.get("multiplier"),
                        }
                    )
    return pd.DataFrame(rows)


def check_coverage(outcomes: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}

    per_season_market_events = (
        outcomes.groupby(["season", "market_key"])["event_id"].nunique().rename("n_events").reset_index()
    )
    result["events_per_season_market"] = per_season_market_events.to_dict("records")

    per_season_market_book_events = (
        outcomes.groupby(["season", "market_key", "book"])["event_id"].nunique().rename("n_events").reset_index()
    )
    result["events_per_season_market_book"] = per_season_market_book_events.to_dict("records")

    per_season_market_players = (
        outcomes.groupby(["season", "market_key"])["description"].nunique().rename("n_unique_players").reset_index()
    )
    result["unique_players_per_season_market"] = per_season_market_players.to_dict("records")

    # Dedup exact duplicate outcomes first: same event/book/market/player/point/side.
    key_cols = ["event_id", "book", "market_key", "description", "point", "side"]
    dup_counts = outcomes.groupby(key_cols).size().rename("n").reset_index()
    dup_counts["n_extra_copies"] = dup_counts["n"] - 1
    dup_rows = dup_counts[dup_counts["n_extra_copies"] > 0]
    dup_by_season_book_market = (
        dup_rows.merge(outcomes[["event_id", "season"]].drop_duplicates(), on="event_id", how="left")
        .groupby(["season", "book", "market_key"])["n_extra_copies"]
        .sum()
        .reset_index()
    )
    result["exact_duplicate_extra_copies_per_season_book_market"] = dup_by_season_book_market.to_dict("records")
    result["total_exact_duplicate_extra_copies"] = int(dup_counts["n_extra_copies"].sum())

    # For duplicate groups, check whether price also matched (harmless dup)
    # or differed (a real conflicting-quote data trap).
    dup_groups_conflicting_price = 0
    if not dup_rows.empty:
        dedup_price_variety = (
            outcomes.merge(dup_rows[key_cols], on=key_cols, how="inner")
            .groupby(key_cols)["price"]
            .nunique()
        )
        dup_groups_conflicting_price = int((dedup_price_variety > 1).sum())
    result["duplicate_groups_with_conflicting_price"] = dup_groups_conflicting_price

    # Dedup before pairing analysis (drop exact-duplicate rows down to one).
    deduped = outcomes.drop_duplicates(subset=key_cols)
    pair_key_cols = ["event_id", "season", "book", "market_key", "description", "point"]
    side_sets = deduped.groupby(pair_key_cols)["side"].apply(lambda s: frozenset(s)).reset_index()
    side_sets["paired"] = side_sets["side"].apply(lambda s: {"Over", "Under"}.issubset(s))
    pairing_summary = (
        side_sets.groupby(["season", "book", "market_key"])
        .agg(n_groups=("paired", "size"), n_paired=("paired", "sum"))
        .reset_index()
    )
    pairing_summary["n_unpaired"] = pairing_summary["n_groups"] - pairing_summary["n_paired"]
    result["paired_ou_completeness_per_season_book_market"] = pairing_summary.to_dict("records")
    result["total_unpaired_outcome_groups"] = int(pairing_summary["n_unpaired"].sum())

    # Key questions from the audit brief.
    def events_with_book_market(season: str, market_key: str, book: str) -> int:
        subset = outcomes[
            (outcomes["season"] == season) & (outcomes["market_key"] == market_key) & (outcomes["book"] == book)
        ]
        return int(subset["event_id"].nunique())

    def events_with_book_any_market(season: str, book: str) -> int:
        subset = outcomes[(outcomes["season"] == season) & (outcomes["book"] == book)]
        return int(subset["event_id"].nunique())

    result["key_questions"] = {
        "2024-25_events_with_betonlineag_saves": events_with_book_market("2024-25", "player_total_saves", "betonlineag"),
        "2024-25_events_with_prizepicks_saves": events_with_book_market("2024-25", "player_total_saves", "prizepicks"),
        "2024-25_events_with_underdog_any_market": events_with_book_any_market("2024-25", "underdog"),
        "2023-24_events_with_underdog_any_market": events_with_book_any_market("2023-24", "underdog"),
    }

    # 2023-24 events with >= 2 books each having at least one paired SOG line.
    sog_2023 = side_sets[(side_sets["season"] == "2023-24") & (side_sets["market_key"] == "player_shots_on_goal")]
    paired_only = sog_2023[sog_2023["paired"]]
    books_per_event = paired_only.groupby("event_id")["book"].nunique()
    result["key_questions"]["2023-24_events_with_ge2_books_paired_sog"] = int((books_per_event >= 2).sum())

    # Schema traps.
    result["schema_traps"] = {
        "n_null_point": int(outcomes["point"].isna().sum()),
        "n_null_or_empty_description": int((outcomes["description"].isna() | (outcomes["description"] == "")).sum()),
        "n_non_null_multiplier": int(outcomes["multiplier"].notna().sum()),
        "market_keys_observed": sorted(outcomes["market_key"].dropna().unique().tolist()),
        "unexpected_market_keys": sorted(
            set(outcomes["market_key"].dropna().unique()) - TWO_SIDED_MARKETS
        ),
        "side_values_observed": sorted(outcomes["side"].dropna().unique().tolist()),
        "unexpected_side_values": sorted(set(outcomes["side"].dropna().unique()) - {"Over", "Under"}),
        "book_keys_observed": sorted(outcomes["book"].dropna().unique().tolist()),
        "book_keys_never_seen": sorted(set(BOOKMAKERS) - set(outcomes["book"].dropna().unique())),
    }

    return result


# ---------------------------------------------------------------------------
# 8. Pairing potential against saves_lines_snapshots.parquet (read-only)
# ---------------------------------------------------------------------------

def check_pairing_potential(outcomes: pd.DataFrame, parquet_path: Path) -> dict[str, Any]:
    if not parquet_path.exists():
        return {"error": f"parquet not found at {parquet_path}"}

    parquet_df = pd.read_parquet(parquet_path)

    start_2425, end_2425 = PASS_SPECS["combined-2024-25"]["window"]
    start_2324, end_2324 = PASS_SPECS["sog-2023-24"]["window"]

    parquet_df = parquet_df.copy()
    parquet_df["game_date_eastern_parsed"] = pd.to_datetime(parquet_df["game_date_eastern"]).dt.date

    saves_2425_purchase_events = set(
        outcomes[(outcomes["season"] == "2024-25") & (outcomes["market_key"] == "player_total_saves")][
            "event_id"
        ].unique()
    )
    parquet_2425_closing_events = set(
        parquet_df[
            (parquet_df["snapshot_pass"] == "closing")
            & (parquet_df["game_date_eastern_parsed"] >= start_2425)
            & (parquet_df["game_date_eastern_parsed"] <= end_2425)
        ]["event_id"].unique()
    )
    intersection_2425 = saves_2425_purchase_events & parquet_2425_closing_events

    sog_2324_purchase_events = set(
        outcomes[(outcomes["season"] == "2023-24") & (outcomes["market_key"] == "player_shots_on_goal")][
            "event_id"
        ].unique()
    )
    parquet_2324_bettime_events = set(
        parquet_df[
            (parquet_df["snapshot_pass"] == "bettime")
            & (parquet_df["game_date_eastern_parsed"] >= start_2324)
            & (parquet_df["game_date_eastern_parsed"] <= end_2324)
        ]["event_id"].unique()
    )
    intersection_2324 = sog_2324_purchase_events & parquet_2324_bettime_events

    return {
        "n_2024_25_purchase_events_with_saves": len(saves_2425_purchase_events),
        "n_parquet_2024_25_closing_events": len(parquet_2425_closing_events),
        "n_2024_25_clv_usable_intersection": len(intersection_2425),
        "n_2023_24_purchase_events_with_sog": len(sog_2324_purchase_events),
        "n_parquet_2023_24_bettime_saves_events": len(parquet_2324_bettime_events),
        "n_2023_24_sog_bettime_saves_intersection": len(intersection_2324),
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(summary: dict[str, Any]) -> None:
    ri = summary["record_integrity"]
    bill = summary["billing"]
    n404 = summary["not_found_404s"]
    anchor = summary["anchor_integrity"]
    cov = summary["coverage"]
    pair = summary["pairing_potential"]

    print("=" * 78)
    print("AUDIT: core_bettime_202607 season-scale purchase")
    print("=" * 78)

    print("\n--- 1. Record integrity ---")
    print(f"  total records on disk:        {ri['total_records_on_disk']}")
    print(f"  parse failures:                {len(ri['parse_failures'])}")
    print(f"  signature mismatches:          {len(ri['signature_mismatches'])}")
    print(f"  filename mismatches:           {len(ri['filename_mismatches'])}")
    print(f"  param/spec violations:         {len(ri['param_violations'])}")
    print(f"  out-of-season-window events:   {len(ri['out_of_season_window'])}")
    print(f"  duplicate event within pass:   {len(ri['duplicate_event_within_pass'])}")
    print(f"  apiKey text found:             {len(ri['apikey_hits'])}")
    print(f"  unique events per pass:        {ri['unique_event_count_per_pass']}")
    print(f"  expected events per pass:      {ri['expected_event_count_per_pass']}")
    print(f"  ALL CLEAN:                     {ri['all_clean']}")

    print("\n--- 2. Billing arithmetic ---")
    print(f"  records considered:                       {bill['n_records_considered']}")
    print(f"  billing formula violations:               {bill['n_billing_violations']}")
    print(f"  sum(x-requests-last) across all records:  {bill['sum_x_requests_last_across_all_records']}")
    print(f"  claimed total billed:                     {bill['claimed_total_billed']}")
    print(f"  MATCHES CLAIMED TOTAL:                    {bill['matches_claimed_total_billed']}")
    print(f"  run_log.jsonl cumulative sum:              {bill['run_log_sum_cumulative_header_last_credits']}")
    print(f"  run_log sum matches record sum:            {bill['run_log_sum_matches_record_sum']}")
    print(f"  non-monotonic remaining transitions:       {bill['n_non_monotonic_remaining_transitions']}")
    print(f"  implied starting balance (first record):   {bill['implied_starting_balance_before_this_purchase']}")
    print(f"  claimed starting balance:                  {bill['claimed_starting_balance']}")
    print(f"  MATCHES CLAIMED STARTING BALANCE:          {bill['matches_claimed_starting_balance']}")
    print(f"  final remaining (last record):             {bill['last_record_by_fetched_at']['x_remaining']}")
    print(f"  claimed final remaining:                   {bill['claimed_final_remaining']}")
    print(f"  MATCHES CLAIMED FINAL REMAINING:           {bill['matches_claimed_final_remaining']}")

    print("\n--- 3. 404s ---")
    print(f"  non-200 records found:      {n404['n_non_200']}")
    print(f"  claimed 404 count:          {n404['claimed_n_404']}")
    print(f"  MATCHES CLAIMED COUNT:      {n404['matches_claimed_count']}")
    print(f"  all zero-cost:              {n404['all_zero_cost']}")
    print(f"  all EVENT_NOT_FOUND:        {n404['all_error_code_event_not_found']}")
    for d in n404["details"]:
        ev = d["event"]
        print(f"    [{d['pass_name']}] event={ev['event_id']} {ev.get('away_team')} @ {ev.get('home_team')} "
              f"commence={ev.get('commence_time')} bettime={ev.get('bettime_ts')}")
        for cand in d["candidate_replacement_events"]:
            print(f"      replacement candidate: event={cand['event_id']} commence={cand['commence_time']} "
                  f"purchased_in_this_pass={cand['purchased_in_this_pass']}")

    print("\n--- 4. Anchor integrity ---")
    gs = anchor["gap_requested_minus_envelope_seconds"]
    print(f"  requested-minus-envelope gap (s): n={gs['n']} median={gs['median_seconds']} "
          f"p95={gs['p95_seconds']} max={gs['max_seconds']} min={gs['min_seconds']}")
    print(f"  envelope-after-requested violations: {anchor['n_envelope_after_requested_ts_violations']}")
    ds = anchor["commence_time_drift_abs_seconds"]
    print(f"  commence-time drift (abs s): n={ds['n']} median={ds['median_seconds']} max={ds['max_seconds']} "
          f"gt5min={ds['n_drift_gt_5min']} gt30min={ds['n_drift_gt_30min']}")
    print(f"  anchor at/after TRUE commence (possible in-game/post-game snapshot): "
          f"{anchor['n_anchor_at_or_after_true_commence']}")
    for ev in anchor["anchor_at_or_after_true_commence_events"][:20]:
        print(f"    event={ev['event_id']} [{ev['pass_name']}] {ev.get('away_team')} @ {ev.get('home_team')} "
              f"requested_bettime={ev['requested_bettime']} true_commence={ev['true_commence_time']} "
              f"cached_commence={ev['cached_commence_time']}")

    print("\n--- 5/7. Coverage, duplicates, schema traps ---")
    print("  events per season/market:")
    for row in cov["events_per_season_market"]:
        print(f"    {row}")
    print(f"  total exact-duplicate extra copies: {cov['total_exact_duplicate_extra_copies']}")
    print(f"  duplicate groups with conflicting price: {cov['duplicate_groups_with_conflicting_price']}")
    print("  exact-duplicate extra copies per season/book/market:")
    for row in cov["exact_duplicate_extra_copies_per_season_book_market"]:
        print(f"    {row}")
    print(f"  total unpaired (one-sided) O/U groups: {cov['total_unpaired_outcome_groups']}")
    print("  key questions:")
    for k, v in cov["key_questions"].items():
        print(f"    {k}: {v}")
    print("  schema traps:")
    for k, v in cov["schema_traps"].items():
        print(f"    {k}: {v}")

    print("\n--- 6. Pairing potential vs. saves_lines_snapshots.parquet ---")
    for k, v in pair.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent read-only audit of the core_bettime_202607 purchase. "
                    "Recomputes every claim from raw cached records; no network calls.",
    )
    parser.add_argument("--cache-dir", type=Path, default=PASS_CACHE_DIR)
    parser.add_argument("--events-cache-dir", type=Path, default=EVENTS_CACHE_DIR)
    parser.add_argument("--parquet-path", type=Path, default=PARQUET_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing audit_summary.json (default: refuse, append-only deliverable).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.summary_path.exists() and not args.force:
        print(f"error: {args.summary_path} already exists; refusing to overwrite (append-only deliverable). "
              "Pass --force to override.", file=sys.stderr)
        return 1

    entries = load_raw_records(args.cache_dir)
    if not entries:
        print(f"error: no core_event=*.json records found under {args.cache_dir}", file=sys.stderr)
        return 1

    record_integrity = check_record_integrity(entries)
    billing = check_billing(entries)
    events_by_id = load_events_cache(args.events_cache_dir)
    not_found_404s = check_404s(entries, events_by_id)
    anchor_integrity = check_anchor_integrity(entries)
    outcomes = flatten_outcomes(entries)
    coverage = check_coverage(outcomes)
    pairing_potential = check_pairing_potential(outcomes, args.parquet_path)

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_cache_dir": str(args.cache_dir),
        "record_integrity": record_integrity,
        "billing": billing,
        "not_found_404s": not_found_404s,
        "anchor_integrity": anchor_integrity,
        "coverage": coverage,
        "pairing_potential": pairing_potential,
    }

    print_report(summary)

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_path, "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    print(f"\nWrote audit summary to {args.summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
