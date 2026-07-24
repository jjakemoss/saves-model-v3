#!/usr/bin/env python3
"""Independent, read-only audit of the saves_fill_2526_202607 purchase.

Mirrors scripts/audit_alt_ladder_pilot.py's structure and stance: this
script is the auditor, not the purchaser. It trusts nothing from
scripts/purchase_2526_bettime_saves_fill.py's own run-log summaries or
module constants, and recomputes every claim directly from the raw cached
response records under
data/raw/betting_lines/passes/saves_fill_2526_202607/ (one
savesfill_event={event_id}_signature={signature}.json per historical
event-odds call, plus plan_saves_fill_2526.json and run_log.jsonl). Every
constant used as a comparison target below (bookmakers, market, sport
path, season window, alignment tolerance, expected buy-set size, credit
floor) is independently RESTATED here rather than imported from the
purchase script -- an audit must not trust the purchaser's own constants
to check the purchaser's own output.

Checks performed, all recomputed independently:
    1. record integrity  -- every record parses; signature ==
                             sha256(canonical json of its request field);
                             filename embeds event_id + signature; request
                             params match the registered spec (single
                             market, nine bookmakers, includeMultipliers,
                             date == recomputed compute_bettime_ts anchor);
                             season-window membership; no duplicate event
                             ids; no apiKey substring anywhere in raw text;
                             on-disk event id set == the frozen plan's buy
                             set exactly (neither missing nor extra ids).
    2. billing arithmetic -- x-requests-last == 10 * (distinct market keys
                             actually returned) on every 200 (0 on a
                             zero-market/404 response); balance-chain
                             reconciliation via a CONSTRUCTIVE/SEQUENTIAL
                             chain built from (x-requests-remaining desc,
                             x-requests-last desc) rather than fetched_at
                             order -- fetched_at has 1-second granularity
                             and same-second collisions produce
                             false-positive chain-break reports if used as
                             a sort key (the lesson from the alt-ladder
                             pilot audit); remaining never below the
                             registered credit floor; run-log cross-check.
    3. non-200s           -- enumerate, confirm zero-cost.
    4. alignment          -- per-event alignment_gap_seconds between the
                             request's date param and an independently
                             recomputed compute_bettime_ts(commence_time);
                             should be ~0 for a correctly-built buy.

Read-only on everything except its own deliverable:
data/raw/betting_lines/passes/saves_fill_2526_202607/audit_summary.json
(written once; refuses to overwrite without --force). Never touches
data/betting.db, never reads .env, never makes a network call.
Deterministic: same raw records, same report.

Usage:
    python scripts/audit_2526_bettime_saves_fill.py
    python scripts/audit_2526_bettime_saves_fill.py --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

# ---------------------------------------------------------------------------
# Fixed locations (all read-only except the summary path, written once).
# ---------------------------------------------------------------------------

PASS_CACHE_DIR = Path("data/raw/betting_lines/passes/saves_fill_2526_202607")
PLAN_FILE_NAME = "plan_saves_fill_2526.json"
RUN_LOG_NAME = "run_log.jsonl"

EASTERN = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# The registered spec this purchase was supposed to follow, independently
# restated here (not imported from purchase_2526_bettime_saves_fill.py).
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
EXPECTED_MARKETS = ("player_total_saves",)
EXPECTED_MARKETS_PARAM = ",".join(EXPECTED_MARKETS)
EXPECTED_SPORT_PATH_TEMPLATE = "/sports/icehockey_nhl/events/{event_id}/odds"

SEASON_WINDOW = (date(2025, 10, 7), date(2026, 4, 19))
ALIGNMENT_TOLERANCE_SECONDS = 300
EXPECTED_BUYSET_SIZE = 481
REGISTERED_CREDIT_FLOOR = 6055


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def compute_bettime_ts(commence_time: str) -> str:
    """Independent re-derivation of the archive's bet-time anchor
    convention: min(22:30Z game date, commence minus 30 minutes)."""
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
# Load raw records and the frozen plan
# ---------------------------------------------------------------------------

def load_raw_records(cache_dir: Path) -> list[dict[str, Any]]:
    """Load every savesfill_event=*.json record, keeping the raw text
    alongside the parsed JSON so later checks (apiKey scan, signature
    recompute) work off the exact bytes on disk."""
    entries = []
    for path in sorted(cache_dir.glob("savesfill_event=*.json")):
        raw_text = path.read_text(encoding="utf-8")
        try:
            record = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            entries.append({"path": path, "raw_text": raw_text, "record": None, "parse_error": str(exc)})
            continue
        entries.append({"path": path, "raw_text": raw_text, "record": record, "parse_error": None})
    return entries


def load_plan(cache_dir: Path) -> dict[str, Any] | None:
    plan_path = cache_dir / PLAN_FILE_NAME
    if not plan_path.exists():
        return None
    try:
        return json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# 1. Record integrity (incl. plan-vs-disk buy-set reconciliation)
# ---------------------------------------------------------------------------

def check_record_integrity(entries: list[dict[str, Any]], plan: dict[str, Any] | None) -> dict[str, Any]:
    total = len(entries)
    parse_failures = [str(e["path"]) for e in entries if e["record"] is None]

    signature_mismatches: list[str] = []
    filename_mismatches: list[str] = []
    param_violations: list[dict[str, Any]] = []
    apikey_hits: list[str] = []
    out_of_window: list[dict[str, Any]] = []
    duplicate_event_ids: list[dict[str, Any]] = []

    seen_event_ids: set[str] = set()

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

        event = rec.get("event", {})
        event_id = event.get("event_id")
        expected_name = f"savesfill_event={event_id}_signature={signature_claimed}.json"
        if path.name != expected_name:
            filename_mismatches.append(str(path))

        if event_id in seen_event_ids:
            duplicate_event_ids.append({"path": str(path), "event_id": event_id})
        seen_event_ids.add(event_id)

        params = request.get("params", {})
        bettime_ts = event.get("bettime_ts")
        reasons = []
        if params.get("bookmakers") != EXPECTED_BOOKMAKERS_PARAM:
            reasons.append(f"bookmakers={params.get('bookmakers')!r}")
        if params.get("markets") != EXPECTED_MARKETS_PARAM:
            reasons.append(f"markets={params.get('markets')!r} expected {EXPECTED_MARKETS_PARAM!r}")
        if params.get("includeMultipliers") != "true":
            reasons.append(f"includeMultipliers={params.get('includeMultipliers')!r}")
        if params.get("date") != bettime_ts:
            reasons.append(f"params.date={params.get('date')!r} != event.bettime_ts={bettime_ts!r}")
        if request.get("method") != "GET":
            reasons.append(f"method={request.get('method')!r}")
        expected_path = EXPECTED_SPORT_PATH_TEMPLATE.format(event_id=event_id)
        if request.get("path") != expected_path:
            reasons.append(f"path={request.get('path')!r} expected {expected_path!r}")
        commence_time = event.get("commence_time")
        if commence_time:
            recomputed_bettime = compute_bettime_ts(commence_time)
            if recomputed_bettime != bettime_ts:
                reasons.append(
                    f"bettime_ts={bettime_ts!r} != recomputed {recomputed_bettime!r} from commence_time"
                )
            game_date = commence_to_eastern_date(commence_time)
            start, end = SEASON_WINDOW
            if not (start <= game_date <= end):
                out_of_window.append(
                    {"path": str(path), "event_id": event_id, "game_date": game_date.isoformat()}
                )
        if reasons:
            param_violations.append({"path": str(path), "reasons": reasons})

    # Plan-vs-disk reconciliation: the on-disk event id set must exactly
    # equal the frozen plan's buy set.
    if plan is None:
        plan_reconciliation: dict[str, Any] = {"error": "plan file missing"}
        plan_recon_clean = False
    else:
        plan_buyset = plan.get("buyset") or []
        plan_ids = {row["event_id"] for row in plan_buyset}
        matches_expected_size = (
            plan.get("buyset_size") == EXPECTED_BUYSET_SIZE and len(plan_buyset) == EXPECTED_BUYSET_SIZE
        )
        on_disk_equals_plan = seen_event_ids == plan_ids
        plan_reconciliation = {
            "plan_buyset_size_field": plan.get("buyset_size"),
            "plan_buyset_list_len": len(plan_buyset),
            "expected_buyset_size": EXPECTED_BUYSET_SIZE,
            "matches_expected_buyset_size": matches_expected_size,
            "n_plan_ids": len(plan_ids),
            "n_on_disk_ids": len(seen_event_ids),
            "on_disk_equals_plan_buyset": on_disk_equals_plan,
            "missing_from_disk": sorted(plan_ids - seen_event_ids),
            "unexpected_on_disk": sorted(seen_event_ids - plan_ids),
        }
        plan_recon_clean = bool(matches_expected_size and on_disk_equals_plan)

    return {
        "total_records_on_disk": total,
        "parse_failures": parse_failures,
        "signature_mismatches": signature_mismatches,
        "filename_mismatches": filename_mismatches,
        "param_violations": param_violations,
        "out_of_season_window": out_of_window,
        "duplicate_event_ids": duplicate_event_ids,
        "apikey_hits": apikey_hits,
        "unique_event_count": len(seen_event_ids),
        "plan_reconciliation": plan_reconciliation,
        "all_clean": not (
            parse_failures
            or signature_mismatches
            or filename_mismatches
            or param_violations
            or out_of_window
            or duplicate_event_ids
            or apikey_hits
        )
        and plan_recon_clean,
    }


# ---------------------------------------------------------------------------
# 2. Billing arithmetic
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


def check_billing(entries: list[dict[str, Any]], run_log_path: Path) -> dict[str, Any]:
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

        expected_last = None
        n_distinct = None
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
                    "n_distinct_markets_returned": n_distinct,
                    "expected": expected_last,
                }
            )

        billing_rows.append(
            {
                "path": str(entry["path"]),
                "fetched_at": rec.get("fetched_at"),
                "status_code": status_code,
                "x_last": x_last,
                "x_remaining": x_remaining,
                "n_distinct_markets": n_distinct,
            }
        )

    df = pd.DataFrame(billing_rows)
    if df.empty:
        return {
            "n_records_considered": 0,
            "n_billing_violations": 0,
            "billing_violations": [],
            "sum_x_requests_last": 0,
            "run_log_sum_cumulative_header_last_credits": 0,
            "run_log_sum_matches_record_sum": True,
            "n_constructive_chain_breaks": 0,
            "constructive_chain_breaks": [],
            "n_naive_fetched_at_order_inversions_diagnostic_only": 0,
            "n_same_second_fetched_at_groups": 0,
            "n_remaining_below_registered_floor": 0,
            "remaining_below_registered_floor": [],
            "implied_starting_balance_before_this_pass": None,
            "final_remaining": None,
            "distinct_market_count_distribution": {},
            "all_clean": True,
        }

    sum_x_last = int(df["x_last"].fillna(0).sum())

    # Balance-chain reconciliation. fetched_at has 1-second granularity, so
    # multiple records can legitimately share a fetched_at value and no
    # on-disk field preserves exact dispatch order within a shared second
    # (the lesson from the alt-ladder pilot audit). The rigorous check
    # therefore verifies the CONSTRUCTIVE chain: sorting by
    # x-requests-remaining descending (x_last descending within ties, which
    # places a billing call before a zero-cost call that left the balance
    # unchanged) must yield a sequence in which every record's remaining
    # equals the previous record's remaining minus its own x_last. This is
    # a real, falsifiable check -- if any x_last did not match the actual
    # balance decrement, no such ordering would exist and breaks would
    # appear. No absolute starting balance is asserted here (unlike a
    # one-off pilot audit with a known pre-purchase balance): the first
    # record in the constructive chain has no predecessor to check against,
    # so only its IMPLIED starting balance is reported, not compared to
    # anything.
    known = df[df["x_remaining"].notna() & df["x_last"].notna()].copy()
    known["x_remaining"] = known["x_remaining"].astype(int)
    known["x_last"] = known["x_last"].astype(int)

    chain = known.sort_values(
        ["x_remaining", "x_last"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
    chain_breaks = []
    prev_remaining: int | None = None
    for i, row in chain.iterrows():
        if prev_remaining is not None:
            expected_remaining = prev_remaining - int(row["x_last"])
            if expected_remaining != int(row["x_remaining"]):
                chain_breaks.append(
                    {
                        "index": int(i),
                        "path": row["path"],
                        "prev_remaining": prev_remaining,
                        "x_last": int(row["x_last"]),
                        "observed_remaining": int(row["x_remaining"]),
                        "expected_remaining": expected_remaining,
                    }
                )
        prev_remaining = int(row["x_remaining"])

    below_floor = [
        {"path": row["path"], "remaining": int(row["x_remaining"])}
        for _, row in known.iterrows()
        if int(row["x_remaining"]) < REGISTERED_CREDIT_FLOOR
    ]

    # Diagnostic only: apparent inversions under naive fetched_at ordering
    # (expected to be nonzero purely from same-second timestamp collisions,
    # not a real integrity problem).
    df_naive = df.sort_values(["fetched_at", "path"], kind="stable").reset_index(drop=True)
    naive_inversions = 0
    same_second_groups = int((df["fetched_at"].value_counts() > 1).sum())
    prev = None
    for _, row in df_naive.iterrows():
        cur = row["x_remaining"]
        if cur is not None and pd.notna(cur):
            cur = int(cur)
            if prev is not None and cur > prev:
                naive_inversions += 1
            prev = cur

    implied_starting_balance = None
    final_remaining = None
    if not chain.empty:
        implied_starting_balance = int(chain.iloc[0]["x_remaining"]) + int(chain.iloc[0]["x_last"])
        final_remaining = int(chain.iloc[-1]["x_remaining"])

    run_log_entries = []
    if run_log_path.exists():
        for line in run_log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                run_log_entries.append(json.loads(line))
    run_log_sum = sum(e.get("cumulative_header_last_credits", 0) or 0 for e in run_log_entries)
    run_log_ok = (run_log_sum == sum_x_last) if run_log_path.exists() else True

    return {
        "n_records_considered": len(billing_rows),
        "n_billing_violations": len(violations),
        "billing_violations": violations[:50],
        "sum_x_requests_last": sum_x_last,
        "run_log_sum_cumulative_header_last_credits": run_log_sum,
        "run_log_sum_matches_record_sum": run_log_ok,
        "n_constructive_chain_breaks": len(chain_breaks),
        "constructive_chain_breaks": chain_breaks[:20],
        "n_naive_fetched_at_order_inversions_diagnostic_only": naive_inversions,
        "n_same_second_fetched_at_groups": same_second_groups,
        "n_remaining_below_registered_floor": len(below_floor),
        "remaining_below_registered_floor": below_floor[:20],
        "implied_starting_balance_before_this_pass": implied_starting_balance,
        "final_remaining": final_remaining,
        "distinct_market_count_distribution": {
            str(k): int(v) for k, v in df["n_distinct_markets"].value_counts(dropna=False).items()
        },
        "all_clean": not (violations or chain_breaks or below_floor) and run_log_ok,
    }


# ---------------------------------------------------------------------------
# 3. Non-200s
# ---------------------------------------------------------------------------

def check_non_200s(entries: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    for entry in entries:
        rec = entry["record"]
        if rec is None:
            continue
        if rec.get("status_code") != 200:
            quota = rec.get("quota_headers", {})
            try:
                body = json.loads(rec.get("raw_body", ""))
            except json.JSONDecodeError:
                body = {}
            details.append(
                {
                    "path": str(entry["path"]),
                    "status_code": rec.get("status_code"),
                    "x_requests_last": quota.get("x-requests-last"),
                    "error_code": body.get("error_code"),
                    "event": rec.get("event", {}),
                    "fetched_at": rec.get("fetched_at"),
                }
            )
    all_zero_cost = all(_as_int(d["x_requests_last"]) == 0 for d in details)
    return {
        "n_non_200": len(details),
        "all_zero_cost": all_zero_cost if details else True,
        "details": details,
        "all_clean": all_zero_cost if details else True,
    }


# ---------------------------------------------------------------------------
# 4. Alignment (per-event alignment_gap_seconds, independently recomputed)
# ---------------------------------------------------------------------------

def check_alignment(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Gap between the request's date param (what was actually asked for)
    and an independently recomputed compute_bettime_ts(commence_time)
    (what should have been asked for). Should be ~0 for every record in a
    correctly-built buy -- still computed and reported per-event, not
    assumed."""
    rows = []
    unresolvable = []
    for entry in entries:
        rec = entry["record"]
        if rec is None or rec.get("status_code") != 200:
            continue
        event = rec.get("event", {})
        event_id = event.get("event_id")
        commence_time = event.get("commence_time")
        request = rec.get("request", {})
        params = request.get("params", {})
        requested_date = params.get("date")
        if not commence_time or not requested_date:
            unresolvable.append({"path": str(entry["path"]), "reason": "missing commence_time or date param"})
            continue
        recomputed_anchor = compute_bettime_ts(commence_time)
        gap_seconds = abs((_parse_utc(requested_date) - _parse_utc(recomputed_anchor)).total_seconds())
        rows.append(
            {
                "event_id": event_id,
                "requested_date": requested_date,
                "recomputed_bettime_ts": recomputed_anchor,
                "alignment_gap_seconds": gap_seconds,
                "within_tolerance": gap_seconds <= ALIGNMENT_TOLERANCE_SECONDS,
            }
        )

    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    exceeding_records: list[dict[str, Any]] = []
    if not df.empty:
        gaps = df["alignment_gap_seconds"]
        summary = {
            "n_events": int(len(df)),
            "n_within_tolerance": int(df["within_tolerance"].sum()),
            "n_exceeding_tolerance": int((~df["within_tolerance"]).sum()),
            "gap_min": float(gaps.min()),
            "gap_median": float(gaps.median()),
            "gap_mean": float(gaps.mean()),
            "gap_max": float(gaps.max()),
        }
        exceeding_records = df[~df["within_tolerance"]].to_dict("records")

    return {
        "tolerance_seconds": ALIGNMENT_TOLERANCE_SECONDS,
        "summary": summary,
        "n_unresolvable": len(unresolvable),
        "unresolvable": unresolvable,
        "n_exceeding_tolerance": len(exceeding_records),
        "exceeding_tolerance": exceeding_records[:20],
        "all_clean": len(unresolvable) == 0 and len(exceeding_records) == 0,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(summary: dict[str, Any]) -> None:
    ri = summary["record_integrity"]
    bill = summary["billing"]
    non200 = summary["non_200s"]
    align = summary["alignment"]

    print("=" * 78)
    print("AUDIT: saves_fill_2526_202607 purchase (2025-26 bet-time completeness fill)")
    print("=" * 78)

    print("\n--- 1. Record integrity ---")
    print(f"  total records on disk:        {ri['total_records_on_disk']}")
    print(f"  parse failures:                {len(ri['parse_failures'])}")
    print(f"  signature mismatches:          {len(ri['signature_mismatches'])}")
    print(f"  filename mismatches:           {len(ri['filename_mismatches'])}")
    print(f"  param/spec violations:         {len(ri['param_violations'])}")
    print(f"  out-of-season-window events:   {len(ri['out_of_season_window'])}")
    print(f"  duplicate event ids:           {len(ri['duplicate_event_ids'])}")
    print(f"  apiKey text found:             {len(ri['apikey_hits'])}")
    print(f"  unique events on disk:         {ri['unique_event_count']}")
    print(f"  plan reconciliation:           {ri['plan_reconciliation']}")
    print(f"  ALL CLEAN:                     {ri['all_clean']}")

    print("\n--- 2. Billing arithmetic ---")
    print(f"  records considered:                       {bill['n_records_considered']}")
    print(f"  billing formula violations:               {bill['n_billing_violations']}")
    print(f"  sum(x-requests-last) all records:         {bill['sum_x_requests_last']}")
    print(f"  run_log sum matches record sum:            {bill['run_log_sum_matches_record_sum']}")
    print(f"  constructive balance-chain breaks:         {bill['n_constructive_chain_breaks']}")
    print(f"  naive fetched_at-order inversions (diag):  {bill['n_naive_fetched_at_order_inversions_diagnostic_only']} "
          f"(same-second fetched_at groups: {bill['n_same_second_fetched_at_groups']})")
    print(f"  remaining ever below floor ({REGISTERED_CREDIT_FLOOR}):     {bill['n_remaining_below_registered_floor']}")
    print(f"  implied starting balance before this pass: {bill['implied_starting_balance_before_this_pass']}")
    print(f"  final remaining:                           {bill['final_remaining']}")
    print(f"  distinct-market-count distribution:        {bill['distinct_market_count_distribution']}")
    print(f"  ALL CLEAN:                                 {bill['all_clean']}")

    print("\n--- 3. Non-200s ---")
    print(f"  non-200 records found:      {non200['n_non_200']}")
    print(f"  all zero-cost:              {non200['all_zero_cost']}")
    for d in non200["details"]:
        print(f"    {d}")
    print(f"  ALL CLEAN:                  {non200['all_clean']}")

    print("\n--- 4. Alignment (alignment_gap_seconds, independently recomputed) ---")
    print(f"  tolerance: {align['tolerance_seconds']}s")
    print(f"  summary: {align['summary']}")
    print(f"  unresolvable: {align['n_unresolvable']}")
    for u in align["unresolvable"][:10]:
        print(f"    {u}")
    print(f"  events exceeding tolerance: {align['n_exceeding_tolerance']}")
    for e in align["exceeding_tolerance"][:10]:
        print(f"    {e}")
    print(f"  ALL CLEAN:                  {align['all_clean']}")

    print("\n" + "=" * 78)
    print(f"VERDICT: {'CLEAN' if summary['overall_integrity_clean'] else 'NOT-CLEAN'}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent read-only audit of the saves_fill_2526_202607 purchase. "
                    "Recomputes every claim from raw cached records; no network calls.",
    )
    parser.add_argument("--cache-dir", type=Path, default=PASS_CACHE_DIR)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing audit_summary.json (default: refuse, append-only deliverable).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary_path = args.summary_path if args.summary_path is not None else args.cache_dir / "audit_summary.json"

    if summary_path.exists() and not args.force:
        print(
            f"error: {summary_path} already exists; refusing to overwrite (append-only deliverable). "
            "Pass --force to override.",
            file=sys.stderr,
        )
        return 1

    entries = load_raw_records(args.cache_dir)
    if not entries:
        print(f"error: no savesfill_event=*.json records found under {args.cache_dir}", file=sys.stderr)
        return 1

    plan = load_plan(args.cache_dir)
    run_log_path = args.cache_dir / RUN_LOG_NAME

    record_integrity = check_record_integrity(entries, plan)
    billing = check_billing(entries, run_log_path)
    non_200s = check_non_200s(entries)
    alignment = check_alignment(entries)

    overall_clean = bool(
        record_integrity["all_clean"]
        and billing["all_clean"]
        and non_200s["all_clean"]
        and alignment["all_clean"]
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_cache_dir": str(args.cache_dir),
        "record_integrity": record_integrity,
        "billing": billing,
        "non_200s": non_200s,
        "alignment": alignment,
        "overall_integrity_clean": overall_clean,
    }

    print_report(summary)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.force else "x"
    with open(summary_path, mode, encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    print(f"\nWrote audit summary to {summary_path}")

    return 0 if overall_clean else 1


if __name__ == "__main__":
    sys.exit(main())
