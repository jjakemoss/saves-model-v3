#!/usr/bin/env python3
"""Season-scale core bet-time purchase for The Odds API (2026-07 window).

Implements the two authorized core purchases from
docs/BREAKTHROUGH_MODEL_PLAN.md section 5.7 (the revised acquisition plan
after the completed, independently verified W1 probe -- see the "Probe
execution actuals" block right after that section) at the established
bet-time anchor:

    combined-2024-25   every 2024-25 regular-season event (2024-10-04 ..
                       2025-04-17 US/Eastern game dates), one historical
                       event-odds call carrying BOTH player_total_saves and
                       player_shots_on_goal (one call, up to 20 credits/
                       event: 10 credits x number of distinct markets
                       actually returned).
    sog-2023-24        every 2023-24 regular-season event (2023-10-10 ..
                       2024-04-18), one call carrying player_shots_on_goal
                       only (up to 10 credits/event). 2023-24 bettime saves
                       are already owned by the existing archive and are
                       deliberately NOT re-bought here.

Both passes use the nine named bookmakers verified by the W1 probe to bill
as one region-equivalent on the historical event-odds endpoint
(draftkings, fanduel, betmgm, williamhill_us, fanatics, bovada, betonlineag,
underdog, prizepicks) plus includeMultipliers=true. Billing rule, verified
by the probe: 10 credits x number of DISTINCT MARKETS ACTUALLY RETURNED per
call; empty responses are free.

This script is deliberately separate from both probe_opening_markets.py (a
bounded, fixed-sample probe) and fetch_historical_odds_snapshots.py (the
canonical phased archive fetcher, not modified here). Its plans are built
entirely from the cached events-list envelopes under
data/raw/betting_lines/cache/ (events_date=*.json), never from
data/betting.db.

DRY-RUN IS THE DEFAULT. This script never makes a network call unless
--execute is passed, and --execute additionally requires --max-credits.
Cache records are append-only under a NEW dedicated directory,
data/raw/betting_lines/passes/core_bettime_202607/, named
core_event={event_id}_signature={signature}.json; a matching existing
signature is always treated as already spent and is never re-requested.

Usage:
    python scripts/purchase_core_bettime_passes.py --pass combined-2024-25
    python scripts/purchase_core_bettime_passes.py --pass sog-2023-24
    python scripts/purchase_core_bettime_passes.py --pass combined-2024-25 \
        --execute --max-credits 100 --limit 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SPORT_KEY = "icehockey_nhl"
BASE_URL = "https://api.the-odds-api.com/v4/historical"
EVENT_ODDS_PATH = f"/sports/{SPORT_KEY}/events/{{event_id}}/odds"
CANONICAL_EVENTS_CACHE = Path("data/raw/betting_lines/cache")
DEFAULT_PASS_CACHE = Path("data/raw/betting_lines/passes/core_bettime_202607")
RUN_LOG_NAME = "run_log.jsonl"

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
QUOTA_HEADERS = ("x-requests-last", "x-requests-used", "x-requests-remaining")
EASTERN = ZoneInfo("America/New_York")
DEFAULT_CREDIT_FLOOR = 11500

# Pre-registered, fixed pass definitions (section 5.7). Not derived from the
# cache at runtime beyond event selection, so a stale/partial cache can't
# silently change what a pass is scoped to buy.
PASS_DEFS: dict[str, dict[str, Any]] = {
    "combined-2024-25": {
        "season": "2024-25",
        "window": ("2024-10-04", "2025-04-17"),
        "markets": ("player_total_saves", "player_shots_on_goal"),
        "max_credits_per_event": 20,
    },
    "sog-2023-24": {
        "season": "2023-24",
        "window": ("2023-10-10", "2024-04-18"),
        "markets": ("player_shots_on_goal",),
        "max_credits_per_event": 10,
    },
}


# ---------------------------------------------------------------------------
# Bet-time anchor + cached-events loading (mirrors probe_opening_markets.py)
# ---------------------------------------------------------------------------

def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def commence_to_eastern_date(commence_time: str) -> str:
    dt = _parse_utc(commence_time)
    return dt.astimezone(EASTERN).date().isoformat()


def compute_bettime_ts(commence_time: str) -> str:
    """Existing archive convention: min(22:30Z game date, start minus 30m)."""
    commence_dt = _parse_utc(commence_time)
    game_date = commence_dt.astimezone(EASTERN).date()
    anchor = datetime(game_date.year, game_date.month, game_date.day, 22, 30, tzinfo=timezone.utc)
    return min(anchor, commence_dt - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_cached_events(cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Load/deduplicate canonical historical events-list envelopes."""
    events_by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(cache_dir.glob("events_date=*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                envelope = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        for event in envelope.get("data") or []:
            if event.get("id") and event.get("commence_time"):
                events_by_id[event["id"]] = event
    return events_by_id


def _season_events(events_by_id: dict[str, dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    """Return one sorted, regular-season-only event row per cached event id,
    using the same Eastern game-date convention as the rest of the archive
    tooling (so a late west-coast game after midnight UTC is still assigned
    to the correct Eastern game date)."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    selected = []
    for event in events_by_id.values():
        commence = event.get("commence_time")
        event_id = event.get("id")
        if not commence or not event_id:
            continue
        event_date = date.fromisoformat(commence_to_eastern_date(commence))
        if start_date <= event_date <= end_date:
            selected.append(event)
    return sorted(selected, key=lambda event: (event["commence_time"], event["id"]))


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_pass(pass_name: str, events_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Full, deterministic, sorted event plan for one pass -- every cached
    event whose US/Eastern game date falls inside the pass's season window,
    sorted by (commence_time, id)."""
    pass_def = PASS_DEFS[pass_name]
    start, end = pass_def["window"]
    plan = []
    for event in _season_events(events_by_id, start, end):
        plan.append(
            {
                "season": pass_def["season"],
                "event_id": event["id"],
                "commence_time": event["commence_time"],
                "bettime_ts": compute_bettime_ts(event["commence_time"]),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            }
        )
    return plan


def request_signature(pass_name: str, plan_event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a stable non-secret signature and its canonical request fields."""
    markets = PASS_DEFS[pass_name]["markets"]
    request = {
        "method": "GET",
        "path": EVENT_ODDS_PATH.format(event_id=plan_event["event_id"]),
        "params": {
            "bookmakers": ",".join(BOOKMAKERS),
            "date": plan_event["bettime_ts"],
            "includeMultipliers": "true",
            "markets": ",".join(markets),
        },
    }
    encoded = json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), request


def _record_path(cache_dir: Path, event_id: str, signature: str) -> Path:
    return cache_dir / f"core_event={event_id}_signature={signature}.json"


def _load_cached_record(path: Path, signature: str) -> dict[str, Any] | None:
    """Treat any complete matching response as spent, including non-200
    replies -- a non-200 was still a billed/attempted call and must never be
    silently re-requested."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return record if record.get("signature") == signature else None


def classify_events(
    pass_name: str, plan: list[dict[str, Any]], cache_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the full plan into (cached, uncached) using on-disk signature
    records. Order is preserved (plan is already sorted by commence_time,
    id)."""
    cached: list[dict[str, Any]] = []
    uncached: list[dict[str, Any]] = []
    for event in plan:
        signature, request = request_signature(pass_name, event)
        path = _record_path(cache_dir, event["event_id"], signature)
        record = _load_cached_record(path, signature)
        enriched = dict(event, signature=signature, request=request, record_path=path)
        if record is not None:
            cached.append(enriched)
        else:
            uncached.append(enriched)
    return cached, uncached


# ---------------------------------------------------------------------------
# Atomic append-only cache writes
# ---------------------------------------------------------------------------

def _atomic_json_create(path: Path, payload: dict[str, Any]) -> None:
    """Append-only atomic write; a collision never replaces prior evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite pass cache record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite pass cache record: {path}")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _quota_headers(headers: Any) -> dict[str, str | None]:
    return {name: headers.get(name) for name in QUOTA_HEADERS}


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _load_api_key() -> str:
    """Load the API key from the environment or .env (API_KEY=... or
    THE_ODDS_API_KEY=...). Never printed or logged anywhere."""
    api_key = os.environ.get("API_KEY") or os.environ.get("THE_ODDS_API_KEY")
    if api_key:
        return api_key
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("API_KEY=") or stripped.startswith("THE_ODDS_API_KEY="):
                    return stripped.split("=", 1)[1].strip().strip("\"'")
    return ""


# ---------------------------------------------------------------------------
# Execute (spends credits -- never invoked by dry-run, requires --execute)
# ---------------------------------------------------------------------------

def execute(
    pass_name: str,
    candidates: list[dict[str, Any]],
    cache_dir: Path,
    max_credits: int,
    credit_floor: int,
) -> dict[str, Any]:
    """Fetch each candidate event exactly once. Reserves the worst-case
    per-event cost against max_credits before dispatching; stops cleanly
    (no error) once the next reservation would exceed the cap. Aborts (does
    not retry) on any non-200 response, any x-requests-remaining below
    credit_floor or below zero, or any network exception raised after a
    request has been dispatched (a dispatched request may have billed even
    if the response never arrives)."""
    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError("no API key found (set API_KEY or THE_ODDS_API_KEY)")

    max_credits_per_event = PASS_DEFS[pass_name]["max_credits_per_event"]

    billing: dict[str, Any] = {
        "pass_name": pass_name,
        "executed": True,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_at": None,
        "calls_attempted": 0,
        "calls_completed": 0,
        "cached_skips": 0,
        "cumulative_conservative_credits": 0,
        "cumulative_header_last_credits": 0,
        "header_last_available_for_calls": 0,
        "latest_headers": {header: None for header in QUOTA_HEADERS},
        "final_remaining": None,
        "events_not_found": 0,
        "aborted_reason": None,
        "records_written": [],
    }

    for event in candidates:
        # Reserve the worst known charge before sending; once sent, it may
        # bill even if the connection dies before a response reaches this
        # process, so the reservation must happen before dispatch, not after.
        if billing["cumulative_conservative_credits"] + max_credits_per_event > max_credits:
            break  # clean stop: next reservation would exceed --max-credits

        signature = event["signature"]
        request = event["request"]
        path = event["record_path"]

        billing["calls_attempted"] += 1
        billing["cumulative_conservative_credits"] += max_credits_per_event

        url = f"{BASE_URL}{request['path']}"
        params = dict(request["params"])
        params["apiKey"] = api_key
        request_url = f"{url}?{urllib.parse.urlencode(params)}"
        http_request = urllib.request.Request(
            request_url,
            headers={"Accept": "application/json", "User-Agent": "saves-model-v3-core-bettime-pass/1"},
        )
        try:
            with urllib.request.urlopen(http_request, timeout=30) as response:
                status_code = response.status
                response_headers = response.headers
                raw_body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            response_headers = exc.headers
            raw_body = exc.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            billing["aborted_reason"] = (
                "ambiguous network failure after request dispatch; no retry to avoid duplicate spend: "
                f"{type(exc).__name__}"
            )
            break

        quota_headers = _quota_headers(response_headers)
        billing["latest_headers"] = quota_headers
        header_last = _as_int(quota_headers["x-requests-last"])
        if header_last is not None:
            billing["cumulative_header_last_credits"] += header_last
            billing["header_last_available_for_calls"] += 1

        record = {
            "schema_version": 1,
            "signature": signature,
            "request": request,
            "event": {
                "season": event["season"],
                "event_id": event["event_id"],
                "commence_time": event["commence_time"],
                "bettime_ts": event["bettime_ts"],
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            },
            "pass_name": pass_name,
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status_code": status_code,
            "quota_headers": quota_headers,
            "raw_body": raw_body,
        }
        try:
            _atomic_json_create(path, record)
            billing["records_written"].append(str(path))
        except FileExistsError:
            # Another process (or a prior partial run) wrote this exact
            # signature first; treat it as cached and move on rather than
            # losing evidence of a call that may have just billed.
            cached = _load_cached_record(path, signature)
            if cached is None:
                billing["aborted_reason"] = "pass cache collision with unreadable or mismatched record"
                break

        billing["calls_completed"] += 1

        if status_code == 404:
            # EVENT_NOT_FOUND: a definitive, free, per-event answer (seen for
            # postponed games whose event id was reissued, e.g. the January
            # 2025 LA-wildfire reschedules). The record above is cached so the
            # id is never re-requested; the run continues.
            billing["events_not_found"] += 1
        elif status_code != 200:
            billing["aborted_reason"] = f"HTTP {status_code}; response cached and run stopped"
            break

        remaining = _as_int(quota_headers["x-requests-remaining"])
        billing["final_remaining"] = remaining
        if remaining is not None and remaining < 0:
            billing["aborted_reason"] = "x-requests-remaining reported below zero"
            break
        if remaining is not None and remaining < credit_floor:
            billing["aborted_reason"] = (
                f"x-requests-remaining ({remaining}) fell below --credit-floor ({credit_floor})"
            )
            break

        time.sleep(0.25)

    billing["ended_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return billing


def append_run_log(cache_dir: Path, pass_name: str, cached_skips: int, billing: dict[str, Any]) -> None:
    entry = {
        "pass_name": pass_name,
        "started_at": billing["started_at"],
        "ended_at": billing["ended_at"],
        "calls_attempted": billing["calls_attempted"],
        "calls_completed": billing["calls_completed"],
        "calls_skipped_cached": cached_skips,
        "cumulative_header_last_credits": billing["cumulative_header_last_credits"],
        "final_x_requests_remaining": billing["final_remaining"],
        "events_not_found": billing["events_not_found"],
        "aborted_reason": billing["aborted_reason"],
    }
    log_path = cache_dir / RUN_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_dry_run_report(
    pass_name: str,
    plan: list[dict[str, Any]],
    cached: list[dict[str, Any]],
    uncached: list[dict[str, Any]],
    limited: list[dict[str, Any]],
    limit: int | None,
) -> None:
    pass_def = PASS_DEFS[pass_name]
    max_per_event = pass_def["max_credits_per_event"]
    start, end = pass_def["window"]
    print(f"\n=== Dry-run plan: {pass_name} ===")
    print(f"  season window: {start} .. {end} (US/Eastern game dates)")
    print(f"  markets:       {','.join(pass_def['markets'])}")
    print(f"  bookmakers:    {','.join(BOOKMAKERS)} (nine named books, verified 1 region-equivalent)")
    print(f"  max credits/event: {max_per_event}")
    print(f"\n  total events in window (cached events-list union): {len(plan)}")
    print(f"  already-cached signature records:                   {len(cached)}")
    print(f"  uncached (not yet purchased):                       {len(uncached)}")
    print(f"  worst-case credits for ALL uncached events:         {len(uncached) * max_per_event}")
    if limit is not None:
        print(f"\n  --limit {limit}: this run would process the first {len(limited)} unfetched events")
        print(f"  worst-case credits for THIS RUN:                    {len(limited) * max_per_event}")
    if limited:
        print("\n  First events this run would target:")
        for event in limited[:8]:
            print(
                f"    event={event['event_id']} commence={event['commence_time']} "
                f"bettime={event['bettime_ts']} {event.get('away_team')} @ {event.get('home_team')}"
            )
    print("\nDry run only -- no network calls made. Pass --execute --max-credits N to actually fetch.")


def print_execute_summary(pass_name: str, cached_skips: int, billing: dict[str, Any]) -> None:
    print(f"\n=== Execute summary: {pass_name} ===")
    print(f"  started_at:  {billing['started_at']}")
    print(f"  ended_at:    {billing['ended_at']}")
    print(f"  calls_attempted:  {billing['calls_attempted']}")
    print(f"  calls_completed:  {billing['calls_completed']}")
    print(f"  calls_skipped_cached (already had a signature record): {cached_skips}")
    print(f"  cumulative x-requests-last credits actually billed: {billing['cumulative_header_last_credits']}")
    print(f"  latest quota headers: {billing['latest_headers']}")
    print(f"  final x-requests-remaining: {billing['final_remaining']}")
    print(f"  events_not_found (HTTP 404, cached, free): {billing['events_not_found']}")
    print(f"  aborted_reason: {billing['aborted_reason']}")
    if billing["records_written"]:
        print("  records written:")
        for path in billing["records_written"]:
            print(f"    {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-by-default season-scale core bet-time purchase for The Odds API. "
                    "See docs/BREAKTHROUGH_MODEL_PLAN.md section 5.7.",
    )
    parser.add_argument(
        "--pass",
        dest="pass_name",
        required=True,
        choices=sorted(PASS_DEFS),
        help="Which pre-registered pass to plan (and, with --execute, run).",
    )
    parser.add_argument("--execute", action="store_true", help="Make the historical API calls.")
    parser.add_argument(
        "--max-credits",
        type=int,
        default=None,
        help="Required with --execute. Hard cap on cumulative reserved (worst-case) credits this run.",
    )
    parser.add_argument(
        "--credit-floor",
        type=int,
        default=DEFAULT_CREDIT_FLOOR,
        help=f"Abort if x-requests-remaining falls below this after any call (default {DEFAULT_CREDIT_FLOOR}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of events processed this run to the first N unfetched events in "
             "sorted (commence_time, id) order. Omit to target every remaining uncached event.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_PASS_CACHE,
        help=f"Dedicated append-only pass cache (default: {DEFAULT_PASS_CACHE}).",
    )
    parser.add_argument(
        "--events-cache-dir",
        type=Path,
        default=CANONICAL_EVENTS_CACHE,
        help=f"Canonical events-list cache to plan from (default: {CANONICAL_EVENTS_CACHE}).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.execute and args.max_credits is None:
        raise SystemExit("error: --execute requires --max-credits")
    if args.max_credits is not None and args.max_credits < 1:
        raise SystemExit("error: --max-credits must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("error: --limit must be >= 1")

    events_by_id = load_cached_events(args.events_cache_dir)
    plan = plan_pass(args.pass_name, events_by_id)
    if not plan:
        raise SystemExit(f"error: zero cached events found for pass {args.pass_name!r} -- check events cache")

    cached, uncached = classify_events(args.pass_name, plan, args.cache_dir)
    limited = uncached if args.limit is None else uncached[: args.limit]

    if not args.execute:
        print_dry_run_report(args.pass_name, plan, cached, uncached, limited, args.limit)
        return 0

    max_credits_per_event = PASS_DEFS[args.pass_name]["max_credits_per_event"]
    worst_case = len(limited) * max_credits_per_event
    print(f"Executing pass={args.pass_name} candidates_this_run={len(limited)} "
          f"worst_case_credits={worst_case} max_credits={args.max_credits} credit_floor={args.credit_floor}")

    billing = execute(args.pass_name, limited, args.cache_dir, args.max_credits, args.credit_floor)
    append_run_log(args.cache_dir, args.pass_name, len(cached), billing)
    print_execute_summary(args.pass_name, len(cached), billing)
    return 0 if billing["aborted_reason"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
