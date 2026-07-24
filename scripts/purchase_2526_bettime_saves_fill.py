#!/usr/bin/env python3
"""2025-26 bet-time player_total_saves archive completeness fill (2026-07).

This is a DATA-ACQUISITION purchase, not a hypothesis test: it fills the
gap in the existing 2025-26 data/processed/saves_lines_snapshots.parquet
bet-time archive so that every in-window 2025-26 event ends up with a
correctly compute_bettime_ts-anchored player_total_saves snapshot. It does
not sample, does not test a market, and is not registered against any
ablation/experiment document -- it exists purely to re-anchor and complete
an archive whose existing bettime rows were found to include events whose
requested_ts drifted more than a few minutes from the archive's own
compute_bettime_ts convention.

The buy set is DETERMINISTIC -- a full-gap fill, not a random sample. There
is no numpy permutation, no seeded sampling, and no target_n. It is built
as follows:

    1. Universe: every in-window 2025-26 cached event, i.e.
       _season_events(load_cached_events(CANONICAL_EVENTS_CACHE),
       "2025-10-07", "2026-04-19") -- both helpers imported verbatim from
       purchase_core_bettime_passes.py. Verified size: 1,232.
    2. Correctly-anchored owned set (excluded from the buy): an event we
       already hold a usable bettime saves line for, at the anchor this
       script would actually request. That anchor is
       compute_bettime_ts(commence_time) with commence_time taken from the
       CACHE (the same source the purchase anchors from); the snapshot
       archive's own commence_time disagrees with the cache by up to 30
       minutes on 85 in-window events, so the exclusion test must anchor
       from the cache, not the snapshot, to test the anchor we buy at. Read
       data/processed/saves_lines_snapshots.parquet, filter
       snapshot_pass == "bettime". An event is correctly-anchored iff AT
       LEAST ONE of its owned bettime snapshots has
           gap_seconds = |requested_ts - anchor_from_cache| <= 300
       (a min-gap-over-all-snapshots test, deliberately NOT a
       single-representative-row dedup: 68 events carry two bettime
       snapshots at different requested_ts, and whether we already hold a
       line at the right anchor cannot depend on which arbitrary row a
       dedup happens to keep). Verified size: 751.
    3. BUY SET = every universe event whose id is NOT in the
       correctly-anchored set, preserving universe (commence_time,
       event_id) order. Verified size: 481 (451 truly-missing with no owned
       bettime snapshot at all, plus 30 owned-but-mis-anchored) -- enforced
       below as EXPECTED_BUYSET_SIZE; a mismatch is a fail-loud RuntimeError
       (STOP-and-investigate), never a silent continue.

Frozen plan artifact: the buy set is computed once, on this script's first
invocation (dry-run or --execute), and persisted to
<cache-dir>/plan_saves_fill_2526.json. Every later invocation LOADS that
frozen file rather than recomputing anything, so the buy set can never
change after it is first frozen -- exactly the discipline
purchase_alt_ladder_pilot.py uses for its own frozen plan files (atomic
create, reload on a losing race, and a _validate_plan_matches_registration
check that a persisted plan still matches this script's own constants).

Script and cache discipline otherwise mirrors purchase_alt_ladder_pilot.py
and purchase_core_bettime_passes.py exactly: dry-run is the default,
--execute additionally requires --max-credits, --max-credits may never
exceed the registered REGISTERED_MAX_CREDITS, --credit-floor may never be
set below the registered REGISTERED_CREDIT_FLOOR, the worst-case per-event
credit cost is reserved before dispatch, and a previously-recorded
signature (any complete response, 200 or non-200) is never re-requested.

The following primitives are imported verbatim from
purchase_core_bettime_passes.py rather than reimplemented: load_cached_events,
_season_events, compute_bettime_ts, _atomic_json_create, _load_cached_record,
_quota_headers, _as_int, _load_api_key, and the constants SPORT_KEY,
BASE_URL, EVENT_ODDS_PATH, BOOKMAKERS, QUOTA_HEADERS, CANONICAL_EVENTS_CACHE.

New dedicated append-only cache directory:
data/raw/betting_lines/passes/saves_fill_2526_202607/, record naming
savesfill_event={event_id}_signature={signature}.json -- a shape that
cannot collide with any other pass/probe archive's record naming.

Registered hard limits, enforced by this script and never raisable via a
flag: --max-credits may not exceed 5000; --credit-floor may not be set
below 6055. Both are checked against the LIVE x-requests-remaining header
after every call, never a locally-estimated figure.

Usage:
    python scripts/purchase_2526_bettime_saves_fill.py
    python scripts/purchase_2526_bettime_saves_fill.py \
        --execute --max-credits 500 --credit-floor 6055 --limit 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from purchase_core_bettime_passes import (  # noqa: E402
    BASE_URL,
    BOOKMAKERS,
    CANONICAL_EVENTS_CACHE,
    EVENT_ODDS_PATH,
    QUOTA_HEADERS,
    SPORT_KEY,
    _as_int,
    _atomic_json_create,
    _load_api_key,
    _load_cached_record,
    _quota_headers,
    _season_events,
    compute_bettime_ts,
    load_cached_events,
)

if SPORT_KEY != "icehockey_nhl":  # sanity: imported, not silently drifted; must not be skippable under -O
    raise RuntimeError(f"unexpected SPORT_KEY imported from purchase_core_bettime_passes: {SPORT_KEY!r}")

DEFAULT_PASS_CACHE = Path("data/raw/betting_lines/passes/saves_fill_2526_202607")
DEFAULT_SAVES_PARQUET = Path("data/processed/saves_lines_snapshots.parquet")
PLAN_FILE_NAME = "plan_saves_fill_2526.json"
RUN_LOG_NAME = "run_log.jsonl"

# Locked constants (this purchase's entire registered design). Not derived
# from any cache/parquet at import time, so a stale/partial cache can't
# silently change what this buy is scoped to.
MARKETS = ("player_total_saves",)
MAX_CREDITS_PER_EVENT = 10
SEASON_LABEL = "2025-26"
SEASON_WINDOW = ("2025-10-07", "2026-04-19")
ALIGNMENT_TOLERANCE_SECONDS = 300
EXPECTED_BUYSET_SIZE = 481  # independently verified; see module docstring

REGISTERED_MAX_CREDITS = 5000
REGISTERED_CREDIT_FLOOR = 6055


# ---------------------------------------------------------------------------
# Buy-set construction (deterministic full-gap fill -- no sampling)
# ---------------------------------------------------------------------------

def _to_utc_timestamp(value: str) -> pd.Timestamp:
    """Parse an archive timestamp string to a tz-aware UTC pandas Timestamp,
    localizing to UTC if the parsed value comes back tz-naive."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def build_buyset(
    events_cache_dir: Path, saves_parquet_path: Path
) -> tuple[list[dict[str, Any]], int, int]:
    """Return (buyset_events, universe_size, correctly_anchored_size).

    Universe: every in-window 2025-26 cached event (already sorted by
    (commence_time, event_id) by _season_events).

    Correctly-anchored (excluded from the buy): an event we already hold a
    usable bettime saves line for, at the anchor this script would actually
    request. The anchor is compute_bettime_ts(commence_time) taken from the
    CACHE event -- the same source the purchase itself anchors from. The
    snapshot archive's own commence_time can disagree with the cache by up
    to 30 minutes (85 in-window events), so anchoring the exclusion test
    from the snapshot would test a different anchor than the one we buy at;
    we anchor from the cache. An event is correctly-anchored iff AT LEAST
    ONE of its owned bettime snapshots has requested_ts within
    ALIGNMENT_TOLERANCE_SECONDS of that cache anchor -- a
    min-gap-over-all-snapshots test, deliberately NOT a
    single-representative-row dedup, because whether we already hold a line
    at the right anchor must not depend on which arbitrary row a dedup keeps
    (68 events carry two bettime snapshots at different requested_ts).

    Buy set: universe events NOT in the correctly-anchored set, preserving
    universe order."""
    events_by_id = load_cached_events(events_cache_dir)
    universe = _season_events(events_by_id, *SEASON_WINDOW)
    universe_ids = {event["id"] for event in universe}
    anchor_by_id = {
        event["id"]: _to_utc_timestamp(compute_bettime_ts(event["commence_time"]))
        for event in universe
    }

    snaps = pd.read_parquet(
        saves_parquet_path,
        columns=["event_id", "requested_ts", "snapshot_pass"],
    )
    bettime = snaps.loc[snaps["snapshot_pass"] == "bettime"]

    correctly_anchored_ids: set[str] = set()
    for row in bettime.itertuples(index=False):
        event_id = row.event_id
        if event_id not in universe_ids or event_id in correctly_anchored_ids:
            continue
        requested_ts = _to_utc_timestamp(row.requested_ts)
        gap_seconds = abs((requested_ts - anchor_by_id[event_id]).total_seconds())
        if gap_seconds <= ALIGNMENT_TOLERANCE_SECONDS:
            correctly_anchored_ids.add(event_id)

    buyset_events = [event for event in universe if event["id"] not in correctly_anchored_ids]
    return buyset_events, len(universe), len(correctly_anchored_ids)


# ---------------------------------------------------------------------------
# Frozen plan artifact: buy set computed once, on first invocation; every
# later invocation loads the persisted file rather than recomputing
# anything, so the buy set can never change after it is first frozen.
# ---------------------------------------------------------------------------

def _plan_path(cache_dir: Path) -> Path:
    return cache_dir / PLAN_FILE_NAME


def build_or_load_plan(cache_dir: Path, events_cache_dir: Path, saves_parquet_path: Path) -> dict[str, Any]:
    plan_path = _plan_path(cache_dir)
    if plan_path.exists():
        with open(plan_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    buyset_events, universe_size, correctly_anchored_size = build_buyset(events_cache_dir, saves_parquet_path)
    buyset_size = len(buyset_events)
    if buyset_size != EXPECTED_BUYSET_SIZE:
        raise RuntimeError(
            f"freshly-built buy set size ({buyset_size}) does not match the independently verified "
            f"expected size ({EXPECTED_BUYSET_SIZE}) -- STOP and investigate before purchasing; either "
            "this script's buy-set logic has drifted from its verified definition or the underlying "
            "events cache / saves_lines_snapshots.parquet has changed since that definition was verified."
        )

    plan = {
        "schema_version": 1,
        "buyset_size": buyset_size,
        "universe_size": universe_size,
        "correctly_anchored_size": correctly_anchored_size,
        "season": SEASON_LABEL,
        "season_window": list(SEASON_WINDOW),
        "tolerance_seconds": ALIGNMENT_TOLERANCE_SECONDS,
        "markets": list(MARKETS),
        "bookmakers": list(BOOKMAKERS),
        "max_credits_per_event": MAX_CREDITS_PER_EVENT,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "buyset": [
            {
                "event_id": event["id"],
                "commence_time": event["commence_time"],
                "bettime_ts": compute_bettime_ts(event["commence_time"]),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            }
            for event in buyset_events
        ],
    }
    try:
        _atomic_json_create(plan_path, plan)
    except FileExistsError:
        # A concurrent invocation won the race and wrote the same
        # deterministic plan first; load its persisted copy rather than
        # trusting this process's own in-memory computation, so every
        # caller ends up looking at the exact same frozen bytes on disk.
        with open(plan_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return plan


def _validate_plan_matches_registration(plan: dict[str, Any], plan_path: Path) -> None:
    """Defensive check: a persisted plan file must still match this
    script's own locked constants. Catches a stale/hand-edited plan file
    rather than silently trusting it -- never silently absorb a mismatch."""
    mismatches = []
    if plan.get("buyset_size") != EXPECTED_BUYSET_SIZE:
        mismatches.append(f"buyset_size: plan={plan.get('buyset_size')!r} expected {EXPECTED_BUYSET_SIZE!r}")
    if len(plan.get("buyset") or []) != EXPECTED_BUYSET_SIZE:
        mismatches.append(f"len(buyset): {len(plan.get('buyset') or [])} expected {EXPECTED_BUYSET_SIZE!r}")
    if plan.get("season_window") != list(SEASON_WINDOW):
        mismatches.append(f"season_window: plan={plan.get('season_window')!r} expected {list(SEASON_WINDOW)!r}")
    if plan.get("tolerance_seconds") != ALIGNMENT_TOLERANCE_SECONDS:
        mismatches.append(
            f"tolerance_seconds: plan={plan.get('tolerance_seconds')!r} expected {ALIGNMENT_TOLERANCE_SECONDS!r}"
        )
    if plan.get("markets") != list(MARKETS):
        mismatches.append(f"markets: plan={plan.get('markets')!r} expected {list(MARKETS)!r}")
    if plan.get("bookmakers") != list(BOOKMAKERS):
        mismatches.append(f"bookmakers: plan={plan.get('bookmakers')!r} expected {list(BOOKMAKERS)!r}")
    if plan.get("max_credits_per_event") != MAX_CREDITS_PER_EVENT:
        mismatches.append(
            f"max_credits_per_event: plan={plan.get('max_credits_per_event')!r} expected {MAX_CREDITS_PER_EVENT!r}"
        )
    if mismatches:
        raise RuntimeError(
            f"persisted plan {plan_path} no longer matches this script's locked constants -- "
            f"refusing to proceed: {mismatches}"
        )


# ---------------------------------------------------------------------------
# Request signature + append-only cache record naming
# ---------------------------------------------------------------------------

def request_signature(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    request = {
        "method": "GET",
        "path": EVENT_ODDS_PATH.format(event_id=event["event_id"]),
        "params": {
            "bookmakers": ",".join(BOOKMAKERS),
            "date": event["bettime_ts"],
            "includeMultipliers": "true",
            "markets": ",".join(MARKETS),
        },
    }
    encoded = json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), request


def _record_path(cache_dir: Path, event_id: str, signature: str) -> Path:
    return cache_dir / f"savesfill_event={event_id}_signature={signature}.json"


def classify_events(
    buyset: list[dict[str, Any]], cache_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the frozen buy set into (cached, uncached) using on-disk
    signature records. Order is preserved (frozen buy-set order)."""
    cached: list[dict[str, Any]] = []
    uncached: list[dict[str, Any]] = []
    for event in buyset:
        signature, request = request_signature(event)
        path = _record_path(cache_dir, event["event_id"], signature)
        record = _load_cached_record(path, signature)
        enriched = dict(event, signature=signature, request=request, record_path=path)
        if record is not None:
            cached.append(enriched)
        else:
            uncached.append(enriched)
    return cached, uncached


# ---------------------------------------------------------------------------
# Execute (spends credits -- never invoked by dry-run, requires --execute)
# ---------------------------------------------------------------------------

def execute(
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
    if the response never arrives). Mirrors
    purchase_core_bettime_passes.py's execute() exactly."""
    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError("no API key found (set API_KEY or THE_ODDS_API_KEY)")

    billing: dict[str, Any] = {
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
        if billing["cumulative_conservative_credits"] + MAX_CREDITS_PER_EVENT > max_credits:
            break  # clean stop: next reservation would exceed --max-credits

        signature = event["signature"]
        request = event["request"]
        path = event["record_path"]

        billing["calls_attempted"] += 1
        billing["cumulative_conservative_credits"] += MAX_CREDITS_PER_EVENT

        url = f"{BASE_URL}{request['path']}"
        params = dict(request["params"])
        params["apiKey"] = api_key
        request_url = f"{url}?{urllib.parse.urlencode(params)}"
        http_request = urllib.request.Request(
            request_url,
            headers={"Accept": "application/json", "User-Agent": "saves-model-v3-saves-fill-2526/1"},
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
                "season": SEASON_LABEL,
                "event_id": event["event_id"],
                "commence_time": event["commence_time"],
                "bettime_ts": event["bettime_ts"],
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            },
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
            # EVENT_NOT_FOUND: a definitive, free, per-event answer. The
            # record above is cached so the id is never re-requested; the
            # run continues.
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


def append_run_log(cache_dir: Path, cached_skips: int, billing: dict[str, Any]) -> None:
    entry = {
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
    plan: dict[str, Any],
    buyset: list[dict[str, Any]],
    cached: list[dict[str, Any]],
    uncached: list[dict[str, Any]],
    limited: list[dict[str, Any]],
    limit: int | None,
) -> None:
    print("\n=== Dry-run plan: 2025-26 bet-time player_total_saves completeness fill ===")
    print(f"  season:                 {plan['season']}")
    print(f"  season window:          {plan['season_window'][0]} .. {plan['season_window'][1]}")
    print(f"  markets:                {','.join(plan['markets'])}")
    print(f"  bookmakers:             {','.join(plan['bookmakers'])} (nine named books)")
    print(f"  max credits/event:      {plan['max_credits_per_event']}")
    print(f"  alignment tolerance:    {plan['tolerance_seconds']}s")
    print(f"\n  universe size (in-window 2025-26 cached events): {plan['universe_size']}")
    print(f"  correctly-anchored (owned, within tolerance):    {plan['correctly_anchored_size']}")
    print(f"  buy set size (universe minus correctly-anchored): {plan['buyset_size']}")
    if plan["buyset_size"] != EXPECTED_BUYSET_SIZE:
        print(f"  *** buyset_size ({plan['buyset_size']}) != EXPECTED_BUYSET_SIZE ({EXPECTED_BUYSET_SIZE}) ***")
    print(f"\n  already-cached signature records: {len(cached)}")
    print(f"  uncached (not yet purchased):     {len(uncached)}")
    print(f"  worst-case credits for ALL uncached: {len(uncached) * MAX_CREDITS_PER_EVENT}")
    if limit is not None:
        print(f"\n  --limit {limit}: this run would process the first {len(limited)} unfetched events")
        print(f"  worst-case credits for THIS RUN:     {len(limited) * MAX_CREDITS_PER_EVENT}")
    print("\n  First 5 events in the frozen buy set:")
    for event in buyset[:5]:
        print(
            f"    event={event['event_id']} commence={event['commence_time']} "
            f"bettime={event['bettime_ts']} {event.get('away_team')} @ {event.get('home_team')}"
        )
    print("\nDry run only -- no network calls made. Pass --execute --max-credits N to actually fetch.")


def print_execute_summary(cached_skips: int, billing: dict[str, Any]) -> None:
    print("\n=== Execute summary: 2025-26 bet-time player_total_saves completeness fill ===")
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
        description="Dry-run-by-default 2025-26 bet-time player_total_saves completeness-fill purchase "
                    "for The Odds API. Deterministic full-gap fill, no sampling.",
    )
    parser.add_argument("--execute", action="store_true", help="Make the historical API calls.")
    parser.add_argument(
        "--max-credits",
        type=int,
        default=None,
        help=f"Required with --execute. Hard cap on cumulative reserved (worst-case) credits this run. "
             f"May never exceed the registered cap of {REGISTERED_MAX_CREDITS}.",
    )
    parser.add_argument(
        "--credit-floor",
        type=int,
        default=REGISTERED_CREDIT_FLOOR,
        help=f"Abort if x-requests-remaining falls below this after any call. May never be set below the "
             f"registered floor of {REGISTERED_CREDIT_FLOOR} (default {REGISTERED_CREDIT_FLOOR}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of events processed this run to the first N unfetched events in the "
             "frozen buy set's order. Omit to target every remaining uncached event in the buy set "
             "(resumable across runs).",
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
    parser.add_argument(
        "--saves-parquet",
        type=Path,
        default=DEFAULT_SAVES_PARQUET,
        help=f"Existing bet-time archive to compute the correctly-anchored owned set from "
             f"(default: {DEFAULT_SAVES_PARQUET}).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.execute and args.max_credits is None:
        raise SystemExit("error: --execute requires --max-credits")
    if args.max_credits is not None:
        if args.max_credits < 1:
            raise SystemExit("error: --max-credits must be >= 1")
        if args.max_credits > REGISTERED_MAX_CREDITS:
            raise SystemExit(
                f"error: --max-credits may not exceed the registered cap of {REGISTERED_MAX_CREDITS}; "
                "a new registration is required to raise it."
            )
    if args.credit_floor < REGISTERED_CREDIT_FLOOR:
        raise SystemExit(
            f"error: --credit-floor may not be set below the registered floor of {REGISTERED_CREDIT_FLOOR}; "
            "a new registration is required to lower it."
        )
    if args.limit is not None and args.limit < 1:
        raise SystemExit("error: --limit must be >= 1")

    plan = build_or_load_plan(args.cache_dir, args.events_cache_dir, args.saves_parquet)
    _validate_plan_matches_registration(plan, _plan_path(args.cache_dir))

    buyset = plan["buyset"]
    cached, uncached = classify_events(buyset, args.cache_dir)
    limited = uncached if args.limit is None else uncached[: args.limit]

    if not args.execute:
        print_dry_run_report(plan, buyset, cached, uncached, limited, args.limit)
        return 0

    worst_case = len(limited) * MAX_CREDITS_PER_EVENT
    print(f"Executing saves-fill-2526 candidates_this_run={len(limited)} "
          f"worst_case_credits={worst_case} max_credits={args.max_credits} credit_floor={args.credit_floor}")

    billing = execute(limited, args.cache_dir, args.max_credits, args.credit_floor)
    append_run_log(args.cache_dir, len(cached), billing)
    print_execute_summary(len(cached), billing)
    return 0 if billing["aborted_reason"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
