#!/usr/bin/env python3
"""Alternate-saves one-sided-ladder feasibility pilot purchase (2026-07 window).

Implements the purchase design registered in
docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 19.3 (Experiment 16) --
that section is the binding contract for every pool definition, sample size,
sampling rule, and forbidden-action clause used here; this docstring
restates the shape of the run, not the rules themselves.

Two legs, each a separate invocation of this script via --leg:

    alt_only_2024_25    markets=player_total_saves_alternate only (up to 10
                         credits/event). Candidate pool: every event with a
                         betonlineag/player_total_saves row in
                         data/processed/core_bettime_202607_snapshots.parquet
                         (1,050 events), minus the 8 already-probed 2024-25
                         events (W1 probe), leaving a 1,043-event pool.
                         Target N = 120 new purchases.
    combined_2025_26     markets=player_total_saves,player_total_saves_alternate
                         (up to 20 credits/event -- guarantees a same-envelope
                         anchor, since the 2025-26 existing bettime archive is
                         NOT reliably compute_bettime_ts-aligned, 19.3).
                         Candidate pool: every in-window 2025-26 event
                         (2025-10-07..2026-04-19, 18.4's window) minus the 8
                         already-probed 2025-26 events, leaving a 1,224-event
                         pool. Target N = 35 new purchases.

Worst case: 120*10 + 35*20 = 1,900 credits, 100 below the registered
--max-credits 2000 hard cap (a genuine backstop, never itself a target).

Seeded sampling (19.3, fixed, never redrawn): each leg's candidate pool is
sorted by (commence_time, event_id) -- exactly as
purchase_core_bettime_passes.py's own _season_events already sorts -- given
a 0-based index in that order, then numpy.random.default_rng(42).permutation
(pool_size) draws a single full permutation of the pool. The target sample
is the first N pool events in permutation order. The FULL permutation (not
just the first N) is persisted to a frozen plan artifact
(<cache-dir>/plan_<leg>.json) the first time a leg is planned; every later
invocation (dry-run or --execute, including a crash-resume) loads that same
frozen file rather than recomputing the pool or redrawing the permutation,
so the sample can never be reordered, resized, or redrawn after this
registration was filed (19.6 items 5/6). --sample-size exists solely to
extend a frozen sample forward along the SAME persisted permutation, for the
narrow 19.6-item-10 case of replacing a confirmed non-200 event; it must
exceed the registered target_n and is never used to shrink or opportunistically
enlarge the sample.

Script and cache discipline mirrors scripts/purchase_core_bettime_passes.py
exactly (19.3): dry-run is the default, --execute additionally requires
--max-credits, plans are built entirely from the cached events-list
envelopes under data/raw/betting_lines/cache/ (never data/betting.db),
worst-case per-event credit reservation happens before dispatch, and a
previously-recorded signature (any complete response, 200 or non-200) is
never re-requested. The following primitives are imported verbatim from
purchase_core_bettime_passes.py rather than reimplemented, per the task's
explicit instruction: load_cached_events, _season_events, compute_bettime_ts,
plus the generic (pass-agnostic) helpers _atomic_json_create,
_load_cached_record, _quota_headers, _as_int, _load_api_key, and the
constants SPORT_KEY, BASE_URL, EVENT_ODDS_PATH, BOOKMAKERS, QUOTA_HEADERS,
CANONICAL_EVENTS_CACHE. _season_events itself internally uses
purchase_core_bettime_passes.py's own commence_to_eastern_date for Eastern
game-date windowing; that function is not separately imported here because
this script never needs Eastern-date windowing on its own (the 2024-25 leg's
pool is scoped by parquet event-id membership, already window-restricted by
construction; the 2025-26 leg's pool is scoped entirely by _season_events).

New dedicated append-only cache directory:
data/raw/betting_lines/passes/alt_ladder_pilot_202607/, record naming
altladder_event={event_id}_signature={signature}.json -- a shape that
cannot collide with probe_opening_markets.py's w1_event=... or
purchase_core_bettime_passes.py's core_event=... records.

Registered hard limits, enforced by this script and never raisable via a
flag: --max-credits may not exceed 2000; --credit-floor may not be set
below 10895. Both are checked against the LIVE x-requests-remaining header
after every call, never a locally-estimated figure.

Usage:
    python scripts/purchase_alt_ladder_pilot.py --leg alt_only_2024_25
    python scripts/purchase_alt_ladder_pilot.py --leg combined_2025_26
    python scripts/purchase_alt_ladder_pilot.py --leg alt_only_2024_25 \
        --execute --max-credits 1200 --credit-floor 10895
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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

DEFAULT_PASS_CACHE = Path("data/raw/betting_lines/passes/alt_ladder_pilot_202607")
DEFAULT_CORE_BETTIME_PARQUET = Path("data/processed/core_bettime_202607_snapshots.parquet")
DEFAULT_PROBE_DIR = Path("data/raw/betting_lines/probes/w1_market_coverage")
RUN_LOG_NAME = "run_log.jsonl"

RNG_SEED = 42
REGISTERED_MAX_CREDITS = 2000
REGISTERED_CREDIT_FLOOR = 10895

# Pre-registered, fixed leg definitions (19.3's table). Not derived from any
# cache/parquet at import time, so a stale/partial cache can't silently
# change what a leg is scoped to buy.
LEG_DEFS: dict[str, dict[str, Any]] = {
    "alt_only_2024_25": {
        "season": "2024-25",
        "markets": ("player_total_saves_alternate",),
        "max_credits_per_event": 10,
        "target_n": 120,
    },
    "combined_2025_26": {
        "season": "2025-26",
        "markets": ("player_total_saves", "player_total_saves_alternate"),
        "max_credits_per_event": 20,
        "target_n": 35,
    },
}
COMBINED_2025_26_WINDOW = ("2025-10-07", "2026-04-19")  # 18.4's window, restated as binding in 19.3

# 19.8's own verified pool-size inventory (1,043 / 1,224), reused here as a
# fail-loud sanity gate: if a freshly-built pool doesn't match, either this
# script's pool logic has a bug or the underlying data has drifted since the
# registration was filed -- either way that is a STOP-and-investigate, never
# a silent continue, per this document family's standing discipline.
EXPECTED_POOL_SIZE = {
    "alt_only_2024_25": 1043,
    "combined_2025_26": 1224,
}


# ---------------------------------------------------------------------------
# Already-probed event exclusion (19.3's "already-owned probe events" rule)
# ---------------------------------------------------------------------------

def load_probe_event_ids(probe_dir: Path) -> dict[str, set[str]]:
    """Read every w1_event=*.json record (ANY status, not just BetOnline-
    alt-saves-covered ones) from the W1 probe archive and return the set of
    ALL probed event ids per season. Used to exclude already-probed events
    from this pilot's NEW candidate pools (19.3) -- an event the W1 probe
    already spent credits requesting is excluded from re-purchase whether or
    not that particular event happened to return BetOnline alternate-saves
    coverage; the coverage-having subset (15 of the 24 probed events: 7 in
    2024-25, 8 in 2025-26) is a separate concept, folded into the analysis
    universe in Experiment 16's Phase 3, not the purchase script."""
    by_season: dict[str, set[str]] = defaultdict(set)
    for path in sorted(probe_dir.glob("w1_event=*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        event = rec.get("event", {})
        season = event.get("season")
        event_id = event.get("event_id")
        if season and event_id:
            by_season[season].add(event_id)
    return dict(by_season)


# ---------------------------------------------------------------------------
# Pool construction (19.3's registered pool definitions, per leg)
# ---------------------------------------------------------------------------

def build_pool_alt_only_2024_25(
    events_by_id: dict[str, dict[str, Any]],
    core_bettime_parquet_path: Path,
    probed_ids: set[str],
) -> list[dict[str, Any]]:
    """1,043-event pool: every event with a betonlineag/player_total_saves
    row in core_bettime_202607_snapshots.parquet, minus already-probed
    2024-25 events, sorted by (commence_time, event_id)."""
    core_df = pd.read_parquet(
        core_bettime_parquet_path, columns=["book_key", "market_key", "event_id"]
    )
    mask = (core_df["book_key"] == "betonlineag") & (core_df["market_key"] == "player_total_saves")
    candidate_ids = set(core_df.loc[mask, "event_id"].unique()) - probed_ids

    candidate_events = []
    missing_from_cache = []
    for event_id in candidate_ids:
        event = events_by_id.get(event_id)
        if event is None:
            missing_from_cache.append(event_id)
            continue
        candidate_events.append(event)

    if missing_from_cache:
        raise RuntimeError(
            f"{len(missing_from_cache)} BetOnline-standard-covered 2024-25 event id(s) are "
            f"missing from the events-list cache (cannot sort or anchor them): "
            f"{sorted(missing_from_cache)[:10]}"
        )

    return sorted(candidate_events, key=lambda event: (event["commence_time"], event["id"]))


def build_pool_combined_2025_26(
    events_by_id: dict[str, dict[str, Any]], probed_ids: set[str]
) -> list[dict[str, Any]]:
    """1,224-event pool: every in-window 2025-26 event (18.4's window),
    minus already-probed 2025-26 events. _season_events already sorts by
    (commence_time, event_id); filtering preserves that order."""
    start, end = COMBINED_2025_26_WINDOW
    window_events = _season_events(events_by_id, start, end)
    return [event for event in window_events if event["id"] not in probed_ids]


# ---------------------------------------------------------------------------
# Frozen plan artifact: pool + full permutation + initial target-N sample.
# Built once per leg, on first invocation (dry-run or execute); every later
# invocation loads the persisted file rather than recomputing anything, so
# the sample can never be redrawn (19.6 items 5/6).
# ---------------------------------------------------------------------------

def _plan_path(cache_dir: Path, leg_name: str) -> Path:
    return cache_dir / f"plan_{leg_name}.json"


def build_or_load_plan(
    leg_name: str,
    cache_dir: Path,
    events_cache_dir: Path,
    core_bettime_parquet_path: Path,
    probe_dir: Path,
) -> dict[str, Any]:
    plan_path = _plan_path(cache_dir, leg_name)
    if plan_path.exists():
        with open(plan_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    leg_def = LEG_DEFS[leg_name]
    events_by_id = load_cached_events(events_cache_dir)
    probed_ids = load_probe_event_ids(probe_dir).get(leg_def["season"], set())

    if leg_name == "alt_only_2024_25":
        pool_events = build_pool_alt_only_2024_25(events_by_id, core_bettime_parquet_path, probed_ids)
    elif leg_name == "combined_2025_26":
        pool_events = build_pool_combined_2025_26(events_by_id, probed_ids)
    else:
        raise ValueError(f"unknown leg: {leg_name!r}")

    pool_size = len(pool_events)
    expected_pool_size = EXPECTED_POOL_SIZE.get(leg_name)
    if expected_pool_size is not None and pool_size != expected_pool_size:
        raise RuntimeError(
            f"leg {leg_name!r}: freshly-built pool_size ({pool_size}) does not match 19.8's verified "
            f"pool size ({expected_pool_size}) -- STOP and investigate before sampling; either the pool "
            "logic here has drifted from the registration or the underlying source data has changed "
            "since 2026-07-16."
        )

    rng = np.random.default_rng(RNG_SEED)
    permutation = rng.permutation(pool_size)

    target_n = leg_def["target_n"]
    pool_records = [
        {
            "pool_index": i,
            "event_id": event["id"],
            "commence_time": event["commence_time"],
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
        }
        for i, event in enumerate(pool_events)
    ]
    sample_preview = []
    for rank in range(min(target_n, pool_size)):
        pool_index = int(permutation[rank])
        rec = pool_records[pool_index]
        sample_preview.append(
            {
                "rank": rank,
                "pool_index": pool_index,
                "event_id": rec["event_id"],
                "commence_time": rec["commence_time"],
                "bettime_ts": compute_bettime_ts(rec["commence_time"]),
                "home_team": rec["home_team"],
                "away_team": rec["away_team"],
            }
        )

    plan = {
        "leg_name": leg_name,
        "season": leg_def["season"],
        "markets": list(leg_def["markets"]),
        "max_credits_per_event": leg_def["max_credits_per_event"],
        "target_n": target_n,
        "pool_size": pool_size,
        "rng_seed": RNG_SEED,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "probed_excluded_event_ids": sorted(probed_ids),
        "n_probed_excluded": len(probed_ids),
        "pool": pool_records,
        "full_permutation": [int(x) for x in permutation],
        "sample": sample_preview,
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


def _validate_plan_matches_registration(leg_name: str, plan: dict[str, Any], plan_path: Path) -> None:
    """Defensive check: a persisted plan file must still match this
    script's own registered LEG_DEFS. Catches a stale/hand-edited plan file
    rather than silently trusting it -- never silently absorb a mismatch."""
    leg_def = LEG_DEFS[leg_name]
    mismatches = []
    if plan.get("leg_name") != leg_name:
        mismatches.append(f"leg_name: plan={plan.get('leg_name')!r} expected {leg_name!r}")
    if plan.get("season") != leg_def["season"]:
        mismatches.append(f"season: plan={plan.get('season')!r} expected {leg_def['season']!r}")
    if plan.get("markets") != list(leg_def["markets"]):
        mismatches.append(f"markets: plan={plan.get('markets')!r} expected {list(leg_def['markets'])!r}")
    if plan.get("max_credits_per_event") != leg_def["max_credits_per_event"]:
        mismatches.append(
            f"max_credits_per_event: plan={plan.get('max_credits_per_event')!r} "
            f"expected {leg_def['max_credits_per_event']!r}"
        )
    if plan.get("target_n") != leg_def["target_n"]:
        mismatches.append(f"target_n: plan={plan.get('target_n')!r} expected {leg_def['target_n']!r}")
    if plan.get("rng_seed") != RNG_SEED:
        mismatches.append(f"rng_seed: plan={plan.get('rng_seed')!r} expected {RNG_SEED!r}")
    if mismatches:
        raise RuntimeError(
            f"persisted plan {plan_path} no longer matches this script's registered LEG_DEFS -- "
            f"refusing to proceed: {mismatches}"
        )


def sample_from_plan(plan: dict[str, Any], sample_size: int) -> list[dict[str, Any]]:
    """The first sample_size pool events in the plan's FROZEN permutation
    order -- the single source of truth for 'which events are in this run's
    working sample,' derived only from the persisted pool + full_permutation,
    never recomputed or reordered."""
    permutation = plan["full_permutation"]
    pool = plan["pool"]
    if sample_size > len(permutation):
        raise ValueError(f"sample_size {sample_size} exceeds pool_size {len(permutation)}")
    out = []
    for rank in range(sample_size):
        pool_index = permutation[rank]
        rec = pool[pool_index]
        out.append(
            {
                "rank": rank,
                "pool_index": pool_index,
                "event_id": rec["event_id"],
                "season": plan["season"],
                "commence_time": rec["commence_time"],
                "bettime_ts": compute_bettime_ts(rec["commence_time"]),
                "home_team": rec.get("home_team"),
                "away_team": rec.get("away_team"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Request signature + append-only cache record naming
# ---------------------------------------------------------------------------

def request_signature(leg_name: str, event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    markets = LEG_DEFS[leg_name]["markets"]
    request = {
        "method": "GET",
        "path": EVENT_ODDS_PATH.format(event_id=event["event_id"]),
        "params": {
            "bookmakers": ",".join(BOOKMAKERS),
            "date": event["bettime_ts"],
            "includeMultipliers": "true",
            "markets": ",".join(markets),
        },
    }
    encoded = json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), request


def _record_path(cache_dir: Path, event_id: str, signature: str) -> Path:
    return cache_dir / f"altladder_event={event_id}_signature={signature}.json"


def classify_events(
    leg_name: str, sample_events: list[dict[str, Any]], cache_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the working sample into (cached, uncached) using on-disk
    signature records. Order is preserved (frozen permutation order)."""
    cached: list[dict[str, Any]] = []
    uncached: list[dict[str, Any]] = []
    for event in sample_events:
        signature, request = request_signature(leg_name, event)
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
    leg_name: str,
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

    max_credits_per_event = LEG_DEFS[leg_name]["max_credits_per_event"]

    billing: dict[str, Any] = {
        "leg_name": leg_name,
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
            headers={"Accept": "application/json", "User-Agent": "saves-model-v3-alt-ladder-pilot/1"},
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
            "leg_name": leg_name,
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


def append_run_log(cache_dir: Path, leg_name: str, cached_skips: int, billing: dict[str, Any]) -> None:
    entry = {
        "leg_name": leg_name,
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
    leg_name: str,
    plan: dict[str, Any],
    working_sample: list[dict[str, Any]],
    cached: list[dict[str, Any]],
    uncached: list[dict[str, Any]],
    limited: list[dict[str, Any]],
    limit: int | None,
    sample_size: int,
) -> None:
    leg_def = LEG_DEFS[leg_name]
    max_per_event = leg_def["max_credits_per_event"]
    print(f"\n=== Dry-run plan: {leg_name} ===")
    print(f"  season:            {leg_def['season']}")
    print(f"  markets:           {','.join(leg_def['markets'])}")
    print(f"  bookmakers:        {','.join(BOOKMAKERS)} (nine named books)")
    print(f"  max credits/event: {max_per_event}")
    print(f"  pool size (candidates, already-probed excluded): {plan['pool_size']}")
    print(f"  n already-probed excluded from pool:              {plan['n_probed_excluded']}")
    print(f"  registered target_n (new purchases):              {leg_def['target_n']}")
    print(f"  this run's working sample_size (permutation-order prefix): {sample_size}")
    if sample_size != leg_def["target_n"]:
        print(
            f"  *** sample_size ({sample_size}) != registered target_n ({leg_def['target_n']}) -- "
            "this must ONLY be an extension to replace confirmed non-200 events (19.6 item 10), "
            "never an opportunistic enlargement (19.6 item 5). ***"
        )
    print(f"\n  already-cached signature records in working sample: {len(cached)}")
    print(f"  uncached (not yet purchased) in working sample:     {len(uncached)}")
    print(f"  worst-case credits for ALL uncached in working sample: {len(uncached) * max_per_event}")
    if limit is not None:
        print(f"\n  --limit {limit}: this run would process the first {len(limited)} unfetched events")
        print(f"  worst-case credits for THIS RUN:                    {len(limited) * max_per_event}")
    print("\n  First 5 events in this leg's working sample (frozen permutation order):")
    for event in working_sample[:5]:
        print(
            f"    rank={event['rank']} pool_index={event['pool_index']} event={event['event_id']} "
            f"commence={event['commence_time']} bettime={event['bettime_ts']} "
            f"{event.get('away_team')} @ {event.get('home_team')}"
        )
    print("\nDry run only -- no network calls made. Pass --execute --max-credits N to actually fetch.")


def print_execute_summary(leg_name: str, cached_skips: int, billing: dict[str, Any]) -> None:
    print(f"\n=== Execute summary: {leg_name} ===")
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
        description="Dry-run-by-default alt-ladder pilot purchase for The Odds API. "
                    "See docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 19.3.",
    )
    parser.add_argument(
        "--leg",
        dest="leg_name",
        required=True,
        choices=sorted(LEG_DEFS),
        help="Which pre-registered leg to plan (and, with --execute, run).",
    )
    parser.add_argument("--execute", action="store_true", help="Make the historical API calls.")
    parser.add_argument(
        "--max-credits",
        type=int,
        default=None,
        help=f"Required with --execute. Hard cap on cumulative reserved (worst-case) credits this run. "
             f"May never exceed the registered pilot cap of {REGISTERED_MAX_CREDITS}.",
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
             "working sample's frozen permutation order. Omit to target every remaining uncached event "
             "in the working sample.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Advanced/extension use only (19.6 item 10): number of pool events (from the SAME frozen "
             "permutation) to include in the working sample. Must exceed the leg's registered target_n; "
             "used only to replace confirmed non-200 events with the next events in permutation order. "
             "Never use this to shrink the sample -- use --limit for that.",
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
        "--core-bettime-parquet",
        type=Path,
        default=DEFAULT_CORE_BETTIME_PARQUET,
        help=f"2024-25 anchor source for the alt-only leg's pool (default: {DEFAULT_CORE_BETTIME_PARQUET}).",
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=DEFAULT_PROBE_DIR,
        help=f"W1 probe archive, for already-probed-event exclusion (default: {DEFAULT_PROBE_DIR}).",
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
                f"error: --max-credits may not exceed the registered pilot cap of {REGISTERED_MAX_CREDITS} "
                "(19.6 item 1); a new registration is required to raise it."
            )
    if args.credit_floor < REGISTERED_CREDIT_FLOOR:
        raise SystemExit(
            f"error: --credit-floor may not be set below the registered floor of {REGISTERED_CREDIT_FLOOR} "
            "(19.6 item 1); a new registration is required to lower it."
        )
    if args.limit is not None and args.limit < 1:
        raise SystemExit("error: --limit must be >= 1")

    leg_def = LEG_DEFS[args.leg_name]
    target_n = leg_def["target_n"]
    if args.sample_size is not None and args.sample_size <= target_n:
        raise SystemExit(
            f"error: --sample-size ({args.sample_size}) must exceed the registered target_n "
            f"({target_n}) for leg {args.leg_name!r}; use --limit to process fewer than the full "
            "working sample this run. --sample-size exists only to extend a frozen sample forward "
            "along the same permutation to replace confirmed non-200 events (19.6 item 10)."
        )
    sample_size = args.sample_size if args.sample_size is not None else target_n

    plan = build_or_load_plan(
        args.leg_name, args.cache_dir, args.events_cache_dir, args.core_bettime_parquet, args.probe_dir
    )
    _validate_plan_matches_registration(args.leg_name, plan, _plan_path(args.cache_dir, args.leg_name))

    working_sample = sample_from_plan(plan, sample_size)
    cached, uncached = classify_events(args.leg_name, working_sample, args.cache_dir)
    limited = uncached if args.limit is None else uncached[: args.limit]

    if not args.execute:
        print_dry_run_report(args.leg_name, plan, working_sample, cached, uncached, limited, args.limit, sample_size)
        return 0

    max_credits_per_event = leg_def["max_credits_per_event"]
    worst_case = len(limited) * max_credits_per_event
    print(f"Executing leg={args.leg_name} candidates_this_run={len(limited)} "
          f"worst_case_credits={worst_case} max_credits={args.max_credits} credit_floor={args.credit_floor}")

    billing = execute(args.leg_name, limited, args.cache_dir, args.max_credits, args.credit_floor)
    append_run_log(args.cache_dir, args.leg_name, len(cached), billing)
    print_execute_summary(args.leg_name, len(cached), billing)
    return 0 if billing["aborted_reason"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
