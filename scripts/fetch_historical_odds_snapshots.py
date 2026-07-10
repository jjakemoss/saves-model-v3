"""
Cache-first historical fetch tool for The Odds API, implementing the
phased purchase plan in docs/OFFSEASON_OPTIMIZATION_PLAN.md section 3.15
("The Odds API historical acquisition plan"). Read that section before
changing anything here -- it is the contract this script implements.

DRY-RUN IS THE DEFAULT. This script never makes a network call unless
--execute is passed, and --execute additionally requires --max-credits.
Every plan is built entirely from src/betting/odds_archive.py's manifest
(scan of data/raw/betting_lines/cache/) plus the events-list responses
already in that cache -- never from data/betting.db (bet-time lines with
outcomes live there, but section 3.15 is explicit that game dates and
event ids for planning purposes come from cached events-list responses,
not the tracker).

Phases (gated -- do not start a phase before the prior one is reviewed):

    probe            events lists + event-odds sampling for 2023-11-15,
                     2024-02-10, 2025-12-10 (~200-500 credits)
    phase1-bettime   bet-time pass, test-fold games 2025-12-04..2026-04-16
                     (~900 events)
    phase1-closing   closing pass (at-commence), 2026-01-04..2026-04-13
                     (~630 events)
    phase2           events + bet-time + closing passes, 2023-24 regular
                     season 2023-10-10..2024-04-18 (~1,312 games)
    phase3           bulk h2h,totals snapshots at 22:30Z daily for the
                     2023-24, 2024-25, and 2025-26 regular seasons
                     (~558 game dates x 2 markets)

BET-TIME ANCHOR rule (per event): bettime_ts = min(22:30:00Z on the
event's US/Eastern game date, commence_time minus 30 minutes). 22:30Z
mirrors the live workflow's closing-fetch cron and the tracker's typical
evening betting pattern; the commence-minus-30-minutes fallback exists so
matinee games (puck drop well before 22:30Z) get a snapshot shortly before
puck drop instead of one taken hours after the game ended.

CLOSING rule (per event): requested_ts = the event's exact commence_time
(the API returns the closest snapshot at-or-before the request; snapshots
are 5-minutely -- this is exactly how the existing archive's closing lines
were built, see the archive's own snapshot-timing audit in section 3.15).

Phase 3 (bulk) uses a fixed 22:30Z daily anchor rather than the per-event
min() formula above: a single bulk call covers many games at different
commence times, so it cannot be tailored per game the way a per-event call
can.

Usage:
    python scripts/fetch_historical_odds_snapshots.py probe
    python scripts/fetch_historical_odds_snapshots.py phase1-bettime
    python scripts/fetch_historical_odds_snapshots.py phase1-closing
    python scripts/fetch_historical_odds_snapshots.py phase2
    python scripts/fetch_historical_odds_snapshots.py phase3
    python scripts/fetch_historical_odds_snapshots.py phase1-bettime --execute --max-credits 16000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, namedtuple
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from betting.odds_archive import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    SPORT_KEY,
    _normalize_field,
    estimate_bulk_odds_credits,
    estimate_event_odds_credits,
    estimate_events_list_credits,
    is_covered,
    save_response,
    scan_archive,
)

BASE_URL = "https://api.the-odds-api.com/v4/historical"
EVENTS_PATH = f"/sports/{SPORT_KEY}/events"
EVENT_ODDS_PATH_TMPL = f"/sports/{SPORT_KEY}/events/{{event_id}}/odds"
BULK_ODDS_PATH = f"/sports/{SPORT_KEY}/odds"

MARKET_SAVES = "player_total_saves"
MARKET_BULK = "h2h,totals"
MASKED_KEY = "***MASKED***"

EASTERN = ZoneInfo("America/New_York")

# Fixed pre-registered date ranges (section 3.15). These are not derived
# from the cache at runtime so a stale/partial cache can't silently change
# what a phase plans to buy.
PHASE1_BETTIME_RANGE = ("2025-12-04", "2026-04-16")
PHASE1_CLOSING_RANGE = ("2026-01-04", "2026-04-13")
PHASE2_RANGE = ("2023-10-10", "2024-04-18")
PHASE3_SEASON_WINDOWS = [
    # (label, start, end). 2023-24 matches PHASE2_RANGE (real NHL season
    # bounds). 2024-25 and 2025-26 bounds are grounded in the cache itself:
    # the existing archive's events files run 2024-10-04..2025-04-17 for
    # 2024-25 (confirmed by scanning the cache, matches the real NHL
    # 2024-25 regular season), and 2025-10-07 for the 2025-26 start
    # (matches the cache's held Oct 2025 events files); the 2025-26 end
    # date reuses PHASE1_BETTIME_RANGE's end (the test-fold's own season
    # boundary, per section 3.15).
    ("2023-24", "2023-10-10", "2024-04-18"),
    ("2024-25", "2024-10-04", "2025-04-17"),
    ("2025-26", "2025-10-07", "2026-04-16"),
]

# Probe sample dates (section 3.15 Phase 0).
PROBE_SAMPLE_DATES = ["2023-11-15", "2024-02-10", "2025-12-10"]

# A real, already-cached 2025-12 closing-line odds file, hardcoded so the
# probe can prove the skip logic works end to end without depending on
# what get scanned elsewhere in the run. Verified present in
# data/raw/betting_lines/cache/ on 2026-07-09:
#   odds_00e0f9a7c3c61e25df9ab047e44864e4_date=2025-12-08T00_10_00Z_
#   markets=player_total_saves_regions=us.json
# envelope timestamp=2025-12-08T00:05:37Z, data.commence_time=
# 2025-12-08T00:10:00Z. The archive is append-only, so this file will
# never move or disappear.
PROBE_PRECACHED_EVENT_ID = "00e0f9a7c3c61e25df9ab047e44864e4"
PROBE_PRECACHED_COMMENCE = "2025-12-08T00:10:00Z"

DEFAULT_GAMES_PER_DATE_ESTIMATE = 8

PlannedCall = namedtuple(
    "PlannedCall",
    ["kind", "pass_name", "status", "requested_ts", "event_id", "markets", "regions", "credits", "note"],
)


# ---------------------------------------------------------------------------
# Bet-time / closing timestamp rules
# ---------------------------------------------------------------------------

def commence_to_eastern_date(commence_time: str) -> str:
    """Convert a UTC commence_time (ISO8601 'Z' string) to the US/Eastern
    calendar date it falls on. Late UTC-evening or early-UTC-morning games
    can land on a different Eastern date than their UTC date."""
    dt = datetime.strptime(commence_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return dt.astimezone(EASTERN).date().isoformat()


def compute_bettime_ts(commence_time: str) -> str:
    """Bet-time anchor for a single event: min(22:30:00Z on the event's
    US/Eastern game date, commence_time minus 30 minutes). See module
    docstring for the rationale (matinee handling)."""
    commence_dt = datetime.strptime(commence_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    eastern_game_date = commence_dt.astimezone(EASTERN).date()
    anchor_2230z = datetime(
        eastern_game_date.year, eastern_game_date.month, eastern_game_date.day,
        22, 30, 0, tzinfo=timezone.utc,
    )
    thirty_before_commence = commence_dt - timedelta(minutes=30)
    bettime_dt = min(anchor_2230z, thirty_before_commence)
    return bettime_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_closing_ts(commence_time: str) -> str:
    """Closing-line requested timestamp: exactly the event's commence_time
    (the API resolves this to the closest snapshot at-or-before it)."""
    return commence_time


def daterange(start: str, end: str) -> list:
    """Inclusive list of ISO calendar date strings from start to end."""
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    out = []
    d = start_d
    while d <= end_d:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Cached events aggregation
# ---------------------------------------------------------------------------

def load_cached_events(cache_dir: Path):
    """
    Aggregate every cached events_date=*.json file's data into a single
    event-id -> event dict map, deduplicated (an event commonly appears in
    several daily snapshots since each events call returns a forward
    window, not just its own requested date), then group by each event's
    US/Eastern game date.

    Deliberately reads ONLY events_*.json files, not odds_*.json files --
    section 3.15 is explicit that game dates and event ids for planning
    come from events-list responses, so a date's coverage here reflects
    what an events-list call actually told us, not what odds calls we
    happen to also hold.

    Returns (events_by_id, events_by_date) where events_by_date maps an
    ISO date string to a list of event dicts (each with at least 'id' and
    'commence_time').
    """
    events_by_id = {}
    for path in sorted(Path(cache_dir).glob("events_date=*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                envelope = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        for ev in envelope.get("data") or []:
            eid = ev.get("id")
            if not eid or not ev.get("commence_time"):
                continue
            events_by_id[eid] = ev

    events_by_date = defaultdict(list)
    for ev in events_by_id.values():
        d = commence_to_eastern_date(ev["commence_time"])
        events_by_date[d].append(ev)
    return events_by_id, dict(events_by_date)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _count(value) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return len(str(value).split(","))


class RunPlanner:
    """Accumulates planned calls for a single script invocation, checking
    each against the on-disk manifest via is_covered() and additionally
    deduplicating identical requests planned twice within the SAME run
    (e.g. phase2's bet-time and closing passes both need the same date's
    events-list call -- only the first counts as a real credit spend)."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.calls = []
        self._seen = set()

    @staticmethod
    def _key(kind, scope, markets, regions, requested_ts):
        return (kind, scope, _normalize_field(markets), _normalize_field(regions), requested_ts)

    def add_events(self, requested_ts, pass_name) -> PlannedCall:
        key = self._key("events", SPORT_KEY, None, None, requested_ts)
        if key in self._seen:
            call = PlannedCall("events", pass_name, "dedup_in_run", requested_ts, None, None, None, 0,
                                "identical events call already planned earlier in this run")
        else:
            self._seen.add(key)
            covered = is_covered(self.manifest, "events", SPORT_KEY, None, None, requested_ts)
            status = "skipped_cached" if covered else "planned"
            credits = 0 if covered else estimate_events_list_credits(1)
            call = PlannedCall("events", pass_name, status, requested_ts, None, None, None, credits, None)
        self.calls.append(call)
        return call

    def add_event_odds(self, event_id, requested_ts, markets, regions, pass_name) -> PlannedCall:
        key = self._key("odds", event_id, markets, regions, requested_ts)
        if key in self._seen:
            call = PlannedCall("odds", pass_name, "dedup_in_run", requested_ts, event_id, markets, regions, 0,
                                "identical odds call already planned earlier in this run")
        else:
            self._seen.add(key)
            covered = is_covered(self.manifest, "odds", event_id, markets, regions, requested_ts)
            creds = estimate_event_odds_credits(1, _count(markets), _count(regions))
            status = "skipped_cached" if covered else "planned"
            call = PlannedCall("odds", pass_name, status, requested_ts, event_id, markets, regions,
                                0 if covered else creds, None)
        self.calls.append(call)
        return call

    def add_estimated_odds(self, requested_ts, markets, regions, pass_name, n_calls, note) -> PlannedCall:
        """Register a schedule-unknown estimate: we know we need odds
        calls around this date/timestamp but have no cached event ids to
        plan them concretely (the events file for that date is itself not
        yet cached). n_calls is the number of event-odds-call-equivalents
        being estimated (e.g. games_per_date_estimate, or 2 events x 2
        snapshot times for the probe)."""
        creds = estimate_event_odds_credits(n_calls, _count(markets), _count(regions))
        call = PlannedCall("odds", pass_name, "estimated", requested_ts, None, markets, regions, creds, note)
        self.calls.append(call)
        return call

    def add_bulk(self, requested_ts, markets, regions, pass_name) -> PlannedCall:
        key = self._key("bulk", SPORT_KEY, markets, regions, requested_ts)
        if key in self._seen:
            call = PlannedCall("bulk", pass_name, "dedup_in_run", requested_ts, None, markets, regions, 0,
                                "identical bulk call already planned earlier in this run")
        else:
            self._seen.add(key)
            covered = is_covered(self.manifest, "bulk", SPORT_KEY, markets, regions, requested_ts)
            creds = estimate_bulk_odds_credits(_count(markets), _count(regions))
            status = "skipped_cached" if covered else "planned"
            call = PlannedCall("bulk", pass_name, status, requested_ts, None, markets, regions,
                                0 if covered else creds, None)
        self.calls.append(call)
        return call


def plan_game_date_pass(planner, events_by_date, date_list, anchor_fn, markets, regions, pass_name,
                         games_per_date_estimate=DEFAULT_GAMES_PER_DATE_ESTIMATE):
    """Plan one pass (bet-time or closing) of event-odds calls over a list
    of US/Eastern game dates, planning the underlying events-list call for
    each date too (deduplicated across passes within the same run by
    RunPlanner)."""
    for d in date_list:
        events_requested_ts = f"{d}T18:00:00Z"
        events_call = planner.add_events(events_requested_ts, pass_name)

        if events_call.status == "dedup_in_run":
            # A different pass in this same run already planned/found this
            # exact events call; re-check the on-disk manifest directly to
            # know whether the file genuinely exists (needed to tell a
            # confirmed off day apart from a truly-unknown date below).
            have_events_file = is_covered(planner.manifest, "events", SPORT_KEY, None, None, events_requested_ts)
        else:
            have_events_file = events_call.status == "skipped_cached"

        events_for_date = events_by_date.get(d)
        if not events_for_date:
            if have_events_file:
                # We already hold an events-list snapshot covering this
                # UTC date and it maps zero games to this Eastern date --
                # a confirmed off day (or its games were folded into a
                # neighboring day's snapshot window). No odds calls.
                continue
            planner.add_estimated_odds(
                events_requested_ts, markets, regions, pass_name, games_per_date_estimate,
                note=f"no cached events for {d}; estimated {games_per_date_estimate} games/date",
            )
            continue

        for ev in sorted(events_for_date, key=lambda e: e.get("commence_time", "")):
            requested_ts = anchor_fn(ev["commence_time"])
            planner.add_event_odds(ev["id"], requested_ts, markets, regions, pass_name)


def plan_probe(planner, events_by_date):
    """Phase 0 (section 3.15): events lists for 3 sample dates, event-odds
    for the first 2 events of each date at bet-time and commence, plus one
    deliberately-already-cached call to prove the skip logic."""
    for d in PROBE_SAMPLE_DATES:
        events_requested_ts = f"{d}T18:00:00Z"
        planner.add_events(events_requested_ts, "probe")

        candidates = sorted(events_by_date.get(d, []), key=lambda e: e.get("commence_time", ""))[:2]
        if candidates:
            for ev in candidates:
                planner.add_event_odds(ev["id"], compute_bettime_ts(ev["commence_time"]),
                                        MARKET_SAVES, "us", "probe")
                planner.add_event_odds(ev["id"], compute_closing_ts(ev["commence_time"]),
                                        MARKET_SAVES, "us", "probe")
        else:
            # No cached events for this sample date -- we don't have real
            # event ids yet. Estimate 2 events x 2 snapshot times (4
            # calls), clearly marked; a real run resolves this after the
            # events call above is actually fetched.
            planner.add_estimated_odds(
                events_requested_ts, MARKET_SAVES, "us", "probe", n_calls=4,
                note=f"no cached events for {d}; estimated 2 events x 2 snapshot times (bettime+closing)",
            )

    # Deliberately-cached check: an at-commence odds call for a real
    # 2025-12 event we already hold. Must resolve to skipped_cached.
    planner.add_event_odds(
        PROBE_PRECACHED_EVENT_ID, compute_closing_ts(PROBE_PRECACHED_COMMENCE),
        MARKET_SAVES, "us", "probe-precached-check",
    )


def plan_phase1_bettime(planner, events_by_date, regions, games_per_date_estimate):
    dates = daterange(*PHASE1_BETTIME_RANGE)
    plan_game_date_pass(planner, events_by_date, dates, compute_bettime_ts, MARKET_SAVES, regions,
                         "phase1-bettime", games_per_date_estimate)


def plan_phase1_closing(planner, events_by_date, regions, games_per_date_estimate):
    dates = daterange(*PHASE1_CLOSING_RANGE)
    plan_game_date_pass(planner, events_by_date, dates, compute_closing_ts, MARKET_SAVES, regions,
                         "phase1-closing", games_per_date_estimate)


def plan_phase2(planner, events_by_date, regions, games_per_date_estimate):
    dates = daterange(*PHASE2_RANGE)
    plan_game_date_pass(planner, events_by_date, dates, compute_bettime_ts, MARKET_SAVES, regions,
                         "phase2-bettime", games_per_date_estimate)
    plan_game_date_pass(planner, events_by_date, dates, compute_closing_ts, MARKET_SAVES, regions,
                         "phase2-closing", games_per_date_estimate)


def plan_phase3(planner, regions):
    for season_label, start, end in PHASE3_SEASON_WINDOWS:
        for d in daterange(start, end):
            requested_ts = f"{d}T22:30:00Z"
            planner.add_bulk(requested_ts, MARKET_BULK, regions, f"phase3-{season_label}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _join(value) -> str:
    if isinstance(value, (list, tuple, set)):
        return ",".join(value)
    return str(value)


def describe_call(call: PlannedCall) -> str:
    """Human-readable planned-call description with the API key masked --
    never the real key, even in dry-run examples."""
    if call.kind == "events":
        url = f"{BASE_URL}{EVENTS_PATH}"
        params = {"date": call.requested_ts, "apiKey": MASKED_KEY}
    elif call.kind == "odds":
        url = f"{BASE_URL}{EVENT_ODDS_PATH_TMPL.format(event_id=call.event_id)}"
        params = {"date": call.requested_ts, "regions": _join(call.regions),
                   "markets": _join(call.markets), "apiKey": MASKED_KEY}
    else:  # bulk
        url = f"{BASE_URL}{BULK_ODDS_PATH}"
        params = {"date": call.requested_ts, "regions": _join(call.regions),
                   "markets": _join(call.markets), "apiKey": MASKED_KEY}
    text = f"GET {url} params={params}"
    if call.note:
        text += f"  note={call.note}"
    return text


def print_plan_report(phase: str, calls, examples: int = 8):
    print(f"\n=== Plan report: {phase} ===")

    by_group = defaultdict(lambda: {"count": 0, "credits": 0})
    for c in calls:
        g = by_group[(c.kind, c.pass_name, c.status)]
        g["count"] += 1
        g["credits"] += c.credits

    print(f"\n  {'kind':6s} {'pass':22s} {'status':16s} {'count':>7s} {'credits':>9s}")
    for (kind, pass_name, status), agg in sorted(by_group.items()):
        print(f"  {kind:6s} {pass_name:22s} {status:16s} {agg['count']:7d} {agg['credits']:9d}")

    total_planned = sum(1 for c in calls if c.status == "planned")
    total_estimated = sum(1 for c in calls if c.status == "estimated")
    total_skipped = sum(1 for c in calls if c.status == "skipped_cached")
    total_dedup = sum(1 for c in calls if c.status == "dedup_in_run")
    total_credits = sum(c.credits for c in calls if c.status in ("planned", "estimated"))

    print(f"\n  TOTAL planned (real, uncached):            {total_planned}")
    print(f"  TOTAL estimated (schedule unknown, heuristic): {total_estimated}")
    print(f"  TOTAL skipped as already cached:            {total_skipped}")
    print(f"  TOTAL deduplicated within this run:         {total_dedup}")
    print(f"  TOTAL estimated credits to spend:           {total_credits}")

    print(f"\n  Example planned/estimated calls (key masked):")
    shown = 0
    for c in calls:
        if c.status not in ("planned", "estimated"):
            continue
        print(f"    [{c.status}] {describe_call(c)}")
        shown += 1
        if shown >= examples:
            break
    if shown == 0:
        print("    (none -- everything already cached)")

    return {
        "planned": total_planned,
        "estimated": total_estimated,
        "skipped_cached": total_skipped,
        "dedup_in_run": total_dedup,
        "credits": total_credits,
    }


# ---------------------------------------------------------------------------
# Execute (spends credits -- never invoked by dry-run, requires --execute)
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """Load the API key from the environment or .env (API_KEY=...). Never
    printed or logged anywhere -- callers must mask it in any output."""
    api_key = os.environ.get("API_KEY") or os.environ.get("THE_ODDS_API_KEY")
    if api_key:
        return api_key
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("API_KEY=") or line.startswith("THE_ODDS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def _build_request(call: PlannedCall, api_key: str):
    if call.kind == "events":
        url = f"{BASE_URL}{EVENTS_PATH}"
        params = {"date": call.requested_ts, "apiKey": api_key}
    elif call.kind == "odds":
        url = f"{BASE_URL}{EVENT_ODDS_PATH_TMPL.format(event_id=call.event_id)}"
        params = {"date": call.requested_ts, "regions": _join(call.regions),
                   "markets": _join(call.markets), "apiKey": api_key}
    elif call.kind == "bulk":
        url = f"{BASE_URL}{BULK_ODDS_PATH}"
        params = {"date": call.requested_ts, "regions": _join(call.regions),
                   "markets": _join(call.markets), "apiKey": api_key}
    else:
        raise ValueError(call.kind)
    return url, params


def _request_with_retry(session, url, params, timeout=20):
    """One retry with exponential backoff on 429/5xx, per section 3.15
    rule 3. Never logs params (which would include the real key).

    Returns the final requests.Response whatever its status code (the
    caller inspects it -- in particular, 401/403 means a bad/expired key
    or plan-tier problem and must abort the whole run, so those are
    returned immediately without a retry: they will not fix themselves
    between attempts). Returns None only if the request itself raised
    even after the retry."""
    import requests

    for attempt in range(2):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.exceptions.RequestException as e:
            if attempt == 0:
                time.sleep(2)
                continue
            print(f"[ERROR] request failed after retry: {e}")
            return None

        if response.status_code == 200:
            return response
        if response.status_code in (401, 403):
            return response
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 0:
                backoff = 2 ** (attempt + 2)
                print(f"[WARNING] HTTP {response.status_code}, retrying in {backoff}s")
                time.sleep(backoff)
                continue
        print(f"[ERROR] HTTP {response.status_code}")
        return response
    return None


def execute_plan(calls, cache_dir: Path, max_credits: int, credit_floor: int, phase: str, regions: str,
                 sleep_seconds: float = 0.25, max_consecutive_failures: int = 5):
    """Actually spend credits: fetch every 'planned' call, save the
    response verbatim via save_response(), abort if the plan would exceed
    max_credits or if x-requests-remaining ever falls below credit_floor.
    'estimated' entries (schedule-unknown dates) cannot be executed
    directly -- fetch their events call first, then re-run to resolve
    them into real per-event calls.

    Pacing and fast-abort hardening:
    - sleep_seconds is slept between successive requests (not before the
      first, not after the last) so a long phase run doesn't hammer the
      API into a 429 storm that one-retry-per-call cannot absorb.
    - a 401/403 response aborts the entire run immediately (bad/expired
      key or plan-tier problem -- every subsequent call would fail the
      same way).
    - max_consecutive_failures consecutive failures (request exception,
      non-200 after retry, or JSON decode failure) abort the run; the
      counter resets on any successful call.
    Every abort path still writes the fetch_log.jsonl entry with the
    counts accumulated so far, plus an aborted_reason field (null on
    clean completion)."""
    import requests

    api_key = _load_api_key()
    if not api_key:
        print("[ERROR] No API key found (set API_KEY in .env or the environment). Aborting.")
        sys.exit(1)

    to_run = [c for c in calls if c.status == "planned"]
    estimated = [c for c in calls if c.status == "estimated"]
    if estimated:
        print(f"[WARNING] {len(estimated)} estimated (schedule-unknown) entries will NOT be executed.")
        print("          Their events-list call is included in this same run; re-run this phase")
        print("          afterward to resolve them into real per-event odds calls.")

    planned_credits = sum(c.credits for c in to_run)
    if planned_credits > max_credits:
        print(f"[ABORT] Planned credits ({planned_credits}) exceed --max-credits ({max_credits}). "
              f"Raise --max-credits or split the run. No calls made.")
        sys.exit(1)

    session = requests.Session()
    run_started = datetime.now(timezone.utc).isoformat()
    fetched = 0
    failed = 0
    skipped_cached = sum(1 for c in calls if c.status == "skipped_cached")
    credits_spent_estimated = 0
    credits_remaining_reported = None
    aborted_reason = None
    consecutive_failures = 0

    def _register_failure(what):
        """Count one failure; returns the abort reason string when the
        consecutive-failure limit is hit, else None."""
        nonlocal failed, consecutive_failures
        failed += 1
        consecutive_failures += 1
        if consecutive_failures >= max_consecutive_failures:
            return (f"{consecutive_failures} consecutive failures "
                    f"(limit {max_consecutive_failures}); last failure: {what}")
        return None

    for i, call in enumerate(to_run):
        if credits_remaining_reported is not None and credits_remaining_reported < credit_floor:
            aborted_reason = (f"credits remaining ({credits_remaining_reported}) "
                              f"below floor ({credit_floor})")
            print(f"[ABORT] {aborted_reason}.")
            break

        if i > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

        url, params = _build_request(call, api_key)
        response = _request_with_retry(session, url, params)

        if response is not None and response.status_code in (401, 403):
            aborted_reason = (f"HTTP {response.status_code} "
                              f"(bad/expired API key or plan tier problem)")
            failed += 1
            print(f"[ABORT] {aborted_reason}. Aborting entire run immediately.")
            break

        if response is None or response.status_code != 200:
            what = "request exception" if response is None else f"HTTP {response.status_code} after retry"
            aborted_reason = _register_failure(what)
            if aborted_reason:
                print(f"[ABORT] {aborted_reason}.")
                break
            continue

        remaining = response.headers.get("x-requests-remaining")
        if remaining is not None:
            try:
                credits_remaining_reported = int(remaining)
            except ValueError:
                pass

        try:
            body = response.json()
        except ValueError:
            aborted_reason = _register_failure("JSON decode failure")
            if aborted_reason:
                print(f"[ABORT] {aborted_reason}.")
                break
            continue

        consecutive_failures = 0

        envelope = {
            "timestamp": body.get("timestamp"),
            "previous_timestamp": body.get("previous_timestamp"),
            "next_timestamp": body.get("next_timestamp"),
            "data": body.get("data"),
        }
        try:
            save_response(cache_dir, call.kind, envelope, call.requested_ts,
                          event_id=call.event_id, markets=call.markets, regions=call.regions)
            fetched += 1
            credits_spent_estimated += call.credits
        except FileExistsError as e:
            print(f"[WARNING] {e}")

        if credits_remaining_reported is not None and credits_remaining_reported < credit_floor:
            aborted_reason = (f"credits remaining ({credits_remaining_reported}) "
                              f"fell below floor ({credit_floor}) after this call")
            print(f"[ABORT] {aborted_reason}.")
            break

    run_ended = datetime.now(timezone.utc).isoformat()
    log_entry = {
        "run_started": run_started,
        "run_ended": run_ended,
        "phase": phase,
        "regions": regions,
        "planned": len(to_run),
        "skipped_cached": skipped_cached,
        "fetched": fetched,
        "failed": failed,
        "credits_spent_estimated": credits_spent_estimated,
        "credits_remaining_reported": credits_remaining_reported,
        "aborted_reason": aborted_reason,
    }
    log_path = Path(cache_dir) / "fetch_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry) + "\n")

    status_word = "aborted" if aborted_reason else "complete"
    print(f"\nRun {status_word}. fetched={fetched} failed={failed} skipped_cached={skipped_cached} "
          f"credits_spent~={credits_spent_estimated} credits_remaining={credits_remaining_reported}")
    if aborted_reason:
        print(f"Abort reason: {aborted_reason}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache-first historical fetch tool for The Odds API (dry-run by default). "
                    "See docs/OFFSEASON_OPTIMIZATION_PLAN.md section 3.15.",
    )
    parser.add_argument(
        "phase",
        choices=["probe", "phase1-bettime", "phase1-closing", "phase2", "phase3"],
        help="Which pre-registered phase to plan (and, with --execute, run).",
    )
    parser.add_argument("--execute", action="store_true",
                        help="Actually spend credits. Default is dry-run (plan + report only).")
    parser.add_argument("--max-credits", type=int, default=None,
                        help="Hard cap on credits to spend this run. Required with --execute.")
    parser.add_argument("--credit-floor", type=int, default=1000,
                        help="Abort if x-requests-remaining falls below this after any call (default 1000).")
    parser.add_argument("--sleep-seconds", type=float, default=0.25,
                        help="Pause between successive requests during --execute (default 0.25). "
                             "Not used in dry-run, which makes no calls.")
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="Abort the run after this many consecutive failed calls during "
                             "--execute; the counter resets on any success (default 5).")
    parser.add_argument("--regions", default="us",
                        help="Comma-separated regions param for event-odds/bulk calls (default 'us').")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                        help=f"Archive directory (default {DEFAULT_CACHE_DIR}).")
    parser.add_argument("--games-per-date-estimate", type=int, default=DEFAULT_GAMES_PER_DATE_ESTIMATE,
                        help="Used only for dry-run credit estimates on dates lacking cached events "
                             f"(default {DEFAULT_GAMES_PER_DATE_ESTIMATE}).")
    parser.add_argument("--examples", type=int, default=8,
                        help="Number of example planned/estimated calls to print (default 8).")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.execute and args.max_credits is None:
        parser.error("--execute requires --max-credits")

    cache_dir = Path(args.cache_dir)
    manifest = scan_archive(cache_dir)
    unparsable = manifest.attrs.get("unparsable", [])
    print(f"Scanned {len(manifest)} cache records from {cache_dir} ({len(unparsable)} unparsable)")
    if unparsable:
        for name in unparsable[:10]:
            print(f"  [unparsable] {name}")
        if len(unparsable) > 10:
            print(f"  ... and {len(unparsable) - 10} more")

    events_by_id, events_by_date = load_cached_events(cache_dir)
    print(f"Aggregated {len(events_by_id)} unique cached events across "
          f"{len(list(Path(cache_dir).glob('events_date=*.json')))} events-list files "
          f"({len(events_by_date)} distinct US/Eastern game dates covered)")

    planner = RunPlanner(manifest)

    if args.phase == "probe":
        plan_probe(planner, events_by_date)
    elif args.phase == "phase1-bettime":
        plan_phase1_bettime(planner, events_by_date, args.regions, args.games_per_date_estimate)
    elif args.phase == "phase1-closing":
        plan_phase1_closing(planner, events_by_date, args.regions, args.games_per_date_estimate)
    elif args.phase == "phase2":
        plan_phase2(planner, events_by_date, args.regions, args.games_per_date_estimate)
    elif args.phase == "phase3":
        plan_phase3(planner, args.regions)

    print_plan_report(args.phase, planner.calls, examples=args.examples)

    if not args.execute:
        print("\nDry run only -- no network calls made. Pass --execute --max-credits N to actually fetch.")
        return

    execute_plan(planner.calls, cache_dir, args.max_credits, args.credit_floor, args.phase, args.regions,
                 sleep_seconds=args.sleep_seconds, max_consecutive_failures=args.max_consecutive_failures)


if __name__ == "__main__":
    main()
