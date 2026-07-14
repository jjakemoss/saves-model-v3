#!/usr/bin/env python3
"""Bounded W1 historical market-coverage probe for The Odds API.

This is deliberately separate from the canonical archive fetcher.  It plans
24 deterministic games (eight per cached regular season), then, only with
``--execute --max-credits N``, asks the historical event-odds endpoint for
the W1 market set at the established bet-time anchor.  The default is a
read-only dry run and makes no network calls or writes.

Responses are stored only under ``data/raw/betting_lines/probes/``.  Each
cache record includes a non-secret request signature, the unmodified raw
response body, and the three quota headers needed to verify historical
billing.  A previously recorded signature is never requested again.
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
DEFAULT_PROBE_CACHE = Path("data/raw/betting_lines/probes/w1_market_coverage")

GAMES_PER_SEASON = 8
MAX_EXECUTE_CREDITS = 1500
ESTIMATED_CREDITS_PER_EVENT = 40  # 4 markets * 10 credits; verified by headers.

MARKETS = (
    "player_total_saves",
    "player_total_saves_alternate",
    "player_shots_on_goal",
    "player_shots_on_goal_alternate",
)
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

# These are copied from fetch_historical_odds_snapshots.py rather than imported
# because importing betting.odds_archive executes betting/__init__.py and pulls
# the full model runtime into a standard-library acquisition probe.
PHASE3_SEASON_WINDOWS = (
    ("2023-24", "2023-10-10", "2024-04-18"),
    ("2024-25", "2024-10-04", "2025-04-17"),
    ("2025-26", "2025-10-07", "2026-04-16"),
)


def commence_to_eastern_date(commence_time: str) -> str:
    dt = _parse_utc(commence_time)
    return dt.astimezone(EASTERN).date().isoformat()


def compute_bettime_ts(commence_time: str) -> str:
    """Existing archive convention: min(22:30Z game date, start minus 30m)."""
    commence_dt = _parse_utc(commence_time)
    game_date = commence_dt.astimezone(EASTERN).date()
    anchor = datetime(game_date.year, game_date.month, game_date.day, 22, 30, tzinfo=timezone.utc)
    return min(anchor, commence_dt - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_cached_events(cache_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
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
    events_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events_by_id.values():
        events_by_date[commence_to_eastern_date(event["commence_time"])].append(event)
    return events_by_id, dict(events_by_date)


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _season_events(events_by_id: dict[str, dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    """Return one sorted, regular-season-only event row per cached event id."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    selected = []
    for event in events_by_id.values():
        commence = event.get("commence_time")
        event_id = event.get("id")
        if not commence or not event_id:
            continue
        # Use the same Eastern game-date convention as load_cached_events,
        # including late west-coast games that are after midnight in UTC.
        event_date = date.fromisoformat(commence_to_eastern_date(commence))
        if start_date <= event_date <= end_date:
            selected.append(event)
    return sorted(selected, key=lambda event: (event["commence_time"], event["id"]))


def _evenly_spaced(events: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Pick deterministic, unique positions spanning a sorted event list."""
    if len(events) < count:
        raise ValueError(f"need {count} cached events, found {len(events)}")
    if count == 1:
        return [events[0]]
    indexes = [round(index * (len(events) - 1) / (count - 1)) for index in range(count)]
    if len(set(indexes)) != count:
        raise ValueError("evenly spaced selection unexpectedly produced duplicate indexes")
    return [events[index] for index in indexes]


def select_sample(events_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the fixed 8-per-season sample from the existing cached events."""
    sample: list[dict[str, Any]] = []
    for season, start, end in PHASE3_SEASON_WINDOWS:
        for event in _evenly_spaced(_season_events(events_by_id, start, end), GAMES_PER_SEASON):
            sample.append(
                {
                    "season": season,
                    "event_id": event["id"],
                    "commence_time": event["commence_time"],
                    "bettime_ts": compute_bettime_ts(event["commence_time"]),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                }
            )
    return sample


def request_signature(sample_event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a stable non-secret signature and its canonical request fields."""
    request = {
        "method": "GET",
        "path": EVENT_ODDS_PATH.format(event_id=sample_event["event_id"]),
        "params": {
            "bookmakers": ",".join(BOOKMAKERS),
            "date": sample_event["bettime_ts"],
            "includeMultipliers": "true",
            "markets": ",".join(MARKETS),
        },
    }
    encoded = json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), request


def _record_path(cache_dir: Path, event_id: str, signature: str) -> Path:
    # This shape intentionally cannot match odds_archive.py's canonical names.
    return cache_dir / f"w1_event={event_id}_signature={signature}.json"


def _atomic_json_create(path: Path, payload: dict[str, Any]) -> None:
    """Append-only atomic write; a collision never replaces prior evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite probe cache record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite probe cache record: {path}")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _load_cached_record(path: Path, signature: str) -> dict[str, Any] | None:
    """Treat any complete matching response as spent, including non-200 replies."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return record if record.get("signature") == signature else None


def _quota_headers(headers: Any) -> dict[str, str | None]:
    return {name: headers.get(name) for name in QUOTA_HEADERS}


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _response_data(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("status_code") != 200:
        return None
    try:
        parsed = json.loads(record.get("raw_body", ""))
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _market_lines(market: dict[str, Any]) -> tuple[set[tuple[str, Any]], set[tuple[str, Any]], int]:
    """Return player-line units, complete O/U units, and multiplier count."""
    sides: dict[tuple[str, Any], set[str]] = defaultdict(set)
    multiplier_outcomes = 0
    for outcome in market.get("outcomes") or []:
        player = outcome.get("description")
        side = str(outcome.get("name") or "").lower()
        if player:
            sides[(str(player), outcome.get("point"))].add(side)
        if outcome.get("multiplier") is not None:
            multiplier_outcomes += 1
    lines = set(sides)
    both_sides = {line for line, values in sides.items() if {"over", "under"}.issubset(values)}
    return lines, both_sides, multiplier_outcomes


def summarize_gates(sample: list[dict[str, Any]], records: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Produce JSON-safe coverage gates from cached/executed probe responses."""
    expected_events = {season: GAMES_PER_SEASON for season, _, _ in PHASE3_SEASON_WINDOWS}
    event_coverage: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    line_sets: dict[tuple[str, str, str], set[tuple[str, str, Any]]] = defaultdict(set)
    both_side_sets: dict[tuple[str, str, str], set[tuple[str, str, Any]]] = defaultdict(set)
    player_names: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    multiplier_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    measured_events: dict[str, int] = defaultdict(int)
    usable_sog_books: dict[tuple[str, str], set[str]] = defaultdict(set)

    for event in sample:
        record = records.get(event["event_id"])
        payload = _response_data(record) if record else None
        if not payload:
            continue
        measured_events[event["season"]] += 1
        data = payload.get("data") or {}
        for bookmaker in data.get("bookmakers") or []:
            book_key = bookmaker.get("key")
            if book_key not in BOOKMAKERS:
                continue
            for market in bookmaker.get("markets") or []:
                market_key = market.get("key")
                if market_key not in MARKETS:
                    continue
                key = (event["season"], book_key, market_key)
                event_coverage[key].add(event["event_id"])
                lines, both_sides, multiplier_count = _market_lines(market)
                line_sets[key].update((event["event_id"], player, point) for player, point in lines)
                both_side_sets[key].update(
                    (event["event_id"], player, point) for player, point in both_sides
                )
                player_names[key].update(player for player, _ in lines)
                multiplier_counts[key] += multiplier_count
                if market_key == "player_shots_on_goal" and both_sides:
                    usable_sog_books[(event["season"], event["event_id"])].add(book_key)

    coverage = []
    for season, _, _ in PHASE3_SEASON_WINDOWS:
        for book in BOOKMAKERS:
            for market in MARKETS:
                key = (season, book, market)
                lines = line_sets[key]
                both_sides = both_side_sets[key]
                coverage.append(
                    {
                        "season": season,
                        "bookmaker": book,
                        "market": market,
                        "sampled_events": expected_events[season],
                        "measured_events": measured_events[season],
                        "events_with_market": len(event_coverage[key]),
                        "unique_player_names": len(player_names[key]),
                        "listed_player_event_point_units": len(lines),
                        "both_side_complete_player_event_point_units": len(both_sides),
                        "both_side_completeness": (
                            round(len(both_sides) / len(lines), 4) if lines else None
                        ),
                        "multiplier_outcomes": multiplier_counts[key],
                    }
                )

    sog_event_gate = []
    for season, _, _ in PHASE3_SEASON_WINDOWS:
        events_with_multiple_books = sum(
            len(usable_sog_books[(season, event["event_id"])]) >= 2
            for event in sample
            if event["season"] == season
        )
        denominator = measured_events[season]
        sog_event_gate.append(
            {
                "season": season,
                "measured_events": denominator,
                "events_with_at_least_two_usable_sog_books": events_with_multiple_books,
                "rate": round(events_with_multiple_books / denominator, 4) if denominator else None,
                "usable_book_definition": (
                    "player_shots_on_goal has at least one player-event-point unit "
                    "with both OVER and UNDER"
                ),
                "passes_70_percent": (
                    events_with_multiple_books / denominator >= 0.70 if denominator else None
                ),
            }
        )

    return {
        "coverage_by_season_book_market": coverage,
        "sog_event_gate": sog_event_gate,
        "sog_gate": [
            row for row in coverage if row["market"] in {"player_shots_on_goal", "player_shots_on_goal_alternate"}
        ],
        "alternate_gate": [row for row in coverage if row["market"].endswith("_alternate")],
        "multipliers_present": sum(row["multiplier_outcomes"] for row in coverage) > 0,
    }


def _load_api_key() -> str:
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


def _run_manifest(
    sample: list[dict[str, Any]],
    records: dict[str, dict[str, Any] | None],
    cache_status: dict[str, str],
    max_credits: int | None,
    execution: dict[str, Any],
) -> dict[str, Any]:
    planned_events = sum(1 for status in cache_status.values() if status == "planned")
    cached_events = sum(1 for status in cache_status.values() if status == "cached")
    fetched_events = sum(1 for status in cache_status.values() if status == "fetched")
    return {
        "probe": "w1_market_coverage",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": not execution["executed"],
        "request": {
            "endpoint": "historical_event_odds",
            "markets": list(MARKETS),
            "bookmakers": list(BOOKMAKERS),
            "includeMultipliers": True,
            "anchor": "compute_bettime_ts",
        },
        "sample": sample,
        "planning": {
            "planned_events": planned_events,
            "cached_events": cached_events,
            "fetched_events": fetched_events,
            "estimated_credits_per_event": ESTIMATED_CREDITS_PER_EVENT,
            "estimated_max_credits": (
                planned_events + fetched_events
            ) * ESTIMATED_CREDITS_PER_EVENT,
            "max_credits": max_credits,
        },
        "billing": execution,
        "gates": summarize_gates(sample, records),
    }


def execute(sample: list[dict[str, Any]], cache_dir: Path, max_credits: int) -> tuple[dict[str, dict[str, Any] | None], dict[str, str], dict[str, Any]]:
    """Execute uncached calls once each; ambiguous failures stop without retrying."""
    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError("no API key found (set API_KEY or THE_ODDS_API_KEY)")

    records: dict[str, dict[str, Any] | None] = {}
    cache_status: dict[str, str] = {}
    candidates = []
    for event in sample:
        signature, request = request_signature(event)
        path = _record_path(cache_dir, event["event_id"], signature)
        cached = _load_cached_record(path, signature)
        if cached is not None:
            records[event["event_id"]] = cached
            cache_status[event["event_id"]] = "cached"
        else:
            candidates.append((event, signature, request, path))
            cache_status[event["event_id"]] = "planned"

    estimated = len(candidates) * ESTIMATED_CREDITS_PER_EVENT
    if estimated > max_credits:
        raise RuntimeError(
            f"uncached probe estimate ({estimated}) exceeds --max-credits ({max_credits}); no calls made"
        )

    billing: dict[str, Any] = {
        "executed": True,
        "calls_attempted": 0,
        "calls_completed": 0,
        "cached_skips": len(sample) - len(candidates),
        "cumulative_conservative_credits": 0,
        "cumulative_header_last_credits": 0,
        "header_last_available_for_calls": 0,
        "latest_headers": {header: None for header in QUOTA_HEADERS},
        "aborted_reason": None,
    }
    for event, signature, request, path in candidates:
        # Reserve the worst known charge before sending; once sent, it may bill
        # even if the connection dies before a response reaches this process.
        if billing["cumulative_conservative_credits"] + ESTIMATED_CREDITS_PER_EVENT > max_credits:
            billing["aborted_reason"] = "cumulative conservative credit cap would be exceeded"
            break
        billing["calls_attempted"] += 1
        billing["cumulative_conservative_credits"] += ESTIMATED_CREDITS_PER_EVENT
        url = f"{BASE_URL}{request['path']}"
        params = dict(request["params"])
        params["apiKey"] = api_key
        request_url = f"{url}?{urllib.parse.urlencode(params)}"
        http_request = urllib.request.Request(
            request_url,
            headers={"Accept": "application/json", "User-Agent": "saves-model-v3-w1-probe/1"},
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
            "event": event,
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status_code": status_code,
            "quota_headers": quota_headers,
            "raw_body": raw_body,
        }
        try:
            _atomic_json_create(path, record)
        except FileExistsError:
            cached = _load_cached_record(path, signature)
            if cached is None:
                billing["aborted_reason"] = "probe cache collision with unreadable or mismatched record"
                break
            record = cached
        records[event["event_id"]] = record
        cache_status[event["event_id"]] = "fetched"
        billing["calls_completed"] += 1

        if status_code != 200:
            billing["aborted_reason"] = f"HTTP {status_code}; response cached and run stopped"
            break
        if billing["cumulative_header_last_credits"] > max_credits:
            billing["aborted_reason"] = "cumulative x-requests-last credits exceeded --max-credits"
            break
        remaining = _as_int(quota_headers["x-requests-remaining"])
        if remaining is not None and remaining < 0:
            billing["aborted_reason"] = "x-requests-remaining reported below zero"
            break
        time.sleep(0.25)
    return records, cache_status, billing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-by-default W1 historical opening-market coverage probe.",
    )
    parser.add_argument("--execute", action="store_true", help="Make the bounded historical API calls.")
    parser.add_argument(
        "--max-credits",
        type=int,
        default=None,
        help=f"Required with --execute; must be 1..{MAX_EXECUTE_CREDITS}.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_PROBE_CACHE,
        help=f"Dedicated append-only probe cache (default: {DEFAULT_PROBE_CACHE}).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.execute and args.max_credits is None:
        raise SystemExit("error: --execute requires --max-credits")
    if args.max_credits is not None and not 1 <= args.max_credits <= MAX_EXECUTE_CREDITS:
        raise SystemExit(f"error: --max-credits must be between 1 and {MAX_EXECUTE_CREDITS}")

    events_by_id, _ = load_cached_events(CANONICAL_EVENTS_CACHE)
    try:
        sample = select_sample(events_by_id)
    except ValueError as exc:
        raise SystemExit(f"error: cannot build deterministic sample: {exc}") from exc

    records: dict[str, dict[str, Any] | None] = {}
    cache_status: dict[str, str] = {}
    for event in sample:
        signature, _ = request_signature(event)
        record = _load_cached_record(_record_path(args.cache_dir, event["event_id"], signature), signature)
        records[event["event_id"]] = record
        cache_status[event["event_id"]] = "cached" if record is not None else "planned"

    execution = {
        "executed": False,
        "calls_attempted": 0,
        "calls_completed": 0,
        "cached_skips": sum(status == "cached" for status in cache_status.values()),
        "cumulative_conservative_credits": 0,
        "cumulative_header_last_credits": 0,
        "header_last_available_for_calls": 0,
        "latest_headers": {header: None for header in QUOTA_HEADERS},
        "aborted_reason": None,
    }
    if args.execute:
        try:
            records, cache_status, execution = execute(sample, args.cache_dir, args.max_credits)
        except RuntimeError as exc:
            execution["executed"] = True
            execution["aborted_reason"] = str(exc)

    manifest = _run_manifest(sample, records, cache_status, args.max_credits, execution)
    if args.execute:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _atomic_json_create(args.cache_dir / f"w1_run_manifest_{stamp}.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if execution["aborted_reason"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
