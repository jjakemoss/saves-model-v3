"""
Cache manifest and skip-logic for The Odds API historical archive.

Ground truth for the archive format and the caching contract lives in
docs/OFFSEASON_OPTIMIZATION_PLAN.md section 3.15 ("The Odds API historical
acquisition plan") -- read that before changing anything here.

The archive at data/raw/betting_lines/cache/ is a flat, append-only
directory of one file per historical API response, stored verbatim (the
full envelope: timestamp / previous_timestamp / next_timestamp / data).
Three filename shapes:

    events_date={ISO}.json
    odds_{eventId32hex}_date={ISO}_markets={m}_regions={r}.json
    bulk_date={ISO}_markets={m}_regions={r}.json

where {ISO} is the REQUESTED snapshot time with colons replaced by
underscores (":" is not filesystem-safe on Windows), e.g.
"2025-10-18T19_10_00Z" <-> "2025-10-18T19:10:00Z". The envelope's own
"timestamp" field is the RESOLVED snapshot time the API actually returned
(closest available snapshot at-or-before the request -- snapshots are
5-minutely), which can differ from the requested time by a few minutes.

This module never makes a network call. It only reads and writes files
already on (or destined for) disk.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/raw/betting_lines/cache")

SPORT_KEY = "icehockey_nhl"

# Credit cost model (The Odds API v4 historical endpoints; figures per the
# section 3.15 plan, to be reconfirmed by the probe's response headers
# before any real spend).
CREDIT_EVENTS_LIST = 1
CREDIT_PER_REGION_PER_MARKET = 10  # both event-odds and bulk-odds calls

# How many bytes of a cache file to read before falling back to a full
# json.load(). The envelope's timestamp / previous_timestamp /
# next_timestamp keys are always written first (see save_response below),
# so a small head read is enough for every file this module has ever
# written (indent=2, ~150-200 bytes to the end of next_timestamp). The
# second, larger read is a safety margin for anything written with
# different formatting; only if both fail do we pay for a full parse.
_HEAD_BYTES_FAST = 700
_HEAD_BYTES_SLOW = 4000

_EVENTS_RE = re.compile(r"^events_date=(?P<date>.+)\.json$")
_ODDS_RE = re.compile(
    r"^odds_(?P<event_id>[0-9a-fA-F]{32})_date=(?P<date>.+?)"
    r"_markets=(?P<markets>.+?)_regions=(?P<regions>.+)\.json$"
)
_BULK_RE = re.compile(
    r"^bulk_date=(?P<date>.+?)_markets=(?P<markets>.+?)_regions=(?P<regions>.+)\.json$"
)

_ENVELOPE_RE = re.compile(
    r'"timestamp"\s*:\s*"(?P<ts>[^"]*)".*?'
    r'"previous_timestamp"\s*:\s*(?:"(?P<prev>[^"]*)"|null).*?'
    r'"next_timestamp"\s*:\s*(?:"(?P<next>[^"]*)"|null)',
    re.DOTALL,
)

MANIFEST_COLUMNS = [
    "kind", "path", "event_id", "scope", "markets", "regions",
    "requested_ts", "resolved_ts", "next_ts",
]


def timestamp_to_filename_component(ts: str) -> str:
    """'2025-10-18T19:10:00Z' -> '2025-10-18T19_10_00Z' (colons are not
    filesystem-safe on Windows, so the existing convention replaces them
    with underscores)."""
    return ts.replace(":", "_")


def filename_component_to_timestamp(component: str) -> str:
    """Inverse of timestamp_to_filename_component."""
    return component.replace("_", ":")


def _normalize_field(value) -> str:
    """Canonicalize a markets/regions value (list, tuple, or comma string)
    into a sorted, comma-joined string, so 'h2h,totals' and 'totals,h2h'
    -- or a list ['h2h', 'totals'] -- all compare equal for coverage
    purposes. Filenames themselves preserve the caller's original order
    (see save_response); only this comparison key is sorted."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = str(value).split(",")
    return ",".join(sorted(p.strip() for p in parts if p.strip()))


def _parse_filename(name: str) -> Optional[dict]:
    """Parse one cache filename into its kind + scope fields, or None if
    it doesn't match any of the three known shapes."""
    m = _ODDS_RE.match(name)
    if m:
        return {
            "kind": "odds",
            "event_id": m.group("event_id"),
            "scope": m.group("event_id"),
            "markets": _normalize_field(m.group("markets")),
            "regions": _normalize_field(m.group("regions")),
            "date": m.group("date"),
        }
    m = _BULK_RE.match(name)
    if m:
        return {
            "kind": "bulk",
            "event_id": None,
            "scope": SPORT_KEY,
            "markets": _normalize_field(m.group("markets")),
            "regions": _normalize_field(m.group("regions")),
            "date": m.group("date"),
        }
    m = _EVENTS_RE.match(name)
    if m:
        return {
            "kind": "events",
            "event_id": None,
            "scope": SPORT_KEY,
            "markets": "",
            "regions": "",
            "date": m.group("date"),
        }
    return None


def _read_envelope_timestamps(path: Path):
    """Read the envelope's resolved timestamp and next_timestamp without a
    full JSON parse when possible (see _HEAD_BYTES_* above). Falls back to
    a full json.load if the fast head-read regex doesn't match, so this
    never silently drops a record -- it either gets a real answer or the
    caller reports the file as unparsable.

    Returns (resolved_ts, next_ts); either may be None.
    """
    for head_bytes in (_HEAD_BYTES_FAST, _HEAD_BYTES_SLOW):
        try:
            with open(path, "rb") as fh:
                head = fh.read(head_bytes)
        except OSError:
            return None, None
        text = head.decode("utf-8", errors="ignore")
        m = _ENVELOPE_RE.search(text)
        if m:
            return m.group("ts") or None, m.group("next") or None
    # Slow path: full parse (only reached if the head-read regex missed).
    try:
        with open(path, "r", encoding="utf-8") as fh:
            envelope = json.load(fh)
        return envelope.get("timestamp"), envelope.get("next_timestamp")
    except (OSError, json.JSONDecodeError, ValueError):
        return None, None


def scan_archive(cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """
    Build the cache manifest by scanning cache_dir: filenames give kind /
    event_id / markets / regions / requested_ts; envelopes (read cheaply,
    see _read_envelope_timestamps) give resolved_ts / next_ts.

    Returns a DataFrame with columns MANIFEST_COLUMNS, one row per parsed
    cache file. Filenames that could not be parsed (wrong shape, or
    envelope timestamps unreadable) are listed at df.attrs['unparsable']
    rather than silently dropped.
    """
    cache_dir = Path(cache_dir)
    records = []
    unparsable = []

    if not cache_dir.exists():
        df = pd.DataFrame(columns=MANIFEST_COLUMNS)
        df.attrs["unparsable"] = unparsable
        return df

    for path in sorted(cache_dir.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        parsed = _parse_filename(path.name)
        if parsed is None:
            unparsable.append(path.name)
            continue
        try:
            requested_ts = filename_component_to_timestamp(parsed["date"])
        except Exception:
            unparsable.append(path.name)
            continue
        resolved_ts, next_ts = _read_envelope_timestamps(path)
        if resolved_ts is None:
            unparsable.append(path.name)
            continue
        records.append({
            "kind": parsed["kind"],
            "path": str(path),
            "event_id": parsed["event_id"],
            "scope": parsed["scope"],
            "markets": parsed["markets"],
            "regions": parsed["regions"],
            "requested_ts": requested_ts,
            "resolved_ts": resolved_ts,
            "next_ts": next_ts,
        })

    df = pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)
    df.attrs["unparsable"] = unparsable
    return df


def _coverage_index(manifest: pd.DataFrame) -> dict:
    """Build (and cache on the DataFrame via .attrs) a
    (kind, scope, markets, regions) -> [(resolved_ts, next_ts, requested_ts), ...]
    index for fast repeated is_covered() lookups, plus a set of UTC
    calendar dates that have an events file. Rebuilt automatically if the
    manifest's row count changes (e.g. a fresh scan_archive() call)."""
    cache_key = "_coverage_index"
    cached = manifest.attrs.get(cache_key)
    if cached is not None and cached.get("n") == len(manifest):
        return cached["index"]

    index: dict = {}
    events_dates: set = set()
    for row in manifest.itertuples(index=False):
        if row.kind == "events":
            events_dates.add(row.requested_ts[:10])
            continue
        key = (row.kind, row.scope, row.markets, row.regions)
        index.setdefault(key, []).append((row.resolved_ts, row.next_ts, row.requested_ts))

    index["_events_dates"] = events_dates
    manifest.attrs[cache_key] = {"n": len(manifest), "index": index}
    return index


def is_covered(manifest: pd.DataFrame, kind: str, scope, markets, regions, requested_ts: str) -> bool:
    """
    Pre-registered skip rule (section 3.15, "Cache-before-call"): a
    planned request is covered -- already answered by something we hold --
    when an existing record with the same (kind, scope, markets, regions)
    has resolved_ts <= requested_ts < next_ts (next_ts of None means "this
    was the most recent snapshot known at write time", open-ended
    upward). An exact filename match (same requested_ts) is always
    covered too, even in the edge case where the interval math above
    would somehow miss it (e.g. a resolved_ts that is unparsable as a
    plain string comparison for some future non-Z timestamp format).

    kind: 'events' | 'odds' | 'bulk'
    scope: event_id for 'odds'; ignored for 'bulk' and 'events' (both are
        sport-level -- this repo only ever queries icehockey_nhl).
    markets / regions: list, tuple, or comma-string, e.g.
        'player_total_saves' or ['h2h', 'totals']. Ignored for 'events'.
    requested_ts: ISO8601 'Z' string, e.g. '2025-12-04T22:30:00Z'.

    Events files historically carry no markets/regions in the filename and
    one request returns a whole slate, so events coverage is looser by
    design: covered if any events file was requested on the same UTC
    calendar date (any time of day), regardless of markets/regions.
    """
    if manifest is None or len(manifest) == 0:
        return False

    if kind == "events":
        index = _coverage_index(manifest)
        return requested_ts[:10] in index.get("_events_dates", set())

    effective_scope = scope if kind == "odds" else SPORT_KEY
    key = (kind, effective_scope, _normalize_field(markets), _normalize_field(regions))
    index = _coverage_index(manifest)
    records = index.get(key, [])
    if not records:
        return False

    for resolved_ts, next_ts, existing_requested_ts in records:
        if existing_requested_ts == requested_ts:
            return True
        if resolved_ts is None:
            continue
        if resolved_ts <= requested_ts and (next_ts is None or requested_ts < next_ts):
            return True
    return False


def save_response(
    cache_dir: Path,
    kind: str,
    envelope: dict,
    requested_ts: str,
    event_id: Optional[str] = None,
    markets=None,
    regions=None,
) -> Path:
    """
    Write one API response envelope verbatim to the archive using the
    existing naming convention, atomically (write to a temp file in the
    same directory, then os.replace). Refuses to overwrite an existing
    path -- the archive is append-only (section 3.15 rule 1): existing
    files are never modified, renamed, or deleted, and a failed/partial
    response never overwrites an existing file.

    envelope must be the full response body: {timestamp, previous_timestamp,
    next_timestamp, data}. It is written exactly as given (no reformatting
    of the data payload; prices stay decimal, matching the archive's
    existing convention).

    markets/regions may be a list/tuple or a comma-string; the filename
    preserves whatever order is given (e.g. 'h2h,totals').

    Returns the path written.
    """
    cache_dir = Path(cache_dir)
    ts_component = timestamp_to_filename_component(requested_ts)

    if kind == "events":
        filename = f"events_date={ts_component}.json"
    elif kind == "odds":
        if not event_id:
            raise ValueError("event_id is required for kind='odds'")
        markets_str = markets if isinstance(markets, str) else ",".join(markets)
        regions_str = regions if isinstance(regions, str) else ",".join(regions)
        filename = (
            f"odds_{event_id}_date={ts_component}"
            f"_markets={markets_str}_regions={regions_str}.json"
        )
    elif kind == "bulk":
        markets_str = markets if isinstance(markets, str) else ",".join(markets)
        regions_str = regions if isinstance(regions, str) else ",".join(regions)
        filename = f"bulk_date={ts_component}_markets={markets_str}_regions={regions_str}.json"
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    final_path = cache_dir / filename
    if final_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing cache file (archive is append-only): {final_path}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_dir / f".{filename}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh, indent=2)

    if final_path.exists():
        # Extremely unlikely race (another process wrote the same file
        # while we were writing our tmp copy); never clobber it.
        tmp_path.unlink(missing_ok=True)
        raise FileExistsError(
            f"refusing to overwrite existing cache file (archive is append-only): {final_path}"
        )
    os.replace(tmp_path, final_path)
    return final_path


def estimate_event_odds_credits(num_events: int, num_markets: int = 1, num_regions: int = 1) -> int:
    """Historical event-odds: 10 credits per region per market per event
    (per call -- 'num_events' here is really 'number of event-odds calls',
    since callers sometimes use this to price several snapshot times for
    the same event)."""
    return CREDIT_PER_REGION_PER_MARKET * num_regions * num_markets * num_events


def estimate_bulk_odds_credits(num_markets: int = 1, num_regions: int = 1) -> int:
    """Historical bulk odds (all events for the sport in one call): 10
    credits per region per market -- NOT multiplied by event count."""
    return CREDIT_PER_REGION_PER_MARKET * num_regions * num_markets


def estimate_events_list_credits(num_calls: int = 1) -> int:
    """Historical events list: 1 credit per call."""
    return CREDIT_EVENTS_LIST * num_calls
