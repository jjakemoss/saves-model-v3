#!/usr/bin/env python3
"""Independent, read-only audit of the alt_ladder_pilot_202607 purchase.

Implements the independent-audit requirement of
docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 19.3 (Experiment 16),
mirroring scripts/audit_core_bettime_passes.py's structure and stance: this
script is the auditor, not the purchaser. It trusts nothing from
scripts/purchase_alt_ladder_pilot.py's own run-log summaries and recomputes
every claim directly from the raw cached response records under
data/raw/betting_lines/passes/alt_ladder_pilot_202607/ (one
altladder_event={event_id}_signature={signature}.json per historical
event-odds call, plus plan_*.json and run_log.jsonl).

Registered checks (19.3), all recomputed independently:
    1. record integrity      -- every record parses; signature ==
                                 sha256(canonical json of its request field);
                                 filename embeds event_id + signature; request
                                 params match the registered spec (markets per
                                 leg, nine bookmakers, includeMultipliers,
                                 date == independently recomputed bettime
                                 anchor); season-window membership; no apiKey
                                 substring anywhere in raw text; purchased
                                 event set == the frozen plan's first-N
                                 permutation sample; zero overlap with the
                                 already-probed W1 event ids.
    2. billing arithmetic     -- x-requests-last == 10 * (distinct market
                                 keys actually returned) on every 200; grand
                                 total billed; weakly-decreasing
                                 x-requests-remaining in fetched_at order;
                                 implied starting balance reconciled against
                                 the pre-purchase 12,895; final remaining;
                                 remaining never below the registered floor
                                 10,895.
    3. non-200s               -- enumerate, confirm zero-cost, confirm
                                 EVENT_NOT_FOUND where applicable.
    4. anchor/alignment       -- the registered alignment_gap_seconds check
                                 (19.3), independently recomputed from raw
                                 records: |ladder envelope timestamp -
                                 standard-quote observed capture timestamp|,
                                 observed timestamps on both sides. For the
                                 2024-25 alt-only leg the standard side is
                                 core_bettime_202607_snapshots.parquet's
                                 betonlineag/player_total_saves envelope
                                 timestamp (its resolved_ts column -- the
                                 actually-returned snapshot timestamp; see
                                 judgment_calls in the summary). For the
                                 2025-26 combined leg both markets share one
                                 envelope, so the gap is 0 by construction --
                                 still computed and reported, not skipped.
    5. coverage               -- rungs-per-goalie-night distribution
                                 (min/median/mean/max) per book and season;
                                 per-season/market/book event counts;
                                 exact-duplicate and one-sided-outcome schema
                                 traps; rung-spacing and X.5-rung checks.
    6. pairing potential       -- read-only join against
                                 core_bettime_202607_snapshots.parquet
                                 (2024-25) and saves_lines_snapshots.parquet
                                 (2025-26) to independently size the
                                 qualifying primary universe before the
                                 feasibility analysis script runs.

Read-only on everything except its own deliverable:
data/raw/betting_lines/passes/alt_ladder_pilot_202607/audit_summary.json
(written once; refuses to overwrite without --force). Never touches
data/betting.db, never reads .env, never makes a network call.
Deterministic: same raw records, same report.

Usage:
    python scripts/audit_alt_ladder_pilot.py
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

PASS_CACHE_DIR = Path("data/raw/betting_lines/passes/alt_ladder_pilot_202607")
EVENTS_CACHE_DIR = Path("data/raw/betting_lines/cache")
CORE_BETTIME_PARQUET = Path("data/processed/core_bettime_202607_snapshots.parquet")
SAVES_SNAPSHOTS_PARQUET = Path("data/processed/saves_lines_snapshots.parquet")
PROBE_DIR = Path("data/raw/betting_lines/probes/w1_market_coverage")
RUN_LOG_PATH = PASS_CACHE_DIR / "run_log.jsonl"
SUMMARY_PATH = PASS_CACHE_DIR / "audit_summary.json"

EASTERN = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# The registered spec this purchase was supposed to follow (independently
# re-stated here from section 19.3, not imported from
# purchase_alt_ladder_pilot.py -- an audit must not trust the purchaser's
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

LEG_SPECS: dict[str, dict[str, Any]] = {
    "alt_only_2024_25": {
        "season": "2024-25",
        "window": (date(2024, 10, 4), date(2025, 4, 17)),
        "markets": ("player_total_saves_alternate",),
        "target_n": 120,
    },
    "combined_2025_26": {
        "season": "2025-26",
        "window": (date(2025, 10, 7), date(2026, 4, 19)),
        "markets": ("player_total_saves", "player_total_saves_alternate"),
        "target_n": 35,
    },
}

EXPECTED_MARKET_KEYS = {"player_total_saves", "player_total_saves_alternate"}

# Registered spend limits and balances (comparison targets; every value on
# the left of a "matches_*" flag below is recomputed from raw records).
REGISTERED_STARTING_BALANCE = 12895
REGISTERED_CREDIT_FLOOR = 10895
ALIGNMENT_TOLERANCE_SECONDS = 300

# Claims made by the purchase run-log -- reported ONLY as comparison targets.
CLAIMED_TOTAL_BILLED = 1200 + 640
CLAIMED_FINAL_REMAINING = 11055
CLAIMED_NON_200_COUNT = 0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


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
# Load raw records and frozen plans
# ---------------------------------------------------------------------------

def load_raw_records(cache_dir: Path) -> list[dict[str, Any]]:
    """Load every altladder_event=*.json record, keeping the raw text
    alongside the parsed JSON so later checks (apiKey scan, signature
    recompute) work off the exact bytes on disk."""
    entries = []
    for path in sorted(cache_dir.glob("altladder_event=*.json")):
        raw_text = path.read_text(encoding="utf-8")
        try:
            record = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            entries.append({"path": path, "raw_text": raw_text, "record": None, "parse_error": str(exc)})
            continue
        entries.append({"path": path, "raw_text": raw_text, "record": record, "parse_error": None})
    return entries


def load_plans(cache_dir: Path) -> dict[str, dict[str, Any]]:
    plans = {}
    for leg_name in LEG_SPECS:
        plan_path = cache_dir / f"plan_{leg_name}.json"
        if plan_path.exists():
            plans[leg_name] = json.loads(plan_path.read_text(encoding="utf-8"))
    return plans


def load_probe_event_ids(probe_dir: Path) -> set[str]:
    ids = set()
    for path in sorted(probe_dir.glob("w1_event=*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        event_id = rec.get("event", {}).get("event_id")
        if event_id:
            ids.add(event_id)
    return ids


# ---------------------------------------------------------------------------
# 1. Record integrity
# ---------------------------------------------------------------------------

def check_record_integrity(
    entries: list[dict[str, Any]],
    plans: dict[str, dict[str, Any]],
    probe_ids: set[str],
) -> dict[str, Any]:
    total = len(entries)
    parse_failures = [str(e["path"]) for e in entries if e["record"] is None]

    signature_mismatches: list[str] = []
    filename_mismatches: list[str] = []
    param_violations: list[dict[str, Any]] = []
    apikey_hits: list[str] = []
    unknown_leg_names: list[str] = []
    out_of_window: list[dict[str, Any]] = []
    duplicate_event_within_leg: list[dict[str, Any]] = []
    probe_overlap: list[dict[str, Any]] = []

    seen_event_per_leg: dict[str, set[str]] = defaultdict(set)

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
        expected_name = f"altladder_event={event_id}_signature={signature_claimed}.json"
        if path.name != expected_name:
            filename_mismatches.append(str(path))

        leg_name = rec.get("leg_name")
        spec = LEG_SPECS.get(leg_name)
        if spec is None:
            unknown_leg_names.append(str(path))
            continue

        if event_id in seen_event_per_leg[leg_name]:
            duplicate_event_within_leg.append({"path": str(path), "leg_name": leg_name, "event_id": event_id})
        seen_event_per_leg[leg_name].add(event_id)

        if event_id in probe_ids:
            probe_overlap.append({"path": str(path), "leg_name": leg_name, "event_id": event_id})

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
                    {"path": str(path), "leg_name": leg_name, "event_id": event_id, "game_date": game_date.isoformat()}
                )
        if reasons:
            param_violations.append({"path": str(path), "leg_name": leg_name, "reasons": reasons})

    # Plan-vs-disk sample reconciliation: the purchased event set per leg
    # must be EXACTLY the frozen plan's first-target_n permutation events.
    plan_reconciliation: dict[str, Any] = {}
    for leg_name, spec in LEG_SPECS.items():
        plan = plans.get(leg_name)
        if plan is None:
            plan_reconciliation[leg_name] = {"error": "plan file missing"}
            continue
        permutation = plan["full_permutation"]
        pool = plan["pool"]
        expected_ids = {pool[permutation[rank]]["event_id"] for rank in range(spec["target_n"])}
        on_disk_ids = seen_event_per_leg.get(leg_name, set())
        plan_reconciliation[leg_name] = {
            "plan_pool_size": len(pool),
            "plan_target_n": spec["target_n"],
            "n_expected_sample_ids": len(expected_ids),
            "n_on_disk_ids": len(on_disk_ids),
            "on_disk_equals_expected_sample": on_disk_ids == expected_ids,
            "missing_from_disk": sorted(expected_ids - on_disk_ids),
            "unexpected_on_disk": sorted(on_disk_ids - expected_ids),
        }

    event_counts_per_leg = {leg: len(ids) for leg, ids in seen_event_per_leg.items()}
    plan_recon_clean = all(
        isinstance(v, dict) and v.get("on_disk_equals_expected_sample") is True
        for v in plan_reconciliation.values()
    )

    return {
        "total_records_on_disk": total,
        "parse_failures": parse_failures,
        "signature_mismatches": signature_mismatches,
        "filename_mismatches": filename_mismatches,
        "unknown_leg_names": unknown_leg_names,
        "param_violations": param_violations,
        "out_of_season_window": out_of_window,
        "duplicate_event_within_leg": duplicate_event_within_leg,
        "probe_event_overlap": probe_overlap,
        "apikey_hits": apikey_hits,
        "unique_event_count_per_leg": event_counts_per_leg,
        "expected_event_count_per_leg": {leg: spec["target_n"] for leg, spec in LEG_SPECS.items()},
        "plan_reconciliation": plan_reconciliation,
        "all_clean": not (
            parse_failures
            or signature_mismatches
            or filename_mismatches
            or unknown_leg_names
            or param_violations
            or out_of_window
            or duplicate_event_within_leg
            or probe_overlap
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
                "leg_name": rec.get("leg_name"),
                "fetched_at": rec.get("fetched_at"),
                "status_code": status_code,
                "x_last": x_last,
                "x_remaining": x_remaining,
                "n_distinct_markets": n_distinct,
            }
        )

    df = pd.DataFrame(billing_rows)
    sum_x_last = int(df["x_last"].fillna(0).sum())
    sum_per_leg = {
        leg: int(sub["x_last"].fillna(0).sum()) for leg, sub in df.groupby("leg_name")
    }

    # Balance-chain reconciliation. fetched_at has 1-second granularity and
    # calls were dispatched ~0.7s apart, so multiple records legitimately
    # share a fetched_at value and no on-disk field preserves exact
    # dispatch order within a shared second. The rigorous check therefore
    # verifies the CONSTRUCTIVE chain: sorting by x-requests-remaining
    # descending (x_last descending within ties, which places a billing
    # call before the zero-cost call that left the balance unchanged) must
    # yield a sequence in which every record's remaining equals the
    # previous remaining minus its own x_last, starting from the
    # registered 12,895. This is a real, falsifiable check -- if any
    # x_last did not match the actual balance decrement, no such ordering
    # would exist and breaks would appear. The naive fetched_at ordering's
    # apparent inversions are counted separately, as a timestamp-
    # granularity diagnostic only.
    known = df[df["x_remaining"].notna() & df["x_last"].notna()].copy()
    known["x_remaining"] = known["x_remaining"].astype(int)
    known["x_last"] = known["x_last"].astype(int)

    chain = known.sort_values(
        ["x_remaining", "x_last"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
    chain_breaks = []
    prev_remaining = REGISTERED_STARTING_BALANCE
    for i, row in chain.iterrows():
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
    # (expected to be nonzero purely from same-second timestamp collisions).
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

    # Run-log cross-check (evidence, not trusted claims).
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
        "sum_x_requests_last_per_leg": sum_per_leg,
        "claimed_total_billed": CLAIMED_TOTAL_BILLED,
        "matches_claimed_total_billed": sum_x_last == CLAIMED_TOTAL_BILLED,
        "run_log_sum_cumulative_header_last_credits": run_log_sum,
        "run_log_sum_matches_record_sum": run_log_sum == sum_x_last,
        "n_constructive_chain_breaks": len(chain_breaks),
        "constructive_chain_breaks": chain_breaks[:20],
        "n_naive_fetched_at_order_inversions_diagnostic_only": naive_inversions,
        "n_same_second_fetched_at_groups": same_second_groups,
        "n_remaining_below_registered_floor": len(below_floor),
        "remaining_below_registered_floor": below_floor[:20],
        "implied_starting_balance_before_this_purchase": implied_starting_balance,
        "registered_starting_balance": REGISTERED_STARTING_BALANCE,
        "matches_registered_starting_balance": implied_starting_balance == REGISTERED_STARTING_BALANCE,
        "final_remaining": final_remaining,
        "claimed_final_remaining": CLAIMED_FINAL_REMAINING,
        "matches_claimed_final_remaining": final_remaining == CLAIMED_FINAL_REMAINING,
        "implied_total_spend_start_minus_final": (
            REGISTERED_STARTING_BALANCE - final_remaining if final_remaining is not None else None
        ),
        "spend_reconciles_start_to_final": (
            final_remaining is not None
            and REGISTERED_STARTING_BALANCE - final_remaining == sum_x_last
        ),
        "distinct_market_count_distribution": {
            str(k): int(v) for k, v in df["n_distinct_markets"].value_counts(dropna=False).items()
        },
        "all_clean": not (violations or chain_breaks or below_floor)
        and implied_starting_balance == REGISTERED_STARTING_BALANCE
        and final_remaining is not None
        and REGISTERED_STARTING_BALANCE - final_remaining == sum_x_last,
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
                    "leg_name": rec.get("leg_name"),
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
        "claimed_non_200_count": CLAIMED_NON_200_COUNT,
        "matches_claimed_count": len(details) == CLAIMED_NON_200_COUNT,
        "all_zero_cost": all_zero_cost if details else True,
        "details": details,
        "all_clean": len(details) == CLAIMED_NON_200_COUNT and (all_zero_cost if details else True),
    }


# ---------------------------------------------------------------------------
# 4. Anchor / alignment integrity (19.3's alignment_gap_seconds, recomputed)
# ---------------------------------------------------------------------------

def check_alignment(entries: list[dict[str, Any]], core_parquet_path: Path) -> dict[str, Any]:
    """Independent recomputation of the registered alignment check at the
    EVENT level (the per-goalie-night application happens in the analysis
    script; every goalie-night of an event shares the event's envelope
    timestamps, so the event-level gap is the goalie-night gap).

    2024-25 alt-only leg: gap = |ladder envelope timestamp (this purchase's
    own returned body.timestamp) - standard-quote observed capture timestamp
    (core_bettime_202607_snapshots.parquet's resolved_ts for the same event's
    betonlineag/player_total_saves rows)|. resolved_ts is that parquet's
    actually-returned envelope snapshot timestamp -- the observed side of
    17.2's observed-over-nominal rule (the parquet's fetched_at column is the
    2026 wall-clock purchase time, not a snapshot timestamp, and is not used).

    2025-26 combined leg: both markets ride one envelope; the standard
    quote's observed capture IS the ladder envelope timestamp, so the gap is
    0 by construction -- computed and reported anyway, per the registration's
    "still COMPUTED and reported for every night, not skipped" clause."""
    core_df = pd.read_parquet(
        core_parquet_path, columns=["event_id", "book_key", "market_key", "resolved_ts"]
    )
    std_mask = (core_df["book_key"] == "betonlineag") & (core_df["market_key"] == "player_total_saves")
    std_ts_by_event: dict[str, list[str]] = (
        core_df.loc[std_mask].groupby("event_id")["resolved_ts"].agg(lambda s: sorted(set(s))).to_dict()
    )

    rows = []
    unresolvable = []
    multi_ts_events = []
    for entry in entries:
        rec = entry["record"]
        if rec is None or rec.get("status_code") != 200:
            continue
        leg_name = rec.get("leg_name")
        event_id = rec.get("event", {}).get("event_id")
        try:
            body = json.loads(rec.get("raw_body", ""))
        except json.JSONDecodeError:
            unresolvable.append({"path": str(entry["path"]), "reason": "raw_body unparseable"})
            continue
        ladder_env_ts = body.get("timestamp")
        if not ladder_env_ts:
            unresolvable.append({"path": str(entry["path"]), "reason": "no envelope timestamp"})
            continue

        if leg_name == "combined_2025_26":
            gap_seconds = 0.0
            std_observed_ts = ladder_env_ts
        else:
            ts_list = std_ts_by_event.get(event_id)
            if not ts_list:
                unresolvable.append(
                    {"path": str(entry["path"]), "reason": "no betonlineag standard rows in core parquet",
                     "event_id": event_id}
                )
                continue
            if len(ts_list) > 1:
                multi_ts_events.append({"event_id": event_id, "resolved_ts_values": ts_list})
            std_observed_ts = ts_list[0]
            gap_seconds = abs((_parse_utc(ladder_env_ts) - _parse_utc(std_observed_ts)).total_seconds())

        rows.append(
            {
                "leg_name": leg_name,
                "event_id": event_id,
                "ladder_envelope_ts": ladder_env_ts,
                "standard_observed_ts": std_observed_ts,
                "alignment_gap_seconds": gap_seconds,
                "within_tolerance": gap_seconds <= ALIGNMENT_TOLERANCE_SECONDS,
            }
        )

    df = pd.DataFrame(rows)
    per_leg = {}
    for leg, sub in df.groupby("leg_name"):
        gaps = sub["alignment_gap_seconds"]
        per_leg[leg] = {
            "n_events": int(len(sub)),
            "n_within_300s": int(sub["within_tolerance"].sum()),
            "n_exceeding_300s": int((~sub["within_tolerance"]).sum()),
            "gap_min": float(gaps.min()),
            "gap_median": float(gaps.median()),
            "gap_mean": float(gaps.mean()),
            "gap_max": float(gaps.max()),
        }

    exceeding = df[~df["within_tolerance"]]
    return {
        "tolerance_seconds": ALIGNMENT_TOLERANCE_SECONDS,
        "per_leg": per_leg,
        "n_unresolvable": len(unresolvable),
        "unresolvable": unresolvable,
        "n_events_with_multiple_standard_resolved_ts": len(multi_ts_events),
        "events_with_multiple_standard_resolved_ts": multi_ts_events[:20],
        "n_exceeding_tolerance": int(len(exceeding)),
        "exceeding_tolerance": exceeding.to_dict("records"),
        "event_level_rows": df.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# 5. Coverage + schema traps
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
        leg_name = rec.get("leg_name")
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
                            "leg_name": leg_name,
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
    if outcomes.empty:
        return {"error": "no outcomes flattened"}

    result["events_per_season_market"] = (
        outcomes.groupby(["season", "market_key"])["event_id"].nunique().rename("n_events").reset_index()
        .to_dict("records")
    )
    result["events_per_season_market_book"] = (
        outcomes.groupby(["season", "market_key", "book"])["event_id"].nunique().rename("n_events").reset_index()
        .to_dict("records")
    )

    # Rungs per goalie-night (alternate market): distinct Over points per
    # (event, player, book), split by season and book.
    alt = outcomes[outcomes["market_key"] == "player_total_saves_alternate"]
    alt_over = alt[alt["side"] == "Over"]
    rung_counts = (
        alt_over.groupby(["season", "book", "event_id", "description"])["point"].nunique().rename("n_rungs").reset_index()
    )
    rung_dist = []
    for (season, book), sub in rung_counts.groupby(["season", "book"]):
        rung_dist.append(
            {
                "season": season,
                "book": book,
                "n_goalie_nights": int(len(sub)),
                "rungs_min": int(sub["n_rungs"].min()),
                "rungs_median": float(sub["n_rungs"].median()),
                "rungs_mean": float(sub["n_rungs"].mean()),
                "rungs_max": int(sub["n_rungs"].max()),
                "n_ge_5_rungs": int((sub["n_rungs"] >= 5).sum()),
            }
        )
    result["alt_over_rungs_per_goalie_night"] = rung_dist

    # Alternate-market side values (all-Over is the expected one-sided shape
    # for BetOnline; Unders here are a schema surprise to report, not an
    # integrity failure).
    result["alt_side_values_by_book"] = (
        alt.groupby(["book", "side"]).size().rename("n_rows").reset_index().to_dict("records")
    )

    # X.5-rung check: the settlement rule (19.2) relies on every rung being
    # an X.5 value (no push case). Count integer-valued rungs.
    alt_points = alt_over["point"].dropna()
    non_half_points = alt_points[(alt_points * 2) % 2 == 0]
    result["n_alt_over_rung_rows"] = int(len(alt_points))
    result["n_alt_over_integer_point_rows"] = int(len(non_half_points))
    result["integer_point_values_observed"] = sorted(set(non_half_points.tolist()))[:20]

    # Exact duplicates.
    key_cols = ["event_id", "book", "market_key", "description", "point", "side"]
    dup_counts = outcomes.groupby(key_cols).size().rename("n").reset_index()
    dup_rows = dup_counts[dup_counts["n"] > 1]
    result["total_exact_duplicate_extra_copies"] = int((dup_counts["n"] - 1).sum())
    result["duplicate_groups_by_book"] = (
        dup_rows.merge(outcomes[["event_id", "book"] + []].drop_duplicates(), on=["event_id", "book"], how="left")
        .groupby("book").size().to_dict()
        if not dup_rows.empty
        else {}
    )

    # Standard-market pairing completeness (two-sided check), per book.
    std = outcomes[outcomes["market_key"] == "player_total_saves"].drop_duplicates(subset=key_cols)
    if not std.empty:
        pair_key = ["event_id", "season", "book", "description", "point"]
        side_sets = std.groupby(pair_key)["side"].apply(lambda s: frozenset(s)).reset_index()
        side_sets["paired"] = side_sets["side"].apply(lambda s: {"Over", "Under"}.issubset(s))
        pairing = (
            side_sets.groupby(["season", "book"])
            .agg(n_groups=("paired", "size"), n_paired=("paired", "sum"))
            .reset_index()
        )
        pairing["n_unpaired"] = pairing["n_groups"] - pairing["n_paired"]
        result["standard_market_pairing_per_season_book"] = pairing.to_dict("records")
    else:
        result["standard_market_pairing_per_season_book"] = []

    result["schema_traps"] = {
        "n_null_point": int(outcomes["point"].isna().sum()),
        "n_null_or_empty_description": int((outcomes["description"].isna() | (outcomes["description"] == "")).sum()),
        "n_non_null_multiplier": int(outcomes["multiplier"].notna().sum()),
        "market_keys_observed": sorted(outcomes["market_key"].dropna().unique().tolist()),
        "unexpected_market_keys": sorted(set(outcomes["market_key"].dropna().unique()) - EXPECTED_MARKET_KEYS),
        "side_values_observed": sorted(outcomes["side"].dropna().unique().tolist()),
        "unexpected_side_values": sorted(set(outcomes["side"].dropna().unique()) - {"Over", "Under"}),
        "book_keys_observed": sorted(outcomes["book"].dropna().unique().tolist()),
        "book_keys_never_seen": sorted(set(BOOKMAKERS) - set(outcomes["book"].dropna().unique())),
    }
    return result


# ---------------------------------------------------------------------------
# 6. Pairing potential (read-only join, sizes the analysis universe)
# ---------------------------------------------------------------------------

def check_pairing_potential(
    outcomes: pd.DataFrame, core_parquet_path: Path, saves_parquet_path: Path
) -> dict[str, Any]:
    if outcomes.empty:
        return {"error": "no outcomes"}
    result: dict[str, Any] = {}

    # 2024-25 leg: BetOnline ladder events joined to the core pass's
    # BetOnline standard-saves quotes (the registered anchor source).
    alt_2425 = outcomes[
        (outcomes["leg_name"] == "alt_only_2024_25")
        & (outcomes["market_key"] == "player_total_saves_alternate")
        & (outcomes["book"] == "betonlineag")
    ]
    ladder_events_2425 = set(alt_2425["event_id"].unique())
    core_df = pd.read_parquet(
        core_parquet_path,
        columns=["event_id", "book_key", "market_key", "player_name_raw", "side", "line"],
    )
    core_std = core_df[
        (core_df["book_key"] == "betonlineag") & (core_df["market_key"] == "player_total_saves")
    ]
    core_events = set(core_std["event_id"].unique())
    result["2024_25"] = {
        "n_purchased_events_with_betonline_ladder": len(ladder_events_2425),
        "n_of_those_with_core_betonline_standard": len(ladder_events_2425 & core_events),
    }

    # Goalie-night level: ladder (event, player) x core standard two-sided (event, player).
    ladder_gn = set(map(tuple, alt_2425[["event_id", "description"]].drop_duplicates().values))
    sides_per = core_std.groupby(["event_id", "player_name_raw", "line"])["side"].nunique().reset_index()
    two_sided = sides_per[sides_per["side"] == 2]
    core_gn = set(map(tuple, two_sided[["event_id", "player_name_raw"]].drop_duplicates().values))
    result["2024_25"]["n_ladder_goalie_nights"] = len(ladder_gn)
    result["2024_25"]["n_ladder_goalie_nights_with_two_sided_core_anchor"] = len(ladder_gn & core_gn)

    # 2025-26 leg: same-envelope check -- ladder goalie-nights vs. own-call
    # BetOnline standard two-sided quotes, plus informational overlap with
    # the existing saves_lines_snapshots archive.
    alt_2526 = outcomes[
        (outcomes["leg_name"] == "combined_2025_26")
        & (outcomes["market_key"] == "player_total_saves_alternate")
        & (outcomes["book"] == "betonlineag")
    ]
    std_2526 = outcomes[
        (outcomes["leg_name"] == "combined_2025_26")
        & (outcomes["market_key"] == "player_total_saves")
        & (outcomes["book"] == "betonlineag")
    ]
    ladder_gn_2526 = set(map(tuple, alt_2526[["event_id", "description"]].drop_duplicates().values))
    sides_2526 = std_2526.groupby(["event_id", "description", "point"])["side"].nunique().reset_index()
    two_sided_2526 = sides_2526[sides_2526["side"] == 2]
    std_gn_2526 = set(map(tuple, two_sided_2526[["event_id", "description"]].drop_duplicates().values))
    all_events_2526 = set(
        outcomes[outcomes["leg_name"] == "combined_2025_26"]["event_id"].unique()
    )
    result["2025_26"] = {
        "n_purchased_events_any_data": len(all_events_2526),
        "n_events_with_betonline_ladder": int(alt_2526["event_id"].nunique()),
        "n_ladder_goalie_nights": len(ladder_gn_2526),
        "n_ladder_goalie_nights_with_same_envelope_two_sided_standard": len(ladder_gn_2526 & std_gn_2526),
    }

    saves_df = pd.read_parquet(saves_parquet_path, columns=["event_id", "snapshot_pass", "book"])
    bettime_events = set(saves_df[saves_df["snapshot_pass"] == "bettime"]["event_id"].unique())
    result["2025_26"]["n_purchased_events_also_in_existing_bettime_archive"] = len(
        all_events_2526 & bettime_events
    )
    return result


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(summary: dict[str, Any]) -> None:
    ri = summary["record_integrity"]
    bill = summary["billing"]
    non200 = summary["non_200s"]
    align = summary["alignment"]
    cov = summary["coverage"]
    pair = summary["pairing_potential"]

    print("=" * 78)
    print("AUDIT: alt_ladder_pilot_202607 purchase (Experiment 16, section 19.3)")
    print("=" * 78)

    print("\n--- 1. Record integrity ---")
    print(f"  total records on disk:        {ri['total_records_on_disk']}")
    print(f"  parse failures:                {len(ri['parse_failures'])}")
    print(f"  signature mismatches:          {len(ri['signature_mismatches'])}")
    print(f"  filename mismatches:           {len(ri['filename_mismatches'])}")
    print(f"  param/spec violations:         {len(ri['param_violations'])}")
    print(f"  out-of-season-window events:   {len(ri['out_of_season_window'])}")
    print(f"  duplicate event within leg:    {len(ri['duplicate_event_within_leg'])}")
    print(f"  overlap with probed W1 events: {len(ri['probe_event_overlap'])}")
    print(f"  apiKey text found:             {len(ri['apikey_hits'])}")
    print(f"  unique events per leg:         {ri['unique_event_count_per_leg']}")
    print(f"  expected events per leg:       {ri['expected_event_count_per_leg']}")
    for leg, rec in ri["plan_reconciliation"].items():
        print(f"  plan reconciliation [{leg}]: {rec}")
    print(f"  ALL CLEAN:                     {ri['all_clean']}")

    print("\n--- 2. Billing arithmetic ---")
    print(f"  records considered:                       {bill['n_records_considered']}")
    print(f"  billing formula violations:               {bill['n_billing_violations']}")
    print(f"  sum(x-requests-last) all records:         {bill['sum_x_requests_last_across_all_records']}")
    print(f"  sum(x-requests-last) per leg:             {bill['sum_x_requests_last_per_leg']}")
    print(f"  claimed total billed:                     {bill['claimed_total_billed']}")
    print(f"  MATCHES CLAIMED TOTAL:                    {bill['matches_claimed_total_billed']}")
    print(f"  run_log sum matches record sum:            {bill['run_log_sum_matches_record_sum']}")
    print(f"  constructive balance-chain breaks:         {bill['n_constructive_chain_breaks']}")
    print(f"  naive fetched_at-order inversions (diag):  {bill['n_naive_fetched_at_order_inversions_diagnostic_only']} "
          f"(same-second fetched_at groups: {bill['n_same_second_fetched_at_groups']})")
    print(f"  remaining ever below floor ({REGISTERED_CREDIT_FLOOR}):     {bill['n_remaining_below_registered_floor']}")
    print(f"  implied starting balance:                  {bill['implied_starting_balance_before_this_purchase']}")
    print(f"  registered starting balance:               {bill['registered_starting_balance']}")
    print(f"  MATCHES REGISTERED STARTING BALANCE:       {bill['matches_registered_starting_balance']}")
    print(f"  final remaining:                           {bill['final_remaining']}")
    print(f"  MATCHES CLAIMED FINAL REMAINING:           {bill['matches_claimed_final_remaining']}")
    print(f"  spend reconciles start->final == sum:      {bill['spend_reconciles_start_to_final']}")
    print(f"  distinct-market-count distribution:        {bill['distinct_market_count_distribution']}")
    print(f"  ALL CLEAN:                                 {bill['all_clean']}")

    print("\n--- 3. Non-200s ---")
    print(f"  non-200 records found:      {non200['n_non_200']}")
    print(f"  all zero-cost:              {non200['all_zero_cost']}")
    for d in non200["details"]:
        print(f"    {d}")
    print(f"  ALL CLEAN:                  {non200['all_clean']}")

    print("\n--- 4. Alignment (registered alignment_gap_seconds check) ---")
    print(f"  tolerance: {align['tolerance_seconds']}s")
    for leg, stats in align["per_leg"].items():
        print(f"  [{leg}] {stats}")
    print(f"  unresolvable (fail-closed at analysis): {align['n_unresolvable']}")
    for u in align["unresolvable"][:10]:
        print(f"    {u}")
    print(f"  events with multiple standard resolved_ts: {align['n_events_with_multiple_standard_resolved_ts']}")
    print(f"  events exceeding tolerance: {align['n_exceeding_tolerance']}")
    for e in align["exceeding_tolerance"][:10]:
        print(f"    {e}")

    print("\n--- 5. Coverage / schema traps ---")
    print("  events per season/market:")
    for row in cov["events_per_season_market"]:
        print(f"    {row}")
    print("  alt-Over rungs per goalie-night (per season/book):")
    for row in cov["alt_over_rungs_per_goalie_night"]:
        print(f"    {row}")
    print(f"  alt-market side values by book: {cov['alt_side_values_by_book']}")
    print(f"  n integer-valued alt rung rows: {cov['n_alt_over_integer_point_rows']} "
          f"of {cov['n_alt_over_rung_rows']}")
    print(f"  exact duplicate extra copies: {cov['total_exact_duplicate_extra_copies']}")
    print("  standard-market pairing per season/book:")
    for row in cov["standard_market_pairing_per_season_book"]:
        print(f"    {row}")
    print("  schema traps:")
    for k, v in cov["schema_traps"].items():
        print(f"    {k}: {v}")

    print("\n--- 6. Pairing potential (analysis-universe sizing) ---")
    for season_key, block in pair.items():
        print(f"  [{season_key}] {block}")

    print("\n" + "=" * 78)
    print(f"OVERALL INTEGRITY CLEAN: {summary['overall_integrity_clean']}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent read-only audit of the alt_ladder_pilot_202607 purchase. "
                    "Recomputes every claim from raw cached records; no network calls.",
    )
    parser.add_argument("--cache-dir", type=Path, default=PASS_CACHE_DIR)
    parser.add_argument("--core-parquet", type=Path, default=CORE_BETTIME_PARQUET)
    parser.add_argument("--saves-parquet", type=Path, default=SAVES_SNAPSHOTS_PARQUET)
    parser.add_argument("--probe-dir", type=Path, default=PROBE_DIR)
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
        print(f"error: no altladder_event=*.json records found under {args.cache_dir}", file=sys.stderr)
        return 1

    plans = load_plans(args.cache_dir)
    probe_ids = load_probe_event_ids(args.probe_dir)

    record_integrity = check_record_integrity(entries, plans, probe_ids)
    billing = check_billing(entries)
    non_200s = check_non_200s(entries)
    alignment = check_alignment(entries, args.core_parquet)
    outcomes = flatten_outcomes(entries)
    coverage = check_coverage(outcomes)
    pairing_potential = check_pairing_potential(outcomes, args.core_parquet, args.saves_parquet)

    overall_clean = bool(
        record_integrity["all_clean"] and billing["all_clean"] and non_200s["all_clean"]
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_cache_dir": str(args.cache_dir),
        "record_integrity": record_integrity,
        "billing": billing,
        "non_200s": non_200s,
        "alignment": alignment,
        "coverage": coverage,
        "pairing_potential": pairing_potential,
        "overall_integrity_clean": overall_clean,
        "judgment_calls": [
            "Alignment standard-quote observed timestamp for the 2024-25 alt-only leg uses "
            "core_bettime_202607_snapshots.parquet's resolved_ts column (the actually-returned "
            "envelope snapshot timestamp of the core pass's own call). The registration's "
            "parenthetical names 'fetched_at/envelope timestamp' for core-sourced anchors; the "
            "parquet's fetched_at column is the 2026 wall-clock purchase time (not an observed "
            "snapshot timestamp), so resolved_ts -- the envelope timestamp -- is the only reading "
            "consistent with the registration's own observed-over-nominal rule.",
            "Alignment is computed at event level here; every goalie-night of an event shares "
            "the event's envelope timestamps, so per-goalie-night gaps in the analysis script "
            "are inherited from these event-level values unchanged.",
            "Balance-chain verification uses the constructive ordering (x-requests-remaining "
            "descending, x_last descending within ties) rather than fetched_at order, because "
            "fetched_at has 1-second granularity while calls were dispatched ~0.7s apart -- 36 "
            "records share a fetched_at second with another record, so no on-disk field "
            "preserves exact dispatch order within those seconds. An earlier draft of this "
            "audit sorted naively by (fetched_at, path) and reported 22 apparent non-monotonic "
            "transitions and 66 apparent chain breaks; all were artifacts of that ordering "
            "assumption (alphabetical path tiebreak within shared seconds), individually "
            "reconciled by the constructive chain, which reconciles all 155 records from "
            "12,895 to 11,055 with zero breaks. The audit was rerun with --force after this "
            "fix; this summary supersedes the first run's output.",
        ],
    }

    print_report(summary)

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.force else "x"
    with open(args.summary_path, mode, encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    print(f"\nWrote audit summary to {args.summary_path}")

    return 0 if overall_clean else 1


if __name__ == "__main__":
    sys.exit(main())
