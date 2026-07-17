#!/usr/bin/env python3
"""Experiment 16: alternate-saves one-sided-ladder feasibility pilot analysis.

Implements sections 19.4/19.5 of docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md
exactly. That registration is the binding contract for every definition,
formula, gate, and pass bar used here; this docstring restates the shape of
the run, not the rules themselves.

Read-only inputs (no writes to any existing file, no network calls, no
touching data/betting.db -- forbidden even for reads per 19.6.2):
  - data/raw/betting_lines/passes/alt_ladder_pilot_202607/ -- the 155 new
    purchase records (audited clean by scripts/audit_alt_ladder_pilot.py;
    this script refuses to load a single record until that audit's
    overall_integrity_clean flag is verified true, per 19.6.9), plus the two
    frozen plan_*.json artifacts.
  - data/raw/betting_lines/probes/w1_market_coverage/ -- the 15 probe-reuse
    events (7 in 2024-25, 8 in 2025-26) folded into the primary universe at
    zero cost, flagged source="probe_reuse" (19.3).
  - data/processed/core_bettime_202607_snapshots.parquet -- the 2024-25
    alt-only leg's standard-quote anchor source (19.3).
  - data/processed/clean_training_data.parquet -- outcomes (saves), the
    goalie-night denominator, and the goalie-identity lookup. The ONLY
    outcome source (19.2).
  - data/processed/saves_lines_snapshots.parquet -- 2023-24 bettime rows for
    the sigma_0 fresh-recompute wiring-gate path (19.4).
  - models/trained/experiment_15_w3_microstructure_20260716_124811/
    juice_goalie_night_features.parquet -- read-only, the sigma_0
    primary-source path (19.4).

Registered computation order: audit gate, structural reconciliation, universe
construction, and the sigma_0 two-path wiring gate all run BEFORE any
registered 19.5 statistic (coverage rate, PRIMARY CI, SECONDARY log-loss) is
computed; a hard stop in that preflight region does not consume the single
registered execution (19.6.10). Once the coverage rate is logged, numbers
stand.

Output: a new timestamped artifact directory under models/trained/ holding
metadata.json (every registered statistic at full precision, the 19.5
verdict, the exclusion funnel, deviations/judgment-call lists, input
checksums), the rung-level paired frame, the funnel frame, the sigma_0 fit
inputs frame, copies of both frozen plan artifacts, and run_log.txt.

Usage:
    python scripts/experiment_16_alt_ladder_pilot.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_odds_snapshots as bos  # noqa: E402  (match_goalie, build_base_lookup, commence_to_eastern_date)
import clv_audit_pace_policy as clv  # noqa: E402  (attach_game_id, pivot_both_sides)
from build_multibook_training_data import TEAM_NAME_TO_ABBREV  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (read-only inputs; data/betting.db is never referenced, per 19.6.2)
# ---------------------------------------------------------------------------

PILOT_CACHE = REPO_ROOT / "data" / "raw" / "betting_lines" / "passes" / "alt_ladder_pilot_202607"
AUDIT_SUMMARY = PILOT_CACHE / "audit_summary.json"
PROBE_DIR = REPO_ROOT / "data" / "raw" / "betting_lines" / "probes" / "w1_market_coverage"
CORE_BETTIME = REPO_ROOT / "data" / "processed" / "core_bettime_202607_snapshots.parquet"
CLEAN_TRAINING = REPO_ROOT / "data" / "processed" / "clean_training_data.parquet"
SAVES_SNAPSHOTS = REPO_ROOT / "data" / "processed" / "saves_lines_snapshots.parquet"
EXP15_JUICE = (
    REPO_ROOT / "models" / "trained" / "experiment_15_w3_microstructure_20260716_124811"
    / "juice_goalie_night_features.parquet"
)
OUTPUT_ROOT = REPO_ROOT / "models" / "trained"

# ---------------------------------------------------------------------------
# Registered constants (19.2-19.5)
# ---------------------------------------------------------------------------

PRIMARY_BOOK = "betonlineag"
DFS_BOOKS = {"prizepicks", "underdog"}
ALT_MARKET = "player_total_saves_alternate"
STD_MARKET = "player_total_saves"
ALIGNMENT_TOLERANCE_SECONDS = 300
RUNG_DEPTH_FLOOR = 5
COVERAGE_GATE = 0.70
CLUSTER_FLOOR = 50
N_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
LOGLOSS_CLIP = 1e-6
SEASON_2023_24_INT = 20232024
WINDOW_2023_24 = ("2023-10-10", "2024-04-18")
EXPECTED_N_NEW = {"alt_only_2024_25": 120, "combined_2025_26": 35}
EXPECTED_N_PROBE = {"2024-25": 7, "2025-26": 8}
WIRING_GATE_ATOL = 1e-12


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_record_dir(dir_path: Path, pattern: str) -> str:
    """Aggregate digest over a directory of records: sha256 of the sorted
    concatenation of 'name:sha256(file)' lines."""
    h = hashlib.sha256()
    for path in sorted(dir_path.glob(pattern)):
        h.update(f"{path.name}:{sha256_file(path)}\n".encode("utf-8"))
    return h.hexdigest()


class Logger:
    def __init__(self, log_path: Path):
        self.handle = open(log_path, "w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.handle.write(msg + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


# ---------------------------------------------------------------------------
# Raw record loading
# ---------------------------------------------------------------------------

def load_pilot_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(PILOT_CACHE.glob("altladder_event=*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        records.append(rec)
    return records


def load_probe_reuse_records() -> list[dict[str, Any]]:
    """The 15 probe-reuse events: W1 probe records in seasons 2024-25 /
    2025-26 whose response carries a BetOnline player_total_saves_alternate
    market (19.3's '7 in 2024-25, 8 in 2025-26'). The 2023-24 probe events
    (no alternates exist there) and the one 2024-25 probed event with no
    BetOnline alternate data are not part of the sampled universe."""
    out = []
    for path in sorted(PROBE_DIR.glob("w1_event=*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        season = rec.get("event", {}).get("season")
        if season not in ("2024-25", "2025-26") or rec.get("status_code") != 200:
            continue
        body = json.loads(rec["raw_body"])
        has_bol_alt = any(
            bm.get("key") == PRIMARY_BOOK
            and any(mk.get("key") == ALT_MARKET for mk in bm.get("markets") or [])
            for bm in (body.get("data") or {}).get("bookmakers") or []
        )
        if has_bol_alt:
            out.append(rec)
    return out


def flatten_book_outcomes(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """All outcome rows of one 200 record (all books; the PRIMARY universe
    filters to betonlineag later, but the non-primary coverage diagnostics
    in 19.5 need every book)."""
    body = json.loads(rec["raw_body"])
    data = body.get("data") or {}
    event_meta = rec["event"]
    rows = []
    for bm in data.get("bookmakers") or []:
        book = bm.get("key")
        for mk in bm.get("markets") or []:
            market_key = mk.get("key")
            for outcome in mk.get("outcomes") or []:
                rows.append(
                    {
                        "event_id": event_meta["event_id"],
                        "season": event_meta["season"],
                        "book": book,
                        "market_key": market_key,
                        "side": outcome.get("name"),
                        "description": outcome.get("description"),
                        "point": outcome.get("point"),
                        "price_decimal": outcome.get("price"),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Event -> clean_training_data game mapping
# ---------------------------------------------------------------------------

def map_event_to_game(
    clean: pd.DataFrame, home_full: str, away_full: str, eastern_date: str
) -> tuple[int | None, str]:
    """Map an Odds API event to a clean_training_data game_id via team
    abbreviations and the Eastern game date, trying exact date first, then
    +/-1 day (the attach_game_id convention's tolerance, applied at the
    event level). Returns (game_id or None, status)."""
    home_ab = TEAM_NAME_TO_ABBREV.get(home_full)
    away_ab = TEAM_NAME_TO_ABBREV.get(away_full)
    if not home_ab or not away_ab:
        return None, "team_name_unmapped"
    base = pd.Timestamp(eastern_date)
    for offset in (0, -1, 1):
        d = (base + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        sub = clean[
            (clean["game_date_str"] == d)
            & (
                ((clean["team_abbrev"] == home_ab) & (clean["opponent_team"] == away_ab))
                | ((clean["team_abbrev"] == away_ab) & (clean["opponent_team"] == home_ab))
            )
        ]
        if len(sub):
            game_ids = sorted(sub["game_id"].unique())
            if len(game_ids) == 1:
                return int(game_ids[0]), "matched"
            return None, "ambiguous_multiple_games"
    return None, "no_game_found"


# ---------------------------------------------------------------------------
# sigma_0 baseline fit (19.4)
# ---------------------------------------------------------------------------

def dedup_max_resolved_ts(df: pd.DataFrame, key_cols: list[str], log, label: str) -> pd.DataFrame:
    """17.2's registered dedup rule (restated as binding by 19.2): within a
    single snapshot_pass, for rows sharing key_cols, keep only the row with
    the MAXIMUM resolved_ts; ties by maximum requested_ts; any remaining tie
    broken deterministically by original row order. Faithful reimplementation
    of experiment_15_w3_microstructure.dedup_max_resolved_ts (the formula is
    the registered source of truth; the sigma_0 two-path wiring gate below
    verifies this reimplementation reproduces Experiment 15's persisted
    values exactly)."""
    df = df.reset_index(drop=True).copy()
    n_before = len(df)
    df["_orig_order"] = np.arange(n_before)
    df["_res"] = pd.to_datetime(df["resolved_ts"], utc=True, errors="coerce")
    df["_req"] = pd.to_datetime(df["requested_ts"], utc=True, errors="coerce")
    df_sorted = df.sort_values(key_cols + ["_res", "_req", "_orig_order"], na_position="first")
    deduped = df_sorted.groupby(key_cols, as_index=False, sort=False).tail(1)
    deduped = deduped.drop(columns=["_orig_order", "_res", "_req"]).reset_index(drop=True)
    log(f"[{label}] 17.2 dedup on {key_cols}: {n_before} -> {len(deduped)} rows.")
    return deduped


def compute_juice_2023_24_fresh(clean: pd.DataFrame, log) -> pd.DataFrame:
    """Fresh recompute of 18.2's juice consensus machinery on 2023-24
    bettime rows of saves_lines_snapshots.parquet (19.4's path B). Identical
    formula to Experiment 15: 17.2 dedup -> DFS exclusion -> resolved
    goalie_id only -> attach_game_id ((goalie_id, game_date_eastern) +/-1
    day) -> pivot_both_sides (both sides of the exact same line required) ->
    proportional de-vig -> modal line (most books, ties -> lowest) ->
    at-modal median p_under_devigged."""
    snapshots = pd.read_parquet(SAVES_SNAPSHOTS)
    bt = snapshots[snapshots["snapshot_pass"] == "bettime"].copy()
    start, end = WINDOW_2023_24
    bt = bt[(bt["game_date_eastern"] >= start) & (bt["game_date_eastern"] <= end)].copy()
    log(f"[sigma0 path B] 2023-24 bettime raw rows: {len(bt)}, events: {bt['event_id'].nunique()}.")

    df = dedup_max_resolved_ts(bt, ["event_id", "goalie_name_raw", "book", "side"], log, "sigma0 path B")
    df = df[~df["book"].isin(DFS_BOOKS)].copy()
    df = df[df["goalie_id"].notna()].copy()
    df["goalie_id"] = df["goalie_id"].astype(int)

    game_lookup_df = clean[["goalie_id", "game_date_str", "game_id"]]
    game_lookup = dict(
        zip(zip(game_lookup_df["goalie_id"], game_lookup_df["game_date_str"]), game_lookup_df["game_id"])
    )
    df, n_unmatched = clv.attach_game_id(df, game_lookup)
    log(f"[sigma0 path B] attach_game_id: {n_unmatched} unmatched dropped -> {len(df)} rows.")

    wide = clv.pivot_both_sides(df, ["event_id", "game_id", "goalie_id", "book", "line"])
    wide["raw_p_over"] = 1.0 / wide["price_decimal_over"]
    wide["raw_p_under"] = 1.0 / wide["price_decimal_under"]
    wide["overround"] = wide["raw_p_over"] + wide["raw_p_under"]
    wide["p_under_devigged"] = wide["raw_p_under"] / wide["overround"]

    n_books = wide.groupby(["game_id", "goalie_id"])["book"].nunique().rename("juice_n_books")
    line_counts = wide.groupby(["game_id", "goalie_id", "line"]).size().reset_index(name="n_at_line")
    line_counts = line_counts.sort_values(
        ["game_id", "goalie_id", "n_at_line", "line"], ascending=[True, True, False, True]
    )
    modal = line_counts.drop_duplicates(subset=["game_id", "goalie_id"], keep="first")[
        ["game_id", "goalie_id", "line"]
    ].rename(columns={"line": "juice_modal_line"})
    wide = wide.merge(modal, on=["game_id", "goalie_id"], how="left")
    at_modal = wide[wide["line"] == wide["juice_modal_line"]]
    consensus = (
        at_modal.groupby(["game_id", "goalie_id"])["p_under_devigged"].median().rename("juice_p_under_consensus")
    )
    feat = (
        n_books.to_frame()
        .join(modal.set_index(["game_id", "goalie_id"]))
        .join(consensus)
        .reset_index()
    )
    feat["juice_matched"] = (feat["juice_n_books"] > 0).astype(int)
    log(f"[sigma0 path B] fresh 2023-24 juice frame: {len(feat)} goalie-nights.")
    return feat


def fit_sigma_0(clean: pd.DataFrame, log) -> tuple[float | None, pd.DataFrame, dict[str, Any]]:
    """19.4's registered sigma_0 fit: two-path sourcing with a mandatory
    wiring gate, then the closed-form OLS-through-origin."""
    # Path A: Experiment 15's persisted per-goalie-night frame (read-only).
    juice_a = pd.read_parquet(EXP15_JUICE)
    season_lookup = clean[["game_id", "goalie_id", "season"]]
    juice_a = juice_a.merge(season_lookup, on=["game_id", "goalie_id"], how="left")
    a_2324 = juice_a[juice_a["season"] == SEASON_2023_24_INT].copy()
    log(f"[sigma0 path A] Experiment 15 persisted frame: {len(juice_a)} rows total, "
        f"{len(a_2324)} in 2023-24.")

    # Path B: fresh recompute from saves_lines_snapshots.parquet.
    b_2324 = compute_juice_2023_24_fresh(clean, log)

    # Wiring gate: both paths must agree (19.4: "either path must reproduce
    # identical values on a spot-check sample before being trusted"; the
    # spot-check here is the FULL 2023-24 frame, a superset of any sample).
    a_keyed = a_2324.set_index(["game_id", "goalie_id"]).sort_index()
    b_keyed = b_2324.set_index(["game_id", "goalie_id"]).sort_index()
    keys_equal = a_keyed.index.equals(b_keyed.index)
    gate: dict[str, Any] = {
        "n_path_a": int(len(a_keyed)),
        "n_path_b": int(len(b_keyed)),
        "key_sets_identical": bool(keys_equal),
    }
    if not keys_equal:
        only_a = a_keyed.index.difference(b_keyed.index)
        only_b = b_keyed.index.difference(a_keyed.index)
        gate["n_only_in_a"] = int(len(only_a))
        gate["n_only_in_b"] = int(len(only_b))
        log(f"[sigma0 wiring gate] KEY MISMATCH: {len(only_a)} only in A, {len(only_b)} only in B.")
        return None, pd.DataFrame(), gate
    for col in ("juice_p_under_consensus", "juice_modal_line", "juice_n_books", "juice_matched"):
        diff = (a_keyed[col].astype(float) - b_keyed[col].astype(float)).abs().max()
        gate[f"max_abs_diff_{col}"] = float(diff)
    gate_pass = all(
        gate[f"max_abs_diff_{c}"] <= WIRING_GATE_ATOL
        for c in ("juice_p_under_consensus", "juice_modal_line", "juice_n_books", "juice_matched")
    )
    gate["gate_pass"] = bool(gate_pass)
    log(f"[sigma0 wiring gate] {gate}")
    if not gate_pass:
        return None, pd.DataFrame(), gate

    # Fit inputs (path A values, now verified identical to path B).
    fit = a_2324[a_2324["juice_matched"] == 1].copy()
    fit = fit.merge(clean[["game_id", "goalie_id", "saves"]], on=["game_id", "goalie_id"], how="inner")
    fit["p_over"] = 1.0 - fit["juice_p_under_consensus"]
    n_before = len(fit)
    exact01 = (fit["p_over"] <= 0.0) | (fit["p_over"] >= 1.0)
    n_exact01 = int(exact01.sum())
    fit = fit[~exact01].copy()
    log(f"[sigma0 fit] juice_matched 2023-24 nights with outcomes: {n_before}; "
        f"excluded exact-0/1 p_over: {n_exact01}; fit n = {len(fit)}.")

    fit["z"] = norm.ppf(1.0 - fit["p_over"])
    fit["x"] = -fit["z"]
    fit["y"] = fit["saves"].astype(float) - (fit["juice_modal_line"].astype(float) + 0.5)
    sxx = float((fit["x"] ** 2).sum())
    sxy = float((fit["x"] * fit["y"]).sum())
    sigma_0 = sxy / sxx if sxx > 0 else None
    gate["n_fit"] = int(len(fit))
    gate["n_excluded_exact_0_or_1"] = n_exact01
    gate["sum_x_squared"] = sxx
    gate["sum_x_y"] = sxy
    log(f"[sigma0 fit] sigma_0 = {sigma_0!r} (sum_xy={sxy!r}, sum_xx={sxx!r}).")
    fit_frame = fit[
        ["game_id", "goalie_id", "juice_modal_line", "juice_p_under_consensus", "p_over", "saves", "x", "y"]
    ].reset_index(drop=True)
    return sigma_0, fit_frame, gate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_16_alt_ladder_pilot_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log = Logger(output_dir / "run_log.txt")
    log(f"Experiment 16 alt-ladder pilot analysis -- run {timestamp} UTC")
    log(f"Output: {output_dir}")

    deviations: list[str] = []
    judgment_calls: list[str] = [
        "Alignment standard-side observed timestamp for the 2024-25 alt-only leg is "
        "core_bettime_202607_snapshots.parquet's resolved_ts (the core pass's actually-returned "
        "envelope snapshot timestamp); the parquet's fetched_at column is the 2026 wall-clock "
        "purchase time and is not an observed snapshot timestamp, so resolved_ts is the only "
        "reading consistent with 19.3's observed-over-nominal rule.",
        "The 19.5(c) rung-depth criterion is implemented as: n_distinct_rungs >= 5, where the "
        "anchor rung counts toward the 5 when present on the ladder. A night whose ladder does "
        "not contain the anchor rung is flagged as a 19.2 anomaly (counted below) but still "
        "processed if it has >= 5 rungs, per 19.2's 'anomaly to flag, not silently drop'.",
    ]

    # ---- Audit gate (19.6.9): no record is loaded before this passes. ----
    if not AUDIT_SUMMARY.exists():
        log("HARD STOP: audit_summary.json not found; the independent audit must run first (19.6.9).")
        return 1
    audit = json.loads(AUDIT_SUMMARY.read_text(encoding="utf-8"))
    if not audit.get("overall_integrity_clean"):
        log("HARD STOP: audit overall_integrity_clean is not true (19.6.9).")
        return 1
    log(f"Audit gate: overall_integrity_clean=True (generated_at={audit.get('generated_at')}).")

    # ---- Load records + structural reconciliation (pre-statistic). ----
    pilot_records = load_pilot_records()
    probe_records = load_probe_reuse_records()
    n_by_leg: dict[str, int] = defaultdict(int)
    for rec in pilot_records:
        n_by_leg[rec["leg_name"]] += 1
    n_probe_by_season: dict[str, int] = defaultdict(int)
    for rec in probe_records:
        n_probe_by_season[rec["event"]["season"]] += 1
    log(f"Pilot records by leg: {dict(n_by_leg)}; probe-reuse by season: {dict(n_probe_by_season)}.")
    if dict(n_by_leg) != EXPECTED_N_NEW or dict(n_probe_by_season) != EXPECTED_N_PROBE:
        log(f"HARD STOP: record counts do not match the registration "
            f"(expected {EXPECTED_N_NEW} new, {EXPECTED_N_PROBE} probe-reuse).")
        return 1

    all_records = [(rec, "new_purchase") for rec in pilot_records] + [
        (rec, "probe_reuse") for rec in probe_records
    ]
    event_ids = [rec["event"]["event_id"] for rec, _ in all_records]
    if len(set(event_ids)) != 170:
        log(f"HARD STOP: expected 170 distinct sampled events, found {len(set(event_ids))}.")
        return 1
    log("Sampled universe: 170 distinct events (155 new_purchase + 15 probe_reuse).")

    # ---- Clean training data + identity lookup. ----
    clean = pd.read_parquet(CLEAN_TRAINING)
    clean["game_date_str"] = pd.to_datetime(clean["game_date"]).dt.strftime("%Y-%m-%d")
    goalie_lookup = bos.build_base_lookup(CLEAN_TRAINING)
    match_cache: dict = {}

    # ---- Event table + event->game mapping. ----
    event_rows = []
    for rec, source in all_records:
        event_meta = rec["event"]
        body = json.loads(rec["raw_body"])
        data = body.get("data") or {}
        true_commence = data.get("commence_time") or event_meta["commence_time"]
        home_full = data.get("home_team") or event_meta.get("home_team")
        away_full = data.get("away_team") or event_meta.get("away_team")
        eastern_date = bos.commence_to_eastern_date(true_commence)
        game_id, map_status = map_event_to_game(clean, home_full, away_full, eastern_date)
        event_rows.append(
            {
                "event_id": event_meta["event_id"],
                "season": event_meta["season"],
                "source": source,
                "commence_time": true_commence,
                "home_team": home_full,
                "away_team": away_full,
                "eastern_date": eastern_date,
                "envelope_ts": body.get("timestamp"),
                "game_id": game_id,
                "map_status": map_status,
            }
        )
    events_df = pd.DataFrame(event_rows)
    n_unmapped = int((events_df["game_id"].isna()).sum())
    if n_unmapped:
        deviations.append(
            f"{n_unmapped} sampled events could not be mapped to a clean_training_data game "
            f"({events_df[events_df['game_id'].isna()]['map_status'].value_counts().to_dict()}); "
            "their goalie-nights are absent from the coverage denominator."
        )
        log(f"WARNING: {n_unmapped} events unmapped to games: "
            f"{events_df[events_df['game_id'].isna()][['event_id', 'map_status']].to_dict('records')}")
    # ---- Denominator: every clean goalie-night of the mapped games. ----
    mapped = events_df[events_df["game_id"].notna()].copy()
    mapped["game_id"] = mapped["game_id"].astype(int)
    dup_games = int(mapped["game_id"].duplicated().sum())
    if dup_games:
        deviations.append(f"{dup_games} sampled events map to an already-mapped game_id; "
                          "duplicates dropped deterministically (first by sorted event_id).")
        mapped = mapped.sort_values("event_id").drop_duplicates(subset=["game_id"], keep="first")
        log(f"WARNING: dropped {dup_games} duplicate-game events.")
    log(f"Event->game mapping: {len(mapped)}/{len(events_df)} events mapped.")
    denom = clean.merge(
        mapped[["game_id", "event_id", "season", "source", "commence_time", "home_team", "away_team",
                "envelope_ts"]],
        on="game_id",
        how="inner",
        suffixes=("", "_event"),
    )[
        ["game_id", "goalie_id", "goalie_name", "saves", "event_id", "season_event", "source",
         "commence_time", "home_team", "away_team", "envelope_ts"]
    ].rename(columns={"season_event": "season"})
    n_denominator = len(denom)
    log(f"Coverage denominator: {n_denominator} goalie-nights across {mapped.shape[0]} mapped events "
        f"(the registered 170-event-derived population).")

    # ---- BetOnline market extraction per event. ----
    # Ladder rows (Over only, 19.2) and same-envelope standard rows, with
    # goalie identity resolved via the archive's own match_goalie machinery.
    core_df = pd.read_parquet(CORE_BETTIME)
    core_std = core_df[
        (core_df["book_key"] == PRIMARY_BOOK) & (core_df["market_key"] == STD_MARKET)
    ].copy()
    core_resolved_ts = core_std.groupby("event_id")["resolved_ts"].agg(lambda s: sorted(set(s))).to_dict()

    ladder_rows = []  # per (event, goalie) rung rows
    std_rows = []  # same-envelope standard rows (probe_reuse + combined leg)
    n_alt_under_outcomes = 0
    n_unresolved_ladder_names = 0
    all_outcomes_rows = []  # every book, for the non-primary coverage diagnostics
    for rec, source in all_records:
        event_meta = rec["event"]
        event_id = event_meta["event_id"]
        body = json.loads(rec["raw_body"])
        data = body.get("data") or {}
        true_commence = data.get("commence_time") or event_meta["commence_time"]
        home_full = data.get("home_team") or event_meta.get("home_team")
        away_full = data.get("away_team") or event_meta.get("away_team")
        for row in flatten_book_outcomes(rec):
            row["source"] = source
            all_outcomes_rows.append(row)
            if row["book"] != PRIMARY_BOOK:
                continue
            goalie_id, goalie_name_matched = bos.match_goalie(
                true_commence, home_full, away_full, row["description"], goalie_lookup, match_cache
            )
            if row["market_key"] == ALT_MARKET:
                if row["side"] != "Over":
                    n_alt_under_outcomes += 1
                    continue
                if goalie_id is None:
                    n_unresolved_ladder_names += 1
                ladder_rows.append(
                    {
                        "event_id": event_id,
                        "source": source,
                        "season": event_meta["season"],
                        "description": row["description"],
                        "goalie_id": goalie_id,
                        "point": float(row["point"]),
                        "price_decimal": float(row["price_decimal"]),
                    }
                )
            elif row["market_key"] == STD_MARKET:
                std_rows.append(
                    {
                        "event_id": event_id,
                        "source": source,
                        "description": row["description"],
                        "goalie_id": goalie_id,
                        "side": row["side"],
                        "point": float(row["point"]),
                        "price_decimal": float(row["price_decimal"]),
                    }
                )
    ladder_df = pd.DataFrame(ladder_rows)
    std_df = pd.DataFrame(std_rows)
    log(f"BetOnline ladder rows (Over): {len(ladder_df)}; same-envelope standard rows: {len(std_df)}; "
        f"alt Under outcomes skipped: {n_alt_under_outcomes}; "
        f"unresolved ladder player names (rows): {n_unresolved_ladder_names}.")

    # Exact-duplicate guard (audit found zero; enforced here fail-loud).
    n_dup_ladder = int(ladder_df.duplicated(subset=["event_id", "description", "point"]).sum())
    if n_dup_ladder:
        log(f"HARD STOP: {n_dup_ladder} duplicate ladder (event, player, point) rows -- "
            "audit reported zero; investigate.")
        return 1

    # ---- Same-envelope standard anchors (probe_reuse + combined_2025_26). ----
    # Two-sided requirement (19.2): both sides at the same point.
    env_anchor: dict[tuple[str, Any], dict[str, Any]] = {}
    n_multi_line_anchor = 0
    if len(std_df):
        piv = std_df.pivot_table(
            index=["event_id", "description", "point"], columns="side", values="price_decimal", aggfunc="first"
        ).reset_index()
        for col in ("Over", "Under"):
            if col not in piv.columns:
                piv[col] = np.nan
        piv = piv.dropna(subset=["Over", "Under"])
        # goalie_id per (event, description) from std_df (identical strings).
        gid_map = std_df.groupby(["event_id", "description"])["goalie_id"].first().to_dict()
        for (event_id, desc), grp in piv.groupby(["event_id", "description"]):
            if len(grp) > 1:
                n_multi_line_anchor += 1
                grp = grp.sort_values("point").head(1)
            row = grp.iloc[0]
            goalie_id = gid_map.get((event_id, desc))
            key = (event_id, goalie_id if goalie_id is not None else f"name:{desc}")
            env_anchor[key] = {
                "L_std": float(row["point"]),
                "price_over": float(row["Over"]),
                "price_under": float(row["Under"]),
            }
    if n_multi_line_anchor:
        judgment_calls.append(
            f"{n_multi_line_anchor} same-envelope goalie-nights carried more than one two-sided "
            "standard line; the LOWEST line was used (18.2's modal-line tie precedent)."
        )
    log(f"Same-envelope two-sided anchors: {len(env_anchor)} goalie-nights "
        f"(multi-line anomalies: {n_multi_line_anchor}).")

    # ---- Core-parquet standard anchors (alt_only_2024_25 leg). ----
    core_anchor: dict[tuple[str, Any], dict[str, Any]] = {}
    n_multi_line_core = 0
    alt_only_event_ids = {
        rec["event"]["event_id"] for rec in pilot_records if rec["leg_name"] == "alt_only_2024_25"
    }
    core_sub = core_std[core_std["event_id"].isin(alt_only_event_ids)].copy()
    core_sub = core_sub.drop_duplicates(subset=["event_id", "player_name_raw", "side", "line", "price_decimal"])
    piv = core_sub.pivot_table(
        index=["event_id", "player_name_raw", "line"], columns="side", values="price_decimal", aggfunc="first"
    ).reset_index()
    for col in ("Over", "Under"):
        if col not in piv.columns:
            piv[col] = np.nan
    piv = piv.dropna(subset=["Over", "Under"])
    core_gid = core_sub.groupby(["event_id", "player_name_raw"])["goalie_id"].first().to_dict()
    for (event_id, name_raw), grp in piv.groupby(["event_id", "player_name_raw"]):
        if len(grp) > 1:
            n_multi_line_core += 1
            grp = grp.sort_values("line").head(1)
        row = grp.iloc[0]
        gid = core_gid.get((event_id, name_raw))
        gid = int(gid) if pd.notna(gid) else None
        key = (event_id, gid if gid is not None else f"name:{name_raw}")
        core_anchor[key] = {
            "L_std": float(row["line"]),
            "price_over": float(row["Over"]),
            "price_under": float(row["Under"]),
        }
    if n_multi_line_core:
        judgment_calls.append(
            f"{n_multi_line_core} core-parquet goalie-nights carried more than one two-sided "
            "standard line; the LOWEST line was used (18.2's modal-line tie precedent)."
        )
    log(f"Core-parquet two-sided anchors (alt-only events): {len(core_anchor)} goalie-nights "
        f"(multi-line anomalies: {n_multi_line_core}).")

    # ---- Ladder rungs grouped per (event, goalie_id). ----
    ladder_by_night: dict[tuple[str, int], pd.DataFrame] = {
        (eid, int(gid)): grp
        for (eid, gid), grp in ladder_df[ladder_df["goalie_id"].notna()].groupby(["event_id", "goalie_id"])
    }
    n_ladder_nights_market = ladder_df[ladder_df["goalie_id"].notna()][
        ["event_id", "goalie_id"]
    ].drop_duplicates().shape[0]
    log(f"BetOnline ladder goalie-nights with resolved identity: {n_ladder_nights_market}.")

    # ---- Funnel over the denominator (19.5): each goalie-night lands in
    # exactly one bucket; buckets sum to the denominator. ----
    funnel_rows = []
    qualifying_rows = []
    n_anchor_not_on_ladder = 0
    alignment_gaps = []
    for row in denom.itertuples(index=False):
        event_id = row.event_id
        goalie_id = int(row.goalie_id)
        source = row.source
        night_key = (event_id, goalie_id)
        bucket = None
        detail: dict[str, Any] = {}

        ladder = ladder_by_night.get(night_key)
        if ladder is None or ladder.empty:
            bucket = "no_resolved_identity_or_ladder"
        else:
            # Anchor: same-envelope first (probe_reuse and combined leg are
            # same-envelope by construction); core parquet for the alt-only
            # leg. A goalie-night can only have one anchor source by design.
            anchor = env_anchor.get(night_key)
            anchor_source = "same_envelope"
            if anchor is None:
                anchor = core_anchor.get(night_key)
                anchor_source = "core_parquet"
            if anchor is None:
                bucket = "no_anchor"
            else:
                # Alignment (19.3): observed timestamps both sides.
                if anchor_source == "same_envelope":
                    gap = 0.0
                else:
                    env_ts = row.envelope_ts
                    ts_list = core_resolved_ts.get(event_id)
                    if not env_ts or not ts_list:
                        gap = None
                    else:
                        gap = abs((_parse_utc(env_ts) - _parse_utc(ts_list[0])).total_seconds())
                if gap is None or gap > ALIGNMENT_TOLERANCE_SECONDS:
                    bucket = "alignment_failure"
                    detail["alignment_gap_seconds"] = gap
                else:
                    alignment_gaps.append(gap)
                    rungs = sorted(ladder["point"].unique())
                    n_rungs = len(rungs)
                    anchor_on_ladder = anchor["L_std"] in rungs
                    if not anchor_on_ladder:
                        n_anchor_not_on_ladder += 1
                    if n_rungs < RUNG_DEPTH_FLOOR:
                        bucket = "rung_depth_failure"
                        detail["n_rungs"] = n_rungs
                    else:
                        bucket = "qualifying"
                        detail = {
                            "anchor_source": anchor_source,
                            "alignment_gap_seconds": gap,
                            "n_rungs": n_rungs,
                            "anchor_on_ladder": anchor_on_ladder,
                            "L_std": anchor["L_std"],
                            "price_over_std": anchor["price_over"],
                            "price_under_std": anchor["price_under"],
                        }
        funnel_rows.append(
            {
                "event_id": event_id,
                "game_id": int(row.game_id),
                "goalie_id": goalie_id,
                "goalie_name": row.goalie_name,
                "season": row.season,
                "source": source,
                "saves": int(row.saves),
                "bucket": bucket,
                **{k: v for k, v in detail.items() if k in ("alignment_gap_seconds", "n_rungs")},
            }
        )
        if bucket == "qualifying":
            qualifying_rows.append(
                {
                    "event_id": event_id,
                    "game_id": int(row.game_id),
                    "goalie_id": goalie_id,
                    "season": row.season,
                    "source": source,
                    "saves": int(row.saves),
                    **detail,
                }
            )

    funnel_df = pd.DataFrame(funnel_rows)
    funnel_counts = {str(k): int(v) for k, v in funnel_df["bucket"].value_counts().items()}
    n_qualifying = int(funnel_counts.get("qualifying", 0))
    if int(sum(funnel_counts.values())) != n_denominator:
        log("HARD STOP: funnel does not reconcile to the denominator.")
        return 1
    log(f"Exclusion funnel (sums to {n_denominator}): {funnel_counts}")
    log(f"Anchor-not-on-ladder anomalies (19.2 flag): {n_anchor_not_on_ladder}")

    # Market-side ladder nights that matched no denominator night (diagnostic).
    denom_keys = {(r.event_id, int(r.goalie_id)) for r in denom.itertuples(index=False)}
    market_only = [k for k in ladder_by_night if k not in denom_keys]
    log(f"Ladder goalie-nights with no clean_training_data denominator row (diagnostic, excluded): "
        f"{len(market_only)}")

    # ---- sigma_0 fit (19.4) with its two-path wiring gate. ----
    sigma_0, sigma_fit_frame, sigma_gate = fit_sigma_0(clean, log)
    if sigma_0 is None or not sigma_gate.get("gate_pass"):
        log("HARD STOP: sigma_0 wiring gate failed or fit undefined -- investigate before any statistic.")
        return 1
    baseline_defined = sigma_0 > 0
    if not baseline_defined:
        log(f"sigma_0 = {sigma_0!r} <= 0: baseline UNDEFINED; calibration will be INSUFFICIENT SAMPLE (19.4).")

    # ---- Rung-level paired frame over qualifying nights. ----
    rung_rows = []
    total_clips = 0
    nights_with_clip = 0
    anchor_consistency_diffs = []
    for q in qualifying_rows:
        night_key = (q["event_id"], q["goalie_id"])
        ladder = ladder_by_night[night_key]
        rungs = ladder.groupby("point")["price_decimal"].first().sort_index()
        overround_std = 1.0 / q["price_over_std"] + 1.0 / q["price_under_std"]
        p_over_std = (1.0 / q["price_over_std"]) / overround_std
        raw_p = 1.0 / rungs.values
        devig_raw = raw_p / overround_std
        # Forward-min monotonicity clip (19.4), rungs ascending.
        devig = np.empty_like(devig_raw)
        n_clip = 0
        for i in range(len(devig_raw)):
            if i == 0:
                devig[i] = devig_raw[i]
            else:
                if devig_raw[i] > devig[i - 1]:
                    n_clip += 1
                    devig[i] = devig[i - 1]
                else:
                    devig[i] = devig_raw[i]
        total_clips += n_clip
        if n_clip:
            nights_with_clip += 1
        if q["anchor_on_ladder"]:
            idx = list(rungs.index).index(q["L_std"])
            anchor_consistency_diffs.append(float(devig_raw[idx] - p_over_std))
        if baseline_defined:
            mu_hat = q["L_std"] + 0.5 - sigma_0 * norm.ppf(1.0 - p_over_std)
        else:
            mu_hat = np.nan
        cluster_id = f"{q['event_id']}_{q['goalie_id']}"
        for i, (L_i, price) in enumerate(rungs.items()):
            baseline_p = (
                float(1.0 - norm.cdf((L_i + 0.5 - mu_hat) / sigma_0)) if baseline_defined else np.nan
            )
            rung_rows.append(
                {
                    "event_id": q["event_id"],
                    "game_id": q["game_id"],
                    "goalie_id": q["goalie_id"],
                    "cluster_id": cluster_id,
                    "season": q["season"],
                    "source": q["source"],
                    "L_std": q["L_std"],
                    "L_i": float(L_i),
                    "is_anchor_rung": bool(L_i == q["L_std"]),
                    "price_decimal_over": float(price),
                    "raw_p_over": float(1.0 / price),
                    "overround_std": float(overround_std),
                    "p_over_devigged_raw": float(devig_raw[i]),
                    "p_over_devigged": float(devig[i]),
                    "p_over_std_anchor": float(p_over_std),
                    "mu_hat": float(mu_hat) if baseline_defined else np.nan,
                    "baseline_p_over": baseline_p,
                    "saves_actual": q["saves"],
                    "label_over": int(q["saves"] > L_i),
                    "n_clips_this_night": n_clip,
                }
            )
    rung_df = pd.DataFrame(rung_rows)
    log(f"Rung-level frame: {len(rung_df)} rows over {n_qualifying} qualifying nights; "
        f"monotonicity clips: {total_clips} rungs across {nights_with_clip} nights.")
    if anchor_consistency_diffs:
        acd = np.array(anchor_consistency_diffs)
        anchor_consistency = {
            "n_nights_with_anchor_on_ladder": int(len(acd)),
            "mean_diff": float(acd.mean()),
            "median_diff": float(np.median(acd)),
            "max_abs_diff": float(np.abs(acd).max()),
        }
    else:
        anchor_consistency = {"n_nights_with_anchor_on_ladder": 0}
    log(f"Anchor-rung consistency diagnostic (ladder devig_raw at L_std minus standard devig): "
        f"{anchor_consistency}")

    # ---- Non-gating coverage diagnostics (19.5). ----
    all_outcomes = pd.DataFrame(all_outcomes_rows)
    alt_all = all_outcomes[
        (all_outcomes["market_key"] == ALT_MARKET) & (all_outcomes["side"] == "Over")
    ]
    per_book_rungs = []
    for (book, season), grp in alt_all.groupby(["book", "season"]):
        rc = grp.groupby(["event_id", "description"])["point"].nunique()
        per_book_rungs.append(
            {
                "book": book,
                "season": season,
                "n_goalie_nights": int(len(rc)),
                "rungs_min": int(rc.min()),
                "rungs_median": float(rc.median()),
                "rungs_mean": float(rc.mean()),
                "rungs_max": int(rc.max()),
            }
        )
    qual_rung_dist = []
    if n_qualifying:
        qdf = pd.DataFrame(qualifying_rows)
        for (season, source), grp in qdf.groupby(["season", "source"]):
            qual_rung_dist.append(
                {
                    "season": season,
                    "source": source,
                    "n_nights": int(len(grp)),
                    "rungs_min": int(grp["n_rungs"].min()),
                    "rungs_median": float(grp["n_rungs"].median()),
                    "rungs_mean": float(grp["n_rungs"].mean()),
                    "rungs_max": int(grp["n_rungs"].max()),
                }
            )
    log(f"Rungs per qualifying night by season x source: {qual_rung_dist}")
    log(f"Per-book alternate-saves coverage (all books, non-gating): {per_book_rungs}")

    # =====================================================================
    # REGISTERED STATISTICS (19.5). From here on, numbers stand (19.6.10).
    # =====================================================================

    # Step 1: zero qualifying -> INSUFFICIENT SAMPLE, no bootstrap.
    coverage_rate = n_qualifying / n_denominator if n_denominator else 0.0
    log("")
    log("=" * 78)
    log("REGISTERED 19.5 STATISTICS")
    log("=" * 78)
    log(f"n_total_sampled_goalie_nights (denominator) = {n_denominator}")
    log(f"n_qualifying = {n_qualifying}")
    log(f"coverage_rate = {coverage_rate!r}")

    primary: dict[str, Any] = {}
    secondary: dict[str, Any] = {}
    verdict = None
    coverage_status = None
    calibration_label = None

    if n_qualifying == 0:
        verdict = "INSUFFICIENT SAMPLE"
        coverage_status = "INSUFFICIENT"
        calibration_label = "NOT COMPUTED (zero qualifying nights, 19.5 step 1)"
        log("19.5 step 1: zero qualifying goalie-nights -> INSUFFICIENT SAMPLE, no bootstrap attempted.")
    else:
        coverage_status = "SUFFICIENT" if coverage_rate >= COVERAGE_GATE else "INSUFFICIENT"
        log(f"Coverage gate (>= {COVERAGE_GATE}): {coverage_status}")

        if not baseline_defined:
            verdict = "INSUFFICIENT SAMPLE"
            calibration_label = "NOT COMPUTED (sigma_0 <= 0, baseline undefined, 19.4)"
            log("sigma_0 <= 0 -> calibration INSUFFICIENT SAMPLE (19.4).")
        else:
            # Calibration population: non-anchor rungs of qualifying nights.
            pop = rung_df[~rung_df["is_anchor_rung"]].copy()
            n_rows = len(pop)
            clusters = pop["cluster_id"].unique()
            n_clusters = len(clusters)
            log(f"Calibration population: {n_rows} (goalie-night, non-anchor rung) rows, "
                f"{n_clusters} goalie-night clusters.")

            y = pop["label_over"].to_numpy(dtype=float)
            p_lad = pop["p_over_devigged"].to_numpy(dtype=float)
            p_base = pop["baseline_p_over"].to_numpy(dtype=float)

            sq_lad = (p_lad - y) ** 2
            sq_base = (p_base - y) ** 2
            p_lad_c = np.clip(p_lad, LOGLOSS_CLIP, 1.0 - LOGLOSS_CLIP)
            p_base_c = np.clip(p_base, LOGLOSS_CLIP, 1.0 - LOGLOSS_CLIP)
            ll_lad = -(y * np.log(p_lad_c) + (1.0 - y) * np.log(1.0 - p_lad_c))
            ll_base = -(y * np.log(p_base_c) + (1.0 - y) * np.log(1.0 - p_base_c))

            brier_lad = float(sq_lad.mean())
            brier_base = float(sq_base.mean())
            delta_brier = brier_lad - brier_base
            logloss_lad = float(ll_lad.mean())
            logloss_base = float(ll_base.mean())
            delta_logloss = logloss_lad - logloss_base

            # Per-cluster aggregates for the cluster bootstrap.
            cl_index = {c: i for i, c in enumerate(clusters)}
            cl_ids = pop["cluster_id"].map(cl_index).to_numpy()
            K = n_clusters
            n_c = np.bincount(cl_ids, minlength=K).astype(float)
            s_lad = np.bincount(cl_ids, weights=sq_lad, minlength=K)
            s_base = np.bincount(cl_ids, weights=sq_base, minlength=K)
            l_lad = np.bincount(cl_ids, weights=ll_lad, minlength=K)
            l_base = np.bincount(cl_ids, weights=ll_base, minlength=K)

            rng = np.random.default_rng(BOOTSTRAP_SEED)
            draw = rng.integers(0, K, size=(N_RESAMPLES, K))
            tot_n = n_c[draw].sum(axis=1)
            d_brier = (s_lad[draw].sum(axis=1) - s_base[draw].sum(axis=1)) / tot_n
            d_logloss = (l_lad[draw].sum(axis=1) - l_base[draw].sum(axis=1)) / tot_n
            ci_brier = np.percentile(d_brier, [2.5, 97.5])
            ci_logloss = np.percentile(d_logloss, [2.5, 97.5])
            ci_lower, ci_upper = float(ci_brier[0]), float(ci_brier[1])

            primary = {
                "metric": "rung_level_brier_delta_ladder_minus_baseline",
                "n_rows": int(n_rows),
                "n_clusters": int(n_clusters),
                "brier_ladder": brier_lad,
                "brier_baseline": brier_base,
                "delta": delta_brier,
                "ci95_lower": ci_lower,
                "ci95_upper": ci_upper,
                "n_resamples": N_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
            }
            secondary = {
                "metric": "rung_level_logloss_delta_ladder_minus_baseline",
                "logloss_ladder": logloss_lad,
                "logloss_baseline": logloss_base,
                "delta": delta_logloss,
                "ci95_lower": float(ci_logloss[0]),
                "ci95_upper": float(ci_logloss[1]),
                "clip_bound": LOGLOSS_CLIP,
            }
            log(f"PRIMARY Brier: ladder={brier_lad!r} baseline={brier_base!r} delta={delta_brier!r}")
            log(f"PRIMARY CI95 (percentile, {N_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}): "
                f"[{ci_lower!r}, {ci_upper!r}]")
            log(f"SECONDARY log-loss: ladder={logloss_lad!r} baseline={logloss_base!r} "
                f"delta={delta_logloss!r} CI95=[{ci_logloss[0]!r}, {ci_logloss[1]!r}]")

            # 19.5's exact PASS/FAIL/INSUFFICIENT arithmetic.
            if coverage_status == "INSUFFICIENT":
                verdict = "INSUFFICIENT SAMPLE"
                calibration_label = "EXPLORATORY-ONLY (coverage gate failed, 19.5 step 2)"
                log("19.5 step 2: coverage_rate < 0.70 -> calibration computed but EXPLORATORY-ONLY; "
                    "overall INSUFFICIENT SAMPLE.")
            elif n_qualifying < CLUSTER_FLOOR:
                verdict = "INSUFFICIENT SAMPLE"
                calibration_label = "POINT ESTIMATES ONLY (cluster floor not met, 19.5 step 3; CI not trusted)"
                log("19.5 step 3: n_qualifying < 50 -> INSUFFICIENT SAMPLE; point estimates reported, "
                    "no CI trusted.")
            else:
                # Registered literally as ci_upper < 0 (19.5's pass bar).
                pilot_pass = ci_upper < 0
                verdict = "PILOT PASS" if pilot_pass else "PILOT FAIL"
                calibration_label = "GATING"
                log(f"19.5 step 4: ci_upper = {ci_upper!r}; PASS requires ci_upper < 0 -> {verdict}.")

    log("")
    log(f"OVERALL VERDICT: {verdict}")

    # ---- Persist artifacts. ----
    rung_df.to_parquet(output_dir / "rung_level_paired_frame.parquet", index=False)
    funnel_df.to_parquet(output_dir / "exclusion_funnel_frame.parquet", index=False)
    sigma_fit_frame.to_parquet(output_dir / "sigma0_fit_inputs.parquet", index=False)
    for leg in ("alt_only_2024_25", "combined_2025_26"):
        shutil.copy2(PILOT_CACHE / f"plan_{leg}.json", output_dir / f"plan_{leg}.json")

    checksums = {
        "clean_training_data.parquet": sha256_file(CLEAN_TRAINING),
        "core_bettime_202607_snapshots.parquet": sha256_file(CORE_BETTIME),
        "saves_lines_snapshots.parquet": sha256_file(SAVES_SNAPSHOTS),
        "exp15_juice_goalie_night_features.parquet": sha256_file(EXP15_JUICE),
        "audit_summary.json": sha256_file(AUDIT_SUMMARY),
        "plan_alt_only_2024_25.json": sha256_file(PILOT_CACHE / "plan_alt_only_2024_25.json"),
        "plan_combined_2025_26.json": sha256_file(PILOT_CACHE / "plan_combined_2025_26.json"),
        "pilot_records_aggregate": sha256_record_dir(PILOT_CACHE, "altladder_event=*.json"),
        "probe_records_aggregate": sha256_record_dir(PROBE_DIR, "w1_event=*.json"),
    }

    metadata = {
        "experiment": "experiment_16_alt_ladder_pilot",
        "registration": "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 19",
        "run_timestamp_utc": timestamp,
        "audit_gate": {
            "audit_summary_generated_at": audit.get("generated_at"),
            "overall_integrity_clean": True,
        },
        "universe": {
            "n_sampled_events": 170,
            "n_new_purchase_events": 155,
            "n_probe_reuse_events": 15,
            "n_events_mapped_to_games": int(len(mapped)),
            "n_events_unmapped": n_unmapped,
            "n_total_sampled_goalie_nights": n_denominator,
            "n_qualifying": n_qualifying,
            "coverage_rate": coverage_rate,
            "coverage_gate_bar": COVERAGE_GATE,
            "coverage_status": coverage_status,
            "cluster_floor": CLUSTER_FLOOR,
            "exclusion_funnel": funnel_counts,
            "funnel_reconciles_to_denominator": True,
            "n_anchor_not_on_ladder_anomalies": n_anchor_not_on_ladder,
            "n_market_ladder_nights_without_denominator_row": len(market_only),
            "alignment_gap_seconds_max_among_qualifying": (
                float(np.max(alignment_gaps)) if alignment_gaps else None
            ),
        },
        "sigma_0": {
            "value": sigma_0,
            "wiring_gate": sigma_gate,
            "baseline_defined": bool(baseline_defined),
        },
        "primary": primary,
        "secondary": secondary,
        "monotonicity_clips": {
            "total_rungs_clipped": int(total_clips),
            "nights_with_at_least_one_clip": int(nights_with_clip),
        },
        "anchor_rung_consistency": anchor_consistency,
        "coverage_diagnostics": {
            "qualifying_rung_distribution_by_season_source": qual_rung_dist,
            "per_book_alt_coverage_all_books": per_book_rungs,
        },
        "verdict": verdict,
        "calibration_label": calibration_label,
        "pass_bar": "ci_upper < 0 (19.5, implemented literally)",
        "deviations_from_registration": deviations,
        "judgment_calls": judgment_calls,
        "input_checksums": checksums,
        "honesty_notes": (
            "Per 19.1: both seasons are development evidence on already-viewed data; 2025-26 is "
            "simultaneously the discovery and live-bet season; any PASS authorizes only DRAFTING a "
            "full-season purchase registration, never a purchase or a betting-policy change."
        ),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")
    log(f"\nWrote metadata.json and artifacts to {output_dir}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
