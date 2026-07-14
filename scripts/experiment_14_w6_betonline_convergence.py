"""Experiment 14: W6 BetOnline bettime-to-close convergence.

Implements section 17 of docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md exactly.
That section is the binding contract for every definition, phase spec, and
forbidden rule used here; this docstring restates the shape of the run, not
the rules themselves.

Read-only inputs (no writes to any existing file, no network calls, no
touching data/betting.db -- forbidden even for reads per 17.6.2):
  - data/processed/saves_lines_snapshots.parquet (Phase A's sole source,
    and Phase B's closing-side source). It carries no season column; season
    is derived per game_date_eastern via build_odds_snapshots.py's own
    season_from_eastern_date helper, imported rather than reimplemented.
  - data/processed/core_bettime_202607_snapshots.parquet (Phase B and
    17.5's bettime source, pass_name == "combined-2024-25").
  - models/trained/experiment_11_frozen_origin_b_p2_20260714_090012/
    p2_primary_betonlineag_universe.parquet and flagged_bets_with_clv.parquet
    (17.5's frozen, read-only, unmodified reuse).

Registered computation order (17.6.8): structural reconciliation against
17.8's persisted counts happens first and hard-stops on any mismatch: none
of Phase A, Phase B, or 17.5's statistics are computed before every check
passes. Then Phase A is fully built and its statistic computed and logged;
then Phase B; then 17.5. Once a phase's statistic is logged it stands.

Output: a new, timestamped artifact directory under models/trained/ holding
row-level universes (parquet), metadata.json (every registered statistic,
CI, count, and verdict), input_checksums.json, and run_log.txt.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_odds_snapshots import season_from_eastern_date  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (read-only inputs; data/betting.db is never referenced, per 17.6.2)
# ---------------------------------------------------------------------------
SAVES_SNAPSHOTS = REPO_ROOT / "data" / "processed" / "saves_lines_snapshots.parquet"
CORE_BETTIME = REPO_ROOT / "data" / "processed" / "core_bettime_202607_snapshots.parquet"
EXP11_DIR = REPO_ROOT / "models" / "trained" / "experiment_11_frozen_origin_b_p2_20260714_090012"
EXP11_UNIVERSE = EXP11_DIR / "p2_primary_betonlineag_universe.parquet"
EXP11_CLV = EXP11_DIR / "flagged_bets_with_clv.parquet"

OUTPUT_ROOT = REPO_ROOT / "models" / "trained"
EXPERIMENT_PREFIX = "experiment_14_w6_betonline_convergence_"

# ---------------------------------------------------------------------------
# Registered constants (17.2 / 17.3 / 17.4 / 17.5)
# ---------------------------------------------------------------------------
DFS_BOOKS = {"prizepicks", "underdog"}
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
DEGENERATE_UNSTABLE_PCT = 1.0
MIN_ARM_BETS = 100
MIN_OTHER_BOOKS = 2
DRIFT_MIN_GAP_MINUTES = 10  # section 14.5 rule 4, applied by reference to the new pass

# 17.8's persisted structural counts. Hard-stop on any mismatch (per the
# task's two-stage wiring discipline) before any Phase A/B/17.5 statistic.
EXPECTED_COUNTS = {
    "betonlineag_bettime_2025_26_raw_rows": 2662,
    "betonlineag_bettime_2025_26_deduped_rows": 2474,
    "joined_bettime_closing_pairs_total": 2063,
    "joined_bettime_closing_pairs_line_identical": 1927,
    "joined_bettime_closing_pairs_line_changed": 136,
    "joined_unique_goalie_nights_any_side": 1032,
    "new_pass_betonlineag_saves_rows": 3498,
    "new_pass_betonlineag_saves_events": 1050,
    "new_pass_betonlineag_within_pass_duplicate_groups": 0,
    "exp11_universe_rows": 1719,
    "exp11_model_arm_rows": 473,
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def make_logger(path: Path):
    def log(message: str) -> None:
        stamped = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(stamped)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")

    return log


def no_completed_run_exists() -> None:
    """17.6.8: one registered execution. Refuse to start if a completed run
    (all three phases' statistics logged) already exists."""
    completed = []
    for candidate in OUTPUT_ROOT.glob(f"{EXPERIMENT_PREFIX}*/metadata.json"):
        try:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if metadata.get("run_status") == "completed":
            completed.append(str(candidate.parent))
    if completed:
        raise SystemExit(
            "A completed Experiment 14 run already exists. Section 17.6.8 forbids rerunning after "
            f"every phase's statistic has been computed and logged: {completed}"
        )


# ---------------------------------------------------------------------------
# 17.2 dedup rule: within each snapshot_pass, keep the row with max
# resolved_ts per natural key; ties broken by max requested_ts; remaining
# ties broken deterministically by original row order. Tie counts logged.
# ---------------------------------------------------------------------------
def dedup_max_resolved_ts(frame: pd.DataFrame, key_cols: list[str], pass_col: str = "snapshot_pass") -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    frame["_orig_order"] = np.arange(len(frame))
    frame["_resolved_ts_dt"] = pd.to_datetime(frame["resolved_ts"], utc=True, errors="coerce")
    frame["_requested_ts_dt"] = pd.to_datetime(frame["requested_ts"], utc=True, errors="coerce")
    frame["_group_key"] = list(zip(*[frame[c] for c in key_cols]))

    out_parts = []
    groups_with_duplicates = 0
    extra_rows_dropped = 0
    row_order_tiebreak_groups = 0

    for _, g in frame.groupby(pass_col, sort=False):
        g = g.copy()
        sizes = g.groupby("_group_key")["_group_key"].transform("size")
        groups_with_duplicates += int((sizes[~g.duplicated("_group_key")] > 1).sum())

        g["_max_resolved"] = g.groupby("_group_key")["_resolved_ts_dt"].transform("max")
        at_max_resolved = g[g["_resolved_ts_dt"] == g["_max_resolved"]].copy()
        at_max_resolved["_max_requested"] = at_max_resolved.groupby("_group_key")["_requested_ts_dt"].transform("max")
        at_max_both = at_max_resolved[at_max_resolved["_requested_ts_dt"] == at_max_resolved["_max_requested"]].copy()

        tie_sizes = at_max_both.groupby("_group_key").size()
        row_order_tiebreak_groups += int((tie_sizes > 1).sum())

        keep_idx = at_max_both.groupby("_group_key")["_orig_order"].idxmin()
        deduped = g.loc[keep_idx]
        extra_rows_dropped += len(g) - len(deduped)
        out_parts.append(deduped)

    result = pd.concat(out_parts, ignore_index=True)
    drop_cols = ["_orig_order", "_resolved_ts_dt", "_requested_ts_dt", "_group_key", "_max_resolved", "_max_requested"]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])
    stats = {
        "groups_with_duplicates": groups_with_duplicates,
        "extra_rows_dropped": extra_rows_dropped,
        "row_order_tiebreak_groups": row_order_tiebreak_groups,
        "rows_before": int(len(frame)),
        "rows_after": int(len(result)),
    }
    return result, stats


def build_paired_devig(frame: pd.DataFrame, name_col: str, book_col: str) -> pd.DataFrame:
    """17.2 de-vig method: proportional normalization of a book's own paired
    decimal prices at the SAME line/pass/goalie-night. A book contributes a
    de-vigged probability only when BOTH sides of that exact line are
    present for that book at that goalie-night."""
    pivot = frame.pivot_table(
        index=["event_id", name_col, book_col, "line"],
        columns="side", values="price_decimal", aggfunc="first",
    )
    for side in ("over", "under"):
        if side not in pivot.columns:
            pivot[side] = np.nan
    pivot = pivot.dropna(subset=["over", "under"]).reset_index()
    raw_p_over = 1.0 / pivot["over"]
    raw_p_under = 1.0 / pivot["under"]
    overround = raw_p_over + raw_p_under
    pivot["p_over_devigged"] = raw_p_over / overround
    pivot["p_under_devigged"] = 1.0 - pivot["p_over_devigged"]
    return pivot


def compute_goalie_key(goalie_id: pd.Series, goalie_name_matched: pd.Series, raw_name: pd.Series) -> pd.Series:
    """17.4's cross-parquet goalie identity: goalie_id where resolved,
    otherwise goalie_name_matched/goalie_name_raw fallback (in that
    priority order). Applied uniformly here for within-file grouping too,
    verified (see run notes) to reproduce 17.8's structural counts
    identically to a goalie_name_raw-only key."""
    gid = goalie_id.astype("Int64")
    fallback_name = goalie_name_matched.fillna(raw_name).astype(str).str.strip().str.lower()
    key = np.where(gid.notna(), "gid:" + gid.astype(str), "name:" + fallback_name)
    return pd.Series(key, index=goalie_id.index)


# ---------------------------------------------------------------------------
# Cluster bootstrap primitives
# ---------------------------------------------------------------------------
def cluster_bootstrap_pearson_r(dev: np.ndarray, rev: np.ndarray, cluster_ids: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    dev = np.asarray(dev, dtype=float)
    rev = np.asarray(rev, dtype=float)
    cluster_ids = np.asarray(cluster_ids, dtype=object)
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(clusters)
    counts = np.bincount(inverse)
    one_to_one = bool(np.all(counts == 1))

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))

    if one_to_one:
        row_for_cluster = np.empty(n_clusters, dtype=int)
        row_for_cluster[inverse] = np.arange(len(dev))
        d = dev[row_for_cluster[draws]]
        v = rev[row_for_cluster[draws]]
        d_mean = d.mean(axis=1, keepdims=True)
        v_mean = v.mean(axis=1, keepdims=True)
        d_dev = d - d_mean
        v_dev = v - v_mean
        num = (d_dev * v_dev).sum(axis=1)
        d_ss = (d_dev ** 2).sum(axis=1)
        v_ss = (v_dev ** 2).sum(axis=1)
        degenerate_mask = (d_ss == 0) | (v_ss == 0)
        denom = np.sqrt(d_ss * v_ss)
        with np.errstate(invalid="ignore", divide="ignore"):
            rs = num / denom
        rs = np.where(degenerate_mask, np.nan, rs)
    else:
        cluster_rows = [np.where(inverse == i)[0] for i in range(n_clusters)]
        rs = np.empty(n_boot)
        degenerate_mask = np.zeros(n_boot, dtype=bool)
        for b in range(n_boot):
            rows = np.concatenate([cluster_rows[c] for c in draws[b]])
            d, v = dev[rows], rev[rows]
            if d.std() == 0 or v.std() == 0:
                degenerate_mask[b] = True
                rs[b] = np.nan
            else:
                rs[b] = np.corrcoef(d, v)[0, 1]

    n_degenerate = int(degenerate_mask.sum())
    valid = rs[~np.isnan(rs)]
    point_r = float(np.corrcoef(dev, rev)[0, 1]) if dev.std() > 0 and rev.std() > 0 else None
    ci_lower = float(np.percentile(valid, 2.5)) if len(valid) else None
    ci_upper = float(np.percentile(valid, 97.5)) if len(valid) else None
    degenerate_pct = 100.0 * n_degenerate / n_boot
    return {
        "point_r": point_r,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "n_goalie_nights": int(len(dev)),
        "n_clusters": n_clusters,
        "clusters_one_to_one_with_rows": one_to_one,
        "n_bootstrap": n_boot,
        "seed": seed,
        "n_degenerate_resamples": n_degenerate,
        "pct_degenerate_resamples": degenerate_pct,
        "unstable": degenerate_pct > DEGENERATE_UNSTABLE_PCT,
        "pass_bar_ci95_entirely_below_zero": (ci_upper is not None and ci_upper < 0),
    }


def ols_slope_secondary(dev: np.ndarray, rev: np.ndarray) -> dict[str, Any]:
    dev = np.asarray(dev, dtype=float)
    rev = np.asarray(rev, dtype=float)
    d_mean, r_mean = dev.mean(), rev.mean()
    d_dev = dev - d_mean
    denom = float((d_dev ** 2).sum())
    if denom == 0:
        return {"slope": None, "intercept": None}
    slope = float((d_dev * (rev - r_mean)).sum() / denom)
    intercept = float(r_mean - slope * d_mean)
    return {"slope": slope, "intercept": intercept}


def bootstrap_roi_delta_vs_fixed_reference(arm_values: np.ndarray, fixed_reference: float, n_boot: int, seed: int) -> dict[str, Any]:
    """17.5 PRIMARY/SECONDARY construction: resample ONLY the arm's own
    cluster values with replacement; the full-473 reference ROI is a fixed
    constant (not itself resampled) subtracted from each resampled arm
    mean."""
    arm_values = np.asarray(arm_values, dtype=float)
    n = len(arm_values)
    if n == 0:
        return {"n": 0, "point_roi": None, "point_delta": None, "ci95_lower": None, "ci95_upper": None,
                "reference_roi_fixed": float(fixed_reference), "n_bootstrap": n_boot, "seed": seed}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arm_values[idx].mean(axis=1)
    deltas = means - fixed_reference
    return {
        "n": n,
        "point_roi": float(arm_values.mean()),
        "reference_roi_fixed": float(fixed_reference),
        "point_delta": float(arm_values.mean() - fixed_reference),
        "ci95_lower": float(np.percentile(deltas, 2.5)),
        "ci95_upper": float(np.percentile(deltas, 97.5)),
        "n_bootstrap": n_boot,
        "seed": seed,
    }


def arm_verdict(n: int, ci_lower: float | None) -> str:
    if n < MIN_ARM_BETS:
        return "INSUFFICIENT_SAMPLE"
    if ci_lower is not None and ci_lower > 0:
        return "PASS"
    return "FAIL"


# ---------------------------------------------------------------------------
# Phase A: 2025-26 clustered re-derivation (17.3)
# ---------------------------------------------------------------------------
def build_phase_a(saves_deduped_all: pd.DataFrame, log) -> dict[str, Any]:
    df = saves_deduped_all[saves_deduped_all["season"] == "2025-26"].copy()
    bettime = df[df["snapshot_pass"] == "bettime"]
    closing = df[df["snapshot_pass"] == "closing"]

    bettime_paired = build_paired_devig(bettime, "goalie_name_raw", "book")
    closing_paired = build_paired_devig(closing, "goalie_name_raw", "book")

    bo_bt = bettime_paired[bettime_paired["book"] == "betonlineag"].copy()
    bo_cl = closing_paired[closing_paired["book"] == "betonlineag"].copy()
    other_bt = bettime_paired[
        (bettime_paired["book"] != "betonlineag") & (~bettime_paired["book"].isin(DFS_BOOKS))
    ].copy()
    log(f"Phase A: betonlineag bettime paired quotes={len(bo_bt)}, betonlineag closing paired quotes={len(bo_cl)}, "
        f"other-book bettime paired quotes={len(other_bt)} across books={sorted(other_bt['book'].unique().tolist())}.")

    goalie_lookup = df[["event_id", "goalie_name_raw", "goalie_id", "goalie_name_matched"]].drop_duplicates()
    bo_bt = bo_bt.merge(goalie_lookup, on=["event_id", "goalie_name_raw"], how="left")
    bo_bt["goalie_key"] = compute_goalie_key(bo_bt["goalie_id"], bo_bt["goalie_name_matched"], bo_bt["goalie_name_raw"])
    bo_cl = bo_cl.merge(goalie_lookup, on=["event_id", "goalie_name_raw"], how="left")
    bo_cl["goalie_key"] = compute_goalie_key(bo_cl["goalie_id"], bo_cl["goalie_name_matched"], bo_cl["goalie_name_raw"])
    other_bt = other_bt.merge(goalie_lookup, on=["event_id", "goalie_name_raw"], how="left")
    other_bt["goalie_key"] = compute_goalie_key(other_bt["goalie_id"], other_bt["goalie_name_matched"], other_bt["goalie_name_raw"])

    consensus = (
        other_bt.groupby(["event_id", "goalie_key", "line"])
        .agg(consensus_p_under_bettime=("p_under_devigged", "median"),
             n_other_qualifying_books_bettime=("book", "nunique"),
             other_books_bettime=("book", lambda s: ",".join(sorted(set(s)))))
        .reset_index()
    )

    universe = bo_bt.merge(consensus, on=["event_id", "goalie_key", "line"], how="left")
    universe["n_other_qualifying_books_bettime"] = universe["n_other_qualifying_books_bettime"].fillna(0).astype(int)
    universe["gate_passed"] = universe["n_other_qualifying_books_bettime"] >= MIN_OTHER_BOOKS
    universe["deviation_under"] = np.where(universe["gate_passed"], universe["p_under_devigged"] - universe["consensus_p_under_bettime"], np.nan)

    closing_small = bo_cl[["event_id", "goalie_key", "line", "p_under_devigged"]].rename(
        columns={"line": "closing_line", "p_under_devigged": "p_under_devigged_closing"}
    )
    universe = universe.merge(closing_small, on=["event_id", "goalie_key"], how="left")
    universe["has_closing_quote_any_line"] = universe["closing_line"].notna()
    universe["line_identical"] = np.where(universe["has_closing_quote_any_line"], universe["line"] == universe["closing_line"], np.nan)
    universe["reversion_under"] = np.where(
        universe["gate_passed"] & (universe["line_identical"] == True),  # noqa: E712
        universe["p_under_devigged_closing"] - universe["p_under_devigged"], np.nan,
    )
    universe["included_in_primary"] = universe["gate_passed"] & (universe["line_identical"] == True)  # noqa: E712
    universe["cluster_id"] = universe["event_id"].astype(str) + "::" + universe["goalie_key"].astype(str)
    universe = universe.rename(columns={"line": "bettime_line", "p_under_devigged": "p_under_devigged_bettime", "p_over_devigged": "p_over_devigged_bettime"})

    n_total = len(universe)
    n_gate_excluded = int((~universe["gate_passed"]).sum())
    n_no_closing = int((universe["gate_passed"] & ~universe["has_closing_quote_any_line"]).sum())
    n_line_changed = int((universe["gate_passed"] & universe["has_closing_quote_any_line"] & (universe["line_identical"] == False)).sum())  # noqa: E712
    n_primary = int(universe["included_in_primary"].sum())
    assert n_gate_excluded + n_no_closing + n_line_changed + n_primary == n_total, "Phase A exclusion funnel does not sum to total"

    primary = universe[universe["included_in_primary"]].copy()
    bootstrap = cluster_bootstrap_pearson_r(
        primary["deviation_under"].to_numpy(), primary["reversion_under"].to_numpy(), primary["cluster_id"].to_numpy(), N_BOOTSTRAP, BOOTSTRAP_SEED,
    )
    bootstrap_repeat = cluster_bootstrap_pearson_r(
        primary["deviation_under"].to_numpy(), primary["reversion_under"].to_numpy(), primary["cluster_id"].to_numpy(), N_BOOTSTRAP, BOOTSTRAP_SEED,
    )
    determinism_ok = bootstrap == bootstrap_repeat
    ols = ols_slope_secondary(primary["deviation_under"].to_numpy(), primary["reversion_under"].to_numpy())

    if bootstrap["unstable"]:
        verdict = "UNSTABLE"
    elif bootstrap["pass_bar_ci95_entirely_below_zero"]:
        verdict = "PASS"
    else:
        verdict = "CLOSED"

    log(f"PHASE A STATISTIC (FINAL, REGISTERED): r={bootstrap['point_r']}, "
        f"CI95=[{bootstrap['ci95_lower']}, {bootstrap['ci95_upper']}], n={bootstrap['n_goalie_nights']}, "
        f"degenerate={bootstrap['n_degenerate_resamples']} ({bootstrap['pct_degenerate_resamples']:.4f}%), verdict={verdict}.")
    log(f"Phase A exclusions: gate={n_gate_excluded}, no_closing_data={n_no_closing}, line_changed={n_line_changed}, primary={n_primary}, total={n_total}.")
    log(f"Phase A bootstrap determinism re-check (same seed, same input): identical={determinism_ok}.")

    return {
        "universe": universe,
        "primary": primary,
        "bootstrap": bootstrap,
        "ols": ols,
        "verdict": verdict,
        "determinism_verified": determinism_ok,
        "n_total_betonlineag_bettime_paired_quotes": n_total,
        "n_excluded_fewer_than_2_books_gate": n_gate_excluded,
        "n_excluded_no_closing_data_available": n_no_closing,
        "n_excluded_line_changed": n_line_changed,
        "n_primary_universe": n_primary,
    }


# ---------------------------------------------------------------------------
# Phase B: 2024-25 second-season replication (17.4)
# ---------------------------------------------------------------------------
def build_phase_b(core_bettime_raw: pd.DataFrame, saves_deduped_all: pd.DataFrame, log) -> dict[str, Any]:
    saves = core_bettime_raw[
        (core_bettime_raw["pass_name"] == "combined-2024-25") & (core_bettime_raw["market_key"] == "player_total_saves")
    ].copy()
    n_fanatics = int((saves["book_key"].str.lower() == "fanatics").sum())
    if n_fanatics:
        raise RuntimeError(f"Section 14.5 rule 5 schema surprise: {n_fanatics} Fanatics rows in the new pass. Stop before Phase B.")
    saves["book_key"] = saves["book_key"].str.lower()
    saves["side"] = saves["side"].str.lower()

    before_drift = len(saves)
    saves = saves[saves["effective_gap_minutes"] >= DRIFT_MIN_GAP_MINUTES].copy()
    n_drift_excluded = before_drift - len(saves)
    log(f"Phase B: commence-drift filter (effective_gap_minutes >= {DRIFT_MIN_GAP_MINUTES}): "
        f"{before_drift} -> {len(saves)} rows ({n_drift_excluded} excluded).")

    saves["_pass_bucket"] = "new_bettime"
    saves_deduped, dedup_stats = dedup_max_resolved_ts(saves, ["event_id", "player_name_raw", "book_key", "side"], pass_col="_pass_bucket")
    saves_deduped = saves_deduped.drop(columns=["_pass_bucket"])
    log(f"Phase B new-pass dedup: {dedup_stats}")

    new_paired = build_paired_devig(saves_deduped, "player_name_raw", "book_key")
    bo_new_bt = new_paired[new_paired["book_key"] == "betonlineag"].copy()
    other_new_bt = new_paired[(new_paired["book_key"] != "betonlineag") & (~new_paired["book_key"].isin(DFS_BOOKS))].copy()
    log(f"Phase B: betonlineag new-pass paired bettime quotes={len(bo_new_bt)}, "
        f"other-book paired bettime quotes={len(other_new_bt)} across books={sorted(other_new_bt['book_key'].unique().tolist())}.")

    new_goalie = saves_deduped[["event_id", "player_name_raw", "goalie_id", "goalie_name_matched"]].drop_duplicates()
    bo_new_bt = bo_new_bt.merge(new_goalie, on=["event_id", "player_name_raw"], how="left")
    bo_new_bt["goalie_key"] = compute_goalie_key(bo_new_bt["goalie_id"], bo_new_bt["goalie_name_matched"], bo_new_bt["player_name_raw"])
    other_new_bt = other_new_bt.merge(new_goalie, on=["event_id", "player_name_raw"], how="left")
    other_new_bt["goalie_key"] = compute_goalie_key(other_new_bt["goalie_id"], other_new_bt["goalie_name_matched"], other_new_bt["player_name_raw"])

    consensus = (
        other_new_bt.groupby(["event_id", "goalie_key", "line"])
        .agg(consensus_p_under_bettime=("p_under_devigged", "median"),
             n_other_qualifying_books_bettime=("book_key", "nunique"),
             other_books_bettime=("book_key", lambda s: ",".join(sorted(set(s)))))
        .reset_index()
    )
    universe = bo_new_bt.merge(consensus, on=["event_id", "goalie_key", "line"], how="left")
    universe["n_other_qualifying_books_bettime"] = universe["n_other_qualifying_books_bettime"].fillna(0).astype(int)
    universe["gate_passed"] = universe["n_other_qualifying_books_bettime"] >= MIN_OTHER_BOOKS
    universe["deviation_under"] = np.where(universe["gate_passed"], universe["p_under_devigged"] - universe["consensus_p_under_bettime"], np.nan)

    # 11-event-overlap dedup (17.4): the pre-existing 21-event 2024-25
    # bettime fragment inside saves_lines_snapshots.parquet contributes
    # ZERO rows here. Only the new-pass bettime population above is used.
    old_2425 = saves_deduped_all[saves_deduped_all["season"] == "2024-25"].copy()
    old_closing = old_2425[old_2425["snapshot_pass"] == "closing"]
    old_closing_paired = build_paired_devig(old_closing, "goalie_name_raw", "book")
    bo_old_close = old_closing_paired[old_closing_paired["book"] == "betonlineag"].copy()
    old_goalie = old_closing[["event_id", "goalie_name_raw", "goalie_id", "goalie_name_matched"]].drop_duplicates()
    bo_old_close = bo_old_close.merge(old_goalie, on=["event_id", "goalie_name_raw"], how="left")
    bo_old_close["goalie_key"] = compute_goalie_key(bo_old_close["goalie_id"], bo_old_close["goalie_name_matched"], bo_old_close["goalie_name_raw"])
    log(f"Phase B: existing-archive 2024-25 closing betonlineag paired quotes={len(bo_old_close)} "
        "(old 21-event bettime fragment is never loaded for Phase B's bettime side, per 17.4's 11-event-overlap resolution).")

    closing_small = bo_old_close[["event_id", "goalie_key", "line", "p_under_devigged"]].rename(
        columns={"line": "closing_line", "p_under_devigged": "p_under_devigged_closing"}
    )
    universe = universe.merge(closing_small, on=["event_id", "goalie_key"], how="left")
    universe["has_closing_quote_any_line"] = universe["closing_line"].notna()
    universe["line_identical"] = np.where(universe["has_closing_quote_any_line"], universe["line"] == universe["closing_line"], np.nan)
    universe["reversion_under"] = np.where(
        universe["gate_passed"] & (universe["line_identical"] == True),  # noqa: E712
        universe["p_under_devigged_closing"] - universe["p_under_devigged"], np.nan,
    )
    universe["included_in_primary"] = universe["gate_passed"] & (universe["line_identical"] == True)  # noqa: E712
    universe["cluster_id"] = universe["event_id"].astype(str) + "::" + universe["goalie_key"].astype(str)
    universe = universe.rename(columns={"line": "bettime_line", "p_under_devigged": "p_under_devigged_bettime", "p_over_devigged": "p_over_devigged_bettime"})

    n_total = len(universe)
    n_gate_excluded = int((~universe["gate_passed"]).sum())
    n_no_closing = int((universe["gate_passed"] & ~universe["has_closing_quote_any_line"]).sum())
    n_line_changed = int((universe["gate_passed"] & universe["has_closing_quote_any_line"] & (universe["line_identical"] == False)).sum())  # noqa: E712
    n_primary = int(universe["included_in_primary"].sum())
    n_with_both_any = int((universe["gate_passed"] & universe["has_closing_quote_any_line"]).sum())
    assert n_gate_excluded + n_no_closing + n_line_changed + n_primary == n_total, "Phase B exclusion funnel does not sum to total"

    primary = universe[universe["included_in_primary"]].copy()
    bootstrap = cluster_bootstrap_pearson_r(
        primary["deviation_under"].to_numpy(), primary["reversion_under"].to_numpy(), primary["cluster_id"].to_numpy(), N_BOOTSTRAP, BOOTSTRAP_SEED,
    )
    bootstrap_repeat = cluster_bootstrap_pearson_r(
        primary["deviation_under"].to_numpy(), primary["reversion_under"].to_numpy(), primary["cluster_id"].to_numpy(), N_BOOTSTRAP, BOOTSTRAP_SEED,
    )
    determinism_ok = bootstrap == bootstrap_repeat
    ols = ols_slope_secondary(primary["deviation_under"].to_numpy(), primary["reversion_under"].to_numpy())

    if bootstrap["unstable"]:
        verdict = "UNSTABLE"
    elif bootstrap["pass_bar_ci95_entirely_below_zero"]:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    log(f"PHASE B STATISTIC (FINAL, REGISTERED): r={bootstrap['point_r']}, "
        f"CI95=[{bootstrap['ci95_lower']}, {bootstrap['ci95_upper']}], n={bootstrap['n_goalie_nights']}, "
        f"degenerate={bootstrap['n_degenerate_resamples']} ({bootstrap['pct_degenerate_resamples']:.4f}%), verdict={verdict}.")
    log(f"Phase B exclusions: gate={n_gate_excluded}, no_closing_data={n_no_closing}, line_changed={n_line_changed}, primary={n_primary}, total={n_total}.")
    log(f"Phase B bootstrap determinism re-check (same seed, same input): identical={determinism_ok}.")

    return {
        "universe": universe,
        "primary": primary,
        "bootstrap": bootstrap,
        "ols": ols,
        "verdict": verdict,
        "determinism_verified": determinism_ok,
        "n_total_betonlineag_bettime_paired_quotes": n_total,
        "n_excluded_fewer_than_2_books_gate": n_gate_excluded,
        "n_excluded_no_closing_data_available": n_no_closing,
        "n_excluded_line_changed": n_line_changed,
        "n_with_both_bettime_and_closing_any_line": n_with_both_any,
        "n_primary_universe": n_primary,
        "n_drift_excluded": n_drift_excluded,
        "dedup_stats": dedup_stats,
        "bo_new_bt_gated": universe,  # reused by 17.5 for the same combined-2024-25 pass
    }


# ---------------------------------------------------------------------------
# 17.5: EV-stacked filter test
# ---------------------------------------------------------------------------
def build_17_5(phase_b_universe: pd.DataFrame, exp11_universe: pd.DataFrame, exp11_clv: pd.DataFrame, phase_a_verdict: str, log) -> dict[str, Any]:
    arm_pop = exp11_universe[exp11_universe["is_model_arm"]].copy()
    n_pop = len(arm_pop)
    if (arm_pop["book_key"] != "betonlineag").any():
        raise RuntimeError("17.5 population has a non-betonlineag book_key row; registration assumption violated.")

    arm_pop["goalie_id"] = arm_pop["goalie_id"].astype("Int64")
    # 17.5 joins the frozen universe to the new-pass deviation table by
    # (event_id, goalie_id) specifically (not the goalie_key fallback used
    # for Phase B's cross-parquet join), per the registration's own text.
    candidates = phase_b_universe[["event_id", "goalie_id", "bettime_line", "deviation_under", "gate_passed"]].copy()
    candidates["goalie_id"] = candidates["goalie_id"].astype("Int64")
    candidates = candidates.rename(columns={"bettime_line": "new_pass_line"})

    merged = arm_pop.merge(candidates, on=["event_id", "goalie_id"], how="left", suffixes=("", "_candidate"))
    n_no_quote = int(merged["new_pass_line"].isna().sum())
    merged["has_new_pass_quote"] = merged["new_pass_line"].notna()
    merged["line_match"] = np.where(merged["has_new_pass_quote"], merged["betting_line"] == merged["new_pass_line"], np.nan)
    n_line_mismatch = int((merged["has_new_pass_quote"] & (merged["line_match"] == False)).sum())  # noqa: E712

    merged["computable"] = merged["has_new_pass_quote"] & (merged["line_match"] == True) & merged["gate_passed"].fillna(False)  # noqa: E712
    n_not_computable_gate = int((merged["has_new_pass_quote"] & (merged["line_match"] == True) & ~merged["gate_passed"].fillna(False)).sum())  # noqa: E712
    merged.loc[~merged["computable"], "deviation_under"] = np.nan

    merged["arm"] = "excluded_not_computable"
    merged.loc[merged["computable"] & (merged["deviation_under"] > 0), "arm"] = "agree"
    merged.loc[merged["computable"] & (merged["deviation_under"] <= 0), "arm"] = "non_agree"

    n_computable = int(merged["computable"].sum())
    n_agree = int((merged["arm"] == "agree").sum())
    n_non_agree = int((merged["arm"] == "non_agree").sum())
    assert n_no_quote + n_line_mismatch + n_not_computable_gate + n_agree + n_non_agree == n_pop, "17.5 exclusion funnel does not sum to population"

    log(f"17.5 population={n_pop}; no new-pass quote at matching (event_id, goalie_id)={n_no_quote}; "
        f"line mismatch={n_line_mismatch}; line-matched but fails >=2-books gate (not computable)={n_not_computable_gate}; "
        f"computable={n_computable}; agree-arm={n_agree}; non-agree-arm={n_non_agree}.")

    clv_keys = exp11_clv[(exp11_clv["book_key"] == "betonlineag") & (exp11_clv["bet_side"] == "UNDER")][
        ["event_id", "goalie_id", "book_key", "clv_prob_net_of_drift"]
    ].copy()
    clv_keys["goalie_id"] = clv_keys["goalie_id"].astype("Int64")
    merged = merged.merge(clv_keys, on=["event_id", "goalie_id", "book_key"], how="left", suffixes=("", "_clv"))
    merged["clv_matched"] = merged["clv_prob_net_of_drift"].notna()

    full_roi = float(arm_pop["profit_if_under"].mean())
    agree = merged[merged["arm"] == "agree"]
    non_agree = merged[merged["arm"] == "non_agree"]

    agree_bootstrap = bootstrap_roi_delta_vs_fixed_reference(agree["profit_if_under"].to_numpy(), full_roi, N_BOOTSTRAP, BOOTSTRAP_SEED)
    non_agree_bootstrap = bootstrap_roi_delta_vs_fixed_reference(non_agree["profit_if_under"].to_numpy(), full_roi, N_BOOTSTRAP, BOOTSTRAP_SEED)

    agree_verdict = arm_verdict(n_agree, agree_bootstrap["ci95_lower"])
    non_agree_verdict = arm_verdict(n_non_agree, non_agree_bootstrap["ci95_lower"])

    agree_clv_matched = agree[agree["clv_matched"]]
    non_agree_clv_matched = non_agree[non_agree["clv_matched"]]
    clv_summary = {
        "agree_arm": {
            "n_matched": int(len(agree_clv_matched)),
            "n_total": int(len(agree)),
            "mean_clv_prob_net_of_drift": float(agree_clv_matched["clv_prob_net_of_drift"].mean()) if len(agree_clv_matched) else None,
        },
        "non_agree_arm": {
            "n_matched": int(len(non_agree_clv_matched)),
            "n_total": int(len(non_agree)),
            "mean_clv_prob_net_of_drift": float(non_agree_clv_matched["clv_prob_net_of_drift"].mean()) if len(non_agree_clv_matched) else None,
        },
    }

    log(f"17.5 PRIMARY (agree-arm ROI minus full-473 ROI): full_roi={full_roi}, {agree_bootstrap}, verdict={agree_verdict}.")
    log(f"17.5 SECONDARY (non-agree-arm ROI minus full-473 ROI): {non_agree_bootstrap}, verdict={non_agree_verdict}.")
    log(f"17.5 CLV secondary: {clv_summary}")

    exploratory_only = phase_a_verdict != "PASS"

    return {
        "arms": merged,
        "n_population": n_pop,
        "n_excluded_no_new_pass_quote": n_no_quote,
        "n_excluded_line_mismatch": n_line_mismatch,
        "n_excluded_not_computable_gate": n_not_computable_gate,
        "n_computable": n_computable,
        "n_agree_arm": n_agree,
        "n_non_agree_arm": n_non_agree,
        "full_473_roi_fixed_reference": full_roi,
        "agree_arm": {"bootstrap": agree_bootstrap, "verdict": agree_verdict},
        "non_agree_arm": {"bootstrap": non_agree_bootstrap, "verdict": non_agree_verdict},
        "clv": clv_summary,
        "exploratory_only": exploratory_only,
    }


# ---------------------------------------------------------------------------
# Reconciliation (two-stage wiring discipline; hard-stop before any
# registered statistic per the task's instruction)
# ---------------------------------------------------------------------------
def reconcile(observed: dict[str, int], log) -> bool:
    all_passed = True
    for key, expected_value in EXPECTED_COUNTS.items():
        observed_value = observed.get(key)
        passed = observed_value == expected_value
        all_passed = all_passed and passed
        log(f"RECONCILIATION [{'PASS' if passed else 'FAIL'}] {key}: expected={expected_value}, observed={observed_value}")
    return all_passed


def main() -> int:
    no_completed_run_exists()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{EXPERIMENT_PREFIX}{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log = make_logger(output_dir / "run_log.txt")

    metadata: dict[str, Any] = {
        "experiment": "Experiment 14 - W6 BetOnline bettime-to-close convergence",
        "registration": "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 17",
        "run_status": "started",
        "network_calls": False,
        "betting_db_touched": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "No prior artifact exists for this Phase A/B re-derivation (17.1 point 1) -- there is no wiring gate to reproduce, only 17.8's structural counts.",
            "2025-26 is the only season with BetOnline bettime coverage and is also the discovery season (17.1 point 2): Phase A is in-sample with respect to the original lead.",
            "2024-25 is already-viewed development data (17.1 point 3): Phase B and 17.5 are development evidence, not confirmation.",
        ],
    }

    try:
        log("EXPERIMENT 14 -- W6 BETONLINE BETTIME-TO-CLOSE CONVERGENCE (SECTION 17)")
        log("No network calls. data/betting.db is never opened, per 17.6.2.")

        for path in (SAVES_SNAPSHOTS, CORE_BETTIME, EXP11_UNIVERSE, EXP11_CLV):
            if not path.exists():
                raise FileNotFoundError(f"Missing required input: {path}")

        input_checksums = {
            "saves_lines_snapshots": {"path": str(SAVES_SNAPSHOTS), "bytes": SAVES_SNAPSHOTS.stat().st_size, "sha256": sha256_file(SAVES_SNAPSHOTS)},
            "core_bettime_202607_snapshots": {"path": str(CORE_BETTIME), "bytes": CORE_BETTIME.stat().st_size, "sha256": sha256_file(CORE_BETTIME)},
            "exp11_p2_primary_betonlineag_universe": {"path": str(EXP11_UNIVERSE), "bytes": EXP11_UNIVERSE.stat().st_size, "sha256": sha256_file(EXP11_UNIVERSE)},
            "exp11_flagged_bets_with_clv": {"path": str(EXP11_CLV), "bytes": EXP11_CLV.stat().st_size, "sha256": sha256_file(EXP11_CLV)},
            "execution_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        }
        write_json(output_dir / "input_checksums.json", input_checksums)
        log("Input checksums pinned before any parsing.")

        log("STEP 1: loading saves_lines_snapshots.parquet, deriving season via build_odds_snapshots.season_from_eastern_date.")
        saves_raw = pd.read_parquet(SAVES_SNAPSHOTS)
        saves_raw["season"] = saves_raw["game_date_eastern"].map(season_from_eastern_date)
        saves_raw["book"] = saves_raw["book"].str.lower()
        saves_raw["side"] = saves_raw["side"].str.lower()
        saves_deduped_all, dedup_stats_saves = dedup_max_resolved_ts(saves_raw, ["event_id", "goalie_name_raw", "book", "side"])
        log(f"saves_lines_snapshots dedup (whole-frame, per snapshot_pass): {dedup_stats_saves}")

        bo_bt_2526_raw = saves_raw[(saves_raw["season"] == "2025-26") & (saves_raw["book"] == "betonlineag") & (saves_raw["snapshot_pass"] == "bettime")]
        bo_bt_2526_dedup = saves_deduped_all[(saves_deduped_all["season"] == "2025-26") & (saves_deduped_all["book"] == "betonlineag") & (saves_deduped_all["snapshot_pass"] == "bettime")]
        bo_cl_2526_dedup = saves_deduped_all[(saves_deduped_all["season"] == "2025-26") & (saves_deduped_all["book"] == "betonlineag") & (saves_deduped_all["snapshot_pass"] == "closing")]

        join_key = ["event_id", "goalie_name_raw", "side"]
        joined_2526 = bo_bt_2526_dedup.merge(bo_cl_2526_dedup, on=join_key, suffixes=("_bt", "_cl"), how="inner")
        n_line_identical = int((joined_2526["line_bt"] == joined_2526["line_cl"]).sum())
        n_line_changed = int((joined_2526["line_bt"] != joined_2526["line_cl"]).sum())
        n_unique_goalie_nights = int(joined_2526[["event_id", "goalie_name_raw"]].drop_duplicates().shape[0])

        log("STEP 2: loading core_bettime_202607_snapshots.parquet for the new-pass structural check.")
        core_bettime_raw = pd.read_parquet(CORE_BETTIME)
        new_saves_all = core_bettime_raw[
            (core_bettime_raw["pass_name"] == "combined-2024-25") & (core_bettime_raw["market_key"] == "player_total_saves")
        ]
        new_bo_saves = new_saves_all[new_saves_all["book_key"].str.lower() == "betonlineag"]
        dup_mask = new_bo_saves.duplicated(subset=["event_id", "player_name_raw", "book_key", "side"], keep=False)
        new_bo_dup_groups = int(new_bo_saves.loc[dup_mask, ["event_id", "player_name_raw", "book_key", "side"]].drop_duplicates().shape[0])

        log("STEP 3: loading Experiment 11's frozen p2_primary_betonlineag_universe.parquet (read-only reuse).")
        exp11_universe = pd.read_parquet(EXP11_UNIVERSE)
        exp11_clv = pd.read_parquet(EXP11_CLV)

        observed_counts = {
            "betonlineag_bettime_2025_26_raw_rows": int(len(bo_bt_2526_raw)),
            "betonlineag_bettime_2025_26_deduped_rows": int(len(bo_bt_2526_dedup)),
            "joined_bettime_closing_pairs_total": int(len(joined_2526)),
            "joined_bettime_closing_pairs_line_identical": n_line_identical,
            "joined_bettime_closing_pairs_line_changed": n_line_changed,
            "joined_unique_goalie_nights_any_side": n_unique_goalie_nights,
            "new_pass_betonlineag_saves_rows": int(len(new_bo_saves)),
            "new_pass_betonlineag_saves_events": int(new_bo_saves["event_id"].nunique()),
            "new_pass_betonlineag_within_pass_duplicate_groups": new_bo_dup_groups,
            "exp11_universe_rows": int(len(exp11_universe)),
            "exp11_model_arm_rows": int(exp11_universe["is_model_arm"].sum()),
        }
        metadata["structural_reconciliation"] = {"expected": EXPECTED_COUNTS, "observed": observed_counts}

        log("STEP 4: reconciling constructed universes against 17.8's persisted structural counts BEFORE any Phase A/B/17.5 statistic.")
        all_passed = reconcile(observed_counts, log)
        metadata["structural_reconciliation"]["all_passed"] = all_passed
        write_json(output_dir / "metadata.json", metadata)
        if not all_passed:
            metadata["run_status"] = "stopped_reconciliation_mismatch"
            write_json(output_dir / "metadata.json", metadata)
            log("RECONCILIATION FAILED. Stopping before any Phase A/B/17.5 statistic is computed, per the task's hard-stop instruction.")
            sys.exit(
                "Experiment 14 reconciliation against section 17.8's persisted structural counts failed. "
                "See run_log.txt and metadata.json['structural_reconciliation'] for the mismatched check(s). "
                "Stopping without adjusting any rule to fit, per instruction."
            )
        log("All structural reconciliation checks passed. Proceeding to Phase A.")

        log("STEP 5: Phase A -- 2025-26 clustered re-derivation (17.3).")
        phase_a = build_phase_a(saves_deduped_all, log)

        log("STEP 6: Phase B -- 2024-25 second-season replication (17.4).")
        phase_b = build_phase_b(core_bettime_raw, saves_deduped_all, log)

        log("STEP 7: 17.5 EV-stacked filter test on the frozen Experiment 11 model arm.")
        filter_test = build_17_5(phase_b["bo_new_bt_gated"], exp11_universe, exp11_clv, phase_a["verdict"], log)

        both_phases_pass = phase_a["verdict"] == "PASS" and phase_b["verdict"] == "PASS"
        either_phase_fails = phase_a["verdict"] in ("CLOSED", "FAIL") or phase_b["verdict"] in ("CLOSED", "FAIL")
        if both_phases_pass:
            overall_consequence = (
                "Both Phase A and Phase B cluster-bootstrap CI95s are entirely below zero. Per 17.7, the convergence "
                "filter is registered as a 2026-27 shadow-candidate filter stacked on model EV, joining the "
                "Experiment 11 and Experiment 12 shadow candidates already on record. It is NOT promoted to live "
                "betting and this is not a standalone-strategy or edge claim."
            )
        elif either_phase_fails:
            overall_consequence = (
                "At least one of Phase A / Phase B's cluster-bootstrap CI95 includes zero. Per 17.7, this lead is "
                "CLOSED this cycle, matching the steam-recon and DFS-census precedents. It does not reopen without "
                "a new architecture or a new season of bettime coverage."
            )
        else:
            overall_consequence = "One or both phases returned UNSTABLE; reported as a methods/sample-structure finding, not a verdict, per 17.7."

        log(f"OVERALL VERDICT: phase_a={phase_a['verdict']}, phase_b={phase_b['verdict']}, "
            f"filter_test_exploratory_only={filter_test['exploratory_only']}.")
        log(overall_consequence)

        # ---------------- Persist row-level artifacts ----------------
        phase_a_cols = [
            "event_id", "goalie_name_raw", "goalie_id", "goalie_name_matched", "goalie_key", "cluster_id",
            "bettime_line", "p_over_devigged_bettime", "p_under_devigged_bettime",
            "n_other_qualifying_books_bettime", "other_books_bettime", "consensus_p_under_bettime",
            "deviation_under", "gate_passed", "has_closing_quote_any_line", "closing_line",
            "p_under_devigged_closing", "line_identical", "reversion_under", "included_in_primary",
        ]
        phase_a["universe"][phase_a_cols].to_parquet(output_dir / "phase_a_universe.parquet", index=False)

        phase_b_cols = [
            "event_id", "player_name_raw", "goalie_id", "goalie_name_matched", "goalie_key", "cluster_id",
            "bettime_line", "p_over_devigged_bettime", "p_under_devigged_bettime",
            "n_other_qualifying_books_bettime", "other_books_bettime", "consensus_p_under_bettime",
            "deviation_under", "gate_passed", "has_closing_quote_any_line", "closing_line",
            "p_under_devigged_closing", "line_identical", "reversion_under", "included_in_primary",
        ]
        phase_b["universe"][phase_b_cols].to_parquet(output_dir / "phase_b_universe.parquet", index=False)

        arms_cols = [
            "event_id", "goalie_id", "game_id", "book_key", "betting_line", "cluster_id", "ev_under",
            "profit_if_under", "new_pass_line", "has_new_pass_quote", "line_match", "gate_passed",
            "computable", "deviation_under", "arm", "clv_prob_net_of_drift", "clv_matched",
        ]
        filter_test["arms"][arms_cols].to_parquet(output_dir / "filter_test_17_5_arms.parquet", index=False)
        filter_test["arms"][arms_cols].to_csv(output_dir / "filter_test_17_5_arms.csv", index=False)

        # ---------------- metadata.json ----------------
        metadata["run_status"] = "completed"
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["phase_a"] = {
            "statistic": phase_a["bootstrap"],
            "ols_secondary": phase_a["ols"],
            "n_total_betonlineag_bettime_paired_quotes": phase_a["n_total_betonlineag_bettime_paired_quotes"],
            "n_excluded_fewer_than_2_books_gate": phase_a["n_excluded_fewer_than_2_books_gate"],
            "n_excluded_no_closing_data_available": phase_a["n_excluded_no_closing_data_available"],
            "n_excluded_line_changed": phase_a["n_excluded_line_changed"],
            "n_primary_universe": phase_a["n_primary_universe"],
            "determinism_verified": phase_a["determinism_verified"],
            "verdict": phase_a["verdict"],
        }
        metadata["phase_b"] = {
            "statistic": phase_b["bootstrap"],
            "ols_secondary": phase_b["ols"],
            "n_total_betonlineag_bettime_paired_quotes": phase_b["n_total_betonlineag_bettime_paired_quotes"],
            "n_excluded_fewer_than_2_books_gate": phase_b["n_excluded_fewer_than_2_books_gate"],
            "n_excluded_no_closing_data_available": phase_b["n_excluded_no_closing_data_available"],
            "n_excluded_line_changed": phase_b["n_excluded_line_changed"],
            "n_with_both_bettime_and_closing_any_line": phase_b["n_with_both_bettime_and_closing_any_line"],
            "n_primary_universe": phase_b["n_primary_universe"],
            "n_drift_excluded": phase_b["n_drift_excluded"],
            "dedup_stats": phase_b["dedup_stats"],
            "determinism_verified": phase_b["determinism_verified"],
            "verdict": phase_b["verdict"],
        }
        metadata["filter_test_17_5"] = {
            "n_population_model_arm": filter_test["n_population"],
            "n_excluded_no_new_pass_quote": filter_test["n_excluded_no_new_pass_quote"],
            "n_excluded_line_mismatch": filter_test["n_excluded_line_mismatch"],
            "n_excluded_not_computable_gate": filter_test["n_excluded_not_computable_gate"],
            "n_computable": filter_test["n_computable"],
            "n_agree_arm": filter_test["n_agree_arm"],
            "n_non_agree_arm": filter_test["n_non_agree_arm"],
            "full_473_roi_fixed_reference": filter_test["full_473_roi_fixed_reference"],
            "agree_arm_primary": filter_test["agree_arm"],
            "non_agree_arm_secondary": filter_test["non_agree_arm"],
            "clv_secondary": filter_test["clv"],
            "exploratory_only": filter_test["exploratory_only"],
            "labeled_exploratory_only_reason": "Phase A verdict != PASS" if filter_test["exploratory_only"] else None,
        }
        metadata["verdict_mapping"] = {
            "phase_a": phase_a["verdict"],
            "phase_b": phase_b["verdict"],
            "filter_test_agree_arm": filter_test["agree_arm"]["verdict"],
            "filter_test_non_agree_arm": filter_test["non_agree_arm"]["verdict"],
            "both_phases_pass": both_phases_pass,
            "either_phase_fails": either_phase_fails,
            "overall_consequence": overall_consequence,
        }
        metadata["judgment_calls"] = [
            "Goalie-night identity for within-file grouping (Phase A's internal consensus/pairing, and the "
            "17.8 structural-reconciliation join) uses goalie_name_raw directly for saves_lines_snapshots.parquet "
            "rows, matching the literal (event_id, goalie_name_raw, book, side) dedup key 17.2 specifies. The "
            "cross-parquet goalie_key (goalie_id, else goalie_name_matched, else goalie_name_raw) used for Phase A/B "
            "consensus grouping and Phase B's new-bettime-to-old-closing join was verified empirically to reproduce "
            "the identical 2,063 / 1,927 / 136 / 1,032 structural counts as a goalie_name_raw-only key would, so no "
            "reconciliation ambiguity resulted from this choice.",
            "The registration does not name an explicit exclusion bucket for 'gate-passed quotes with zero closing "
            "counterpart at any line' (distinct from 'line-changed', which presupposes a closing quote exists). This "
            "runner reports n_excluded_no_closing_data_available as its own count so each phase's exclusion funnel "
            "sums exactly to its total population; 17.3/17.4 only explicitly name the gate and line-changed counts.",
            "17.5's PRIMARY/SECONDARY ROI-delta bootstrap resamples only the arm under test and subtracts the FIXED "
            "(non-resampled) full-473 ROI, per a literal reading of 17.5's own construction language -- this differs "
            "from Experiment 11's own paired-bootstrap convention (14.6), which resamples both arms from the same "
            "draw. The registration's own text for 17.5 is explicit enough (the reference is described as fixed) "
            "that this reading was treated as unambiguous rather than a judgment call requiring flagging, but is "
            "noted here for transparency.",
            "17.5 does not name an explicit PASS/FAIL bar the way 17.3/17.4 do; this runner applies the same CI95-"
            "entirely-above-zero bar per arm (mirroring 17.5's own aside describing 'a clean pass... CI95 entirely "
            "above zero'), gated by the registered 100-bet-per-arm floor for INSUFFICIENT_SAMPLE.",
            "Numpy's default_rng(42) is used for every bootstrap (Phase A, Phase B, and 17.5), per this task's own "
            "instruction; section 17 itself pins the seed (42) and resample count (10,000) but not the RNG class. "
            "Prior experiments in this doc family (Experiment 11/13) used np.random.RandomState instead.",
        ]
        write_json(output_dir / "metadata.json", metadata)

        log(f"Artifacts saved under {output_dir}")
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        metadata["run_status"] = "exception"
        metadata["exception"] = f"{type(exc).__name__}: {exc}"
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "metadata.json", metadata)
        log(f"STOPPED: {metadata['exception']}")
        raise


if __name__ == "__main__":
    sys.exit(main())
