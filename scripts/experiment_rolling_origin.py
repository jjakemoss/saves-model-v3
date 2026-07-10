"""
Rolling-origin confirmation on fresh folds (section 3.15 pre-registration:
docs/OFFSEASON_OPTIMIZATION_PLAN.md, "Rolling-origin confirmation" bullet,
read together with section 3.14's Claude-verification block for the
statistical standard).

Tests whether the pace_shots RECIPE (same feature families, same config
grid, same validation-only selection convention) -- not the frozen
production model -- reproduces its market-parity/edge result on seasons it
has never touched:

  Origin A: train <= 2022-23, test = 2023-24 (new frame built here).
  Origin B: train <= 2023-24, test = 2024-25 (existing frame's closing rows).

All modeling code is REUSED by importing it from
experiments.distributional_saves / experiments.harness /
scripts/experiment_pace_distributional.py / scripts/clv_audit_pace_policy.py
-- nothing here reimplements shots/save-rate training, the distribution
math, bet grading, or bootstrap CIs. This script only adds: (1) the 2023-24
frame builder, (2) the rolling-origin train/val/test date carving (the
production harness's split_by_date has the production dates hardcoded and
cannot be reused for other origins), and (3) glue/reporting.

Design decisions made explicit here (see run_log.txt / metadata.json for
the same text):

  - EV threshold is FIXED at 0.05 for both origins' test evaluation (the
    frozen production pace_shots threshold), never reselected via a
    validation sweep. This is necessary, not just convenient: no betting-
    line odds exist anywhere for the 2022-23 season (the vendor's historical
    player-props archive starts 2023-05-03), so Origin A's validation window
    (carved from inside <=2022-23, per the design's explicit instruction)
    has no market data to sweep against. The same fixed threshold is used
    for Origin B so both origins are evaluated identically.
  - Validation windows are the last 49 days of each origin's training-pool
    seasons (49 = the length of the production experiment's Oct16-Dec3
    validation window), used ONLY for shots/save-rate hyperparameter config
    selection (val MAE / weighted log-loss, no market data touched). This
    places val INSIDE the training-pool seasons rather than inside the test
    season -- required so the test season is touched exactly once.
  - PRIMARY prices = closing pass for both origins. Origin A additionally
    reports a SECONDARY descriptive policy-ROI line on the 2023-24 bettime
    pass (realistic execution prices), per the task design.

Usage:
    python scripts/experiment_rolling_origin.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Plain sys.path imports (not importlib) -- this is the proven-working
# pattern already used by scripts/clv_audit_pace_policy.py to import this
# same experiment module. It sidesteps the importlib/dataclass registration
# pitfall entirely rather than working around it.
import experiment_pace_distributional as epd  # noqa: E402
import clv_audit_pace_policy as clv  # noqa: E402
from experiments.distributional_saves import (  # noqa: E402
    CAP,
    SavesDistribution,
    compute_distribution_predictions,
    fit_dispersion,
    join_and_price,
    train_save_rate_model,
    train_shots_model,
)
from experiments.harness import (  # noqa: E402
    betting_metrics_bundle,
    decide_bet,
    fold_wide_auc_brier,
    grade_bets,
    split_by_date,
)
from betting.odds_utils import calculate_payout, decimal_to_american  # noqa: E402
from betting.tracking_db import devig_prob  # noqa: E402
from features.feature_engineering import compute_line_relative_features  # noqa: E402

make_logger = epd.make_logger


SNAPSHOTS_PATH = REPO_ROOT / "data" / "processed" / "saves_lines_snapshots.parquet"
CLOSING_FRAME_PATH = REPO_ROOT / "data" / "processed" / "multibook_frame_2023_24.parquet"
BETTIME_FRAME_PATH = REPO_ROOT / "data" / "processed" / "multibook_frame_2023_24_bettime.parquet"
PRODUCTION_ARTIFACT_DIR = REPO_ROOT / "models" / "trained" / "experiment_pace_distributional_20260709_100802"
OUTPUT_ROOT = REPO_ROOT / "models" / "trained"

# clean_training_data.parquet / multibook_classification_training_data.parquet
# both store `season` as an integer, e.g. 20222023 for the 2022-23 season.
SEASON_2022_23 = 20222023
SEASON_2023_24 = 20232024
SEASON_2024_25 = 20242025

VAL_WINDOW_DAYS = 49  # matches the production experiment's Oct16-Dec3 (inclusive) val window length
FIXED_EV_THRESHOLD = 0.05  # frozen production pace_shots threshold; never reselected here

# CAP=70 (imported from distributional_saves) is the production model's PMF
# truncation bound and MUST stay 70 for the wiring gate to exactly reproduce
# the frozen artifact. It is not adequate for the new origins: Origin A's
# much smaller training pool produces a wider/less-regularized shots model,
# and one 2023-24 test-fold goalie-night (mu=42.47, a real high-shots game)
# lost >0.1% of its PMF mass above shots=70, failing the harness's own
# normalization assert. This is a numerical-support choice made before
# looking at any grading/ROI number (verified empirically to restore
# normalization for both origins, not tuned against a result), applied
# identically to Origin A and Origin B.
ORIGIN_CAP = 90

DROPPED_BOOK_KEYS = {"prizepicks", "manual", "unknown"}

WIRING_GATE_EXPECTED = {
    "val_brier": 0.25327,
    "test_brier": 0.24904,
    "test_roi": 9.02,
    "test_bets": 616,
}

N_BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# WIRING GATE: reproduce the production pace_shots val/test metrics from the
# frozen artifact before any new number is trusted.
# ---------------------------------------------------------------------------


def run_wiring_gate(log) -> dict:
    log("=" * 80)
    log("WIRING GATE: reproduce production pace_shots val/test metrics from the frozen artifact")
    log("=" * 80)
    metadata_path = PRODUCTION_ARTIFACT_DIR / "metadata.json"
    frozen_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frozen_result = frozen_metadata["results"]["pace_shots"]

    frame = epd.load_pace_modeling_frame(
        epd.DATA_PATH_CLEAN, epd.DATA_PATH_CONTEXT, epd.DATA_PATH_PACE, epd.DATA_PATH_PACE_METADATA, log,
    )
    clean_split = split_by_date(frame.df, log, "clean_training_data (production dates)")

    variant = next(v for v in epd.VARIANTS if v.name == "pace_shots")
    shots_cols, rate_cols = epd.feature_cols_for_variant(frame, variant)
    if shots_cols != frozen_result["shots_feature_cols"] or rate_cols != frozen_result["rate_feature_cols"]:
        raise AssertionError(
            "WIRING GATE FAILED: reconstructed pace_shots feature lists do not match the frozen artifact."
        )
    log(f"Feature identity check passed: {len(shots_cols)} shots cols, {len(rate_cols)} rate cols match the frozen artifact.")

    shots_model = xgb.XGBRegressor()
    shots_model.load_model(str(PRODUCTION_ARTIFACT_DIR / "pace_shots_shots_model.json"))
    rate_model = xgb.XGBRegressor()
    rate_model.load_model(str(PRODUCTION_ARTIFACT_DIR / "pace_shots_save_rate_model.json"))
    log("Loaded frozen pace_shots shots_model and save_rate_model from the production artifact (no retraining).")

    alpha_recomputed, _dispersion_method, _diag = fit_dispersion(
        shots_model, frame.df, clean_split.train_idx, shots_cols, log, "wiring_gate pace_shots",
    )
    alpha_frozen = float(frozen_result["dispersion"]["alpha"])
    if abs(alpha_recomputed - alpha_frozen) > 1e-6:
        raise AssertionError(
            f"WIRING GATE FAILED: recomputed dispersion alpha ({alpha_recomputed}) does not match "
            f"the frozen artifact's alpha ({alpha_frozen})."
        )
    alpha = alpha_frozen
    log(f"Dispersion alpha matches frozen artifact: {alpha:.6f}")

    dist = SavesDistribution(CAP)
    dist_preds_val = compute_distribution_predictions(
        frame.df, clean_split.val_idx, shots_model, rate_model, alpha, shots_cols, rate_cols, dist, log,
        "wiring_gate VAL",
    )
    dist_preds_test = compute_distribution_predictions(
        frame.df, clean_split.test_idx, shots_model, rate_model, alpha, shots_cols, rate_cols, dist, log,
        "wiring_gate TEST",
    )

    df_bet = epd.build_betting_frame(epd.DATA_PATH_MULTIBOOK, log)
    bet_split = split_by_date(df_bet, log, "multibook_classification_training_data (production dates)")
    df_bet_val = df_bet.iloc[bet_split.val_idx].reset_index(drop=True)
    df_bet_test = df_bet.iloc[bet_split.test_idx].reset_index(drop=True)

    p_over_val, _p_under_val, _p_push_val, matched_val, _cov_val = join_and_price(
        df_bet_val, dist_preds_val, dist, log, "wiring_gate VAL betting frame",
    )
    _val_auc, val_brier = fold_wide_auc_brier(
        p_over_val, matched_val, df_bet_val["saves"].values, df_bet_val["betting_line"].values,
        df_bet_val["game_id"].values, df_bet_val["goalie_id"].values, log, "wiring_gate VAL",
    )

    p_over_test, p_under_test, _p_push_test, matched_test, _cov_test = join_and_price(
        df_bet_test, dist_preds_test, dist, log, "wiring_gate TEST betting frame",
    )
    test_bundle, _test_auc, test_brier = epd.evaluate_test_once(
        df_bet_test, p_over_test, p_under_test, matched_test, 0.05, log, "wiring_gate TEST",
    )

    observed = {
        "val_brier": round(float(val_brier), 5),
        "test_brier": round(float(test_brier), 5),
        "test_roi": round(float(test_bundle["summary"]["roi"]), 2),
        "test_bets": int(test_bundle["summary"]["bets"]),
    }
    log(f"Observed: {observed}")
    log(f"Expected: {WIRING_GATE_EXPECTED}")
    if observed != WIRING_GATE_EXPECTED:
        raise AssertionError(
            f"WIRING GATE FAILED: expected {WIRING_GATE_EXPECTED}, observed {observed}. "
            "Stopping per task instructions rather than improvising."
        )
    log(
        "WIRING GATE PASSED: reproduced production pace_shots val Brier=0.25327, "
        "test Brier=0.24904, test ROI=+9.02%, 616 bets."
    )
    return observed


# ---------------------------------------------------------------------------
# STEP 1: build the 2023-24 multibook-style frame(s) from the snapshots
# parquet, mirroring scripts/build_multibook_training_data.py's conventions
# (per-book rows, never averaged).
# ---------------------------------------------------------------------------


def build_season_multibook_frame(
    base_df_season: pd.DataFrame,
    snapshots_all: pd.DataFrame,
    pass_name: str,
    log,
) -> pd.DataFrame:
    base_df_season = base_df_season.copy()
    base_df_season["game_id"] = base_df_season["game_id"].astype(int)
    base_df_season["goalie_id"] = base_df_season["goalie_id"].astype(int)

    game_lookup_df = base_df_season[["goalie_id", "game_date", "game_id"]].copy()
    game_lookup_df["date_str"] = pd.to_datetime(game_lookup_df["game_date"]).dt.strftime("%Y-%m-%d")
    dup_keys = int(game_lookup_df.duplicated(subset=["goalie_id", "date_str"]).sum())
    if dup_keys:
        raise AssertionError(
            f"{dup_keys} duplicate (goalie_id, date) keys in the season base frame; "
            "game_id lookup would be ambiguous."
        )
    game_lookup = dict(
        zip(zip(game_lookup_df["goalie_id"], game_lookup_df["date_str"]), game_lookup_df["game_id"])
    )

    if pass_name == "closing":
        pass_df = clv.clean_closing_pass(snapshots_all)
    elif pass_name == "bettime":
        pass_df = clv.clean_bettime_pass(snapshots_all)
    else:
        raise ValueError(f"Unknown pass_name: {pass_name}")

    pass_df = pass_df[~pass_df["book"].astype(str).str.lower().isin(DROPPED_BOOK_KEYS)].copy()

    pass_df, n_unmatched = clv.attach_game_id(pass_df, game_lookup)
    log(
        f"[{pass_name}] attach_game_id via (goalie_id, game_date_eastern): {n_unmatched} rows unmatched and "
        "dropped (expected: rows outside the 2023-24 season, plus any unresolved goalie_id)."
    )

    valid_game_ids = set(base_df_season["game_id"].tolist())
    before = len(pass_df)
    pass_df = pass_df[pass_df["game_id"].isin(valid_game_ids)].copy()
    log(
        f"[{pass_name}] {before - len(pass_df)} additional rows outside the 2023-24 season game_id "
        "universe dropped (sanity filter -- should be 0 given the lookup is season-scoped already)."
    )

    group_cols = ["event_id", "game_id", "goalie_id", "book", "line"]
    wide = clv.pivot_both_sides(pass_df, group_cols)
    log(
        f"[{pass_name}] both-sides pivot: {len(wide)} (event, game, goalie, book, line) rows "
        "with Over and Under both quoted."
    )

    wide["odds_over_american"] = wide["price_decimal_over"].apply(decimal_to_american)
    wide["odds_under_american"] = wide["price_decimal_under"].apply(decimal_to_american)
    wide = wide.rename(
        columns={
            "price_decimal_over": "odds_over_decimal",
            "price_decimal_under": "odds_under_decimal",
            "line": "betting_line",
            "book": "book_key",
        }
    )
    wide["num_books"] = 1

    merged = wide.merge(base_df_season, on=["game_id", "goalie_id"], how="left", validate="many_to_one")
    n_missing_base = int(merged["saves"].isna().sum())
    if n_missing_base:
        raise AssertionError(f"[{pass_name}] {n_missing_base} rows failed to join season base features; investigate.")

    merged["over_hit"] = (merged["saves"] > merged["betting_line"]).astype(int)
    merged["line_margin"] = merged["saves"] - merged["betting_line"]
    merged = compute_line_relative_features(merged)

    before_dedup = len(merged)
    merged = merged.drop_duplicates(
        subset=["game_id", "goalie_id", "book_key", "betting_line"], keep="last"
    ).reset_index(drop=True)
    log(
        f"[{pass_name}] deduplicated on (game_id, goalie_id, book_key, betting_line): "
        f"{before_dedup - len(merged)} rows dropped."
    )

    merged = merged.sort_values("game_date").reset_index(drop=True)

    n_events = merged["event_id"].nunique()
    n_nights = merged[["game_id", "goalie_id"]].drop_duplicates().shape[0]
    base_universe = base_df_season[["game_id", "goalie_id"]].drop_duplicates().shape[0]
    log(
        f"[{pass_name}] FINAL: {len(merged)} rows, {n_events} distinct events, {n_nights} distinct "
        f"goalie-nights ({n_nights / base_universe * 100:.2f}% of the {base_universe}-goalie-night "
        "2023-24 base universe)."
    )
    log(f"[{pass_name}] books: {merged['book_key'].value_counts().to_dict()}")

    return merged


# ---------------------------------------------------------------------------
# Rolling-origin date carving. experiments.harness.split_by_date has the
# production dates hardcoded (TRAIN_END_EXCLUSIVE etc. are module constants)
# and cannot be reused for a different origin; this is the one genuinely new
# piece of chronological-split logic in this script, and it follows the same
# mechanics (contiguous, chronological, no leakage) as split_by_date.
# ---------------------------------------------------------------------------


def season_date_range(clean_full: pd.DataFrame, seasons: list[int]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """clean_training_data.parquet's own unambiguous `season` column, used to
    look up date bounds. NOTE: frame.df (the pace-merged modeling frame) is
    NOT used for season membership -- the pace-features merge in
    load_pace_modeling_frame also carries a `season` verification column, so
    a plain, non-key-column merge suffixes both into season_x/season_y.
    Date-range masks on frame.df["game_date"] (a genuine, unambiguous join
    key) sidestep that entirely and are exact since frame.df has no rows
    outside real game dates (no off-season filler rows)."""
    mask = clean_full["season"].isin(seasons)
    dates = clean_full.loc[mask, "game_date"]
    if dates.empty:
        raise ValueError(f"No clean_training_data rows found for seasons {seasons}.")
    return dates.min(), dates.max()


def carve_origin_split(
    frame_df: pd.DataFrame, pool_min: pd.Timestamp, pool_max: pd.Timestamp, val_window_days: int, log, label: str
):
    pool_mask = (frame_df["game_date"] >= pool_min) & (frame_df["game_date"] <= pool_max)
    val_start = pool_max - pd.Timedelta(days=val_window_days - 1)

    train_mask = pool_mask & (frame_df["game_date"] < val_start)
    val_mask = pool_mask & (frame_df["game_date"] >= val_start) & (frame_df["game_date"] <= pool_max)

    train_idx = np.where(train_mask.values)[0]
    val_idx = np.where(val_mask.values)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"[{label}] Empty train or val fold after carving; investigate.")

    boundaries = {
        "pool_date_range": [str(pool_min.date()), str(pool_max.date())],
        "train": {
            "start": str(frame_df.loc[train_mask, "game_date"].min().date()),
            "end": str(frame_df.loc[train_mask, "game_date"].max().date()),
            "rows": int(len(train_idx)),
        },
        "val": {"start": str(val_start.date()), "end": str(pool_max.date()), "rows": int(len(val_idx))},
        "val_window_days": val_window_days,
    }
    log(
        f"[{label}] train/val carve: train {boundaries['train']['start']}..{boundaries['train']['end']} "
        f"(n={boundaries['train']['rows']}); val {boundaries['val']['start']}..{boundaries['val']['end']} "
        f"(n={boundaries['val']['rows']}), the last {val_window_days} days of the training-pool date range "
        f"{boundaries['pool_date_range']}."
    )
    return train_idx, val_idx, boundaries


def date_range_test_idx(
    frame_df: pd.DataFrame, test_min: pd.Timestamp, test_max: pd.Timestamp, log, label: str
) -> np.ndarray:
    mask = (frame_df["game_date"] >= test_min) & (frame_df["game_date"] <= test_max)
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        raise ValueError(f"[{label}] No rows found for test date range {test_min}..{test_max}.")
    log(f"[{label}] test date range {test_min.date()}..{test_max.date()} (n={len(idx)})")
    return idx


# ---------------------------------------------------------------------------
# Paired per-row Brier delta vs the de-vigged market (same statistic as the
# 3.14 Claude-verification block), and the per-row prediction table.
# ---------------------------------------------------------------------------


def paired_brier_delta(df_bet: pd.DataFrame, p_over: np.ndarray, matched: np.ndarray, log, label: str):
    n = len(df_bet)
    saves_arr = df_bet["saves"].values.astype(float)
    lines_arr = df_bet["betting_line"].values.astype(float)
    odds_o_arr = df_bet["odds_over_american"].astype(float).values
    odds_u_arr = df_bet["odds_under_american"].astype(float).values
    game_id_arr = df_bet["game_id"].values
    goalie_id_arr = df_bet["goalie_id"].values

    market_p_over_arr = np.full(n, np.nan)
    market_p_under_arr = np.full(n, np.nan)

    model_sq_list = []
    market_sq_list = []
    cluster_ids = []
    n_devig_fail = 0
    for i in range(n):
        if not matched[i]:
            continue
        y = 1.0 if saves_arr[i] > lines_arr[i] else 0.0
        p_mkt_over, p_mkt_under = devig_prob(odds_o_arr[i], odds_u_arr[i])
        if p_mkt_over is None:
            n_devig_fail += 1
            continue
        market_p_over_arr[i] = p_mkt_over
        market_p_under_arr[i] = p_mkt_under
        model_sq_list.append((float(p_over[i]) - y) ** 2)
        market_sq_list.append((p_mkt_over - y) ** 2)
        cluster_ids.append(f"{int(game_id_arr[i])}_{int(goalie_id_arr[i])}")

    model_sq = np.array(model_sq_list)
    market_sq = np.array(market_sq_list)
    delta = model_sq - market_sq
    cluster_ids_arr = np.array(cluster_ids, dtype=object)

    stat = clv.cluster_bootstrap_mean_ci(
        delta, cluster_ids_arr, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, ci_pct=95.0
    )
    log(
        f"[{label}] paired Brier delta (model-market): model_brier={model_sq.mean():.5f} "
        f"market_brier={market_sq.mean():.5f} delta={stat['mean']:+.5f} "
        f"95% CI=[{stat['lower']:+.5f}, {stat['upper']:+.5f}] n_rows={stat['n_bets']} "
        f"n_clusters={stat['n_clusters']} devig_failures={n_devig_fail}"
    )
    return (
        {
            "model_brier_mean": float(model_sq.mean()) if len(model_sq) else None,
            "market_brier_mean": float(market_sq.mean()) if len(market_sq) else None,
            "delta_mean": stat["mean"],
            "delta_ci95_lower": stat["lower"],
            "delta_ci95_upper": stat["upper"],
            "n_rows": stat["n_bets"],
            "n_clusters": stat["n_clusters"],
            "devig_failures": n_devig_fail,
        },
        market_p_over_arr,
        market_p_under_arr,
    )


def build_row_predictions(
    df_bet: pd.DataFrame,
    p_over: np.ndarray,
    p_under: np.ndarray,
    matched: np.ndarray,
    market_p_over_arr: np.ndarray,
    market_p_under_arr: np.ndarray,
    threshold: float,
    origin_label: str,
    price_pass: str,
) -> pd.DataFrame:
    n = len(df_bet)
    saves_arr = df_bet["saves"].values.astype(float)
    lines_arr = df_bet["betting_line"].values.astype(float)
    odds_o_arr = df_bet["odds_over_american"].astype(float).values
    odds_u_arr = df_bet["odds_under_american"].astype(float).values
    game_id_arr = df_bet["game_id"].values
    goalie_id_arr = df_bet["goalie_id"].values
    goalie_name_arr = df_bet["goalie_name"].values if "goalie_name" in df_bet.columns else [None] * n
    team_arr = df_bet["team_abbrev"].values if "team_abbrev" in df_bet.columns else [None] * n
    opp_arr = df_bet["opponent_team"].values if "opponent_team" in df_bet.columns else [None] * n
    game_date_arr = df_bet["game_date"].values
    book_arr = df_bet["book_key"].values if "book_key" in df_bet.columns else [None] * n

    rows = []
    for i in range(n):
        if not matched[i]:
            continue
        saves_i = saves_arr[i]
        line_i = lines_arr[i]
        is_push = bool(saves_i == line_i)
        outcome = None if is_push else int(saves_i > line_i)
        bet_side, ev = decide_bet(float(p_over[i]), float(p_under[i]), float(odds_o_arr[i]), float(odds_u_arr[i]), threshold)

        profit = None
        won = None
        if bet_side is not None and not is_push:
            if bet_side == "OVER":
                won = bool(saves_i > line_i)
                profit = float(calculate_payout(1.0, odds_o_arr[i], won))
            else:
                won = bool(saves_i < line_i)
                profit = float(calculate_payout(1.0, odds_u_arr[i], won))

        mkt_over = market_p_over_arr[i]
        mkt_under = market_p_under_arr[i]
        rows.append(
            {
                "origin": origin_label,
                "price_pass": price_pass,
                "game_id": int(game_id_arr[i]),
                "goalie_id": int(goalie_id_arr[i]),
                "goalie_name": goalie_name_arr[i],
                "team_abbrev": team_arr[i],
                "opponent_team": opp_arr[i],
                "game_date": str(pd.Timestamp(game_date_arr[i]).date()),
                "book": book_arr[i],
                "betting_line": line_i,
                "odds_over_american": float(odds_o_arr[i]),
                "odds_under_american": float(odds_u_arr[i]),
                "model_p_over": float(p_over[i]),
                "model_p_under": float(p_under[i]),
                "market_p_over_devigged": None if pd.isna(mkt_over) else float(mkt_over),
                "market_p_under_devigged": None if pd.isna(mkt_under) else float(mkt_under),
                "saves": saves_i,
                "outcome_over": outcome,
                "is_push": is_push,
                "bet_side": bet_side,
                "ev": ev,
                "won": won,
                "profit": profit,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-origin runner: train the pace_shots recipe on the origin's train/val
# split, price the origin's test season, report metrics, save artifacts.
# ---------------------------------------------------------------------------


def run_origin(
    origin_label: str,
    frame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    price_frames: dict,
    output_dir: Path,
    log,
) -> dict:
    log("\n" + "=" * 80)
    log(f"ORIGIN {origin_label}: pace_shots recipe -- train/val config selection, frozen-threshold test pricing")
    log("=" * 80)

    variant = next(v for v in epd.VARIANTS if v.name == "pace_shots")
    shots_cols, rate_cols = epd.feature_cols_for_variant(frame, variant)
    log(f"Shots feature count: {len(shots_cols)}  Save-rate feature count: {len(rate_cols)}")

    shots_model, shots_winner, shots_evals = train_shots_model(
        frame.df, train_idx, val_idx, shots_cols, log, f"origin_{origin_label} pace_shots"
    )
    alpha, dispersion_method, dispersion_diag = fit_dispersion(
        shots_model, frame.df, train_idx, shots_cols, log, f"origin_{origin_label} pace_shots"
    )
    rate_model, rate_winner, rate_evals = train_save_rate_model(
        frame.df, train_idx, val_idx, rate_cols, log, f"origin_{origin_label} pace_shots"
    )

    shots_path = output_dir / f"origin_{origin_label.lower()}_pace_shots_shots_model.json"
    rate_path = output_dir / f"origin_{origin_label.lower()}_pace_shots_save_rate_model.json"
    shots_model.get_booster().save_model(str(shots_path))
    rate_model.get_booster().save_model(str(rate_path))
    log(f"Saved origin {origin_label} shots model to: {shots_path}")
    log(f"Saved origin {origin_label} save-rate model to: {rate_path}")

    dist = SavesDistribution(ORIGIN_CAP)
    dist_preds_test = compute_distribution_predictions(
        frame.df, test_idx, shots_model, rate_model, alpha, shots_cols, rate_cols, dist, log,
        f"origin_{origin_label} TEST",
    )

    pass_results = {}
    all_row_frames = []
    for pass_name, df_bet in price_frames.items():
        label = f"origin_{origin_label} TEST {pass_name}"
        p_over, p_under, _p_push, matched, cov = join_and_price(df_bet, dist_preds_test, dist, log, label)
        auc, brier_val = fold_wide_auc_brier(
            p_over, matched, df_bet["saves"].values, df_bet["betting_line"].values,
            df_bet["game_id"].values, df_bet["goalie_id"].values, log, label,
        )

        brier_delta_stat, market_p_over_arr, market_p_under_arr = paired_brier_delta(
            df_bet, p_over, matched, log, label
        )

        bet_results = grade_bets(
            p_over, p_under, df_bet["saves"].values.astype(float), df_bet["betting_line"].values.astype(float),
            df_bet["odds_over_american"].astype(float).values, df_bet["odds_under_american"].astype(float).values,
            df_bet["game_id"].values, df_bet["goalie_id"].values, FIXED_EV_THRESHOLD, matched, log, label,
        )
        bundle = betting_metrics_bundle(bet_results, df_bet["game_id"].values, df_bet["goalie_id"].values, len(df_bet))
        log(
            f"[{label}] {bundle['summary']['bets']} bets, {bundle['summary']['bet_rate']:.1f}% bet rate, "
            f"{bundle['summary']['hit_rate']:.1f}% hit rate, {bundle['summary']['roi']:+.2f}% ROI"
        )
        log(
            f"[{label}] ROI 95% CI (cluster): [{bundle['roi_ci_cluster']['lower']:+.2f}%, "
            f"{bundle['roi_ci_cluster']['upper']:+.2f}%] (n_clusters={bundle['roi_ci_cluster']['n_clusters']})"
        )
        log(
            f"[{label}] side breakdown: OVER {bundle['side_breakdown']['OVER']['bets']} bets "
            f"({bundle['side_breakdown']['OVER']['roi']:+.2f}%), UNDER {bundle['side_breakdown']['UNDER']['bets']} "
            f"bets ({bundle['side_breakdown']['UNDER']['roi']:+.2f}%)"
        )

        row_df = build_row_predictions(
            df_bet, p_over, p_under, matched, market_p_over_arr, market_p_under_arr,
            FIXED_EV_THRESHOLD, origin_label, pass_name,
        )
        all_row_frames.append(row_df)

        pass_results[pass_name] = {
            "join_coverage_pct": cov,
            "fold_wide_auc": auc,
            "fold_wide_brier": brier_val,
            "paired_brier_delta_vs_market": brier_delta_stat,
            "policy_roi": bundle,
            "ev_threshold": FIXED_EV_THRESHOLD,
        }

    predictions_df = pd.concat(all_row_frames, ignore_index=True)
    predictions_path = output_dir / f"origin_{origin_label.lower()}_test_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    log(f"Saved {len(predictions_df)} per-row test predictions to: {predictions_path}")

    return {
        "origin": origin_label,
        "shots_feature_cols": shots_cols,
        "rate_feature_cols": rate_cols,
        "shots_model": {"winner": shots_winner, "val_evaluations": shots_evals, "model_path": str(shots_path)},
        "save_rate_model": {"winner": rate_winner, "val_evaluations": rate_evals, "model_path": str(rate_path)},
        "dispersion": {"alpha": alpha, "method": dispersion_method, "diagnostics": dispersion_diag},
        "test_idx_n": int(len(test_idx)),
        "price_passes": pass_results,
        "predictions_path": str(predictions_path),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_rolling_origin_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log, flush_log = make_logger(log_path)

    metadata: dict = {"timestamp": datetime.now().isoformat()}
    try:
        log("=" * 80)
        log("ROLLING-ORIGIN CONFIRMATION: pace_shots recipe on fresh 2023-24 / 2024-25 folds")
        log("Pre-registration: docs/OFFSEASON_OPTIMIZATION_PLAN.md section 3.15")
        log("=" * 80)
        log(f"Output directory: {output_dir}")

        for path in (
            SNAPSHOTS_PATH, epd.DATA_PATH_CLEAN, epd.DATA_PATH_CONTEXT, epd.DATA_PATH_PACE,
            epd.DATA_PATH_PACE_METADATA, epd.DATA_PATH_MULTIBOOK, PRODUCTION_ARTIFACT_DIR / "metadata.json",
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required input: {path}")

        # ---- WIRING GATE ----
        gate_observed = run_wiring_gate(log)
        metadata["wiring_gate"] = {"observed": gate_observed, "expected": WIRING_GATE_EXPECTED, "passed": True}

        # ---- STEP 1: build the 2023-24 multibook-style frame(s) ----
        log("\n" + "=" * 80)
        log("STEP 1: build data/processed/multibook_frame_2023_24.parquet (closing pass, PRIMARY deliverable)")
        log("=" * 80)
        clean_full = pd.read_parquet(epd.DATA_PATH_CLEAN)
        clean_full["game_date"] = pd.to_datetime(clean_full["game_date"])
        base_2023_24 = clean_full[clean_full["season"] == SEASON_2023_24].reset_index(drop=True)
        log(f"2023-24 base goalie-night universe (clean_training_data.parquet): {len(base_2023_24)} rows.")

        snapshots_all = pd.read_parquet(SNAPSHOTS_PATH)
        log(f"saves_lines_snapshots.parquet: {len(snapshots_all)} total rows across all fetched windows.")

        closing_frame = build_season_multibook_frame(base_2023_24, snapshots_all, "closing", log)
        CLOSING_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
        closing_frame.to_parquet(CLOSING_FRAME_PATH, index=False)
        log(f"Saved closing-pass 2023-24 multibook frame to: {CLOSING_FRAME_PATH}")

        bettime_frame = build_season_multibook_frame(base_2023_24, snapshots_all, "bettime", log)
        BETTIME_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
        bettime_frame.to_parquet(BETTIME_FRAME_PATH, index=False)
        log(
            "Saved bettime-pass 2023-24 multibook frame (bonus artifact, not a mandatory deliverable, "
            f"kept for reproducibility) to: {BETTIME_FRAME_PATH}"
        )

        metadata["frame_2023_24"] = {
            "closing_path": str(CLOSING_FRAME_PATH),
            "closing_rows": int(len(closing_frame)),
            "closing_goalie_nights": int(closing_frame[["game_id", "goalie_id"]].drop_duplicates().shape[0]),
            "closing_events": int(closing_frame["event_id"].nunique()),
            "closing_books": closing_frame["book_key"].value_counts().to_dict(),
            "bettime_path": str(BETTIME_FRAME_PATH),
            "bettime_rows": int(len(bettime_frame)),
            "bettime_goalie_nights": int(bettime_frame[["game_id", "goalie_id"]].drop_duplicates().shape[0]),
            "bettime_events": int(bettime_frame["event_id"].nunique()),
            "base_goalie_night_universe": int(base_2023_24[["game_id", "goalie_id"]].drop_duplicates().shape[0]),
        }
        flush_log()

        # ---- load the shared pace modeling frame (all seasons) once ----
        log("\n" + "=" * 80)
        log("Loading shared pace modeling frame (all seasons) for origin training")
        log("=" * 80)
        frame = epd.load_pace_modeling_frame(
            epd.DATA_PATH_CLEAN, epd.DATA_PATH_CONTEXT, epd.DATA_PATH_PACE, epd.DATA_PATH_PACE_METADATA, log,
        )
        df_bet_multibook_full = epd.build_betting_frame(epd.DATA_PATH_MULTIBOOK, log)

        # ---- ORIGIN A: train <= 2022-23, test = 2023-24 ----
        log("\n" + "=" * 80)
        log("ORIGIN A: train <= 2022-23, test = 2023-24")
        log("=" * 80)
        pool_min_a, pool_max_a = season_date_range(clean_full, [SEASON_2022_23])
        train_idx_a, val_idx_a, boundaries_a = carve_origin_split(
            frame.df, pool_min_a, pool_max_a, VAL_WINDOW_DAYS, log, "Origin A"
        )
        test_min_a, test_max_a = season_date_range(clean_full, [SEASON_2023_24])
        test_idx_a = date_range_test_idx(frame.df, test_min_a, test_max_a, log, "Origin A")

        df_bet_test_a_closing = epd.build_betting_frame(CLOSING_FRAME_PATH, log)
        df_bet_test_a_bettime = epd.build_betting_frame(BETTIME_FRAME_PATH, log)

        origin_a_result = run_origin(
            "A", frame, train_idx_a, val_idx_a, test_idx_a,
            {"closing": df_bet_test_a_closing, "bettime": df_bet_test_a_bettime},
            output_dir, log,
        )
        origin_a_result["fold_boundaries"] = {
            **boundaries_a, "test_season": SEASON_2023_24, "test_rows": int(len(test_idx_a)),
        }
        flush_log()

        # ---- ORIGIN B: train <= 2023-24, test = 2024-25 ----
        log("\n" + "=" * 80)
        log("ORIGIN B: train <= 2023-24, test = 2024-25")
        log("=" * 80)
        pool_min_b, pool_max_b = season_date_range(clean_full, [SEASON_2022_23, SEASON_2023_24])
        train_idx_b, val_idx_b, boundaries_b = carve_origin_split(
            frame.df, pool_min_b, pool_max_b, VAL_WINDOW_DAYS, log, "Origin B"
        )
        test_min_b, test_max_b = season_date_range(clean_full, [SEASON_2024_25])
        test_idx_b = date_range_test_idx(frame.df, test_min_b, test_max_b, log, "Origin B")

        df_bet_test_b_closing = df_bet_multibook_full[
            df_bet_multibook_full["season"] == SEASON_2024_25
        ].reset_index(drop=True)
        log(
            "Origin B test betting frame (multibook_classification_training_data.parquet, season "
            f"2024-25, closing lines from the odds cache): {len(df_bet_test_b_closing)} rows."
        )

        origin_b_result = run_origin(
            "B", frame, train_idx_b, val_idx_b, test_idx_b,
            {"closing": df_bet_test_b_closing},
            output_dir, log,
        )
        origin_b_result["fold_boundaries"] = {
            **boundaries_b, "test_season": SEASON_2024_25, "test_rows": int(len(test_idx_b)),
        }
        flush_log()

        metadata["origin_a"] = origin_a_result
        metadata["origin_b"] = origin_b_result
        metadata["fixed_ev_threshold"] = FIXED_EV_THRESHOLD
        metadata["caveat"] = (
            "Origin A trains on roughly one season (2022-23) plus 2021-22 pace-feature priors -- it "
            "tests the mechanism, not the production model. Small training data may honestly produce "
            "a weaker model; report whatever comes out."
        )
        metadata["design_notes"] = [
            "EV threshold fixed at 0.05 for both origins' test evaluation (the frozen production "
            "pace_shots threshold), not reselected via a validation sweep. Necessary because no "
            "betting-line odds exist anywhere for the 2022-23 season (the vendor's historical props "
            "archive starts 2023-05-03), so Origin A's validation window has no market data to sweep "
            "against. The same fixed threshold is applied to Origin B so both origins are evaluated "
            "identically.",
            "Validation windows are the last 49 days of each origin's training-pool seasons (matching "
            "the length of the production experiment's Oct16-Dec3 validation window), used only for "
            "shots/save-rate hyperparameter config selection (val MAE / weighted log-loss) -- no "
            "market data is touched during selection for either origin. This places val inside the "
            "training-pool seasons rather than inside the test season, which is required so the test "
            "season is touched exactly once.",
            "PRIMARY metrics use the 2023-24 / 2024-25 CLOSING pass for both origins. Origin A "
            "additionally reports a SECONDARY descriptive policy-ROI line on the 2023-24 BETTIME pass "
            "(realistic execution prices) per the task design.",
            f"PMF cap widened from the production CAP=70 to ORIGIN_CAP={ORIGIN_CAP} for both origins' "
            "distributional predictions (the wiring gate still uses CAP=70 exactly, to reproduce the "
            "frozen artifact bit-for-bit). Origin A's smaller training pool produced a wider shots "
            "model; one 2023-24 test-fold goalie-night (mu=42.47) lost >0.1% of its PMF mass above "
            "shots=70 and failed the harness's own normalization assert. The wider cap was chosen by "
            "checking PMF-sum normalization only, before any grading/ROI number was computed, and "
            "applied identically to both origins.",
        ]

        elapsed = time.time() - start_time
        metadata["wall_clock_seconds"] = elapsed
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        log(f"\nSaved metadata to: {metadata_path}")
        log(f"Wall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        log("\n" + "=" * 80)
        log("ROLLING-ORIGIN CONFIRMATION COMPLETE")
        log("=" * 80)
        flush_log()
    except Exception:
        flush_log()
        raise

    print(f"Saved run log to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
