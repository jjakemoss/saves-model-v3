"""
Experiment 8 -- Origin C market-state replication.

Contract: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 11 (11.1-11.11),
registered 2026-07-13 by the lead reviewer BEFORE any Origin C model or test
prediction existed anywhere in this repo. Read section 11 in full before
touching this script; section 1 (shared conventions) and section 10.1 item 1
(the dispersion-correction lesson from Experiment 3) also bind this run.

Hypothesis (11.1): Experiment 5's Origin B result -- market-state features
(Component C) produce a real paired-Brier improvement over the no-pace
control -- replicates on Origin C's test fold (season 2025-26), a fold no
model in this repo has ever been trained toward. Secondarily, the post-hoc
observation from section 10.1 item 5 (the market-state model's UNDER picks
beat blind-UNDER on the same quote universe on Origin B) generalizes to the
executable venue and window (BetOnline bettime).

Frozen recipe (11.2) -- nothing here is reselected. This script does not
reimplement the modeling recipe; it IMPORTS scripts/experiment_market_state_
features.py (the frozen Origin B run's own module) and calls its functions
directly: build_market_state_events, attach_market_state_features,
run_shots_variant, paired_brier_delta_vs_variant, MARKET_FEATURE_COLS,
MARKET_INDICATOR_COL, ALL_MARKET_COLS, VARIANT_NAMES. This is the only way
to guarantee "same script lineage" (11.2) rather than a hand-copied
approximation of it. The 104-column no-pace-control shots feature list and
the 7 mkt_* + mkt_matched columns are additionally asserted byte-for-byte
against models/trained/experiment_market_state_20260710_213106/metadata.json
before any model is trained (see assert_feature_identity below) -- a defense
against silent drift in clean_training_data.parquet's column set, not a
reselection.

Wiring gate (11.5, mandatory before any 2025-26 data is touched): this
script re-runs Origin B (train <= 2023-24, test = 2024-25) through the exact
same emsf.run_shots_variant code path used for Origin C below, and the
resulting closing-pass paired Brier delta (control_plus_market_state minus
no_pace_control) must match experiment_market_state_20260710_213106's
recorded Origin B number (mean -0.0041404240194266384, n_bets 7463,
n_clusters 2510) to within 1e-4 on the mean and exactly on n_bets/n_clusters.
If it does not reproduce, the script stops before loading any 2025-26 row.

Origin C folds (11.3): pool = seasons 2022-23 + 2023-24 + 2024-25 (carved via
experiment_rolling_origin.carve_origin_split, val = final 49 days of the
pool date range, train = the rest); test = season 20252026. 2025-26 rows
never enter train or val in any form -- enforced both structurally (the pool
date range is bounded by the three training seasons only, so 2025-26 dates
fall strictly after pool_max) and by an explicit runtime assertion.

Dispersion policy (11.9, per the Experiment 3 correction in section 10.1
item 1): HEADLINE results use validation-fitted NB2 dispersion, exactly
matching the frozen Origin B recipe (emsf.run_shots_variant always fits
alpha on val_idx, not train_idx -- see that module's docstring point 6).
Train-fitted dispersion is produced side-by-side as a pre-registered
sensitivity check, not an alternate headline: if either P1 or P2 flips sign
under train-fitted dispersion, the replication is reported as
DISPERSION-FRAGILE regardless of what the val-fitted numbers say. Because
emsf.run_shots_variant hardcodes val_idx internally and must not be edited
(no changes to any existing file, per task constraints), the train-fitted
sensitivity pass is produced here by RELOADING each variant's already-
trained shots-model JSON (bit-identical to the in-memory model used for the
headline pass -- XGBoost JSON round-trips exactly) and refitting alpha on
train_idx via the same shared experiments.distributional_saves.fit_dispersion
function, then repricing test-fold predictions under that alpha. A sanity
check asserts the reloaded val-fitted p_over matches emsf.run_shots_variant's
own in-memory p_over to within 1e-9 before any P1/P2 number is trusted.

Primary metrics (11.7):
  P1 (accuracy replication) -- paired Brier delta (market-state variant
  minus no-pace control), model probability at each posted line, CLOSING
  pass, ALL BOOKS, NON-PUSH rows, goalie-night cluster bootstrap. PASS =
  CI95 entirely below zero. emsf.paired_brier_delta_vs_variant does NOT
  exclude push rows (it treats saves==line as an UNDER win, matching Origin
  A/B precedent, where this was never flagged because pushes are rare); P1
  is computed by a small dedicated function in this script,
  paired_brier_delta_p1, using the identical squared-error/cluster-bootstrap
  math but with an explicit non-push filter, because 11.7 requires it
  explicitly. (On this data the two conventions turn out to coincide: the
  2025-26 closing pass has zero push rows, verified before running.)

  P2 (executable selection effect) -- on U = paired, gradeable betonlineag
  BETTIME quotes for test-fold games (pushes excluded): model arm = UNDER
  bets where the market-state model's EV(UNDER) >= 0.05 (raw vig-inclusive
  single-side implied probability, matching experiments.harness.decide_bet's
  literal selection code / section 1's "Selection at inference/pricing
  time" convention); blind arm = UNDER on every quote in U. A dedicated
  paired cluster bootstrap (p2_paired_bootstrap below -- resamples goalie-
  night clusters ONCE per resample, both arms computed from that same draw)
  reports delta = ROI_model - ROI_blind. PASS = CI95 entirely above zero.
  Empty-model-arm resamples and small-sample guards are implemented exactly
  per 11.7's text.

Data sources (11.4): clean_training_data.parquet, game_context_features.
parquet, pace_features.parquet (loaded but unused for features, matching
emsf's own no-pace convention), market_game_features.parquet (closing-pass
market-state join), multibook_classification_training_data.parquet season
20252026 (closing-pass saves quotes, 5,729 rows), saves_lines_snapshots.
parquet snapshot_pass=="bettime" (bettime-pass saves quotes). The bettime
frame is built with experiment_rolling_origin.build_season_multibook_frame
-- the SAME general-purpose pairing function (event_id/game_id/goalie_id/
book/line -> both-sides-paired rows via clv_audit_pace_policy.pivot_both_
sides) that scripts/experiment_cross_line_pricing.py's development-phase
pairing logic mirrors; reusing the already-generalized repo function is
preferred over hand-copying that script's version, per this repo's stated
convention of importing rather than reimplementing.

Book-key diagnosis (11.4, mandatory before grading): resolved by
diagnose_book_key_split below, using only the two parquet files already in
scope (never touching data/betting.db from this script -- see that
function's docstring for the full evidence trail and the code-provenance
reasoning that is cited, not re-executed, from scripts/build_multibook_
training_data.py and src/betting/odds_fetcher.py).

Zero network calls. All inputs are already on disk.

Usage:
    python scripts/experiment_market_state_origin_c.py
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

# Proven sys.path import pattern (scripts/experiment_season_funnel.py lines
# 76-135; same pattern used by clv_audit_pace_policy.py and
# experiment_rolling_origin.py). Nothing below is reimplemented that already
# exists in these modules.
import experiment_pace_distributional as epd  # noqa: E402
import experiment_rolling_origin as ero  # noqa: E402
import clv_audit_pace_policy as clv  # noqa: E402
import experiment_market_state_features as emsf  # noqa: E402
from experiments.distributional_saves import (  # noqa: E402
    SavesDistribution,
    build_betting_frame,
    compute_distribution_predictions,
    fit_dispersion,
    join_and_price,
    load_modeling_frame,
    train_save_rate_model,
)
from experiments.harness import (  # noqa: E402
    betting_metrics_bundle,
    decide_bet,
    grade_bets,
)
from betting.odds_utils import calculate_ev, calculate_payout  # noqa: E402
from betting.tracking_db import devig_prob  # noqa: E402

make_logger = epd.make_logger

DATA_PATH_CLEAN = REPO_ROOT / "data" / "processed" / "clean_training_data.parquet"
DATA_PATH_CONTEXT = REPO_ROOT / "data" / "processed" / "game_context_features.parquet"
DATA_PATH_MULTIBOOK = REPO_ROOT / "data" / "processed" / "multibook_classification_training_data.parquet"
MARKET_PATH = REPO_ROOT / "data" / "processed" / "market_game_features.parquet"
SNAPSHOTS_PATH = REPO_ROOT / "data" / "processed" / "saves_lines_snapshots.parquet"
FROZEN_MARKET_STATE_METADATA = (
    REPO_ROOT / "models" / "trained" / "experiment_market_state_20260710_213106" / "metadata.json"
)
OUTPUT_ROOT = REPO_ROOT / "models" / "trained"

SEASON_2025_26 = 20252026

FIXED_EV_THRESHOLD = ero.FIXED_EV_THRESHOLD  # 0.05, never reselected
ORIGIN_CAP = ero.ORIGIN_CAP  # 90
N_BOOTSTRAP_RESAMPLES = ero.N_BOOTSTRAP_RESAMPLES  # 10000
BOOTSTRAP_SEED = ero.BOOTSTRAP_SEED  # 42

WIRING_GATE_EXPECTED_MEAN = -0.0041404240194266384
WIRING_GATE_EXPECTED_N_BETS = 7463
WIRING_GATE_EXPECTED_N_CLUSTERS = 2510
WIRING_GATE_TOLERANCE = 1e-4

P2_MIN_MODEL_ARM_BETS = 100
P2_MAX_EMPTY_RESAMPLE_PCT = 1.0


# ---------------------------------------------------------------------------
# 11.4: betonline / betonlineag book_key diagnosis, BEFORE grading.
# ---------------------------------------------------------------------------


def diagnose_book_key_split(df_closing_c: pd.DataFrame, log) -> dict:
    """Diagnose the 'betonline' vs 'betonlineag' book_key split in the
    2025-26 closing pass (multibook_classification_training_data.parquet),
    per pre-reg 11.4. Uses ONLY the two parquet files already loaded by this
    script (df_closing_c here; saves_lines_snapshots.parquet's bettime-pass
    book coverage is checked separately in main()) -- this function never
    opens data/betting.db.

    Evidence trail (code-provenance reasoning, cited from source files
    already read during this script's preparation, not re-executed here):
    scripts/build_multibook_training_data.py builds this parquet from TWO
    sources. parse_odds_cache() reads the raw historical Odds-API JSON
    cache and uses the API's own bookmaker `key` field verbatim --
    src/betting/odds_fetcher.py's BOOKMAKER_DISPLAY_NAMES dict maps the API
    key 'betonlineag' -> display name 'BetOnline', confirming 'betonlineag'
    is the canonical Odds-API key for this book. parse_betting_db()
    supplements dates the raw cache does not cover by reading data/
    betting.db's live tracker rows and lowercasing whatever the user typed
    into the 'book' field (str(row['book']).lower()); a manually-recorded
    'BetOnline' entry normalizes to 'betonline' (missing the API's '-ag'
    suffix), never 'betonlineag'. The empirically observed pattern below
    (zero shared goalie-nights, non-overlapping back-to-back date windows)
    is exactly what that two-source construction predicts if both labels
    refer to the same underlying book.
    """
    log("\n" + "=" * 80)
    log("BOOK_KEY DIAGNOSIS: 'betonline' vs 'betonlineag' in the 2025-26 closing pass (pre-reg 11.4)")
    log("=" * 80)
    bo = df_closing_c[df_closing_c["book_key"] == "betonline"]
    boag = df_closing_c[df_closing_c["book_key"] == "betonlineag"]
    n_bo, n_boag = int(len(bo)), int(len(boag))
    bo_dates = (
        (str(pd.to_datetime(bo["game_date"]).min().date()), str(pd.to_datetime(bo["game_date"]).max().date()))
        if n_bo
        else (None, None)
    )
    boag_dates = (
        (str(pd.to_datetime(boag["game_date"]).min().date()), str(pd.to_datetime(boag["game_date"]).max().date()))
        if n_boag
        else (None, None)
    )
    bo_keys = set(zip(bo["game_id"].tolist(), bo["goalie_id"].tolist()))
    boag_keys = set(zip(boag["game_id"].tolist(), boag["goalie_id"].tolist()))
    overlap = bo_keys & boag_keys

    log(f"  betonline:   {n_bo} rows, {len(bo_keys)} distinct goalie-nights, date range {bo_dates}")
    log(f"  betonlineag: {n_boag} rows, {len(boag_keys)} distinct goalie-nights, date range {boag_dates}")
    log(f"  Overlapping (game_id, goalie_id) between the two labels: {len(overlap)}")

    contiguous = False
    if n_bo and n_boag:
        contiguous = (bo_dates[0] > boag_dates[1]) or (boag_dates[0] > bo_dates[1])
    log(f"  Date ranges non-overlapping (one label's window starts strictly after the other ends): {contiguous}")
    log(
        "  Source-code provenance (read, not executed, from scripts/build_multibook_training_data.py "
        "and src/betting/odds_fetcher.py -- see this function's docstring for the full trace): "
        "'betonlineag' comes from the raw Odds-API JSON cache using the API's own bookmaker key; "
        "'betonline' comes from the live betting-tracker-DB fallback used for dates the raw cache does "
        "not cover, lowercasing the user's manually-typed 'BetOnline' book label."
    )

    resolution = (
        "merge_same_book"
        if (len(overlap) == 0 and contiguous and n_bo > 0 and n_boag > 0)
        else "provenance_unclear_exclude_betonline"
    )
    log(f"  RESOLUTION: {resolution}")
    if resolution == "merge_same_book":
        log(
            "  Zero shared goalie-nights plus non-overlapping, essentially back-to-back date windows are "
            "consistent with a single underlying book (BetOnline / BetOnline.ag) captured through two "
            "different data-provenance paths, not two distinct venues. Both labels are treated as the "
            "SAME book for any BetOnline-specific secondary cut below (canonicalized under 'betonlineag'). "
            "This does NOT affect P1 (all-books, book-identity-agnostic) or P2 (betonlineag-only in the "
            "BETTIME pass, where 'betonline' never appears at all across the whole 2025-26 season -- "
            "verified separately against saves_lines_snapshots.parquet's bettime-pass book coverage)."
        )
    else:
        log(
            "  Provenance unclear -- excluding 'betonline' rows from any BetOnline-specific cut; only "
            "'betonlineag' rows are treated as venue-accessible BetOnline quotes in the closing pass."
        )

    return {
        "betonline_rows": n_bo,
        "betonlineag_rows": n_boag,
        "betonline_nights": len(bo_keys),
        "betonlineag_nights": len(boag_keys),
        "betonline_date_range": bo_dates,
        "betonlineag_date_range": boag_dates,
        "overlap_nights": len(overlap),
        "date_ranges_non_overlapping": bool(contiguous),
        "resolution": resolution,
    }


# ---------------------------------------------------------------------------
# P1: paired Brier delta, closing pass, all books, NON-PUSH rows.
# ---------------------------------------------------------------------------


def paired_brier_delta_p1(
    df_bet: pd.DataFrame,
    p_over_base: np.ndarray,
    matched_base: np.ndarray,
    p_over_new: np.ndarray,
    matched_new: np.ndarray,
    log,
    label: str,
) -> dict:
    """Pre-reg 11.7 P1: paired Brier delta (control_plus_market_state minus
    no_pace_control), model probability at each posted line, closing pass,
    ALL BOOKS, NON-PUSH rows, goalie-night cluster bootstrap 95% CI. Same
    squared-error/cluster-bootstrap math as emsf.paired_brier_delta_vs_
    variant, but with an explicit non-push filter (emsf's version does not
    exclude pushes; 11.7 requires it explicitly for P1)."""
    saves = df_bet["saves"].values.astype(float)
    lines = df_bet["betting_line"].values.astype(float)
    non_push = saves != lines
    n_push = int((~non_push).sum())
    both_matched = matched_base & matched_new & non_push
    y = (saves > lines).astype(float)
    sq_base = (p_over_base - y) ** 2
    sq_new = (p_over_new - y) ** 2
    delta = np.where(both_matched, sq_new - sq_base, np.nan)
    game_id = df_bet["game_id"].values
    goalie_id = df_bet["goalie_id"].values
    cluster_ids = np.array([f"{int(g)}_{int(o)}" for g, o in zip(game_id, goalie_id)], dtype=object)
    stat = clv.cluster_bootstrap_mean_ci(
        delta, cluster_ids, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, ci_pct=95.0
    )
    stat["n_push_excluded"] = n_push
    log(
        f"[{label}] P1 paired Brier delta (control_plus_market_state - no_pace_control), non-push: "
        f"mean={stat['mean']:+.6f} 95% CI=[{stat['lower']:+.6f}, {stat['upper']:+.6f}] "
        f"n_rows={stat['n_bets']} n_clusters={stat['n_clusters']} (n_push_excluded={n_push})"
    )
    return stat


# ---------------------------------------------------------------------------
# P2: paired cluster bootstrap of (model-arm UNDER ROI) - (blind-UNDER ROI)
# on the same resampled goalie-night clusters.
# ---------------------------------------------------------------------------


def p2_paired_bootstrap(
    cluster_ids: np.ndarray,
    profit: np.ndarray,
    is_model_arm: np.ndarray,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci_pct: float = 95.0,
) -> dict:
    """Pre-reg 11.7 P2. U is the full set of rows passed in (already
    filtered to paired/gradeable/non-push betonlineag bettime quotes by the
    caller). blind arm = every row in U (bet UNDER always); model arm =
    rows where is_model_arm is True (EV(UNDER) >= 0.05). Per resample:
    cluster indices are drawn ONCE and used for BOTH arms; delta = ROI_model
    - ROI_blind for that draw. Resamples where the model arm is empty after
    resampling are counted and excluded from the delta distribution (per
    11.7, reported separately as a stability diagnostic, not silently
    dropped)."""
    cluster_ids = np.asarray(cluster_ids, dtype=object)
    profit = np.asarray(profit, dtype=float)
    is_model_arm = np.asarray(is_model_arm, dtype=bool)
    n_total = len(profit)

    unique_clusters, inv = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)

    blind_sum = np.zeros(n_clusters)
    blind_cnt = np.zeros(n_clusters)
    np.add.at(blind_sum, inv, profit)
    np.add.at(blind_cnt, inv, 1)

    model_sum = np.zeros(n_clusters)
    model_cnt = np.zeros(n_clusters)
    if is_model_arm.any():
        np.add.at(model_sum, inv[is_model_arm], profit[is_model_arm])
        np.add.at(model_cnt, inv[is_model_arm], 1)

    n_model_bets = int(is_model_arm.sum())
    roi_blind_point = float(blind_sum.sum() / blind_cnt.sum() * 100.0) if blind_cnt.sum() > 0 else None
    roi_model_point = float(model_sum.sum() / model_cnt.sum() * 100.0) if model_cnt.sum() > 0 else None
    delta_point = (
        (roi_model_point - roi_blind_point) if (roi_model_point is not None and roi_blind_point is not None) else None
    )

    rng = np.random.RandomState(seed)
    deltas = []
    n_empty = 0
    for _ in range(n_resamples):
        draw = rng.randint(0, n_clusters, size=n_clusters)
        counts = np.bincount(draw, minlength=n_clusters)
        total_blind_profit = np.dot(counts, blind_sum)
        total_blind_n = np.dot(counts, blind_cnt)
        total_model_profit = np.dot(counts, model_sum)
        total_model_n = np.dot(counts, model_cnt)
        if total_model_n == 0:
            n_empty += 1
            continue
        roi_blind_b = total_blind_profit / total_blind_n * 100.0 if total_blind_n > 0 else np.nan
        roi_model_b = total_model_profit / total_model_n * 100.0
        deltas.append(roi_model_b - roi_blind_b)

    deltas_arr = np.asarray(deltas, dtype=float)
    deltas_arr = deltas_arr[~np.isnan(deltas_arr)]
    alpha = (100.0 - ci_pct) / 2.0
    if len(deltas_arr):
        ci_lower = float(np.percentile(deltas_arr, alpha))
        ci_upper = float(np.percentile(deltas_arr, 100.0 - alpha))
        boot_mean = float(np.mean(deltas_arr))
    else:
        ci_lower = ci_upper = boot_mean = None

    return {
        "n_universe_rows": int(n_total),
        "n_clusters_total": int(n_clusters),
        "n_model_arm_bets": n_model_bets,
        "roi_model_point": roi_model_point,
        "roi_blind_point": roi_blind_point,
        "mean": delta_point,
        "delta_mean_bootstrap": boot_mean,
        "lower": ci_lower,
        "upper": ci_upper,
        "n_resamples": n_resamples,
        "n_empty_model_arm_resamples": n_empty,
        "pct_empty_model_arm_resamples": n_empty / n_resamples * 100.0,
    }


def build_p2_universe(
    df_bettime: pd.DataFrame,
    p_under: np.ndarray,
    matched: np.ndarray,
    ev_threshold: float,
    log,
    label: str,
) -> pd.DataFrame:
    """U = paired, gradeable (matched to a test-fold model prediction),
    non-push rows of df_bettime, with per-row UNDER profit-if-bet, EV(UNDER)
    (raw vig-inclusive single-side implied probability, matching
    experiments.harness.decide_bet), and a goalie-night cluster id."""
    saves = df_bettime["saves"].values.astype(float)
    lines = df_bettime["betting_line"].values.astype(float)
    odds_under = df_bettime["odds_under_american"].astype(float).values
    non_push = saves != lines
    gradeable = matched & non_push
    n_push = int((~non_push).sum())
    n_unmatched = int((~matched).sum())
    log(
        f"[{label}] universe construction: {len(df_bettime)} rows -> {n_unmatched} unmatched to a "
        f"test-fold model prediction dropped, {n_push} push rows dropped -> {int(gradeable.sum())} "
        "gradeable rows remain."
    )
    u = df_bettime.loc[gradeable].copy()
    p_under_u = p_under[gradeable]
    odds_under_u = odds_under[gradeable]
    saves_u = saves[gradeable]
    lines_u = lines[gradeable]

    won_under = saves_u < lines_u
    profit = np.array(
        [calculate_payout(1.0, odds_under_u[i], bool(won_under[i])) for i in range(len(u))], dtype=float
    )
    ev_under = np.array([calculate_ev(float(p_under_u[i]), float(odds_under_u[i])) for i in range(len(u))], dtype=float)

    u = u.reset_index(drop=True)
    u["p_under_model"] = p_under_u
    u["profit_if_under"] = profit
    u["ev_under"] = ev_under
    u["is_model_arm"] = ev_under >= ev_threshold
    u["cluster_id"] = u["game_id"].astype(str) + "_" + u["goalie_id"].astype(str)
    log(
        f"[{label}] U = {len(u)} gradeable rows, {u['cluster_id'].nunique()} distinct goalie-nights; "
        f"model arm (EV(UNDER)>={ev_threshold}): {int(u['is_model_arm'].sum())} rows."
    )
    return u


# ---------------------------------------------------------------------------
# Secondary: OVER/UNDER ROI split on an arbitrary row subset.
# ---------------------------------------------------------------------------


def side_breakdown_for_subset(
    df_bet: pd.DataFrame,
    p_over: np.ndarray,
    p_under: np.ndarray,
    matched: np.ndarray,
    subset_mask: np.ndarray,
    threshold: float,
    log,
    label: str,
) -> dict:
    sub_matched = matched & subset_mask
    bet_results = grade_bets(
        p_over,
        p_under,
        df_bet["saves"].values.astype(float),
        df_bet["betting_line"].values.astype(float),
        df_bet["odds_over_american"].astype(float).values,
        df_bet["odds_under_american"].astype(float).values,
        df_bet["game_id"].values,
        df_bet["goalie_id"].values,
        threshold,
        sub_matched,
        log,
        label,
    )
    bundle = betting_metrics_bundle(
        bet_results, df_bet["game_id"].values, df_bet["goalie_id"].values, int(subset_mask.sum())
    )
    log(
        f"[{label}] {bundle['summary']['bets']} bets, {bundle['summary']['roi']:+.2f}% ROI -- "
        f"OVER {bundle['side_breakdown']['OVER']['bets']} bets ({bundle['side_breakdown']['OVER']['roi']:+.2f}%), "
        f"UNDER {bundle['side_breakdown']['UNDER']['bets']} bets ({bundle['side_breakdown']['UNDER']['roi']:+.2f}%)"
    )
    return bundle


# ---------------------------------------------------------------------------
# Secondary: bettime-to-close CLV of flagged bets, net of the unconditional
# drift baseline (same matched-quote convention as the Component G run,
# adapted from decimal-odds devig to this repo's American-odds devig_prob
# per section 1's stated CLV de-vig convention -- both are the identical
# additive-normalization formula, just different input representations).
# ---------------------------------------------------------------------------


def add_devig_cols(df: pd.DataFrame, log, label: str) -> pd.DataFrame:
    df = df.copy()
    pairs = [devig_prob(o, u) for o, u in zip(df["odds_over_american"].values, df["odds_under_american"].values)]
    df["devig_prob_over"] = [p[0] for p in pairs]
    df["devig_prob_under"] = [p[1] for p in pairs]
    n_fail = int(df["devig_prob_over"].isna().sum())
    if n_fail:
        log(f"[{label}] {n_fail}/{len(df)} rows failed de-vig (missing/degenerate odds).")
    return df


def build_closing_consensus(df_closing_devigged: pd.DataFrame, log) -> pd.DataFrame:
    consensus = (
        df_closing_devigged.groupby(["game_id", "goalie_id", "betting_line"])
        .agg(
            consensus_prob_over=("devig_prob_over", "mean"),
            consensus_prob_under=("devig_prob_under", "mean"),
            n_closing_books=("book_key", "nunique"),
        )
        .reset_index()
    )
    log(
        f"Closing consensus table (2025-26): {len(consensus)} distinct (game, goalie, line) entries "
        f"from {len(df_closing_devigged)} closing book-quote rows."
    )
    return consensus


def compute_drift_baseline(bettime_devigged: pd.DataFrame, closing_consensus: pd.DataFrame, log) -> dict:
    merged = bettime_devigged.merge(closing_consensus, on=["game_id", "goalie_id", "betting_line"], how="left")
    merged["cluster_id"] = merged["game_id"].astype(str) + "_" + merged["goalie_id"].astype(str)
    merged["drift_over"] = merged["consensus_prob_over"] - merged["devig_prob_over"]
    merged["drift_under"] = merged["consensus_prob_under"] - merged["devig_prob_under"]
    over_stat = clv.cluster_bootstrap_mean_ci(merged["drift_over"].values, merged["cluster_id"].values)
    under_stat = clv.cluster_bootstrap_mean_ci(merged["drift_under"].values, merged["cluster_id"].values)
    n_matched = int(merged["consensus_prob_over"].notna().sum())
    log(
        f"Unconditional 2025-26 bettime-to-close drift baseline: {n_matched}/{len(merged)} bettime "
        "quotes matched a closing consensus at their exact line."
    )
    log(f"  Drift OVER : mean={over_stat['mean']}  CI=[{over_stat['lower']}, {over_stat['upper']}]  n={over_stat['n_bets']}")
    log(f"  Drift UNDER: mean={under_stat['mean']}  CI=[{under_stat['lower']}, {under_stat['upper']}]  n={under_stat['n_bets']}")
    return {"over": over_stat, "under": under_stat, "n_matched": n_matched, "n_total_bettime_quotes": int(len(merged))}


def flag_bets_full_policy(
    df_bet: pd.DataFrame, p_over: np.ndarray, p_under: np.ndarray, matched: np.ndarray, threshold: float, log, label: str
) -> pd.DataFrame:
    n = len(df_bet)
    odds_o = df_bet["odds_over_american"].astype(float).values
    odds_u = df_bet["odds_under_american"].astype(float).values
    saves = df_bet["saves"].values.astype(float)
    lines = df_bet["betting_line"].values.astype(float)
    rows_idx, bet_sides = [], []
    for i in range(n):
        if not matched[i] or saves[i] == lines[i]:
            continue
        side, _ev = decide_bet(float(p_over[i]), float(p_under[i]), float(odds_o[i]), float(odds_u[i]), threshold)
        if side is None:
            continue
        rows_idx.append(i)
        bet_sides.append(side)
    flagged = df_bet.iloc[rows_idx].copy().reset_index(drop=True)
    flagged["bet_side"] = bet_sides
    log(f"[{label}] flagged {len(flagged)}/{n} rows using the full decide_bet policy (threshold={threshold}).")
    return flagged


def attach_clv(flagged_devigged: pd.DataFrame, closing_consensus: pd.DataFrame, drift_baseline: dict, log) -> pd.DataFrame:
    df = flagged_devigged.merge(closing_consensus, on=["game_id", "goalie_id", "betting_line"], how="left")
    df["closing_consensus_prob_chosen_side"] = np.where(
        df["bet_side"] == "OVER", df["consensus_prob_over"], df["consensus_prob_under"]
    )
    df["bettime_devig_prob_chosen_side"] = np.where(
        df["bet_side"] == "OVER", df["devig_prob_over"], df["devig_prob_under"]
    )
    df["clv_prob"] = df["closing_consensus_prob_chosen_side"] - df["bettime_devig_prob_chosen_side"]
    drift_over_mean = drift_baseline["over"]["mean"] if drift_baseline["over"]["mean"] is not None else 0.0
    drift_under_mean = drift_baseline["under"]["mean"] if drift_baseline["under"]["mean"] is not None else 0.0
    drift_scalar = np.where(df["bet_side"] == "OVER", drift_over_mean, drift_under_mean)
    df["clv_prob_net_of_drift"] = df["clv_prob"] - drift_scalar
    n_cov = int(df["clv_prob"].notna().sum())
    log(f"  CLV coverage: {n_cov}/{len(df)} flagged bettime bets had a closing consensus at their exact line.")
    return df


# ---------------------------------------------------------------------------
# Reload helper for the train-fitted dispersion sensitivity pass.
# ---------------------------------------------------------------------------


def reload_shots_model(model_path: str) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def assert_allclose(a: np.ndarray, b: np.ndarray, tol: float, label: str) -> None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = np.nanmax(np.abs(a - b)) if len(a) else 0.0
    if diff > tol:
        raise AssertionError(f"[{label}] reload sanity check FAILED: max abs diff {diff} exceeds tolerance {tol}.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_market_state_origin_c_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log, flush_log = make_logger(log_path)

    metadata: dict = {
        "timestamp": datetime.now().isoformat(),
        "registration": "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 11 (Experiment 8 -- Origin C market-state replication)",
        "deviations_from_registration": [],
    }
    try:
        log("=" * 80)
        log("EXPERIMENT 8 -- ORIGIN C MARKET-STATE REPLICATION")
        log("docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 11")
        log("=" * 80)
        log(f"Output directory: {output_dir}")

        for path in (
            DATA_PATH_CLEAN,
            DATA_PATH_CONTEXT,
            DATA_PATH_MULTIBOOK,
            MARKET_PATH,
            SNAPSHOTS_PATH,
            FROZEN_MARKET_STATE_METADATA,
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required input: {path}")

        frozen_meta = json.loads(FROZEN_MARKET_STATE_METADATA.read_text(encoding="utf-8"))

        # ---- STEP 1: market-state event table + no-pace modeling frame ----
        log("\n" + "=" * 80)
        log("STEP 1: build market-state event table (emsf.build_market_state_events, unchanged)")
        log("=" * 80)
        events, market_stats = emsf.build_market_state_events(MARKET_PATH, log)
        metadata["market_source_stats"] = market_stats

        log("\n" + "=" * 80)
        log("STEP 2: load no-pace control modeling frame + market-state join (emsf functions, unchanged)")
        log("=" * 80)
        frame = load_modeling_frame(DATA_PATH_CLEAN, DATA_PATH_CONTEXT, log)
        df_full = emsf.attach_market_state_features(frame.df, events, log)

        no_pace_cols = frame.base_feature_cols + frame.engineered_cols
        market_shots_cols = no_pace_cols + emsf.ALL_MARKET_COLS
        log(f"no_pace_control shots/rate feature count: {len(no_pace_cols)}")
        log(f"control_plus_market_state shots feature count: {len(market_shots_cols)}")

        # ---- 11.2: feature-identity check against the frozen recipe ----
        frozen_no_pace = frozen_meta["feature_sets"]["no_pace_control"]
        frozen_mkt_cols = frozen_meta["feature_sets"]["market_feature_cols"]
        frozen_mkt_indicator = frozen_meta["feature_sets"]["market_indicator_col"]
        identity_ok = (
            no_pace_cols == frozen_no_pace
            and emsf.MARKET_FEATURE_COLS == frozen_mkt_cols
            and emsf.MARKET_INDICATOR_COL == frozen_mkt_indicator
        )
        log(
            f"11.2 feature-identity check vs frozen {FROZEN_MARKET_STATE_METADATA}: "
            f"no_pace_control match={no_pace_cols == frozen_no_pace}, "
            f"market_feature_cols match={emsf.MARKET_FEATURE_COLS == frozen_mkt_cols}, "
            f"market_indicator_col match={emsf.MARKET_INDICATOR_COL == frozen_mkt_indicator}"
        )
        if not identity_ok:
            raise AssertionError(
                "11.2 feature-identity check FAILED: reconstructed feature lists do not match the "
                "frozen Origin B recipe. Stopping per task instructions rather than improvising."
            )
        metadata["feature_identity_check"] = {"passed": True, "no_pace_control_n": len(no_pace_cols)}
        log("Feature-identity check PASSED.")
        flush_log()

        df_bet_multibook_full = build_betting_frame(DATA_PATH_MULTIBOOK, log)

        # =====================================================================
        # WIRING GATE (11.5): re-run Origin B through this script's own code
        # path (emsf.run_shots_variant, the same function Origin C uses below)
        # and reproduce the frozen recorded closing-pass paired Brier delta.
        # =====================================================================
        log("\n" + "=" * 80)
        log("WIRING GATE (11.5): re-run Origin B through emsf.run_shots_variant, reproduce frozen numbers")
        log("=" * 80)

        pool_min_b, pool_max_b = ero.season_date_range(df_full, [ero.SEASON_2022_23, ero.SEASON_2023_24])
        train_idx_b, val_idx_b, boundaries_b = ero.carve_origin_split(
            df_full, pool_min_b, pool_max_b, ero.VAL_WINDOW_DAYS, log, "Origin B (wiring gate)"
        )
        test_min_b, test_max_b = ero.season_date_range(df_full, [ero.SEASON_2024_25])
        test_idx_b = ero.date_range_test_idx(df_full, test_min_b, test_max_b, log, "Origin B (wiring gate)")
        df_bet_test_b_closing = df_bet_multibook_full[df_bet_multibook_full["season"] == ero.SEASON_2024_25].reset_index(
            drop=True
        )

        rate_model_b, rate_winner_b, rate_evals_b = train_save_rate_model(
            df_full, train_idx_b, val_idx_b, no_pace_cols, log, "wiring_gate_origin_b_shared_save_rate"
        )

        variant_results_b = {}
        variant_probs_b = {}
        for variant_name in emsf.VARIANT_NAMES:
            shots_cols = no_pace_cols if variant_name == "no_pace_control" else market_shots_cols
            result_json, probs_by_pass, _row_predictions, _mu_test = emsf.run_shots_variant(
                "B_wiring_gate",
                variant_name,
                df_full,
                train_idx_b,
                val_idx_b,
                test_idx_b,
                shots_cols,
                no_pace_cols,
                rate_model_b,
                {"closing": df_bet_test_b_closing},
                output_dir,
                log,
            )
            variant_results_b[variant_name] = result_json
            variant_probs_b[variant_name] = probs_by_pass
            flush_log()

        base_p_over_b, base_matched_b = variant_probs_b["no_pace_control"]["closing"]
        new_p_over_b, new_matched_b = variant_probs_b["control_plus_market_state"]["closing"]
        wiring_stat = emsf.paired_brier_delta_vs_variant(
            df_bet_test_b_closing, base_p_over_b, base_matched_b, new_p_over_b, new_matched_b, log,
            "WIRING GATE origin_B closing",
        )

        gate_mean_diff = abs(wiring_stat["mean"] - WIRING_GATE_EXPECTED_MEAN)
        gate_mean_ok = gate_mean_diff <= WIRING_GATE_TOLERANCE
        gate_n_bets_ok = wiring_stat["n_bets"] == WIRING_GATE_EXPECTED_N_BETS
        gate_n_clusters_ok = wiring_stat["n_clusters"] == WIRING_GATE_EXPECTED_N_CLUSTERS
        gate_passed = gate_mean_ok and gate_n_bets_ok and gate_n_clusters_ok

        log("\n--- WIRING GATE RESULT ---")
        log(f"  Expected: mean={WIRING_GATE_EXPECTED_MEAN}, n_bets={WIRING_GATE_EXPECTED_N_BETS}, n_clusters={WIRING_GATE_EXPECTED_N_CLUSTERS}")
        log(f"  Observed: mean={wiring_stat['mean']}, n_bets={wiring_stat['n_bets']}, n_clusters={wiring_stat['n_clusters']}")
        log(f"  |mean diff|={gate_mean_diff:.10f} (tolerance {WIRING_GATE_TOLERANCE}) -> {'OK' if gate_mean_ok else 'FAIL'}")
        log(f"  n_bets match: {gate_n_bets_ok}; n_clusters match: {gate_n_clusters_ok}")
        log(f"  WIRING GATE: {'PASSED' if gate_passed else 'FAILED'}")

        metadata["wiring_gate"] = {
            "expected": {
                "mean": WIRING_GATE_EXPECTED_MEAN,
                "n_bets": WIRING_GATE_EXPECTED_N_BETS,
                "n_clusters": WIRING_GATE_EXPECTED_N_CLUSTERS,
            },
            "observed": {"mean": wiring_stat["mean"], "n_bets": wiring_stat["n_bets"], "n_clusters": wiring_stat["n_clusters"]},
            "mean_abs_diff": gate_mean_diff,
            "tolerance": WIRING_GATE_TOLERANCE,
            "passed": gate_passed,
        }
        flush_log()

        if not gate_passed:
            log("\nWIRING GATE FAILED. Per task instructions: STOP. Do not touch 2025-26 data.")
            metadata["stopped_at"] = "wiring_gate"
            metadata_path = output_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            log(f"\nSaved metadata (wiring-gate-failure record) to: {metadata_path}")
            flush_log()
            print(f"Saved run log to: {log_path}")
            return 1

        log("\nWiring gate passed. Proceeding to Origin C (season 2025-26 test fold).")

        # =====================================================================
        # ORIGIN C (11.3): pool = 2022-23 + 2023-24 + 2024-25, test = 2025-26.
        # =====================================================================
        log("\n" + "=" * 80)
        log("ORIGIN C: pool = 2022-23 + 2023-24 + 2024-25, test = season 2025-26")
        log("=" * 80)

        pool_min_c, pool_max_c = ero.season_date_range(
            df_full, [ero.SEASON_2022_23, ero.SEASON_2023_24, ero.SEASON_2024_25]
        )
        train_idx_c, val_idx_c, boundaries_c = ero.carve_origin_split(
            df_full, pool_min_c, pool_max_c, ero.VAL_WINDOW_DAYS, log, "Origin C"
        )
        test_min_c, test_max_c = ero.season_date_range(df_full, [SEASON_2025_26])
        test_idx_c = ero.date_range_test_idx(df_full, test_min_c, test_max_c, log, "Origin C")

        train_seasons_c = set(df_full["season"].values[train_idx_c].tolist())
        val_seasons_c = set(df_full["season"].values[val_idx_c].tolist())
        test_seasons_c = set(df_full["season"].values[test_idx_c].tolist())
        if SEASON_2025_26 in train_seasons_c or SEASON_2025_26 in val_seasons_c:
            raise AssertionError("2025-26 rows leaked into Origin C train/val -- refusing to proceed.")
        if test_seasons_c != {SEASON_2025_26}:
            raise AssertionError(f"Origin C test fold is not exactly season 2025-26: {test_seasons_c}")
        log(f"Fence check passed: train seasons={sorted(train_seasons_c)}, val seasons={sorted(val_seasons_c)}, test seasons={sorted(test_seasons_c)}")
        log(f"Origin C test fold size: {len(test_idx_c)} rows (expected 2,624 per 11.3).")

        join_coverage_c = emsf.report_join_coverage(
            df_full, {"Origin C": {"train": train_idx_c, "val": val_idx_c, "test": test_idx_c}}, log
        )
        metadata["join_coverage_origin_c"] = join_coverage_c
        flush_log()

        # ---- betting frames: closing (existing parquet) + bettime (built) ----
        df_bet_test_c_closing = df_bet_multibook_full[df_bet_multibook_full["season"] == SEASON_2025_26].reset_index(
            drop=True
        )
        log(f"\nOrigin C closing betting frame (multibook_classification_training_data.parquet, season 2025-26): "
            f"{len(df_bet_test_c_closing)} rows.")

        book_resolution = diagnose_book_key_split(df_bet_test_c_closing, log)
        metadata["book_key_resolution"] = book_resolution
        flush_log()

        clean_full = pd.read_parquet(DATA_PATH_CLEAN)
        clean_full["game_date"] = pd.to_datetime(clean_full["game_date"])
        base_2025_26 = clean_full[clean_full["season"] == SEASON_2025_26].reset_index(drop=True)
        snapshots_all = pd.read_parquet(SNAPSHOTS_PATH)
        log(f"\nsaves_lines_snapshots.parquet: {len(snapshots_all)} total rows across all fetched windows.")

        log("\nBuilding Origin C bettime betting frame via experiment_rolling_origin.build_season_multibook_frame "
            "(the reference pairing implementation, reused unchanged -- see module docstring).")
        bettime_frame_c = ero.build_season_multibook_frame(base_2025_26, snapshots_all, "bettime", log)
        bettime_path = output_dir / "origin_c_bettime_frame_allbooks.parquet"
        bettime_frame_c.to_parquet(bettime_path, index=False)
        log(f"Saved Origin C all-books bettime frame to: {bettime_path}")

        # sanity: 'betonline' (without -ag) must never appear in the bettime
        # snapshot pipeline -- confirms P2's betonlineag-only universe is
        # unaffected by the closing-pass book_key ambiguity.
        bettime_book_keys = set(bettime_frame_c["book_key"].unique().tolist())
        log(f"Origin C bettime frame book_key values: {sorted(bettime_book_keys)}")
        if "betonline" in bettime_book_keys:
            raise AssertionError(
                "'betonline' (without -ag) unexpectedly appears in the bettime pass; the P2 universe "
                "definition (betonlineag only) needs re-examination before grading."
            )
        metadata["bettime_frame_book_keys"] = sorted(bettime_book_keys)
        flush_log()

        # ---- shared save-rate model ----
        rate_model_c, rate_winner_c, rate_evals_c = train_save_rate_model(
            df_full, train_idx_c, val_idx_c, no_pace_cols, log, "origin_c_shared_save_rate"
        )
        rate_path_c = output_dir / "origin_c_shared_save_rate_model.json"
        rate_model_c.get_booster().save_model(str(rate_path_c))
        log(f"Saved Origin C shared save-rate model to: {rate_path_c}")

        # ---- both variants, val-fitted dispersion (headline, per 11.9) ----
        variant_results_c = {}
        variant_probs_c = {}
        all_row_frames = []
        for variant_name in emsf.VARIANT_NAMES:
            shots_cols = no_pace_cols if variant_name == "no_pace_control" else market_shots_cols
            result_json, probs_by_pass, row_predictions, _mu_test = emsf.run_shots_variant(
                "C",
                variant_name,
                df_full,
                train_idx_c,
                val_idx_c,
                test_idx_c,
                shots_cols,
                no_pace_cols,
                rate_model_c,
                {"closing": df_bet_test_c_closing, "bettime": bettime_frame_c},
                output_dir,
                log,
            )
            variant_results_c[variant_name] = result_json
            variant_probs_c[variant_name] = probs_by_pass
            all_row_frames.append(row_predictions)
            flush_log()

        predictions_df = pd.concat(all_row_frames, ignore_index=True)
        predictions_path = output_dir / "origin_c_test_predictions.parquet"
        predictions_df.to_parquet(predictions_path, index=False)
        log(f"\nSaved {len(predictions_df)} per-row Origin C test predictions (both variants, closing+bettime) to: {predictions_path}")

        # ---- reload models, rebuild p_under (needed for P2), sanity-check ----
        log("\n" + "=" * 80)
        log("Reloading trained shots models to derive p_under (val-fitted) and the train-fitted sensitivity pass")
        log("=" * 80)

        dist = SavesDistribution(ORIGIN_CAP)
        reload_val = {}
        reload_train = {}
        for variant_name in emsf.VARIANT_NAMES:
            shots_cols = no_pace_cols if variant_name == "no_pace_control" else market_shots_cols
            result_json = variant_results_c[variant_name]
            shots_path = result_json["shots_model"]["model_path"]
            alpha_val = result_json["dispersion"]["alpha"]
            shots_model_reloaded = reload_shots_model(shots_path)

            dist_preds_val = compute_distribution_predictions(
                df_full, test_idx_c, shots_model_reloaded, rate_model_c, alpha_val, shots_cols, no_pace_cols, dist, log,
                f"origin_c_{variant_name} reload VAL-FITTED",
            )
            p_over_closing_val, p_under_closing_val, _pp, matched_closing_val, _cov = join_and_price(
                df_bet_test_c_closing, dist_preds_val, dist, log, f"origin_c_{variant_name} closing reload (val-fitted)"
            )
            p_over_bettime_val, p_under_bettime_val, _pp2, matched_bettime_val, _cov2 = join_and_price(
                bettime_frame_c, dist_preds_val, dist, log, f"origin_c_{variant_name} bettime reload (val-fitted)"
            )

            # Sanity check: reloaded p_over must match emsf.run_shots_variant's
            # own in-memory p_over exactly (same model, same alpha, same idx).
            orig_p_over_closing, orig_matched_closing = variant_probs_c[variant_name]["closing"]
            assert_allclose(p_over_closing_val, orig_p_over_closing, 1e-9, f"{variant_name} closing reload vs original")
            assert (matched_closing_val == orig_matched_closing).all(), f"{variant_name} closing matched-mask mismatch on reload."
            orig_p_over_bettime, orig_matched_bettime = variant_probs_c[variant_name]["bettime"]
            assert_allclose(p_over_bettime_val, orig_p_over_bettime, 1e-9, f"{variant_name} bettime reload vs original")
            assert (matched_bettime_val == orig_matched_bettime).all(), f"{variant_name} bettime matched-mask mismatch on reload."
            log(f"[{variant_name}] reload sanity check PASSED (closing + bettime p_over match to 1e-9).")

            reload_val[variant_name] = {
                "alpha": alpha_val,
                "p_over_closing": p_over_closing_val,
                "p_under_closing": p_under_closing_val,
                "matched_closing": matched_closing_val,
                "p_over_bettime": p_over_bettime_val,
                "p_under_bettime": p_under_bettime_val,
                "matched_bettime": matched_bettime_val,
            }

            # ---- train-fitted dispersion sensitivity pass (11.9) ----
            alpha_train, method_train, diag_train = fit_dispersion(
                shots_model_reloaded, df_full, train_idx_c, shots_cols, log, f"origin_c_{variant_name} TRAIN-FITTED sensitivity"
            )
            dist_preds_train = compute_distribution_predictions(
                df_full, test_idx_c, shots_model_reloaded, rate_model_c, alpha_train, shots_cols, no_pace_cols, dist, log,
                f"origin_c_{variant_name} reload TRAIN-FITTED",
            )
            p_over_closing_train, p_under_closing_train, _pp3, matched_closing_train, _cov3 = join_and_price(
                df_bet_test_c_closing, dist_preds_train, dist, log, f"origin_c_{variant_name} closing (train-fitted)"
            )
            p_over_bettime_train, p_under_bettime_train, _pp4, matched_bettime_train, _cov4 = join_and_price(
                bettime_frame_c, dist_preds_train, dist, log, f"origin_c_{variant_name} bettime (train-fitted)"
            )
            reload_train[variant_name] = {
                "alpha": alpha_train,
                "method": method_train,
                "diagnostics": diag_train,
                "p_over_closing": p_over_closing_train,
                "p_under_closing": p_under_closing_train,
                "matched_closing": matched_closing_train,
                "p_over_bettime": p_over_bettime_train,
                "p_under_bettime": p_under_bettime_train,
                "matched_bettime": matched_bettime_train,
            }
            flush_log()

        # =====================================================================
        # P1 (11.7): paired Brier delta, closing pass, all books, non-push.
        # =====================================================================
        log("\n" + "=" * 80)
        log("P1: paired Brier delta (control_plus_market_state - no_pace_control), closing, all books, non-push")
        log("=" * 80)
        p1_val = paired_brier_delta_p1(
            df_bet_test_c_closing,
            reload_val["no_pace_control"]["p_over_closing"], reload_val["no_pace_control"]["matched_closing"],
            reload_val["control_plus_market_state"]["p_over_closing"], reload_val["control_plus_market_state"]["matched_closing"],
            log, "P1 VAL-FITTED (headline)",
        )
        p1_train = paired_brier_delta_p1(
            df_bet_test_c_closing,
            reload_train["no_pace_control"]["p_over_closing"], reload_train["no_pace_control"]["matched_closing"],
            reload_train["control_plus_market_state"]["p_over_closing"], reload_train["control_plus_market_state"]["matched_closing"],
            log, "P1 TRAIN-FITTED (sensitivity)",
        )
        p1_pass = p1_val["upper"] is not None and p1_val["upper"] < 0
        log(f"P1 verdict (val-fitted headline): {'PASS' if p1_pass else 'FAIL'} (CI95 entirely below zero required).")

        # =====================================================================
        # P2 (11.7): paired resample selection-over-blind-UNDER delta on the
        # betonlineag bettime universe.
        # =====================================================================
        log("\n" + "=" * 80)
        log("P2: selection-over-blind-UNDER delta, betonlineag bettime universe")
        log("=" * 80)

        u_val_allbooks = build_p2_universe(
            bettime_frame_c, reload_val["control_plus_market_state"]["p_under_bettime"],
            reload_val["control_plus_market_state"]["matched_bettime"], FIXED_EV_THRESHOLD, log, "P2 VAL-FITTED all-books",
        )
        u_val_bo = u_val_allbooks[u_val_allbooks["book_key"] == "betonlineag"].reset_index(drop=True)
        log(f"P2 primary universe (betonlineag only, val-fitted): {len(u_val_bo)} rows, "
            f"{u_val_bo['cluster_id'].nunique()} goalie-nights.")

        p2_val_primary = p2_paired_bootstrap(
            u_val_bo["cluster_id"].values, u_val_bo["profit_if_under"].values, u_val_bo["is_model_arm"].values
        )
        p2_val_allbooks_secondary = p2_paired_bootstrap(
            u_val_allbooks["cluster_id"].values, u_val_allbooks["profit_if_under"].values, u_val_allbooks["is_model_arm"].values
        )
        log(
            f"P2 PRIMARY (val-fitted, betonlineag): n_model_arm_bets={p2_val_primary['n_model_arm_bets']} "
            f"roi_model={p2_val_primary['roi_model_point']} roi_blind={p2_val_primary['roi_blind_point']} "
            f"delta={p2_val_primary['mean']} CI95=[{p2_val_primary['lower']}, {p2_val_primary['upper']}] "
            f"empty_resamples={p2_val_primary['n_empty_model_arm_resamples']} "
            f"({p2_val_primary['pct_empty_model_arm_resamples']:.3f}%)"
        )
        log(
            f"P2 SECONDARY (val-fitted, all books): n_model_arm_bets={p2_val_allbooks_secondary['n_model_arm_bets']} "
            f"delta={p2_val_allbooks_secondary['mean']} "
            f"CI95=[{p2_val_allbooks_secondary['lower']}, {p2_val_allbooks_secondary['upper']}]"
        )

        u_train_allbooks = build_p2_universe(
            bettime_frame_c, reload_train["control_plus_market_state"]["p_under_bettime"],
            reload_train["control_plus_market_state"]["matched_bettime"], FIXED_EV_THRESHOLD, log,
            "P2 TRAIN-FITTED all-books (sensitivity)",
        )
        u_train_bo = u_train_allbooks[u_train_allbooks["book_key"] == "betonlineag"].reset_index(drop=True)
        p2_train_primary = p2_paired_bootstrap(
            u_train_bo["cluster_id"].values, u_train_bo["profit_if_under"].values, u_train_bo["is_model_arm"].values
        )
        log(
            f"P2 PRIMARY (train-fitted sensitivity, betonlineag): n_model_arm_bets={p2_train_primary['n_model_arm_bets']} "
            f"delta={p2_train_primary['mean']} CI95=[{p2_train_primary['lower']}, {p2_train_primary['upper']}]"
        )

        if p2_val_primary["n_model_arm_bets"] < P2_MIN_MODEL_ARM_BETS:
            p2_verdict = "INSUFFICIENT_SAMPLE"
        elif p2_val_primary["pct_empty_model_arm_resamples"] > P2_MAX_EMPTY_RESAMPLE_PCT:
            p2_verdict = "UNSTABLE"
        elif p2_val_primary["lower"] is not None and p2_val_primary["lower"] > 0:
            p2_verdict = "PASS"
        else:
            p2_verdict = "FAIL"
        log(f"P2 verdict (val-fitted headline): {p2_verdict}")

        # =====================================================================
        # 11.9 dispersion sensitivity: does either primary flip sign?
        # =====================================================================
        def _sign_flip(a, b):
            if a is None or b is None or a == 0 or b == 0:
                return False
            return (a > 0) != (b > 0)

        flip_p1 = _sign_flip(p1_val["mean"], p1_train["mean"])
        flip_p2 = _sign_flip(p2_val_primary["mean"], p2_train_primary["mean"])
        dispersion_fragile = bool(flip_p1) or bool(flip_p2)
        log(
            f"\n11.9 dispersion sensitivity: P1 sign flip={flip_p1} (val mean={p1_val['mean']:+.6f}, "
            f"train mean={p1_train['mean']:+.6f}); P2 sign flip={flip_p2} (val mean={p2_val_primary['mean']}, "
            f"train mean={p2_train_primary['mean']}) -> DISPERSION-FRAGILE={dispersion_fragile}"
        )

        metadata["P1"] = {"val_fitted_headline": p1_val, "train_fitted_sensitivity": p1_train, "pass": p1_pass}
        metadata["P2"] = {
            "val_fitted_primary_betonlineag": p2_val_primary,
            "val_fitted_secondary_allbooks": p2_val_allbooks_secondary,
            "train_fitted_primary_betonlineag_sensitivity": p2_train_primary,
            "verdict": p2_verdict,
        }
        metadata["dispersion_sensitivity"] = {
            "p1_sign_flip": flip_p1,
            "p2_sign_flip": flip_p2,
            "dispersion_fragile": dispersion_fragile,
        }
        flush_log()

        # =====================================================================
        # SECONDARIES (11.8) -- report, no gate.
        # =====================================================================
        log("\n" + "=" * 80)
        log("SECONDARIES (11.8) -- report only, no gate")
        log("=" * 80)

        secondaries: dict = {}

        # (a) Brier vs de-vigged market, closing + bettime, control_plus_market_state
        secondaries["brier_vs_market_closing"] = variant_results_c["control_plus_market_state"]["price_passes"]["closing"][
            "paired_brier_delta_vs_market"
        ]
        secondaries["brier_vs_market_bettime_allbooks"] = variant_results_c["control_plus_market_state"]["price_passes"][
            "bettime"
        ]["paired_brier_delta_vs_market"]

        # (b) OVER/UNDER splits, both passes, all-books (already computed inside run_shots_variant)
        secondaries["over_under_split_closing_allbooks"] = variant_results_c["control_plus_market_state"]["price_passes"][
            "closing"
        ]["policy_roi"]["side_breakdown"]
        secondaries["over_under_split_bettime_allbooks"] = variant_results_c["control_plus_market_state"]["price_passes"][
            "bettime"
        ]["policy_roi"]["side_breakdown"]

        # BetOnline cuts
        if book_resolution["resolution"] == "merge_same_book":
            betonline_labels_closing = {"betonline", "betonlineag"}
        else:
            betonline_labels_closing = {"betonlineag"}
        mask_betonline_closing = df_bet_test_c_closing["book_key"].isin(betonline_labels_closing).values
        mask_betonline_bettime = (bettime_frame_c["book_key"] == "betonlineag").values

        secondaries["over_under_split_closing_betonline_cut"] = side_breakdown_for_subset(
            df_bet_test_c_closing,
            reload_val["control_plus_market_state"]["p_over_closing"],
            reload_val["control_plus_market_state"]["p_under_closing"],
            reload_val["control_plus_market_state"]["matched_closing"],
            mask_betonline_closing, FIXED_EV_THRESHOLD, log, "closing BetOnline cut",
        )
        secondaries["over_under_split_bettime_betonline_cut"] = side_breakdown_for_subset(
            bettime_frame_c,
            reload_val["control_plus_market_state"]["p_over_bettime"],
            reload_val["control_plus_market_state"]["p_under_bettime"],
            reload_val["control_plus_market_state"]["matched_bettime"],
            mask_betonline_bettime, FIXED_EV_THRESHOLD, log, "bettime BetOnline(ag) cut",
        )

        # (c) all-books bettime selection-over-blind delta -- already computed as p2_val_allbooks_secondary
        secondaries["bettime_selection_over_blind_allbooks"] = p2_val_allbooks_secondary

        # (d) bettime-to-close CLV of flagged bets, net of unconditional drift baseline
        log("\n--- CLV net of drift (secondary) ---")
        closing_devigged = add_devig_cols(df_bet_test_c_closing, log, "closing devig")
        bettime_devigged = add_devig_cols(bettime_frame_c, log, "bettime devig")
        closing_consensus = build_closing_consensus(closing_devigged, log)
        drift_baseline = compute_drift_baseline(bettime_devigged, closing_consensus, log)

        flagged_bettime = flag_bets_full_policy(
            bettime_frame_c,
            reload_val["control_plus_market_state"]["p_over_bettime"],
            reload_val["control_plus_market_state"]["p_under_bettime"],
            reload_val["control_plus_market_state"]["matched_bettime"],
            FIXED_EV_THRESHOLD, log, "flagged bettime bets (all books)",
        )
        flagged_devigged = add_devig_cols(flagged_bettime, log, "flagged bettime devig")
        flagged_clv = attach_clv(flagged_devigged, closing_consensus, drift_baseline, log)
        flagged_clv_cluster = flagged_clv["game_id"].astype(str) + "_" + flagged_clv["goalie_id"].astype(str)
        clv_net_allbooks = clv.cluster_bootstrap_mean_ci(
            flagged_clv["clv_prob_net_of_drift"].values, flagged_clv_cluster.values
        )
        log(f"Flagged-bet CLV net of drift (all books): {clv_net_allbooks}")

        flagged_clv_bo = flagged_clv[flagged_clv["book_key"] == "betonlineag"]
        clv_net_betonline = clv.cluster_bootstrap_mean_ci(
            flagged_clv_bo["clv_prob_net_of_drift"].values,
            (flagged_clv_bo["game_id"].astype(str) + "_" + flagged_clv_bo["goalie_id"].astype(str)).values,
        )
        log(f"Flagged-bet CLV net of drift (betonlineag only): {clv_net_betonline}")

        secondaries["clv_net_of_drift"] = {
            "drift_baseline": drift_baseline,
            "n_flagged_bettime_bets_allbooks": int(len(flagged_clv)),
            "clv_net_of_drift_allbooks": clv_net_allbooks,
            "n_flagged_bettime_bets_betonlineag": int(len(flagged_clv_bo)),
            "clv_net_of_drift_betonlineag": clv_net_betonline,
        }

        # (e) shots-model signed bias and MAE, both variants, test fold
        secondaries["shots_bias_mae"] = {
            v: variant_results_c[v]["workload_shots_against_test"] for v in emsf.VARIANT_NAMES
        }

        # (f) join coverage by fold (market-state join + betting-frame join)
        secondaries["join_coverage_market_state"] = join_coverage_c
        secondaries["join_coverage_betting_closing"] = {
            v: variant_results_c[v]["price_passes"]["closing"]["join_coverage_pct"] for v in emsf.VARIANT_NAMES
        }
        secondaries["join_coverage_betting_bettime"] = {
            v: variant_results_c[v]["price_passes"]["bettime"]["join_coverage_pct"] for v in emsf.VARIANT_NAMES
        }

        metadata["secondaries"] = secondaries
        flush_log()

        # =====================================================================
        # Consequence mapping (11.11) -- fixed in advance, applied mechanically.
        # =====================================================================
        log("\n" + "=" * 80)
        log("CONSEQUENCE MAPPING (11.11)")
        log("=" * 80)
        if dispersion_fragile:
            consequence = "DISPERSION-FRAGILE: at least one primary flips sign under train-fitted dispersion. Not a clean pass regardless of P1/P2 individually."
        elif p1_pass and p2_verdict == "PASS":
            consequence = "PASS: promoted per plan step 6e -- 2026-27 shadow/token-stake candidacy; 2024-25 bettime-pass purchase worth reconsidering."
        elif not p1_pass:
            consequence = "FAIL of P1: Experiment 5's Origin B result is treated as origin-specific; Component C drops out of the front of the queue."
        elif p1_pass and p2_verdict == "INSUFFICIENT_SAMPLE":
            consequence = "INSUFFICIENT SAMPLE on P2: report; closing-pass all-books selection delta (secondary) informs but does not decide a bettime re-test on 2026-27 live data."
        elif p1_pass and p2_verdict in ("FAIL", "UNSTABLE"):
            consequence = "FAIL of P2 alone (P1 passing): accuracy gain real, executable-venue selection effect not demonstrated at bettime. No purchase, no promotion, revisit with 6b/6c architecture work."
        else:
            consequence = "Unclassified combination -- report raw P1/P2/dispersion results for manual review."
        log(consequence)
        metadata["consequence"] = consequence

        # =====================================================================
        # Fold boundaries, feature sets, etc.
        # =====================================================================
        metadata["origin_c_fold_boundaries"] = {
            **boundaries_c, "test_season": SEASON_2025_26, "test_rows": int(len(test_idx_c)),
        }
        metadata["origin_b_wiring_gate_fold_boundaries"] = {
            **boundaries_b, "test_season": ero.SEASON_2024_25, "test_rows": int(len(test_idx_b)),
        }
        metadata["feature_sets"] = {
            "no_pace_control": no_pace_cols,
            "market_feature_cols": emsf.MARKET_FEATURE_COLS,
            "market_indicator_col": emsf.MARKET_INDICATOR_COL,
        }
        metadata["fixed_ev_threshold"] = FIXED_EV_THRESHOLD
        metadata["origin_cap"] = ORIGIN_CAP
        metadata["origin_c_variants"] = variant_results_c
        metadata["origin_c_dispersion_reload"] = {
            v: {
                "alpha_val_fitted": reload_val[v]["alpha"],
                "alpha_train_fitted": reload_train[v]["alpha"],
                "train_fitted_method": reload_train[v]["method"],
                "train_fitted_diagnostics": reload_train[v]["diagnostics"],
            }
            for v in emsf.VARIANT_NAMES
        }
        metadata["deviations_from_registration"] = [
            "11.7 P1 is computed with a dedicated function (paired_brier_delta_p1) rather than reusing "
            "emsf.paired_brier_delta_vs_variant verbatim, because 11.7 requires non-push exclusion and "
            "emsf's helper does not exclude pushes (matches Origin A/B precedent, where this was never "
            "flagged since pushes are rare). On this data the two conventions coincide: the 2025-26 "
            "closing pass has zero push rows, verified before running -- so this is a definitional "
            "correction with no numeric effect this round, not a substantive deviation.",
            "The CLV-net-of-drift secondary (11.8) adapts scripts/experiment_cross_line_pricing.py's "
            "decimal-odds devig_pair_decimal to this repo's American-odds betting.tracking_db.devig_prob "
            "(section 1's stated CLV de-vig convention) -- both are the identical additive-normalization "
            "formula on the same two-sided quote, just different input representations (decimal vs "
            "American odds); this is a representation match, not a methodology change.",
            "The train-fitted dispersion sensitivity pass (11.9) reloads each variant's already-trained "
            "shots-model JSON rather than retraining, because emsf.run_shots_variant hardcodes val_idx "
            "internally and may not be edited (no changes to any existing file, per task constraints). "
            "A sanity check (assert_allclose, 1e-9 tolerance) confirms the reloaded val-fitted p_over "
            "matches emsf.run_shots_variant's own in-memory p_over exactly before any P1/P2 number "
            "derived from the reload is trusted.",
            "The book_key diagnosis (11.4) reads only multibook_classification_training_data.parquet and "
            "saves_lines_snapshots.parquet at runtime; it never opens data/betting.db (the code-provenance "
            "evidence citing scripts/build_multibook_training_data.py and src/betting/odds_fetcher.py is "
            "documented reasoning from reading those source files during preparation, not a runtime query).",
        ]

        elapsed = time.time() - start_time
        metadata["wall_clock_seconds"] = elapsed
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        log(f"\nSaved metadata to: {metadata_path}")
        log(f"Wall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        log("\n" + "=" * 80)
        log("EXPERIMENT 8 -- ORIGIN C MARKET-STATE REPLICATION COMPLETE")
        log("=" * 80)
        flush_log()
    except Exception:
        flush_log()
        raise

    print(f"Saved run log to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
