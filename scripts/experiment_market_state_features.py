"""
Breakthrough Model Plan section 4.3 (Component C) / section 6.2 item 5:
market-implied game-state features on the rolling-origin harness.

Pre-registration alignment: this is Experiment 5 of
docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md (read sections 1 and 6 together
with this docstring). Pinned conventions honored here: ONLY the
is_latest_pregame_snapshot timing-safe view of market_game_features.parquet
is used (never a mean across all snapshot dates for an event); the
cross-book consensus aggregation built in STEP 1 below is new preprocessing
(the file ships none), chosen as an ordinary development decision and
applied identically to both origins; EV threshold fixed at 0.05;
goalie-night cluster key f"{game_id}_{goalie_id}", 10,000 resamples, seed
42, 95% CI; PMF cap ORIGIN_CAP=90. The pre-registered pass bar (section
6.4): a cluster-CI-excluding-zero improvement in shots MAE or paired Brier
on BOTH origins relative to the baseline; a marginal CI-spanning-zero
result is a valid negative and is reported as such, never rounded up.

De-vig conventions -- two distinct ones coexist here and are NOT conflated
(pre-registration section 1, "Selection at inference/pricing time"):
  (a) market_game_features.parquet's implied_prob_devig column carries the
      archive's own per-book MULTIPLICATIVE (proportional) two-way de-vig of
      h2h/totals pairs. It is used ONLY to build the market-state FEATURE
      columns (win probabilities, expected-goals approximation).
  (b) betting.tracking_db.devig_prob's additive-normalization convention on
      American-odds pairs is used ONLY inside the paired-Brier-vs-market
      METRIC (via the reused experiment_rolling_origin.paired_brier_delta),
      exactly as in every prior origin experiment. Bet SELECTION
      (decide_bet/calculate_ev) uses raw vig-inclusive single-side implied
      probabilities, unchanged.

Baseline note: pre-registration section 6.3 nominally baselines Experiment 5
against "the Experiment 2 Gate-A candidate." That candidate did not exist as
a locked artifact when this run was assigned and executed (Experiments 1-3
run concurrently; pre-registration section 0.2 acknowledges this round's
concurrency). Per this task's assignment, the baseline here is the no-pace
control (Experiment 1's variant), trained fresh inside this script on each
origin's own train/val split. If Experiments 2-3 later produce a different
Gate-A candidate, the marginal effect of market-state features on THAT
architecture is a separate follow-up, not covered by this run.

Question: does adding timing-safe market-implied game-state features
(data/processed/market_game_features.parquet -- consensus game total,
de-vigged home/away win probability, favorite/underdog strength, cross-book
dispersion, and an approximate opponent expected-goals figure derived from
total + moneyline) on top of the existing no-pace control feature set improve
the distributional saves model?

This reuses scripts/experiment_rolling_origin.py's rolling-origin machinery
UNCHANGED (folds, betting-frame construction, paired de-vigged-market Brier
delta, goalie-night cluster bootstrap, output conventions) so the numbers
here are directly comparable to that script's pace_shots result:

  Origin A: train <= 2022-23, test = 2023-24.
  Origin B: train <= 2023-24, test = 2024-25.

Two variants per origin, both trained fresh on that origin's own train/val
split (nothing is tuned against, or even touches, the worn Dec 2025-Apr 2026
fold):
  no_pace_control            -- exact "control" feature set from
                                 experiment_pace_distributional.py: base +
                                 engineered features only, no game-context,
                                 no pace.
  control_plus_market_state  -- no_pace_control's shots-against feature list
                                 plus 7 market-state columns (see
                                 MARKET_FEATURE_COLS below) plus one
                                 mkt_matched missingness indicator, added to
                                 the SHOTS-AGAINST model only (mirrors the
                                 pace_shots precedent of adding new families
                                 to the shots model and leaving the save-rate
                                 model unchanged, so the ablation measures one
                                 thing). The save-rate model is therefore
                                 identical -- and literally trained once, not
                                 twice -- across both variants of an origin.

CRITICAL FIRST STEP (required by the task, not optional context): join
coverage of market_game_features.parquet against each origin's TRAIN, VAL,
and TEST rows is computed and logged/saved BEFORE any model is trained. The
h2h/totals bulk archive this parquet is built from starts 2023-10-10 -- there
is zero market-state coverage for the 2022-23 season under any book. That
means Origin A's entire train+val pool (which lives inside 2022-23) has 0%
coverage: the control_plus_market_state shots model for Origin A literally
never sees a non-missing market-state value during training or validation,
so its Origin A test-fold result cannot be interpreted as evidence the
features help or hurt -- the tree has no non-missing training exposure to
split on. This is reported and handled honestly (NaN + explicit mkt_matched
indicator, never silently imputed), not swept under the rug. Origin B's
train pool spans 2022-23 and 2023-24, so it gets partial (not full) real
training exposure; see the join-coverage table for the exact number.

Methodology choices made in this script that the plan left open (repeated in
the run log / metadata and in the final chat report, not just here):

  1. Market features enter the SHOTS-AGAINST model only, not the save-rate
     model -- mirrors the pace_shots precedent (pace families 1-4,6 shots
     only; family 5 reserved for the secondary pace_both variant) so this
     ablation measures a single clean marginal effect.
  2. "No-pace control" is the literal experiment_pace_distributional.py
     "control" variant: base_feature_cols + engineered_cols from
     experiments.distributional_saves.load_modeling_frame. No
     game_context_features.parquet columns, no pace_features.parquet
     columns -- this script never even loads pace_features.parquet.
  3. Consensus game total = median of the totals `point` value (the line
     itself, not a probability) across books at the is_latest_pregame_snapshot
     view; cross-book total dispersion = std of that same point value across
     books. Averaging LINE values across books is not the odds-averaging bug
     documented in docs/HISTORICAL_DATA_ANALYSIS.md section 1 (which was
     arithmetic averaging of vig-inclusive American ODDS/PROBABILITIES).
     Every probability used here is first de-vigged per book (via the
     archive's own proportional per-book devig, already computed in
     implied_prob_devig) and only de-vigged probabilities are ever averaged
     across books -- the same devig-then-average pattern already used by
     scripts/clv_audit_pace_policy.py's build_closing_consensus().
  4. Approximate opponent expected goals = consensus_total *
     opponent_win_prob_devigged (each team's expected share of the game's
     total goals assumed proportional to its moneyline-implied win
     probability -- a standard "implied team total" heuristic, not a
     mechanistic goals model; this archive has no spread/puckline data to do
     better, matching the plan's own "approximate" framing).
  5. A market feature is left as NaN (routed by XGBoost's native
     missing-value handling) everywhere the event did not join or a
     sub-market (h2h or totals) was unquoted for a joined event, PLUS an
     explicit mkt_matched indicator column is added. Never silently imputed.
  6. Dispersion (negative-binomial alpha) is fit on VALIDATION residuals
     (val_idx passed to the reused experiments.distributional_saves.
     fit_dispersion), not training residuals -- a deliberate deviation from
     the pace_shots precedent (which fits on train_idx), because this task's
     design explicitly requires validation-or-rolling-out-of-sample
     dispersion fitting and Gate A explicitly asks for "validation-fitted
     dispersion without extreme edge inflation." fit_dispersion's internal
     log lines say "TRAIN residuals" verbatim regardless of which index
     array is passed (hardcoded string in the reused function) -- read those
     lines in run_log.txt as "the residuals of whatever index set was
     passed," which is val_idx here.
  7. EV threshold fixed at 0.05 for both origins (same as
     experiment_rolling_origin.py), not reselected via a validation sweep --
     for direct ROI comparability with the pace_shots rolling-origin result,
     and because Origin A's validation window has zero market-state (and
     zero saves-market odds) coverage, ruling out a validation sweep there.
  8. PMF cap = 90 (ORIGIN_CAP, reused unchanged from
     experiment_rolling_origin.py), not the production CAP=70, for the same
     documented reason: a smaller/less-regularized origin shots model can
     need wider PMF support.
  9. "Lower-tail calibration" uses goalie TOI < 50 minutes (the Breakthrough
     Model Plan section 3.2 definition of a short/early-replacement start).
     "OVER/UNDER calibration separately" is a 2-bucket calibration check
     (rows where the model's own p_over >= 0.5 vs < 0.5: mean predicted
     P(over) vs actual over-hit rate in each bucket) -- a calibration
     diagnostic, distinct from the betting-policy OVER/UNDER ROI split
     already reported inside policy_roi.side_breakdown. Both are computed on
     one row per goalie-night (first book kept) to avoid overweighting
     heavily-quoted goalie-nights.

Zero network calls. All inputs are already on disk.

Usage:
    python scripts/experiment_market_state_features.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Reuse the rolling-origin machinery unchanged -- folds, betting-frame
# construction, paired de-vigged-market Brier delta, row-prediction schema.
import experiment_rolling_origin as ero  # noqa: E402
import clv_audit_pace_policy as clv  # noqa: E402
from experiments.distributional_saves import (  # noqa: E402
    SavesDistribution,
    build_betting_frame,
    compute_distribution_predictions,
    fit_dispersion,
    join_and_price,
    load_modeling_frame,
    mae,
    train_save_rate_model,
    train_shots_model,
)
from experiments.harness import (  # noqa: E402
    betting_metrics_bundle,
    brier as harness_brier,
    fold_wide_auc_brier,
    grade_bets,
)

DATA_PATH_CLEAN = REPO_ROOT / "data" / "processed" / "clean_training_data.parquet"
DATA_PATH_CONTEXT = REPO_ROOT / "data" / "processed" / "game_context_features.parquet"
DATA_PATH_MULTIBOOK = REPO_ROOT / "data" / "processed" / "multibook_classification_training_data.parquet"
MARKET_PATH = REPO_ROOT / "data" / "processed" / "market_game_features.parquet"
OUTPUT_ROOT = REPO_ROOT / "models" / "trained"

MARKET_FEATURE_COLS = [
    "mkt_consensus_total",
    "mkt_team_win_prob_devigged",
    "mkt_opponent_win_prob_devigged",
    "mkt_favorite_underdog_strength",
    "mkt_total_dispersion",
    "mkt_h2h_dispersion",
    "mkt_opponent_expected_goals_approx",
]
MARKET_INDICATOR_COL = "mkt_matched"
ALL_MARKET_COLS = MARKET_FEATURE_COLS + [MARKET_INDICATOR_COL]

VARIANT_NAMES = ("no_pace_control", "control_plus_market_state")


# ---------------------------------------------------------------------------
# STEP 1: build the event-level market-state table from
# market_game_features.parquet, restricted to the timing-safe last-snapshot-
# before-commence view.
# ---------------------------------------------------------------------------


def build_market_state_events(market_path: Path, log) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(market_path)
    n_raw = len(raw)
    n_events_raw = int(raw["event_id"].nunique())
    latest = raw[raw["is_latest_pregame_snapshot"]].copy()
    log(
        f"market_game_features.parquet: {n_raw} rows / {n_events_raw} distinct events total; "
        f"{len(latest)} rows / {latest['event_id'].nunique()} distinct events at "
        "is_latest_pregame_snapshot==True (the timing-safe last-snapshot-before-commence view -- "
        "the only rows used anywhere below)."
    )

    h2h = latest[latest["market"] == "h2h"].copy()
    h2h["is_home_outcome"] = h2h["outcome_label"] == h2h["home_abbrev"]
    h2h_book = h2h.loc[h2h["is_home_outcome"], ["event_id", "book", "implied_prob_devig"]].rename(
        columns={"implied_prob_devig": "home_prob"}
    )
    h2h_summary = h2h_book.groupby("event_id")["home_prob"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "home_win_prob_devigged", "std": "h2h_dispersion", "count": "n_h2h_books"}
    )
    h2h_summary["h2h_dispersion"] = h2h_summary["h2h_dispersion"].fillna(0.0)

    totals = latest[latest["market"] == "totals"].copy()
    totals_book_point = totals[["event_id", "book", "point"]].drop_duplicates()
    dup_book_point = totals_book_point.duplicated(subset=["event_id", "book"]).sum()
    if dup_book_point:
        raise AssertionError(
            f"{dup_book_point} (event_id, book) pairs quote more than one totals point at the "
            "latest pregame snapshot; the median/std-of-points construction below assumes exactly "
            "one point per book (verified empirically before writing this script)."
        )
    totals_summary = totals_book_point.groupby("event_id")["point"].agg(["median", "std", "count"]).rename(
        columns={"median": "consensus_total", "std": "total_dispersion", "count": "n_totals_books"}
    )
    totals_summary["total_dispersion"] = totals_summary["total_dispersion"].fillna(0.0)

    event_meta = latest[["event_id", "game_date_eastern", "home_abbrev", "away_abbrev"]].drop_duplicates(
        subset=["event_id"]
    )
    dup_meta_keys = int(event_meta.duplicated(subset=["game_date_eastern", "home_abbrev", "away_abbrev"]).sum())
    if dup_meta_keys:
        raise AssertionError(
            f"{dup_meta_keys} duplicate (date, home, away) event keys; the goalie-frame join below "
            "would be ambiguous."
        )

    events = event_meta.merge(h2h_summary, on="event_id", how="left").merge(totals_summary, on="event_id", how="left")
    events["home_expected_goals_approx"] = events["consensus_total"] * events["home_win_prob_devigged"]
    events["away_expected_goals_approx"] = events["consensus_total"] * (1.0 - events["home_win_prob_devigged"])

    n_h2h = int(events["home_win_prob_devigged"].notna().sum())
    n_totals = int(events["consensus_total"].notna().sum())
    n_both = int((events["home_win_prob_devigged"].notna() & events["consensus_total"].notna()).sum())
    log(
        f"Event-level market-state table: {len(events)} distinct events; {n_h2h} with an h2h "
        f"summary, {n_totals} with a totals summary, {n_both} with both."
    )

    stats = {
        "n_raw_rows": n_raw,
        "n_raw_events": n_events_raw,
        "n_latest_pregame_rows": int(len(latest)),
        "n_latest_pregame_events": int(latest["event_id"].nunique()),
        "n_events_with_h2h": n_h2h,
        "n_events_with_totals": n_totals,
        "n_events_with_both": n_both,
    }
    return events, stats


# ---------------------------------------------------------------------------
# STEP 2: join the event-level table onto the goalie-game modeling frame,
# reoriented to the goalie's own team vs opponent perspective.
# ---------------------------------------------------------------------------


def attach_market_state_features(df: pd.DataFrame, events: pd.DataFrame, log) -> pd.DataFrame:
    df = df.copy()
    overlap = set(ALL_MARKET_COLS) & set(df.columns)
    if overlap:
        raise ValueError(f"Market feature names collide with modeling frame columns: {sorted(overlap)}")

    df["_date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
    df["_home_abbrev"] = np.where(df["is_home"] == 1, df["team_abbrev"], df["opponent_team"])
    df["_away_abbrev"] = np.where(df["is_home"] == 1, df["opponent_team"], df["team_abbrev"])

    ev = events.rename(columns={"game_date_eastern": "_date_str", "home_abbrev": "_home_abbrev", "away_abbrev": "_away_abbrev"})
    keep_cols = [
        "_date_str", "_home_abbrev", "_away_abbrev", "home_win_prob_devigged", "h2h_dispersion",
        "consensus_total", "total_dispersion", "home_expected_goals_approx", "away_expected_goals_approx",
    ]
    ev = ev[keep_cols]

    before = len(df)
    df = df.merge(ev, on=["_date_str", "_home_abbrev", "_away_abbrev"], how="left", indicator="_mkt_merge")
    if len(df) != before:
        raise AssertionError("Market-state merge changed modeling-frame row count.")

    matched = (df["_mkt_merge"] == "both").values
    df["mkt_matched"] = matched.astype(int)
    df["mkt_consensus_total"] = df["consensus_total"]
    df["mkt_total_dispersion"] = df["total_dispersion"]
    df["mkt_h2h_dispersion"] = df["h2h_dispersion"]
    df["mkt_team_win_prob_devigged"] = np.where(
        df["is_home"] == 1, df["home_win_prob_devigged"], 1.0 - df["home_win_prob_devigged"]
    )
    df["mkt_opponent_win_prob_devigged"] = 1.0 - df["mkt_team_win_prob_devigged"]
    df["mkt_favorite_underdog_strength"] = df["mkt_team_win_prob_devigged"] - 0.5
    df["mkt_opponent_expected_goals_approx"] = np.where(
        df["is_home"] == 1, df["away_expected_goals_approx"], df["home_expected_goals_approx"]
    )

    drop_cols = [
        "_date_str", "_home_abbrev", "_away_abbrev", "_mkt_merge", "home_win_prob_devigged", "h2h_dispersion",
        "consensus_total", "total_dispersion", "home_expected_goals_approx", "away_expected_goals_approx",
    ]
    df = df.drop(columns=drop_cols)

    coverage_pct = float(matched.mean() * 100) if len(df) else 0.0
    log(
        f"\nMarket-state join to modeling frame: {int(matched.sum())}/{len(df)} rows matched "
        f"({coverage_pct:.2f}% overall across ALL seasons combined -- see the per-origin/split "
        "breakdown below for the number that actually determines trainability)."
    )

    mat = df[MARKET_FEATURE_COLS].values.astype(np.float64)
    n_inf = int(np.isinf(mat).sum())
    if n_inf:
        raise AssertionError(f"Market feature matrix contains {n_inf} infinite values.")
    null_counts = {c: int(v) for c, v in df[MARKET_FEATURE_COLS].isna().sum().items() if int(v) > 0}
    log(
        "Market feature NaN counts (retained for XGBoost native missing-value handling; NaN exactly "
        f"where mkt_matched==0 or a sub-market was unquoted for a matched event): {null_counts}"
    )

    return df


def report_join_coverage(df_full: pd.DataFrame, origin_splits: dict, log) -> dict:
    log("\n" + "=" * 80)
    log("CRITICAL FIRST STEP: market_game_features.parquet join coverage")
    log("=" * 80)
    matched = df_full["mkt_matched"].values.astype(bool)

    log("\nPer-season coverage across the full clean_training_data.parquet universe "
        "(answers: which seasons have quotes at all?):")
    season_cov = df_full.groupby("season").apply(
        lambda g: pd.Series({"n_rows": len(g), "n_matched": int(g["mkt_matched"].sum())})
    )
    season_cov["pct_matched"] = season_cov["n_matched"] / season_cov["n_rows"] * 100
    coverage: dict = {"by_season": {}}
    for season, row in season_cov.iterrows():
        pct = float(row["pct_matched"])
        log(f"  season {int(season)}: {int(row['n_matched'])}/{int(row['n_rows'])} rows matched ({pct:.2f}%)")
        coverage["by_season"][int(season)] = {
            "n_rows": int(row["n_rows"]), "n_matched": int(row["n_matched"]), "pct_matched": pct,
        }

    log("\nPer-origin, per-split coverage (this determines whether the market-state variant is "
        "actually trainable for a given origin):")
    for origin_label, splits in origin_splits.items():
        coverage[origin_label] = {}
        log(f"\n{origin_label}:")
        for split_name, idx in splits.items():
            n = len(idx)
            m = int(matched[idx].sum())
            pct = m / n * 100 if n else 0.0
            coverage[origin_label][split_name] = {"n_rows": n, "n_matched": m, "pct_matched": pct}
            log(f"  {split_name:<6}: {m}/{n} rows matched to a market_game_features event ({pct:.2f}%)")

    return coverage


# ---------------------------------------------------------------------------
# Calibration diagnostics (lower-tail, OVER/UNDER), computed one row per
# goalie-night to avoid overweighting heavily-quoted goalie-nights.
# ---------------------------------------------------------------------------


def parse_toi_minutes(values) -> np.ndarray:
    """Parse goalie TOI stored as 'MM:SS' strings (the on-disk format in
    every frame here, e.g. '60:00') into float minutes. NaN for
    missing/unparseable values."""

    def _parse(v):
        if v is None:
            return np.nan
        if isinstance(v, float):
            return v if not np.isnan(v) else np.nan
        s = str(v)
        if ":" in s:
            mm, ss = s.split(":", 1)
            try:
                return float(mm) + float(ss) / 60.0
            except ValueError:
                return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    return np.array([_parse(v) for v in values], dtype=float)


def dedup_positions(df_bet: pd.DataFrame, matched: np.ndarray) -> np.ndarray:
    idx = np.where(matched)[0]
    d = pd.DataFrame(
        {"game_id": df_bet["game_id"].values[idx], "goalie_id": df_bet["goalie_id"].values[idx], "_pos": idx}
    )
    d = d.drop_duplicates(subset=["game_id", "goalie_id"], keep="first")
    return d["_pos"].values


def side_calibration(p_over: np.ndarray, y: np.ndarray, log, label: str) -> dict:
    """2-bucket calibration diagnostic: rows where the model's own p_over is
    on the OVER side (>=0.5) vs the UNDER side (<0.5). Mean predicted P(over)
    vs actual over-hit rate in each bucket. This is a calibration check, not
    the betting-policy OVER/UNDER ROI split (reported separately inside
    policy_roi.side_breakdown)."""
    over_favored = p_over >= 0.5
    result = {}
    for name, mask in (("model_favors_over", over_favored), ("model_favors_under", ~over_favored)):
        n = int(mask.sum())
        if n == 0:
            result[name] = {"n": 0}
            continue
        mean_pred = float(p_over[mask].mean())
        actual_rate = float(y[mask].mean())
        result[name] = {
            "n": n,
            "mean_predicted_p_over": mean_pred,
            "actual_over_rate": actual_rate,
            "calibration_gap": mean_pred - actual_rate,
        }
    log(f"[{label}] side calibration (one row per goalie-night): {result}")
    return result


def lower_tail_calibration(p_over: np.ndarray, y: np.ndarray, toi: np.ndarray, log, label: str) -> dict:
    """Calibration restricted to short/early-replacement starts (TOI < 50
    minutes, the Breakthrough Model Plan section 3.2 definition)."""
    mask = toi < 50
    n = int(np.nansum(mask))
    if n == 0:
        result = {"n": 0}
        log(f"[{label}] lower-tail (toi<50) calibration: no rows.")
        return result
    b = float(harness_brier(p_over[mask], y[mask]))
    mean_pred = float(p_over[mask].mean())
    actual_rate = float(y[mask].mean())
    result = {"n": n, "brier": b, "mean_predicted_p_over": mean_pred, "actual_over_rate": actual_rate}
    log(f"[{label}] lower-tail (toi<50) calibration (one row per goalie-night): {result}")
    return result


def paired_shots_mae_delta(
    df_full: pd.DataFrame,
    test_idx: np.ndarray,
    mu_base: np.ndarray,
    mu_new: np.ndarray,
    log,
    label: str,
) -> dict:
    """Paired per-goalie-night absolute-shots-error delta between the two
    variants on the SAME test rows: |mu_market - actual| - |mu_control -
    actual|. Cluster key f"{game_id}_{goalie_id}" (each clean-data test row
    IS one goalie-night, so clusters are singletons here; the cluster
    bootstrap is retained for convention consistency with every other CI in
    this program). Negative delta means the market-state variant's shots
    model was MORE accurate. This is the statistic the pre-registered pass
    bar (PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 6.4) tests for a
    CI-excluding-zero improvement."""
    actual = df_full["shots_against"].values[test_idx].astype(float)
    delta = np.abs(mu_new - actual) - np.abs(mu_base - actual)
    cluster_ids = np.array(
        [
            f"{int(g)}_{int(o)}"
            for g, o in zip(df_full["game_id"].values[test_idx], df_full["goalie_id"].values[test_idx])
        ],
        dtype=object,
    )
    stat = clv.cluster_bootstrap_mean_ci(
        delta, cluster_ids, n_resamples=ero.N_BOOTSTRAP_RESAMPLES, seed=ero.BOOTSTRAP_SEED, ci_pct=95.0
    )
    log(
        f"[{label}] paired shots |error| delta (control_plus_market_state - no_pace_control): "
        f"mean={stat['mean']:+.5f} 95% CI=[{stat['lower']:+.5f}, {stat['upper']:+.5f}] "
        f"n={stat['n_bets']} (negative means the market-state variant's shots model was BETTER)"
    )
    return stat


def paired_brier_delta_vs_variant(
    df_bet: pd.DataFrame,
    p_over_base: np.ndarray,
    matched_base: np.ndarray,
    p_over_new: np.ndarray,
    matched_new: np.ndarray,
    log,
    label: str,
) -> dict:
    """Paired per-row Brier delta between two variants' predictions on the
    SAME betting-frame rows (control_plus_market_state minus no_pace_control),
    goalie-night cluster bootstrap 95% CI. Positive delta means the
    market-state variant was WORSE."""
    both_matched = matched_base & matched_new
    y = (df_bet["saves"].values.astype(float) > df_bet["betting_line"].values.astype(float)).astype(float)
    sq_base = (p_over_base - y) ** 2
    sq_new = (p_over_new - y) ** 2
    delta = np.where(both_matched, sq_new - sq_base, np.nan)
    cluster_ids = np.array(
        [f"{int(g)}_{int(o)}" for g, o in zip(df_bet["game_id"].values, df_bet["goalie_id"].values)], dtype=object
    )
    stat = clv.cluster_bootstrap_mean_ci(
        delta, cluster_ids, n_resamples=ero.N_BOOTSTRAP_RESAMPLES, seed=ero.BOOTSTRAP_SEED, ci_pct=95.0
    )
    log(
        f"[{label}] paired Brier delta (control_plus_market_state - no_pace_control): "
        f"mean={stat['mean']} 95% CI=[{stat['lower']}, {stat['upper']}] "
        f"n_rows={stat['n_bets']} n_clusters={stat['n_clusters']} "
        "(negative means the market-state variant was BETTER)"
    )
    return stat


# ---------------------------------------------------------------------------
# STEP 3: per-origin, per-variant shots-model training and evaluation.
# ---------------------------------------------------------------------------


def run_shots_variant(
    origin_label: str,
    variant_name: str,
    df_full: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    shots_cols: list[str],
    rate_cols: list[str],
    rate_model,
    price_frames: dict,
    output_dir: Path,
    log,
) -> tuple[dict, dict, pd.DataFrame, np.ndarray]:
    label_base = f"origin_{origin_label.lower()}_{variant_name}"
    log("\n" + "=" * 80)
    log(f"ORIGIN {origin_label} / VARIANT {variant_name}")
    log("=" * 80)
    log(f"Shots feature count: {len(shots_cols)}  (save-rate model is shared across variants, {len(rate_cols)} features)")

    shots_model, shots_winner, shots_evals = train_shots_model(df_full, train_idx, val_idx, shots_cols, log, label_base)

    log(
        f"[{label_base}] NOTE: fit_dispersion is called with val_idx (validation residuals), not "
        "train_idx, per this experiment's explicit dispersion-fitting requirement -- its own log "
        "lines below say 'TRAIN residuals' verbatim (hardcoded string in the reused function) but "
        "the residuals are val_idx's."
    )
    alpha, dispersion_method, dispersion_diag = fit_dispersion(shots_model, df_full, val_idx, shots_cols, log, label_base)

    shots_path = output_dir / f"{label_base}_shots_model.json"
    shots_model.get_booster().save_model(str(shots_path))
    log(f"Saved {label_base} shots model to: {shots_path}")

    dist = SavesDistribution(ero.ORIGIN_CAP)
    dist_preds_test = compute_distribution_predictions(
        df_full, test_idx, shots_model, rate_model, alpha, shots_cols, rate_cols, dist, log, f"{label_base} TEST"
    )

    mu_test = dist_preds_test["mu"]
    actual_shots_test = df_full["shots_against"].values[test_idx].astype(float)
    bias = float(np.mean(mu_test - actual_shots_test))
    workload_mae = mae(actual_shots_test, mu_test)
    log(
        f"[{label_base}] shots-against workload on TEST: mean_bias={bias:+.4f} "
        f"MAE={workload_mae:.4f} (n={len(test_idx)})"
    )

    pass_results = {}
    probs_by_pass = {}
    row_frames = []
    for pass_name, df_bet in price_frames.items():
        label = f"{label_base} TEST {pass_name}"
        p_over, p_under, p_push, matched, cov = join_and_price(df_bet, dist_preds_test, dist, log, label)
        probs_by_pass[pass_name] = (p_over, matched)

        auc, brier_val = fold_wide_auc_brier(
            p_over, matched, df_bet["saves"].values, df_bet["betting_line"].values,
            df_bet["game_id"].values, df_bet["goalie_id"].values, log, label,
        )
        brier_delta_stat, market_p_over_arr, market_p_under_arr = ero.paired_brier_delta(df_bet, p_over, matched, log, label)

        bet_results = grade_bets(
            p_over, p_under, df_bet["saves"].values.astype(float), df_bet["betting_line"].values.astype(float),
            df_bet["odds_over_american"].astype(float).values, df_bet["odds_under_american"].astype(float).values,
            df_bet["game_id"].values, df_bet["goalie_id"].values, ero.FIXED_EV_THRESHOLD, matched, log, label,
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

        keep_pos = dedup_positions(df_bet, matched)
        y_full = (df_bet["saves"].values.astype(float) > df_bet["betting_line"].values.astype(float)).astype(float)
        toi_full = (
            parse_toi_minutes(df_bet["toi"].values) if "toi" in df_bet.columns else np.full(len(df_bet), np.nan)
        )
        p_over_dd = p_over[keep_pos]
        y_dd = y_full[keep_pos]
        toi_dd = toi_full[keep_pos]
        side_cal = side_calibration(p_over_dd, y_dd, log, label)
        tail_cal = lower_tail_calibration(p_over_dd, y_dd, toi_dd, log, label)

        row_df = ero.build_row_predictions(
            df_bet, p_over, p_under, matched, market_p_over_arr, market_p_under_arr,
            ero.FIXED_EV_THRESHOLD, origin_label, pass_name,
        )
        row_df["variant"] = variant_name
        row_frames.append(row_df)

        pass_results[pass_name] = {
            "join_coverage_pct": cov,
            "fold_wide_auc": auc,
            "fold_wide_brier": brier_val,
            "paired_brier_delta_vs_market": brier_delta_stat,
            "policy_roi": bundle,
            "side_calibration": side_cal,
            "lower_tail_calibration_toi_lt_50": tail_cal,
            "ev_threshold": ero.FIXED_EV_THRESHOLD,
            "n_goalie_nights_deduped": int(len(keep_pos)),
        }

    result_json = {
        "origin": origin_label,
        "variant": variant_name,
        "shots_feature_count": len(shots_cols),
        "shots_feature_cols": shots_cols,
        "shots_model": {"winner": shots_winner, "val_evaluations": shots_evals, "model_path": str(shots_path)},
        "dispersion": {
            "alpha": alpha, "method": dispersion_method, "diagnostics": dispersion_diag,
            "fit_on": "val_idx (validation residuals, not train_idx -- see module docstring point 6)",
        },
        "workload_shots_against_test": {"mean_bias": bias, "mae": workload_mae, "n_test_rows": int(len(test_idx))},
        "price_passes": pass_results,
    }
    row_predictions = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    return result_json, probs_by_pass, row_predictions, mu_test


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_market_state_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log, flush_log = ero.make_logger(log_path)

    metadata: dict = {"timestamp": datetime.now().isoformat()}
    try:
        log("=" * 80)
        log("MARKET-STATE FEATURES EXPERIMENT (Breakthrough Model Plan section 4.3 / Component C)")
        log("Rolling-origin harness reused unchanged from scripts/experiment_rolling_origin.py")
        log("=" * 80)
        log(f"Output directory: {output_dir}")

        for path in (DATA_PATH_CLEAN, DATA_PATH_CONTEXT, DATA_PATH_MULTIBOOK, MARKET_PATH, ero.CLOSING_FRAME_PATH, ero.BETTIME_FRAME_PATH):
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required input: {path}")

        # ---- STEP 1: market-state event table ----
        log("\n" + "=" * 80)
        log("STEP 1: build event-level market-state table from market_game_features.parquet")
        log("=" * 80)
        events, market_stats = build_market_state_events(MARKET_PATH, log)
        metadata["market_source_stats"] = market_stats

        # ---- STEP 2: no-pace modeling frame + market-state join ----
        log("\n" + "=" * 80)
        log("STEP 2: load no-pace control modeling frame and join market-state features")
        log("=" * 80)
        frame = load_modeling_frame(DATA_PATH_CLEAN, DATA_PATH_CONTEXT, log)
        df_full = attach_market_state_features(frame.df, events, log)

        no_pace_cols = frame.base_feature_cols + frame.engineered_cols
        market_shots_cols = no_pace_cols + ALL_MARKET_COLS
        log(f"\nno_pace_control shots/rate feature count: {len(no_pace_cols)}")
        log(f"control_plus_market_state shots feature count: {len(market_shots_cols)} "
            f"({len(no_pace_cols)} no-pace-control + {len(ALL_MARKET_COLS)} market-state)")
        metadata["feature_sets"] = {
            "no_pace_control": no_pace_cols,
            "market_feature_cols": MARKET_FEATURE_COLS,
            "market_indicator_col": MARKET_INDICATOR_COL,
        }

        # ---- origin splits ----
        pool_min_a, pool_max_a = ero.season_date_range(df_full, [ero.SEASON_2022_23])
        train_idx_a, val_idx_a, boundaries_a = ero.carve_origin_split(
            df_full, pool_min_a, pool_max_a, ero.VAL_WINDOW_DAYS, log, "Origin A"
        )
        test_min_a, test_max_a = ero.season_date_range(df_full, [ero.SEASON_2023_24])
        test_idx_a = ero.date_range_test_idx(df_full, test_min_a, test_max_a, log, "Origin A")

        pool_min_b, pool_max_b = ero.season_date_range(df_full, [ero.SEASON_2022_23, ero.SEASON_2023_24])
        train_idx_b, val_idx_b, boundaries_b = ero.carve_origin_split(
            df_full, pool_min_b, pool_max_b, ero.VAL_WINDOW_DAYS, log, "Origin B"
        )
        test_min_b, test_max_b = ero.season_date_range(df_full, [ero.SEASON_2024_25])
        test_idx_b = ero.date_range_test_idx(df_full, test_min_b, test_max_b, log, "Origin B")

        # ---- STEP 3: CRITICAL FIRST STEP -- join coverage, before any modeling ----
        join_coverage = report_join_coverage(
            df_full,
            {
                "Origin A": {"train": train_idx_a, "val": val_idx_a, "test": test_idx_a},
                "Origin B": {"train": train_idx_b, "val": val_idx_b, "test": test_idx_b},
            },
            log,
        )
        metadata["join_coverage"] = join_coverage
        flush_log()

        # ---- betting (saves-market) frames ----
        df_bet_test_a_closing = build_betting_frame(ero.CLOSING_FRAME_PATH, log)
        df_bet_test_a_bettime = build_betting_frame(ero.BETTIME_FRAME_PATH, log)
        df_bet_multibook_full = build_betting_frame(DATA_PATH_MULTIBOOK, log)
        df_bet_test_b_closing = df_bet_multibook_full[
            df_bet_multibook_full["season"] == ero.SEASON_2024_25
        ].reset_index(drop=True)
        log(
            "\nOrigin B test betting frame (multibook_classification_training_data.parquet, season "
            f"2024-25, closing lines): {len(df_bet_test_b_closing)} rows."
        )

        origin_configs = [
            ("A", train_idx_a, val_idx_a, test_idx_a, boundaries_a, ero.SEASON_2023_24,
             {"closing": df_bet_test_a_closing, "bettime": df_bet_test_a_bettime}),
            ("B", train_idx_b, val_idx_b, test_idx_b, boundaries_b, ero.SEASON_2024_25,
             {"closing": df_bet_test_b_closing}),
        ]

        results = {}
        for origin_label, train_idx, val_idx, test_idx, boundaries, test_season, price_frames in origin_configs:
            log("\n" + "=" * 80)
            log(f"ORIGIN {origin_label}: shared save-rate model, then both shots-model variants")
            log("=" * 80)

            rate_label = f"origin_{origin_label.lower()}_shared_save_rate"
            rate_model, rate_winner, rate_evals = train_save_rate_model(
                df_full, train_idx, val_idx, no_pace_cols, log, rate_label
            )
            rate_path = output_dir / f"{rate_label}_model.json"
            rate_model.get_booster().save_model(str(rate_path))
            log(f"Saved {rate_label} model to: {rate_path} (shared by both variants of Origin {origin_label})")

            variant_json = {}
            variant_probs = {}
            variant_mu = {}
            all_row_frames = []
            for variant_name in VARIANT_NAMES:
                shots_cols = no_pace_cols if variant_name == "no_pace_control" else market_shots_cols
                result_json, probs_by_pass, row_predictions, mu_test = run_shots_variant(
                    origin_label, variant_name, df_full, train_idx, val_idx, test_idx,
                    shots_cols, no_pace_cols, rate_model, price_frames, output_dir, log,
                )
                variant_json[variant_name] = result_json
                variant_probs[variant_name] = probs_by_pass
                variant_mu[variant_name] = mu_test
                all_row_frames.append(row_predictions)
                flush_log()

            predictions_df = pd.concat(all_row_frames, ignore_index=True)
            predictions_path = output_dir / f"origin_{origin_label.lower()}_test_predictions.parquet"
            predictions_df.to_parquet(predictions_path, index=False)
            log(f"\nSaved {len(predictions_df)} per-row test predictions (both variants) to: {predictions_path}")

            base_p_over, base_matched = variant_probs["no_pace_control"]["closing"]
            new_p_over, new_matched = variant_probs["control_plus_market_state"]["closing"]
            brier_vs_control = paired_brier_delta_vs_variant(
                price_frames["closing"], base_p_over, base_matched, new_p_over, new_matched, log,
                f"origin_{origin_label} closing",
            )
            shots_mae_delta = paired_shots_mae_delta(
                df_full, test_idx, variant_mu["no_pace_control"], variant_mu["control_plus_market_state"],
                log, f"origin_{origin_label} TEST",
            )

            results[origin_label] = {
                "variants": variant_json,
                "rate_model": {
                    "winner": rate_winner, "val_evaluations": rate_evals, "model_path": str(rate_path),
                    "feature_cols": no_pace_cols,
                },
                "brier_vs_control_closing": brier_vs_control,
                "shots_mae_delta_vs_control": shots_mae_delta,
                "fold_boundaries": {**boundaries, "test_season": test_season, "test_rows": int(len(test_idx))},
                "predictions_path": str(predictions_path),
            }
            flush_log()

        metadata["origin_a"] = results["A"]
        metadata["origin_b"] = results["B"]
        metadata["fixed_ev_threshold"] = ero.FIXED_EV_THRESHOLD
        metadata["origin_cap"] = ero.ORIGIN_CAP
        metadata["design_notes"] = [
            "Market features enter the SHOTS-AGAINST model only; the save-rate model is unchanged "
            "and trained once per origin, shared by both variants.",
            "no_pace_control is the literal experiment_pace_distributional.py 'control' variant: "
            "base_feature_cols + engineered_cols, no game-context columns, no pace columns.",
            "Consensus game total = median of the totals point value (a line, not a probability) "
            "across books at the latest pregame snapshot; this is a legitimate line average, "
            "distinct from the odds-averaging bug (docs/HISTORICAL_DATA_ANALYSIS.md section 1), "
            "which was arithmetic averaging of vig-inclusive American odds. Every probability used "
            "here is de-vigged per book before any cross-book averaging.",
            "Approximate opponent expected goals = consensus_total * opponent_win_prob_devigged "
            "(a standard 'implied team total' heuristic; no spread/puckline data exists in this "
            "archive to do better).",
            "A market feature is NaN (native XGBoost missing-value routing) wherever the event did "
            "not join or a sub-market was unquoted; mkt_matched is an explicit 0/1 indicator. "
            "Never silently imputed.",
            "Dispersion is fit on VALIDATION residuals (val_idx), not training residuals -- a "
            "deliberate deviation from the pace_shots precedent (train_idx), because this "
            "experiment's design explicitly requires validation-or-rolling-out-of-sample dispersion "
            "fitting. fit_dispersion's own log lines say 'TRAIN residuals' regardless of which "
            "index array was passed (hardcoded string in the reused function).",
            "EV threshold fixed at 0.05 for both origins (matches experiment_rolling_origin.py's "
            "pace_shots recipe), not reselected via a validation sweep, for direct ROI comparability "
            "and because Origin A's validation window has zero market-state coverage.",
            f"PMF cap = {ero.ORIGIN_CAP} (ORIGIN_CAP, reused unchanged from "
            "experiment_rolling_origin.py), not the production CAP=70.",
            "Lower-tail calibration uses goalie TOI < 50 minutes (Breakthrough Model Plan section "
            "3.2). OVER/UNDER calibration is a 2-bucket diagnostic on the model's own p_over>=0.5 "
            "split, distinct from the betting-policy side_breakdown ROI split. Both are computed one "
            "row per goalie-night.",
            "market_game_features.parquet has ZERO coverage for the 2022-23 season under any book "
            "(the h2h/totals bulk archive starts 2023-10-10). Origin A's entire train+val pool lives "
            "inside 2022-23, so the control_plus_market_state variant for Origin A has 0% real "
            "training exposure to these features -- its Origin A test-fold result is a mechanical "
            "check, not evidence the features help or hurt. See join_coverage for exact numbers.",
            "Two de-vig conventions coexist and are not conflated (pre-registration section 1): the "
            "archive's per-book MULTIPLICATIVE de-vig (implied_prob_devig) is used only to build the "
            "market-state feature columns; betting.tracking_db.devig_prob's additive convention is "
            "used only inside the paired-Brier-vs-market metric, via the reused "
            "experiment_rolling_origin.paired_brier_delta. Bet selection uses raw vig-inclusive "
            "single-side implied probabilities (decide_bet/calculate_ev), unchanged.",
            "Baseline is the no-pace control, not the pre-registration section 6.3 'Experiment 2 "
            "Gate-A candidate' -- that candidate did not exist as a locked artifact when this run "
            "was assigned/executed (concurrent implementation round, pre-registration section 0.2). "
            "Re-testing the marginal effect on the eventual Gate-A candidate is a separate follow-up.",
            "Pre-registered pass bar (section 6.4): cluster-CI-excluding-zero improvement in shots "
            "MAE or paired Brier on BOTH origins; marginal CI-spanning-zero results are valid "
            "negatives. Evaluated and recorded in metadata['pre_registered_pass_bar'].",
        ]

        # ---- head-to-head summary ----
        log("\n" + "=" * 80)
        log("HEAD-TO-HEAD SUMMARY (closing pass, TEST fold)")
        log("=" * 80)
        log(
            f"{'origin':<8} {'variant':<26} {'shots_mae':>10} {'shots_bias':>11} "
            f"{'brier':>9} {'vs_mkt':>9} {'roi':>9} {'bets':>6}"
        )
        for origin_label in ("A", "B"):
            res = results[origin_label]
            for variant_name in VARIANT_NAMES:
                v = res["variants"][variant_name]
                wl = v["workload_shots_against_test"]
                cp = v["price_passes"]["closing"]
                log(
                    f"{origin_label:<8} {variant_name:<26} {wl['mae']:>10.4f} {wl['mean_bias']:>+11.4f} "
                    f"{cp['fold_wide_brier']:>9.5f} {cp['paired_brier_delta_vs_market']['delta_mean']:>+9.5f} "
                    f"{cp['policy_roi']['summary']['roi']:>+8.2f}% {cp['policy_roi']['summary']['bets']:>6}"
                )
            bvc = res["brier_vs_control_closing"]
            if bvc["n_bets"]:
                log(
                    f"  Origin {origin_label}: Brier(control_plus_market_state) - Brier(no_pace_control) "
                    f"on closing pass: mean={bvc['mean']:+.5f}  95% CI=[{bvc['lower']:+.5f}, "
                    f"{bvc['upper']:+.5f}]  n_rows={bvc['n_bets']}  n_clusters={bvc['n_clusters']}"
                )
            else:
                log(f"  Origin {origin_label}: no shared matched rows to compute a Brier-vs-control delta.")
            smd = res["shots_mae_delta_vs_control"]
            log(
                f"  Origin {origin_label}: shots |error| delta (market - control): mean={smd['mean']:+.5f}  "
                f"95% CI=[{smd['lower']:+.5f}, {smd['upper']:+.5f}]  n={smd['n_bets']}"
            )
        log("vs_mkt is model Brier minus the de-vigged market Brier; lower is better.")
        log("Brier(control_plus_market_state) - Brier(no_pace_control) negative means market-state features HELPED.")
        log("Do not interpret Origin A's control_plus_market_state result without reading the join-coverage table above.")

        # ---- pre-registered pass bar (PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 6.4) ----
        log("\n" + "=" * 80)
        log("PRE-REGISTERED PASS BAR (PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 6.4)")
        log("=" * 80)
        log("Pass requires a cluster-CI-excluding-zero IMPROVEMENT (CI upper bound < 0 on the "
            "market-minus-control delta) in shots MAE or paired Brier, on BOTH origins. A marginal "
            "CI-spanning-zero result is a valid negative.")
        pass_bar: dict = {}
        for metric_key, metric_label in (
            ("shots_mae_delta_vs_control", "shots |error| delta"),
            ("brier_vs_control_closing", "paired Brier delta (closing)"),
        ):
            per_origin = {}
            for origin_label in ("A", "B"):
                stat = results[origin_label][metric_key]
                improved = stat["upper"] is not None and stat["upper"] < 0
                per_origin[origin_label] = {
                    "mean": stat["mean"], "ci95": [stat["lower"], stat["upper"]],
                    "ci_excludes_zero_improvement": bool(improved),
                }
                log(
                    f"  {metric_label} / Origin {origin_label}: mean={stat['mean']:+.5f} "
                    f"CI=[{stat['lower']:+.5f}, {stat['upper']:+.5f}] -> "
                    f"{'CI-excluding-zero improvement' if improved else 'NOT a CI-excluding-zero improvement'}"
                )
            both = all(per_origin[o]["ci_excludes_zero_improvement"] for o in ("A", "B"))
            per_origin["passes_on_both_origins"] = both
            pass_bar[metric_key] = per_origin
        overall_pass = any(pass_bar[m]["passes_on_both_origins"] for m in pass_bar)
        pass_bar["overall_pass"] = overall_pass
        log(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'} against the pre-registered section 6.4 bar.")
        if not overall_pass:
            log("Per section 6.4 this is a valid negative: the features are not carried into the "
                "combined model (or are retained only as an input to Experiment 6's exposure "
                "component). It does not block Gate A.")
        log("Caveat that travels with any Origin A number: Origin A's train+val pool has 0% "
            "market-state coverage, so its market-variant result is a mechanical check, not "
            "evidence about the features (see join-coverage table).")
        metadata["pre_registered_pass_bar"] = pass_bar

        elapsed = time.time() - start_time
        metadata["wall_clock_seconds"] = elapsed
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        log(f"\nSaved metadata to: {metadata_path}")
        log(f"Wall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        log("\n" + "=" * 80)
        log("MARKET-STATE FEATURES EXPERIMENT COMPLETE")
        log("=" * 80)
        flush_log()
    except Exception:
        flush_log()
        raise

    print(f"Saved run log to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
