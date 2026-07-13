"""
Season-normalized pace / attempt-to-SOG funnel experiment on fresh rolling
origins (BREAKTHROUGH_MODEL_PLAN.md sections 4.2 + 6.2 items 1-3, evaluated
against Gate A in section 7; pre-registered evaluation standard in
docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md sections 1-4).

Variants (plan section 4.2 required-ablations table), all trained and
evaluated with the identical recipe (same SHOTS_CONFIGS/SAVE_RATE_CONFIGS
grids, validation-only submodel selection, fixed EV threshold 0.05):

  (a) no_pace_control     -- exact no-pace/no-context recipe (epd VARIANTS[0])
  (b) pace_shots_raw      -- raw 41-column pace recipe (reproduces the known
                             failure; also serves as this script's wiring
                             gate against experiment_rolling_origin's
                             recorded numbers)
  (c) season_normalized   -- the 37 raw-level pace columns replaced by
                             prior-only current-season league z-scores, plus
                             the 4 pre-built league_relative_zscores columns
                             unchanged
  (d) attempt_sog_funnel  -- explicit attempt -> unblocked attempt -> SOG ->
                             SOG-during-projected-exposure funnel stages,
                             each expressed as a rate or level relative to
                             the prior-only current-season league
                             environment, fed to the shots model as features
                             (plus the deterministic funnel projection
                             reported as a mechanism diagnostic)

Dispersion (plan section 6.2 item 3): for ALL result variants the
negative-binomial dispersion alpha is fitted on VALIDATION-fold residuals
(same closed-form NB2 moment matching as
src/experiments/distributional_saves.py::fit_dispersion, but computed from
the winner shots model's held-out validation residuals, never training
residuals). The train-residual alpha is computed as a DIAGNOSTIC ONLY for
every variant; the only predictions ever produced from a train-fitted alpha
are the variant (b) wiring-gate reproduction of
experiment_rolling_origin_20260709_222639's recorded numbers.

Origins (confirmed from scripts/experiment_rolling_origin.py and its
metadata, not assumed from the plan):

  Origin A: train/val inside season 2022-23 (val = last 49 days),
            test = season 2023-24. Betting frames:
            data/processed/multibook_frame_2023_24.parquet (closing,
            PRIMARY) + multibook_frame_2023_24_bettime.parquet (SECONDARY).
  Origin B: train/val inside seasons 2022-23 + 2023-24 (val = last 49
            days), test = season 2024-25. Betting frame:
            multibook_classification_training_data.parquet filtered to
            season == 20242025 (closing only).

All shared machinery is imported, never reimplemented: fold carving and the
paired market-Brier statistic from scripts/experiment_rolling_origin.py,
model training / distribution math from
src/experiments/distributional_saves.py, bet grading and cluster bootstrap
from src/experiments/harness.py, the generic cluster-bootstrap-mean CI from
scripts/clv_audit_pace_policy.py, and the prior-only league-environment
statistics helper from src/features/pace_features.py.

Usage:
    python scripts/experiment_season_funnel.py
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

# Proven sys.path import pattern (same as scripts/clv_audit_pace_policy.py
# and scripts/experiment_rolling_origin.py).
import experiment_pace_distributional as epd  # noqa: E402
import experiment_rolling_origin as ero  # noqa: E402
import clv_audit_pace_policy as clv  # noqa: E402
from experiments.distributional_saves import (  # noqa: E402
    SavesDistribution,
    compute_distribution_predictions,
    fit_dispersion,
    intrinsic_quality_metrics,
    join_and_price,
    train_save_rate_model,
    train_shots_model,
)
from experiments.harness import (  # noqa: E402
    betting_metrics_bundle,
    fold_wide_auc_brier,
    grade_bets,
)
from features.pace_features import FAMILY_COLUMNS, _prior_league_stats_by_date  # noqa: E402

make_logger = epd.make_logger

OUTPUT_ROOT = REPO_ROOT / "models" / "trained"
CLOSING_FRAME_PATH = REPO_ROOT / "data" / "processed" / "multibook_frame_2023_24.parquet"
BETTIME_FRAME_PATH = REPO_ROOT / "data" / "processed" / "multibook_frame_2023_24_bettime.parquet"

SEASON_2022_23 = 20222023
SEASON_2023_24 = 20232024
SEASON_2024_25 = 20242025

VAL_WINDOW_DAYS = 49          # matches experiment_rolling_origin exactly
FIXED_EV_THRESHOLD = 0.05     # pre-registered, never reselected (prereg section 1)
ORIGIN_CAP = 90               # pre-registered PMF cap for origin-carved runs
N_BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42

LOW_SAVES_THRESHOLDS = [10, 12, 15, 18, 20]

# The 37 raw-level pace columns = families 1-4 (the 41 pace_shots columns
# minus the 4 pre-built league_relative_zscores).
RAW_PACE_FAMILIES = [
    "opponent_offense_pace",
    "team_shot_suppression",
    "combined_pace",
    "special_teams_volume",
]
PREBUILT_Z_FAMILY = "league_relative_zscores"

# ---------------------------------------------------------------------------
# Wiring-gate constants: variant (b) trained here with TRAIN-fitted
# dispersion must reproduce experiment_rolling_origin_20260709_222639's
# recorded closing-pass numbers exactly (same data, same deterministic
# training, same ORIGIN_CAP=90). Values transcribed from that run's
# metadata.json.
# ---------------------------------------------------------------------------
ERO_GATE = {
    "A": {
        "alpha": 0.012962625150880053,
        "dispersion_method": "negative_binomial",
        "closing": {"brier_r5": 0.26338, "delta_r5": 0.01338, "roi_r2": -8.30, "bets": 3895},
    },
    "B": {
        "alpha": 0.0,
        "dispersion_method": "poisson_fallback",
        "closing": {"brier_r5": 0.26439, "delta_r5": 0.01559, "roi_r2": -3.00, "bets": 4376},
    },
}

# Step-0 / coordinator cross-checks (soft: logged, flagged, never used to
# tune anything).
STEP0_CONTROL_BIAS = {"A": 0.4420, "B": 0.0308}
STEP0_PACE_VAL_ALPHA = {"A": 0.0303, "B": 0.0270}


VARIANT_SPECS = [
    {
        "name": "no_pace_control",
        "description": "No pace, no context: base + engineered features only in both submodels (epd control variant).",
    },
    {
        "name": "pace_shots_raw",
        "description": "Raw pace families 1-4 and 6 (41 columns) added to the shots model (epd pace_shots variant; the known failure).",
    },
    {
        "name": "season_normalized",
        "description": (
            "Season-normalized pace: the 37 raw-level pace columns replaced by prior-only current-season "
            "league z-scores (sz_*), plus the 4 pre-built league_relative_zscores columns unchanged. "
            "The shots model can no longer read absolute pooled attempt levels, only position relative "
            "to the current league environment."
        ),
    },
    {
        "name": "attempt_sog_funnel",
        "description": (
            "Explicit attempt-to-SOG funnel: expected attempts (opponent offense x team defense vs prior-only "
            "current-season league attempt level), unblocked fraction (Fenwick/Corsi vs league fraction), "
            "Fenwick-to-SOG conversion anchored to the prior-only current-season league SOG level (the stage "
            "where the documented cross-season drift lives), and starter-exposure share; stage features plus "
            "the deterministic funnel projection fed to the shots model."
        ),
    },
]


# ---------------------------------------------------------------------------
# Feature construction for variants (c) and (d). All league-environment
# statistics are prior-only within season by game date (reusing
# src/features/pace_features.py::_prior_league_stats_by_date), so nothing
# from the current date or later ever enters a feature.
# ---------------------------------------------------------------------------


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def attach_prior_league_stats(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """Return df with prior_mean/prior_std columns for source_col, prior-only
    within (season, game_date). Caller must drop the helper columns."""
    stats = _prior_league_stats_by_date(df[["season", "game_date", source_col]].copy(), source_col)
    merged = df.merge(stats, on=["season", "game_date"], how="left")
    if len(merged) != len(df):
        raise AssertionError(f"prior-league-stats merge changed row count for {source_col}.")
    return merged


def add_season_column(frame_df: pd.DataFrame, clean_full: pd.DataFrame, log) -> pd.DataFrame:
    """frame.df carries season_x (from clean_training_data) / season_y (pace
    verification column) after the pace merge. Recover a single unambiguous
    `season` column and verify it against clean_training_data."""
    df = frame_df.copy()
    if "season" in df.columns:
        return df
    if "season_x" not in df.columns:
        raise AssertionError("frame.df has neither season nor season_x; cannot recover season column.")
    df["season"] = df["season_x"].astype(int)
    check = clean_full[["game_id", "goalie_id", "season"]].rename(columns={"season": "season_check"})
    merged = df[["game_id", "goalie_id", "season"]].merge(check, on=["game_id", "goalie_id"], how="left")
    if len(merged) != len(df):
        raise AssertionError("Season verification merge changed row count.")
    n_bad = int((merged["season"] != merged["season_check"]).sum())
    if n_bad:
        raise AssertionError(f"{n_bad} rows have season_x != clean_training_data season.")
    log(f"Season column recovered from season_x and verified against clean_training_data ({len(df)} rows).")
    return df


def add_season_normalized_features(df: pd.DataFrame, log) -> tuple[pd.DataFrame, list[str]]:
    """Variant (c): prior-only current-season league z-scores of the 37
    raw-level pace columns."""
    raw_cols: list[str] = []
    for fam in RAW_PACE_FAMILIES:
        raw_cols.extend(FAMILY_COLUMNS[fam])
    if len(raw_cols) != 37:
        raise AssertionError(f"Expected 37 raw-level pace columns, found {len(raw_cols)}.")

    sz_cols = []
    for col in raw_cols:
        merged = attach_prior_league_stats(df, col)
        z_col = f"sz_{col}"
        df[z_col] = safe_div(merged[col] - merged["prior_mean"], merged["prior_std"]).values
        sz_cols.append(z_col)

    null_counts = {c: int(df[c].isna().sum()) for c in sz_cols}
    worst = sorted(null_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    log(
        f"Season-normalized features: {len(sz_cols)} sz_* columns added (prior-only within-season league "
        f"z-scores). Top null counts (early-season rows, retained for XGBoost): {worst}"
    )
    return df, sz_cols


def add_funnel_features(df: pd.DataFrame, log) -> tuple[pd.DataFrame, list[str]]:
    """Variant (d): explicit attempt -> unblocked -> SOG -> exposure funnel.

    Stages (every league statistic is a prior-only expanding mean over
    strictly earlier game dates in the CURRENT season -- computed on the
    goalie-game frame, so it is start-weighted, documented as a methodology
    choice):

      L_att   = league prior mean of opp_off_all_corsi_ema5 (per-game
                all-situation attempt volume).
      E[att]  = opp_off_all_corsi_ema5 * team_def_all_corsi_against_ema5
                / L_att  (standard multiplicative matchup adjustment).
      frac    = Fenwick/Corsi unblocked fraction, opponent-offense and
                team-defense sides, each used relative to the league prior
                fraction L_frac.
      E[fen]  = E[att] * opp_frac * team_frac / L_frac.
      L_conv  = (league prior mean actual team SOG per game, from
                clean_training_data's opp_shots outcome column on earlier
                dates only) / (league prior mean opp_off_all_fenwick_ema5).
                This is the attempt-to-SOG conversion stage where the
                documented cross-season drift lives; anchoring it to the
                current season's own prior games is the entire point.
      conv rel = team/opponent-specific SOG-per-Fenwick (rolling SOG from
                clean data over Fenwick EMA), each relative to L_conv, fed
                as features rather than hard-multiplied.
      L_share = league prior mean of shots_against / opp_shots (starter's
                share of the team's faced SOG; outcome columns, earlier
                dates only).
      mu_funnel = E[fen] * L_conv * L_share  (deterministic league-conversion
                projection, also fed as a feature and reported as a
                mechanism diagnostic).
    """
    needed = [
        "opp_off_all_corsi_ema5", "team_def_all_corsi_against_ema5",
        "opp_off_all_fenwick_ema5", "team_def_all_fenwick_against_ema5",
        "opp_shots_rolling_10", "team_shots_against_rolling_10",
        "opp_shots", "shots_against",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise AssertionError(f"Funnel feature construction missing inputs: {missing}")

    # League prior attempt level.
    m = attach_prior_league_stats(df, "opp_off_all_corsi_ema5")
    df["fnl_league_att_pg"] = m["prior_mean"].values

    df["fnl_expected_attempts"] = safe_div(
        df["opp_off_all_corsi_ema5"] * df["team_def_all_corsi_against_ema5"], df["fnl_league_att_pg"]
    )
    df["fnl_expected_attempts_rel"] = safe_div(df["fnl_expected_attempts"], df["fnl_league_att_pg"])

    # Unblocked fraction stage.
    df["fnl_opp_unblocked_frac"] = safe_div(df["opp_off_all_fenwick_ema5"], df["opp_off_all_corsi_ema5"])
    df["fnl_team_unblocked_frac"] = safe_div(
        df["team_def_all_fenwick_against_ema5"], df["team_def_all_corsi_against_ema5"]
    )
    m = attach_prior_league_stats(df, "fnl_opp_unblocked_frac")
    df["fnl_league_unblocked_frac"] = m["prior_mean"].values
    df["fnl_expected_fenwick"] = safe_div(
        df["fnl_expected_attempts"] * df["fnl_opp_unblocked_frac"] * df["fnl_team_unblocked_frac"],
        df["fnl_league_unblocked_frac"],
    )

    # Fenwick-to-SOG conversion stage, anchored to the current season's own
    # prior games (outcome columns of strictly earlier dates -- prior-only).
    m = attach_prior_league_stats(df, "opp_shots")
    df["fnl_league_sog_pg"] = m["prior_mean"].values
    m = attach_prior_league_stats(df, "opp_off_all_fenwick_ema5")
    df["fnl_league_fen_pg"] = m["prior_mean"].values
    df["fnl_league_fen_to_sog"] = safe_div(df["fnl_league_sog_pg"], df["fnl_league_fen_pg"])

    df["fnl_opp_sog_conv_rel"] = safe_div(
        safe_div(df["opp_shots_rolling_10"], df["opp_off_all_fenwick_ema5"]), df["fnl_league_fen_to_sog"]
    )
    df["fnl_team_sog_conv_rel"] = safe_div(
        safe_div(df["team_shots_against_rolling_10"], df["team_def_all_fenwick_against_ema5"]),
        df["fnl_league_fen_to_sog"],
    )

    # Exposure stage: league prior starter share of team faced SOG.
    share = safe_div(df["shots_against"].astype(float), df["opp_shots"].astype(float))
    df["_starter_share_outcome"] = share
    m = attach_prior_league_stats(df, "_starter_share_outcome")
    df["fnl_starter_share"] = m["prior_mean"].values
    df = df.drop(columns=["_starter_share_outcome"])

    # Deterministic funnel projection.
    df["fnl_mu_league_conv"] = df["fnl_expected_fenwick"] * df["fnl_league_fen_to_sog"] * df["fnl_starter_share"]
    df["fnl_mu_team_adj"] = df["fnl_mu_league_conv"] * np.sqrt(
        df["fnl_opp_sog_conv_rel"].clip(0.5, 2.0) * df["fnl_team_sog_conv_rel"].clip(0.5, 2.0)
    )

    funnel_cols = [
        "fnl_league_att_pg", "fnl_expected_attempts", "fnl_expected_attempts_rel",
        "fnl_opp_unblocked_frac", "fnl_team_unblocked_frac", "fnl_league_unblocked_frac",
        "fnl_expected_fenwick", "fnl_league_sog_pg", "fnl_league_fen_to_sog",
        "fnl_opp_sog_conv_rel", "fnl_team_sog_conv_rel", "fnl_starter_share",
        "fnl_mu_league_conv", "fnl_mu_team_adj",
    ]
    null_counts = {c: int(df[c].isna().sum()) for c in funnel_cols}
    worst = sorted(null_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    log(
        f"Funnel features: {len(funnel_cols)} fnl_* columns added (all stages prior-only). "
        f"Top null counts (early-season rows, retained for XGBoost): {worst}"
    )
    return df, funnel_cols


# ---------------------------------------------------------------------------
# Validation-fitted dispersion (plan section 6.2 item 3). Identical NB2
# moment-matching math to distributional_saves.fit_dispersion, but on the
# winner shots model's held-out VALIDATION residuals.
# ---------------------------------------------------------------------------


def fit_dispersion_val(shots_model, df, val_idx, feature_cols, log, label):
    X_val = df[feature_cols].iloc[val_idx].astype(np.float32)
    y_val = df["shots_against"].values[val_idx].astype(float)
    mu_val = np.clip(shots_model.predict(X_val), 1e-3, None)
    resid = y_val - mu_val

    val_mean = float(np.mean(mu_val))
    val_var = float(np.mean(resid**2))
    log(f"\n--- {label}: dispersion fit on VALIDATION residuals ---")
    log(f"  mean(predicted mu on val) = {val_mean:.4f}")
    log(f"  mean(val residual^2)      = {val_var:.4f}")

    if val_var <= val_mean:
        log("  VAL variance <= VAL mean: falling back to Poisson (alpha=0).")
        return 0.0, "poisson_fallback_val", {"val_mean": val_mean, "val_var": val_var}

    mean_mu2 = float(np.mean(mu_val**2))
    alpha = max((val_var - val_mean) / mean_mu2, 1e-6)
    log(f"  NB2 dispersion alpha (val-fitted) = {alpha:.6f}  (Var = mean + alpha*mean^2)")
    return alpha, "negative_binomial_val", {"val_mean": val_mean, "val_var": val_var, "mean_mu2": mean_mu2}


# ---------------------------------------------------------------------------
# Test-fold intrinsic metrics beyond intrinsic_quality_metrics: signed bias
# with bootstrap CI, low-saves tail calibration, TOI<50 subset diagnostics.
# ---------------------------------------------------------------------------


def parse_toi_minutes(toi_series: pd.Series) -> np.ndarray:
    def _parse(v):
        try:
            s = str(v)
            if ":" in s:
                mm, ss = s.split(":")
                return float(mm) + float(ss) / 60.0
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    return np.array([_parse(v) for v in toi_series], dtype=float)


def shots_bias_stats(df, test_idx, dist_preds, log, label):
    y = df["shots_against"].values[test_idx].astype(float)
    mu = dist_preds["mu"]
    diff = mu - y
    game_ids = df["game_id"].values[test_idx]
    goalie_ids = df["goalie_id"].values[test_idx]
    clusters = np.array([f"{int(g)}_{int(q)}" for g, q in zip(game_ids, goalie_ids)], dtype=object)
    stat = clv.cluster_bootstrap_mean_ci(diff, clusters, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    log(
        f"[{label}] shots signed bias (mu - actual): {stat['mean']:+.4f} "
        f"95% CI [{stat['lower']:+.4f}, {stat['upper']:+.4f}] (n={stat['n_bets']})"
    )
    return {
        "bias_mean": stat["mean"],
        "bias_ci95_lower": stat["lower"],
        "bias_ci95_upper": stat["upper"],
        "mae": float(np.mean(np.abs(diff))),
        "n": int(len(diff)),
    }


def lower_tail_metrics(df, test_idx, dist_preds, dist, log, label):
    saves_actual = df["saves"].values[test_idx].astype(int)
    saves_c = np.clip(saves_actual, 0, dist.cap)
    pmf = dist_preds["pmf"]
    cdf = np.cumsum(pmf, axis=1)
    n = len(test_idx)

    table = {}
    abs_dev_sum = 0.0
    for s in LOW_SAVES_THRESHOLDS:
        pred = float(np.mean(cdf[:, s]))
        actual = float(np.mean(saves_actual <= s))
        table[str(s)] = {"predicted_p_le": pred, "actual_freq": actual, "gap": pred - actual}
        abs_dev_sum += abs(pred - actual)

    toi_min = parse_toi_minutes(df["toi"].values[test_idx])
    lt50 = toi_min < 50
    n_lt50 = int(np.nansum(lt50))
    lt50_diag = {"n": n_lt50}
    if n_lt50 > 0:
        idx_lt = np.where(lt50)[0]
        rng = np.random.RandomState(123)
        pit = np.zeros(len(idx_lt))
        for j, i in enumerate(idx_lt):
            y = saves_c[i]
            hi = cdf[i, y]
            lo = cdf[i, y - 1] if y > 0 else 0.0
            pit[j] = lo + rng.uniform() * (hi - lo)
        lt50_diag["mean_pit"] = float(np.mean(pit))
        lt50_diag["pit_below_0p2_freq"] = float(np.mean(pit < 0.2))
        p_at = np.clip(pmf[idx_lt, saves_c[idx_lt]], 1e-12, None)
        lt50_diag["mean_logscore"] = float(np.mean(-np.log(p_at)))

    log(f"[{label}] lower-tail P(saves<=s) predicted vs actual:")
    for s in LOW_SAVES_THRESHOLDS:
        t = table[str(s)]
        log(f"    s={s:>3}: predicted {t['predicted_p_le']:.4f} vs actual {t['actual_freq']:.4f} (gap {t['gap']:+.4f})")
    log(f"[{label}] lower-tail summed |gap| = {abs_dev_sum:.4f}; TOI<50 subset diagnostics: {lt50_diag}")

    return {
        "p_le_table": table,
        "summed_abs_gap": abs_dev_sum,
        "toi_lt50": lt50_diag,
    }


def side_calibration(df_bet, p_over, p_under, matched, log, label):
    saves_arr = df_bet["saves"].values.astype(float)[matched]
    lines_arr = df_bet["betting_line"].values.astype(float)[matched]
    po = np.asarray(p_over)[matched]
    pu = np.asarray(p_under)[matched]
    y_over = (saves_arr > lines_arr).astype(float)
    y_under = (saves_arr < lines_arr).astype(float)

    def _binned(prob, y):
        bins = np.clip((prob * 10).astype(int), 0, 9)
        out = []
        for b in range(10):
            m = bins == b
            if m.sum() == 0:
                out.append({"bin": f"{b/10:.1f}-{(b+1)/10:.1f}", "n": 0, "mean_pred": None, "empirical": None})
            else:
                out.append({
                    "bin": f"{b/10:.1f}-{(b+1)/10:.1f}",
                    "n": int(m.sum()),
                    "mean_pred": float(prob[m].mean()),
                    "empirical": float(y[m].mean()),
                })
        return out

    over_lean = po >= 0.5
    summary = {
        "over_reliability_10bin": _binned(po, y_over),
        "under_reliability_10bin": _binned(pu, y_under),
        "over_leaning": {
            "n": int(over_lean.sum()),
            "mean_p_over": float(po[over_lean].mean()) if over_lean.any() else None,
            "empirical_over_rate": float(y_over[over_lean].mean()) if over_lean.any() else None,
        },
        "under_leaning": {
            "n": int((~over_lean).sum()),
            "mean_p_under": float(pu[~over_lean].mean()) if (~over_lean).any() else None,
            "empirical_under_rate": float(y_under[~over_lean].mean()) if (~over_lean).any() else None,
        },
    }
    ol, ul = summary["over_leaning"], summary["under_leaning"]
    log(
        f"[{label}] side calibration: OVER-leaning n={ol['n']} mean_p={ol['mean_p_over']} "
        f"empirical={ol['empirical_over_rate']}; UNDER-leaning n={ul['n']} mean_p={ul['mean_p_under']} "
        f"empirical={ul['empirical_under_rate']}"
    )
    return summary


def brier_delta_vs_control(df_bet, p_over_variant, p_over_control, matched, log, label):
    saves_arr = df_bet["saves"].values.astype(float)
    lines_arr = df_bet["betting_line"].values.astype(float)
    game_ids = df_bet["game_id"].values
    goalie_ids = df_bet["goalie_id"].values

    idx = np.where(matched)[0]
    y = (saves_arr[idx] > lines_arr[idx]).astype(float)
    sq_v = (np.asarray(p_over_variant)[idx] - y) ** 2
    sq_c = (np.asarray(p_over_control)[idx] - y) ** 2
    delta = sq_v - sq_c
    clusters = np.array([f"{int(game_ids[i])}_{int(goalie_ids[i])}" for i in idx], dtype=object)
    stat = clv.cluster_bootstrap_mean_ci(delta, clusters, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    log(
        f"[{label}] paired Brier delta vs no_pace_control: {stat['mean']:+.5f} "
        f"95% CI [{stat['lower']:+.5f}, {stat['upper']:+.5f}] (negative = variant better)"
    )
    return {
        "delta_mean": stat["mean"],
        "delta_ci95_lower": stat["lower"],
        "delta_ci95_upper": stat["upper"],
        "n_rows": stat["n_bets"],
        "n_clusters": stat["n_clusters"],
    }


def logscore_delta_vs_control(df, test_idx, dist_preds_v, dist_preds_c, dist, log, label):
    saves_actual = np.clip(df["saves"].values[test_idx].astype(int), 0, dist.cap)
    rows = np.arange(len(test_idx))
    ls_v = -np.log(np.clip(dist_preds_v["pmf"][rows, saves_actual], 1e-12, None))
    ls_c = -np.log(np.clip(dist_preds_c["pmf"][rows, saves_actual], 1e-12, None))
    delta = ls_v - ls_c
    game_ids = df["game_id"].values[test_idx]
    goalie_ids = df["goalie_id"].values[test_idx]
    clusters = np.array([f"{int(g)}_{int(q)}" for g, q in zip(game_ids, goalie_ids)], dtype=object)
    stat = clv.cluster_bootstrap_mean_ci(delta, clusters, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    log(
        f"[{label}] full-distribution log-score delta vs control: {stat['mean']:+.5f} "
        f"95% CI [{stat['lower']:+.5f}, {stat['upper']:+.5f}] (negative = variant better)"
    )
    return {
        "mean_logscore_variant": float(np.mean(ls_v)),
        "mean_logscore_control": float(np.mean(ls_c)),
        "delta_mean": stat["mean"],
        "delta_ci95_lower": stat["lower"],
        "delta_ci95_upper": stat["upper"],
    }


def funnel_projection_diagnostic(df, test_idx, log, label):
    """Mechanism diagnostic: bias/MAE of the DETERMINISTIC funnel projections
    (no XGBoost) on the test fold."""
    y = df["shots_against"].values[test_idx].astype(float)
    out = {}
    for col in ("fnl_mu_league_conv", "fnl_mu_team_adj"):
        mu = df[col].values[test_idx].astype(float)
        ok = ~np.isnan(mu)
        out[col] = {
            "coverage_pct": float(ok.mean() * 100),
            "bias_mean": float(np.mean(mu[ok] - y[ok])),
            "mae": float(np.mean(np.abs(mu[ok] - y[ok]))),
        }
        log(
            f"[{label}] deterministic {col}: bias {out[col]['bias_mean']:+.4f}, MAE {out[col]['mae']:.4f} "
            f"({out[col]['coverage_pct']:.1f}% rows non-null)"
        )
    return out


# ---------------------------------------------------------------------------
# Per-variant, per-origin runner.
# ---------------------------------------------------------------------------


def price_pass(
    df_bet, dist_preds_test, dist, threshold, control_p_over, log, label,
    origin_label, pass_name, variant_name,
):
    p_over, p_under, _p_push, matched, cov = join_and_price(df_bet, dist_preds_test, dist, log, label)
    auc, brier_val = fold_wide_auc_brier(
        p_over, matched, df_bet["saves"].values, df_bet["betting_line"].values,
        df_bet["game_id"].values, df_bet["goalie_id"].values, log, label,
    )
    brier_delta_stat, market_p_over_arr, market_p_under_arr = ero.paired_brier_delta(
        df_bet, p_over, matched, log, label
    )

    vs_control = None
    if control_p_over is not None:
        vs_control = brier_delta_vs_control(df_bet, p_over, control_p_over, matched, log, label)

    bet_results = grade_bets(
        p_over, p_under, df_bet["saves"].values.astype(float), df_bet["betting_line"].values.astype(float),
        df_bet["odds_over_american"].astype(float).values, df_bet["odds_under_american"].astype(float).values,
        df_bet["game_id"].values, df_bet["goalie_id"].values, threshold, matched, log, label,
    )
    bundle = betting_metrics_bundle(bet_results, df_bet["game_id"].values, df_bet["goalie_id"].values, len(df_bet))
    log(
        f"[{label}] {bundle['summary']['bets']} bets, {bundle['summary']['bet_rate']:.1f}% bet rate, "
        f"{bundle['summary']['roi']:+.2f}% ROI, cluster CI [{bundle['roi_ci_cluster']['lower']:+.2f}%, "
        f"{bundle['roi_ci_cluster']['upper']:+.2f}%]"
    )
    log(
        f"[{label}] side breakdown: OVER {bundle['side_breakdown']['OVER']['bets']} "
        f"({bundle['side_breakdown']['OVER']['roi']:+.2f}%), UNDER {bundle['side_breakdown']['UNDER']['bets']} "
        f"({bundle['side_breakdown']['UNDER']['roi']:+.2f}%)"
    )

    calib = side_calibration(df_bet, p_over, p_under, matched, log, label)

    row_df = ero.build_row_predictions(
        df_bet, p_over, p_under, matched, market_p_over_arr, market_p_under_arr,
        threshold, origin_label, pass_name,
    )
    row_df["variant"] = variant_name

    result = {
        "join_coverage_pct": cov,
        "fold_wide_auc": auc,
        "fold_wide_brier": brier_val,
        "paired_brier_delta_vs_market": brier_delta_stat,
        "brier_delta_vs_no_pace_control": vs_control,
        "policy_roi": bundle,
        "side_calibration": calib,
        "ev_threshold": threshold,
    }
    return result, p_over, matched, row_df


def run_variant_origin(
    variant_name, shots_cols, rate_cols, frame_df, train_idx, val_idx, test_idx,
    price_frames, output_dir, origin_label, control_cache, log,
):
    log("\n" + "=" * 80)
    log(f"ORIGIN {origin_label} / VARIANT {variant_name}")
    log("=" * 80)
    log(f"Shots feature count: {len(shots_cols)}  Save-rate feature count: {len(rate_cols)}")

    shots_model, shots_winner, shots_evals = train_shots_model(
        frame_df, train_idx, val_idx, shots_cols, log, f"origin_{origin_label} {variant_name}"
    )
    rate_model, rate_winner, rate_evals = train_save_rate_model(
        frame_df, train_idx, val_idx, rate_cols, log, f"origin_{origin_label} {variant_name}"
    )

    # Train-residual alpha: DIAGNOSTIC ONLY for result variants.
    alpha_train, method_train, diag_train = fit_dispersion(
        shots_model, frame_df, train_idx, shots_cols, log, f"origin_{origin_label} {variant_name} (train, diagnostic)"
    )
    # Validation-residual alpha: used for ALL result predictions.
    alpha_val, method_val, diag_val = fit_dispersion_val(
        shots_model, frame_df, val_idx, shots_cols, log, f"origin_{origin_label} {variant_name}"
    )
    log(
        f"[origin_{origin_label} {variant_name}] dispersion: train-fitted alpha={alpha_train:.6f} "
        f"({method_train}, diagnostic only) vs val-fitted alpha={alpha_val:.6f} ({method_val}, USED)"
    )

    shots_path = output_dir / f"origin_{origin_label.lower()}_{variant_name}_shots_model.json"
    rate_path = output_dir / f"origin_{origin_label.lower()}_{variant_name}_save_rate_model.json"
    shots_model.get_booster().save_model(str(shots_path))
    rate_model.get_booster().save_model(str(rate_path))
    log(f"Saved shots model to {shots_path}")
    log(f"Saved save-rate model to {rate_path}")

    dist = SavesDistribution(ORIGIN_CAP)
    dist_preds_test = compute_distribution_predictions(
        frame_df, test_idx, shots_model, rate_model, alpha_val, shots_cols, rate_cols, dist, log,
        f"origin_{origin_label} {variant_name} TEST (val-fitted alpha)",
    )

    label_base = f"origin_{origin_label} {variant_name}"
    bias = shots_bias_stats(frame_df, test_idx, dist_preds_test, log, label_base)
    intrinsics = intrinsic_quality_metrics(frame_df, test_idx, dist_preds_test, dist, log, f"{label_base} TEST")
    tail = lower_tail_metrics(frame_df, test_idx, dist_preds_test, dist, log, label_base)

    logscore_vs_control = None
    if control_cache is not None:
        logscore_vs_control = logscore_delta_vs_control(
            frame_df, test_idx, dist_preds_test, control_cache["dist_preds_test"], dist, log, label_base
        )

    funnel_diag = None
    if variant_name == "attempt_sog_funnel":
        funnel_diag = funnel_projection_diagnostic(frame_df, test_idx, log, label_base)

    pass_results = {}
    p_over_by_pass = {}
    row_frames = []
    for pass_name, df_bet in price_frames.items():
        label = f"{label_base} TEST {pass_name}"
        control_p_over = control_cache["p_over_by_pass"][pass_name] if control_cache is not None else None
        result, p_over, matched, row_df = price_pass(
            df_bet, dist_preds_test, dist, FIXED_EV_THRESHOLD, control_p_over, log, label,
            origin_label, pass_name, variant_name,
        )
        if control_cache is not None:
            if not np.array_equal(matched, control_cache["matched_by_pass"][pass_name]):
                raise AssertionError(f"[{label}] matched mask differs from control's; variants must share join universe.")
        pass_results[pass_name] = result
        p_over_by_pass[pass_name] = p_over
        row_frames.append(row_df)

    result = {
        "variant": variant_name,
        "shots_feature_count": len(shots_cols),
        "rate_feature_count": len(rate_cols),
        "shots_feature_cols": shots_cols,
        "rate_feature_cols": rate_cols,
        "shots_model": {"winner": shots_winner, "val_evaluations": shots_evals, "model_path": str(shots_path)},
        "save_rate_model": {"winner": rate_winner, "val_evaluations": rate_evals, "model_path": str(rate_path)},
        "dispersion": {
            "used": {"alpha": alpha_val, "method": method_val, "diagnostics": diag_val, "fitted_on": "validation residuals"},
            "train_fitted_diagnostic_only": {"alpha": alpha_train, "method": method_train, "diagnostics": diag_train},
        },
        "shots_bias_test": bias,
        "intrinsic_test": intrinsics,
        "lower_tail_test": tail,
        "logscore_delta_vs_control": logscore_vs_control,
        "funnel_projection_diagnostic": funnel_diag,
        "price_passes": pass_results,
    }
    cache = {
        "dist_preds_test": dist_preds_test,
        "p_over_by_pass": p_over_by_pass,
        "matched_by_pass": {
            k: (np.asarray(~np.isnan(p_over_by_pass[k]))) for k in p_over_by_pass
        },
        "shots_model": shots_model,
        "rate_model": rate_model,
        "alpha_train": alpha_train,
        "shots_cols": shots_cols,
        "rate_cols": rate_cols,
    }
    return result, cache, row_frames


# ---------------------------------------------------------------------------
# Wiring gate: variant (b) with TRAIN-fitted dispersion must reproduce
# experiment_rolling_origin's recorded closing-pass numbers.
# ---------------------------------------------------------------------------


def run_wiring_gate(origin_label, cache_b, frame_df, test_idx, df_bet_closing, log):
    log("\n" + "-" * 80)
    log(f"WIRING GATE origin {origin_label}: pace_shots_raw + TRAIN-fitted alpha must reproduce "
        "experiment_rolling_origin_20260709_222639's recorded closing-pass numbers")
    log("-" * 80)
    expected = ERO_GATE[origin_label]

    alpha_train = cache_b["alpha_train"]
    if abs(alpha_train - expected["alpha"]) > 1e-6:
        raise AssertionError(
            f"WIRING GATE FAILED (origin {origin_label}): train-fitted alpha {alpha_train} != "
            f"recorded {expected['alpha']}."
        )
    log(f"Train-fitted alpha matches recorded value: {alpha_train:.10f}")

    dist = SavesDistribution(ORIGIN_CAP)
    dist_preds = compute_distribution_predictions(
        frame_df, test_idx, cache_b["shots_model"], cache_b["rate_model"], alpha_train,
        cache_b["shots_cols"], cache_b["rate_cols"], dist, log,
        f"wiring_gate origin_{origin_label} (train-fitted alpha)",
    )
    p_over, p_under, _pp, matched, _cov = join_and_price(
        df_bet_closing, dist_preds, dist, log, f"wiring_gate origin_{origin_label} closing"
    )
    _auc, brier_val = fold_wide_auc_brier(
        p_over, matched, df_bet_closing["saves"].values, df_bet_closing["betting_line"].values,
        df_bet_closing["game_id"].values, df_bet_closing["goalie_id"].values, log,
        f"wiring_gate origin_{origin_label}",
    )
    delta_stat, _mo, _mu = ero.paired_brier_delta(df_bet_closing, p_over, matched, log, f"wiring_gate origin_{origin_label}")
    bet_results = grade_bets(
        p_over, p_under, df_bet_closing["saves"].values.astype(float),
        df_bet_closing["betting_line"].values.astype(float),
        df_bet_closing["odds_over_american"].astype(float).values,
        df_bet_closing["odds_under_american"].astype(float).values,
        df_bet_closing["game_id"].values, df_bet_closing["goalie_id"].values,
        FIXED_EV_THRESHOLD, matched, log, f"wiring_gate origin_{origin_label}",
    )
    bundle = betting_metrics_bundle(
        bet_results, df_bet_closing["game_id"].values, df_bet_closing["goalie_id"].values, len(df_bet_closing)
    )

    observed = {
        "brier_r5": round(float(brier_val), 5),
        "delta_r5": round(float(delta_stat["delta_mean"]), 5),
        "roi_r2": round(float(bundle["summary"]["roi"]), 2),
        "bets": int(bundle["summary"]["bets"]),
    }
    log(f"Observed: {observed}")
    log(f"Expected: {expected['closing']}")
    if observed != expected["closing"]:
        raise AssertionError(
            f"WIRING GATE FAILED (origin {origin_label}): expected {expected['closing']}, observed {observed}. "
            "Stopping rather than improvising."
        )
    log(f"WIRING GATE PASSED (origin {origin_label}).")
    return observed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_season_funnel_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log, flush_log = make_logger(log_path)

    metadata: dict = {"timestamp": datetime.now().isoformat()}
    try:
        log("=" * 80)
        log("SEASON-NORMALIZED PACE / ATTEMPT-TO-SOG FUNNEL EXPERIMENT")
        log("Plan: docs/BREAKTHROUGH_MODEL_PLAN.md sections 4.2, 6.2 items 1-3, Gate A (section 7)")
        log("Pre-registration: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md sections 1-4")
        log("=" * 80)
        log(f"Output directory: {output_dir}")

        for path in (
            epd.DATA_PATH_CLEAN, epd.DATA_PATH_CONTEXT, epd.DATA_PATH_PACE, epd.DATA_PATH_PACE_METADATA,
            epd.DATA_PATH_MULTIBOOK, CLOSING_FRAME_PATH, BETTIME_FRAME_PATH,
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required input: {path}")

        # ---- load shared modeling frame and add new feature families ----
        frame = epd.load_pace_modeling_frame(
            epd.DATA_PATH_CLEAN, epd.DATA_PATH_CONTEXT, epd.DATA_PATH_PACE, epd.DATA_PATH_PACE_METADATA, log,
        )
        clean_full = pd.read_parquet(epd.DATA_PATH_CLEAN)
        clean_full["game_date"] = pd.to_datetime(clean_full["game_date"])

        frame_df = add_season_column(frame.df, clean_full, log)
        order_key_before = list(zip(frame_df["game_id"].tolist(), frame_df["goalie_id"].tolist()))

        frame_df, sz_cols = add_season_normalized_features(frame_df, log)
        frame_df, funnel_cols = add_funnel_features(frame_df, log)

        order_key_after = list(zip(frame_df["game_id"].tolist(), frame_df["goalie_id"].tolist()))
        if order_key_before != order_key_after:
            raise AssertionError("Feature construction changed frame row order; fold indices would be invalid.")
        log(f"Row order and count verified unchanged after feature construction ({len(frame_df)} rows).")

        # ---- variant feature lists ----
        control_spec = next(v for v in epd.VARIANTS if v.name == "control")
        pace_spec = next(v for v in epd.VARIANTS if v.name == "pace_shots")
        shots_cols_a, rate_cols_a = epd.feature_cols_for_variant(frame, control_spec)
        shots_cols_b, rate_cols_b = epd.feature_cols_for_variant(frame, pace_spec)
        prebuilt_z = FAMILY_COLUMNS[PREBUILT_Z_FAMILY]
        shots_cols_c = shots_cols_a + sz_cols + prebuilt_z
        shots_cols_d = shots_cols_a + funnel_cols
        variant_features = {
            "no_pace_control": (shots_cols_a, rate_cols_a),
            "pace_shots_raw": (shots_cols_b, rate_cols_b),
            "season_normalized": (shots_cols_c, rate_cols_a),
            "attempt_sog_funnel": (shots_cols_d, rate_cols_a),
        }
        for name, (sc, rc) in variant_features.items():
            if len(sc) != len(set(sc)) or len(rc) != len(set(rc)):
                raise AssertionError(f"{name} feature list contains duplicates.")
            log(f"Variant {name}: {len(sc)} shots features, {len(rc)} rate features.")

        # ---- betting frames (reuse existing artifacts; never rebuilt) ----
        df_bet_a_closing = epd.build_betting_frame(CLOSING_FRAME_PATH, log)
        df_bet_a_bettime = epd.build_betting_frame(BETTIME_FRAME_PATH, log)
        df_bet_multibook_full = epd.build_betting_frame(epd.DATA_PATH_MULTIBOOK, log)
        df_bet_b_closing = df_bet_multibook_full[
            df_bet_multibook_full["season"] == SEASON_2024_25
        ].reset_index(drop=True)
        log(f"Origin B closing betting frame (multibook, season 2024-25): {len(df_bet_b_closing)} rows.")

        # ---- origin fold carving (reusing experiment_rolling_origin's code) ----
        origins = {}
        pool_min_a, pool_max_a = ero.season_date_range(clean_full, [SEASON_2022_23])
        train_idx_a, val_idx_a, bounds_a = ero.carve_origin_split(
            frame_df, pool_min_a, pool_max_a, VAL_WINDOW_DAYS, log, "Origin A"
        )
        test_min_a, test_max_a = ero.season_date_range(clean_full, [SEASON_2023_24])
        test_idx_a = ero.date_range_test_idx(frame_df, test_min_a, test_max_a, log, "Origin A")
        origins["A"] = {
            "train_idx": train_idx_a, "val_idx": val_idx_a, "test_idx": test_idx_a,
            "bounds": {**bounds_a, "test_season": SEASON_2023_24, "test_rows": int(len(test_idx_a))},
            "price_frames": {"closing": df_bet_a_closing, "bettime": df_bet_a_bettime},
        }

        pool_min_b, pool_max_b = ero.season_date_range(clean_full, [SEASON_2022_23, SEASON_2023_24])
        train_idx_b, val_idx_b, bounds_b = ero.carve_origin_split(
            frame_df, pool_min_b, pool_max_b, VAL_WINDOW_DAYS, log, "Origin B"
        )
        test_min_b, test_max_b = ero.season_date_range(clean_full, [SEASON_2024_25])
        test_idx_b = ero.date_range_test_idx(frame_df, test_min_b, test_max_b, log, "Origin B")
        origins["B"] = {
            "train_idx": train_idx_b, "val_idx": val_idx_b, "test_idx": test_idx_b,
            "bounds": {**bounds_b, "test_season": SEASON_2024_25, "test_rows": int(len(test_idx_b))},
            "price_frames": {"closing": df_bet_b_closing},
        }

        # ---- run all variants per origin ----
        results = {"A": {}, "B": {}}
        cross_checks = {}
        for origin_label in ("A", "B"):
            o = origins[origin_label]
            control_cache = None
            row_frames_all = []
            for spec in VARIANT_SPECS:
                name = spec["name"]
                sc, rc = variant_features[name]
                result, cache, row_frames = run_variant_origin(
                    name, sc, rc, frame_df, o["train_idx"], o["val_idx"], o["test_idx"],
                    o["price_frames"], output_dir, origin_label, control_cache, log,
                )
                result["description"] = spec["description"]
                results[origin_label][name] = result
                row_frames_all.extend(row_frames)

                if name == "no_pace_control":
                    control_cache = cache
                    # Soft cross-check vs step-0's independently reproduced control bias.
                    obs_bias = result["shots_bias_test"]["bias_mean"]
                    exp_bias = STEP0_CONTROL_BIAS[origin_label]
                    ok = abs(obs_bias - exp_bias) <= 0.05
                    cross_checks[f"control_bias_origin_{origin_label}"] = {
                        "observed": obs_bias, "step0_expected": exp_bias, "within_0.05": bool(ok),
                    }
                    log(
                        f"CROSS-CHECK control bias origin {origin_label}: observed {obs_bias:+.4f} vs step-0 "
                        f"{exp_bias:+.4f} -> {'OK' if ok else 'MISMATCH (investigate)'}"
                    )
                    if not ok:
                        log("WARNING: control bias does not match step-0 reproduction within 0.05; results reported anyway, flagged.")

                if name == "pace_shots_raw":
                    # Wiring gate: exact reproduction with train-fitted alpha.
                    gate = run_wiring_gate(
                        origin_label, cache, frame_df, o["test_idx"],
                        o["price_frames"]["closing"], log,
                    )
                    cross_checks[f"wiring_gate_origin_{origin_label}"] = {
                        "observed": gate, "expected": ERO_GATE[origin_label]["closing"], "passed": True,
                    }
                    # Soft cross-check vs step-0's val-implied alpha for pace_shots.
                    obs_alpha = result["dispersion"]["used"]["alpha"]
                    exp_alpha = STEP0_PACE_VAL_ALPHA[origin_label]
                    ok = abs(obs_alpha - exp_alpha) <= 0.005
                    cross_checks[f"pace_val_alpha_origin_{origin_label}"] = {
                        "observed": obs_alpha, "step0_expected_approx": exp_alpha, "within_0.005": bool(ok),
                    }
                    log(
                        f"CROSS-CHECK pace_shots val-fitted alpha origin {origin_label}: observed {obs_alpha:.4f} "
                        f"vs step-0 ~{exp_alpha:.4f} -> {'OK' if ok else 'MISMATCH (investigate)'}"
                    )
                flush_log()

            predictions_df = pd.concat(row_frames_all, ignore_index=True)
            predictions_path = output_dir / f"origin_{origin_label.lower()}_test_predictions.parquet"
            predictions_df.to_parquet(predictions_path, index=False)
            log(f"Saved {len(predictions_df)} per-row test predictions to: {predictions_path}")
            results[origin_label]["_predictions_path"] = str(predictions_path)

        # ---- summary table ----
        log("\n" + "=" * 80)
        log("SUMMARY (closing pass, val-fitted dispersion, fixed EV threshold 0.05)")
        log("=" * 80)
        log(f"{'origin':<7} {'variant':<20} {'bias':>8} {'mae':>7} {'brier':>9} {'d_mkt':>9} {'d_ctl':>9} {'roi':>8} {'bets':>6} {'bet%':>6}")
        for origin_label in ("A", "B"):
            for spec in VARIANT_SPECS:
                r = results[origin_label][spec["name"]]
                pp = r["price_passes"]["closing"]
                d_ctl = pp["brier_delta_vs_no_pace_control"]
                d_ctl_s = f"{d_ctl['delta_mean']:+.5f}" if d_ctl else "      --"
                log(
                    f"{origin_label:<7} {spec['name']:<20} {r['shots_bias_test']['bias_mean']:>+8.4f} "
                    f"{r['shots_bias_test']['mae']:>7.4f} {pp['fold_wide_brier']:>9.5f} "
                    f"{pp['paired_brier_delta_vs_market']['delta_mean']:>+9.5f} {d_ctl_s:>9} "
                    f"{pp['policy_roi']['summary']['roi']:>+8.2f} {pp['policy_roi']['summary']['bets']:>6} "
                    f"{pp['policy_roi']['summary']['bet_rate']:>6.1f}"
                )
        log("bias = mean(mu - actual shots against); d_mkt = paired Brier delta vs de-vigged market "
            "(positive = worse than market); d_ctl = paired Brier delta vs no_pace_control "
            "(negative = better than control).")

        metadata["design"] = {
            "plan_reference": "docs/BREAKTHROUGH_MODEL_PLAN.md sections 4.2, 6.2 items 1-3, 6.3, 7 Gate A",
            "preregistration_reference": "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md sections 1-4",
            "origin_cap": ORIGIN_CAP,
            "fixed_ev_threshold": FIXED_EV_THRESHOLD,
            "val_window_days": VAL_WINDOW_DAYS,
            "bootstrap": {"n_resamples": N_BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                          "cluster_key": "f\"{game_id}_{goalie_id}\""},
            "dispersion_convention": (
                "All result variants use NB2 alpha moment-matched on the winner shots model's held-out "
                "VALIDATION residuals (fit_dispersion_val in this script; identical closed form to "
                "distributional_saves.fit_dispersion). Train-residual alpha computed as diagnostic only. "
                "The only train-alpha predictions are the variant (b) wiring-gate reproduction runs."
            ),
            "variant_specs": VARIANT_SPECS,
            "low_saves_thresholds": LOW_SAVES_THRESHOLDS,
        }
        metadata["fold_boundaries"] = {k: origins[k]["bounds"] for k in origins}
        metadata["cross_checks"] = cross_checks
        metadata["results"] = results
        metadata["methodology_choices"] = [
            "Season normalization (variant c) implemented as prior-only within-season league z-scores of the "
            "37 raw-level pace columns, computed on the goalie-game frame (start-weighted league environment) "
            "via the same _prior_league_stats_by_date machinery that built the 4 pre-built z columns; those 4 "
            "pre-built columns are passed through unchanged rather than re-normalized.",
            "The attempt-to-SOG funnel (variant d) is implemented as deterministic prior-only stage features "
            "(expected attempts, unblocked fraction, league Fenwick-to-SOG conversion anchored to the current "
            "season's earlier games, starter exposure share, and the composed projection) fed to the same "
            "XGBoost count:poisson shots model used by every other variant, rather than as a chain of four "
            "separately trained stage models. This keeps train/selection methodology identical across variants; "
            "the deterministic projection's own bias/MAE is reported as a mechanism diagnostic.",
            "The league Fenwick-per-game denominator in the conversion stage uses the prior league mean of the "
            "opponent Fenwick EMA (a smoothed per-game estimate) because per-game raw Fenwick is not carried in "
            "pace_features.parquet; SOG-per-game and starter-share numerators use actual outcome columns of "
            "strictly earlier dates.",
            "The save-rate model is identical (base + engineered features) for all four variants; only the "
            "shots-against model's feature set varies, matching the epd pace_shots convention.",
            "EV threshold fixed at 0.05, never swept (no market data exists inside Origin A's validation window).",
            "Origin A reports closing (PRIMARY) and bettime (SECONDARY) passes; Origin B closing only.",
            "OVER/UNDER calibration implemented as 10-bin reliability tables for p_over vs (saves>line) and "
            "p_under vs (saves<line) plus over-leaning/under-leaning summary rows.",
            "Lower-tail calibration implemented as predicted-vs-actual P(saves<=s) at s in "
            f"{LOW_SAVES_THRESHOLDS}, plus PIT histogram/coverage from intrinsic_quality_metrics on the test "
            "fold and a TOI<50-subset PIT diagnostic.",
        ]
        metadata["caveats"] = [
            "2023-24 and 2024-25 outcomes were already viewed by experiment_rolling_origin; per plan section "
            "6.1 everything here is development-tier evidence, not confirmatory.",
            "Origin A trains on roughly one season; small-sample effects are expected and reported as-is.",
            "League-environment statistics are start-weighted (computed over goalie-game rows), not "
            "team-game-weighted.",
        ]

        elapsed = time.time() - start_time
        metadata["wall_clock_seconds"] = elapsed
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        log(f"\nSaved metadata to: {metadata_path}")
        log(f"Wall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        log("\n" + "=" * 80)
        log("EXPERIMENT COMPLETE")
        log("=" * 80)
        flush_log()
    except Exception:
        flush_log()
        raise

    print(f"Saved run log to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
