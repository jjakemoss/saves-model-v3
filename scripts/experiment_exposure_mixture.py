"""
Gate A / Component A: exposure-state mixture for the goalie saves distribution.

Pre-registration: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 7
(Experiment 6) and section 1 (shared conventions), which operationalize
docs/BREAKTHROUGH_MODEL_PLAN.md sections 3.2, 4.1, 6.2 item 6, and 7 (Gate A).

Motivation (plan section 3.2, independently re-verified here and in the
pre-registration): 572 of 10,496 starts (5.45%) in
data/processed/clean_training_data.parquet have goalie TOI below 50 minutes
("early replacement" -- cause not reliably separable from injury in this
data, so no causal label is used). A single negative-binomial count process
fit across both populations at once represents the saves distribution's
lower tail poorly, because it blends two different exposure regimes (a
~28-minute-mean early-exit population and a ~60-minute-mean normal
population) into one mean/dispersion pair.

This script builds and evaluates, WITHOUT spending any Odds API credits:

  Component A part 1: a calibrated binary probability of TOI < 50, trained
  on pregame-safe features only: the base+engineered rolling feature set the
  existing "control" distributional recipe uses (already leakage-audited by
  experiments.distributional_saves.load_modeling_frame /
  FORBIDDEN_FEATURE_COLS) plus the pace goalie_workload_quality family
  (4 columns), which the pre-registration's section 7.2 names as candidate
  exposure-risk features. The plan explicitly expects only weak
  discrimination here (a prior quick attempt got AUC 0.53-0.56); per the
  pre-registration's section 7.5, AUC is NOT the point and is reported only
  against that ceiling. The value under test is CALIBRATION.

  Component A part 2: pooled (non-parametric, Laplace-smoothed) empirical TOI
  distributions for the early-replacement and normal-start populations, fit
  on TRAIN rows only. The normal-start TOI distribution naturally embeds the
  real historical regulation-vs-overtime mix (about 22% of normal starts have
  TOI > 60 minutes) -- this is the "simpler historical game-state baseline"
  the plan allows in lieu of a market-derived OT probability (plan section
  4.1; pre-registration section 7.2 makes the market-derived variant optional).

  Component A part 3 (feeds Component E's mixture formula): a shots-against-
  per-60-minutes rate model (same feature set and hyperparameter grid family
  as the existing shots model, different target), used to scale expected
  shots by each exposure state's TOI, combined with a save-rate model
  trained ONCE per origin and shared between the mixture and the controls
  (save percentage is not expected to depend on exposure length, and sharing
  it holds that piece constant so the comparison isolates the exposure
  mechanism specifically).

  Mixture assembly (plan section 4.5):
    P(saves = s) = P(early) * sum_t w_early(t) * NB2Binomial(mu60*t/60, alpha, q)[s]
                 + (1-P(early)) * sum_t w_normal(t) * NB2Binomial(mu60*t/60, alpha, q)[s]
  where w_early/w_normal are the pooled TOI-bin weights from part 2, and
  alpha is fit on VALIDATION (out-of-sample) residuals -- never training
  residuals, per plan sections 2/4.5 and pre-registration Experiment 3.

Comparators (both trained identically on the SAME rolling-origin folds as
scripts/experiment_rolling_origin.py, reusing its fold-carving code directly):

  control_train_disp -- the existing no-pace distributional recipe
  (experiments.distributional_saves.train_shots_model / fit_dispersion, i.e.
  train-residual dispersion), exactly as used elsewhere in this repo
  (pre-registration Experiment 1's architecture).

  control_val_disp -- the SAME shots/save-rate models as control_train_disp,
  but with alpha refit on VAL (out-of-sample) residuals instead of train
  residuals (pre-registration Experiment 3's change, layered on the control).

PRIMARY COMPARATOR DESIGNATION, locked here before this script's first full
run: the pre-registration's Experiment 6 pass bar compares the mixture
against "Experiment 2's Gate-A candidate, without the mixture." Experiment 2
(season-normalized funnel) is a concurrent, separate experiment whose
artifacts are not available to this run, so the closest available analog is
used and the substitution is reported rather than hidden:

  PRIMARY:   control_val_disp  (single NB2 process, validation-fitted
             dispersion -- matches the pre-registration's cross-experiment
             note that Experiments 4-6 are evaluated against the candidate
             WITH Experiment 3's dispersion fix, and uses the same dispersion
             convention as the mixture, so the comparison isolates the
             exposure-mixture mechanism itself)
  SECONDARY: control_train_disp (the literal existing single-NB2 recipe)

PASS BAR (pre-registration section 7.4, fixed before any number below
existed): on the TOI<50 subset of BOTH origins' test folds, the mixture's
summed central-coverage deviation from nominal, |cov50-50| + |cov80-80|,
must be smaller than the single-NB2 baseline's on the same subset. The
early subset is only ~126-164 goalie-nights per origin, so cluster CIs on
this slice are wide -- reported prominently per section 7.3/7.6, not hidden.

Folds (confirmed from scripts/experiment_rolling_origin.py and the
pre-registration's section 1 table, reused via direct import):
  Origin A: train <= 2022-23, val = last 49 days of that pool, test = 2023-24.
  Origin B: train <= 2023-24, val = last 49 days of that pool, test = 2024-25.

Shared conventions (pre-registration section 1): EV threshold fixed at 0.05
(never swept against either origin); goalie-night cluster bootstrap with
cluster key f"{game_id}_{goalie_id}", 10,000 resamples, seed 42, 95% CI;
PMF cap ORIGIN_CAP=90. Each origin's TEST fold is priced exactly once, after
every selection step (classifier config, Platt calibration, rate-model
config, dispersion, TOI bins) is locked on TRAIN/VAL only. Nothing here
touches the worn December 2025-April 2026 fold.

Zero Odds API / network calls. All data read from parquet files already on
disk. This script does not modify data/betting.db, src/betting/predictor.py,
any workflow file, or any existing file under models/trained/ -- it only
creates a new models/trained/experiment_exposure_mixture_<timestamp>/
directory.

Usage:
    python scripts/experiment_exposure_mixture.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Plain sys.path imports -- the proven-working pattern already used by
# scripts/clv_audit_pace_policy.py and scripts/experiment_rolling_origin.py
# to import each other. All three modules below are import-safe (function/
# constant definitions only, guarded `if __name__ == "__main__"` at the
# bottom) so importing them here triggers no computation.
import experiment_pace_distributional as epd  # noqa: E402
import experiment_rolling_origin as erx  # noqa: E402
import clv_audit_pace_policy as clv  # noqa: E402
from experiments import distributional_saves as ds  # noqa: E402
from experiments import harness as hn  # noqa: E402

make_logger = epd.make_logger

OUTPUT_ROOT = REPO_ROOT / "models" / "trained"

# Sanity gate: reproduce the plan's headline claim (section 3.2) exactly
# before building anything on top of it.
EXPECTED_TOTAL_ROWS = 10496
EXPECTED_EARLY_ROWS = 572
EARLY_TOI_THRESHOLD_MIN = 50.0

# Known ceiling from the plan's prior quick attempt (section 3.2). AUC is
# reported against this for context only; it is NOT part of the pass bar
# (pre-registration section 7.4/7.5).
PRIOR_EXPOSURE_AUC_CEILING = (0.53, 0.56)

# TOI floor applied only when constructing the shots-per-60 TRAINING target,
# to avoid a handful of near-zero-TOI outings (min observed TOI is 0.65
# minutes) producing an unbounded implied rate. Chosen by inspecting the
# target distribution before any model was fit (max raw rate hits 96
# shots/60 for the single most extreme row; only 10/10496 rows exceed 60
# even after flooring) -- not tuned against any evaluation metric.
RATE_TARGET_TOI_FLOOR_MIN = 10.0
RATE_TARGET_WINSOR_PCT = 99.5

# Pooled empirical TOI-bin widths (see build_toi_bin_distribution). 5-minute
# bins for the early-replacement population (train-pool early counts are
# small -- 97 for Origin A, 271 for Origin B -- so coarser bins keep
# per-bin sample size reasonable); 1-minute bins for the normal-start
# population (much larger sample, and 1-minute resolution is needed to
# preserve the real regulation (60) vs overtime (60-65) split).
EARLY_BIN_WIDTH_MIN = 5.0
EARLY_BIN_LO, EARLY_BIN_HI = 0.0, EARLY_TOI_THRESHOLD_MIN
NORMAL_BIN_WIDTH_MIN = 1.0
NORMAL_BIN_LO, NORMAL_BIN_HI = EARLY_TOI_THRESHOLD_MIN, 66.0
TOI_BIN_LAPLACE = 1.0

LOWER_TAIL_KS = [5, 10, 15, 20, 25, 30]
RELIABILITY_N_BINS = 8
PIT_SEED = 123  # matches intrinsic_quality_metrics' RandomState(123) for comparability

PRIMARY_BASELINE = "control_val_disp"
SECONDARY_BASELINE = "control_train_disp"

N_BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# TOI parsing and exposure label
# ---------------------------------------------------------------------------


def toi_to_minutes(toi_str) -> float:
    """clean_training_data.parquet stores TOI as 'MM:SS' strings (no NaNs in
    the current file -- verified directly). No existing repo helper does this
    conversion; the pre-registration's own re-verification used the same
    parse."""
    minutes_str, seconds_str = str(toi_str).split(":")
    return int(minutes_str) + int(seconds_str) / 60.0


# ---------------------------------------------------------------------------
# Component A part 1: exposure classifier + Platt calibration
# ---------------------------------------------------------------------------


def train_exposure_classifier(df, train_idx, val_idx, feature_cols, log, label):
    """Binary calibrated P(TOI < 50). Mirrors the structure of
    experiments.distributional_saves.train_save_rate_model (same
    hyperparameter grid, reused directly for consistency with the rest of
    this repo's recipes) but with no sample weighting (there is no natural
    per-row weight for the exposure label, unlike save-rate's shots-against
    weighting) and selection on validation log loss rather than a weighted
    metric, since log loss is exactly the primary metric this component is
    judged on (pre-registration section 7.3)."""
    ds.assert_feature_matrix_clean(df, feature_cols, f"{label} exposure", allow_nan=True)
    X_train = df[feature_cols].iloc[train_idx].astype(np.float32)
    y_train = df["is_early"].values[train_idx].astype(float)
    X_val = df[feature_cols].iloc[val_idx].astype(np.float32)
    y_val = df["is_early"].values[val_idx].astype(float)

    log(f"\n--- {label}: exposure classifier (binary:logistic), selected on VAL log loss ---")
    log(f"  TRAIN n={len(y_train)} positives={int(y_train.sum())} ({y_train.mean() * 100:.2f}%)")
    log(f"  VAL   n={len(y_val)} positives={int(y_val.sum())} ({y_val.mean() * 100:.2f}%)")
    log(f"{'config':<20} {'val_logloss':>12}")
    evaluations = []
    models = {}
    for name, cfg in ds.SAVE_RATE_CONFIGS:
        params = dict(objective="binary:logistic", random_state=42, n_jobs=-1, verbosity=0, **cfg)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred_val = np.clip(model.predict(X_val), 1e-6, 1 - 1e-6)
        val_ll = float(log_loss(y_val, pred_val, labels=[0, 1]))
        evaluations.append({"config": name, "hyperparams": cfg, "val_logloss": val_ll})
        models[name] = model
        log(f"{name:<20} {val_ll:>12.5f}")

    winner_entry = min(evaluations, key=lambda e: e["val_logloss"])
    winner_model = models[winner_entry["config"]]
    log(f"{label} exposure winner: {winner_entry['config']}  val_logloss={winner_entry['val_logloss']:.5f}")

    raw_val_pred = np.clip(winner_model.predict(X_val), 1e-6, 1 - 1e-6)
    logit_val = np.log(raw_val_pred / (1 - raw_val_pred)).reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(logit_val, y_val.astype(int))
    log(
        f"{label} Platt (sigmoid) calibration fit on VAL (n={len(y_val)}, "
        f"{int(y_val.sum())} positives): coef={platt.coef_[0][0]:.4f} intercept={platt.intercept_[0]:.4f}"
    )
    if int(y_val.sum()) < 60:
        log(
            f"  CAVEAT: VAL positive count ({int(y_val.sum())}) is small; Platt calibration "
            "(2 free parameters) is the most sample-efficient standard choice, but treat calibrated "
            "probabilities in the extreme tails with caution."
        )

    return winner_model, winner_entry, evaluations, platt


def apply_platt(raw_probs: np.ndarray, platt: LogisticRegression) -> np.ndarray:
    raw_probs = np.clip(raw_probs, 1e-6, 1 - 1e-6)
    logit = np.log(raw_probs / (1 - raw_probs)).reshape(-1, 1)
    return platt.predict_proba(logit)[:, 1]


def evaluate_exposure(y_true: np.ndarray, p_calibrated: np.ndarray, p_raw: np.ndarray, log, label: str) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    base_rate = float(y_true.mean())
    ll_raw = float(log_loss(y_true, p_raw, labels=[0, 1]))
    ll_cal = float(log_loss(y_true, p_calibrated, labels=[0, 1]))
    brier_raw = hn.brier(p_raw, y_true)
    brier_cal = hn.brier(p_calibrated, y_true)
    ll_baseline = float(log_loss(y_true, np.full(n, base_rate), labels=[0, 1]))
    brier_baseline = base_rate * (1 - base_rate)
    auc = float(roc_auc_score(y_true, p_raw)) if len(set(y_true)) > 1 else float("nan")

    reliability = reliability_table(y_true, p_calibrated, RELIABILITY_N_BINS)

    log(f"\n--- {label}: exposure component evaluation (TEST fold, n={n}, base_rate={base_rate:.4f}) ---")
    log(f"  constant-base-rate baseline: logloss={ll_baseline:.5f} brier={brier_baseline:.5f}")
    log(f"  raw model:                   logloss={ll_raw:.5f} brier={brier_raw:.5f}")
    log(f"  Platt-calibrated model:      logloss={ll_cal:.5f} brier={brier_cal:.5f}")
    log(
        f"  AUC={auc:.4f} -- context only, vs the plan's prior quick-attempt ceiling "
        f"{PRIOR_EXPOSURE_AUC_CEILING[0]:.2f}-{PRIOR_EXPOSURE_AUC_CEILING[1]:.2f}. AUC is NOT part of the "
        "pass bar (pre-registration sections 7.4/7.5); no individualized pull/injury skill is claimed."
    )
    log("  reliability table (calibrated probability, quantile bins):")
    for row in reliability:
        log(
            f"    n={row['n']:>5}  mean_pred={row['mean_pred']:.4f}  "
            f"mean_actual={row['mean_actual']:.4f}  diff={row['mean_pred'] - row['mean_actual']:+.4f}"
        )

    return {
        "n": n,
        "base_rate": base_rate,
        "raw": {"logloss": ll_raw, "brier": brier_raw, "auc": auc},
        "calibrated": {"logloss": ll_cal, "brier": brier_cal},
        "constant_baseline": {"logloss": ll_baseline, "brier": brier_baseline},
        "auc_prior_ceiling": list(PRIOR_EXPOSURE_AUC_CEILING),
        "reliability_table": reliability,
    }


def reliability_table(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int) -> list[dict]:
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(p_pred, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"], n_bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), mean_actual=("y", "mean"), n=("y", "size")
    )
    return [
        {"mean_pred": float(r.mean_pred), "mean_actual": float(r.mean_actual), "n": int(r.n)}
        for r in grouped.itertuples()
    ]


# ---------------------------------------------------------------------------
# Component A part 2: pooled empirical TOI-bin distributions (TRAIN only)
# ---------------------------------------------------------------------------


def build_toi_bin_distribution(
    toi_values: np.ndarray, bin_width: float, lo: float, hi: float, log, label: str, laplace: float = TOI_BIN_LAPLACE
) -> list[tuple[float, float]]:
    edges = np.arange(lo, hi + bin_width, bin_width)
    if edges[-1] < hi:
        edges = np.append(edges, hi)
    counts, edges = np.histogram(toi_values, bins=edges)
    counts = counts.astype(float) + laplace
    weights = counts / counts.sum()
    mids = (edges[:-1] + edges[1:]) / 2.0
    bins = list(zip(weights.tolist(), mids.tolist()))
    log(
        f"[{label}] pooled TOI-bin distribution: n_obs={len(toi_values)}, n_bins={len(bins)}, "
        f"bin_width={bin_width}min, range=[{lo},{hi}), laplace_smoothing={laplace}"
    )
    log(f"[{label}]   bin midpoints: {[round(t, 2) for _, t in bins]}")
    log(f"[{label}]   bin weights:   {[round(w, 4) for w, _ in bins]}")
    return bins


# ---------------------------------------------------------------------------
# Component A part 3: shots-per-60 rate model (feeds the mixture)
# ---------------------------------------------------------------------------


def train_shots_rate60_model(
    df, train_idx, val_idx, feature_cols, log, label,
    toi_floor: float = RATE_TARGET_TOI_FLOOR_MIN, winsor_pct: float = RATE_TARGET_WINSOR_PCT,
):
    """Same feature set and hyperparameter grid family as the control's
    shots-against model (experiments.distributional_saves.train_shots_model),
    but the target is shots_against per 60 minutes of TOI rather than raw
    shots_against, so that a row's exposure length does not confound the
    workload-rate estimate. TOI floored at toi_floor minutes and the TRAIN
    target winsorized at winsor_pct before fitting -- both applied only to
    the TRAINING target construction, never to VAL/TEST evaluation data."""
    ds.assert_feature_matrix_clean(df, feature_cols, f"{label} shots_rate60", allow_nan=True)
    floored_toi = np.maximum(df["toi_min"].values, toi_floor)
    rate60_all = df["shots_against"].values.astype(float) / (floored_toi / 60.0)

    y_train_raw = rate60_all[train_idx]
    winsor_cap = float(np.percentile(y_train_raw, winsor_pct))
    y_train = np.clip(y_train_raw, 0, winsor_cap)
    y_val = rate60_all[val_idx]

    n_clipped = int((y_train_raw > winsor_cap).sum())
    log(
        f"\n--- {label}: shots-per-60 rate model (count:poisson), selected on VAL MAE ---\n"
        f"  TOI floor={toi_floor}min for target construction; TRAIN target winsorized at "
        f"p{winsor_pct}={winsor_cap:.2f} shots/60 ({n_clipped}/{len(y_train_raw)} train rows clipped)."
    )

    X_train = df[feature_cols].iloc[train_idx].astype(np.float32)
    X_val = df[feature_cols].iloc[val_idx].astype(np.float32)

    log(f"{'config':<20} {'val_mae':>9}")
    evaluations = []
    models = {}
    for name, cfg in ds.SHOTS_CONFIGS:
        params = dict(objective="count:poisson", random_state=42, n_jobs=-1, verbosity=0, **cfg)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred_val = np.clip(model.predict(X_val), 1e-3, None)
        val_mae = ds.mae(y_val, pred_val)
        evaluations.append({"config": name, "hyperparams": cfg, "val_mae": val_mae})
        models[name] = model
        log(f"{name:<20} {val_mae:>9.4f}")

    winner_entry = min(evaluations, key=lambda e: e["val_mae"])
    winner_model = models[winner_entry["config"]]
    log(f"{label} shots-rate60 winner: {winner_entry['config']}  val_mae={winner_entry['val_mae']:.4f}")
    return winner_model, winner_entry, evaluations, winsor_cap


def fit_dispersion_oos(mu_arr: np.ndarray, y_arr: np.ndarray, log, label: str) -> tuple[float, str, dict]:
    """Same NB2 alpha derivation as experiments.distributional_saves.fit_dispersion
    (Var = mean + alpha*mean^2), but applied to caller-supplied (mu, y) pairs
    from a held-out fold rather than training residuals -- the plan's explicit
    fix for the diagnosed in-sample-dispersion-fitting failure mode
    (docs/BREAKTHROUGH_MODEL_PLAN.md section 2; pre-registration Experiment 3)."""
    resid = y_arr - mu_arr
    mean_mu = float(np.mean(mu_arr))
    var_resid = float(np.mean(resid**2))
    log(f"\n--- {label}: dispersion fit on VAL (out-of-sample) residuals ---")
    log(f"  mean(predicted mu) = {mean_mu:.4f}")
    log(f"  mean(residual^2)   = {var_resid:.4f}")
    if var_resid <= mean_mu:
        log("  VAL variance <= VAL mean: falling back to Poisson (alpha=0).")
        return 0.0, "poisson_fallback_oos", {"val_mean": mean_mu, "val_var": var_resid}
    mean_mu2 = float(np.mean(mu_arr**2))
    alpha = max((var_resid - mean_mu) / mean_mu2, 1e-6)
    log(f"  NB2 dispersion alpha (OOS) = {alpha:.6f}  (Var = mean + alpha*mean^2)")
    return alpha, "negative_binomial_oos", {"val_mean": mean_mu, "val_var": var_resid, "mean_mu2": mean_mu2}


# ---------------------------------------------------------------------------
# Mixture assembly (plan section 4.5)
# ---------------------------------------------------------------------------


def compute_mixture_distribution_predictions(
    df, idx, rate60_model, save_rate_model, alpha,
    rate_feature_cols, saverate_feature_cols, p_early_arr,
    early_bins, normal_bins, dist, log, label,
):
    """Builds a dist_preds dict with the SAME shape/keys as
    experiments.distributional_saves.compute_distribution_predictions's
    return value (mu, q, pmf, keys, lookup, idx), so the existing
    join_and_price / intrinsic_quality_metrics helpers can be reused
    unchanged against the mixture's PMF exactly as they are against the
    control's PMF."""
    X_rate = df[rate_feature_cols].iloc[idx].astype(np.float32)
    X_sr = df[saverate_feature_cols].iloc[idx].astype(np.float32)
    mu60_arr = np.clip(rate60_model.predict(X_rate), 1e-3, None)
    q_arr = np.clip(save_rate_model.predict(X_sr), 1e-6, 1 - 1e-6)

    n = len(idx)
    pmf_arr = np.zeros((n, dist.cap + 1))
    mu_expected_arr = np.zeros(n)
    for i in range(n):
        pmf_early = np.zeros(dist.cap + 1)
        e_early = 0.0
        for w, t in early_bins:
            mu_t = float(mu60_arr[i]) * t / 60.0
            pmf_t, _ = dist.saves_pmf(mu_t, alpha, float(q_arr[i]))
            pmf_early += w * pmf_t
            e_early += w * mu_t

        pmf_normal = np.zeros(dist.cap + 1)
        e_normal = 0.0
        for w, t in normal_bins:
            mu_t = float(mu60_arr[i]) * t / 60.0
            pmf_t, _ = dist.saves_pmf(mu_t, alpha, float(q_arr[i]))
            pmf_normal += w * pmf_t
            e_normal += w * mu_t

        pe = float(p_early_arr[i])
        pmf_arr[i] = pe * pmf_early + (1 - pe) * pmf_normal
        mu_expected_arr[i] = pe * e_early + (1 - pe) * e_normal

    sums = pmf_arr.sum(axis=1)
    bad = np.where(sums < 0.999)[0]
    log(
        f"[{label}] mixture pmf normalization check: n={n}, min_sum={sums.min():.6f}, "
        f"max_sum={sums.max():.6f}, rows below 0.999: {len(bad)}"
    )
    assert len(bad) == 0, f"[{label}] {len(bad)} rows have mixture pmf sum < 0.999 (cap={dist.cap} may be too small)."

    game_ids = df["game_id"].values[idx]
    goalie_ids = df["goalie_id"].values[idx]
    keys = list(zip(game_ids.tolist(), goalie_ids.tolist()))
    assert len(set(keys)) == len(keys), f"[{label}] duplicate (game_id, goalie_id) keys in mixture predictions."
    lookup = {k: i for i, k in enumerate(keys)}
    return {
        "mu": mu_expected_arr, "q": q_arr, "pmf": pmf_arr,
        "keys": keys, "lookup": lookup, "idx": idx,
        "p_early": np.asarray(p_early_arr, dtype=float), "mu60": mu60_arr,
    }


# ---------------------------------------------------------------------------
# Diagnostics: coverage/PIT (the pre-registered lower-tail pass-bar metric),
# lower-tail P(saves<=k), side calibration, mixture-vs-control paired delta
# ---------------------------------------------------------------------------


def coverage_pit_metrics(pmf_arr: np.ndarray, saves_actual_arr: np.ndarray, cap: int, seed: int = PIT_SEED) -> dict:
    """Central 50%/80% interval coverage and randomized PIT, replicating
    experiments.distributional_saves.intrinsic_quality_metrics' construction
    exactly (searchsorted on the CDF at the 25/75 and 10/90 percentiles,
    randomized PIT with RandomState(seed)) so the pass-bar numbers are
    directly comparable with every other experiment in this repo. Returns
    the per-row in-interval indicator arrays as well, so a paired cluster
    bootstrap can be run on the mixture-vs-control coverage difference."""
    saves_c = np.clip(np.asarray(saves_actual_arr, dtype=int), 0, cap)
    cdf = np.cumsum(pmf_arr, axis=1)
    n = len(saves_c)
    in50 = np.zeros(n)
    in80 = np.zeros(n)
    for i in range(n):
        row_cdf = cdf[i]
        lo50 = np.searchsorted(row_cdf, 0.25, side="left")
        hi50 = np.searchsorted(row_cdf, 0.75, side="left")
        lo80 = np.searchsorted(row_cdf, 0.10, side="left")
        hi80 = np.searchsorted(row_cdf, 0.90, side="left")
        if lo50 <= saves_c[i] <= hi50:
            in50[i] = 1.0
        if lo80 <= saves_c[i] <= hi80:
            in80[i] = 1.0

    rng = np.random.RandomState(seed)
    pit_vals = np.zeros(n)
    for i in range(n):
        y = saves_c[i]
        cdf_y = cdf[i, y]
        cdf_y_minus1 = cdf[i, y - 1] if y > 0 else 0.0
        pit_vals[i] = cdf_y_minus1 + rng.uniform() * (cdf_y - cdf_y_minus1)
    hist, _ = np.histogram(pit_vals, bins=10, range=(0, 1))
    freqs = (hist / n).tolist() if n else []

    cov50_pct = float(in50.mean() * 100) if n else float("nan")
    cov80_pct = float(in80.mean() * 100) if n else float("nan")
    return {
        "n": n,
        "cov50_pct": cov50_pct,
        "cov80_pct": cov80_pct,
        "summed_coverage_deviation": abs(cov50_pct - 50.0) + abs(cov80_pct - 80.0),
        "pit_histogram_10bins": [round(f, 4) for f in freqs],
        "_in50": in50,
        "_in80": in80,
    }


def strip_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def lower_tail_calibration(pmf_arr: np.ndarray, saves_actual_arr: np.ndarray, cap: int, ks: list[int], log, label: str) -> list[dict]:
    saves_actual_c = np.clip(saves_actual_arr, 0, cap)
    cdf = np.cumsum(pmf_arr, axis=1)
    n = len(saves_actual_c)
    rows = []
    for k in ks:
        if k > cap:
            continue
        pred_mean = float(cdf[:, k].mean())
        actual = float((saves_actual_c <= k).mean())
        rows.append({"k": k, "predicted_mean_P_le_k": pred_mean, "actual_P_le_k": actual, "diff": pred_mean - actual, "n": n})
    log(f"[{label}] lower-tail calibration (n={n}):")
    log(f"  {'k':>4} {'pred_P(saves<=k)':>18} {'actual_P(saves<=k)':>20} {'diff':>8}")
    for r in rows:
        log(f"  {r['k']:>4} {r['predicted_mean_P_le_k']:>18.4f} {r['actual_P_le_k']:>20.4f} {r['diff']:>+8.4f}")
    return rows


def side_calibration(p_over: np.ndarray, matched: np.ndarray, saves_actual: np.ndarray, lines: np.ndarray, log, label: str) -> dict:
    idx = matched
    y = (saves_actual[idx] > lines[idx]).astype(int)
    p = p_over[idx]
    over_mask = p >= 0.5
    under_mask = ~over_mask
    result: dict = {}
    if over_mask.sum() > 0:
        result["OVER_favored"] = {
            "n": int(over_mask.sum()),
            "mean_predicted_p_over": float(p[over_mask].mean()),
            "actual_over_rate": float(y[over_mask].mean()),
        }
    if under_mask.sum() > 0:
        result["UNDER_favored"] = {
            "n": int(under_mask.sum()),
            "mean_predicted_p_under": float((1 - p[under_mask]).mean()),
            "actual_under_rate": float((1 - y[under_mask]).mean()),
        }
    log(f"[{label}] OVER/UNDER calibration by predicted side: {result}")
    return result


def paired_delta_between_models(
    df_bet: pd.DataFrame, p_over_a: np.ndarray, matched_a: np.ndarray, p_over_b: np.ndarray, matched_b: np.ndarray,
    model_a_name: str, model_b_name: str, log, label: str,
) -> dict:
    """Same paired-per-row-squared-error-difference + goalie-night cluster
    bootstrap methodology as experiment_rolling_origin.paired_brier_delta,
    generalized to compare two model probability arrays against each other
    instead of one model against the de-vigged market."""
    both_matched = matched_a & matched_b
    saves_arr = df_bet["saves"].values.astype(float)
    lines_arr = df_bet["betting_line"].values.astype(float)
    game_id_arr = df_bet["game_id"].values
    goalie_id_arr = df_bet["goalie_id"].values

    y = (saves_arr[both_matched] > lines_arr[both_matched]).astype(float)
    sq_a = (p_over_a[both_matched] - y) ** 2
    sq_b = (p_over_b[both_matched] - y) ** 2
    delta = sq_a - sq_b
    cluster_ids = np.array(
        [f"{int(g)}_{int(go)}" for g, go in zip(game_id_arr[both_matched], goalie_id_arr[both_matched])], dtype=object
    )

    stat = clv.cluster_bootstrap_mean_ci(delta, cluster_ids, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, ci_pct=95.0)
    log(
        f"[{label}] paired Brier delta ({model_a_name} - {model_b_name}): "
        f"{model_a_name}_brier={sq_a.mean():.5f} {model_b_name}_brier={sq_b.mean():.5f} "
        f"delta={stat['mean']:+.5f} 95% CI=[{stat['lower']:+.5f}, {stat['upper']:+.5f}] "
        f"n_rows={stat['n_bets']} n_clusters={stat['n_clusters']}"
    )
    return {
        f"{model_a_name}_brier_mean": float(sq_a.mean()) if len(sq_a) else None,
        f"{model_b_name}_brier_mean": float(sq_b.mean()) if len(sq_b) else None,
        "delta_mean": stat["mean"], "delta_ci95_lower": stat["lower"], "delta_ci95_upper": stat["upper"],
        "n_rows": stat["n_bets"], "n_clusters": stat["n_clusters"],
    }


def paired_coverage_difference_ci(
    in_mixture: np.ndarray, in_baseline: np.ndarray, cluster_ids: np.ndarray, log, label: str
) -> dict:
    """Cluster-bootstrap CI on the per-goalie-night difference of in-interval
    indicators (mixture minus baseline). On clean-training test folds each
    row IS one goalie-night, so this is effectively a row bootstrap, run
    through the same cluster machinery for convention compliance
    (pre-registration section 1)."""
    diff = np.asarray(in_mixture, dtype=float) - np.asarray(in_baseline, dtype=float)
    stat = clv.cluster_bootstrap_mean_ci(diff, cluster_ids, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, ci_pct=95.0)
    log(
        f"[{label}] paired coverage-indicator difference (mixture - baseline): "
        f"mean={stat['mean']:+.4f} 95% CI=[{stat['lower']:+.4f}, {stat['upper']:+.4f}] "
        f"n={stat['n_bets']} clusters={stat['n_clusters']}"
    )
    return stat


# ---------------------------------------------------------------------------
# Per-origin runner
# ---------------------------------------------------------------------------


def run_origin(
    origin_label: str, frame_df: pd.DataFrame, core_feature_cols: list[str], exposure_feature_cols: list[str],
    train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
    price_frames: dict, test_season: int, output_dir: Path, log,
) -> dict:
    log("\n" + "=" * 80)
    log(f"ORIGIN {origin_label}: exposure-state mixture vs single-NB2 no-pace controls")
    log("=" * 80)

    # ---- CONTROL: reuse the existing recipe unchanged (shots + save-rate on core features) ----
    control_shots_model, control_shots_winner, control_shots_evals = ds.train_shots_model(
        frame_df, train_idx, val_idx, core_feature_cols, log, f"origin_{origin_label} control"
    )
    control_alpha_train, control_disp_train_method, control_disp_train_diag = ds.fit_dispersion(
        control_shots_model, frame_df, train_idx, core_feature_cols, log, f"origin_{origin_label} control"
    )
    save_rate_model, save_rate_winner, save_rate_evals = ds.train_save_rate_model(
        frame_df, train_idx, val_idx, core_feature_cols, log, f"origin_{origin_label} shared_save_rate"
    )

    X_val_ctrl = frame_df[core_feature_cols].iloc[val_idx].astype(np.float32)
    mu_val_ctrl = np.clip(control_shots_model.predict(X_val_ctrl), 1e-3, None)
    shots_val_actual = frame_df["shots_against"].values[val_idx].astype(float)
    control_alpha_val, control_disp_val_method, control_disp_val_diag = fit_dispersion_oos(
        mu_val_ctrl, shots_val_actual, log, f"origin_{origin_label} control_val_disp"
    )

    shots_path = output_dir / f"origin_{origin_label.lower()}_control_shots_model.json"
    save_rate_path = output_dir / f"origin_{origin_label.lower()}_shared_save_rate_model.json"
    control_shots_model.get_booster().save_model(str(shots_path))
    save_rate_model.get_booster().save_model(str(save_rate_path))
    log(f"Saved control shots model to: {shots_path}")
    log(f"Saved shared save-rate model to: {save_rate_path}")

    # ---- MIXTURE: exposure classifier (core + goalie_workload_quality pace family) ----
    exposure_model, exposure_winner, exposure_evals, platt = train_exposure_classifier(
        frame_df, train_idx, val_idx, exposure_feature_cols, log, f"origin_{origin_label} exposure"
    )
    exposure_model_path = output_dir / f"origin_{origin_label.lower()}_exposure_classifier.json"
    exposure_model.get_booster().save_model(str(exposure_model_path))
    log(f"Saved exposure classifier to: {exposure_model_path}")

    # ---- MIXTURE: shots-per-60 rate model (core features) ----
    rate60_model, rate60_winner, rate60_evals, rate60_winsor_cap = train_shots_rate60_model(
        frame_df, train_idx, val_idx, core_feature_cols, log, f"origin_{origin_label} rate60"
    )
    rate60_path = output_dir / f"origin_{origin_label.lower()}_shots_rate60_model.json"
    rate60_model.get_booster().save_model(str(rate60_path))
    log(f"Saved shots-rate60 model to: {rate60_path}")

    X_val_rate = frame_df[core_feature_cols].iloc[val_idx].astype(np.float32)
    pred_rate60_val = np.clip(rate60_model.predict(X_val_rate), 1e-3, None)
    real_toi_val = frame_df["toi_min"].values[val_idx]
    mu_actual_val_mixture = pred_rate60_val * (real_toi_val / 60.0)
    mixture_alpha, mixture_disp_method, mixture_disp_diag = fit_dispersion_oos(
        mu_actual_val_mixture, shots_val_actual, log, f"origin_{origin_label} mixture"
    )

    # ---- Component A part 2: pooled TOI-bin distributions from TRAIN only ----
    train_toi = frame_df["toi_min"].values[train_idx]
    train_is_early = frame_df["is_early"].values[train_idx].astype(bool)
    early_bins = build_toi_bin_distribution(
        train_toi[train_is_early], EARLY_BIN_WIDTH_MIN, EARLY_BIN_LO, EARLY_BIN_HI, log, f"origin_{origin_label} early"
    )
    normal_bins = build_toi_bin_distribution(
        train_toi[~train_is_early], NORMAL_BIN_WIDTH_MIN, NORMAL_BIN_LO, NORMAL_BIN_HI, log, f"origin_{origin_label} normal"
    )
    normal_ot_frac = float((train_toi[~train_is_early] > 60.05).mean())
    log(
        f"[origin_{origin_label}] historical OT/SO baseline embedded in the normal-start TOI "
        f"distribution (TRAIN, fraction with TOI>60.05min): {normal_ot_frac:.4f} -- used in place of a "
        "market-derived OT probability, per plan section 4.1's allowed simplification (pre-registration "
        "7.2 makes the market-derived variant optional)."
    )

    # ---- calibrated exposure probabilities on TEST ----
    X_test_exp = frame_df[exposure_feature_cols].iloc[test_idx].astype(np.float32)
    p_early_raw_test = np.clip(exposure_model.predict(X_test_exp), 1e-6, 1 - 1e-6)
    p_early_cal_test = apply_platt(p_early_raw_test, platt)

    is_early_test = frame_df["is_early"].values[test_idx].astype(int)
    exposure_eval = evaluate_exposure(is_early_test, p_early_cal_test, p_early_raw_test, log, f"origin_{origin_label} exposure")

    # ---- assemble PMFs on TEST ----
    dist = ds.SavesDistribution(erx.ORIGIN_CAP)

    dist_preds_control_train = ds.compute_distribution_predictions(
        frame_df, test_idx, control_shots_model, save_rate_model, control_alpha_train,
        core_feature_cols, core_feature_cols, dist, log, f"origin_{origin_label} control_train_disp TEST",
    )
    dist_preds_control_val = ds.compute_distribution_predictions(
        frame_df, test_idx, control_shots_model, save_rate_model, control_alpha_val,
        core_feature_cols, core_feature_cols, dist, log, f"origin_{origin_label} control_val_disp TEST",
    )
    dist_preds_mixture = compute_mixture_distribution_predictions(
        frame_df, test_idx, rate60_model, save_rate_model, mixture_alpha,
        core_feature_cols, core_feature_cols, p_early_cal_test,
        early_bins, normal_bins, dist, log, f"origin_{origin_label} mixture TEST",
    )

    model_pmfs = {
        "mixture": dist_preds_mixture,
        "control_train_disp": dist_preds_control_train,
        "control_val_disp": dist_preds_control_val,
    }

    # ---- intrinsic quality (shots MAE, PIT, coverage) on the full TEST fold ----
    intrinsic = {}
    for model_name, dist_preds in model_pmfs.items():
        intrinsic[model_name] = ds.intrinsic_quality_metrics(
            frame_df, test_idx, dist_preds, dist, log, f"origin_{origin_label}_{model_name}"
        )

    # ---- explicit shots bias (Gate A criterion 1 context) ----
    shots_actual_test = frame_df["shots_against"].values[test_idx].astype(float)
    shots_bias = {
        model_name: float(np.mean(dist_preds["mu"] - shots_actual_test)) for model_name, dist_preds in model_pmfs.items()
    }
    log(f"[origin_{origin_label}] TEST shots bias (mean predicted mu - actual shots_against): {shots_bias}")

    # ---- PRE-REGISTERED PASS BAR (section 7.4): coverage on the TOI<50 subset ----
    saves_actual_test = frame_df["saves"].values[test_idx].astype(int)
    is_early_test_bool = is_early_test.astype(bool)
    n_early_test = int(is_early_test_bool.sum())
    early_cluster_ids = np.array(
        [
            f"{int(g)}_{int(go)}"
            for g, go in zip(
                frame_df["game_id"].values[test_idx][is_early_test_bool],
                frame_df["goalie_id"].values[test_idx][is_early_test_bool],
            )
        ],
        dtype=object,
    )
    all_cluster_ids = np.array(
        [
            f"{int(g)}_{int(go)}"
            for g, go in zip(frame_df["game_id"].values[test_idx], frame_df["goalie_id"].values[test_idx])
        ],
        dtype=object,
    )

    log("\n" + "-" * 80)
    log(
        f"[origin_{origin_label}] PRE-REGISTERED PASS-BAR METRIC (PREREGISTRATION_NO_CREDIT_ABLATIONS.md "
        f"section 7.4): central 50%/80% coverage on the TOI<50 TEST subset (n={n_early_test} goalie-nights)."
    )
    log(
        f"[origin_{origin_label}] SMALL-SAMPLE CAVEAT (section 7.3/7.6): {n_early_test} goalie-nights is a "
        "small slice; CIs on this subset are wide and the coverage comparison is directionally "
        "informative, not tightly estimated."
    )
    log("-" * 80)

    coverage_results: dict = {}
    for subset_name, subset_mask, subset_clusters in (
        ("ALL", np.ones(len(test_idx), dtype=bool), all_cluster_ids),
        ("EARLY_SUBSET", is_early_test_bool, early_cluster_ids),
    ):
        coverage_results[subset_name] = {}
        for model_name, dist_preds in model_pmfs.items():
            cov = coverage_pit_metrics(dist_preds["pmf"][subset_mask], saves_actual_test[subset_mask], dist.cap)
            coverage_results[subset_name][model_name] = cov
            log(
                f"[origin_{origin_label}] {subset_name:<13} {model_name:<20} n={cov['n']:>5} "
                f"cov50={cov['cov50_pct']:6.2f}% cov80={cov['cov80_pct']:6.2f}% "
                f"summed_dev={cov['summed_coverage_deviation']:7.3f}  PIT={cov['pit_histogram_10bins']}"
            )

    pass_bar = {}
    for baseline_name, role in ((PRIMARY_BASELINE, "PRIMARY"), (SECONDARY_BASELINE, "SECONDARY")):
        mix_dev = coverage_results["EARLY_SUBSET"]["mixture"]["summed_coverage_deviation"]
        base_dev = coverage_results["EARLY_SUBSET"][baseline_name]["summed_coverage_deviation"]
        passed = bool(mix_dev < base_dev)
        cov50_diff_ci = paired_coverage_difference_ci(
            coverage_results["EARLY_SUBSET"]["mixture"]["_in50"],
            coverage_results["EARLY_SUBSET"][baseline_name]["_in50"],
            early_cluster_ids, log, f"origin_{origin_label} EARLY cov50 mixture-vs-{baseline_name}",
        )
        cov80_diff_ci = paired_coverage_difference_ci(
            coverage_results["EARLY_SUBSET"]["mixture"]["_in80"],
            coverage_results["EARLY_SUBSET"][baseline_name]["_in80"],
            early_cluster_ids, log, f"origin_{origin_label} EARLY cov80 mixture-vs-{baseline_name}",
        )
        log(
            f"[origin_{origin_label}] PASS BAR vs {baseline_name} ({role}): mixture summed_dev={mix_dev:.3f} "
            f"vs baseline {base_dev:.3f} -> {'PASS' if passed else 'FAIL'} (point comparison per section 7.4; "
            "CIs above are the honesty check on the small subset)."
        )
        pass_bar[baseline_name] = {
            "role": role,
            "mixture_summed_dev": mix_dev,
            "baseline_summed_dev": base_dev,
            "passed_point_comparison": passed,
            "early_subset_n": n_early_test,
            "cov50_indicator_diff_ci": cov50_diff_ci,
            "cov80_indicator_diff_ci": cov80_diff_ci,
        }

    # strip per-row indicator arrays before coverage_results goes to metadata
    coverage_results_meta = {
        subset: {m: strip_private(v) for m, v in models.items()} for subset, models in coverage_results.items()
    }

    # ---- lower-tail P(saves<=k) diagnostics: all TEST rows, and the early subset ----
    lower_tail = {}
    for model_name, dist_preds in model_pmfs.items():
        lower_tail[f"{model_name}__ALL"] = lower_tail_calibration(
            dist_preds["pmf"], saves_actual_test, dist.cap, LOWER_TAIL_KS, log, f"origin_{origin_label} {model_name} ALL"
        )
        lower_tail[f"{model_name}__EARLY_SUBSET"] = lower_tail_calibration(
            dist_preds["pmf"][is_early_test_bool], saves_actual_test[is_early_test_bool], dist.cap, LOWER_TAIL_KS,
            log, f"origin_{origin_label} {model_name} EARLY_SUBSET(n={n_early_test})",
        )

    # ---- price against betting frames, per model, per price pass ----
    pass_results = {}
    all_row_frames = []
    for pass_name, df_bet in price_frames.items():
        pricing = {}
        for model_name, dist_preds in model_pmfs.items():
            label = f"origin_{origin_label} {model_name} TEST {pass_name}"
            p_over, p_under, p_push, matched, cov = ds.join_and_price(df_bet, dist_preds, dist, log, label)
            auc, brier_val = hn.fold_wide_auc_brier(
                p_over, matched, df_bet["saves"].values, df_bet["betting_line"].values,
                df_bet["game_id"].values, df_bet["goalie_id"].values, log, label,
            )
            brier_delta_stat, market_p_over_arr, market_p_under_arr = erx.paired_brier_delta(df_bet, p_over, matched, log, label)
            bet_results = hn.grade_bets(
                p_over, p_under, df_bet["saves"].values.astype(float), df_bet["betting_line"].values.astype(float),
                df_bet["odds_over_american"].astype(float).values, df_bet["odds_under_american"].astype(float).values,
                df_bet["game_id"].values, df_bet["goalie_id"].values, erx.FIXED_EV_THRESHOLD, matched, log, label,
            )
            bundle = hn.betting_metrics_bundle(bet_results, df_bet["game_id"].values, df_bet["goalie_id"].values, len(df_bet))
            log(
                f"[{label}] {bundle['summary']['bets']} bets, {bundle['summary']['bet_rate']:.1f}% bet rate, "
                f"{bundle['summary']['hit_rate']:.1f}% hit rate, {bundle['summary']['roi']:+.2f}% ROI"
            )
            side_cal = side_calibration(p_over, matched, df_bet["saves"].values.astype(float), df_bet["betting_line"].values.astype(float), log, label)

            row_df = erx.build_row_predictions(
                df_bet, p_over, p_under, matched, market_p_over_arr, market_p_under_arr,
                erx.FIXED_EV_THRESHOLD, origin_label, pass_name,
            )
            row_df["model"] = model_name
            all_row_frames.append(row_df)

            pricing[model_name] = {
                "join_coverage_pct": cov, "fold_wide_auc": auc, "fold_wide_brier": brier_val,
                "paired_brier_delta_vs_market": brier_delta_stat, "policy_roi": bundle,
                "side_calibration": side_cal, "ev_threshold": erx.FIXED_EV_THRESHOLD,
                "p_over": p_over, "matched": matched,
            }

        deltas = {}
        for baseline_name in (PRIMARY_BASELINE, SECONDARY_BASELINE):
            deltas[f"mixture_vs_{baseline_name}"] = paired_delta_between_models(
                df_bet, pricing["mixture"]["p_over"], pricing["mixture"]["matched"],
                pricing[baseline_name]["p_over"], pricing[baseline_name]["matched"],
                "mixture", baseline_name, log, f"origin_{origin_label} mixture_vs_{baseline_name} TEST {pass_name}",
            )

        for model_name in pricing:
            del pricing[model_name]["p_over"]
            del pricing[model_name]["matched"]
        pass_results[pass_name] = {"per_model": pricing, "model_vs_model_brier_deltas": deltas}

    predictions_df = pd.concat(all_row_frames, ignore_index=True)
    predictions_path = output_dir / f"origin_{origin_label.lower()}_test_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    log(f"Saved {len(predictions_df)} per-row test predictions (mixture + both controls) to: {predictions_path}")

    exposure_df = pd.DataFrame(
        {
            "game_id": frame_df["game_id"].values[test_idx],
            "goalie_id": frame_df["goalie_id"].values[test_idx],
            "goalie_name": frame_df["goalie_name"].values[test_idx] if "goalie_name" in frame_df.columns else None,
            "team_abbrev": frame_df["team_abbrev"].values[test_idx] if "team_abbrev" in frame_df.columns else None,
            "opponent_team": frame_df["opponent_team"].values[test_idx] if "opponent_team" in frame_df.columns else None,
            "game_date": frame_df["game_date"].values[test_idx],
            "test_season": test_season,
            "actual_toi_min": frame_df["toi_min"].values[test_idx],
            "actual_is_early": is_early_test,
            "p_early_raw": p_early_raw_test,
            "p_early_calibrated": p_early_cal_test,
            "actual_saves": frame_df["saves"].values[test_idx],
            "actual_shots_against": frame_df["shots_against"].values[test_idx],
        }
    )
    exposure_preds_path = output_dir / f"origin_{origin_label.lower()}_exposure_predictions.parquet"
    exposure_df.to_parquet(exposure_preds_path, index=False)
    log(f"Saved {len(exposure_df)} per-goalie-night exposure predictions to: {exposure_preds_path}")

    return {
        "origin": origin_label,
        "test_season": test_season,
        "core_feature_count": len(core_feature_cols),
        "exposure_feature_count": len(exposure_feature_cols),
        "exposure_only_extra_cols": [c for c in exposure_feature_cols if c not in core_feature_cols],
        "control_shots_model": {"winner": control_shots_winner, "val_evaluations": control_shots_evals, "model_path": str(shots_path)},
        "shared_save_rate_model": {"winner": save_rate_winner, "val_evaluations": save_rate_evals, "model_path": str(save_rate_path)},
        "control_dispersion_train": {"alpha": control_alpha_train, "method": control_disp_train_method, "diagnostics": control_disp_train_diag},
        "control_dispersion_val": {"alpha": control_alpha_val, "method": control_disp_val_method, "diagnostics": control_disp_val_diag},
        "exposure_classifier": {"winner": exposure_winner, "val_evaluations": exposure_evals, "model_path": str(exposure_model_path),
                                  "platt_coef": float(platt.coef_[0][0]), "platt_intercept": float(platt.intercept_[0])},
        "shots_rate60_model": {"winner": rate60_winner, "val_evaluations": rate60_evals, "model_path": str(rate60_path),
                                 "train_winsor_cap": rate60_winsor_cap},
        "mixture_dispersion": {"alpha": mixture_alpha, "method": mixture_disp_method, "diagnostics": mixture_disp_diag},
        "early_bins": early_bins, "normal_bins": normal_bins, "normal_ot_fraction_train": normal_ot_frac,
        "exposure_eval_test": exposure_eval,
        "intrinsic_quality_test": intrinsic,
        "shots_bias_test": shots_bias,
        "coverage_results_test": coverage_results_meta,
        "preregistered_pass_bar_section_7_4": pass_bar,
        "lower_tail_calibration_test": lower_tail,
        "price_passes": pass_results,
        "predictions_path": str(predictions_path),
        "exposure_predictions_path": str(exposure_preds_path),
        "test_idx_n": int(len(test_idx)),
        "test_early_n": n_early_test,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_exposure_mixture_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log, flush_log = make_logger(log_path)

    metadata: dict = {"timestamp": datetime.now().isoformat()}
    try:
        log("=" * 80)
        log("EXPOSURE-STATE MIXTURE EXPERIMENT (Gate A / Component A / pre-registration Experiment 6)")
        log("Pre-registration: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md sections 1 and 7")
        log("Plan: docs/BREAKTHROUGH_MODEL_PLAN.md sections 3.2, 4.1, 6.2 item 6, 7 (Gate A)")
        log("=" * 80)
        log(f"Output directory: {output_dir}")

        for path in (
            epd.DATA_PATH_CLEAN, epd.DATA_PATH_CONTEXT, epd.DATA_PATH_PACE, epd.DATA_PATH_PACE_METADATA,
            epd.DATA_PATH_MULTIBOOK, erx.CLOSING_FRAME_PATH, erx.BETTIME_FRAME_PATH,
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required input: {path}")

        # ---- SANITY GATE: reproduce the plan's 572/10496 = 5.45% claim ----
        log("\n" + "=" * 80)
        log("SANITY GATE: reproduce plan section 3.2's TOI<50 claim from clean_training_data.parquet")
        log("=" * 80)
        clean_raw = pd.read_parquet(epd.DATA_PATH_CLEAN)
        clean_raw["toi_min"] = clean_raw["toi"].apply(toi_to_minutes)
        n_total = len(clean_raw)
        n_early = int((clean_raw["toi_min"] < EARLY_TOI_THRESHOLD_MIN).sum())
        pct_early = n_early / n_total * 100
        log(f"Observed: {n_early}/{n_total} = {pct_early:.2f}%  Expected: {EXPECTED_EARLY_ROWS}/{EXPECTED_TOTAL_ROWS} = 5.45%")
        if n_total != EXPECTED_TOTAL_ROWS or n_early != EXPECTED_EARLY_ROWS:
            raise AssertionError(
                f"SANITY GATE FAILED: expected {EXPECTED_EARLY_ROWS}/{EXPECTED_TOTAL_ROWS}, "
                f"observed {n_early}/{n_total}. Stopping rather than building on an unverified premise."
            )
        log("SANITY GATE PASSED.")
        metadata["sanity_gate"] = {"n_total": n_total, "n_early": n_early, "pct_early": pct_early}

        # ---- shared modeling frame ----
        log("\n" + "=" * 80)
        log("Loading shared modeling frame (base + engineered + pace, per pre-registration 7.2)")
        log("=" * 80)
        frame = epd.load_pace_modeling_frame(
            epd.DATA_PATH_CLEAN, epd.DATA_PATH_CONTEXT, epd.DATA_PATH_PACE, epd.DATA_PATH_PACE_METADATA, log,
        )
        frame_df = frame.df.copy()
        frame_df["toi_min"] = frame_df["toi"].apply(toi_to_minutes)
        frame_df["is_early"] = (frame_df["toi_min"] < EARLY_TOI_THRESHOLD_MIN).astype(int)

        core_feature_cols = frame.base_feature_cols + frame.engineered_cols
        exposure_feature_cols = core_feature_cols + frame.pace_goalie_workload_cols
        log(
            f"Core feature set (shots-rate60 / save-rate / control shots): {len(core_feature_cols)} columns "
            "-- identical to the existing no-pace control recipe. No pace/market-context features added to "
            "the count models, keeping this component isolated per plan section 6.2."
        )
        log(
            f"Exposure-classifier feature set: {len(exposure_feature_cols)} columns = core + the pace "
            f"goalie_workload_quality family ({frame.pace_goalie_workload_cols}), the candidate exposure-risk "
            "features named in pre-registration section 7.2. All pregame-safe (prior-only rolling/EMA)."
        )

        clean_full = pd.read_parquet(epd.DATA_PATH_CLEAN)
        clean_full["game_date"] = pd.to_datetime(clean_full["game_date"])

        df_bet_full = ds.build_betting_frame(epd.DATA_PATH_MULTIBOOK, log)

        origin_specs = [
            ("A", [erx.SEASON_2022_23], erx.SEASON_2023_24),
            ("B", [erx.SEASON_2022_23, erx.SEASON_2023_24], erx.SEASON_2024_25),
        ]

        results = {}
        for origin_label, train_seasons, test_season in origin_specs:
            log("\n" + "=" * 80)
            log(f"Carving Origin {origin_label} folds (reusing experiment_rolling_origin's carve logic)")
            log("=" * 80)
            pool_min, pool_max = erx.season_date_range(clean_full, train_seasons)
            train_idx, val_idx, boundaries = erx.carve_origin_split(
                frame_df, pool_min, pool_max, erx.VAL_WINDOW_DAYS, log, f"Origin {origin_label}"
            )
            test_min, test_max = erx.season_date_range(clean_full, [test_season])
            test_idx = erx.date_range_test_idx(frame_df, test_min, test_max, log, f"Origin {origin_label}")

            if origin_label == "A":
                price_frames = {
                    "closing": ds.build_betting_frame(erx.CLOSING_FRAME_PATH, log),
                    "bettime": ds.build_betting_frame(erx.BETTIME_FRAME_PATH, log),
                }
                log(
                    "Origin A price frames loaded from existing rolling-origin artifacts (closing is PRIMARY; "
                    "bettime is a secondary descriptive pass, matching experiment_rolling_origin.py's convention)."
                )
            else:
                closing_b = df_bet_full[df_bet_full["season"] == test_season].reset_index(drop=True)
                price_frames = {"closing": closing_b}
                log(
                    f"Origin B price frame: multibook_classification_training_data.parquet season {test_season} "
                    f"closing rows ({len(closing_b)}). No usable 2024-25 bettime pass exists (pre-registration 1.2)."
                )

            origin_result = run_origin(
                origin_label, frame_df, core_feature_cols, exposure_feature_cols,
                train_idx, val_idx, test_idx, price_frames, test_season, output_dir, log,
            )
            origin_result["fold_boundaries"] = {**boundaries, "test_season": test_season, "test_rows": int(len(test_idx))}
            results[origin_label] = origin_result
            flush_log()

        # ---- overall pre-registered verdict: both origins must pass ----
        log("\n" + "=" * 80)
        log("OVERALL PRE-REGISTERED VERDICT (section 7.4: pass requires BOTH origins)")
        log("=" * 80)
        overall = {}
        for baseline_name in (PRIMARY_BASELINE, SECONDARY_BASELINE):
            a_pass = results["A"]["preregistered_pass_bar_section_7_4"][baseline_name]["passed_point_comparison"]
            b_pass = results["B"]["preregistered_pass_bar_section_7_4"][baseline_name]["passed_point_comparison"]
            overall[baseline_name] = {"origin_a_pass": a_pass, "origin_b_pass": b_pass, "overall_pass": bool(a_pass and b_pass)}
            log(
                f"  vs {baseline_name}: Origin A {'PASS' if a_pass else 'FAIL'}, "
                f"Origin B {'PASS' if b_pass else 'FAIL'} -> overall {'PASS' if (a_pass and b_pass) else 'FAIL'}"
            )
        log(
            "  Reminder (pre-registration 7.6 / plan 6.1): Origin A improvement is hypothesis support only; "
            "both origins agreeing is the strongest no-purchase evidence available; the early-subset sample "
            "is small and CIs are wide. Gate A is judged on the COMBINED architecture, not this experiment alone."
        )
        metadata["overall_preregistered_verdict"] = overall

        metadata["origin_a"] = results["A"]
        metadata["origin_b"] = results["B"]
        metadata["constants"] = {
            "early_toi_threshold_min": EARLY_TOI_THRESHOLD_MIN,
            "rate_target_toi_floor_min": RATE_TARGET_TOI_FLOOR_MIN,
            "rate_target_winsor_pct": RATE_TARGET_WINSOR_PCT,
            "early_bin_width_min": EARLY_BIN_WIDTH_MIN,
            "normal_bin_width_min": NORMAL_BIN_WIDTH_MIN,
            "toi_bin_laplace": TOI_BIN_LAPLACE,
            "lower_tail_ks": LOWER_TAIL_KS,
            "origin_cap": erx.ORIGIN_CAP,
            "fixed_ev_threshold": erx.FIXED_EV_THRESHOLD,
            "pit_seed": PIT_SEED,
            "primary_baseline": PRIMARY_BASELINE,
            "secondary_baseline": SECONDARY_BASELINE,
            "prior_exposure_auc_ceiling": list(PRIOR_EXPOSURE_AUC_CEILING),
            "n_bootstrap_resamples": N_BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }
        metadata["methodology_notes"] = [
            "Pre-registration alignment: this run implements Experiment 6 of "
            "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md. The pass bar is section 7.4's summed "
            "lower-tail coverage deviation (|cov50-50| + |cov80-80|) on the TOI<50 subset of both "
            "origins' test folds, mixture vs single-NB2 baseline. Exposure-classifier AUC is reported "
            "only against the known 0.53-0.56 ceiling and is NOT part of the pass bar.",
            "PRIMARY-baseline substitution, locked before this script's first full run: section 7.4 "
            "names 'Experiment 2's Gate-A candidate without the mixture' as the baseline, but Experiment "
            "2 (season-normalized funnel) runs concurrently in a separate agent and its artifacts are "
            "not available here. control_val_disp (single NB2, validation-fitted dispersion -- Experiment "
            "1's architecture with Experiment 3's dispersion fix layered on, per the pre-registration's "
            "own cross-experiment layering note) is the designated PRIMARY stand-in; control_train_disp "
            "(the literal existing recipe) is SECONDARY. Both comparisons are reported.",
            "Exposure classifier features = base pre-game rolling features + line-independent engineered "
            "features + the pace goalie_workload_quality family (4 prior-only EMA/rolling columns), the "
            "candidate exposure-risk features named in pre-registration 7.2. FORBIDDEN_FEATURE_COLS "
            "(current-game saves, shots_against, goals_against, toi, decision, situational splits, "
            "team/opp current-game goals and shots) is asserted clean by the shared loader and by "
            "assert_feature_matrix_clean before every fit. Backup-goalie quality was NOT derived (would "
            "require pregame roster/depth-chart data this repo does not have) -- reported as a choice, "
            "not an oversight.",
            "OT probability is NOT derived from moneyline/total market inputs (Component C is Experiment "
            "5, separate). The pooled empirical TOI distribution for normal starts is used directly, "
            "which embeds the real historical OT/SO rate (~22% of normal starts have TOI>60min) -- the "
            "'simpler historical game-state baseline' plan section 4.1 explicitly allows.",
            "Save-rate model is trained ONCE per origin and shared between the controls and the "
            "mixture's per-state q, isolating the exposure-scaling mechanism as the sole difference "
            "under test (beyond the dispersion convention, which control_val_disp equalizes).",
            "Dispersion for the mixture and control_val_disp is fit on VAL (out-of-sample) residuals, "
            "never train residuals. control_train_disp intentionally keeps the existing train-residual "
            "fit unchanged, since that is the literal existing single-process recipe.",
            "Pooled (non-parametric, Laplace-smoothed) empirical TOI-bin distributions, fit on TRAIN "
            "rows only, are used for both exposure states rather than a per-row predictive TOI model -- "
            "matching plan section 9's disposition ('retain a pooled calibrated early-replacement "
            "mixture... do not claim individualized pull/injury skill').",
            "Shots-per-60 rate model target is floored at 10 minutes TOI and winsorized at the 99.5th "
            "percentile (TRAIN only) before fitting, to prevent a handful of near-zero-TOI outings (min "
            "observed TOI 0.65 minutes) from producing unbounded implied rates.",
            "The mixture's alpha is fit on VAL residuals of the rate60 model's actual-TOI-scaled mu "
            "(mu60 * actual_val_TOI / 60), so it captures rate dispersion beyond exposure; exposure "
            "variance itself is carried by the TOI-bin mixture at test time, not double-counted in alpha.",
            "Each origin's TEST fold was priced exactly once, after all selection (classifier config, "
            "Platt calibration, rate60 config, dispersion, TOI bins) was locked on TRAIN/VAL. The worn "
            "December 2025-April 2026 fold was not touched.",
        ]

        elapsed = time.time() - start_time
        metadata["wall_clock_seconds"] = elapsed
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        log(f"\nSaved metadata to: {metadata_path}")
        log(f"Wall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        log("\n" + "=" * 80)
        log("EXPOSURE-STATE MIXTURE EXPERIMENT COMPLETE")
        log("=" * 80)
        flush_log()
    except Exception:
        flush_log()
        raise

    print(f"Saved run log to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
