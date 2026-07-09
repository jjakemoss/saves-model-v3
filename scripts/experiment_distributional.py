"""
Roadmap item 9: distributional saves model prototype.

docs/OFFSEASON_OPTIMIZATION_PLAN.md sections 3.5, 3.9, 3.10, 3.11 (motivation)
and 3.3 (selection methodology bar this experiment must clear).

BUSINESS QUESTION: model saves as a COUNT DISTRIBUTION -- decomposed into
shots-against and save-rate submodels -- and price P(over line) analytically
for any posted line, then run it head-to-head against the existing classifier
on the same chronological folds through the honest harness. Empirical
motivation (verified 2026-07-08, section 3.11): saves correlates 0.979 with
shots against (R^2 0.959) and only 0.559 with save percentage -- a saves line
is a shot-volume prop first, and the classifier barely models shot volume.
This prototype trains on ALL goalie-games (no odds required), roughly 2.3x
the odds-matched sample used by the classifier.

This script does not iterate toward a positive result; it iterates only on
correctness. A clean negative is a successful experiment.

ARCHITECTURE (pre-registered, kept simple and inspectable):
  1. Shots model: XGBoost regressor, objective=count:poisson, for E[shots].
     Overdispersion: negative-binomial dispersion fit on TRAIN residuals by
     method of moments (variance vs mean); falls back to Poisson if TRAIN
     variance <= TRAIN mean.
  2. Save-rate model: XGBoost regressor, objective=binary:logistic, trained
     on the per-game save fraction with sample_weight=shots_against
     (equivalent to a per-shot Bernoulli likelihood). Output q = P(save).
  3. Saves distribution: P(saves=s) = sum_n P(shots=n | NB or Poisson) *
     Binomial(s; n, q), n truncated at cap=70. Verified to sum to >0.999 for
     every priced row.
  4. P(over line) = P(saves > line); integer lines report P(over)/P(push)/
     P(under).

DATA: data/processed/clean_training_data.parquet (10,496 goalie-games, no
betting lines needed for training). Features are pre-game rolling/context
columns only -- current-game outcome columns, identifiers, and anything
requiring betting_line are excluded (this parquet has no betting_line column
at all, so the exclusion is enforced structurally as well as by name).

SPLIT: by DATE, identical boundaries to scripts/experiment_market_anchor.py:
  train = game_date <  2025-10-16
  val   = 2025-10-16 to 2025-12-03 inclusive
  test  = game_date >= 2025-12-04
Applied identically to clean_training_data.parquet (submodel training/
intrinsic quality) and multibook_classification_training_data.parquet
(betting evaluation, joined by (game_id, goalie_id)).

SELECTION METHODOLOGY:
  Submodels: each has its own pre-registered grid of up to 6 configs,
  selected on VAL ONLY by the submodel's own loss (shots: MAE, tie-broken
  informationally by Poisson deviance; save-rate: weighted log-loss).
  Distribution-level choices (dispersion method, truncation cap=70) are
  fixed in advance, not tuned.
  Betting: EV thresholds {0.05, 0.10, 0.12, 0.15} evaluated on VAL ONLY
  (the priced probabilities are fixed once the two submodels are chosen --
  there is no separate hyperparameter retrain per threshold). Filter to
  15-35% val bet rate, rank by val ROI (fallback 10-40%, then all with
  disclosure). Exactly ONE test-fold touch of the winning threshold.

Artifacts (models/trained/experiment_distributional_{timestamp}/):
  run_log.txt        -- full stdout, tee'd
  metadata.json       -- fold boundaries, feature lists, submodel val
                          selections, dispersion parameter, intrinsic
                          quality metrics, join coverage, val betting sweep,
                          the winner, the single test touch with CIs,
                          head-to-head table
  shots_model.json    -- trained XGBoost shots submodel (native format)
  save_rate_model.json -- trained XGBoost save-rate submodel (native format)

Do NOT modify: data/betting.db, models/trained/tuned_v1_20260201_155204/,
models/trained/tuned_v2_clean_20260707_212023/, src/betting/, docs/. This
script only reads the two parquets and writes its own artifact directory.

Usage:
    python scripts/experiment_distributional.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import gammaln
from sklearn.metrics import roc_auc_score, brier_score_loss

from betting.odds_utils import calculate_ev, calculate_payout

DATA_PATH_CLEAN = Path('data/processed/clean_training_data.parquet')
DATA_PATH_MULTIBOOK = Path('data/processed/multibook_classification_training_data.parquet')
OUTPUT_ROOT = Path('models/trained')

CAP = 70                 # shots/saves pmf truncation cap
EV_THRESHOLDS = [0.05, 0.10, 0.12, 0.15]
NAIVE_BASELINE_COL = 'shots_against_rolling_5'  # goalie's own rolling-5 shots-against average

# Reference numbers from the pre-registered protocol, cited verbatim from
# models/trained/experiment_market_anchor_20260708_184452/metadata.json.
# Do NOT re-run that experiment; these are final and independently verified.
HEAD_TO_HEAD_REFERENCE = {
    'classifier_control_A': {
        'description': '114-feature classifier, standard EV betting rule',
        'test_roi_pct': -5.97, 'test_bets': 843,
        'test_auc_row': 0.5067, 'test_auc_night': 0.5019,
        'test_brier': 0.27668,
        'source': 'models/trained/experiment_market_anchor_20260708_184452/metadata.json (results.A.test)',
    },
    'market_anchored_B': {
        'description': '114 features + per-row market features + book one-hots',
        'test_roi_pct': -3.10, 'test_bets': 516,
        'test_auc_row': 0.5243, 'test_auc_night': 0.5214,
        'test_brier': 0.26368,
        'source': 'models/trained/experiment_market_anchor_20260708_184452/metadata.json (results.B.test)',
    },
    'market_baseline_D': {
        'description': 'no-model market probability (fair_prob_over), zero bets by EV construction',
        'test_roi_pct': None, 'test_bets': 0,
        'test_auc_row': 0.5218, 'test_auc_night': 0.5170,
        'test_brier': 0.24961,
        'source': 'models/trained/experiment_market_anchor_20260708_184452/metadata.json (baseline_D.test)',
    },
}

# ============================================================
# Pre-registered hyperparameter grids -- max 6 configs each.
# ============================================================

BASE_SHOTS = dict(max_depth=3, learning_rate=0.05, min_child_weight=10,
                   subsample=0.8, colsample_bytree=0.8, n_estimators=400, reg_lambda=1.0)
SHOTS_CONFIGS = [
    ('base', dict(BASE_SHOTS)),
    ('depth2', {**BASE_SHOTS, 'max_depth': 2}),
    ('depth4_mcw20', {**BASE_SHOTS, 'max_depth': 4, 'min_child_weight': 20}),
    ('more_trees_lowlr', {**BASE_SHOTS, 'n_estimators': 800, 'learning_rate': 0.03}),
    ('shallow_highreg', {**BASE_SHOTS, 'max_depth': 2, 'min_child_weight': 30, 'reg_lambda': 5.0}),
    ('deep_reg', {**BASE_SHOTS, 'max_depth': 5, 'min_child_weight': 30, 'n_estimators': 300}),
]
assert len(SHOTS_CONFIGS) <= 6

BASE_RATE = dict(max_depth=3, learning_rate=0.05, min_child_weight=10,
                  subsample=0.8, colsample_bytree=0.8, n_estimators=400, reg_lambda=1.0)
SAVE_RATE_CONFIGS = [
    ('base', dict(BASE_RATE)),
    ('depth2', {**BASE_RATE, 'max_depth': 2}),
    ('depth4_mcw20', {**BASE_RATE, 'max_depth': 4, 'min_child_weight': 20}),
    ('more_trees_lowlr', {**BASE_RATE, 'n_estimators': 800, 'learning_rate': 0.03}),
    ('shallow_highreg', {**BASE_RATE, 'max_depth': 2, 'min_child_weight': 30, 'reg_lambda': 5.0}),
    ('deep_reg', {**BASE_RATE, 'max_depth': 5, 'min_child_weight': 30, 'n_estimators': 300}),
]
assert len(SAVE_RATE_CONFIGS) <= 6


# ============================================================
# Section 1: feature engineering (pre-game only, no betting_line dependence)
# ============================================================

IDENTIFIER_COLS = ['game_id', 'game_date', 'season', 'goalie_id', 'goalie_name',
                    'team_abbrev', 'opponent_team']

# Current-game outcome columns -- exactly the set named in the protocol, plus
# the current-game special-teams stat columns.
CURRENT_GAME_OUTCOME_COLS = [
    'saves', 'shots_against', 'goals_against', 'save_percentage', 'toi', 'decision',
    'team_goals', 'team_shots', 'opp_goals', 'opp_shots',
    'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
    'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
    'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
]

FORBIDDEN_FEATURE_COLS = set(IDENTIFIER_COLS) | set(CURRENT_GAME_OUTCOME_COLS) | {'betting_line'}


def add_engineered_features_no_line(df):
    """Subset of the 18 engineered features from optimize_features.py /
    experiment_market_anchor.py's add_all_engineered_features -- with the 3
    features that require betting_line removed (volatility_vs_line_5,
    volatility_vs_line_10, line_vs_opp_implied_saves). clean_training_data.parquet
    has no betting_line column, so this parquet cannot compute those 3 anyway;
    this function documents the omission explicitly rather than silently."""
    df = df.copy()

    for w in [3, 5, 10]:
        sr = f'saves_rolling_{w}'
        sar = f'shots_against_rolling_{w}'
        if sr in df.columns and sar in df.columns:
            df[f'save_efficiency_{w}'] = df[sr] / df[sar].clip(lower=1)

    for w in [5, 10]:
        es = f'even_strength_saves_rolling_{w}'
        sr = f'saves_rolling_{w}'
        if es in df.columns and sr in df.columns:
            df[f'es_saves_proportion_{w}'] = df[es] / df[sr].clip(lower=1)

    if 'opp_shots_rolling_5' in df.columns and 'team_shots_against_rolling_5' in df.columns:
        df['opp_vs_team_shots_5'] = df['opp_shots_rolling_5'] - df['team_shots_against_rolling_5']
    if 'opp_shots_rolling_10' in df.columns and 'team_shots_against_rolling_10' in df.columns:
        df['opp_vs_team_shots_10'] = df['opp_shots_rolling_10'] - df['team_shots_against_rolling_10']

    for w in [5, 10]:
        mean_col = f'saves_rolling_{w}'
        std_col = f'saves_rolling_std_{w}'
        if mean_col in df.columns and std_col in df.columns:
            df[f'saves_cv_{w}'] = df[std_col] / df[mean_col].clip(lower=1)
        # volatility_vs_line_{w} intentionally omitted -- requires betting_line

    for stat in ['saves', 'shots_against', 'goals_against']:
        short = f'{stat}_rolling_3'
        long = f'{stat}_rolling_10'
        if short in df.columns and long in df.columns:
            df[f'{stat}_momentum'] = df[short] - df[long]

    sp_short = 'save_percentage_rolling_3'
    sp_long = 'save_percentage_rolling_10'
    if sp_short in df.columns and sp_long in df.columns:
        df['save_pct_momentum'] = df[sp_short] - df[sp_long]

    if 'opp_shots_rolling_5' in df.columns and 'shots_against_rolling_5' in df.columns:
        df['expected_workload_diff'] = df['opp_shots_rolling_5'] - df['shots_against_rolling_5']

    # line_vs_opp_implied_saves intentionally omitted -- requires betting_line

    if 'goalie_days_rest' in df.columns and 'saves_rolling_5' in df.columns:
        df['rest_x_performance'] = df['goalie_days_rest'].clip(upper=7) * df['saves_rolling_5']

    return df


def build_modeling_frame(log):
    df = pd.read_parquet(DATA_PATH_CLEAN)
    log(f"Raw clean_training_data.parquet: {len(df)} rows, {len(df.columns)} columns.")
    df = df.sort_values('game_date').reset_index(drop=True)

    assert 'betting_line' not in df.columns, \
        "clean_training_data.parquet unexpectedly has a betting_line column -- protocol assumes it does not."

    base_feature_cols = [c for c in df.columns if c not in IDENTIFIER_COLS and c not in CURRENT_GAME_OUTCOME_COLS]
    log(f"Base pre-game/context features (identifiers + current-game outcomes excluded): {len(base_feature_cols)}")

    df = add_engineered_features_no_line(df)
    engineered_cols = [c for c in df.columns
                        if c not in base_feature_cols and c not in IDENTIFIER_COLS and c not in CURRENT_GAME_OUTCOME_COLS]
    log(f"Engineered features (line-dependent ones omitted): {len(engineered_cols)} -> {engineered_cols}")

    feature_cols = base_feature_cols + engineered_cols
    log(f"Final feature set: {len(feature_cols)} features.")

    # --- Hard assert: no identifier/post-game/betting_line columns leaked in ---
    leaked = set(feature_cols) & FORBIDDEN_FEATURE_COLS
    assert not leaked, f"Feature set leaks forbidden columns: {leaked}"
    assert 'betting_line' not in df.columns

    # --- Hard assert: no infinities/NaNs in the feature matrix ---
    mat = df[feature_cols].values.astype(np.float64)
    n_inf = np.isinf(mat).sum()
    n_nan = np.isnan(mat).sum()
    assert n_inf == 0, f"Feature matrix contains {n_inf} infinite values."
    assert n_nan == 0, f"Feature matrix contains {n_nan} NaN values."
    log("No infinities or NaNs found in the feature matrix.")

    # --- Drop the 1 shots_against==0 row from shots-model target sanity note ---
    n_zero_shots = int((df['shots_against'] == 0).sum())
    if n_zero_shots:
        log(f"Note: {n_zero_shots} row(s) with shots_against==0 present (valid Poisson/NB target=0; "
            f"excluded from save-rate model training only, since saves/shots_against is undefined there).")

    return df, feature_cols, base_feature_cols, engineered_cols


def split_by_date(df, log, label):
    train_mask = df['game_date'] < '2025-10-16'
    val_mask = (df['game_date'] >= '2025-10-16') & (df['game_date'] <= '2025-12-03')
    test_mask = df['game_date'] >= '2025-12-04'

    assert (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int) <= 1).all(), \
        f"[{label}] A row satisfies more than one fold's date condition."
    assert (train_mask | val_mask | test_mask).all(), \
        f"[{label}] A row satisfies none of the fold date conditions."
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == len(df)

    train_dates = set(df.loc[train_mask, 'game_date'])
    val_dates = set(df.loc[val_mask, 'game_date'])
    test_dates = set(df.loc[test_mask, 'game_date'])
    assert train_dates.isdisjoint(val_dates), f"[{label}] Train/val date overlap."
    assert train_dates.isdisjoint(test_dates), f"[{label}] Train/test date overlap."
    assert val_dates.isdisjoint(test_dates), f"[{label}] Val/test date overlap."

    train_idx = np.where(train_mask.values)[0]
    val_idx = np.where(val_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]

    log(f"\n[{label}] Fold boundaries (date-based split):")
    log(f"  Train: {df['game_date'].iloc[train_idx].min().date()} to "
        f"{df['game_date'].iloc[train_idx].max().date()}  (n={len(train_idx)})")
    log(f"  Val:   {df['game_date'].iloc[val_idx].min().date()} to "
        f"{df['game_date'].iloc[val_idx].max().date()}  (n={len(val_idx)})")
    log(f"  Test:  {df['game_date'].iloc[test_idx].min().date()} to "
        f"{df['game_date'].iloc[test_idx].max().date()}  (n={len(test_idx)})")
    log(f"  [{label}] Fold date-disjointness verified.")

    return train_idx, val_idx, test_idx


# ============================================================
# Section 2: submodel loss functions
# ============================================================

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def poisson_deviance(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)
    term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    dev = 2 * (term - (y_true - y_pred))
    return float(np.mean(dev))


def weighted_logloss(y_true, y_pred, weights):
    eps = 1e-7
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    weights = np.asarray(weights, dtype=float)
    ll = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return float(np.average(ll, weights=weights))


def top_features(model, k=5):
    booster = model.get_booster()
    gain = booster.get_score(importance_type='gain')
    top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{'feature': f, 'gain': float(g)} for f, g in top]


# ============================================================
# Section 3: submodel training
# ============================================================

def train_shots_model(df, train_idx, val_idx, feature_cols, log):
    X_train = df[feature_cols].iloc[train_idx].astype(np.float32)
    y_train = df['shots_against'].values[train_idx].astype(float)
    X_val = df[feature_cols].iloc[val_idx].astype(np.float32)
    y_val = df['shots_against'].values[val_idx].astype(float)

    log("\n--- Shots model (count:poisson) -- config search, selected on VAL MAE ---")
    log(f"{'config':<20} {'val_mae':>9} {'val_poisson_dev':>16}")
    evaluations = []
    models = {}
    for name, cfg in SHOTS_CONFIGS:
        params = dict(objective='count:poisson', random_state=42, n_jobs=-1, verbosity=0, **cfg)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred_val = np.clip(model.predict(X_val), 1e-3, None)
        val_mae = mae(y_val, pred_val)
        val_dev = poisson_deviance(y_val, pred_val)
        evaluations.append({'config': name, 'hyperparams': cfg, 'val_mae': val_mae, 'val_poisson_deviance': val_dev})
        models[name] = model
        log(f"{name:<20} {val_mae:>9.4f} {val_dev:>16.4f}")

    winner_entry = min(evaluations, key=lambda e: e['val_mae'])
    winner_model = models[winner_entry['config']]
    log(f"\nShots model winner (VAL MAE): {winner_entry['config']}  "
        f"val_mae={winner_entry['val_mae']:.4f}  val_poisson_deviance={winner_entry['val_poisson_deviance']:.4f}")
    return winner_model, winner_entry, evaluations


def train_save_rate_model(df, train_idx, val_idx, feature_cols, log):
    shots = df['shots_against'].values.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        save_rate = df['saves'].values.astype(float) / shots
    valid = shots > 0

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(len(df), dtype=bool)
    val_mask[val_idx] = True
    train_use = train_mask & valid
    val_use = val_mask & valid

    n_dropped_train = int(train_mask.sum() - train_use.sum())
    n_dropped_val = int(val_mask.sum() - val_use.sum())
    if n_dropped_train or n_dropped_val:
        log(f"\nSave-rate model: dropped {n_dropped_train} TRAIN / {n_dropped_val} VAL row(s) "
            f"with shots_against==0 (save rate undefined).")

    X_train = df[feature_cols].loc[train_use].astype(np.float32)
    y_train = save_rate[train_use]
    w_train = shots[train_use]
    X_val = df[feature_cols].loc[val_use].astype(np.float32)
    y_val = save_rate[val_use]
    w_val = shots[val_use]

    log("\n--- Save-rate model (binary:logistic, sample_weight=shots_against) -- "
        "config search, selected on VAL weighted log-loss ---")
    log(f"{'config':<20} {'val_weighted_logloss':>21}")
    evaluations = []
    models = {}
    for name, cfg in SAVE_RATE_CONFIGS:
        params = dict(objective='binary:logistic', random_state=42, n_jobs=-1, verbosity=0, **cfg)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val], verbose=False)
        pred_val = np.clip(model.predict(X_val), 1e-6, 1 - 1e-6)
        val_ll = weighted_logloss(y_val, pred_val, w_val)
        evaluations.append({'config': name, 'hyperparams': cfg, 'val_weighted_logloss': val_ll})
        models[name] = model
        log(f"{name:<20} {val_ll:>21.5f}")

    winner_entry = min(evaluations, key=lambda e: e['val_weighted_logloss'])
    winner_model = models[winner_entry['config']]
    log(f"\nSave-rate model winner (VAL weighted log-loss): {winner_entry['config']}  "
        f"val_weighted_logloss={winner_entry['val_weighted_logloss']:.5f}")
    return winner_model, winner_entry, evaluations


def fit_dispersion(shots_model, df, train_idx, feature_cols, log):
    X_train = df[feature_cols].iloc[train_idx].astype(np.float32)
    y_train = df['shots_against'].values[train_idx].astype(float)
    mu_train = np.clip(shots_model.predict(X_train), 1e-3, None)
    resid = y_train - mu_train

    train_mean = float(np.mean(mu_train))
    train_var = float(np.mean(resid ** 2))
    log(f"\n--- Dispersion fit on TRAIN residuals (method of moments) ---")
    log(f"  mean(predicted mu) = {train_mean:.4f}")
    log(f"  mean(residual^2)   = {train_var:.4f}   (empirical Var(Y|mu) estimate)")

    if train_var <= train_mean:
        log("  TRAIN variance <= TRAIN mean: no overdispersion detected. Falling back to Poisson (alpha=0).")
        return 0.0, 'poisson_fallback', {'train_mean': train_mean, 'train_var': train_var}

    mean_mu2 = float(np.mean(mu_train ** 2))
    alpha = max((train_var - train_mean) / mean_mu2, 1e-6)
    log(f"  TRAIN variance > TRAIN mean: overdispersion detected.")
    log(f"  NB2 method-of-moments dispersion alpha = {alpha:.6f}  (Var = mean + alpha*mean^2)")
    return alpha, 'negative_binomial', {'train_mean': train_mean, 'train_var': train_var, 'mean_mu2': mean_mu2}


# ============================================================
# Section 4: saves distribution (shots NB/Poisson mixed with Binomial)
# ============================================================

class SavesDistribution:
    """Precomputes the (n, s) binomial-coefficient grid once; per-row pricing
    only needs to plug in mu, alpha, q."""

    def __init__(self, cap):
        self.cap = cap
        self.n_arr = np.arange(cap + 1).astype(np.float64)
        n_idx = self.n_arr.reshape(-1, 1)
        s_idx = self.n_arr.reshape(1, -1)
        diff = n_idx - s_idx
        self.valid = diff >= 0
        diff_safe = np.where(self.valid, diff, 0)
        self.diff_safe = diff_safe
        self.s_idx = s_idx
        # log multinomial-coefficient part of the binomial pmf, independent of q
        self.log_C = gammaln(n_idx + 1) - gammaln(s_idx + 1) - gammaln(diff_safe + 1)

    def shots_pmf(self, mu, alpha):
        mu = max(mu, 1e-6)
        n = self.n_arr
        if alpha <= 0:
            logpmf = n * np.log(mu) - mu - gammaln(n + 1)
        else:
            r = 1.0 / alpha
            p = r / (r + mu)
            logpmf = gammaln(n + r) - gammaln(r) - gammaln(n + 1) + r * np.log(p) + n * np.log(1 - p)
        return np.exp(logpmf)

    def saves_pmf(self, mu, alpha, q):
        shots_pmf = self.shots_pmf(mu, alpha)
        q = min(max(q, 1e-6), 1 - 1e-6)
        logB = self.log_C + self.s_idx * np.log(q) + self.diff_safe * np.log(1 - q)
        B = np.where(self.valid, np.exp(logB), 0.0)
        saves_pmf = shots_pmf @ B
        return saves_pmf, shots_pmf

    def price_line(self, saves_pmf, line):
        s = self.n_arr
        p_over = float(saves_pmf[s > line].sum())
        p_under = float(saves_pmf[s < line].sum())
        p_push = float(saves_pmf[s == line].sum())
        return p_over, p_under, p_push


def compute_distribution_predictions(df, idx, shots_model, rate_model, alpha, feature_cols, dist, log, label):
    X = df[feature_cols].iloc[idx].astype(np.float32)
    mu_arr = np.clip(shots_model.predict(X), 1e-3, None)
    q_arr = np.clip(rate_model.predict(X), 1e-6, 1 - 1e-6)

    n = len(idx)
    pmf_arr = np.zeros((n, dist.cap + 1))
    for i in range(n):
        saves_pmf, _ = dist.saves_pmf(float(mu_arr[i]), alpha, float(q_arr[i]))
        pmf_arr[i] = saves_pmf

    sums = pmf_arr.sum(axis=1)
    bad = np.where(sums < 0.999)[0]
    log(f"[{label}] pmf normalization check: n={n}, min_sum={sums.min():.6f}, max_sum={sums.max():.6f}, "
        f"rows below 0.999: {len(bad)}")
    assert len(bad) == 0, f"[{label}] {len(bad)} rows have pmf sum < 0.999 (cap={dist.cap} may be too small)."

    game_ids = df['game_id'].values[idx]
    goalie_ids = df['goalie_id'].values[idx]
    keys = list(zip(game_ids.tolist(), goalie_ids.tolist()))
    assert len(set(keys)) == len(keys), f"[{label}] duplicate (game_id, goalie_id) keys in distribution predictions."
    lookup = {k: i for i, k in enumerate(keys)}

    return {'mu': mu_arr, 'q': q_arr, 'pmf': pmf_arr, 'keys': keys, 'lookup': lookup, 'idx': idx}


# ============================================================
# Section 5: intrinsic distribution quality (VAL only)
# ============================================================

def intrinsic_quality_metrics(df, val_idx, dist_preds_val, dist, log):
    y_val_shots = df['shots_against'].values[val_idx].astype(float)
    mu_val = dist_preds_val['mu']
    naive_pred = df[NAIVE_BASELINE_COL].values[val_idx].astype(float)

    model_mae = mae(y_val_shots, mu_val)
    naive_mae = mae(y_val_shots, naive_pred)
    log(f"\n--- Intrinsic distribution quality on VAL ---")
    log(f"Shots MAE: model={model_mae:.4f}  naive({NAIVE_BASELINE_COL})={naive_mae:.4f}  "
        f"({'model beats naive' if model_mae < naive_mae else 'model does NOT beat naive'})")

    saves_actual = df['saves'].values[val_idx].astype(int)
    saves_actual_c = np.clip(saves_actual, 0, dist.cap)
    pmf_arr = dist_preds_val['pmf']
    n = len(val_idx)

    row_idx = np.arange(n)
    p_actual = np.clip(pmf_arr[row_idx, saves_actual_c], 1e-12, None)
    mean_pmf_logloss = float(np.mean(-np.log(p_actual)))
    log(f"Mean pmf log-loss at actual saves value: {mean_pmf_logloss:.4f}")

    cdf = np.cumsum(pmf_arr, axis=1)
    cov50 = 0
    cov80 = 0
    for i in range(n):
        row_cdf = cdf[i]
        lo50 = np.searchsorted(row_cdf, 0.25, side='left')
        hi50 = np.searchsorted(row_cdf, 0.75, side='left')
        lo80 = np.searchsorted(row_cdf, 0.10, side='left')
        hi80 = np.searchsorted(row_cdf, 0.90, side='left')
        if lo50 <= saves_actual_c[i] <= hi50:
            cov50 += 1
        if lo80 <= saves_actual_c[i] <= hi80:
            cov80 += 1
    cov50_pct = cov50 / n * 100
    cov80_pct = cov80 / n * 100
    log(f"Empirical coverage: central 50% interval hit {cov50_pct:.1f}% of the time (nominal 50%)")
    log(f"Empirical coverage: central 80% interval hit {cov80_pct:.1f}% of the time (nominal 80%)")
    if abs(cov50_pct - 50) >= 10:
        log(f"  WARNING: 50% interval coverage gap is {abs(cov50_pct - 50):.1f} points -- miscalibrated.")
    if abs(cov80_pct - 80) >= 10:
        log(f"  WARNING: 80% interval coverage gap is {abs(cov80_pct - 80):.1f} points -- miscalibrated.")

    rng = np.random.RandomState(123)
    pit_vals = np.zeros(n)
    for i in range(n):
        y = saves_actual_c[i]
        cdf_y = cdf[i, y]
        cdf_y_minus1 = cdf[i, y - 1] if y > 0 else 0.0
        v = rng.uniform()
        pit_vals[i] = cdf_y_minus1 + v * (cdf_y - cdf_y_minus1)
    hist, _ = np.histogram(pit_vals, bins=10, range=(0, 1))
    freqs = (hist / n).tolist()
    log(f"PIT histogram (10 bins, randomized for discreteness), frequencies: "
        f"{[round(f, 3) for f in freqs]}  (nominal 0.100 each)")
    log(f"PIT bin frequency min={min(freqs):.3f}  max={max(freqs):.3f}")

    return {
        'naive_baseline_col': NAIVE_BASELINE_COL,
        'shots_mae_model': model_mae,
        'shots_mae_naive': naive_mae,
        'mean_pmf_logloss_at_actual': mean_pmf_logloss,
        'coverage_50pct_nominal_actual': cov50_pct,
        'coverage_80pct_nominal_actual': cov80_pct,
        'pit_histogram_10bins': freqs,
        'pit_bin_freq_min': min(freqs),
        'pit_bin_freq_max': max(freqs),
        'n_val_rows': n,
    }


# ============================================================
# Section 6: betting harness (copied patterns from
# experiment_market_anchor.py -- grade_bets, bootstrap CIs, dedup, AUC/Brier)
# ============================================================

CONTAMINATED_MARKET_COLS = [
    'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
    'market_vig', 'impl_prob_over', 'impl_prob_under',
    'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
    'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low',
]


def build_betting_frame(log):
    df = pd.read_parquet(DATA_PATH_MULTIBOOK)
    log(f"\nRaw multibook_classification_training_data.parquet: {len(df)} rows, {len(df.columns)} columns.")
    df = df.drop(columns=[c for c in CONTAMINATED_MARKET_COLS if c in df.columns], errors='ignore')
    df = df[df['odds_over_american'].notna() & df['odds_under_american'].notna()].copy()
    df = df.sort_values('game_date').reset_index(drop=True)
    log(f"After both-side-odds filter + chronological sort: {len(df)} rows.")
    assert len(df) == 13192, f"Expected 13192 rows (matches experiment_market_anchor.py's modeling frame), got {len(df)}."
    return df


def decide_bet(p_over, p_under, odds_over, odds_under, ev_threshold):
    ev_over = calculate_ev(p_over, odds_over)
    ev_under = calculate_ev(p_under, odds_under)
    if ev_over >= ev_threshold and ev_over > ev_under:
        return 'OVER', ev_over
    elif ev_under >= ev_threshold:
        return 'UNDER', ev_under
    return None, None


def grade_bets(p_over_arr, p_under_arr, saves_actual, lines, odds_over, odds_under,
               game_id, goalie_id, ev_threshold, matched_mask, log=None, label=None):
    results = []
    n_push = 0
    for i in range(len(p_over_arr)):
        if not matched_mask[i]:
            continue
        bet, ev = decide_bet(p_over_arr[i], p_under_arr[i], odds_over[i], odds_under[i], ev_threshold)
        if bet is None:
            continue
        actual = saves_actual[i]
        line = lines[i]
        if actual == line:
            n_push += 1
            continue  # void bet -- excluded from grading (stake not at risk)
        if bet == 'OVER':
            won = actual > line
            profit = calculate_payout(1.0, odds_over[i], won)
        else:
            won = actual < line
            profit = calculate_payout(1.0, odds_under[i], won)
        results.append({
            'local_idx': int(i), 'bet': bet, 'profit': float(profit), 'won': bool(won),
            'ev': float(ev), 'cluster_id': f"{int(game_id[i])}_{int(goalie_id[i])}",
        })
    if log is not None and n_push:
        log(f"[{label}] {n_push} push bet(s) excluded from grading (actual saves == line).")
    return results


def summarize_bets(results, fold_size):
    n_bets = len(results)
    if n_bets == 0:
        return {'bets': 0, 'bet_rate': 0.0, 'hit_rate': 0.0, 'roi': 0.0, 'profit': 0.0}
    wins = sum(r['won'] for r in results)
    profit = sum(r['profit'] for r in results)
    return {
        'bets': n_bets, 'bet_rate': n_bets / fold_size * 100,
        'hit_rate': wins / n_bets * 100, 'roi': profit / n_bets * 100, 'profit': profit,
    }


def side_breakdown(results):
    breakdown = {}
    for side in ('OVER', 'UNDER'):
        side_bets = [r for r in results if r['bet'] == side]
        n_side = len(side_bets)
        if n_side == 0:
            breakdown[side] = {'bets': 0, 'hit_rate': 0.0, 'roi': 0.0, 'profit': 0.0}
            continue
        wins = sum(r['won'] for r in side_bets)
        profit = sum(r['profit'] for r in side_bets)
        breakdown[side] = {'bets': n_side, 'hit_rate': wins / n_side * 100,
                            'roi': profit / n_side * 100, 'profit': profit}
    return breakdown


def bootstrap_roi_ci(results, n_resamples=10000, seed=42, ci_pct=95.0):
    profits = np.asarray([r['profit'] for r in results], dtype=float)
    n_bets = len(profits)
    if n_bets == 0:
        return {'lower': 0.0, 'upper': 0.0, 'n_bets': 0}
    rng = np.random.RandomState(seed)
    resample_idx = rng.randint(0, n_bets, size=(n_resamples, n_bets))
    boot_rois = profits[resample_idx].mean(axis=1) * 100
    alpha = (100.0 - ci_pct) / 2.0
    return {'lower': float(np.percentile(boot_rois, alpha)),
            'upper': float(np.percentile(boot_rois, 100.0 - alpha)), 'n_bets': int(n_bets)}


def cluster_bootstrap_roi_ci(results, n_resamples=10000, seed=42, ci_pct=95.0):
    profits = np.asarray([r['profit'] for r in results], dtype=float)
    cluster_ids = np.asarray([r['cluster_id'] for r in results], dtype=object)
    if len(profits) == 0:
        return {'lower': 0.0, 'upper': 0.0, 'n_clusters': 0}
    unique_clusters, inv = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    cluster_sum = np.zeros(n_clusters)
    cluster_count = np.zeros(n_clusters)
    np.add.at(cluster_sum, inv, profits)
    np.add.at(cluster_count, inv, 1)
    rng = np.random.RandomState(seed)
    boot_rois = np.empty(n_resamples)
    for b in range(n_resamples):
        draw = rng.randint(0, n_clusters, size=n_clusters)
        counts = np.bincount(draw, minlength=n_clusters)
        total_profit = np.dot(counts, cluster_sum)
        total_bets = np.dot(counts, cluster_count)
        boot_rois[b] = (total_profit / total_bets) * 100 if total_bets > 0 else 0.0
    alpha = (100.0 - ci_pct) / 2.0
    return {'lower': float(np.percentile(boot_rois, alpha)),
            'upper': float(np.percentile(boot_rois, 100.0 - alpha)), 'n_clusters': int(n_clusters)}


def dedup_one_per_goalie_night(prob, y_true, game_id, goalie_id):
    d = pd.DataFrame({'prob': np.asarray(prob), 'y': np.asarray(y_true),
                       'game_id': np.asarray(game_id), 'goalie_id': np.asarray(goalie_id)})
    d = d.drop_duplicates(subset=['game_id', 'goalie_id'], keep='first')
    return d['prob'].values, d['y'].values


def auc_both(prob, y_true, game_id, goalie_id):
    row_auc = roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else float('nan')
    dedup_prob, dedup_y = dedup_one_per_goalie_night(prob, y_true, game_id, goalie_id)
    night_auc = roc_auc_score(dedup_y, dedup_prob) if len(set(dedup_y)) > 1 else float('nan')
    return {'row_level': float(row_auc), 'one_per_goalie_night': float(night_auc),
            'n_rows': int(len(prob)), 'n_goalie_nights': int(len(dedup_prob))}


def brier(prob, y_true):
    return float(brier_score_loss(y_true, prob))


def goalie_night_count(game_id, goalie_id):
    return len(set(zip(np.asarray(game_id).tolist(), np.asarray(goalie_id).tolist())))


def bet_goalie_night_count(results):
    return len(set(r['cluster_id'] for r in results))


def join_and_price(df_bet_fold, dist_preds, dist, log, label):
    game_ids = df_bet_fold['game_id'].values
    goalie_ids = df_bet_fold['goalie_id'].values
    lines = df_bet_fold['betting_line'].values.astype(float)
    n = len(df_bet_fold)
    p_over = np.full(n, np.nan)
    p_under = np.full(n, np.nan)
    p_push = np.full(n, np.nan)
    matched = np.zeros(n, dtype=bool)
    for i in range(n):
        key = (int(game_ids[i]), int(goalie_ids[i]))
        pi = dist_preds['lookup'].get(key)
        if pi is None:
            continue
        matched[i] = True
        pmf = dist_preds['pmf'][pi]
        po, pu, pp = dist.price_line(pmf, lines[i])
        p_over[i], p_under[i], p_push[i] = po, pu, pp
    coverage = matched.mean() * 100 if n else 0.0
    log(f"[{label}] join coverage: {int(matched.sum())}/{n} rows matched a distribution prediction ({coverage:.2f}%)")
    if coverage < 95.0:
        log(f"  WARNING: join coverage below 95% -- investigate before trusting downstream betting numbers.")
    return p_over, p_under, p_push, matched, coverage


def fold_wide_auc_brier(p_over, matched, saves, lines, game_id, goalie_id, log, label):
    m = matched
    y = (saves[m] > lines[m]).astype(int)
    prob = p_over[m]
    gid = game_id[m]
    gaid = goalie_id[m]
    auc = auc_both(prob, y, gid, gaid)
    b = brier(prob, y)
    log(f"[{label}] fold-wide AUC row-level={auc['row_level']:.4f} "
        f"one-per-goalie-night={auc['one_per_goalie_night']:.4f}  Brier={b:.5f}  (n={len(prob)})")
    return auc, b


def run_val_sweep(df_bet_val, p_over_val, p_under_val, matched_val, log):
    saves_val = df_bet_val['saves'].values.astype(float)
    lines_val = df_bet_val['betting_line'].values.astype(float)
    odds_over_val = df_bet_val['odds_over_american'].astype(float).values
    odds_under_val = df_bet_val['odds_under_american'].astype(float).values
    game_id_val = df_bet_val['game_id'].values
    goalie_id_val = df_bet_val['goalie_id'].values
    n_val = len(df_bet_val)

    log(f"\n--- VAL EV threshold sweep (probabilities fixed; only the threshold varies) ---")
    log(f"{'thresh':>7} {'bets':>6} {'bet_rate':>9} {'hit_rate':>9} {'roi':>9}")
    evaluations = []
    for thresh in EV_THRESHOLDS:
        results = grade_bets(p_over_val, p_under_val, saves_val, lines_val, odds_over_val, odds_under_val,
                              game_id_val, goalie_id_val, thresh, matched_val, log, 'VAL')
        summary = summarize_bets(results, n_val)
        evaluations.append({'threshold': thresh, 'summary': summary, 'results': results})
        log(f"{thresh:>7.2f} {summary['bets']:>6} {summary['bet_rate']:>8.1f}% "
            f"{summary['hit_rate']:>8.1f}% {summary['roi']:>+8.2f}%")

    in_range = [e for e in evaluations if 15 <= e['summary']['bet_rate'] <= 35]
    deviation = None
    if not in_range:
        in_range = [e for e in evaluations if 10 <= e['summary']['bet_rate'] <= 40]
        deviation = "No EV threshold landed in the pre-registered 15-35% val bet-rate band; widened to 10-40%."
        log(f"WARNING: {deviation}")
        if not in_range:
            in_range = evaluations
            deviation += " Even the widened 10-40% band was empty; fell back to ALL 4 thresholds."
            log(f"WARNING: {deviation}")

    winner = max(in_range, key=lambda e: e['summary']['roi'])
    log(f"\nVAL sweep winner: EV threshold={winner['threshold']:.2f}  "
        f"bets={winner['summary']['bets']}  bet_rate={winner['summary']['bet_rate']:.1f}%  "
        f"roi={winner['summary']['roi']:+.2f}%")
    return evaluations, winner, deviation


# ============================================================
# Section 7: line-sensitivity demonstration (VAL only, illustrative)
# ============================================================

def line_sensitivity_demo(df_bet_val, dist_preds, dist, log):
    median_line = float(df_bet_val['betting_line'].median())
    idx_closest = (df_bet_val['betting_line'] - median_line).abs().idxmin()
    row = df_bet_val.loc[idx_closest]
    key = (int(row['game_id']), int(row['goalie_id']))
    pi = dist_preds['lookup'].get(key)
    if pi is None:
        log("Line-sensitivity demo: no matched distribution prediction for the median-line goalie-night.")
        return None

    pmf = dist_preds['pmf'][pi]
    mu = float(dist_preds['mu'][pi])
    q = float(dist_preds['q'][pi])
    game_date = row['game_date']
    game_date_str = game_date.date().isoformat() if hasattr(game_date, 'date') else str(game_date)

    log(f"\n--- Line-sensitivity demonstration (VAL only, illustrative, no selection) ---")
    log(f"Goalie-night: goalie_id={int(row['goalie_id'])}, game_id={int(row['game_id'])}, "
        f"game_date={game_date_str}")
    log(f"VAL median betting_line={median_line:.2f}; this goalie-night's own quoted line={row['betting_line']:.1f}")
    log(f"Predicted mu(shots)={mu:.2f}, q(save rate)={q:.4f}")
    log(f"{'line':>6} {'P(over)':>9} {'P(under)':>9} {'P(push)':>9}")
    demo_rows = []
    line = 22.5
    while line <= 27.5 + 1e-9:
        po, pu, pp = dist.price_line(pmf, line)
        log(f"{line:>6.1f} {po:>9.4f} {pu:>9.4f} {pp:>9.4f}")
        demo_rows.append({'line': float(line), 'p_over': po, 'p_under': pu, 'p_push': pp})
        line += 1.0

    return {'goalie_id': int(row['goalie_id']), 'game_id': int(row['game_id']),
            'game_date': game_date_str, 'mu': mu, 'q': q,
            'val_median_line': median_line, 'own_line': float(row['betting_line']), 'rows': demo_rows}


# ============================================================
# Main
# ============================================================

def main():
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_ROOT / f'experiment_distributional_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'run_log.txt'
    log_lines = []

    def log(msg=''):
        print(msg)
        log_lines.append(str(msg))

    def flush_log():
        with open(log_path, 'w') as f:
            f.write('\n'.join(log_lines))

    log("=" * 80)
    log("DISTRIBUTIONAL SAVES MODEL PROTOTYPE (roadmap item 9)")
    log(f"Run timestamp: {datetime.now().isoformat()}")
    log("=" * 80)

    # --- 1. Build modeling frame (clean_training_data.parquet) ---
    df, feature_cols, base_feature_cols, engineered_cols = build_modeling_frame(log)
    train_idx, val_idx, test_idx = split_by_date(df, log, 'clean_training_data')

    # --- 2. Train submodels ---
    shots_model, shots_winner, shots_evals = train_shots_model(df, train_idx, val_idx, feature_cols, log)
    alpha, dispersion_method, dispersion_diag = fit_dispersion(shots_model, df, train_idx, feature_cols, log)
    rate_model, rate_winner, rate_evals = train_save_rate_model(df, train_idx, val_idx, feature_cols, log)

    shots_model_path = output_dir / 'shots_model.json'
    rate_model_path = output_dir / 'save_rate_model.json'
    shots_model.get_booster().save_model(str(shots_model_path))
    rate_model.get_booster().save_model(str(rate_model_path))
    log(f"\nSaved shots model to: {shots_model_path}")
    log(f"Saved save-rate model to: {rate_model_path}")

    shots_top5 = top_features(shots_model, 5)
    rate_top5 = top_features(rate_model, 5)
    log(f"\nShots model top-5 features by gain: {shots_top5}")
    log(f"Save-rate model top-5 features by gain: {rate_top5}")

    # --- 3. Distribution predictions on VAL + TEST (clean_training_data) ---
    dist = SavesDistribution(CAP)
    combined_idx = np.concatenate([val_idx, test_idx])
    dist_preds_all = compute_distribution_predictions(
        df, combined_idx, shots_model, rate_model, alpha, feature_cols, dist, log, 'VAL+TEST (clean_training_data)'
    )
    n_val = len(val_idx)
    dist_preds_val = {
        'mu': dist_preds_all['mu'][:n_val], 'q': dist_preds_all['q'][:n_val],
        'pmf': dist_preds_all['pmf'][:n_val],
    }

    # --- 4. Intrinsic quality on VAL only ---
    intrinsic = intrinsic_quality_metrics(df, val_idx, dist_preds_val, dist, log)

    # --- 5. Betting evaluation on the multibook odds frame ---
    df_bet = build_betting_frame(log)
    train_idx_bet, val_idx_bet, test_idx_bet = split_by_date(df_bet, log, 'multibook_classification_training_data')
    df_bet_val = df_bet.iloc[val_idx_bet].reset_index(drop=True)
    df_bet_test = df_bet.iloc[test_idx_bet].reset_index(drop=True)

    p_over_val, p_under_val, p_push_val, matched_val, cov_val = join_and_price(
        df_bet_val, dist_preds_all, dist, log, 'VAL betting frame')
    p_over_test, p_under_test, p_push_test, matched_test, cov_test = join_and_price(
        df_bet_test, dist_preds_all, dist, log, 'TEST betting frame')

    val_fold_auc, val_fold_brier = fold_wide_auc_brier(
        p_over_val, matched_val, df_bet_val['saves'].values, df_bet_val['betting_line'].values,
        df_bet_val['game_id'].values, df_bet_val['goalie_id'].values, log, 'VAL')

    # --- 6. VAL EV threshold sweep + selection ---
    val_evaluations, val_winner, selection_deviation = run_val_sweep(df_bet_val, p_over_val, p_under_val, matched_val, log)
    winner_threshold = val_winner['threshold']

    val_roi_ci = bootstrap_roi_ci(val_winner['results'])
    val_cluster_ci = cluster_bootstrap_roi_ci(val_winner['results'])
    val_side = side_breakdown(val_winner['results'])
    val_goalie_nights_total = goalie_night_count(df_bet_val['game_id'].values, df_bet_val['goalie_id'].values)
    val_goalie_nights_bet = bet_goalie_night_count(val_winner['results'])
    log(f"\nVAL winner ROI 95% CI (row bootstrap):     [{val_roi_ci['lower']:+.2f}%, {val_roi_ci['upper']:+.2f}%]")
    log(f"VAL winner ROI 95% CI (cluster bootstrap):  [{val_cluster_ci['lower']:+.2f}%, {val_cluster_ci['upper']:+.2f}%]  "
        f"(n_clusters={val_cluster_ci['n_clusters']})")
    log(f"VAL winner side breakdown: OVER {val_side['OVER']['bets']} bets ({val_side['OVER']['roi']:+.2f}%), "
        f"UNDER {val_side['UNDER']['bets']} bets ({val_side['UNDER']['roi']:+.2f}%)")
    log(f"VAL goalie-nights: {val_goalie_nights_total} total, {val_goalie_nights_bet} with a bet")

    # --- 7. SINGLE test-fold touch ---
    log(f"\n{'=' * 80}")
    log(f"SINGLE TEST TOUCH -- EV threshold={winner_threshold:.2f} (selected on VAL only)")
    log(f"{'=' * 80}")
    test_fold_auc, test_fold_brier = fold_wide_auc_brier(
        p_over_test, matched_test, df_bet_test['saves'].values, df_bet_test['betting_line'].values,
        df_bet_test['game_id'].values, df_bet_test['goalie_id'].values, log, 'TEST')

    saves_test = df_bet_test['saves'].values.astype(float)
    lines_test = df_bet_test['betting_line'].values.astype(float)
    odds_over_test = df_bet_test['odds_over_american'].astype(float).values
    odds_under_test = df_bet_test['odds_under_american'].astype(float).values
    game_id_test = df_bet_test['game_id'].values
    goalie_id_test = df_bet_test['goalie_id'].values
    n_test = len(df_bet_test)

    test_results = grade_bets(p_over_test, p_under_test, saves_test, lines_test, odds_over_test, odds_under_test,
                               game_id_test, goalie_id_test, winner_threshold, matched_test, log, 'TEST')
    test_summary = summarize_bets(test_results, n_test)
    test_roi_ci = bootstrap_roi_ci(test_results)
    test_cluster_ci = cluster_bootstrap_roi_ci(test_results)
    test_side = side_breakdown(test_results)
    test_goalie_nights_total = goalie_night_count(game_id_test, goalie_id_test)
    test_goalie_nights_bet = bet_goalie_night_count(test_results)

    log(f"\nTEST: {test_summary['bets']} bets, {test_summary['bet_rate']:.1f}% bet rate, "
        f"{test_summary['hit_rate']:.1f}% hit rate, {test_summary['roi']:+.2f}% ROI")
    log(f"TEST ROI 95% CI (row bootstrap):     [{test_roi_ci['lower']:+.2f}%, {test_roi_ci['upper']:+.2f}%]")
    log(f"TEST ROI 95% CI (cluster bootstrap):  [{test_cluster_ci['lower']:+.2f}%, {test_cluster_ci['upper']:+.2f}%]  "
        f"(n_clusters={test_cluster_ci['n_clusters']})")
    log(f"TEST side breakdown: OVER {test_side['OVER']['bets']} bets ({test_side['OVER']['roi']:+.2f}%), "
        f"UNDER {test_side['UNDER']['bets']} bets ({test_side['UNDER']['roi']:+.2f}%)")
    log(f"TEST goalie-nights: {test_goalie_nights_total} total, {test_goalie_nights_bet} with a bet")
    log(f"TEST AUC row-level={test_fold_auc['row_level']:.4f}  one-per-goalie-night={test_fold_auc['one_per_goalie_night']:.4f}")
    log(f"TEST Brier={test_fold_brier:.5f}")

    # --- 8. Line-sensitivity demonstration (VAL only) ---
    line_demo = line_sensitivity_demo(df_bet_val, dist_preds_all, dist, log)

    # --- 9. Head-to-head comparison table ---
    log(f"\n{'=' * 80}")
    log("HEAD-TO-HEAD COMPARISON (test fold, all single-touch numbers)")
    log(f"{'=' * 80}")
    log(f"{'model':<28} {'test_roi':>10} {'bets':>6} {'auc_row':>9} {'auc_night':>10} {'brier':>9}")
    dist_row = {
        'test_roi_pct': test_summary['roi'], 'test_bets': test_summary['bets'],
        'test_auc_row': test_fold_auc['row_level'], 'test_auc_night': test_fold_auc['one_per_goalie_night'],
        'test_brier': test_fold_brier,
    }
    log(f"{'distributional (this exp)':<28} {dist_row['test_roi_pct']:>+9.2f}% {dist_row['test_bets']:>6} "
        f"{dist_row['test_auc_row']:>9.4f} {dist_row['test_auc_night']:>10.4f} {dist_row['test_brier']:>9.5f}")
    for key, ref in HEAD_TO_HEAD_REFERENCE.items():
        roi_str = f"{ref['test_roi_pct']:+.2f}%" if ref['test_roi_pct'] is not None else "n/a"
        log(f"{key:<28} {roi_str:>10} {ref['test_bets']:>6} "
            f"{ref['test_auc_row']:>9.4f} {ref['test_auc_night']:>10.4f} {ref['test_brier']:>9.5f}")

    elapsed = time.time() - start_time
    log(f"\nWall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # --- Save metadata ---
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'wall_clock_seconds': elapsed,
        'data_paths': {'clean': str(DATA_PATH_CLEAN), 'multibook': str(DATA_PATH_MULTIBOOK)},
        'cap': CAP,
        'fold_boundaries_clean_training_data': {
            'train': {'start': str(df['game_date'].iloc[train_idx].min().date()),
                      'end': str(df['game_date'].iloc[train_idx].max().date()), 'rows': int(len(train_idx))},
            'val': {'start': str(df['game_date'].iloc[val_idx].min().date()),
                    'end': str(df['game_date'].iloc[val_idx].max().date()), 'rows': int(len(val_idx))},
            'test': {'start': str(df['game_date'].iloc[test_idx].min().date()),
                     'end': str(df['game_date'].iloc[test_idx].max().date()), 'rows': int(len(test_idx))},
        },
        'fold_boundaries_multibook': {
            'train': {'rows': int(len(train_idx_bet))},
            'val': {'start': str(df_bet['game_date'].iloc[val_idx_bet].min().date()),
                    'end': str(df_bet['game_date'].iloc[val_idx_bet].max().date()), 'rows': int(len(val_idx_bet))},
            'test': {'start': str(df_bet['game_date'].iloc[test_idx_bet].min().date()),
                     'end': str(df_bet['game_date'].iloc[test_idx_bet].max().date()), 'rows': int(len(test_idx_bet))},
        },
        'feature_cols': {
            'base_feature_cols': base_feature_cols, 'engineered_cols': engineered_cols,
            'all_feature_cols': feature_cols, 'n_features': len(feature_cols),
        },
        'shots_model': {
            'configs': [{'name': n, 'params': c} for n, c in SHOTS_CONFIGS],
            'val_evaluations': shots_evals, 'winner': shots_winner, 'top5_features': shots_top5,
            'model_path': str(shots_model_path),
        },
        'dispersion': {'alpha': alpha, 'method': dispersion_method, 'diagnostics': dispersion_diag},
        'save_rate_model': {
            'configs': [{'name': n, 'params': c} for n, c in SAVE_RATE_CONFIGS],
            'val_evaluations': rate_evals, 'winner': rate_winner, 'top5_features': rate_top5,
            'model_path': str(rate_model_path),
        },
        'intrinsic_quality_val': intrinsic,
        'join_coverage': {'val_pct': cov_val, 'test_pct': cov_test,
                           'val_n_rows': int(len(df_bet_val)), 'test_n_rows': int(len(df_bet_test))},
        'val_fold_wide_auc': val_fold_auc, 'val_fold_wide_brier': val_fold_brier,
        'ev_thresholds': EV_THRESHOLDS,
        'val_betting_sweep': [
            {'threshold': e['threshold'], 'summary': e['summary']} for e in val_evaluations
        ],
        'val_winner': {
            'threshold': winner_threshold, 'summary': val_winner['summary'],
            'roi_ci_row': val_roi_ci, 'roi_ci_cluster': val_cluster_ci,
            'side_breakdown': val_side,
            'goalie_nights_total': val_goalie_nights_total, 'goalie_nights_bet': val_goalie_nights_bet,
        },
        'selection_deviation': selection_deviation,
        'test_single_touch': {
            'threshold': winner_threshold, 'summary': test_summary,
            'roi_ci_row': test_roi_ci, 'roi_ci_cluster': test_cluster_ci,
            'side_breakdown': test_side, 'auc': test_fold_auc, 'brier': test_fold_brier,
            'goalie_nights_total': test_goalie_nights_total, 'goalie_nights_bet': test_goalie_nights_bet,
        },
        'line_sensitivity_demo_val': line_demo,
        'head_to_head': {
            'distributional_this_experiment': dist_row,
            **HEAD_TO_HEAD_REFERENCE,
        },
        'touch_count_audit': {
            'shots_model_val_evaluations': len(shots_evals),
            'save_rate_model_val_evaluations': len(rate_evals),
            'betting_val_evaluations': len(val_evaluations),
            'betting_test_evaluations': 1,
        },
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    log(f"\nSaved metadata to: {metadata_path}")

    flush_log()
    print(f"Saved run log to: {log_path}")

    log("\n" + "=" * 80)
    log("EXPERIMENT COMPLETE")
    log("=" * 80)
    flush_log()


if __name__ == '__main__':
    main()
