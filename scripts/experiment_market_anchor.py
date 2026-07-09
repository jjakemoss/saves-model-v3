"""
Roadmap item 7: market-anchored residual model experiment.

docs/OFFSEASON_OPTIMIZATION_PLAN.md sections 3.4, 3.9, 3.10, 3.11 (motivation)
and 3.3 (selection methodology bar this experiment must clear).

BUSINESS QUESTION: does giving the model the market's own information
(per-row no-vig implied probability, vig, book identity) produce an honest
chronological-holdout improvement over both (a) the market itself and (b)
the existing 114-feature model -- or does the model simply collapse to the
market? Either answer is valuable. This script does not iterate toward a
positive result; it iterates only on correctness. A clean negative is a
successful experiment.

FOUR FEATURE SETS, identical harness:
  A (control):        the standard 114 features.
  B (anchored):        114 + recomputed per-row market features + book one-hots.
  C (market-only):     recomputed per-row market features + book one-hots + betting_line.
  D (no-model market baseline): probability = fair_prob_over directly, no training.

MARKET FEATURES ARE RECOMPUTED FRESH, per-row, from that row's OWN odds only
(no cross-row aggregation, no averaging). The 12 precomputed market columns
already in the parquet are dropped first because they were built by an old
odds-averaging bug (see docs/HISTORICAL_DATA_ANALYSIS.md section 1) -- this
script never reads them.

SPLIT: by DATE, not row index (fixes the known fold-straddle problem where a
single calendar date's multibook rows could land in two different folds
under a row-index split):
  train = game_date <  2025-10-16
  val   = 2025-10-16 to 2025-12-03 inclusive
  test  = game_date >= 2025-12-04

SELECTION METHODOLOGY (matches scripts/tune_hyperparameters.py section 3.3):
  6 hyperparameter configs x 4 EV thresholds = 24 evaluations per feature
  set (A/B/C), ALL on validation only. Filter to 15-35% val bet rate, rank
  by val ROI (fallback: widen to 10-40% if nothing lands in range). Exactly
  ONE test-fold evaluation per feature set thereafter. Baseline D involves
  no training and no selection -- its test-fold betting-rule performance is
  reported at all four thresholds, unselected.

Artifacts (models/trained/experiment_market_anchor_{timestamp}/):
  run_log.txt              -- full stdout, tee'd
  metadata.json             -- fold boundaries, all evaluations, winners,
                                test results with CIs, anchoring diagnostics,
                                D baseline, protocol deviations

Do NOT modify: data/betting.db, models/trained/tuned_v1_20260201_155204/,
models/trained/tuned_v2_clean_20260707_212023/, src/betting/. This script
only reads the multibook parquet and writes its own artifact directory.

Usage:
    python scripts/experiment_market_anchor.py
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
from sklearn.metrics import roc_auc_score, brier_score_loss

from betting.odds_utils import calculate_ev, calculate_payout, american_to_implied_prob

DATA_PATH = Path('data/processed/multibook_classification_training_data.parquet')
OUTPUT_ROOT = Path('models/trained')

EV_THRESHOLDS = [0.05, 0.10, 0.12, 0.15]

# Pre-registered hyperparameter grid: v2 winner + exactly five variants.
BASE_CONFIG = dict(
    max_depth=3, learning_rate=0.05, min_child_weight=10, gamma=1.0,
    reg_alpha=10, reg_lambda=40, n_estimators=1200, subsample=0.7,
    colsample_bytree=0.8,
)
CONFIGS = [
    ('v2_winner', dict(BASE_CONFIG)),
    ('max_depth_4', {**BASE_CONFIG, 'max_depth': 4}),
    ('max_depth_6_mcw_30', {**BASE_CONFIG, 'max_depth': 6, 'min_child_weight': 30}),
    ('n_estimators_600', {**BASE_CONFIG, 'n_estimators': 600}),
    ('reg_lambda_10', {**BASE_CONFIG, 'reg_lambda': 10}),
    ('lr_002_est_2400', {**BASE_CONFIG, 'learning_rate': 0.02, 'n_estimators': 2400}),
]
assert len(CONFIGS) == 6
assert len(EV_THRESHOLDS) == 4

# ============================================================
# Section 1: feature engineering -- verbatim copy of the function in
# scripts/calibrate_model.py / scripts/tune_hyperparameters.py. Copied,
# not imported (tune_hyperparameters.py has no __main__ guard and running
# it as an import launches a full multi-minute tuning run).
# ============================================================

def add_all_engineered_features(df):
    """Reproduce the 18 engineered features from optimize_features.py
    (verbatim copy)."""
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
        if std_col in df.columns and 'betting_line' in df.columns:
            df[f'volatility_vs_line_{w}'] = df[std_col] / df['betting_line'].clip(lower=1)

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

    if 'opp_shots_rolling_5' in df.columns and 'opp_goals_rolling_5' in df.columns:
        opp_saves_implied = df['opp_shots_rolling_5'] - df['opp_goals_rolling_5']
        df['line_vs_opp_implied_saves'] = df['betting_line'] - opp_saves_implied

    if 'goalie_days_rest' in df.columns and 'saves_rolling_5' in df.columns:
        df['rest_x_performance'] = df['goalie_days_rest'].clip(upper=7) * df['saves_rolling_5']

    return df


# The 12 precomputed market-feature columns dropped before any recomputation
# -- contaminated by the old odds-averaging bug (docs/HISTORICAL_DATA_ANALYSIS.md
# section 1). Only 8 of these 12 exist in the current parquet; drop is a no-op
# for the rest.
CONTAMINATED_MARKET_COLS = [
    'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
    'market_vig', 'impl_prob_over', 'impl_prob_under',
    'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
    'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low',
]

# v2 exclusion list -- verbatim copy from scripts/tune_hyperparameters.py /
# scripts/calibrate_model.py. Produces exactly 114 feature columns (asserted
# below). This IS the "forbidden identifier / post-game-stat" column list
# referenced by the hard asserts in this script.
EXCLUDED_V2 = [
    'game_id', 'goalie_id', 'game_date', 'over_hit',
    'odds_over_american', 'odds_under_american',
    'odds_over_decimal', 'odds_under_decimal', 'num_books',
    'team_abbrev', 'opponent_team', 'toi', 'season',
    'saves', 'shots_against', 'goals_against', 'save_percentage',
    'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
    'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
    'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
    'team_goals', 'team_shots', 'opp_goals', 'opp_shots', 'line_margin',
    'book_key', 'decision', 'team_id', 'goalie_name',
    'saves_margin', 'over_line',
    '_game_date_str', '_lookup_key',
]

FORBIDDEN_IDENTIFIER_COLS = [
    'goalie_name', 'book_key', 'team_abbrev', 'opponent_team', 'season',
    'game_id', 'goalie_id', 'game_date', 'decision', 'toi',
    'saves', 'shots_against', 'goals_against', 'save_percentage',
    'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
    'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
    'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
    'team_goals', 'team_shots', 'opp_goals', 'opp_shots',
]
assert set(FORBIDDEN_IDENTIFIER_COLS).issubset(set(EXCLUDED_V2))

MARKET_FEATURE_COLS = ['impl_prob_over', 'impl_prob_under', 'market_vig', 'fair_prob_over']


# ============================================================
# Section 2: data pipeline
# ============================================================

def build_modeling_frame(log):
    """Reproduce the v2 modeling frame, then recompute market features
    fresh per-row from that row's own odds only, plus book one-hots.
    Returns (df, feature_cols_A, feature_cols_B, feature_cols_C, book_cols).
    """
    df_raw = pd.read_parquet(DATA_PATH)
    log(f"Raw parquet: {len(df_raw)} rows, {len(df_raw.columns)} columns.")

    df_raw = df_raw.drop(columns=[c for c in CONTAMINATED_MARKET_COLS if c in df_raw.columns],
                          errors='ignore')
    df_raw = df_raw[df_raw['odds_over_american'].notna() & df_raw['odds_under_american'].notna()].copy()
    df_raw = df_raw.sort_values('game_date').reset_index(drop=True)
    log(f"After both-side-odds filter + chronological sort: {len(df_raw)} rows.")

    df = add_all_engineered_features(df_raw)

    assert len(df) == 13192, f"Expected 13192 rows (v2 modeling frame), got {len(df)}."

    # --- Feature set A: standard 114 features ---
    feature_cols_A = [c for c in df.columns if c not in EXCLUDED_V2]
    assert len(feature_cols_A) == 114, f"Expected 114 features for set A, got {len(feature_cols_A)}."
    log(f"Feature set A (control): {len(feature_cols_A)} features.")

    # --- Recompute market features fresh, per-row, from own odds only ---
    odds_over = df['odds_over_american'].astype(float).values
    odds_under = df['odds_under_american'].astype(float).values
    impl_over = np.array([american_to_implied_prob(o) for o in odds_over])
    impl_under = np.array([american_to_implied_prob(o) for o in odds_under])
    df['impl_prob_over'] = impl_over
    df['impl_prob_under'] = impl_under
    df['market_vig'] = impl_over + impl_under - 1.0
    df['fair_prob_over'] = impl_over / (impl_over + impl_under)

    # Spot-check: market features derived only from same-row odds (no
    # cross-row aggregation). Re-derive for a fixed sample and compare.
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(df), size=min(200, len(df)), replace=False)
    for i in sample_idx:
        oo = float(df['odds_over_american'].iloc[i])
        ou = float(df['odds_under_american'].iloc[i])
        io = american_to_implied_prob(oo)
        iu = american_to_implied_prob(ou)
        expected_vig = io + iu - 1.0
        expected_fair = io / (io + iu)
        assert abs(df['impl_prob_over'].iloc[i] - io) < 1e-12
        assert abs(df['impl_prob_under'].iloc[i] - iu) < 1e-12
        assert abs(df['market_vig'].iloc[i] - expected_vig) < 1e-12
        assert abs(df['fair_prob_over'].iloc[i] - expected_fair) < 1e-12
    log(f"Market feature per-row recomputation spot-check passed on {len(sample_idx)} rows.")

    # --- Book one-hots ---
    book_dummies = pd.get_dummies(df['book_key'], prefix='book').astype(np.float32)
    book_cols = list(book_dummies.columns)
    log(f"Book identities ({len(book_cols)}): {book_cols}")
    df = pd.concat([df, book_dummies], axis=1)

    feature_cols_B = feature_cols_A + MARKET_FEATURE_COLS + book_cols
    feature_cols_C = MARKET_FEATURE_COLS + book_cols + ['betting_line']

    log(f"Feature set B (anchored): {len(feature_cols_B)} features "
        f"(114 + {len(MARKET_FEATURE_COLS)} market + {len(book_cols)} book).")
    log(f"Feature set C (market-only): {len(feature_cols_C)} features "
        f"({len(MARKET_FEATURE_COLS)} market + {len(book_cols)} book + betting_line).")

    # --- Hard asserts: no forbidden identifier / post-game-stat columns ---
    for name, cols in [('A', feature_cols_A), ('B', feature_cols_B), ('C', feature_cols_C)]:
        leaked = set(cols) & set(FORBIDDEN_IDENTIFIER_COLS)
        assert not leaked, f"Feature set {name} leaks forbidden columns: {leaked}"
        assert 'book_key' not in cols, f"Feature set {name} contains raw book_key string column."

    # --- Hard assert: no infinities in any feature matrix ---
    for name, cols in [('A', feature_cols_A), ('B', feature_cols_B), ('C', feature_cols_C)]:
        mat = df[cols].values.astype(np.float64)
        n_inf = np.isinf(mat).sum()
        assert n_inf == 0, f"Feature set {name} contains {n_inf} infinite values."
    log("No infinities found in any feature matrix (A/B/C).")

    return df, feature_cols_A, feature_cols_B, feature_cols_C, book_cols


def split_by_date(df, log):
    """Date-based chronological split (not row-index based -- fixes the
    known fold-straddle problem)."""
    train_mask = df['game_date'] < '2025-10-16'
    val_mask = (df['game_date'] >= '2025-10-16') & (df['game_date'] <= '2025-12-03')
    test_mask = df['game_date'] >= '2025-12-04'

    assert (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int) <= 1).all(), \
        "A row satisfies more than one fold's date condition."
    assert (train_mask | val_mask | test_mask).all(), \
        "A row satisfies none of the fold date conditions."
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == len(df)

    train_dates = set(df.loc[train_mask, 'game_date'])
    val_dates = set(df.loc[val_mask, 'game_date'])
    test_dates = set(df.loc[test_mask, 'game_date'])
    assert train_dates.isdisjoint(val_dates), "Train/val date overlap."
    assert train_dates.isdisjoint(test_dates), "Train/test date overlap."
    assert val_dates.isdisjoint(test_dates), "Val/test date overlap."

    train_idx = np.where(train_mask.values)[0]
    val_idx = np.where(val_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]

    log("\nFold boundaries (date-based split):")
    log(f"  Train: {df['game_date'].iloc[train_idx].min().date()} to "
        f"{df['game_date'].iloc[train_idx].max().date()}  (n={len(train_idx)})")
    log(f"  Val:   {df['game_date'].iloc[val_idx].min().date()} to "
        f"{df['game_date'].iloc[val_idx].max().date()}  (n={len(val_idx)})")
    log(f"  Test:  {df['game_date'].iloc[test_idx].min().date()} to "
        f"{df['game_date'].iloc[test_idx].max().date()}  (n={len(test_idx)})")
    log("  Fold date-disjointness verified.")

    return train_idx, val_idx, test_idx


# ============================================================
# Section 3: betting-sim harness (replicates
# ClassifierTrainer.evaluate_profitability's decision logic exactly via the
# same calculate_ev/calculate_payout functions, plus per-bet cluster ids
# for cluster bootstrap CIs which evaluate_profitability does not return).
# ============================================================

def decide_bet(p_over, p_under, odds_over, odds_under, ev_threshold):
    ev_over = calculate_ev(p_over, odds_over)
    ev_under = calculate_ev(p_under, odds_under)
    if ev_over >= ev_threshold and ev_over > ev_under:
        return 'OVER', ev_over
    elif ev_under >= ev_threshold:
        return 'UNDER', ev_under
    return None, None


def grade_bets(prob_over, y_true, odds_over, odds_under, game_id, goalie_id, ev_threshold):
    """All arguments are aligned local arrays (same fold, same order).
    Returns a list of per-bet dicts."""
    results = []
    for i in range(len(prob_over)):
        p_over = float(prob_over[i])
        p_under = 1 - p_over
        bet, ev = decide_bet(p_over, p_under, odds_over[i], odds_under[i], ev_threshold)
        if bet is None:
            continue
        actual_over = y_true[i]
        if bet == 'OVER':
            won = (actual_over == 1)
            profit = calculate_payout(1.0, odds_over[i], won)
        else:
            won = (actual_over == 0)
            profit = calculate_payout(1.0, odds_under[i], won)
        results.append({
            'local_idx': int(i),
            'bet': bet,
            'profit': float(profit),
            'won': bool(won),
            'ev': float(ev),
            'cluster_id': f"{int(game_id[i])}_{int(goalie_id[i])}",
        })
    return results


def summarize_bets(results, fold_size):
    n_bets = len(results)
    if n_bets == 0:
        return {'bets': 0, 'bet_rate': 0.0, 'hit_rate': 0.0, 'roi': 0.0, 'profit': 0.0}
    wins = sum(r['won'] for r in results)
    profit = sum(r['profit'] for r in results)
    return {
        'bets': n_bets,
        'bet_rate': n_bets / fold_size * 100,
        'hit_rate': wins / n_bets * 100,
        'roi': profit / n_bets * 100,
        'profit': profit,
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
        breakdown[side] = {
            'bets': n_side,
            'hit_rate': wins / n_side * 100,
            'roi': profit / n_side * 100,
            'profit': profit,
        }
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
    return {
        'lower': float(np.percentile(boot_rois, alpha)),
        'upper': float(np.percentile(boot_rois, 100.0 - alpha)),
        'n_bets': int(n_bets),
    }


def cluster_bootstrap_roi_ci(results, n_resamples=10000, seed=42, ci_pct=95.0):
    """Cluster (goalie-night) bootstrap: resamples whole (game_id, goalie_id)
    clusters with replacement, since multibook rows on the same goalie-night
    are correlated."""
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
    return {
        'lower': float(np.percentile(boot_rois, alpha)),
        'upper': float(np.percentile(boot_rois, 100.0 - alpha)),
        'n_clusters': int(n_clusters),
    }


def dedup_one_per_goalie_night(prob, y_true, game_id, goalie_id):
    """First occurrence per (game_id, goalie_id) in the given (date-sorted)
    row order. Deterministic; documented method for the one-row-per-goalie-
    night AUC diagnostic."""
    d = pd.DataFrame({
        'prob': np.asarray(prob), 'y': np.asarray(y_true),
        'game_id': np.asarray(game_id), 'goalie_id': np.asarray(goalie_id),
    })
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


def bet_goalie_night_count(results, cluster_id_lookup=None):
    return len(set(r['cluster_id'] for r in results))


# ============================================================
# Section 4: per-feature-set harness
# ============================================================

def run_feature_set(name, feature_cols, df, train_idx, val_idx, test_idx, log):
    log("\n" + "=" * 80)
    log(f"FEATURE SET {name}: {len(feature_cols)} features")
    log("=" * 80)

    # Fit on a DataFrame (not .values) so the saved booster keeps real
    # feature names -- required for the top-20 gain importance diagnostic
    # to report actual feature names instead of generic f0/f1/... indices.
    X_df = df[feature_cols].astype(np.float32)
    y = df['over_hit'].values.astype(int)
    odds_over = df['odds_over_american'].astype(float).values
    odds_under = df['odds_under_american'].astype(float).values
    game_id = df['game_id'].values
    goalie_id = df['goalie_id'].values

    X_train, y_train = X_df.iloc[train_idx], y[train_idx]
    X_val, y_val = X_df.iloc[val_idx], y[val_idx]
    X_test, y_test = X_df.iloc[test_idx], y[test_idx]
    odds_over_val, odds_under_val = odds_over[val_idx], odds_under[val_idx]
    odds_over_test, odds_under_test = odds_over[test_idx], odds_under[test_idx]
    gid_val, gaid_val = game_id[val_idx], goalie_id[val_idx]
    gid_test, gaid_test = game_id[test_idx], goalie_id[test_idx]

    val_evaluations = []
    models_by_config = {}
    probs_val_by_config = {}

    n_val = len(val_idx)
    log(f"\n{'config':<20} {'thresh':>7} {'bets':>6} {'bet_rate':>9} {'hit_rate':>9} {'roi':>9}")
    for cfg_name, cfg in CONFIGS:
        params = dict(objective='binary:logistic', eval_metric=['logloss', 'auc'],
                      random_state=42, n_jobs=-1, verbosity=0, **cfg)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models_by_config[cfg_name] = model

        prob_val = model.predict_proba(X_val)[:, 1]
        probs_val_by_config[cfg_name] = prob_val

        for thresh in EV_THRESHOLDS:
            results = grade_bets(prob_val, y_val, odds_over_val, odds_under_val,
                                  gid_val, gaid_val, thresh)
            summary = summarize_bets(results, n_val)
            entry = {
                'feature_set': name, 'config': cfg_name, 'hyperparams': cfg,
                'threshold': thresh, **summary,
            }
            val_evaluations.append({'entry': entry, 'results': results})
            log(f"{cfg_name:<20} {thresh:>7.2f} {summary['bets']:>6} "
                f"{summary['bet_rate']:>8.1f}% {summary['hit_rate']:>8.1f}% {summary['roi']:>+8.2f}%")

    assert len(val_evaluations) == 24, f"Expected 24 val evaluations for set {name}, got {len(val_evaluations)}"

    # --- Selection: filter to 15-35% val bet rate, rank by val ROI ---
    in_range = [e for e in val_evaluations if 15 <= e['entry']['bet_rate'] <= 35]
    deviation = None
    if not in_range:
        in_range = [e for e in val_evaluations if 10 <= e['entry']['bet_rate'] <= 40]
        deviation = (f"Feature set {name}: no config landed in the pre-registered 15-35% "
                     f"val bet-rate band; widened to 10-40% per protocol fallback.")
        log(f"\nWARNING: {deviation}")
        if not in_range:
            in_range = val_evaluations
            deviation += " Even the widened 10-40% band was empty; fell back to ALL 24 evaluations."
            log(f"WARNING: {deviation}")

    winner = max(in_range, key=lambda e: e['entry']['roi'])
    winner_cfg_name = winner['entry']['config']
    winner_thresh = winner['entry']['threshold']
    winner_model = models_by_config[winner_cfg_name]
    winner_prob_val = probs_val_by_config[winner_cfg_name]

    log(f"\nWinner for set {name}: config={winner_cfg_name}, threshold={winner_thresh:.2f}")
    log(f"  Val: {winner['entry']['bets']} bets, {winner['entry']['bet_rate']:.1f}% bet rate, "
        f"{winner['entry']['roi']:+.2f}% ROI")

    val_roi_ci = bootstrap_roi_ci(winner['results'])
    val_cluster_ci = cluster_bootstrap_roi_ci(winner['results'])
    val_side = side_breakdown(winner['results'])
    val_auc = auc_both(winner_prob_val, y_val, gid_val, gaid_val)
    val_brier = brier(winner_prob_val, y_val)
    val_goalie_nights_total = goalie_night_count(gid_val, gaid_val)
    val_goalie_nights_bet = bet_goalie_night_count(winner['results'])

    log(f"  Val ROI 95% CI (row bootstrap):     [{val_roi_ci['lower']:+.2f}%, {val_roi_ci['upper']:+.2f}%]")
    log(f"  Val ROI 95% CI (cluster bootstrap):  [{val_cluster_ci['lower']:+.2f}%, {val_cluster_ci['upper']:+.2f}%]  "
        f"(n_clusters={val_cluster_ci['n_clusters']})")
    log(f"  Val AUC row-level={val_auc['row_level']:.4f}  one-per-goalie-night={val_auc['one_per_goalie_night']:.4f}")
    log(f"  Val Brier={val_brier:.5f}")
    log(f"  Val goalie-nights: {val_goalie_nights_total} total, {val_goalie_nights_bet} with a bet")
    log(f"  Val side breakdown: OVER {val_side['OVER']['bets']} bets "
        f"({val_side['OVER']['roi']:+.2f}%), UNDER {val_side['UNDER']['bets']} bets "
        f"({val_side['UNDER']['roi']:+.2f}%)")

    # --- Single test-fold touch ---
    log(f"\n--- SINGLE TEST TOUCH: feature set {name}, config={winner_cfg_name}, threshold={winner_thresh:.2f} ---")
    prob_test = winner_model.predict_proba(X_test)[:, 1]
    test_results = grade_bets(prob_test, y_test, odds_over_test, odds_under_test,
                               gid_test, gaid_test, winner_thresh)
    n_test = len(test_idx)
    test_summary = summarize_bets(test_results, n_test)
    test_roi_ci = bootstrap_roi_ci(test_results)
    test_cluster_ci = cluster_bootstrap_roi_ci(test_results)
    test_side = side_breakdown(test_results)
    test_auc = auc_both(prob_test, y_test, gid_test, gaid_test)
    test_brier = brier(prob_test, y_test)
    test_goalie_nights_total = goalie_night_count(gid_test, gaid_test)
    test_goalie_nights_bet = bet_goalie_night_count(test_results)

    log(f"  Test: {test_summary['bets']} bets, {test_summary['bet_rate']:.1f}% bet rate, "
        f"{test_summary['hit_rate']:.1f}% hit rate, {test_summary['roi']:+.2f}% ROI")
    log(f"  Test ROI 95% CI (row bootstrap):     [{test_roi_ci['lower']:+.2f}%, {test_roi_ci['upper']:+.2f}%]")
    log(f"  Test ROI 95% CI (cluster bootstrap):  [{test_cluster_ci['lower']:+.2f}%, {test_cluster_ci['upper']:+.2f}%]  "
        f"(n_clusters={test_cluster_ci['n_clusters']})")
    log(f"  Test AUC row-level={test_auc['row_level']:.4f}  one-per-goalie-night={test_auc['one_per_goalie_night']:.4f}")
    log(f"  Test Brier={test_brier:.5f}")
    log(f"  Test goalie-nights: {test_goalie_nights_total} total, {test_goalie_nights_bet} with a bet")
    log(f"  Test side breakdown: OVER {test_side['OVER']['bets']} bets "
        f"({test_side['OVER']['roi']:+.2f}%), UNDER {test_side['UNDER']['bets']} bets "
        f"({test_side['UNDER']['roi']:+.2f}%)")

    result = {
        'feature_set': name,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'val_evaluations': [e['entry'] for e in val_evaluations],
        'selection_deviation': deviation,
        'winner': {
            'config': winner_cfg_name, 'hyperparams': winner['entry']['hyperparams'],
            'threshold': winner_thresh,
            'val_summary': winner['entry'],
            'val_roi_ci_row': val_roi_ci,
            'val_roi_ci_cluster': val_cluster_ci,
            'val_side_breakdown': val_side,
            'val_auc': val_auc,
            'val_brier': val_brier,
            'val_goalie_nights_total': val_goalie_nights_total,
            'val_goalie_nights_bet': val_goalie_nights_bet,
        },
        'test': {
            'summary': test_summary,
            'roi_ci_row': test_roi_ci,
            'roi_ci_cluster': test_cluster_ci,
            'side_breakdown': test_side,
            'auc': test_auc,
            'brier': test_brier,
            'goalie_nights_total': test_goalie_nights_total,
            'goalie_nights_bet': test_goalie_nights_bet,
        },
    }

    return result, winner_model, winner_prob_val, y_val, test_results


# ============================================================
# Section 5: baseline D -- no-model market probability
# ============================================================

def run_baseline_D(df, val_idx, test_idx, log):
    log("\n" + "=" * 80)
    log("FEATURE SET D: no-model market baseline (probability = fair_prob_over)")
    log("=" * 80)

    y = df['over_hit'].values.astype(int)
    odds_over = df['odds_over_american'].astype(float).values
    odds_under = df['odds_under_american'].astype(float).values
    game_id = df['game_id'].values
    goalie_id = df['goalie_id'].values
    fair_prob = df['fair_prob_over'].values.astype(float)

    out = {}
    for split_name, idx in (('val', val_idx), ('test', test_idx)):
        prob = fair_prob[idx]
        y_split = y[idx]
        gid = game_id[idx]
        gaid = goalie_id[idx]
        odds_o = odds_over[idx]
        odds_u = odds_under[idx]

        auc = auc_both(prob, y_split, gid, gaid)
        b = brier(prob, y_split)
        log(f"\n  [{split_name}] AUC row-level={auc['row_level']:.4f} "
            f"one-per-goalie-night={auc['one_per_goalie_night']:.4f}  Brier={b:.5f}")

        threshold_results = {}
        log(f"  [{split_name}] betting-rule performance at all 4 thresholds (no selection):")
        log(f"    {'thresh':>7} {'bets':>6} {'bet_rate':>9} {'hit_rate':>9} {'roi':>9}")
        for thresh in EV_THRESHOLDS:
            results = grade_bets(prob, y_split, odds_o, odds_u, gid, gaid, thresh)
            summary = summarize_bets(results, len(idx))
            row_ci = bootstrap_roi_ci(results)
            cluster_ci = cluster_bootstrap_roi_ci(results)
            side = side_breakdown(results)
            threshold_results[thresh] = {
                'summary': summary, 'roi_ci_row': row_ci, 'roi_ci_cluster': cluster_ci,
                'side_breakdown': side, 'goalie_nights_bet': bet_goalie_night_count(results),
            }
            log(f"    {thresh:>7.2f} {summary['bets']:>6} {summary['bet_rate']:>8.1f}% "
                f"{summary['hit_rate']:>8.1f}% {summary['roi']:>+8.2f}%")

        out[split_name] = {
            'auc': auc, 'brier': b,
            'goalie_nights_total': goalie_night_count(gid, gaid),
            'threshold_results': {str(t): v for t, v in threshold_results.items()},
        }

    return out


# ============================================================
# Section 6: anchoring diagnostics for model B (val only)
# ============================================================

def anchoring_diagnostics_B(winner_model_B, prob_val_B, y_val, df, val_idx, winner_thresh_B, log):
    log("\n" + "=" * 80)
    log("ANCHORING DIAGNOSTICS -- feature set B (validation fold only, never test)")
    log("=" * 80)

    market_val = df['fair_prob_over'].values[val_idx].astype(float)
    corr = float(np.corrcoef(prob_val_B, market_val)[0, 1])
    disagreement = prob_val_B - market_val

    quantile_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    disagreement_quantiles = {f'p{q}': float(np.percentile(disagreement, q)) for q in quantile_levels}

    log(f"\nCorrelation(model B prob, fair_prob_over) on val: {corr:.4f}")
    log("Disagreement (model - market) quantiles on val:")
    for q in quantile_levels:
        log(f"  p{q:>3}: {disagreement_quantiles[f'p{q}']:+.4f}")

    odds_over_val = df['odds_over_american'].astype(float).values[val_idx]
    odds_under_val = df['odds_under_american'].astype(float).values[val_idx]
    game_id_val = df['game_id'].values[val_idx]
    goalie_id_val = df['goalie_id'].values[val_idx]

    bet_results = grade_bets(prob_val_B, y_val, odds_over_val, odds_under_val,
                              game_id_val, goalie_id_val, winner_thresh_B)

    buckets = [(0.05, 0.10), (0.10, 0.15), (0.15, np.inf)]
    bucket_report = {}
    log(f"\nVal bets bucketed by |model - market| disagreement magnitude "
        f"(winner B threshold={winner_thresh_B:.2f}):")
    for lo, hi in buckets:
        label = f"[{lo:.2f}, {hi:.2f})" if np.isfinite(hi) else f">={lo:.2f}"
        bucket_bets = [r for r in bet_results if lo <= abs(disagreement[r['local_idx']]) < hi]
        summary = summarize_bets(bucket_bets, len(bucket_bets) if bucket_bets else 1)
        bucket_report[label] = summary
        log(f"  {label:<14} n={summary['bets']:>4}  hit_rate={summary['hit_rate']:>6.1f}%  roi={summary['roi']:>+7.2f}%")

    # Top-20 gain feature importances.
    booster = winner_model_B.get_booster()
    gain_scores = booster.get_score(importance_type='gain')
    top20 = sorted(gain_scores.items(), key=lambda kv: kv[1], reverse=True)[:20]
    log("\nTop-20 features by gain (model B winner):")
    market_or_book_in_top20 = 0
    for rank, (feat, gain) in enumerate(top20, 1):
        is_market = feat in MARKET_FEATURE_COLS or feat.startswith('book_')
        if is_market:
            market_or_book_in_top20 += 1
        tag = "  <-- market/book" if is_market else ""
        log(f"  {rank:>2}. {feat:<35} gain={gain:>12.2f}{tag}")
    log(f"\nMarket/book features in top 20 by gain: {market_or_book_in_top20}/20")

    return {
        'correlation_model_vs_market_val': corr,
        'disagreement_quantiles_val': disagreement_quantiles,
        'disagreement_buckets_val': bucket_report,
        'top20_gain_features': [{'feature': f, 'gain': float(g), 'is_market_or_book': (f in MARKET_FEATURE_COLS or f.startswith('book_'))} for f, g in top20],
        'market_or_book_in_top20': market_or_book_in_top20,
    }


# ============================================================
# Main
# ============================================================

def main():
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_ROOT / f'experiment_market_anchor_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'run_log.txt'
    log_lines = []

    def log(msg=''):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 80)
    log("MARKET-ANCHORED RESIDUAL MODEL EXPERIMENT (roadmap item 7)")
    log(f"Run timestamp: {datetime.now().isoformat()}")
    log("=" * 80)

    df, feature_cols_A, feature_cols_B, feature_cols_C, book_cols = build_modeling_frame(log)
    train_idx, val_idx, test_idx = split_by_date(df, log)

    assert len(feature_cols_C) == len(MARKET_FEATURE_COLS) + len(book_cols) + 1

    results = {}
    winner_models = {}
    winner_probs_val = {}

    for name, feature_cols in [('A', feature_cols_A), ('B', feature_cols_B), ('C', feature_cols_C)]:
        result, winner_model, winner_prob_val, y_val_local, test_results = run_feature_set(
            name, feature_cols, df, train_idx, val_idx, test_idx, log
        )
        results[name] = result
        winner_models[name] = winner_model
        winner_probs_val[name] = winner_prob_val

    baseline_D = run_baseline_D(df, val_idx, test_idx, log)

    y_val = df['over_hit'].values.astype(int)[val_idx]
    anchoring_B = anchoring_diagnostics_B(
        winner_models['B'], winner_probs_val['B'], y_val, df, val_idx,
        results['B']['winner']['threshold'], log
    )

    # --- Cross-cutting summary questions ---
    log("\n" + "=" * 80)
    log("SUMMARY -- ANSWERING THE FOUR PROTOCOL QUESTIONS")
    log("=" * 80)

    b_test_auc = results['B']['test']['auc']['row_level']
    d_test_auc = baseline_D['test']['auc']['row_level']
    a_test_auc = results['A']['test']['auc']['row_level']
    b_test_brier = results['B']['test']['brier']
    d_test_brier = baseline_D['test']['brier']
    a_test_brier = results['A']['test']['brier']

    log(f"\n1) Does B beat D (market) on test AUC/Brier?")
    log(f"   B row-AUC={b_test_auc:.4f} vs D row-AUC={d_test_auc:.4f}  "
        f"({'B beats D' if b_test_auc > d_test_auc else 'B does NOT beat D'})")
    log(f"   B Brier={b_test_brier:.5f} vs D Brier={d_test_brier:.5f}  "
        f"({'B beats D' if b_test_brier < d_test_brier else 'B does NOT beat D'} -- lower Brier is better)")

    log(f"\n2) Does B beat A (control)?")
    log(f"   B row-AUC={b_test_auc:.4f} vs A row-AUC={a_test_auc:.4f}  "
        f"({'B beats A' if b_test_auc > a_test_auc else 'B does NOT beat A'})")
    log(f"   B Brier={b_test_brier:.5f} vs A Brier={a_test_brier:.5f}  "
        f"({'B beats A' if b_test_brier < a_test_brier else 'B does NOT beat A'} -- lower Brier is better)")
    log(f"   B test ROI={results['B']['test']['summary']['roi']:+.2f}% vs "
        f"A test ROI={results['A']['test']['summary']['roi']:+.2f}%")

    log(f"\n3) Does the betting sim for B/C survive its single test touch with a "
        f"cluster CI excluding zero?")
    for name in ('B', 'C'):
        ci = results[name]['test']['roi_ci_cluster']
        roi = results[name]['test']['summary']['roi']
        excludes_zero = ci['lower'] > 0 or ci['upper'] < 0
        log(f"   {name}: test ROI={roi:+.2f}%, cluster CI=[{ci['lower']:+.2f}%, {ci['upper']:+.2f}%]  "
            f"({'excludes zero' if excludes_zero else 'SPANS ZERO -- not distinguishable from breakeven'})")

    log(f"\n4) Does model B collapse to the market or maintain disagreement, and do "
        f"val disagreements resolve in the model's favor?")
    log(f"   Correlation(model B, fair_prob_over) on val: {anchoring_B['correlation_model_vs_market_val']:.4f}")
    log(f"   Disagreement p50={anchoring_B['disagreement_quantiles_val']['p50']:+.4f}, "
        f"p5={anchoring_B['disagreement_quantiles_val']['p5']:+.4f}, "
        f"p95={anchoring_B['disagreement_quantiles_val']['p95']:+.4f}")
    for label, summ in anchoring_B['disagreement_buckets_val'].items():
        log(f"   Bucket {label}: n={summ['bets']}, hit_rate={summ['hit_rate']:.1f}%, roi={summ['roi']:+.2f}%")
    log(f"   Market/book features in top-20 gain: {anchoring_B['market_or_book_in_top20']}/20")

    # --- Sanity check vs v2 result (informational, not a gate) ---
    log("\n" + "=" * 80)
    log("SANITY CHECK vs v2 clean retrain (date-based split will NOT match exactly)")
    log("=" * 80)
    log(f"  v2 (row-index split): val +5.66% ROI (CI spans zero), test -7.09% ROI (CI negative)")
    log(f"  A  (date-based split): val {results['A']['winner']['val_summary']['roi']:+.2f}% ROI, "
        f"test {results['A']['test']['summary']['roi']:+.2f}% ROI")

    elapsed = time.time() - start_time
    log(f"\nWall-clock time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # --- Save metadata ---
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'wall_clock_seconds': elapsed,
        'data_path': str(DATA_PATH),
        'n_rows_modeling_frame': int(len(df)),
        'fold_boundaries': {
            'train': {
                'start': str(df['game_date'].iloc[train_idx].min().date()),
                'end': str(df['game_date'].iloc[train_idx].max().date()),
                'rows': int(len(train_idx)),
            },
            'val': {
                'start': str(df['game_date'].iloc[val_idx].min().date()),
                'end': str(df['game_date'].iloc[val_idx].max().date()),
                'rows': int(len(val_idx)),
            },
            'test': {
                'start': str(df['game_date'].iloc[test_idx].min().date()),
                'end': str(df['game_date'].iloc[test_idx].max().date()),
                'rows': int(len(test_idx)),
            },
        },
        'book_columns': book_cols,
        'market_feature_columns': MARKET_FEATURE_COLS,
        'ev_thresholds': EV_THRESHOLDS,
        'configs': [{'name': n, 'params': c} for n, c in CONFIGS],
        'results': {
            name: {k: v for k, v in r.items() if k != 'feature_cols'}
            for name, r in results.items()
        },
        'feature_cols': {name: r['feature_cols'] for name, r in results.items()},
        'baseline_D': baseline_D,
        'anchoring_diagnostics_B': anchoring_B,
        'protocol_deviations': [r['selection_deviation'] for r in results.values() if r['selection_deviation']],
        'touch_count_audit': {
            name: {'val_evaluations': len(r['val_evaluations']), 'test_evaluations': 1}
            for name, r in results.items()
        },
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    log(f"\nSaved metadata to: {metadata_path}")

    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"Saved run log to: {log_path}")

    log("\n" + "=" * 80)
    log("EXPERIMENT COMPLETE")
    log("=" * 80)

    # Re-write log with the final lines included.
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))


if __name__ == '__main__':
    main()
