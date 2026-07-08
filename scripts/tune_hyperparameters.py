"""
Hyperparameter tuning script for the optimized 114-feature model.

Uses the engineered features from optimize_features.py and searches across
hyperparameter space to maximize ROI while targeting a 15-35% bet rate.

SELECTION METHODOLOGY (v2, clean):
  The test fold is NEVER used to pick a config. During the search, every
  (hyperparameter config, EV threshold) pair is scored ONLY on the
  validation fold: candidates are filtered to a 15-35% validation bet
  rate and ranked by validation ROI. Only after a single winner is chosen
  this way is that one model + EV threshold evaluated on the test fold --
  exactly once, for the whole script run. This replaces the prior
  methodology, which filtered by test bet rate and ranked by combined
  val+test ROI, leaking the test fold into model selection. That leak is
  why the old headline numbers (from the corrupted, smaller training set)
  were retired as contaminated. This run also trains on a corrected,
  regenerated training parquet -- see docs/HISTORICAL_DATA_ANALYSIS.md
  and HANDOVER/HANDOVER.md for the underlying data bug and fix.

  Bootstrap 95% CIs (10,000 resamples, percentile method, numpy seed 42)
  are reported on ROI for both the winner's validation result and the
  single test-fold touch, along with an OVER/UNDER recommendation-side
  breakdown for each, since the live betting record shows those two
  sides behaving very differently.

Runtime: ~10-40 minutes.

Usage:
    python scripts/tune_hyperparameters.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
from models.classifier_trainer import ClassifierTrainer
import json
import xgboost as xgb
import warnings
import itertools
warnings.filterwarnings('ignore')


# ============================================================
# Data Loading + Feature Engineering (same as optimize_features.py)
# ============================================================

def add_all_engineered_features(df):
    """Reproduce the 18 engineered features from optimize_features.py."""
    df = df.copy()

    # Interaction features
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

    # Volatility features
    for w in [5, 10]:
        mean_col = f'saves_rolling_{w}'
        std_col = f'saves_rolling_std_{w}'
        if mean_col in df.columns and std_col in df.columns:
            df[f'saves_cv_{w}'] = df[std_col] / df[mean_col].clip(lower=1)
        if std_col in df.columns and 'betting_line' in df.columns:
            df[f'volatility_vs_line_{w}'] = df[std_col] / df['betting_line'].clip(lower=1)

    # Trend/momentum features
    for stat in ['saves', 'shots_against', 'goals_against']:
        short = f'{stat}_rolling_3'
        long = f'{stat}_rolling_10'
        if short in df.columns and long in df.columns:
            df[f'{stat}_momentum'] = df[short] - df[long]

    sp_short = 'save_percentage_rolling_3'
    sp_long = 'save_percentage_rolling_10'
    if sp_short in df.columns and sp_long in df.columns:
        df['save_pct_momentum'] = df[sp_short] - df[sp_long]

    # Matchup context features
    if 'opp_shots_rolling_5' in df.columns and 'shots_against_rolling_5' in df.columns:
        df['expected_workload_diff'] = df['opp_shots_rolling_5'] - df['shots_against_rolling_5']

    if 'opp_shots_rolling_5' in df.columns and 'opp_goals_rolling_5' in df.columns:
        opp_saves_implied = df['opp_shots_rolling_5'] - df['opp_goals_rolling_5']
        df['line_vs_opp_implied_saves'] = df['betting_line'] - opp_saves_implied

    if 'goalie_days_rest' in df.columns and 'saves_rolling_5' in df.columns:
        df['rest_x_performance'] = df['goalie_days_rest'].clip(upper=7) * df['saves_rolling_5']

    return df


print("=" * 80)
print("HYPERPARAMETER TUNING (114 engineered features) -- v2 clean selection")
print("=" * 80)

data_path = 'data/processed/multibook_classification_training_data.parquet'
df_raw = pd.read_parquet(data_path)

market_features = [
    'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
    'market_vig', 'impl_prob_over', 'impl_prob_under',
    'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
    'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low'
]
df_raw = df_raw.drop(columns=[c for c in market_features if c in df_raw.columns], errors='ignore')
df_raw = df_raw.sort_values('game_date').reset_index(drop=True)
df_raw = df_raw[df_raw['odds_over_american'].notna() & df_raw['odds_under_american'].notna()].reset_index(drop=True)

# Add engineered features
df = add_all_engineered_features(df_raw)
print(f"Loaded {len(df)} samples with engineered features")

EXCLUDED = [
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

feature_cols = [c for c in df.columns if c not in EXCLUDED]
print(f"Features: {len(feature_cols)}")

# Hard-fail guardrails: this training run must use exactly the 114 approved
# features, with no metadata/identifier columns leaking in as predictors.
assert len(feature_cols) == 114, (
    f"Expected exactly 114 feature columns, got {len(feature_cols)}. "
    f"Feature set must not change for this run."
)
_forbidden_cols = [
    'goalie_name', 'book_key', 'team_abbrev', 'opponent_team',
    'season', 'game_id', 'goalie_id', 'game_date',
]
_leaked = [c for c in _forbidden_cols if c in feature_cols]
assert not _leaked, f"Forbidden metadata/identifier columns leaked into feature_cols: {_leaked}"

# Chronological split 60/20/20
n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

X = df[feature_cols].values
y = df['over_hit'].values

train_idx = np.arange(0, train_end)
val_idx = np.arange(train_end, val_end)
test_idx = np.arange(val_end, n)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

val_total = len(X_val)
test_total = len(X_test)

print(f"Train: {len(X_train)} | Val: {val_total} | Test: {test_total}")
print(f"Target bet rate: 15-35% (Val: {int(val_total*0.15)}-{int(val_total*0.35)} bets, Test: {int(test_total*0.15)}-{int(test_total*0.35)} bets)")

# Fold date boundaries -- printed so downstream results can be interpreted
train_dates = df.iloc[train_idx]['game_date']
val_dates = df.iloc[val_idx]['game_date']
test_dates = df.iloc[test_idx]['game_date']
print("\nFold date boundaries:")
print(f"  Train: {train_dates.min()} to {train_dates.max()}  ({len(train_idx)} rows)")
print(f"  Val:   {val_dates.min()} to {val_dates.max()}  ({len(val_idx)} rows)")
print(f"  Test:  {test_dates.min()} to {test_dates.max()}  ({len(test_idx)} rows)")


# ============================================================
# Bootstrap CI + side-breakdown helpers
# ============================================================

def bootstrap_roi_ci(bet_results, n_resamples=10000, seed=42, ci_pct=95.0):
    """
    Bootstrap a percentile-method confidence interval on ROI by resampling
    per-bet profits with replacement.

    Args:
        bet_results: list of dicts with a 'profit' key (one entry per bet)
        n_resamples: number of bootstrap resamples
        seed: numpy random seed (fixed for reproducibility)
        ci_pct: confidence level, e.g. 95.0 for a 95% CI

    Returns:
        dict with 'lower', 'upper' (ROI percent) and 'n_bets'
    """
    n_bets = len(bet_results)
    if n_bets == 0:
        return {'lower': 0.0, 'upper': 0.0, 'n_bets': 0}

    profits = np.array([r['profit'] for r in bet_results], dtype=float)

    rng = np.random.RandomState(seed)
    resample_idx = rng.randint(0, n_bets, size=(n_resamples, n_bets))
    boot_rois = profits[resample_idx].mean(axis=1) * 100  # ROI = mean profit per bet, as %

    alpha = (100.0 - ci_pct) / 2.0
    lower = float(np.percentile(boot_rois, alpha))
    upper = float(np.percentile(boot_rois, 100.0 - alpha))
    return {'lower': lower, 'upper': upper, 'n_bets': n_bets}


def side_breakdown(bet_results):
    """Split bet_results by recommendation side (OVER/UNDER) and summarize."""
    breakdown = {}
    for side in ('OVER', 'UNDER'):
        side_bets = [r for r in bet_results if r['bet'] == side]
        n_side = len(side_bets)
        if n_side == 0:
            breakdown[side] = {'bets': 0, 'win_rate': 0.0, 'roi': 0.0, 'profit': 0.0}
            continue
        wins = sum(1 for r in side_bets if r['won'])
        profit = sum(r['profit'] for r in side_bets)
        breakdown[side] = {
            'bets': n_side,
            'win_rate': wins / n_side,
            'roi': (profit / n_side) * 100,
            'profit': profit,
        }
    return breakdown


def print_side_breakdown(label, breakdown):
    for side in ('OVER', 'UNDER'):
        b = breakdown[side]
        print(f"    {label} {side}: {b['bets']} bets | win rate {b['win_rate']*100:.1f}% | ROI {b['roi']:+.2f}%")


# ============================================================
# Evaluation function -- VALIDATION FOLD ONLY (search loop)
# ============================================================

def evaluate_val_only(model, ev_threshold):
    """Evaluate model on the validation fold only. The test fold is never
    touched here -- this function is the only thing the search loop calls."""
    trainer = ClassifierTrainer()
    trainer.model = model
    trainer.feature_names = feature_cols

    val_m = trainer.evaluate_profitability(
        X_val, y_val, df, val_idx, dataset_name='Val', ev_threshold=ev_threshold
    )

    val_bet_rate = val_m['total_bets'] / val_total * 100

    return {
        'val_roi': val_m['roi'], 'val_bets': val_m['total_bets'], 'val_bet_rate': val_bet_rate,
        'val_win_rate': val_m['win_rate'], 'val_profit': val_m['total_profit'],
        'val_bet_results': val_m['bet_results'],
    }


# ============================================================
# Hyperparameter Grid
# ============================================================

# Dimensions to search (focused on what matters most for XGBoost)
param_grid = {
    'max_depth':        [3, 4, 5, 6],
    'learning_rate':    [0.01, 0.02, 0.05],
    'min_child_weight': [10, 15, 20, 30],
    'gamma':            [0.5, 1.0, 2.0],
    'reg_alpha':        [5, 10, 20],
    'reg_lambda':       [20, 40, 60],
    'n_estimators':     [600, 800, 1200],
    'subsample':        [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
}

# EV thresholds to try per config (controls bet selectivity)
EV_THRESHOLDS = [0.08, 0.10, 0.12, 0.15]

# Full grid is too large -- use randomized search
total_combos = 1
for v in param_grid.values():
    total_combos *= len(v)
print(f"\nFull grid: {total_combos} combinations (too many)")

N_RANDOM = 40
print(f"Random search: {N_RANDOM} configs x {len(EV_THRESHOLDS)} EV thresholds = {N_RANDOM * len(EV_THRESHOLDS)} evaluations")

np.random.seed(42)
random_configs = []
for _ in range(N_RANDOM):
    config = {}
    for key, values in param_grid.items():
        config[key] = values[np.random.randint(len(values))]
    random_configs.append(config)

# Also include the current best config and baseline
random_configs.insert(0, {
    'max_depth': 4, 'learning_rate': 0.02, 'min_child_weight': 15,
    'gamma': 2.0, 'reg_alpha': 20, 'reg_lambda': 60,
    'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8,
})
random_configs.insert(1, {
    'max_depth': 4, 'learning_rate': 0.02, 'min_child_weight': 15,
    'gamma': 1.0, 'reg_alpha': 10, 'reg_lambda': 40,
    'n_estimators': 800, 'subsample': 0.8, 'colsample_bytree': 0.8,
})

print(f"Total configs to train: {len(random_configs)}")
print(f"\nStarting search (validation fold only -- test fold is not touched)...\n")

# ============================================================
# Search Loop -- VALIDATION FOLD ONLY. Test fold is never referenced
# anywhere in this loop, directly or indirectly.
# ============================================================

all_results = []

for i, config in enumerate(random_configs):
    label = "Current best" if i == 0 else ("Config #4398 baseline" if i == 1 else f"Random #{i-1}")
    print(f"[{i+1}/{len(random_configs)}] {label}: depth={config['max_depth']} lr={config['learning_rate']} mcw={config['min_child_weight']} gamma={config['gamma']} alpha={config['reg_alpha']} lambda={config['reg_lambda']} est={config['n_estimators']}")

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        **config
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    for ev_thresh in EV_THRESHOLDS:
        metrics = evaluate_val_only(model, ev_thresh)

        result = {
            'config_idx': i,
            'label': label,
            'ev_threshold': ev_thresh,
            **config,
            **metrics,
            'model': model,
        }
        all_results.append(result)

    # Print quick summary for best EV threshold (validation-only selection)
    best_ev = max(
        [r for r in all_results if r['config_idx'] == i],
        key=lambda r: r['val_roi'] if 15 <= r['val_bet_rate'] <= 35 else r['val_roi'] - 50
    )
    print(f"  -> Best: EV>={best_ev['ev_threshold']*100:.0f}% | Val: {best_ev['val_roi']:+.1f}% ({best_ev['val_bets']} bets, {best_ev['val_bet_rate']:.0f}%)")


# ============================================================
# Results Analysis -- VALIDATION FOLD ONLY. This is where the winner
# is chosen. The test fold has not been referenced anywhere above.
# ============================================================

print("\n" + "=" * 80)
print("TOP 15 CONFIGS (sorted by VALIDATION ROI, filtered to 15-35% validation bet rate)")
print("=" * 80)

# Filter to configs in the target bet rate range (validation only)
in_range = [r for r in all_results if 15 <= r['val_bet_rate'] <= 35]
fallback_used = False
if not in_range:
    print("No configs in target validation bet rate range! Showing all:")
    in_range = all_results
    fallback_used = True

in_range_sorted = sorted(in_range, key=lambda r: r['val_roi'], reverse=True)

print(f"\n{'#':>3} {'Label':<22} {'EV':>4} {'Depth':>5} {'LR':>5} {'MCW':>4} {'Gam':>4} {'Alpha':>5} {'Lam':>4} {'Est':>5} | {'V ROI':>7} {'V Bets':>6} {'V Bt%':>5}")
print("-" * 110)

for rank, r in enumerate(in_range_sorted[:15], 1):
    print(f"{rank:>3} {r['label']:<22} {r['ev_threshold']*100:>3.0f}% {r['max_depth']:>5} {r['learning_rate']:>5} {r['min_child_weight']:>4} {r['gamma']:>4} {r['reg_alpha']:>5} {r['reg_lambda']:>4} {r['n_estimators']:>5} | {r['val_roi']:>+6.1f}% {r['val_bets']:>6} {r['val_bet_rate']:>4.0f}%")

# ============================================================
# Select winner -- based on VALIDATION ROI ONLY
# ============================================================

best = in_range_sorted[0]
print("\n" + "=" * 80)
print("WINNING CONFIG (selected on validation fold only)")
print("=" * 80)
print(f"  Label: {best['label']}")
print(f"  EV Threshold: {best['ev_threshold']*100:.0f}%")
print(f"  max_depth: {best['max_depth']}")
print(f"  learning_rate: {best['learning_rate']}")
print(f"  min_child_weight: {best['min_child_weight']}")
print(f"  gamma: {best['gamma']}")
print(f"  reg_alpha: {best['reg_alpha']}")
print(f"  reg_lambda: {best['reg_lambda']}")
print(f"  n_estimators: {best['n_estimators']}")
print(f"  subsample: {best['subsample']}")
print(f"  colsample_bytree: {best['colsample_bytree']}")
print(f"\n  Val ROI: {best['val_roi']:+.2f}% ({best['val_bets']} bets, {best['val_bet_rate']:.1f}%)")
print(f"  Val Win Rate: {best['val_win_rate']*100:.1f}%")

val_roi_ci = bootstrap_roi_ci(best['val_bet_results'], n_resamples=10000, seed=42, ci_pct=95.0)
print(f"  Val ROI 95% CI (bootstrap, 10000 resamples): [{val_roi_ci['lower']:+.2f}%, {val_roi_ci['upper']:+.2f}%]")

val_breakdown = side_breakdown(best['val_bet_results'])
print("  Val OVER/UNDER breakdown:")
print_side_breakdown("Val", val_breakdown)

# ============================================================
# SINGLE TEST-FOLD TOUCH -- the only place in this script that
# evaluates on the test fold. Uses the winning model + EV threshold,
# both chosen entirely from validation-fold performance above.
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST-FOLD EVALUATION (single touch -- test fold was not used for selection)")
print("=" * 80)

test_trainer = ClassifierTrainer()
test_trainer.model = best['model']
test_trainer.feature_names = feature_cols

test_m = test_trainer.evaluate_profitability(
    X_test, y_test, df, test_idx, dataset_name='Test', ev_threshold=best['ev_threshold']
)
test_bet_rate = test_m['total_bets'] / test_total * 100

print(f"\n  Test ROI: {test_m['roi']:+.2f}% ({test_m['total_bets']} bets, {test_bet_rate:.1f}%)")
print(f"  Test Win Rate: {test_m['win_rate']*100:.1f}%")

test_roi_ci = bootstrap_roi_ci(test_m['bet_results'], n_resamples=10000, seed=42, ci_pct=95.0)
print(f"  Test ROI 95% CI (bootstrap, 10000 resamples): [{test_roi_ci['lower']:+.2f}%, {test_roi_ci['upper']:+.2f}%]")

test_breakdown = side_breakdown(test_m['bet_results'])
print("  Test OVER/UNDER breakdown:")
print_side_breakdown("Test", test_breakdown)

if test_roi_ci['lower'] < 0 < test_roi_ci['upper']:
    print("\n  WARNING: Test ROI 95% CI spans zero -- result is not statistically distinguishable from breakeven.")
if test_m['roi'] < 0:
    print("\n  WARNING: Test ROI is negative.")
if fallback_used:
    print("\n  WARNING: No configs fell in the 15-35% validation bet rate range; fallback to all configs was used.")

# ============================================================
# Save Best Model
# ============================================================

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_dir = Path('models/trained') / f'tuned_v2_clean_{timestamp}'
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / 'classifier_model.json'
feature_path = model_dir / 'classifier_feature_names.json'
metadata_path = model_dir / 'classifier_metadata.json'

best['model'].get_booster().save_model(str(model_path))
with open(feature_path, 'w') as f:
    json.dump(feature_cols, f, indent=2)

metadata = {
    'config_name': f'Tuned V2 Clean ({best["label"]})',
    'trained_date': datetime.now().isoformat(),
    'selection_method': 'validation_only_test_touched_once',
    'training_data': 'multibook_classification_training_data.parquet (13192 rows, regenerated 2026-07-07 post-corruption-fix)',
    'ev_threshold': best['ev_threshold'],
    'hyperparameters': {
        'max_depth': best['max_depth'],
        'learning_rate': best['learning_rate'],
        'min_child_weight': best['min_child_weight'],
        'gamma': best['gamma'],
        'reg_alpha': best['reg_alpha'],
        'reg_lambda': best['reg_lambda'],
        'n_estimators': best['n_estimators'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
    },
    'num_features': len(feature_cols),
    'fold_boundaries': {
        'train': {'start': str(train_dates.min()), 'end': str(train_dates.max()), 'rows': int(len(train_idx))},
        'val': {'start': str(val_dates.min()), 'end': str(val_dates.max()), 'rows': int(len(val_idx))},
        'test': {'start': str(test_dates.min()), 'end': str(test_dates.max()), 'rows': int(len(test_idx))},
    },
    'val_roi': best['val_roi'],
    'val_bets': best['val_bets'],
    'val_bet_rate': best['val_bet_rate'],
    'val_win_rate': best['val_win_rate'],
    'val_roi_ci_95': val_roi_ci,
    'val_breakdown': val_breakdown,
    'test_roi': test_m['roi'],
    'test_bets': test_m['total_bets'],
    'test_bet_rate': test_bet_rate,
    'test_win_rate': test_m['win_rate'],
    'test_roi_ci_95': test_roi_ci,
    'test_breakdown': test_breakdown,
    'features': feature_cols,
}
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel saved to: {model_dir}")
print(f"\nTo use this model:")
print(f"  1. Update predictor.py default paths to: {model_dir}")
print(f"  2. Update EV threshold in predictor.py _determine_recommendation() to: {best['ev_threshold']}")
print(f"  3. Add engineered features to feature_calculator.py")
print(f"  4. Run: python scripts/fetch_and_predict.py -v")

print("\n" + "=" * 80)
print("TUNING COMPLETE")
print("=" * 80)
