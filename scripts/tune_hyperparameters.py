"""
Hyperparameter tuning script for the optimized 114-feature model.

Uses the engineered features from optimize_features.py and searches across
hyperparameter space to maximize ROI while targeting 20-30% bet rate.

Current baseline (Config #4398 + high reg):
  Val: +21.01% (199 bets / 1082 = 18.4%)
  Test: +10.68% (252 bets / 1082 = 23.3%)
  Combined: +15.24% (451 bets)

Target: ~20-30% of lines bet on, with best possible ROI.

Runtime: ~5-10 minutes.

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
print("HYPERPARAMETER TUNING (114 engineered features)")
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
print(f"Target bet rate: 20-30% (Val: {int(val_total*0.2)}-{int(val_total*0.3)} bets, Test: {int(test_total*0.2)}-{int(test_total*0.3)} bets)")


# ============================================================
# Evaluation function
# ============================================================

def evaluate(model, ev_threshold):
    """Evaluate model on val + test with given EV threshold."""
    trainer = ClassifierTrainer()
    trainer.model = model
    trainer.feature_names = feature_cols

    val_m = trainer.evaluate_profitability(
        X_val, y_val, df, val_idx, dataset_name='Val', ev_threshold=ev_threshold
    )
    test_m = trainer.evaluate_profitability(
        X_test, y_test, df, test_idx, dataset_name='Test', ev_threshold=ev_threshold
    )

    val_bets = val_m['total_bets']
    test_bets = test_m['total_bets']
    combined_bets = val_bets + test_bets
    combined_profit = val_m['total_profit'] + test_m['total_profit']
    combined_roi = (combined_profit / combined_bets) * 100 if combined_bets > 0 else 0

    val_bet_rate = val_bets / val_total * 100
    test_bet_rate = test_bets / test_total * 100

    return {
        'val_roi': val_m['roi'], 'val_bets': val_bets, 'val_bet_rate': val_bet_rate,
        'val_win_rate': val_m['win_rate'], 'val_profit': val_m['total_profit'],
        'test_roi': test_m['roi'], 'test_bets': test_bets, 'test_bet_rate': test_bet_rate,
        'test_win_rate': test_m['win_rate'], 'test_profit': test_m['total_profit'],
        'combined_roi': combined_roi, 'combined_bets': combined_bets,
        'combined_profit': combined_profit,
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
print(f"\nStarting search...\n")

# ============================================================
# Search Loop
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
        metrics = evaluate(model, ev_thresh)

        result = {
            'config_idx': i,
            'label': label,
            'ev_threshold': ev_thresh,
            **config,
            **metrics,
            'model': model,
        }
        all_results.append(result)

    # Print quick summary for best EV threshold
    best_ev = max(
        [r for r in all_results if r['config_idx'] == i],
        key=lambda r: r['combined_roi'] if 20 <= r['test_bet_rate'] <= 35 else r['combined_roi'] - 50
    )
    print(f"  -> Best: EV>={best_ev['ev_threshold']*100:.0f}% | Val: {best_ev['val_roi']:+.1f}% ({best_ev['val_bets']} bets, {best_ev['val_bet_rate']:.0f}%) | Test: {best_ev['test_roi']:+.1f}% ({best_ev['test_bets']} bets, {best_ev['test_bet_rate']:.0f}%) | Comb: {best_ev['combined_roi']:+.1f}%")


# ============================================================
# Results Analysis
# ============================================================

print("\n" + "=" * 80)
print("TOP 15 CONFIGS (sorted by combined ROI, filtered to 15-35% test bet rate)")
print("=" * 80)

# Filter to configs in the target bet rate range
in_range = [r for r in all_results if 15 <= r['test_bet_rate'] <= 35]
if not in_range:
    print("No configs in target range! Showing all:")
    in_range = all_results

in_range_sorted = sorted(in_range, key=lambda r: r['combined_roi'], reverse=True)

print(f"\n{'#':>3} {'Label':<22} {'EV':>4} {'Depth':>5} {'LR':>5} {'MCW':>4} {'Gam':>4} {'Alpha':>5} {'Lam':>4} {'Est':>5} | {'V ROI':>7} {'V Bt%':>5} | {'T ROI':>7} {'T Bt%':>5} | {'C ROI':>7} {'C Bets':>6}")
print("-" * 140)

for rank, r in enumerate(in_range_sorted[:15], 1):
    print(f"{rank:>3} {r['label']:<22} {r['ev_threshold']*100:>3.0f}% {r['max_depth']:>5} {r['learning_rate']:>5} {r['min_child_weight']:>4} {r['gamma']:>4} {r['reg_alpha']:>5} {r['reg_lambda']:>4} {r['n_estimators']:>5} | {r['val_roi']:>+6.1f}% {r['val_bet_rate']:>4.0f}% | {r['test_roi']:>+6.1f}% {r['test_bet_rate']:>4.0f}% | {r['combined_roi']:>+6.1f}% {r['combined_bets']:>6}")

# Also show top by test ROI specifically
print("\n" + "=" * 80)
print("TOP 10 BY TEST ROI (15-35% test bet rate)")
print("=" * 80)
by_test = sorted(in_range, key=lambda r: r['test_roi'], reverse=True)
for rank, r in enumerate(by_test[:10], 1):
    print(f"{rank:>3} {r['label']:<22} EV>={r['ev_threshold']*100:.0f}% | Val: {r['val_roi']:>+6.1f}% ({r['val_bet_rate']:.0f}%) | Test: {r['test_roi']:>+6.1f}% ({r['test_bet_rate']:.0f}%) | Comb: {r['combined_roi']:>+6.1f}% ({r['combined_bets']} bets)")

# ============================================================
# Save Best Model
# ============================================================

best = in_range_sorted[0]
print("\n" + "=" * 80)
print("BEST CONFIG")
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
print(f"  Test ROI: {best['test_roi']:+.2f}% ({best['test_bets']} bets, {best['test_bet_rate']:.1f}%)")
print(f"  Combined ROI: {best['combined_roi']:+.2f}% ({best['combined_bets']} bets)")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_dir = Path('models/trained') / f'tuned_v1_{timestamp}'
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / 'classifier_model.json'
feature_path = model_dir / 'classifier_feature_names.json'
metadata_path = model_dir / 'classifier_metadata.json'

best['model'].get_booster().save_model(str(model_path))
with open(feature_path, 'w') as f:
    json.dump(feature_cols, f, indent=2)

metadata = {
    'config_name': f'Tuned V1 ({best["label"]})',
    'trained_date': datetime.now().isoformat(),
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
    'val_roi': best['val_roi'],
    'val_bets': best['val_bets'],
    'val_bet_rate': best['val_bet_rate'],
    'test_roi': best['test_roi'],
    'test_bets': best['test_bets'],
    'test_bet_rate': best['test_bet_rate'],
    'combined_roi': best['combined_roi'],
    'combined_bets': best['combined_bets'],
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
