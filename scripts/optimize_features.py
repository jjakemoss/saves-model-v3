"""
Feature engineering optimization script.

Tests multiple feature configurations to find the best-performing model:
1. Baseline (current 96 features)
2. New engineered features (interactions, ratios, trend features)
3. Feature selection (drop low-importance features)
4. Hyperparameter variations alongside feature changes

Runtime: ~5-10 minutes depending on hardware.

Usage:
    python scripts/optimize_features.py
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
warnings.filterwarnings('ignore')

EV_THRESHOLD = 0.12

# ============================================================
# Data Loading
# ============================================================
print("=" * 80)
print("FEATURE ENGINEERING OPTIMIZATION")
print("=" * 80)

data_path = 'data/processed/multibook_classification_training_data.parquet'
df_raw = pd.read_parquet(data_path)
print(f"Loaded {len(df_raw)} samples")

# Drop market-derived features (not available at inference)
market_features = [
    'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
    'market_vig', 'impl_prob_over', 'impl_prob_under',
    'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
    'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low'
]
df_raw = df_raw.drop(columns=[c for c in market_features if c in df_raw.columns], errors='ignore')
df_raw = df_raw.sort_values('game_date').reset_index(drop=True)
df_raw = df_raw[df_raw['odds_over_american'].notna() & df_raw['odds_under_american'].notna()].reset_index(drop=True)
print(f"After filtering: {len(df_raw)} samples with odds")

# Excluded columns (raw stats, identifiers, target)
EXCLUDED_BASE = [
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


# ============================================================
# Feature Engineering Functions
# ============================================================

def add_interaction_features(df):
    """Add interaction features that combine existing rolling stats."""
    df = df.copy()

    # Saves-to-shots ratio rolling (how efficient is this goalie recently?)
    for w in [3, 5, 10]:
        sr = f'saves_rolling_{w}'
        sar = f'shots_against_rolling_{w}'
        if sr in df.columns and sar in df.columns:
            df[f'save_efficiency_{w}'] = df[sr] / df[sar].clip(lower=1)

    # Even strength proportion of total saves
    for w in [5, 10]:
        es = f'even_strength_saves_rolling_{w}'
        sr = f'saves_rolling_{w}'
        if es in df.columns and sr in df.columns:
            df[f'es_saves_proportion_{w}'] = df[es] / df[sr].clip(lower=1)

    # Opponent shots vs team shots against (opponent firepower relative to team defense)
    if 'opp_shots_rolling_5' in df.columns and 'team_shots_against_rolling_5' in df.columns:
        df['opp_vs_team_shots_5'] = df['opp_shots_rolling_5'] - df['team_shots_against_rolling_5']
    if 'opp_shots_rolling_10' in df.columns and 'team_shots_against_rolling_10' in df.columns:
        df['opp_vs_team_shots_10'] = df['opp_shots_rolling_10'] - df['team_shots_against_rolling_10']

    return df


def add_volatility_features(df):
    """Add features that capture goalie consistency/volatility."""
    df = df.copy()

    # Coefficient of variation (std/mean) for saves - high = inconsistent
    for w in [5, 10]:
        mean_col = f'saves_rolling_{w}'
        std_col = f'saves_rolling_std_{w}'
        if mean_col in df.columns and std_col in df.columns:
            df[f'saves_cv_{w}'] = df[std_col] / df[mean_col].clip(lower=1)

    # Range proxy: std relative to line (more volatile = harder to predict)
    for w in [5, 10]:
        std_col = f'saves_rolling_std_{w}'
        if std_col in df.columns and 'betting_line' in df.columns:
            df[f'volatility_vs_line_{w}'] = df[std_col] / df['betting_line'].clip(lower=1)

    return df


def add_trend_features(df):
    """Add features that capture recent form changes."""
    df = df.copy()

    # Short-term vs long-term momentum (3-game vs 10-game avg)
    for stat in ['saves', 'shots_against', 'goals_against']:
        short = f'{stat}_rolling_3'
        long = f'{stat}_rolling_10'
        if short in df.columns and long in df.columns:
            df[f'{stat}_momentum'] = df[short] - df[long]

    # Save percentage momentum
    sp_short = 'save_percentage_rolling_3'
    sp_long = 'save_percentage_rolling_10'
    if sp_short in df.columns and sp_long in df.columns:
        df['save_pct_momentum'] = df[sp_short] - df[sp_long]

    return df


def add_matchup_context_features(df):
    """Add features combining goalie performance with opponent/team context."""
    df = df.copy()

    # Expected workload: opponent shots tendency vs goalie's recent workload
    if 'opp_shots_rolling_5' in df.columns and 'shots_against_rolling_5' in df.columns:
        df['expected_workload_diff'] = df['opp_shots_rolling_5'] - df['shots_against_rolling_5']

    # Line vs opponent-implied saves (line relative to what opponent usually forces)
    if 'opp_shots_rolling_5' in df.columns and 'opp_goals_rolling_5' in df.columns:
        # Opponent usually forces this many saves on goalies
        opp_saves_implied = df['opp_shots_rolling_5'] - df['opp_goals_rolling_5']
        df['line_vs_opp_implied_saves'] = df['betting_line'] - opp_saves_implied

    # Rest advantage interaction: rest * recent performance
    if 'goalie_days_rest' in df.columns and 'saves_rolling_5' in df.columns:
        df['rest_x_performance'] = df['goalie_days_rest'].clip(upper=7) * df['saves_rolling_5']

    return df


def add_all_engineered_features(df):
    """Apply all feature engineering."""
    df = add_interaction_features(df)
    df = add_volatility_features(df)
    df = add_trend_features(df)
    df = add_matchup_context_features(df)
    return df


# ============================================================
# Evaluation Harness
# ============================================================

def evaluate_config(df, feature_cols, config_name, params, ev_threshold=0.12):
    """Train and evaluate a feature configuration. Returns metrics dict."""
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

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    trainer = ClassifierTrainer()
    trainer.model = model
    trainer.feature_names = feature_cols

    val_metrics = trainer.evaluate_profitability(
        X_val, y_val, df, val_idx, dataset_name='Val', ev_threshold=ev_threshold
    )
    test_metrics = trainer.evaluate_profitability(
        X_test, y_test, df, test_idx, dataset_name='Test', ev_threshold=ev_threshold
    )

    val_roi = val_metrics['roi']
    val_bets = val_metrics['total_bets']
    test_roi = test_metrics['roi']
    test_bets = test_metrics['total_bets']
    combined_bets = val_bets + test_bets
    combined_profit = val_metrics['total_profit'] + test_metrics['total_profit']
    combined_roi = (combined_profit / combined_bets) * 100 if combined_bets > 0 else 0

    # Get feature importance
    importance = model.get_booster().get_score(importance_type='gain')
    feat_importance = {}
    for feat_key, gain in importance.items():
        idx = int(feat_key.replace('f', ''))
        if idx < len(feature_cols):
            feat_importance[feature_cols[idx]] = gain

    return {
        'config_name': config_name,
        'n_features': len(feature_cols),
        'val_roi': val_roi,
        'val_bets': val_bets,
        'val_profit': val_metrics['total_profit'],
        'val_win_rate': val_metrics['win_rate'],
        'test_roi': test_roi,
        'test_bets': test_bets,
        'test_profit': test_metrics['total_profit'],
        'test_win_rate': test_metrics['win_rate'],
        'combined_roi': combined_roi,
        'combined_bets': combined_bets,
        'combined_profit': combined_profit,
        'feature_cols': feature_cols,
        'feature_importance': feat_importance,
        'model': model,
    }


# ============================================================
# Configurations to Test
# ============================================================

# Base hyperparameters (Config #4398)
BASE_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'max_depth': 4,
    'learning_rate': 0.02,
    'min_child_weight': 15,
    'gamma': 1.0,
    'reg_alpha': 10,
    'reg_lambda': 40,
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

results = []

# --- Config 1: Baseline (current 96 features) ---
print("\n" + "=" * 80)
print("[1/8] Baseline (current 96 features)")
print("=" * 80)
df1 = df_raw.copy()
feat1 = [c for c in df1.columns if c not in EXCLUDED_BASE]
r = evaluate_config(df1, feat1, "Baseline (96 features)", BASE_PARAMS)
results.append(r)
print(f"  Val: {r['val_roi']:+.2f}% ({r['val_bets']} bets) | Test: {r['test_roi']:+.2f}% ({r['test_bets']} bets) | Combined: {r['combined_roi']:+.2f}%")

# --- Config 2: Baseline + all new engineered features ---
print("\n" + "=" * 80)
print("[2/8] Baseline + interaction/volatility/trend/matchup features")
print("=" * 80)
df2 = add_all_engineered_features(df_raw.copy())
feat2 = [c for c in df2.columns if c not in EXCLUDED_BASE]
r = evaluate_config(df2, feat2, "Baseline + all engineered", BASE_PARAMS)
results.append(r)
new_feats = [f for f in feat2 if f not in feat1]
print(f"  New features ({len(new_feats)}): {new_feats}")
print(f"  Val: {r['val_roi']:+.2f}% ({r['val_bets']} bets) | Test: {r['test_roi']:+.2f}% ({r['test_bets']} bets) | Combined: {r['combined_roi']:+.2f}%")

# --- Config 3: Baseline + only interaction features ---
print("\n" + "=" * 80)
print("[3/8] Baseline + interaction features only")
print("=" * 80)
df3 = add_interaction_features(df_raw.copy())
feat3 = [c for c in df3.columns if c not in EXCLUDED_BASE]
r = evaluate_config(df3, feat3, "Baseline + interactions", BASE_PARAMS)
results.append(r)
print(f"  Val: {r['val_roi']:+.2f}% ({r['val_bets']} bets) | Test: {r['test_roi']:+.2f}% ({r['test_bets']} bets) | Combined: {r['combined_roi']:+.2f}%")

# --- Config 4: Baseline + trend features only ---
print("\n" + "=" * 80)
print("[4/8] Baseline + trend/momentum features only")
print("=" * 80)
df4 = add_trend_features(df_raw.copy())
feat4 = [c for c in df4.columns if c not in EXCLUDED_BASE]
r = evaluate_config(df4, feat4, "Baseline + trends", BASE_PARAMS)
results.append(r)
print(f"  Val: {r['val_roi']:+.2f}% ({r['val_bets']} bets) | Test: {r['test_roi']:+.2f}% ({r['test_bets']} bets) | Combined: {r['combined_roi']:+.2f}%")

# --- Config 5: Drop low-importance features (keep top 50) ---
print("\n" + "=" * 80)
print("[5/8] Feature selection: top 50 features by importance")
print("=" * 80)
baseline_importance = results[0]['feature_importance']
top_50 = sorted(baseline_importance, key=baseline_importance.get, reverse=True)[:50]
r = evaluate_config(df_raw.copy(), top_50, "Top 50 features", BASE_PARAMS)
results.append(r)
print(f"  Val: {r['val_roi']:+.2f}% ({r['val_bets']} bets) | Test: {r['test_roi']:+.2f}% ({r['test_bets']} bets) | Combined: {r['combined_roi']:+.2f}%")

# --- Config 6: Drop low-importance features (keep top 30) ---
print("\n" + "=" * 80)
print("[6/8] Feature selection: top 30 features by importance")
print("=" * 80)
top_30 = sorted(baseline_importance, key=baseline_importance.get, reverse=True)[:30]
r = evaluate_config(df_raw.copy(), top_30, "Top 30 features", BASE_PARAMS)
results.append(r)
print(f"  Val: {r['val_roi']:+.2f}% ({r['val_bets']} bets) | Test: {r['test_roi']:+.2f}% ({r['test_bets']} bets) | Combined: {r['combined_roi']:+.2f}%")

# --- Config 7: Best engineered features + deeper trees ---
print("\n" + "=" * 80)
print("[7/8] All engineered features + deeper trees (max_depth=5)")
print("=" * 80)
deeper_params = {**BASE_PARAMS, 'max_depth': 5, 'min_child_weight': 20, 'n_estimators': 1000}
df7 = add_all_engineered_features(df_raw.copy())
feat7 = [c for c in df7.columns if c not in EXCLUDED_BASE]
r = evaluate_config(df7, feat7, "Engineered + deeper trees", deeper_params)
results.append(r)
print(f"  Val: {r['val_roi']:+.2f}% ({r['val_bets']} bets) | Test: {r['test_roi']:+.2f}% ({r['test_bets']} bets) | Combined: {r['combined_roi']:+.2f}%")

# --- Config 8: All engineered + more regularization ---
print("\n" + "=" * 80)
print("[8/8] All engineered features + higher regularization")
print("=" * 80)
reg_params = {**BASE_PARAMS, 'reg_alpha': 20, 'reg_lambda': 60, 'gamma': 2.0, 'n_estimators': 1000}
df8 = add_all_engineered_features(df_raw.copy())
feat8 = [c for c in df8.columns if c not in EXCLUDED_BASE]
r = evaluate_config(df8, feat8, "Engineered + high reg", reg_params)
results.append(r)
print(f"  Val: {r['val_roi']:+.2f}% ({r['val_bets']} bets) | Test: {r['test_roi']:+.2f}% ({r['test_bets']} bets) | Combined: {r['combined_roi']:+.2f}%")


# ============================================================
# Results Summary
# ============================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"\n{'Config':<40} | {'Feats':>5} | {'Val ROI':>8} | {'V Bets':>6} | {'Test ROI':>8} | {'T Bets':>6} | {'Comb ROI':>8} | {'C Bets':>6}")
print("-" * 120)

for r in sorted(results, key=lambda x: x['combined_roi'], reverse=True):
    print(f"{r['config_name']:<40} | {r['n_features']:>5} | {r['val_roi']:>+7.2f}% | {r['val_bets']:>6} | {r['test_roi']:>+7.2f}% | {r['test_bets']:>6} | {r['combined_roi']:>+7.2f}% | {r['combined_bets']:>6}")

# Find best config
best = max(results, key=lambda x: x['combined_roi'])
print(f"\nBest config: {best['config_name']}")
print(f"  Combined ROI: {best['combined_roi']:+.2f}% ({best['combined_bets']} bets)")
print(f"  Val: {best['val_roi']:+.2f}% ({best['val_bets']} bets)")
print(f"  Test: {best['test_roi']:+.2f}% ({best['test_bets']} bets)")

# Show top features for best config
print(f"\nTop 20 features for best config:")
sorted_imp = sorted(best['feature_importance'].items(), key=lambda x: x[1], reverse=True)
for i, (feat, gain) in enumerate(sorted_imp[:20], 1):
    marker = ""
    if feat.startswith('line_vs_') or feat.startswith('line_z_'):
        marker = " [LINE-RELATIVE]"
    elif feat == 'betting_line':
        marker = " [BETTING LINE]"
    elif feat not in feat1:
        marker = " [NEW]"
    print(f"  {i:2d}. {feat:40s} gain={gain:>8.1f}{marker}")

# Save best model if it beats baseline
baseline = results[0]
if best['combined_roi'] > baseline['combined_roi'] and best['config_name'] != baseline['config_name']:
    print(f"\nBest config beats baseline by {best['combined_roi'] - baseline['combined_roi']:+.2f}% ROI")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = Path('models/trained') / f'optimized_v1_{timestamp}'
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'classifier_model.json'
    feature_path = model_dir / 'classifier_feature_names.json'
    metadata_path = model_dir / 'classifier_metadata.json'

    best['model'].get_booster().save_model(str(model_path))
    with open(feature_path, 'w') as f:
        json.dump(best['feature_cols'], f, indent=2)

    metadata = {
        'config_name': best['config_name'],
        'trained_date': datetime.now().isoformat(),
        'ev_threshold': EV_THRESHOLD,
        'num_features': best['n_features'],
        'val_roi': best['val_roi'],
        'val_bets': best['val_bets'],
        'test_roi': best['test_roi'],
        'test_bets': best['test_bets'],
        'combined_roi': best['combined_roi'],
        'combined_bets': best['combined_bets'],
        'features': best['feature_cols'],
        'improvement_over_baseline': best['combined_roi'] - baseline['combined_roi'],
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {model_dir}")
    print(f"To use this model, update predictor.py model_path to:")
    print(f"  {model_dir / 'classifier_model.json'}")

    # List any new features that need to be added to inference
    new_features = [f for f in best['feature_cols'] if f not in feat1]
    if new_features:
        print(f"\nNew features that need inference support ({len(new_features)}):")
        for f in new_features:
            print(f"  - {f}")
        print("\nYou will need to add computation for these in feature_calculator.py")
else:
    print(f"\nBaseline is already the best or tied. No new model saved.")

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)
