"""
Train classifier on multi-book training data with line-relative features.

This uses the same Config #4398 hyperparameters as baseline, but with:
- Multi-book training data (multiple bookmaker lines per goalie-game)
- 6 new line-relative features (line_vs_rolling_*, line_z_score_*)
- Total features: ~96 (90 original + 6 new)

Usage:
    python scripts/train_production_multibook.py
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

# Same hyperparameters as Config #4398
CONFIG = {
    'max_depth': 4,
    'learning_rate': 0.02,
    'min_child_weight': 15,
    'gamma': 1.0,
    'reg_alpha': 10,
    'reg_lambda': 40,
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'use_sample_weights': False
}

EV_THRESHOLD = 0.12

print("=" * 80)
print("TRAINING MULTI-BOOK MODEL WITH LINE-RELATIVE FEATURES")
print("=" * 80)
print(f"\nHyperparameters (same as Config #4398):")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print(f"\nEV Threshold: {EV_THRESHOLD*100:.0f}%")

# Load multi-book training data
print("\n" + "=" * 80)
print("Loading multi-book training data...")

data_path = 'data/processed/multibook_classification_training_data.parquet'
df = pd.read_parquet(data_path)
print(f"Loaded {len(df)} total samples")

# Remove market-derived features (same as original)
market_features = [
    'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
    'market_vig', 'impl_prob_over', 'impl_prob_under',
    'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
    'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low'
]
df = df.drop(columns=[col for col in market_features if col in df.columns], errors='ignore')

# Sort by date and filter to samples with odds
df = df.sort_values('game_date').reset_index(drop=True)
df = df[df['odds_over_american'].notna() & df['odds_under_american'].notna()].reset_index(drop=True)
print(f"Filtered to {len(df)} samples with odds data")
print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

# Show multi-book stats
if 'book_key' in df.columns:
    print(f"\nBooks in data:")
    for book, count in df['book_key'].value_counts().items():
        print(f"  {book}: {count} rows")

game_goalie_counts = df.groupby(['game_id', 'goalie_id']).size()
multi = game_goalie_counts[game_goalie_counts > 1]
print(f"\nGoalie-games with multiple lines: {len(multi)} / {len(game_goalie_counts)}")

# Define feature columns
excluded_cols = [
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
    'saves_margin', 'over_line',  # Alternative target column names
    '_game_date_str', '_lookup_key',  # Internal columns from data builder
]

feature_cols = [col for col in df.columns if col not in excluded_cols]
print(f"\nFeature columns: {len(feature_cols)}")

# Verify line-relative features are present
lr_features = [f'line_vs_rolling_{w}' for w in [3, 5, 10]] + \
              [f'line_z_score_{w}' for w in [3, 5, 10]]
present = [f for f in lr_features if f in feature_cols]
missing = [f for f in lr_features if f not in feature_cols]
print(f"Line-relative features present: {len(present)}/{len(lr_features)}")
if missing:
    print(f"  [WARNING] Missing: {missing}")

# Show all features
print("\nAll features:")
for i, col in enumerate(sorted(feature_cols), 1):
    print(f"  {i}. {col}")

# Split: 60/20/20 chronological
print("\n" + "=" * 80)
print("Splitting data: 60% train, 20% validation, 20% test...")

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

train_dates = df.iloc[train_idx]['game_date']
val_dates = df.iloc[val_idx]['game_date']
test_dates = df.iloc[test_idx]['game_date']

print(f"Train: {len(X_train)} samples ({train_dates.min()} to {train_dates.max()})")
print(f"Val: {len(X_val)} samples ({val_dates.min()} to {val_dates.max()})")
print(f"Test: {len(X_test)} samples ({test_dates.min()} to {test_dates.max()})")

# Verify no date overlap between splits
assert train_dates.max() <= val_dates.min(), "Train/Val date overlap!"
assert val_dates.max() <= test_dates.min(), "Val/Test date overlap!"
print("[OK] No date overlap between splits")

# Train model
print("\n" + "=" * 80)
print("Training model...")

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'max_depth': CONFIG['max_depth'],
    'learning_rate': CONFIG['learning_rate'],
    'min_child_weight': CONFIG['min_child_weight'],
    'gamma': CONFIG['gamma'],
    'reg_alpha': CONFIG['reg_alpha'],
    'reg_lambda': CONFIG['reg_lambda'],
    'n_estimators': CONFIG['n_estimators'],
    'subsample': CONFIG['subsample'],
    'colsample_bytree': CONFIG['colsample_bytree'],
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 1
}

model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True
)

# Store model in trainer for evaluation
trainer = ClassifierTrainer()
trainer.model = model
trainer.feature_names = feature_cols

print("\n[OK] Training complete")

# Evaluate
print("\n" + "=" * 80)
print(f"Evaluating (EV threshold = {EV_THRESHOLD*100:.0f}%)...")

val_metrics = trainer.evaluate_profitability(
    X_val, y_val, df, val_idx, dataset_name='Validation', ev_threshold=EV_THRESHOLD
)

test_metrics = trainer.evaluate_profitability(
    X_test, y_test, df, test_idx, dataset_name='Test', ev_threshold=EV_THRESHOLD
)

val_roi = val_metrics['roi']
val_bets = val_metrics['total_bets']
val_profit = val_metrics['total_profit']

test_roi = test_metrics['roi']
test_bets = test_metrics['total_bets']
test_profit = test_metrics['total_profit']

combined_bets = val_bets + test_bets
combined_profit = val_profit + test_profit
combined_roi = (combined_profit / combined_bets) * 100 if combined_bets > 0 else 0

# Check feature importance for line-relative features
print("\n" + "=" * 80)
print("Feature importance (top 20):")
importance = model.get_booster().get_score(importance_type='gain')
# Map feature indices to names
feat_importance = {}
for feat_key, gain in importance.items():
    idx = int(feat_key.replace('f', ''))
    if idx < len(feature_cols):
        feat_importance[feature_cols[idx]] = gain

sorted_importance = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
for i, (feat, gain) in enumerate(sorted_importance[:20], 1):
    marker = " <-- LINE-RELATIVE" if feat.startswith('line_vs_') or feat.startswith('line_z_') else ""
    marker = " <-- BETTING LINE" if feat == 'betting_line' else marker
    print(f"  {i:2d}. {feat:40s} gain={gain:.1f}{marker}")

# Show line-relative feature importance specifically
print("\nLine-related feature importance:")
for feat in ['betting_line'] + lr_features:
    gain = feat_importance.get(feat, 0)
    print(f"  {feat:25s} gain={gain:.1f}")

# Save model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_dir = Path('models/trained') / f'multibook_v1_{timestamp}'
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / 'classifier_model.json'
feature_path = model_dir / 'classifier_feature_names.json'
metadata_path = model_dir / 'classifier_metadata.json'

print("\n" + "=" * 80)
print(f"Saving model to {model_dir}...")

# Use Booster interface for saving (compatible with predictor.py which loads as Booster)
model.get_booster().save_model(str(model_path))

with open(feature_path, 'w') as f:
    json.dump(feature_cols, f, indent=2)

metadata = {
    'config_name': 'Multibook V1 (line-relative features)',
    'base_config': 'Config #4398',
    'trained_date': datetime.now().isoformat(),
    'hyperparameters': params,
    'ev_threshold': EV_THRESHOLD,
    'training_samples': len(X_train),
    'validation_samples': len(X_val),
    'test_samples': len(X_test),
    'total_samples': len(df),
    'validation_roi': val_roi,
    'validation_total_bets': val_bets,
    'test_roi': test_roi,
    'test_total_bets': test_bets,
    'combined_roi': combined_roi,
    'combined_total_bets': combined_bets,
    'num_features': len(feature_cols),
    'features': feature_cols,
    'line_relative_features': lr_features,
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"[OK] Model saved: {model_path}")
print(f"[OK] Features saved: {feature_path}")
print(f"[OK] Metadata saved: {metadata_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nModel directory: {model_dir}")
print(f"Feature count: {len(feature_cols)}")
print(f"EV Threshold: {EV_THRESHOLD*100:.0f}%")
print(f"\nResults:")
print(f"  Validation ROI: {val_roi:+.2f}% ({val_bets} bets)")
print(f"  Test ROI: {test_roi:+.2f}% ({test_bets} bets)")
print(f"  Combined ROI: {combined_roi:+.2f}% ({combined_bets} bets)")
print(f"\nBaseline (Config #4398, single-book):")
print(f"  Val: +2.54% (363 bets)")
print(f"  Test: +0.62% (336 bets)")
print(f"  Combined: +1.60% (699 bets)")
print(f"\nNext steps:")
print(f"  1. Update BettingPredictor default paths to: {model_dir}")
print(f"  2. Run: python scripts/fetch_and_predict.py -v")
print(f"  3. Verify different lines produce different P(over) for same goalie")
print("=" * 80)
