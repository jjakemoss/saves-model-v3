"""
Train Config #4398 for production use with 2% EV threshold

CRITICAL: This script uses the EXACT same feature preparation as tune_comprehensive.py
to ensure the model matches the tuning results.

Config #4398 emerged as the best performer at 2% EV threshold:
- Val: +2.54% ROI (363 bets)
- Test: +0.62% ROI (336 bets)
- Combined: +1.60% ROI (699 bets)
- More volume than Config #5419 @ 4% (699 vs 581 bets)

Hyperparameters:
- weights: False (no sample weighting)
- max_depth: 4
- min_child_weight: 15
- gamma: 1.0
- learning_rate: 0.02
- reg_alpha: 10
- reg_lambda: 40
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

# Config #4398 hyperparameters (exactly as tuned)
CONFIG_4398 = {
    'max_depth': 4,
    'learning_rate': 0.02,
    'min_child_weight': 15,
    'gamma': 1.0,
    'reg_alpha': 10,
    'reg_lambda': 40,
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'use_sample_weights': False  # No sample weighting
}

EV_THRESHOLD = 0.02  # 2%

print("=" * 80)
print("TRAINING CONFIG #4398 FOR PRODUCTION")
print("Using EXACT same features as tune_comprehensive.py (90 features)")
print("=" * 80)
print(f"\nHyperparameters:")
for key, value in CONFIG_4398.items():
    print(f"  {key}: {value}")
print(f"\nEV Threshold: {EV_THRESHOLD*100:.0f}%")

# Load data - EXACTLY as tune_comprehensive.py does
print("\n" + "=" * 80)
print("Loading data (matching tune_comprehensive.py exactly)...")

data_path = 'data/processed/classification_training_data.parquet'
df = pd.read_parquet(data_path)
print(f"Loaded {len(df)} total samples")

# Step 1: Remove market-derived features (EXACTLY as tune_comprehensive.py line 22-29)
market_features = [
    'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
    'market_vig', 'impl_prob_over', 'impl_prob_under',
    'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
    'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low'
]
df = df.drop(columns=[col for col in market_features if col in df.columns], errors='ignore')
print(f"Removed {len(market_features)} market-derived features")

# Step 2: Sort by date and filter to samples with odds (EXACTLY as tune_comprehensive.py line 31-33)
df = df.sort_values('game_date').reset_index(drop=True)
df = df[df['odds_over_american'].notna() & df['odds_under_american'].notna()].reset_index(drop=True)
print(f"Filtered to {len(df)} samples with odds data")
print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

# Step 3: Define feature columns (EXACTLY as tune_comprehensive.py line 59-69)
excluded_cols = [
    'game_id', 'goalie_id', 'game_date', 'over_hit',
    'odds_over_american', 'odds_under_american',
    'odds_over_decimal', 'odds_under_decimal', 'num_books',
    'team_abbrev', 'opponent_team', 'toi', 'season',
    'saves', 'shots_against', 'goals_against', 'save_percentage',
    'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
    'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
    'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
    'team_goals', 'team_shots', 'opp_goals', 'opp_shots', 'line_margin'
]

feature_cols = [col for col in df.columns if col not in excluded_cols]
print(f"\nFeature columns: {len(feature_cols)}")

# Verify we have exactly 90 features (same as tuning)
if len(feature_cols) != 90:
    print(f"WARNING: Expected 90 features, got {len(feature_cols)}")
    print("Feature list:")
    for i, col in enumerate(sorted(feature_cols), 1):
        print(f"  {i}. {col}")
else:
    print("Feature count matches tuning (90 features)")

# Split: 60/20/20 (same as tuning to verify results match)
print("\n" + "=" * 80)
print("Splitting data: 60% train, 20% validation, 20% test (same as tuning)...")

n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

# Prepare features and target
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

# No sample weights for Config #4398
print("\nSample weighting: DISABLED (as per Config #4398)")

# Train model
print("\n" + "=" * 80)
print("Training model...")

trainer = ClassifierTrainer()

# Build params dict
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'max_depth': CONFIG_4398['max_depth'],
    'learning_rate': CONFIG_4398['learning_rate'],
    'min_child_weight': CONFIG_4398['min_child_weight'],
    'gamma': CONFIG_4398['gamma'],
    'reg_alpha': CONFIG_4398['reg_alpha'],
    'reg_lambda': CONFIG_4398['reg_lambda'],
    'n_estimators': CONFIG_4398['n_estimators'],
    'subsample': CONFIG_4398['subsample'],
    'colsample_bytree': CONFIG_4398['colsample_bytree'],
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 1
}

print(f"Hyperparameters: {params}")

# Train using XGBClassifier directly (same as trainer.train does internally)
model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True
)

# Store model in trainer for evaluation
trainer.model = model
trainer.feature_names = feature_cols

print("\n[OK] Training complete")

# Evaluate on validation and test sets
print("\n" + "=" * 80)
print(f"Evaluating on validation and test sets (EV threshold = {EV_THRESHOLD*100:.0f}%)...")

val_metrics = trainer.evaluate_profitability(
    X_val, y_val, df, val_idx, dataset_name='Validation', ev_threshold=EV_THRESHOLD
)

test_metrics = trainer.evaluate_profitability(
    X_test, y_test, df, test_idx, dataset_name='Test', ev_threshold=EV_THRESHOLD
)

# Extract metrics from dict
val_roi = val_metrics['roi']
val_win_rate = val_metrics['win_rate']
val_bets = val_metrics['total_bets']
val_profit = val_metrics['total_profit']

test_roi = test_metrics['roi']
test_win_rate = test_metrics['win_rate']
test_bets = test_metrics['total_bets']
test_profit = test_metrics['total_profit']

# Calculate combined metrics
combined_bets = val_bets + test_bets
combined_profit = val_profit + test_profit
combined_roi = (combined_profit / combined_bets) * 100 if combined_bets > 0 else 0

# Save model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_dir = Path('models/trained') / f'config_4398_ev2pct_{timestamp}'
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / 'classifier_model.json'
feature_path = model_dir / 'classifier_feature_names.json'
metadata_path = model_dir / 'classifier_metadata.json'

print("\n" + "=" * 80)
print(f"Saving model to {model_dir}...")

# Save model
model.save_model(str(model_path))

# Save feature names (the 90 features used)
with open(feature_path, 'w') as f:
    json.dump(feature_cols, f, indent=2)

# Save metadata
metadata = {
    'config_name': 'Config #4398',
    'trained_date': datetime.now().isoformat(),
    'hyperparameters': params,
    'use_sample_weights': CONFIG_4398['use_sample_weights'],
    'ev_threshold': EV_THRESHOLD,
    'training_samples': len(X_train),
    'validation_samples': len(X_val),
    'test_samples': len(X_test),
    'validation_roi': val_roi,
    'validation_win_rate': val_win_rate,
    'validation_total_bets': val_bets,
    'test_roi': test_roi,
    'test_win_rate': test_win_rate,
    'test_total_bets': test_bets,
    'combined_roi': combined_roi,
    'combined_total_bets': combined_bets,
    'num_features': len(feature_cols),
    'features': feature_cols
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
print(f"Feature count: {len(feature_cols)} (matches tuning)")
print(f"EV Threshold: {EV_THRESHOLD*100:.0f}%")
print(f"\nActual Results (60/20/20 split):")
print(f"  Validation ROI: {val_roi:+.2f}% ({val_bets} bets)")
print(f"  Test ROI: {test_roi:+.2f}% ({test_bets} bets)")
print(f"  Combined ROI: {combined_roi:+.2f}% ({combined_bets} bets)")
print(f"\nExpected Performance (from tuning with 60/20/20 split):")
print(f"  Val: +2.54% (363 bets)")
print(f"  Test: +0.62% (336 bets)")
print(f"  Combined: +1.60% (699 bets)")
print("\nNext steps:")
print(f"  1. Update BettingPredictor default paths to: {model_dir}")
print(f"  2. Verify feature calculator produces same 90 features")
print(f"  3. Run generate_predictions.py to test")
print("=" * 80)
