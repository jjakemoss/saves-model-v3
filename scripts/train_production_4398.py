"""
Train Config #4398 for production use with 2% EV threshold

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

# Config #4398 hyperparameters
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
print("=" * 80)
print(f"\nHyperparameters:")
for key, value in CONFIG_4398.items():
    print(f"  {key}: {value}")
print(f"\nEV Threshold: {EV_THRESHOLD*100:.0f}%")

# Load data
print("\n" + "=" * 80)
print("Loading data...")
trainer = ClassifierTrainer()
df = trainer.load_data('data/processed/classification_training_data.parquet')

print(f"Total samples: {len(df)}")
print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

# Split: 80% train, 20% validation
print("\n" + "=" * 80)
print("Splitting data: 80% train, 20% validation...")

# Sort by date for chronological split
df = df.sort_values('game_date')
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:]

print(f"Train: {len(train_df)} samples ({train_df['game_date'].min()} to {train_df['game_date'].max()})")
print(f"Val: {len(val_df)} samples ({val_df['game_date'].min()} to {val_df['game_date'].max()})")

# Prepare features
print("\n" + "=" * 80)
print("Preparing features...")

X_train, y_train, feature_names = trainer.prepare_features(train_df)
X_val, y_val, _ = trainer.prepare_features(val_df)

print(f"Features: {len(feature_names)}")
print(f"Train samples: {len(X_train)}")
print(f"Val samples: {len(X_val)}")

# Calculate sample weights (though use_sample_weights=False, we still need the array)
sample_weights_train = None
if CONFIG_4398['use_sample_weights']:
    sample_weights_train = trainer.calculate_sample_weights(train_df)
    print(f"Sample weights: mean={sample_weights_train.mean():.2f}, std={sample_weights_train.std():.2f}")
else:
    print("Sample weighting: DISABLED")

# Train model
print("\n" + "=" * 80)
print("Training model...")

trainer.train(
    X_train, y_train, X_val, y_val,
    sample_weight=sample_weights_train,
    params={
        'max_depth': CONFIG_4398['max_depth'],
        'learning_rate': CONFIG_4398['learning_rate'],
        'min_child_weight': CONFIG_4398['min_child_weight'],
        'gamma': CONFIG_4398['gamma'],
        'reg_alpha': CONFIG_4398['reg_alpha'],
        'reg_lambda': CONFIG_4398['reg_lambda'],
        'n_estimators': CONFIG_4398['n_estimators'],
        'subsample': CONFIG_4398['subsample'],
        'colsample_bytree': CONFIG_4398['colsample_bytree']
    }
)

print("\n[OK] Training complete")

# Evaluate on validation set
print("\n" + "=" * 80)
print(f"Evaluating on validation set (EV threshold = {EV_THRESHOLD*100:.0f}%)...")

# Get validation indices (last 20% chronologically)
val_idx = np.arange(split_idx, len(df))

val_metrics = trainer.evaluate_profitability(
    X_val, y_val, df, val_idx, dataset_name='Validation', ev_threshold=EV_THRESHOLD
)

# Extract metrics from dict
val_roi = val_metrics['roi']
val_win_rate = val_metrics['win_rate']
val_bets = val_metrics['total_bets']
val_profit = val_metrics['total_profit']

# Save model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'config_4398_ev2pct_{timestamp}.json'
model_path = Path('models/trained') / model_filename

print("\n" + "=" * 80)
print(f"Saving model to {model_path}...")

# Set feature names before saving
trainer.feature_names = feature_names

# Save model
trainer.save_model(model_path)

# Save metadata
metadata = {
    'config_name': 'Config #4398',
    'trained_date': datetime.now().isoformat(),
    'hyperparameters': {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'n_estimators': CONFIG_4398['n_estimators'],
        'max_depth': CONFIG_4398['max_depth'],
        'learning_rate': CONFIG_4398['learning_rate'],
        'subsample': CONFIG_4398['subsample'],
        'colsample_bytree': CONFIG_4398['colsample_bytree'],
        'min_child_weight': CONFIG_4398['min_child_weight'],
        'gamma': CONFIG_4398['gamma'],
        'reg_alpha': CONFIG_4398['reg_alpha'],
        'reg_lambda': CONFIG_4398['reg_lambda'],
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    },
    'use_sample_weights': CONFIG_4398['use_sample_weights'],
    'ev_threshold': EV_THRESHOLD,
    'training_samples': len(train_df),
    'validation_samples': len(val_df),
    'validation_roi': val_roi,
    'validation_win_rate': val_win_rate,
    'validation_total_bets': val_bets,
    'features': feature_names
}

metadata_path = model_path.parent / f'{model_path.stem}_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"[OK] Model saved: {model_path}")
print(f"[OK] Metadata saved: {metadata_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nModel: {model_filename}")
print(f"EV Threshold: {EV_THRESHOLD*100:.0f}%")
print(f"Validation ROI: {val_roi:+.2f}% ({val_bets} bets)")
print("\nExpected Performance (from tuning):")
print(f"  Val: +2.54% (363 bets)")
print(f"  Test: +0.62% (336 bets)")
print(f"  Combined: +1.60% (699 bets)")
print("\nNext steps:")
print(f"  1. Update BettingPredictor to use {model_filename}")
print(f"  2. Create feature order file: training_feature_order_config_4398.txt")
print(f"  3. Update BETTING_TRACKER_README.md with new model details")
print(f"  4. Test with: python scripts/test_config_4398_integration.py")
print("=" * 80)
