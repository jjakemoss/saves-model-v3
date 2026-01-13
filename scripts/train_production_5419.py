"""
Train Config #5419 on ALL data for production use with 4% EV threshold.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from src.models.classifier_trainer import ClassifierTrainer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Config #5419 hyperparameters
CONFIG_5419 = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'n_estimators': 800,
    'max_depth': 5,
    'learning_rate': 0.015,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 12,
    'gamma': 0.5,
    'reg_alpha': 10,
    'reg_lambda': 30,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 1
}

USE_SAMPLE_WEIGHTS = True
PRODUCTION_EV_THRESHOLD = 0.04  # 4% EV threshold

print("=" * 80)
print("TRAINING CONFIG #5419 FOR PRODUCTION")
print("=" * 80)
print(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nHyperparameters:")
for key, value in CONFIG_5419.items():
    print(f"  {key}: {value}")
print(f"  use_sample_weights: {USE_SAMPLE_WEIGHTS}")
print(f"\nProduction EV Threshold: {int(PRODUCTION_EV_THRESHOLD*100)}%")
print("=" * 80)

# Load data
logging.info("Loading data...")
data_path = 'data/processed/classification_training_data.parquet'
df = pd.read_parquet(data_path)
logging.info(f"Loaded {len(df)} samples")

# Remove market-derived features (data leakage)
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

logging.info(f"Samples with odds data: {len(df)}")

# Prepare features
exclude_cols = [
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

feature_cols = [col for col in df.columns if col not in exclude_cols]

# Use 80% for training, 20% for validation
n = len(df)
train_end = int(n * 0.8)

train_idx = np.arange(0, train_end)
val_idx = np.arange(train_end, n)

X = df[feature_cols].values
y = df['over_hit'].values

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

logging.info(f"Training on {len(X_train)} samples (80% of data)")
logging.info(f"Validation on {len(X_val)} samples (20% of data)")

# Initialize ClassifierTrainer
trainer = ClassifierTrainer()

# Calculate sample weights
train_weights = None
if USE_SAMPLE_WEIGHTS:
    logging.info("Calculating sample weights...")
    train_weights = trainer.calculate_sample_weights(df, train_idx)

# Train model
logging.info("Training XGBoost classifier on ALL available data...")
trainer.train(X_train, y_train, X_val, y_val, params=CONFIG_5419, sample_weight=train_weights)
logging.info("Training complete!")

# Evaluate on validation set at production threshold
print("\n" + "=" * 80)
print("VALIDATION SET PERFORMANCE AT 4% EV THRESHOLD")
print("=" * 80)

val_results = trainer.evaluate_profitability(
    X_val, y_val, df, val_idx,
    dataset_name='Validation', ev_threshold=PRODUCTION_EV_THRESHOLD
)

print(f"\nExpected performance in production:")
print(f"  Bets per season (est): {val_results['total_bets'] * 5:.0f} (assuming 5 seasons of data)")
print(f"  ROI: {val_results['roi']:+.2f}%")
print(f"  Win rate: {val_results['win_rate']*100:.1f}%")
print(f"  Expected profit per 100 bets: {val_results['roi']:.2f} units")

# Save the model
model_filename = f"models/trained/config_5419_ev4pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
logging.info(f"Saving model to {model_filename}...")
trainer.model.save_model(model_filename)

# Save metadata
metadata = {
    'config_name': 'Config #5419',
    'trained_date': datetime.now().isoformat(),
    'hyperparameters': CONFIG_5419,
    'use_sample_weights': USE_SAMPLE_WEIGHTS,
    'ev_threshold': PRODUCTION_EV_THRESHOLD,
    'training_samples': len(X_train),
    'validation_samples': len(X_val),
    'validation_roi': val_results['roi'],
    'validation_win_rate': val_results['win_rate'],
    'validation_total_bets': val_results['total_bets'],
    'features': feature_cols
}

import json
metadata_filename = model_filename.replace('.json', '_metadata.json')
with open(metadata_filename, 'w') as f:
    json.dump(metadata, f, indent=2)

logging.info(f"Metadata saved to {metadata_filename}")

print("\n" + "=" * 80)
print("PRODUCTION MODEL READY!")
print("=" * 80)
print(f"Model saved: {model_filename}")
print(f"Metadata saved: {metadata_filename}")
print(f"\nTo use in production:")
print(f"  1. Load model: model = xgb.Booster(); model.load_model('{model_filename}')")
print(f"  2. Use EV threshold: {int(PRODUCTION_EV_THRESHOLD*100)}%")
print(f"  3. Bet on OVER or UNDER when EV >= {int(PRODUCTION_EV_THRESHOLD*100)}%")
print("=" * 80)
