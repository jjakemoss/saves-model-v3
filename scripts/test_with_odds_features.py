"""
Quick test: What happens if we include betting odds as features?

This tests whether including the juice (e.g., -110, -115) as features
helps or hurts the model's ability to find +EV bets.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.models.classifier_trainer import ClassifierTrainer

def test_with_and_without_odds():
    """Compare model performance with and without odds as features."""

    print("="*80)
    print("TESTING: Odds as Features vs No Odds as Features")
    print("="*80)

    # Load data
    data_path = 'data/processed/classification_training_data.parquet'
    df = pd.read_parquet(data_path)

    # Remove market features (they hurt performance)
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

    print(f"\nLoaded {len(df)} samples with odds data")

    # Create chronological splits (60/20/20)
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    print(f"Train: {len(train_idx)} samples")
    print(f"Validation: {len(val_idx)} samples")
    print(f"Test: {len(test_idx)} samples\n")

    # Base exclusions
    base_exclusions = [
        'game_id', 'goalie_id', 'game_date', 'over_hit',
        'odds_over_decimal', 'odds_under_decimal', 'num_books',
        'team_abbrev', 'opponent_team', 'toi', 'season',
        # CRITICAL: Exclude actual game results to prevent data leakage
        'saves', 'shots_against', 'goals_against', 'save_percentage',
        'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
        'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
        'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
        'team_goals', 'team_shots', 'opp_goals', 'opp_shots', 'line_margin'
    ]

    # Test 1: WITHOUT odds as features (current approach)
    print("="*80)
    print("TEST 1: WITHOUT ODDS AS FEATURES (baseline)")
    print("="*80)

    feature_cols_without = [col for col in df.columns if col not in
                           base_exclusions + ['odds_over_american', 'odds_under_american']]

    print(f"Features used: {len(feature_cols_without)}")

    X_without = df[feature_cols_without].values
    y = df['over_hit'].values

    trainer_without = ClassifierTrainer()

    X_train = X_without[train_idx]
    X_val = X_without[val_idx]
    X_test = X_without[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    train_weights = trainer_without.calculate_sample_weights(df, train_idx)

    print("\nTraining model WITHOUT odds features...")
    trainer_without.train(X_train, y_train, X_val, y_val, sample_weight=train_weights)

    print("\nEvaluating on test set at multiple EV thresholds...")
    results_without = trainer_without.test_ev_thresholds(
        X_test, y_test, df, test_idx,
        dataset_name='Test',
        thresholds=[0.01, 0.02, 0.03, 0.05]
    )

    # Test 2: WITH odds as features
    print("\n" + "="*80)
    print("TEST 2: WITH ODDS AS FEATURES (experimental)")
    print("="*80)

    feature_cols_with = [col for col in df.columns if col not in base_exclusions]

    print(f"Features used: {len(feature_cols_with)} (includes odds_over_american, odds_under_american)")

    X_with = df[feature_cols_with].values

    trainer_with = ClassifierTrainer()

    X_train_with = X_with[train_idx]
    X_val_with = X_with[val_idx]
    X_test_with = X_with[test_idx]

    train_weights_with = trainer_with.calculate_sample_weights(df, train_idx)

    print("\nTraining model WITH odds features...")
    trainer_with.train(X_train_with, y_train, X_val_with, y_val, sample_weight=train_weights_with)

    print("\nEvaluating on test set at multiple EV thresholds...")
    results_with = trainer_with.test_ev_thresholds(
        X_test_with, y_test, df, test_idx,
        dataset_name='Test',
        thresholds=[0.01, 0.02, 0.03, 0.05]
    )

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON: WITHOUT vs WITH Odds as Features")
    print("="*80)
    print("\n{:<15} {:<20} {:<20}".format("EV Threshold", "WITHOUT Odds", "WITH Odds"))
    print("-"*80)

    for threshold in [0.01, 0.02, 0.03, 0.05]:
        without_metrics = results_without[threshold]
        with_metrics = results_with[threshold]

        without_str = f"ROI: {without_metrics['roi']:>6.2f}% | Bets: {without_metrics['total_bets']:>3}"
        with_str = f"ROI: {with_metrics['roi']:>6.2f}% | Bets: {with_metrics['total_bets']:>3}"

        print(f"{threshold*100:>3.0f}%          {without_str:<20} {with_str:<20}")

    # Find best for each
    best_without_roi = max([m['roi'] for m in results_without.values() if m['total_bets'] > 0], default=-100)
    best_with_roi = max([m['roi'] for m in results_with.values() if m['total_bets'] > 0], default=-100)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Best ROI WITHOUT odds features: {best_without_roi:.2f}%")
    print(f"Best ROI WITH odds features:    {best_with_roi:.2f}%")

    if best_without_roi > best_with_roi:
        print(f"\nWINNER: WITHOUT odds (by {best_without_roi - best_with_roi:.2f}%)")
        print("\nThis suggests the model performs better when it makes independent")
        print("predictions rather than learning from the market's odds.")
    elif best_with_roi > best_without_roi:
        print(f"\nWINNER: WITH odds (by {best_with_roi - best_without_roi:.2f}%)")
        print("\nThis suggests including odds as features helps the model identify")
        print("market inefficiencies and find better betting opportunities.")
    else:
        print("\nRESULT: TIE - No significant difference")

if __name__ == '__main__':
    test_with_and_without_odds()
