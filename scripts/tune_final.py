"""
Final hyperparameter tuning with sample weights as a tunable parameter.
This script tests whether sample weights help or hurt profitability.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
from src.models.classifier_trainer import ClassifierTrainer


def load_training_data():
    """Load and prepare data exactly as production does."""
    data_path = 'data/processed/classification_training_data.parquet'
    df = pd.read_parquet(data_path)

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

    print(f"Loaded {len(df)} samples with odds data")
    return df


def evaluate_config(params, use_sample_weights, df, train_idx, val_idx, ev_threshold=0.02):
    """
    Train model and evaluate on validation ROI.

    Args:
        params: XGBoost hyperparameters
        use_sample_weights: bool, whether to use sample weighting
        df: Full dataframe
        train_idx: Training indices
        val_idx: Validation indices
        ev_threshold: Minimum EV required to place bet

    Returns:
        dict: Validation metrics
    """
    try:
        trainer = ClassifierTrainer()

        # Prepare features (exclude actual results and odds)
        feature_cols = [col for col in df.columns if col not in [
            'game_id', 'goalie_id', 'game_date', 'over_hit',
            'odds_over_american', 'odds_under_american',
            'odds_over_decimal', 'odds_under_decimal', 'num_books',
            'team_abbrev', 'opponent_team', 'toi', 'season',
            'saves', 'shots_against', 'goals_against', 'save_percentage',
            'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
            'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
            'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
            'team_goals', 'team_shots', 'opp_goals', 'opp_shots', 'line_margin'
        ]]

        X = df[feature_cols].values
        y = df['over_hit'].values

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Calculate sample weights if requested
        train_weights = None
        if use_sample_weights:
            train_weights = trainer.calculate_sample_weights(df, train_idx)

        # Train model
        trainer.train(X_train, y_train, X_val, y_val, params=params, sample_weight=train_weights)

        # Evaluate on validation set
        val_metrics = trainer.evaluate_profitability(
            X_val, y_val, df, val_idx,
            dataset_name='Validation',
            ev_threshold=ev_threshold
        )

        return {
            'roi': val_metrics['roi'],
            'total_bets': val_metrics['total_bets'],
            'win_rate': val_metrics['win_rate'],
            'total_profit': val_metrics['total_profit'],
            'success': True
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'roi': -100.0,
            'total_bets': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'success': False,
            'error': str(e)
        }


def grid_search():
    """Perform grid search including sample weights as a parameter."""
    from time import time
    start_time = time()

    print("="*80)
    print("FINAL HYPERPARAMETER TUNING")
    print("Testing sample weights as a hyperparameter")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    df = load_training_data()

    # Create chronological splits (60/20/20)
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    print(f"Train: {len(train_idx)} samples")
    print(f"Val: {len(val_idx)} samples")
    print(f"Test: {len(test_idx)} samples\n")

    # Hyperparameter grid (smaller grid focused on promising regions)
    param_grid = {
        'max_depth': [3, 4],
        'min_child_weight': [10, 15],
        'gamma': [1.0, 5.0],
        'learning_rate': [0.015, 0.02],
        'reg_alpha': [10, 15],
        'reg_lambda': [30, 40],
        'use_sample_weights': [True, False]  # NEW: Test with and without weights
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total_configs = np.prod([len(v) for v in values])

    print(f"Testing {total_configs} configurations")
    print(f"Hyperparameter grid:")
    for key, vals in param_grid.items():
        print(f"  {key}: {vals}")
    print()

    best_roi = -float('inf')
    best_config = None
    results = []

    config_num = 0
    for combo in product(*values):
        config_num += 1
        config_start = time()

        # Parse config
        config = dict(zip(keys, combo))
        use_weights = config.pop('use_sample_weights')

        # Build params dict
        params = {k: v for k, v in config.items()}

        print(f"[{config_num}/{total_configs}] Testing: weights={use_weights}, " +
              f"depth={params['max_depth']}, child_wt={params['min_child_weight']}, " +
              f"gamma={params['gamma']}, lr={params['learning_rate']}, " +
              f"alpha={params['reg_alpha']}, lambda={params['reg_lambda']}")

        # Evaluate
        result = evaluate_config(params, use_weights, df, train_idx, val_idx, ev_threshold=0.02)

        # Track results
        result['config'] = config
        result['use_sample_weights'] = use_weights
        results.append(result)

        # Print result
        roi_indicator = " <- NEW BEST!" if result['roi'] > best_roi else ""
        print(f"  Val ROI: {result['roi']:.2f}% | Bets: {result['total_bets']} | " +
              f"Win Rate: {result['win_rate']:.1f}%{roi_indicator}")

        # Update best
        if result['roi'] > best_roi:
            best_roi = result['roi']
            best_config = {**params, 'use_sample_weights': use_weights}

        # Timing
        config_time = time() - config_start
        elapsed = time() - start_time
        avg_time = elapsed / config_num
        remaining = avg_time * (total_configs - config_num)

        print(f"  Time: {config_time:.1f}s | Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min\n")

    # Final summary
    print("\n" + "="*80)
    print("TUNING COMPLETE")
    print("="*80)
    print(f"Total time: {(time() - start_time)/60:.1f} minutes")
    print(f"\nBest configuration:")
    print(f"  Val ROI: {best_roi:.2f}%")
    for key, val in best_config.items():
        print(f"  {key}: {val}")

    # Show top 10
    results_sorted = sorted(results, key=lambda x: x['roi'], reverse=True)
    print(f"\nTop 10 configurations:")
    for i, r in enumerate(results_sorted[:10], 1):
        cfg = r['config']
        weights_str = "WITH weights" if r['use_sample_weights'] else "NO weights"
        print(f"{i}. ROI={r['roi']:+.2f}% ({r['total_bets']} bets) - {weights_str} - " +
              f"depth={cfg['max_depth']}, child={cfg['min_child_weight']}, " +
              f"gamma={cfg['gamma']}, lr={cfg['learning_rate']}")

    return best_config, results


if __name__ == '__main__':
    best_config, results = grid_search()
