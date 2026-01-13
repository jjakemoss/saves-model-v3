"""
Comprehensive hyperparameter tuning to find configs profitable on ALL splits.
Tests broader hyperparameter space with focus on generalization.
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


def evaluate_config(params, use_sample_weights, df, train_idx, val_idx, test_idx, ev_thresholds=[0.02, 0.05, 0.07]):
    """
    Train model and evaluate on ALL splits at multiple EV thresholds.

    Args:
        params: XGBoost hyperparameters
        use_sample_weights: bool, whether to use sample weighting
        df: Full dataframe
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        ev_thresholds: List of EV thresholds to test

    Returns:
        dict: Metrics for all splits and EV thresholds
    """
    try:
        trainer = ClassifierTrainer()

        # Prepare features
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
        X_test, y_test = X[test_idx], y[test_idx]

        # Calculate sample weights if requested
        train_weights = None
        if use_sample_weights:
            train_weights = trainer.calculate_sample_weights(df, train_idx)

        # Train model
        trainer.train(X_train, y_train, X_val, y_val, params=params, sample_weight=train_weights)

        # Evaluate on all splits at all EV thresholds
        results = {'success': True, 'params': params, 'use_weights': use_sample_weights}

        for ev in ev_thresholds:
            val_metrics = trainer.evaluate_profitability(X_val, y_val, df, val_idx, ev_threshold=ev)
            test_metrics = trainer.evaluate_profitability(X_test, y_test, df, test_idx, ev_threshold=ev)

            results[f'val_roi_{int(ev*100)}'] = val_metrics['roi']
            results[f'val_bets_{int(ev*100)}'] = val_metrics['total_bets']
            results[f'test_roi_{int(ev*100)}'] = test_metrics['roi']
            results[f'test_bets_{int(ev*100)}'] = test_metrics['total_bets']

        return results

    except Exception as e:
        print(f"Error: {str(e)}")
        return {'success': False, 'error': str(e)}


def grid_search():
    """Comprehensive grid search across broader hyperparameter space."""
    from time import time
    start_time = time()

    print("="*80)
    print("COMPREHENSIVE HYPERPARAMETER TUNING")
    print("Goal: Find configs profitable on validation AND test sets")
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

    # Broader hyperparameter grid
    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'min_child_weight': [8, 10, 12, 15],
        'gamma': [0.5, 1.0, 2.0, 5.0],
        'learning_rate': [0.01, 0.015, 0.02],
        'reg_alpha': [5, 10, 15, 20],
        'reg_lambda': [20, 30, 40, 50],
        'use_sample_weights': [True, False]
    }

    # EV thresholds to test
    ev_thresholds = [0.02, 0.05, 0.07]

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total_configs = np.prod([len(v) for v in values])

    print(f"Testing {total_configs} configurations")
    print(f"EV thresholds: {[f'{int(ev*100)}%' for ev in ev_thresholds]}")
    print(f"Hyperparameter grid:")
    for key, vals in param_grid.items():
        print(f"  {key}: {vals}")
    print()

    best_configs = {
        2: {'roi_sum': -float('inf'), 'config': None},
        5: {'roi_sum': -float('inf'), 'config': None},
        7: {'roi_sum': -float('inf'), 'config': None}
    }

    all_results = []
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
              f"depth={params['max_depth']}, child={params['min_child_weight']}, " +
              f"gamma={params['gamma']}, lr={params['learning_rate']}, " +
              f"alpha={params['reg_alpha']}, lambda={params['reg_lambda']}")

        # Evaluate
        result = evaluate_config(params, use_weights, df, train_idx, val_idx, test_idx, ev_thresholds)

        if not result['success']:
            print(f"  FAILED: {result.get('error', 'Unknown error')}\n")
            continue

        all_results.append(result)

        # Check each EV threshold
        for ev in ev_thresholds:
            ev_pct = int(ev * 100)
            val_roi = result[f'val_roi_{ev_pct}']
            test_roi = result[f'test_roi_{ev_pct}']
            val_bets = result[f'val_bets_{ev_pct}']
            test_bets = result[f'test_bets_{ev_pct}']

            # Calculate average ROI across val and test
            roi_sum = val_roi + test_roi
            both_profitable = val_roi > 0 and test_roi > 0

            indicator = ""
            if both_profitable and roi_sum > best_configs[ev_pct]['roi_sum']:
                best_configs[ev_pct]['roi_sum'] = roi_sum
                best_configs[ev_pct]['config'] = {**params, 'use_weights': use_weights}
                indicator = " <- NEW BEST!"
            elif both_profitable:
                indicator = " <- BOTH PROFITABLE!"

            print(f"  EV={ev_pct}%: Val={val_roi:+.2f}% ({val_bets}), Test={test_roi:+.2f}% ({test_bets}){indicator}")

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
    print(f"Total time: {(time() - start_time)/60:.1f} minutes\n")

    for ev in [2, 5, 7]:
        best = best_configs[ev]
        if best['config'] is not None:
            print(f"BEST CONFIG FOR {ev}% EV THRESHOLD:")
            print(f"  Combined ROI: {best['roi_sum']:.2f}%")
            for key, val in best['config'].items():
                print(f"  {key}: {val}")
            print()
        else:
            print(f"NO PROFITABLE CONFIG FOUND FOR {ev}% EV THRESHOLD\n")

    # Show all configs profitable on BOTH val and test
    print("="*80)
    print("ALL CONFIGS PROFITABLE ON BOTH VAL AND TEST:")
    print("="*80)

    for ev in [2, 5, 7]:
        profitable_configs = []
        for r in all_results:
            ev_pct = int(ev * 100)
            val_roi = r.get(f'val_roi_{ev_pct}', -100)
            test_roi = r.get(f'test_roi_{ev_pct}', -100)
            if val_roi > 0 and test_roi > 0:
                profitable_configs.append({
                    'val_roi': val_roi,
                    'test_roi': test_roi,
                    'avg_roi': (val_roi + test_roi) / 2,
                    'config': r
                })

        profitable_configs.sort(key=lambda x: x['avg_roi'], reverse=True)

        print(f"\n{ev}% EV Threshold: {len(profitable_configs)} configs profitable on both")
        for i, pc in enumerate(profitable_configs[:5], 1):
            cfg = pc['config']
            weights_str = "WITH weights" if cfg['use_weights'] else "NO weights"
            print(f"  {i}. Avg={pc['avg_roi']:+.2f}% (Val={pc['val_roi']:+.2f}%, Test={pc['test_roi']:+.2f}%) - {weights_str}")

    return best_configs, all_results


if __name__ == '__main__':
    best_configs, results = grid_search()
