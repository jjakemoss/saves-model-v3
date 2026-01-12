"""
Hyperparameter tuning WITH odds as features (experimental approach).
Evaluates configurations based on validation ROI (not accuracy).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from itertools import product
from src.models.classifier_trainer import ClassifierTrainer

def load_training_data():
    """Load the classification training data."""
    data_path = 'data/processed/classification_training_data.parquet'
    df = pd.read_parquet(data_path)

    # Remove market features if they exist (they hurt performance)
    market_features = [
        'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
        'market_vig', 'impl_prob_over', 'impl_prob_under',
        'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
        'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low'
    ]
    df = df.drop(columns=[col for col in market_features if col in df.columns], errors='ignore')

    # Sort by date for chronological split
    df = df.sort_values('game_date').reset_index(drop=True)

    # Filter to samples with odds
    df = df[df['odds_over_american'].notna() & df['odds_under_american'].notna()].reset_index(drop=True)

    print(f"Loaded {len(df)} samples with odds data")

    return df

def evaluate_config(params, df, train_idx, val_idx, ev_threshold=0.02):
    """
    Train model with given params and evaluate on validation ROI.
    Returns validation ROI and other metrics.
    """
    try:
        # Create trainer
        trainer = ClassifierTrainer()

        # Prepare data - INCLUDE odds as features
        # CRITICAL: Exclude actual game results to prevent data leakage
        feature_cols = [col for col in df.columns if col not in [
            'game_id', 'goalie_id', 'game_date', 'over_hit',
            # odds_over_american and odds_under_american ARE INCLUDED (testing with odds)
            'odds_over_decimal', 'odds_under_decimal', 'num_books',
            'team_abbrev', 'opponent_team', 'toi', 'season',
            # EXCLUDE ACTUAL GAME RESULTS (data leakage):
            'saves', 'shots_against', 'goals_against', 'save_percentage',
            'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
            'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
            'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
            'team_goals', 'team_shots', 'opp_goals', 'opp_shots', 'line_margin'
        ]]

        print(f"  Features: {len(feature_cols)} (WITH odds)")

        X = df[feature_cols].values
        y = df['over_hit'].values

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Calculate sample weights for training
        train_weights = trainer.calculate_sample_weights(df, train_idx)

        # Train model with custom params
        trainer.train(
            X_train, y_train, X_val, y_val,
            params=params,
            sample_weight=train_weights
        )

        # Evaluate on validation set
        val_metrics = trainer.evaluate_profitability(
            X_val, y_val, df, val_idx,
            dataset_name='Validation',
            ev_threshold=ev_threshold
        )

        # Also get basic accuracy metrics
        y_pred = trainer.model.predict(X_val)
        accuracy = np.mean(y_pred == y_val)

        return {
            'roi': val_metrics['roi'],
            'total_bets': val_metrics['total_bets'],
            'win_rate': val_metrics['win_rate'],
            'total_profit': val_metrics['total_profit'],
            'accuracy': accuracy,
            'success': True
        }

    except Exception as e:
        print(f"Error with params {params}: {str(e)}")
        return {
            'roi': -100.0,
            'total_bets': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'accuracy': 0.0,
            'success': False,
            'error': str(e)
        }

def grid_search():
    """
    Perform grid search over hyperparameter space.
    """
    from time import time
    start_time = time()

    print("="*80)
    print("HYPERPARAMETER TUNING: WITH ODDS AS FEATURES")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
    print(f"Validation: {len(val_idx)} samples")
    print(f"Test: {len(test_idx)} samples")

    # Define parameter grid
    param_grid = {
        'max_depth': [3, 4, 5],
        'min_child_weight': [10, 15, 20, 25],
        'gamma': [1.0, 5.0, 10.0, 15.0],
        'learning_rate': [0.01, 0.015, 0.02],
        'reg_alpha': [5, 10, 15, 20],
        'reg_lambda': [10, 20, 30, 40],
    }

    # Fixed params
    fixed_params = {
        'n_estimators': 800,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'tree_method': 'hist',
    }

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    total_configs = len(all_combinations)
    print(f"\nTesting {total_configs} configurations WITH odds features...")
    print(f"Evaluating on validation ROI at 2% EV threshold\n")

    # Store all results
    all_results = []
    best_so_far = {'roi': -float('inf'), 'config_num': None}

    # Test each configuration
    for i, combo in enumerate(all_combinations, 1):
        config_start = time()
        # Create params dict
        params = fixed_params.copy()
        for param_name, param_value in zip(param_names, combo):
            params[param_name] = param_value

        print(f"[{i}/{total_configs}] Testing: max_depth={params['max_depth']}, "
              f"min_child_weight={params['min_child_weight']}, "
              f"gamma={params['gamma']}, "
              f"lr={params['learning_rate']}, "
              f"alpha={params['reg_alpha']}, "
              f"lambda={params['reg_lambda']}")

        # Evaluate this configuration
        result = evaluate_config(params, df, train_idx, val_idx, ev_threshold=0.02)

        # Store result
        result['params'] = {k: v for k, v in params.items() if k in param_names}
        result['config_num'] = i
        all_results.append(result)

        config_time = time() - config_start
        elapsed = time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (total_configs - i)

        if result['success']:
            roi_indicator = "NEW BEST!" if result['roi'] > best_so_far['roi'] else ""
            if result['roi'] > best_so_far['roi']:
                best_so_far = {'roi': result['roi'], 'config_num': i}

            print(f"  Val ROI: {result['roi']:.2f}% | Bets: {result['total_bets']} | "
                  f"Win Rate: {result['win_rate']:.1f}% | Accuracy: {result['accuracy']:.1f}% {roi_indicator}")
            print(f"  Time: {config_time:.1f}s | Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min\n")
        else:
            print(f"  FAILED: {result.get('error', 'Unknown error')}")
            print(f"  Time: {config_time:.1f}s | Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min\n")

    # Find best configuration by validation ROI
    successful_results = [r for r in all_results if r['success'] and r['total_bets'] >= 50]

    if not successful_results:
        print("\nNo successful configurations found with at least 50 bets!")
        print("Saving all results anyway...")

        # Save all results
        results_path = 'models/metadata/hyperparameter_tuning_WITH_odds.json'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'approach': 'WITH odds as features',
                'total_configs': total_configs,
                'all_results': all_results,
                'best_config': None
            }, f, indent=2)

        print(f"Results saved to {results_path}")
        return

    # Sort by ROI
    successful_results.sort(key=lambda x: x['roi'], reverse=True)
    best_result = successful_results[0]

    print("\n" + "="*80)
    print("BEST CONFIGURATION FOUND (WITH ODDS)")
    print("="*80)
    print(f"Validation ROI: {best_result['roi']:.2f}%")
    print(f"Validation Bets: {best_result['total_bets']}")
    print(f"Validation Win Rate: {best_result['win_rate']:.1f}%")
    print(f"Validation Accuracy: {best_result['accuracy']:.1f}%")
    print(f"\nBest Parameters:")
    for param, value in best_result['params'].items():
        print(f"  {param}: {value}")

    # Now test best config on test set with multiple EV thresholds
    print("\n" + "="*80)
    print("TESTING BEST CONFIG ON TEST SET")
    print("="*80)

    best_params = fixed_params.copy()
    best_params.update(best_result['params'])

    trainer = ClassifierTrainer()

    # Prepare data - same inclusions as training (WITH odds)
    feature_cols = [col for col in df.columns if col not in [
        'game_id', 'goalie_id', 'game_date', 'over_hit',
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

    # Train on train + val combined
    train_val_idx = np.concatenate([train_idx, val_idx])
    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]

    train_val_weights = trainer.calculate_sample_weights(df, train_val_idx)

    print(f"Training on {len(train_val_idx)} samples (train + validation)...")

    # Create a dummy val set for the train() method
    dummy_val_idx = val_idx[:100]
    X_dummy_val = X[dummy_val_idx]
    y_dummy_val = y[dummy_val_idx]

    trainer.train(
        X_train_val, y_train_val, X_dummy_val, y_dummy_val,
        params=best_params,
        sample_weight=train_val_weights
    )

    # Test on multiple EV thresholds
    X_test = X[test_idx]
    y_test = y[test_idx]

    test_results = trainer.test_ev_thresholds(
        X_test, y_test, df, test_idx,
        dataset_name='Test',
        thresholds=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    )

    # Save all results
    results_path = 'models/metadata/hyperparameter_tuning_WITH_odds.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'approach': 'WITH odds as features',
        'total_configs_tested': total_configs,
        'successful_configs': len(successful_results),
        'best_validation_result': best_result,
        'test_results_by_ev_threshold': test_results,
        'top_10_configs': successful_results[:10],
        'all_results': all_results
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nAll results saved to {results_path}")

    # Find best test threshold
    best_test_threshold = None
    best_test_roi = -float('inf')

    for threshold, metrics in test_results.items():
        if metrics['total_bets'] > 0 and metrics['roi'] > best_test_roi:
            best_test_roi = metrics['roi']
            best_test_threshold = threshold

    if best_test_threshold and best_test_roi > 0:
        print("\n" + "="*80)
        print("BEST TEST RESULT (WITH ODDS)")
        print("="*80)
        print(f"EV Threshold: {best_test_threshold*100:.0f}%")
        print(f"Test ROI: {best_test_roi:.2f}%")
        print(f"Test Bets: {test_results[best_test_threshold]['total_bets']}")
        print(f"Test Win Rate: {test_results[best_test_threshold]['win_rate']:.1f}%")
    else:
        print("\nNo profitable threshold found on test set.")

    total_time = time() - start_time
    print(f"\n{'='*80}")
    print(f"Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == '__main__':
    grid_search()
