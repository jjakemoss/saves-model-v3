"""
Hyperparameter Tuning for XGBoost Goalie Saves Model

This script performs hyperparameter optimization using both:
1. Random search for initial exploration
2. Grid search around promising regions

Key hyperparameters to tune:
- n_estimators: Number of boosting rounds
- max_depth: Maximum tree depth
- learning_rate: Step size shrinkage
- subsample: Fraction of samples per tree
- colsample_bytree: Fraction of features per tree
- min_child_weight: Minimum sum of instance weight in a child
- gamma: Minimum loss reduction for split
- reg_alpha: L1 regularization
- reg_lambda: L2 regularization
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from src.models.trainer import GoalieModelTrainer
from itertools import product
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def random_search_hyperparameters(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    trainer,
    n_iterations=30
):
    """
    Random search over hyperparameter space

    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Target vectors
        trainer: GoalieModelTrainer instance
        n_iterations: Number of random configurations to try

    Returns:
        List of results dictionaries
    """
    logger.info(f"\n{'='*70}")
    logger.info("RANDOM SEARCH - Exploring Hyperparameter Space")
    logger.info(f"Testing {n_iterations} random configurations")
    logger.info(f"{'='*70}\n")

    # Define hyperparameter ranges
    param_distributions = {
        'n_estimators': [300, 500, 700, 1000, 1500],
        'max_depth': [4, 5, 6, 7, 8, 10],
        'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 5, 7],
        'gamma': [0, 0.05, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.01, 0.05, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    }

    results = []

    for i in range(n_iterations):
        # Randomly sample hyperparameters
        params = {
            key: np.random.choice(values)
            for key, values in param_distributions.items()
        }

        logger.info(f"\n--- Random Search Iteration {i+1}/{n_iterations} ---")
        logger.info(f"Testing parameters: {params}")

        try:
            # Train model with these hyperparameters
            model = trainer.train(
                X_train, y_train, X_val, y_val,
                **params
            )

            # Evaluate on validation and test sets
            val_metrics = trainer.evaluate(X_val, y_val, dataset_name="Validation")
            test_metrics = trainer.evaluate(X_test, y_test, dataset_name="Test")

            # Store results
            result = {
                'iteration': i + 1,
                'params': params,
                'val_rmse': val_metrics['rmse'],
                'val_mae': val_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae']
            }
            results.append(result)

            logger.info(f"Validation RMSE: {val_metrics['rmse']:.3f} | Test RMSE: {test_metrics['rmse']:.3f}")

        except Exception as e:
            logger.error(f"Error in iteration {i+1}: {e}")
            continue

    return results


def grid_search_refined(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    trainer,
    base_params
):
    """
    Grid search around best parameters from random search

    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Target vectors
        trainer: GoalieModelTrainer instance
        base_params: Best parameters from random search

    Returns:
        List of results dictionaries
    """
    logger.info(f"\n{'='*70}")
    logger.info("GRID SEARCH - Refining Best Configuration")
    logger.info(f"Base parameters: {base_params}")
    logger.info(f"{'='*70}\n")

    # Create grid around best parameters
    param_grid = {
        'n_estimators': [
            int(base_params['n_estimators'] * 0.8),
            base_params['n_estimators'],
            int(base_params['n_estimators'] * 1.2)
        ],
        'max_depth': [
            max(3, base_params['max_depth'] - 1),
            base_params['max_depth'],
            min(12, base_params['max_depth'] + 1)
        ],
        'learning_rate': [
            base_params['learning_rate'] * 0.8,
            base_params['learning_rate'],
            base_params['learning_rate'] * 1.2
        ],
        'subsample': [
            max(0.5, base_params['subsample'] - 0.1),
            base_params['subsample'],
            min(1.0, base_params['subsample'] + 0.1)
        ],
        'colsample_bytree': [
            max(0.5, base_params['colsample_bytree'] - 0.1),
            base_params['colsample_bytree'],
            min(1.0, base_params['colsample_bytree'] + 0.1)
        ]
    }

    # Fix other parameters
    fixed_params = {
        'min_child_weight': base_params['min_child_weight'],
        'gamma': base_params['gamma'],
        'reg_alpha': base_params['reg_alpha'],
        'reg_lambda': base_params['reg_lambda']
    }

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    all_combinations = list(product(*param_values))

    logger.info(f"Testing {len(all_combinations)} grid combinations")

    results = []

    for i, combination in enumerate(all_combinations):
        params = dict(zip(param_names, combination))
        params.update(fixed_params)

        logger.info(f"\n--- Grid Search {i+1}/{len(all_combinations)} ---")
        logger.info(f"Parameters: {params}")

        try:
            # Train model
            model = trainer.train(
                X_train, y_train, X_val, y_val,
                **params
            )

            # Evaluate
            val_metrics = trainer.evaluate(X_val, y_val, dataset_name="Validation")
            test_metrics = trainer.evaluate(X_test, y_test, dataset_name="Test")

            result = {
                'iteration': i + 1,
                'params': params,
                'val_rmse': val_metrics['rmse'],
                'val_mae': val_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae']
            }
            results.append(result)

            logger.info(f"Validation RMSE: {val_metrics['rmse']:.3f} | Test RMSE: {test_metrics['rmse']:.3f}")

        except Exception as e:
            logger.error(f"Error in grid iteration {i+1}: {e}")
            continue

    return results


def main():
    """Main hyperparameter tuning workflow"""

    logger.info("="*70)
    logger.info("XGBoost Hyperparameter Tuning")
    logger.info("="*70)

    # Load training data
    logger.info("\nLoading training data...")
    data_path = Path('data/processed/training_data.parquet')
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Initialize trainer with EWA (best from previous optimization)
    config = {
        'model': {'random_state': 42},
        'features': {'rolling_windows': [3, 5, 10]}
    }
    trainer = GoalieModelTrainer(config)

    # Prepare data with EWA
    logger.info("\nPreparing data with EWA (span=1.25)...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        df,
        target_col='saves',
        use_ewa=True,
        ewa_span_multiplier=1.25
    )

    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Validation: {len(X_val)} samples")
    logger.info(f"Test: {len(X_test)} samples")

    # PHASE 1: Random Search
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: Random Search")
    logger.info("="*70)

    random_results = random_search_hyperparameters(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        trainer,
        n_iterations=30
    )

    # Find best from random search
    random_df = pd.DataFrame(random_results)
    random_df = random_df.sort_values('val_rmse')
    best_random = random_df.iloc[0]

    logger.info(f"\n{'='*70}")
    logger.info("BEST RESULT FROM RANDOM SEARCH")
    logger.info(f"{'='*70}")
    logger.info(f"Validation RMSE: {best_random['val_rmse']:.3f}")
    logger.info(f"Test RMSE: {best_random['test_rmse']:.3f}")
    logger.info(f"Parameters: {best_random['params']}")

    # Save random search results
    random_output = Path('models/metadata/hyperparameter_random_search.json')
    random_df.to_json(random_output, orient='records', indent=2)
    logger.info(f"\nRandom search results saved to {random_output}")

    # PHASE 2: Grid Search around best parameters
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: Grid Search Refinement")
    logger.info(f"{'='*70}")

    grid_results = grid_search_refined(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        trainer,
        base_params=best_random['params']
    )

    # Find overall best
    all_results = random_results + grid_results
    all_df = pd.DataFrame(all_results)
    all_df = all_df.sort_values('val_rmse')
    best_overall = all_df.iloc[0]

    logger.info(f"\n{'='*70}")
    logger.info("BEST HYPERPARAMETERS (OVERALL)")
    logger.info(f"{'='*70}")
    logger.info(f"Validation RMSE: {best_overall['val_rmse']:.3f}")
    logger.info(f"Test RMSE: {best_overall['test_rmse']:.3f}")
    logger.info(f"\nOptimal Parameters:")
    for param, value in best_overall['params'].items():
        logger.info(f"  {param}: {value}")

    # Save all results
    output_path = Path('models/metadata/hyperparameter_tuning_results.json')
    all_df.to_json(output_path, orient='records', indent=2)
    logger.info(f"\nAll results saved to {output_path}")

    # Save best parameters separately
    best_params_path = Path('models/metadata/best_hyperparameters.json')
    # Convert numpy types to native Python types for JSON serialization
    best_params_serializable = {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v)
                                for k, v in best_overall['params'].items()}
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_params': best_params_serializable,
            'val_rmse': float(best_overall['val_rmse']),
            'test_rmse': float(best_overall['test_rmse']),
            'tuning_date': datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"Best parameters saved to {best_params_path}")

    # Compare to baseline
    logger.info(f"\n{'='*70}")
    logger.info("IMPROVEMENT SUMMARY")
    logger.info(f"{'='*70}")
    baseline_rmse = 6.510  # Current best with EWA
    improvement = baseline_rmse - best_overall['test_rmse']
    pct_improvement = (improvement / baseline_rmse) * 100

    logger.info(f"Baseline (EWA, default params): {baseline_rmse:.3f} RMSE")
    logger.info(f"Optimized hyperparameters:      {best_overall['test_rmse']:.3f} RMSE")
    logger.info(f"Improvement:                    {improvement:.3f} saves ({pct_improvement:.2f}%)")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
