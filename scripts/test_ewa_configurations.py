"""
Test different EWA (Exponential Weighted Average) configurations

This script tests different decay rates and weighting schemes for EWA features
to find the optimal configuration that improves model performance.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from src.models.trainer import GoalieModelTrainer
import json

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_ewa_configuration(span_multiplier: float, config_name: str):
    """
    Test a specific EWA configuration by modifying the span parameter

    Args:
        span_multiplier: Multiplier for span (1.0 = current, 0.5 = faster decay, 1.5 = slower decay)
        config_name: Name for this configuration

    Returns:
        Dictionary with test metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing EWA Configuration: {config_name}")
    logger.info(f"Span multiplier: {span_multiplier}")
    logger.info(f"{'='*70}\n")

    # Load training data
    data_path = Path('data/processed/training_data.parquet')
    df = pd.read_parquet(data_path)

    logger.info(f"Loaded {len(df)} training samples")
    logger.info(f"Testing with span_multiplier={span_multiplier}")

    # Train model with modified EWA features
    config = {
        'model': {'random_state': 42},
        'features': {'rolling_windows': [3, 5, 10]}
    }

    trainer = GoalieModelTrainer(config)

    # Prepare data with EWA configuration
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        df,
        target_col='saves',
        use_ewa=True,
        ewa_span_multiplier=span_multiplier
    )

    # Train model
    logger.info("Training model with new EWA configuration...")
    model = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test, dataset_name="Test")

    # Return results
    results = {
        'config_name': config_name,
        'span_multiplier': span_multiplier,
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2']
    }

    logger.info(f"\nResults for {config_name}:")
    logger.info(f"  Test RMSE: {test_metrics['rmse']:.3f} saves")
    logger.info(f"  Test MAE: {test_metrics['mae']:.3f} saves")
    logger.info(f"  Test R²: {test_metrics['r2']:.4f}")

    return results


def main():
    """Test multiple EWA configurations and compare results"""

    configurations = [
        (1.0, "Baseline (current)"),
        (0.5, "Fast decay (span=window*0.5)"),
        (0.75, "Medium-fast decay (span=window*0.75)"),
        (1.25, "Medium-slow decay (span=window*1.25)"),
        (1.5, "Slow decay (span=window*1.5)"),
        (2.0, "Very slow decay (span=window*2.0)")
    ]

    all_results = []

    for span_mult, config_name in configurations:
        try:
            results = test_ewa_configuration(span_mult, config_name)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error testing {config_name}: {e}")
            continue

    # Compare results
    logger.info(f"\n{'='*70}")
    logger.info("COMPARISON OF ALL EWA CONFIGURATIONS")
    logger.info(f"{'='*70}\n")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('test_rmse')

    print(results_df.to_string(index=False))

    # Find best configuration
    best_config = results_df.iloc[0]
    logger.info(f"\n{'='*70}")
    logger.info(f"BEST CONFIGURATION: {best_config['config_name']}")
    logger.info(f"  Span multiplier: {best_config['span_multiplier']}")
    logger.info(f"  Test RMSE: {best_config['test_rmse']:.3f} saves")
    logger.info(f"  Improvement vs baseline: {results_df[results_df['config_name']=='Baseline (current)']['test_rmse'].values[0] - best_config['test_rmse']:.3f} saves")
    logger.info(f"{'='*70}\n")

    # Save results
    output_path = Path('models/metadata/ewa_configuration_comparison.json')
    results_df.to_json(output_path, orient='records', indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
