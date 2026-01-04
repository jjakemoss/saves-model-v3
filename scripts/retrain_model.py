"""Model retraining script

Retrain model with latest data and compare performance to previous version.
Run weekly/monthly to keep model current with latest NHL trends.

Usage:
    python scripts/retrain_model.py
    python scripts/retrain_model.py --force  # Deploy even if performance worse
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml
import json
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.trainer import GoalieModelTrainer
from models.evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_previous_metrics(model_dir: Path, model_name: str) -> dict:
    """
    Load metrics from previous model version

    Args:
        model_dir: Directory containing model files
        model_name: Base name of model

    Returns:
        Dictionary of previous metrics or empty dict if not found
    """
    metadata_path = model_dir / f"{model_name}_metadata.json"

    if not metadata_path.exists():
        logger.warning(f"No previous metadata found at {metadata_path}")
        return {}

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Extract test metrics if available
        if 'test_metrics' in metadata:
            return metadata['test_metrics']
        else:
            logger.warning("No test metrics found in previous metadata")
            return {}

    except Exception as e:
        logger.error(f"Error loading previous metadata: {e}")
        return {}


def compare_performance(new_metrics: dict, old_metrics: dict) -> dict:
    """
    Compare new model performance to previous version

    Args:
        new_metrics: Metrics from newly trained model
        old_metrics: Metrics from previous model

    Returns:
        Dictionary with comparison results
    """
    if not old_metrics:
        logger.info("No previous metrics to compare against")
        return {
            'is_improvement': True,
            'reason': 'No previous model to compare'
        }

    # Compare key metrics (lower is better for RMSE/MAE, higher for R²)
    rmse_improved = new_metrics['rmse'] < old_metrics.get('rmse', float('inf'))
    mae_improved = new_metrics['mae'] < old_metrics.get('mae', float('inf'))
    r2_improved = new_metrics['r2'] > old_metrics.get('r2', -float('inf'))

    # Calculate improvements
    rmse_change = new_metrics['rmse'] - old_metrics.get('rmse', 0)
    mae_change = new_metrics['mae'] - old_metrics.get('mae', 0)
    r2_change = new_metrics['r2'] - old_metrics.get('r2', 0)

    logger.info("\n" + "="*60)
    logger.info("Performance Comparison")
    logger.info("="*60)
    logger.info(f"RMSE: {old_metrics.get('rmse', 0):.3f} → {new_metrics['rmse']:.3f} ({rmse_change:+.3f})")
    logger.info(f"MAE:  {old_metrics.get('mae', 0):.3f} → {new_metrics['mae']:.3f} ({mae_change:+.3f})")
    logger.info(f"R²:   {old_metrics.get('r2', 0):.4f} → {new_metrics['r2']:.4f} ({r2_change:+.4f})")

    # Determine if overall improvement (majority of metrics improved)
    improvements = sum([rmse_improved, mae_improved, r2_improved])
    is_improvement = improvements >= 2  # At least 2 out of 3 metrics improved

    comparison = {
        'is_improvement': is_improvement,
        'rmse_improved': rmse_improved,
        'mae_improved': mae_improved,
        'r2_improved': r2_improved,
        'rmse_change': rmse_change,
        'mae_change': mae_change,
        'r2_change': r2_change,
        'improvements_count': improvements
    }

    if is_improvement:
        logger.info("\n✓ New model shows improvement!")
    else:
        logger.warning("\n✗ New model performance is worse or mixed")

    return comparison


def backup_model(model_dir: Path, model_name: str):
    """
    Backup current model before replacing

    Args:
        model_dir: Directory containing model files
        model_name: Base name of model
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = model_dir / 'backups' / f"{model_name}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Files to backup
    files_to_backup = [
        f"{model_name}.pkl",
        f"{model_name}_features.json",
        f"{model_name}_feature_importance.csv",
        f"{model_name}_metadata.json"
    ]

    logger.info(f"Backing up current model to {backup_dir}")

    for filename in files_to_backup:
        src = model_dir / filename
        if src.exists():
            dst = backup_dir / filename
            shutil.copy2(src, dst)
            logger.info(f"  Backed up {filename}")

    logger.info("Backup complete")


def deploy_model(temp_dir: Path, model_dir: Path, model_name: str):
    """
    Deploy new model by moving from temp to production directory

    Args:
        temp_dir: Temporary directory with new model
        model_dir: Production model directory
        model_name: Base name of model
    """
    logger.info("Deploying new model...")

    # Files to deploy
    files_to_deploy = [
        f"{model_name}.pkl",
        f"{model_name}_features.json",
        f"{model_name}_feature_importance.csv",
        f"{model_name}_metadata.json"
    ]

    for filename in files_to_deploy:
        src = temp_dir / filename
        dst = model_dir / filename

        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"  Deployed {filename}")
        else:
            logger.warning(f"  {filename} not found in temp directory")

    logger.info("Deployment complete")


def main():
    """Main retraining workflow"""
    parser = argparse.ArgumentParser(description='Retrain model with latest data')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--force', action='store_true', help='Deploy even if performance worse')
    parser.add_argument('--skip-backup', action='store_true', help='Skip backing up current model')

    args = parser.parse_args()

    print("=" * 70)
    print("NHL GOALIE SAVES MODEL - RETRAINING")
    print("=" * 70)
    print()

    # Load config
    config = load_config(args.config)

    # Paths
    data_path = Path(config['paths']['processed_data']) / 'training_data.parquet'
    model_dir = Path(config['paths']['models'])
    model_name = 'xgboost_goalie_model'

    # Check if training data exists
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}")
        logger.error("Please run feature engineering first: python scripts/engineer_features.py")
        sys.exit(1)

    # Load training data
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Load previous model metrics
    logger.info("Loading previous model metrics...")
    previous_metrics = load_previous_metrics(model_dir, model_name)

    if previous_metrics:
        logger.info(f"Previous model RMSE: {previous_metrics.get('rmse', 'N/A')}")
        logger.info(f"Previous model MAE: {previous_metrics.get('mae', 'N/A')}")
        logger.info(f"Previous model R²: {previous_metrics.get('r2', 'N/A')}")
    else:
        logger.info("No previous metrics found (first training)")

    # Initialize trainer
    logger.info("\nInitializing model trainer")
    trainer = GoalieModelTrainer(config)

    # Prepare data
    test_size = config['model'].get('test_size', 0.15)
    val_size = config['model'].get('validation_size', 0.15)

    logger.info(f"Preparing train/val/test splits (test={test_size:.1%}, val={val_size:.1%})")

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        df,
        target_col='saves',
        test_size=test_size,
        val_size=val_size,
        random_state=config['model'].get('random_state', 42)
    )

    # Train new model
    logger.info("\nStarting model training...")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")

    model = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate new model
    logger.info("\n" + "="*60)
    logger.info("New Model Evaluation")
    logger.info("="*60)

    train_metrics = trainer.evaluate(X_train, y_train, "Training")
    val_metrics = trainer.evaluate(X_val, y_val, "Validation")
    test_metrics = trainer.evaluate(X_test, y_test, "Test")

    # Compare to previous model
    comparison = compare_performance(test_metrics, previous_metrics)

    # Decide whether to deploy
    should_deploy = comparison['is_improvement'] or args.force

    if not should_deploy:
        logger.warning("\n" + "="*60)
        logger.warning("New model does not show improvement")
        logger.warning("Skipping deployment (use --force to override)")
        logger.warning("="*60)
        print("\nRetraining completed but new model NOT deployed (no improvement)")
        sys.exit(0)

    # Create temporary directory for new model
    temp_dir = model_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Save new model to temp directory
    logger.info(f"\nSaving new model to temporary directory: {temp_dir}")

    # Temporarily save to temp directory
    original_model = trainer.model
    trainer.save_model(temp_dir, model_name=model_name)

    # Update metadata with test metrics
    metadata_path = temp_dir / f"{model_name}_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    metadata['test_metrics'] = test_metrics
    metadata['retrain_date'] = datetime.now().isoformat()
    metadata['comparison_to_previous'] = comparison

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Backup current production model
    if not args.skip_backup and (model_dir / f"{model_name}.pkl").exists():
        backup_model(model_dir, model_name)

    # Deploy new model
    deploy_model(temp_dir, model_dir, model_name)

    # Clean up temp directory
    shutil.rmtree(temp_dir)
    logger.info("Cleaned up temporary files")

    # Summary
    print()
    print("=" * 70)
    print("RETRAINING COMPLETE")
    print("=" * 70)
    print(f"Dataset size: {len(df)} samples")
    print(f"Test RMSE: {test_metrics['rmse']:.3f} saves")
    print(f"Test MAE: {test_metrics['mae']:.3f} saves")
    print(f"Test R²: {test_metrics['r2']:.4f}")

    if previous_metrics:
        print(f"\nImprovement vs previous:")
        print(f"  RMSE: {comparison['rmse_change']:+.3f} saves")
        print(f"  MAE: {comparison['mae_change']:+.3f} saves")
        print(f"  R²: {comparison['r2_change']:+.4f}")

    print(f"\nModel deployed to: {model_dir}")
    print()

    logger.info("Retraining completed successfully")


if __name__ == '__main__':
    main()
