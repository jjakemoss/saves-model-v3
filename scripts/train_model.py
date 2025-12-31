"""Train XGBoost model for goalie saves prediction

This script:
1. Loads processed training data
2. Splits into train/val/test sets (time-based)
3. Trains XGBoost model
4. Evaluates on all sets
5. Saves model and evaluation results
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import yaml
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.trainer import GoalieModelTrainer
from models.evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training workflow"""
    parser = argparse.ArgumentParser(description='Train goalie saves prediction model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/training_data.parquet',
        help='Path to training data'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='xgboost_goalie_model',
        help='Name for saved model'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=None,
        help='Test set size (overrides config)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=None,
        help='Validation set size (overrides config)'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip detailed evaluation (faster training)'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save evaluation plots'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("NHL Goalie Saves Prediction - Model Training")
    logger.info("="*60)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Load training data
    logger.info(f"Loading training data from {args.data}")
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}")
        logger.error("Please run feature engineering first: python scripts/engineer_features.py")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Check for target column
    if 'saves' not in df.columns:
        logger.error("Target column 'saves' not found in training data")
        sys.exit(1)

    # Initialize trainer
    logger.info("Initializing model trainer")
    trainer = GoalieModelTrainer(config)

    # Prepare data splits
    test_size = args.test_size if args.test_size is not None else config['model'].get('test_size', 0.15)
    val_size = args.val_size if args.val_size is not None else config['model'].get('validation_size', 0.15)

    logger.info(f"Preparing train/val/test splits (test={test_size:.1%}, val={val_size:.1%})")

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        df,
        target_col='saves',
        test_size=test_size,
        val_size=val_size,
        random_state=config['model'].get('random_state', 42)
    )

    # Train model
    logger.info("Starting model training...")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")

    model = trainer.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val
    )

    logger.info("Model training complete!")

    # Evaluate on all sets
    logger.info("\n" + "="*60)
    logger.info("Model Evaluation")
    logger.info("="*60)

    train_metrics = trainer.evaluate(X_train, y_train, "Training")
    val_metrics = trainer.evaluate(X_val, y_val, "Validation")
    test_metrics = trainer.evaluate(X_test, y_test, "Test")

    # Check if model meets targets
    target_rmse = config['model'].get('target_rmse', 5.0)  # Target: predict within 5 saves
    target_mae = config['model'].get('target_mae', 3.5)   # Target: average error under 3.5 saves

    logger.info("\n" + "="*60)
    logger.info("Performance vs Targets")
    logger.info("="*60)
    logger.info(f"Test RMSE: {test_metrics['rmse']:.3f} saves (target: <{target_rmse:.1f}) {'✓' if test_metrics['rmse'] < target_rmse else '✗'}")
    logger.info(f"Test MAE: {test_metrics['mae']:.3f} saves (target: <{target_mae:.1f}) {'✓' if test_metrics['mae'] < target_mae else '✗'}")
    logger.info(f"Test R²: {test_metrics['r2']:.4f} (higher is better)")

    # Detailed evaluation (optional)
    if not args.skip_evaluation:
        logger.info("\n" + "="*60)
        logger.info("Detailed Evaluation")
        logger.info("="*60)

        evaluator = ModelEvaluator(model, trainer.feature_names)

        # Run detailed evaluation on test set
        evaluator.evaluate_classification_metrics(X_test, y_test, "Test")

        # Calibration analysis
        logger.info("\nCalibration Analysis:")
        evaluator.analyze_calibration(X_test, y_test)

        # Feature importance
        logger.info("\nFeature Importance:")
        evaluator.get_feature_importance(importance_type='gain', top_n=20)

        # Save evaluation plots
        if args.save_plots:
            logger.info("\nGenerating evaluation plots...")
            output_dir = Path('models') / 'evaluation'
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            try:
                evaluator.plot_calibration_curve(
                    X_test, y_test,
                    save_path=output_dir / f'calibration_curve_{timestamp}.png'
                )
                evaluator.plot_roc_curve(
                    X_test, y_test,
                    save_path=output_dir / f'roc_curve_{timestamp}.png'
                )
                evaluator.plot_feature_importance(
                    top_n=20,
                    save_path=output_dir / f'feature_importance_{timestamp}.png'
                )
                evaluator.plot_prediction_distribution(
                    X_test, y_test,
                    save_path=output_dir / f'prediction_distribution_{timestamp}.png'
                )
                logger.info(f"Evaluation plots saved to {output_dir}")
            except Exception as e:
                logger.warning(f"Failed to generate plots: {e}")

        # Save evaluation report
        logger.info("\nSaving evaluation report...")
        eval_dir = Path('models') / 'metadata'
        evaluator.save_evaluation_report(eval_dir, report_name=args.model_name)

    # Save model
    logger.info("\n" + "="*60)
    logger.info("Saving Model")
    logger.info("="*60)

    model_dir = Path(config['paths'].get('models', 'models/trained'))
    trainer.save_model(model_dir, model_name=args.model_name)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Model saved to: {model_dir / f'{args.model_name}.pkl'}")
    logger.info(f"Features: {len(trainer.feature_names)}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.3f} saves")
    logger.info(f"Test MAE: {test_metrics['mae']:.3f} saves")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info("\nNext steps:")
    logger.info("  1. Review evaluation results in models/metadata/")
    logger.info("  2. Test predictions: python scripts/predict_games.py")
    logger.info("  3. Set up automation: python scripts/update_daily_data.py")


if __name__ == '__main__':
    main()
