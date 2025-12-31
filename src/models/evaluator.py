"""Model evaluation and analysis module for regression

Provides detailed evaluation metrics, residual analysis, and feature importance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis for regression

    Provides:
    - Regression metrics (RMSE, MAE, R², MAPE)
    - Residual analysis
    - Feature importance analysis
    - Prediction distribution analysis
    - Error analysis by subgroups
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize evaluator

        Args:
            model: Trained model (XGBoost regressor)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.evaluation_results = {}

    def evaluate_regression_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics

        Args:
            X: Features
            y: True values (actual saves)
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {dataset_name} set...")

        # Get predictions
        y_pred = self.model.predict(X)

        # Calculate metrics
        metrics = {
            'dataset': dataset_name,
            'n_samples': len(y),

            # Regression metrics
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': mean_absolute_percentage_error(y, y_pred) * 100,  # As percentage

            # Prediction statistics
            'mean_actual': y.mean(),
            'std_actual': y.std(),
            'min_actual': y.min(),
            'max_actual': y.max(),
            'mean_predicted': y_pred.mean(),
            'std_predicted': y_pred.std(),
            'min_predicted': y_pred.min(),
            'max_predicted': y_pred.max(),

            # Residual statistics
            'mean_residual': (y - y_pred).mean(),
            'std_residual': (y - y_pred).std(),
        }

        # Store results
        self.evaluation_results[dataset_name] = metrics

        # Log key metrics
        logger.info(f"\n{dataset_name} Set Metrics:")
        logger.info(f"  Samples: {metrics['n_samples']}")
        logger.info(f"  RMSE: {metrics['rmse']:.3f} saves")
        logger.info(f"  MAE: {metrics['mae']:.3f} saves")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  Mean Actual: {metrics['mean_actual']:.2f} saves")
        logger.info(f"  Mean Predicted: {metrics['mean_predicted']:.2f} saves")

        return metrics

    def analyze_residuals(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Analyze prediction residuals

        Args:
            X: Features
            y: True values
            n_bins: Number of bins for residual distribution

        Returns:
            DataFrame with residual analysis
        """
        y_pred = self.model.predict(X)
        residuals = y - y_pred

        # Residual statistics
        residual_stats = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'q25': residuals.quantile(0.25),
            'median': residuals.median(),
            'q75': residuals.quantile(0.75),
        }

        logger.info(f"\nResidual Analysis:")
        logger.info(f"  Mean: {residual_stats['mean']:.3f}")
        logger.info(f"  Std Dev: {residual_stats['std']:.3f}")
        logger.info(f"  Range: [{residual_stats['min']:.1f}, {residual_stats['max']:.1f}]")
        logger.info(f"  IQR: [{residual_stats['q25']:.1f}, {residual_stats['q75']:.1f}]")

        return pd.DataFrame([residual_stats])

    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from model

        Args:
            importance_type: 'weight', 'gain', or 'cover'
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'get_score'):
            # XGBoost get_score method
            importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
            importances = np.array([importance_dict.get(f'f{i}', 0.0) for i in range(len(self.feature_names))])
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop {top_n} Features by {importance_type}:")
        for idx, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df.head(top_n)

    def plot_residuals(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[Path] = None
    ):
        """
        Plot residuals histogram and Q-Q plot

        Args:
            X: Features
            y: True values
            save_path: Path to save plot (optional)
        """
        y_pred = self.model.predict(X)
        residuals = y - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of residuals
        ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', label='Zero Error')
        ax1.set_xlabel('Residual (Actual - Predicted)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Residual Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Residuals vs predicted
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_xlabel('Predicted Saves')
        ax2.set_ylabel('Residual (Actual - Predicted)')
        ax2.set_title('Residuals vs Predicted Values')
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_predicted_vs_actual(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[Path] = None
    ):
        """
        Plot predicted vs actual values

        Args:
            X: Features
            y: True values
            save_path: Path to save plot (optional)
        """
        y_pred = self.model.predict(X)

        plt.figure(figsize=(8, 8))
        plt.scatter(y, y_pred, alpha=0.5)

        # Perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.xlabel('Actual Saves')
        plt.ylabel('Predicted Saves')
        plt.title('Predicted vs Actual Saves')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predicted vs Actual plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_feature_importance(
        self,
        top_n: int = 20,
        importance_type: str = 'gain',
        save_path: Optional[Path] = None
    ):
        """
        Plot feature importance

        Args:
            top_n: Number of top features to plot
            importance_type: 'weight', 'gain', or 'cover'
            save_path: Path to save plot (optional)
        """
        importance_df = self.get_feature_importance(importance_type=importance_type, top_n=top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel(f'Importance ({importance_type})')
        plt.title(f'Top {top_n} Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()

    def plot_prediction_distribution(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[Path] = None
    ):
        """
        Plot distribution of predicted vs actual saves

        Args:
            X: Features
            y: True values
            save_path: Path to save plot (optional)
        """
        y_pred = self.model.predict(X)

        plt.figure(figsize=(10, 6))

        # Plot distributions
        plt.hist(y, bins=30, alpha=0.5, label='Actual Saves', color='blue', edgecolor='black')
        plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted Saves', color='red', edgecolor='black')

        plt.axvline(y.mean(), color='blue', linestyle='--', label=f'Actual Mean: {y.mean():.1f}')
        plt.axvline(y_pred.mean(), color='red', linestyle='--', label=f'Predicted Mean: {y_pred.mean():.1f}')

        plt.xlabel('Saves')
        plt.ylabel('Frequency')
        plt.title('Distribution of Actual vs Predicted Saves')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def save_evaluation_report(
        self,
        output_dir: Path,
        report_name: str = "evaluation_report"
    ):
        """
        Save comprehensive evaluation report

        Args:
            output_dir: Directory to save report files
            report_name: Base name for report files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        import json
        metrics_path = output_dir / f"{report_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for dataset, metrics in self.evaluation_results.items():
                serializable_results[dataset] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics.items()
                }
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Evaluation metrics saved to {metrics_path}")

        # Save feature importance as CSV
        importance_df = self.get_feature_importance(top_n=50)
        importance_path = output_dir / f"{report_name}_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        logger.info(f"Feature importance saved to {importance_path}")

        logger.info(f"Evaluation report saved to {output_dir}")
