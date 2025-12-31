"""Model evaluation and analysis module

Provides detailed evaluation metrics, calibration analysis, and feature importance analysis.
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
    log_loss, roc_auc_score, brier_score_loss, accuracy_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis

    Provides:
    - Classification metrics (log loss, ROC-AUC, Brier score)
    - Calibration analysis
    - Feature importance analysis
    - Prediction distribution analysis
    - Error analysis by subgroups
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize evaluator

        Args:
            model: Trained model (XGBoost classifier)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.evaluation_results = {}

    def evaluate_classification_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics

        Args:
            X: Features
            y: True labels
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {dataset_name} set...")

        # Get predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        metrics = {
            'dataset': dataset_name,
            'n_samples': len(y),
            'n_positive': y.sum(),
            'positive_rate': y.mean(),

            # Probabilistic metrics
            'log_loss': log_loss(y, y_pred_proba),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'brier_score': brier_score_loss(y, y_pred_proba),

            # Classification metrics
            'accuracy': accuracy_score(y, y_pred),
            'baseline_accuracy': max(y.mean(), 1 - y.mean()),  # Always predict majority class

            # Prediction statistics
            'mean_predicted_prob': y_pred_proba.mean(),
            'std_predicted_prob': y_pred_proba.std(),
            'min_predicted_prob': y_pred_proba.min(),
            'max_predicted_prob': y_pred_proba.max(),
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0

        # Store results
        self.evaluation_results[dataset_name] = metrics

        # Log key metrics
        logger.info(f"\n{dataset_name} Set Metrics:")
        logger.info(f"  Samples: {metrics['n_samples']} ({metrics['n_positive']} positive, {metrics['positive_rate']:.1%})")
        logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f} (baseline: {metrics['baseline_accuracy']:.4f})")
        if 'precision' in metrics:
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")

        return metrics

    def analyze_calibration(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_bins: int = 10,
        strategy: str = 'quantile'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze probability calibration

        Args:
            X: Features
            y: True labels
            n_bins: Number of bins for calibration curve
            strategy: 'uniform' or 'quantile' binning

        Returns:
            (prob_true, prob_pred) - Calibration curve data
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        prob_true, prob_pred = calibration_curve(
            y, y_pred_proba, n_bins=n_bins, strategy=strategy
        )

        # Calculate calibration error
        calibration_error = np.mean(np.abs(prob_true - prob_pred))

        logger.info(f"Calibration Error (mean absolute): {calibration_error:.4f}")

        return prob_true, prob_pred

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

    def analyze_predictions_by_line(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        betting_lines: pd.Series,
        line_bins: List[float] = [0, 25, 30, 35, 100]
    ) -> pd.DataFrame:
        """
        Analyze model performance by betting line ranges

        Args:
            X: Features
            y: True labels
            betting_lines: Betting line values
            line_bins: Bin edges for grouping lines

        Returns:
            DataFrame with metrics by line range
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Create line bins
        line_categories = pd.cut(betting_lines, bins=line_bins, include_lowest=True)

        results = []
        for category in line_categories.cat.categories:
            mask = line_categories == category
            if mask.sum() == 0:
                continue

            y_subset = y[mask]
            y_pred_subset = y_pred_proba[mask]

            result = {
                'line_range': str(category),
                'n_samples': mask.sum(),
                'actual_over_rate': y_subset.mean(),
                'predicted_over_rate': y_pred_subset.mean(),
                'log_loss': log_loss(y_subset, y_pred_subset),
                'roc_auc': roc_auc_score(y_subset, y_pred_subset) if len(y_subset.unique()) > 1 else np.nan,
                'brier_score': brier_score_loss(y_subset, y_pred_subset)
            }
            results.append(result)

        results_df = pd.DataFrame(results)

        logger.info("\nPerformance by Betting Line Range:")
        logger.info(results_df.to_string(index=False))

        return results_df

    def analyze_errors(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5,
        top_n: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze prediction errors (false positives and false negatives)

        Args:
            X: Features
            y: True labels
            threshold: Classification threshold
            top_n: Number of worst errors to return

        Returns:
            (false_positives_df, false_negatives_df)
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # False positives (predicted over, actual under)
        fp_mask = (y_pred == 1) & (y == 0)
        fp_df = pd.DataFrame({
            'predicted_prob': y_pred_proba[fp_mask],
            'actual': y[fp_mask],
            'error_magnitude': y_pred_proba[fp_mask] - y[fp_mask]
        }).sort_values('predicted_prob', ascending=False).head(top_n)

        # False negatives (predicted under, actual over)
        fn_mask = (y_pred == 0) & (y == 1)
        fn_df = pd.DataFrame({
            'predicted_prob': y_pred_proba[fn_mask],
            'actual': y[fn_mask],
            'error_magnitude': y[fn_mask] - y_pred_proba[fn_mask]
        }).sort_values('predicted_prob', ascending=True).head(top_n)

        logger.info(f"\nError Analysis:")
        logger.info(f"  False Positives: {fp_mask.sum()} ({fp_mask.mean():.1%})")
        logger.info(f"  False Negatives: {fn_mask.sum()} ({fn_mask.mean():.1%})")

        return fp_df, fn_df

    def plot_calibration_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[Path] = None
    ):
        """
        Plot calibration curve

        Args:
            X: Features
            y: True labels
            save_path: Path to save plot (optional)
        """
        prob_true, prob_pred = self.analyze_calibration(X, y)

        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(prob_pred, prob_true, 'o-', label='Model')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Actual Frequency')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration plot saved to {save_path}")
        else:
            plt.show()

    def plot_roc_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[Path] = None
    ):
        """
        Plot ROC curve

        Args:
            X: Features
            y: True labels
            save_path: Path to save plot (optional)
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
        plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
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
        Plot distribution of predicted probabilities

        Args:
            X: Features
            y: True labels
            save_path: Path to save plot (optional)
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        plt.figure(figsize=(10, 6))

        # Plot distributions for each class
        plt.hist(y_pred_proba[y == 0], bins=30, alpha=0.5, label='Actual Under', color='blue')
        plt.hist(y_pred_proba[y == 1], bins=30, alpha=0.5, label='Actual Over', color='red')

        plt.axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
        plt.xlabel('Predicted Probability (Over)')
        plt.ylabel('Frequency')
        plt.title('Prediction Distribution by Actual Outcome')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution plot saved to {save_path}")
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
