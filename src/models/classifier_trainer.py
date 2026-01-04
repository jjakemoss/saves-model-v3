"""
Trainer for binary classification model (OVER/UNDER prediction)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    classification_report, confusion_matrix
)
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ClassifierTrainer:
    """Train XGBoost binary classifier for OVER/UNDER prediction"""

    def __init__(self, config_path='config/config.json'):
        """Initialize trainer with config"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self.feature_names = None

    def _load_config(self):
        """Load configuration from JSON"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def load_data(self, data_path='data/processed/classification_training_data.parquet'):
        """Load classification training data (already clean, no need to recalculate)"""
        logger.info(f"Loading data from {data_path}")

        df = pd.read_parquet(data_path)

        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        logger.info(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

        # Data is already clean from create_clean_features.py - no recalculation needed
        logger.info("Using pre-calculated clean features (no data leakage)")

        return df

    def _recalculate_rolling_features(self, df):
        """
        Recalculate rolling features WITHOUT data leakage

        For each game, rolling features use ONLY data from prior games.
        Uses shift(1) to ensure current game is excluded from rolling averages.
        """
        # Define stats to calculate rolling features for
        goalie_stats = [
            'saves',
            'save_percentage',
            'shots_against',
            'goals_against',
            'even_strength_save_pct',
            'power_play_save_pct'
        ]

        # Only use stats that exist in the dataframe
        goalie_stats = [stat for stat in goalie_stats if stat in df.columns]

        windows = [3, 5, 10]

        df_result = df.copy()

        # Sort by goalie and date to ensure proper time ordering
        df_result = df_result.sort_values(['goalie_id', 'game_date'])

        # Calculate rolling averages with shift(1) to exclude current game
        for stat in goalie_stats:
            for window in windows:
                # All possible column name formats that might exist from merge operations
                col_name_x = f"{stat}_rolling_{window}_x"
                col_name_y = f"{stat}_rolling_{window}_y"
                col_name = f"{stat}_rolling_{window}"
                col_std_x = f"{stat}_rolling_std_{window}_x"
                col_std_y = f"{stat}_rolling_std_{window}_y"
                col_std = f"{stat}_rolling_std_{window}"

                # CRITICAL: Drop ALL existing rolling columns (they contain data leakage)
                for col in [col_name_x, col_name_y, col_name, col_std_x, col_std_y, col_std]:
                    if col in df_result.columns:
                        df_result = df_result.drop(columns=[col])

                # Recalculate MEAN with proper shift (excludes current game)
                df_result[col_name] = df_result.groupby('goalie_id')[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )

                # Recalculate STD with proper shift (excludes current game)
                df_result[col_std] = df_result.groupby('goalie_id')[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
                )

        # Also recalculate EWA features
        ewa_stats = ['saves', 'save_percentage', 'shots_against']
        ewa_windows = [3, 5, 10]

        for stat in [s for s in ewa_stats if s in df.columns]:
            for window in ewa_windows:
                col_name = f"{stat}_ewa_{window}"

                # Drop if exists
                if col_name in df_result.columns:
                    df_result = df_result.drop(columns=[col_name])

                # Recalculate with shift
                df_result[col_name] = df_result.groupby('goalie_id')[stat].transform(
                    lambda x: x.ewm(span=window, adjust=False, min_periods=1).mean().shift(1)
                )

        logger.info(f"Recalculated rolling features (mean + std) for {len(goalie_stats)} stats across {len(windows)} windows")

        return df_result

    def prepare_features(self, df):
        """
        Prepare features and target for classification

        Args:
            df: DataFrame with all data

        Returns:
            X: Feature matrix
            y: Target vector (over_hit)
            feature_names: List of feature column names
        """
        logger.info("Preparing features and target...")

        # Target variable
        y = df['over_hit'].values

        # Columns to exclude from features
        exclude_cols = [
            # Metadata
            'goalie_id', 'game_id', 'game_date', 'season', 'team_abbrev',
            'opponent_team', 'toi', 'decision', 'team_id', 'opponent_id',
            # Target variables
            'saves', 'over_hit', 'line_margin',  # betting_line IS a valid feature (known pre-game)
            # Constant features
            'is_starter',  # Always True in training data
            # CRITICAL: Exclude ALL current-game outcome features (not knowable before game)
            # These are the RAW stats from the current game being predicted
            # We CAN use their ROLLING AVERAGES (e.g., opp_shots_rolling_5)
            'shots_against', 'total_shots_against', 'goals_against',
            'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
            'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
            'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
            'save_percentage', 'even_strength_save_pct', 'power_play_save_pct', 'short_handed_save_pct',
            # Current-game team/opponent stats (RAW values from this specific game)
            'opp_shots', 'opp_goals', 'opp_powerplay_goals', 'opp_powerplay_opportunities',
            'team_goals', 'team_shots', 'team_powerplay_goals', 'team_powerplay_opportunities',
            'team_shooting_pct', 'team_powerplay_pct',
            'team_hits', 'team_blocked_shots', 'team_pim', 'team_faceoff_win_pct',
            'pim',
            # Current-game shot quality stats (RAW values)
            'high_danger_saves', 'high_danger_shots_against', 'high_danger_goals_against', 'high_danger_save_pct',
            'mid_danger_saves', 'mid_danger_shots_against', 'mid_danger_goals_against', 'mid_danger_save_pct',
            'low_danger_saves', 'low_danger_shots_against', 'low_danger_goals_against', 'low_danger_save_pct',
            'total_xg_against', 'high_danger_xg_against', 'mid_danger_xg_against', 'low_danger_xg_against',
            'rebounds_created', 'rebound_rate', 'dangerous_rebound_pct',
            'avg_shot_distance', 'avg_shot_angle',
            'toi_seconds', 'saves_volatility_10',
            # CRITICAL: Exclude Corsi/Fenwick from CURRENT game (only use rolling averages)
            'team_corsi_for', 'team_corsi_against', 'team_corsi_for_pct',
            'team_fenwick_for', 'team_fenwick_against', 'team_fenwick_for_pct',
            'opp_blocked_shots',  # Used to calculate Corsi, but is current-game data
            # CRITICAL: Exclude game state from CURRENT game (only use rolling averages)
            'is_win', 'is_loss', 'goal_differential',
        ]

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Handle any remaining non-numeric columns
        X = df[feature_cols].copy()

        # Convert any object columns to category codes
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category').cat.codes

        # Fill any NaN values
        X = X.fillna(0)

        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Target distribution: OVER={y.sum()} ({y.mean()*100:.1f}%), UNDER={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")

        return X.values, y, feature_cols

    def split_data(self, df, X, y, test_size=0.2, val_size=0.15):
        """
        Split data CHRONOLOGICALLY to prevent temporal leakage

        Train on early season games, validate on mid-season, test on late season.
        This ensures the model only predicts future games it hasn't seen.

        Args:
            df: Original dataframe with game_date column
            X: Feature matrix
            y: Target vector
            test_size: Fraction for test set (most recent games)
            val_size: Fraction for validation set

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Splitting data CHRONOLOGICALLY (train=early season, test=late season)...")

        # Sort by date to ensure chronological split
        df_sorted = df.reset_index(drop=True)
        date_sorted_idx = df_sorted.sort_values('game_date').index.values

        n_samples = len(date_sorted_idx)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val

        # Chronological split
        train_idx = date_sorted_idx[:n_train]
        val_idx = date_sorted_idx[n_train:n_train+n_val]
        test_idx = date_sorted_idx[n_train+n_val:]

        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        # Log date ranges
        train_dates = df_sorted.iloc[train_idx]['game_date']
        val_dates = df_sorted.iloc[val_idx]['game_date']
        test_dates = df_sorted.iloc[test_idx]['game_date']

        logger.info(f"Train set: {len(train_idx)} samples ({train_dates.min()} to {train_dates.max()})")
        logger.info(f"Val set:   {len(val_idx)} samples ({val_dates.min()} to {val_dates.max()})")
        logger.info(f"Test set:  {len(test_idx)} samples ({test_dates.min()} to {test_dates.max()})")
        logger.info(f"\nTrain set: {len(X_train)} samples (OVER: {y_train.sum()}, UNDER: {len(y_train)-y_train.sum()})")
        logger.info(f"Val set:   {len(X_val)} samples (OVER: {y_val.sum()}, UNDER: {len(y_val)-y_val.sum()})")
        logger.info(f"Test set:  {len(X_test)} samples (OVER: {y_test.sum()}, UNDER: {len(y_test)-y_test.sum()})")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_train, y_train, X_val, y_val, params=None):
        """
        Train XGBoost classifier

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Optional hyperparameters

        Returns:
            Trained model
        """
        logger.info("Training XGBoost classifier...")

        # Default parameters for classification
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'n_estimators': 600,
            'max_depth': 4,
            'learning_rate': 0.012,
            'subsample': 0.9,
            'colsample_bytree': 1.0,
            'min_child_weight': 7,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'reg_lambda': 2.0,
            'random_state': self.config.get('model', {}).get('random_state', 42),
            'n_jobs': -1,
            'verbosity': 1
        }

        # Override with provided params
        if params:
            default_params.update(params)

        logger.info(f"Hyperparameters: {default_params}")

        # Train model
        self.model = xgb.XGBClassifier(**default_params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )

        logger.info("Training complete")

        return self.model

    def evaluate(self, X, y, dataset_name='Test'):
        """
        Evaluate model on dataset

        Args:
            X: Features
            y: True labels
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary of metrics
        """
        logger.info(f"\nEvaluating on {dataset_name} set...")

        # Predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]  # Probability of OVER
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        logloss = log_loss(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Classification report
        report = classification_report(y, y_pred, target_names=['UNDER', 'OVER'], output_dict=True)

        metrics = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'auc_roc': auc,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'precision_over': report['OVER']['precision'],
            'recall_over': report['OVER']['recall'],
            'f1_over': report['OVER']['f1-score'],
            'precision_under': report['UNDER']['precision'],
            'recall_under': report['UNDER']['recall'],
            'f1_under': report['UNDER']['f1-score']
        }

        # Log metrics
        logger.info(f"{dataset_name} Accuracy: {accuracy:.4f}")
        logger.info(f"{dataset_name} Log Loss: {logloss:.4f}")
        logger.info(f"{dataset_name} AUC-ROC: {auc:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn:4d}  |  FP: {fp:4d}")
        logger.info(f"  FN: {fn:4d}  |  TP: {tp:4d}")
        logger.info(f"\nClassification Report:")
        logger.info(f"  OVER  - Precision: {metrics['precision_over']:.3f}, Recall: {metrics['recall_over']:.3f}, F1: {metrics['f1_over']:.3f}")
        logger.info(f"  UNDER - Precision: {metrics['precision_under']:.3f}, Recall: {metrics['recall_under']:.3f}, F1: {metrics['f1_under']:.3f}")

        return metrics

    def save_model(self, output_dir='models'):
        """Save trained model and metadata"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / 'classifier_model.json'
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save feature names
        feature_names_path = output_dir / 'classifier_feature_names.json'
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Feature names saved to {feature_names_path}")

        # Save metadata
        metadata = {
            'model_type': 'binary_classifier',
            'objective': 'predict_over_under',
            'num_features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'xgboost_version': xgb.__version__
        }

        metadata_path = output_dir / 'classifier_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

    def save_metrics(self, train_metrics, val_metrics, test_metrics, output_dir='models/metadata'):
        """Save evaluation metrics"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
            'evaluation_date': datetime.now().isoformat()
        }

        metrics_path = output_dir / 'classifier_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")


def main():
    """Main training pipeline"""
    logger.info("="*70)
    logger.info("CLASSIFICATION MODEL TRAINING")
    logger.info("="*70)

    # Initialize trainer
    trainer = ClassifierTrainer()

    # Load data
    df = trainer.load_data()

    # Prepare features
    X, y, feature_names = trainer.prepare_features(df)
    trainer.feature_names = feature_names

    # Split data chronologically (df needed for game_date)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(df, X, y)

    # Train model
    model = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate
    train_metrics = trainer.evaluate(X_train, y_train, 'Train')
    val_metrics = trainer.evaluate(X_val, y_val, 'Validation')
    test_metrics = trainer.evaluate(X_test, y_test, 'Test')

    # Save model and metrics
    trainer.save_model()
    trainer.save_metrics(train_metrics, val_metrics, test_metrics)

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"Test Log Loss: {test_metrics['log_loss']:.4f}")


if __name__ == "__main__":
    main()
