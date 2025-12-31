"""Model training module for XGBoost regressor

Trains a regression model to predict the number of saves a goalie will make.
At prediction time, compare predicted saves to betting line to determine over/under.
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class GoalieModelTrainer:
    """
    Train XGBoost regression model to predict goalie saves

    Uses time-based train/validation/test split to prevent data leakage.
    Predicts actual number of saves (not over/under classification).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_importance = None

        logger.info("Model trainer initialized")

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'saves',
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data with time-based train/validation/test split

        CRITICAL: Uses chronological split, not random split

        Args:
            df: Complete dataset
            target_col: Name of target column
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Preparing data with time-based split...")

        # Sort by date to ensure chronological split
        df = df.sort_values('game_date').reset_index(drop=True)

        # Identify feature columns (exclude metadata and target)
        exclude_cols = [
            'goalie_id', 'game_id', 'game_date', 'season', 'team_abbrev',
            'opponent_team', 'toi', 'decision', target_col
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Store feature names
        self.feature_names = feature_cols

        logger.info(f"Using {len(feature_cols)} features for training")

        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Time-based split
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))

        X_train = X.iloc[:val_idx]
        X_val = X.iloc[val_idx:test_idx]
        X_test = X.iloc[test_idx:]

        y_train = y.iloc[:val_idx]
        y_val = y.iloc[val_idx:test_idx]
        y_test = y.iloc[test_idx:]

        logger.info(f"Train set: {len(X_train)} samples ({df['game_date'].iloc[:val_idx].min()} to {df['game_date'].iloc[val_idx-1]})")
        logger.info(f"Validation set: {len(X_val)} samples ({df['game_date'].iloc[val_idx]} to {df['game_date'].iloc[test_idx-1]})")
        logger.info(f"Test set: {len(X_test)} samples ({df['game_date'].iloc[test_idx]} to {df['game_date'].iloc[-1]})")

        # Handle missing values
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)

        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_val = X_val.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **xgb_params
    ) -> XGBRegressor:
        """
        Train XGBoost regressor

        Args:
            X_train: Training features
            y_train: Training targets (saves)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **xgb_params: Additional XGBoost parameters

        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost regression model...")

        # Default XGBoost parameters for regression
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.config.get('model', {}).get('random_state', 42),
            'n_jobs': -1,
            'verbosity': 1
        }

        # Override with provided parameters
        default_params.update(xgb_params)

        logger.info(f"XGBoost parameters: {default_params}")

        # Initialize model
        self.model = XGBRegressor(**default_params)

        # Prepare evaluation set if validation data provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        # Train model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("Model training complete")
        logger.info(f"Top 10 features by importance:")
        for idx, row in self.feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return self.model

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X: Features
            y: True targets (actual saves)
            dataset_name: Name of dataset (for logging)

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Get predictions
        y_pred = self.model.predict(X)

        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mean_actual': y.mean(),
            'mean_predicted': y_pred.mean(),
            'std_actual': y.std(),
            'std_predicted': y_pred.std()
        }

        logger.info(f"\n{dataset_name} Set Metrics:")
        logger.info(f"  RMSE: {metrics['rmse']:.3f} saves")
        logger.info(f"  MAE: {metrics['mae']:.3f} saves")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  Mean Actual: {metrics['mean_actual']:.2f} saves")
        logger.info(f"  Mean Predicted: {metrics['mean_predicted']:.2f} saves")

        return metrics

    def save_model(self, model_dir: Path, model_name: str = "xgboost_goalie_model"):
        """
        Save trained model and metadata

        Args:
            model_dir: Directory to save model
            model_name: Base name for model files
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"Model saved to {model_path}")

        # Save feature names
        feature_path = model_dir / f"{model_name}_features.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)

        logger.info(f"Feature names saved to {feature_path}")

        # Save feature importance
        importance_path = model_dir / f"{model_name}_feature_importance.csv"
        self.feature_importance.to_csv(importance_path, index=False)

        logger.info(f"Feature importance saved to {importance_path}")

        # Save metadata
        metadata = {
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'model_params': self.model.get_params(),
            'xgboost_version': self.model.__version__ if hasattr(self.model, '__version__') else 'unknown'
        }

        metadata_path = model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

    def load_model(self, model_dir: Path, model_name: str = "xgboost_goalie_model"):
        """
        Load trained model and metadata

        Args:
            model_dir: Directory containing model files
            model_name: Base name of model files
        """
        model_dir = Path(model_dir)

        # Load model
        model_path = model_dir / f"{model_name}.pkl"
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        logger.info(f"Model loaded from {model_path}")

        # Load feature names
        feature_path = model_dir / f"{model_name}_features.json"
        with open(feature_path, 'r') as f:
            self.feature_names = json.load(f)

        logger.info(f"Feature names loaded from {feature_path}")

        # Load feature importance
        importance_path = model_dir / f"{model_name}_feature_importance.csv"
        self.feature_importance = pd.read_csv(importance_path)

        logger.info(f"Feature importance loaded from {importance_path}")
