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
        random_state: int = 42,
        use_ewa: bool = False,
        ewa_span_multiplier: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data with time-based train/validation/test split

        CRITICAL: Uses chronological split, not random split
        CRITICAL: Recalculates rolling features for each split to prevent data leakage

        Args:
            df: Complete dataset with base features (NO rolling features yet)
            target_col: Name of target column
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Preparing data with time-based split...")
        logger.info("CRITICAL: Recalculating rolling features to prevent data leakage")

        # Sort by date to ensure chronological split
        df = df.sort_values('game_date').reset_index(drop=True)

        # Time-based split FIRST (before calculating rolling features)
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))

        logger.info(f"Train set: {val_idx} samples ({df['game_date'].iloc[0]} to {df['game_date'].iloc[val_idx-1]})")
        logger.info(f"Validation set: {test_idx - val_idx} samples ({df['game_date'].iloc[val_idx]} to {df['game_date'].iloc[test_idx-1]})")
        logger.info(f"Test set: {n - test_idx} samples ({df['game_date'].iloc[test_idx]} to {df['game_date'].iloc[-1]})")

        # Recalculate rolling features for each split to prevent leakage
        logger.info("Recalculating rolling features WITHOUT future data...")

        # Identify GOALIE rolling feature columns (to remove and recalculate)
        # Keep team/opponent rolling features - they're already properly calculated
        # ALSO keep Corsi/Fenwick rolling features - they're already properly calculated
        # ALSO keep opponent_ rolling features (e.g., opponent_shooting_pct_rolling_)
        goalie_rolling_cols = [col for col in df.columns
                              if ('_rolling_' in col or '_ewa_' in col)
                              and not col.startswith('team_defense_')
                              and not col.startswith('opp_offense_')
                              and not col.startswith('opponent_')
                              and not (('corsi' in col.lower() or 'fenwick' in col.lower()) and 'rolling' in col.lower())]
        logger.info(f"Found {len(goalie_rolling_cols)} goalie rolling features to recalculate")

        # Count team features being preserved
        team_rolling_cols = [col for col in df.columns
                           if ('_rolling_' in col or '_ewa_' in col or '_std_' in col)
                           and (col.startswith('team_defense_') or col.startswith('opp_offense_'))]
        logger.info(f"Preserving {len(team_rolling_cols)} team/opponent rolling features")

        # Count Corsi/Fenwick rolling features being preserved
        corsi_fenwick_rolling_cols = [col for col in df.columns
                                     if ('corsi' in col.lower() or 'fenwick' in col.lower()) and 'rolling' in col.lower()]
        logger.info(f"Preserving {len(corsi_fenwick_rolling_cols)} Corsi/Fenwick rolling features")

        # Count opponent rolling features being preserved
        opponent_rolling_cols = [col for col in df.columns
                                if col.startswith('opponent_') and '_rolling_' in col]
        logger.info(f"Preserving {len(opponent_rolling_cols)} opponent rolling features (e.g., shooting_pct)")

        # Get base columns (drop only goalie rolling features, keep team features)
        base_df = df.drop(columns=goalie_rolling_cols)

        # Recalculate rolling features properly
        df_with_proper_rolling = self._recalculate_rolling_features(
            base_df,
            train_idx=val_idx,
            val_idx=test_idx,
            use_ewa=use_ewa,
            ewa_span_multiplier=ewa_span_multiplier
        )

        # Now identify feature columns (exclude metadata, target, AND current-game outcomes)
        exclude_cols = [
            # Metadata
            'goalie_id', 'game_id', 'game_date', 'season', 'team_abbrev',
            'opponent_team', 'toi', 'decision',
            # Target variable
            target_col,
            # Constant features (all training samples have same value)
            'is_starter',  # Always True in training data (we filter to starters only)
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
            # ZERO-IMPORTANCE FEATURES (remove to reduce noise)
            # Powerplay features (14 features with 0.0 importance)
            'team_defense_team_powerplay_goals_rolling_3',
            'team_defense_team_powerplay_goals_rolling_5',
            'team_defense_team_powerplay_goals_rolling_10',
            'team_defense_team_powerplay_opportunities_rolling_3',
            'team_defense_team_powerplay_opportunities_rolling_5',
            'team_defense_team_powerplay_opportunities_rolling_10',
            'team_defense_opp_powerplay_opportunities_rolling_3',
            'team_defense_opp_powerplay_opportunities_rolling_5',
            'team_defense_opp_powerplay_opportunities_rolling_10',
            'opp_offense_team_powerplay_goals_rolling_3',
            'opp_offense_team_powerplay_goals_rolling_5',
            'opp_offense_team_powerplay_goals_rolling_10',
            'opp_offense_team_powerplay_opportunities_rolling_3',
            'opp_offense_team_powerplay_opportunities_rolling_5',
            'opp_offense_team_powerplay_opportunities_rolling_10',
            # Physical play features (12 features with 0.0 importance)
            'team_defense_team_blocked_shots_rolling_3',
            'team_defense_team_blocked_shots_rolling_5',
            'team_defense_team_blocked_shots_rolling_10',
            'team_defense_team_hits_rolling_3',
            'team_defense_team_hits_rolling_5',
            'team_defense_team_hits_rolling_10',
            'opp_offense_team_blocked_shots_rolling_3',
            'opp_offense_team_blocked_shots_rolling_5',
            'opp_offense_team_blocked_shots_rolling_10',
            'opp_offense_team_hits_rolling_3',
            'opp_offense_team_hits_rolling_5',
            'opp_offense_team_hits_rolling_10',
            # Goalie back-to-back (1 feature with 0.0 importance)
            'goalie_is_back_to_back'
        ]

        # Build feature list with special handling for team/opponent rolling features
        feature_cols = []
        for col in df_with_proper_rolling.columns:
            # ALWAYS include team defense and opponent offense rolling features
            # These are historical averages, not current-game outcomes
            if col.startswith('team_defense_') or col.startswith('opp_offense_'):
                feature_cols.append(col)
            # ALWAYS include Corsi/Fenwick rolling features (safe historical averages)
            elif ('corsi' in col.lower() or 'fenwick' in col.lower()) and 'rolling' in col.lower():
                feature_cols.append(col)
            # ALWAYS include opponent rolling features (e.g., opponent_shooting_pct_rolling_)
            elif col.startswith('opponent_') and '_rolling_' in col:
                feature_cols.append(col)
            # For other columns, check if they're in the exclusion list
            elif col not in exclude_cols:
                feature_cols.append(col)

        # Store feature names
        self.feature_names = feature_cols

        logger.info(f"Using {len(feature_cols)} features for training")
        logger.info(f"  - Team defense features: {len([c for c in feature_cols if c.startswith('team_defense_')])}")
        logger.info(f"  - Opponent offense features: {len([c for c in feature_cols if c.startswith('opp_offense_')])}")
        logger.info(f"  - Goalie features: {len([c for c in feature_cols if not c.startswith('team_defense_') and not c.startswith('opp_offense_')])}")

        # Extract features and target
        X = df_with_proper_rolling[feature_cols].copy()
        y = df_with_proper_rolling[target_col].copy()

        # Split into train/val/test
        X_train = X.iloc[:val_idx]
        X_val = X.iloc[val_idx:test_idx]
        X_test = X.iloc[test_idx:]

        y_train = y.iloc[:val_idx]
        y_val = y.iloc[val_idx:test_idx]
        y_test = y.iloc[test_idx:]

        # Handle missing values
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)

        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_val = X_val.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _recalculate_rolling_features(
        self,
        df: pd.DataFrame,
        train_idx: int,
        val_idx: int,
        use_ewa: bool = False,
        ewa_span_multiplier: float = 1.0
    ) -> pd.DataFrame:
        """
        Recalculate rolling features WITHOUT data leakage

        For each game, rolling features use ONLY data from prior games.

        Args:
            df: DataFrame with base features (no rolling features)
            train_idx: Index where training set ends
            val_idx: Index where validation set ends
            use_ewa: If True, use exponential weighted averages instead of simple rolling
            ewa_span_multiplier: Multiplier for EWA span (1.0 = span equals window size)

        Returns:
            DataFrame with properly calculated rolling features
        """
        if use_ewa:
            logger.info(f"Calculating EWA features (span_multiplier={ewa_span_multiplier}) with proper time-based windows...")
        else:
            logger.info("Calculating rolling features with proper time-based windows...")

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

        # Calculate rolling averages OR EWA for each goalie
        for stat in goalie_stats:
            for window in windows:
                col_name = f"{stat}_rolling_{window}"

                if use_ewa:
                    # Exponential weighted average (recent games weighted more)
                    span = window * ewa_span_multiplier
                    df_result[col_name] = df_result.groupby('goalie_id')[stat].transform(
                        lambda x: x.ewm(span=span, adjust=False, min_periods=1).mean().shift(1)
                    )
                else:
                    # Simple rolling mean
                    df_result[col_name] = df_result.groupby('goalie_id')[stat].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                    )

        logger.info(f"Recalculated {len(goalie_stats) * len(windows)} {'EWA' if use_ewa else 'rolling'} features")

        return df_result

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
