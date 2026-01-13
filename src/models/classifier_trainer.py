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

        return X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx

    def train(self, X_train, y_train, X_val, y_val, params=None, sample_weight=None):
        """
        Train XGBoost classifier

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Optional hyperparameters
            sample_weight: Optional sample weights for training

        Returns:
            Trained model
        """
        logger.info("Training XGBoost classifier...")

        # Default parameters for classification (conservative: profitable at 3% EV threshold)
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'n_estimators': 800,  # More trees with slower learning
            'max_depth': 3,  # Shallower trees to reduce overfitting
            'learning_rate': 0.01,  # Slower learning for better generalization
            'subsample': 0.8,  # Sample 80% of data per tree
            'colsample_bytree': 0.8,  # Sample 80% of features per tree
            'min_child_weight': 20,  # Require more samples per leaf
            'gamma': 10,  # Much higher pruning threshold
            'reg_alpha': 15,  # Stronger L1 regularization
            'reg_lambda': 30,  # Stronger L2 regularization
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

        # Log sample weight usage
        if sample_weight is not None:
            logger.info(f"Training with sample weights (mean={sample_weight.mean():.2f})")

        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
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

    def calculate_sample_weights(self, df, split_idx):
        """
        Calculate sample weights based on market efficiency (vig).

        Strategy:
        - Sharp lines (low vig <5%): Higher weight = 1.5 (trustworthy market signal)
        - Soft lines (high vig >10%): Lower weight = 0.8 (uncertain market)
        - Missing odds: Default weight = 1.0

        Args:
            df: DataFrame with odds columns
            split_idx: Indices for the split (train/val/test)

        Returns:
            np.array: Sample weights for XGBoost
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from betting.odds_utils import american_to_implied_prob

        weights = np.ones(len(split_idx))
        df_split = df.iloc[split_idx]

        for i, (idx, row) in enumerate(df_split.iterrows()):
            odds_over = row.get('odds_over_american')
            odds_under = row.get('odds_under_american')

            # Skip if no odds (use default weight = 1.0)
            if pd.isna(odds_over) or pd.isna(odds_under):
                continue

            # Calculate implied probabilities
            impl_prob_over = american_to_implied_prob(odds_over)
            impl_prob_under = american_to_implied_prob(odds_under)

            # Calculate vig (market overround)
            vig = impl_prob_over + impl_prob_under - 1.0

            # Weight by market sharpness
            if vig < 0.05:  # <5% vig = sharp line
                weights[i] = 1.5
            elif vig > 0.10:  # >10% vig = soft line
                weights[i] = 0.8

        logger.info(f"Sample weights calculated (mean={weights.mean():.2f}, std={weights.std():.2f})")
        sharp_count = np.sum(weights > 1.2)
        soft_count = np.sum(weights < 0.9)
        logger.info(f"  Sharp lines (weight=1.5): {sharp_count} ({sharp_count/len(weights)*100:.1f}%)")
        logger.info(f"  Soft lines (weight=0.8): {soft_count} ({soft_count/len(weights)*100:.1f}%)")

        return weights

    def evaluate_profitability(self, X, y, df, split_idx, dataset_name='Test', ev_threshold=0.02):
        """
        Evaluate model performance on betting profitability metrics.

        Backtests using actual historical odds to calculate real profits.

        Args:
            X: Feature matrix
            y: True labels
            df: Original DataFrame with odds columns
            split_idx: Indices for the split
            dataset_name: Name for logging
            ev_threshold: Minimum EV required to place bet (default 2%)

        Returns:
            dict: Profitability metrics
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from betting.odds_utils import calculate_ev, calculate_payout

        logger.info(f"\nEvaluating betting profitability on {dataset_name} set...")

        # Get predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]  # Probability of OVER

        df_split = df.iloc[split_idx].reset_index(drop=True)

        results = []
        skipped_no_odds = 0

        for i in range(len(y_pred_proba)):
            prob_over = y_pred_proba[i]
            prob_under = 1 - prob_over
            actual_over = y[i]  # 1 if OVER hit, 0 if UNDER hit

            odds_over = df_split.iloc[i].get('odds_over_american')
            odds_under = df_split.iloc[i].get('odds_under_american')

            # Skip if no odds available
            if pd.isna(odds_over) or pd.isna(odds_under):
                skipped_no_odds += 1
                continue

            # Calculate EV for both sides
            ev_over = calculate_ev(prob_over, odds_over)
            ev_under = calculate_ev(prob_under, odds_under)

            # Make bet decision based on EV threshold
            if ev_over >= ev_threshold and ev_over > ev_under:
                bet = 'OVER'
                won = (actual_over == 1)
                profit = calculate_payout(1.0, odds_over, won)
                ev = ev_over
            elif ev_under >= ev_threshold:
                bet = 'UNDER'
                won = (actual_over == 0)
                profit = calculate_payout(1.0, odds_under, won)
                ev = ev_under
            else:
                continue  # NO BET

            results.append({
                'bet': bet,
                'profit': profit,
                'won': won,
                'ev': ev
            })

        # Calculate metrics
        if len(results) == 0:
            logger.warning(f"  No bets placed on {dataset_name} set (no +EV opportunities)")
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'roi': 0.0,
                'avg_ev': 0.0,
                'skipped_no_odds': skipped_no_odds
            }

        total_bets = len(results)
        wins = sum(r['won'] for r in results)
        losses = total_bets - wins
        win_rate = wins / total_bets
        total_profit = sum(r['profit'] for r in results)
        roi = (total_profit / total_bets) * 100  # ROI as percentage
        avg_ev = np.mean([r['ev'] for r in results])

        logger.info(f"  Total bets: {total_bets}")
        logger.info(f"  Wins: {wins}, Losses: {losses}")
        logger.info(f"  Win rate: {win_rate*100:.1f}%")
        logger.info(f"  Total profit: {total_profit:+.2f} units")
        logger.info(f"  ROI: {roi:+.2f}%")
        logger.info(f"  Avg EV when betting: {avg_ev*100:+.1f}%")
        logger.info(f"  Games skipped (no odds): {skipped_no_odds}")

        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': roi,
            'avg_ev': avg_ev,
            'skipped_no_odds': skipped_no_odds
        }

    def test_ev_thresholds(self, X, y, df, split_idx, dataset_name='Test', thresholds=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10]):
        """
        Test multiple EV thresholds to find optimal betting strategy.

        Args:
            X: Feature matrix
            y: True labels
            df: Original DataFrame with odds columns
            split_idx: Indices for the split
            dataset_name: Name for logging
            thresholds: List of EV thresholds to test

        Returns:
            dict: Results for each threshold
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing EV Thresholds on {dataset_name} Set")
        logger.info(f"{'='*70}")

        results = {}
        for threshold in thresholds:
            metrics = self.evaluate_profitability(X, y, df, split_idx, dataset_name, ev_threshold=threshold)
            results[threshold] = metrics

        # Print summary table
        logger.info(f"\n{'='*70}")
        logger.info(f"EV THRESHOLD COMPARISON - {dataset_name} Set")
        logger.info(f"{'='*70}")
        logger.info(f"{'Threshold':>10} | {'Bets':>6} | {'Win Rate':>9} | {'ROI':>8} | {'Total P/L':>10}")
        logger.info(f"{'-'*70}")

        for threshold in thresholds:
            m = results[threshold]
            if m['total_bets'] > 0:
                logger.info(
                    f"{threshold*100:>9.0f}% | {m['total_bets']:>6} | "
                    f"{m['win_rate']*100:>8.1f}% | {m['roi']:>7.2f}% | "
                    f"{m['total_profit']:>9.2f} units"
                )
            else:
                logger.info(f"{threshold*100:>9.0f}% | {'NO BETS':>6} |")

        # Find best threshold by ROI
        profitable = {t: m for t, m in results.items() if m['total_bets'] > 0 and m['roi'] > 0}
        if profitable:
            best_threshold = max(profitable.keys(), key=lambda t: profitable[t]['roi'])
            best_metrics = profitable[best_threshold]
            logger.info(f"\n{'='*70}")
            logger.info(f"BEST THRESHOLD: {best_threshold*100:.0f}% EV")
            logger.info(f"  ROI: {best_metrics['roi']:+.2f}%")
            logger.info(f"  Win Rate: {best_metrics['win_rate']*100:.1f}%")
            logger.info(f"  Total Bets: {best_metrics['total_bets']}")
            logger.info(f"  Total Profit: {best_metrics['total_profit']:+.2f} units")
            logger.info(f"{'='*70}")
        else:
            logger.info(f"\n{'='*70}")
            logger.info(f"NO PROFITABLE THRESHOLD FOUND")
            logger.info(f"{'='*70}")

        return results

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

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        metrics = {
            'train': convert_to_native(train_metrics),
            'validation': convert_to_native(val_metrics),
            'test': convert_to_native(test_metrics),
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
    X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = trainer.split_data(df, X, y)

    # Calculate sample weights for training set
    sample_weights = trainer.calculate_sample_weights(df, train_idx)

    # Train model with sample weights
    model = trainer.train(X_train, y_train, X_val, y_val, sample_weight=sample_weights)

    # Evaluate on accuracy metrics
    train_metrics = trainer.evaluate(X_train, y_train, 'Train')
    val_metrics = trainer.evaluate(X_val, y_val, 'Validation')
    test_metrics = trainer.evaluate(X_test, y_test, 'Test')

    # Evaluate on profitability metrics (with real odds at 2% EV threshold)
    train_profit = trainer.evaluate_profitability(X_train, y_train, df, train_idx, 'Train')
    val_profit = trainer.evaluate_profitability(X_val, y_val, df, val_idx, 'Validation')
    test_profit = trainer.evaluate_profitability(X_test, y_test, df, test_idx, 'Test')

    # Test multiple EV thresholds on test set to find optimal strategy
    ev_threshold_results = trainer.test_ev_thresholds(
        X_test, y_test, df, test_idx, 'Test',
        thresholds=[0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    )

    # Combine metrics
    train_metrics.update({'profitability': train_profit})
    val_metrics.update({'profitability': val_profit})
    test_metrics.update({'profitability': test_profit})

    # Save model and metrics
    trainer.save_model()
    trainer.save_metrics(train_metrics, val_metrics, test_metrics)

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"Test Log Loss: {test_metrics['log_loss']:.4f}")
    logger.info(f"Test ROI (2% EV): {test_profit['roi']:+.2f}%")


if __name__ == "__main__":
    main()
