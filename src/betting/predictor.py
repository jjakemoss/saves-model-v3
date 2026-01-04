"""
Betting predictor - wrapper around trained XGBoost model
"""
import xgboost as xgb
import numpy as np
from pathlib import Path


class BettingPredictor:
    """Make predictions using trained classifier model"""

    def __init__(self, model_path='models/classifier_model.json'):
        """
        Initialize predictor with trained model

        Args:
            model_path: Path to trained XGBoost model JSON file
        """
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained XGBoost classifier"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))

    def predict(self, features_df, betting_line=None):
        """
        Generate prediction for a single game

        Args:
            features_df: pd.DataFrame with single row of 89 features
            betting_line: Optional betting line (saves o/u) for estimating predicted saves

        Returns:
            dict: {
                'predicted_saves': float,
                'prob_over': float (0-1),
                'confidence_pct': float (0-100),
                'confidence_bucket': str,
                'recommendation': str (OVER/UNDER/NO BET)
            }
        """
        # Get probability predictions
        prob_over = self.model.predict_proba(features_df)[0, 1]

        # Calculate confidence (distance from 0.5)
        confidence = abs(prob_over - 0.5)
        confidence_pct = confidence * 200  # Convert to 0-100 scale

        # Determine confidence bucket
        confidence_bucket = self.get_confidence_bucket(confidence)

        # Make recommendation
        if prob_over > 0.55:
            recommendation = 'OVER'
        elif prob_over < 0.45:
            recommendation = 'UNDER'
        else:
            recommendation = 'NO BET'

        # Estimate predicted saves based on betting line and probability
        # If prob_over = 0.6 and line = 25.5, estimate ~26.5 saves (slightly over)
        # If prob_over = 0.4 and line = 25.5, estimate ~24.5 saves (slightly under)
        if betting_line is not None:
            # Map probability to estimated distance from line
            # prob_over 0.5 -> 0 difference, prob_over 0.75 -> +2.5, prob_over 0.25 -> -2.5
            estimated_offset = (prob_over - 0.5) * 5  # Scale factor of 5 gives reasonable range
            predicted_saves = round(betting_line + estimated_offset, 1)
        else:
            # If no betting line provided, use league average
            predicted_saves = 25.0

        return {
            'predicted_saves': predicted_saves,
            'prob_over': prob_over,
            'confidence_pct': confidence_pct,
            'confidence_bucket': confidence_bucket,
            'recommendation': recommendation
        }

    def get_confidence_bucket(self, confidence):
        """
        Map confidence to bucket label

        Args:
            confidence: float (0-0.5 range, distance from 0.5)

        Returns:
            str: Confidence bucket label
        """
        if confidence < 0.05:
            return '50-55%'
        elif confidence < 0.10:
            return '55-60%'
        elif confidence < 0.15:
            return '60-65%'
        elif confidence < 0.20:
            return '65-70%'
        elif confidence < 0.25:
            return '70-75%'
        else:
            return '75%+'

    def predict_batch(self, features_df_list):
        """
        Generate predictions for multiple games

        Args:
            features_df_list: List of pd.DataFrame, each with single row

        Returns:
            list: List of prediction dicts
        """
        predictions = []
        for features_df in features_df_list:
            pred = self.predict(features_df)
            predictions.append(pred)

        return predictions
