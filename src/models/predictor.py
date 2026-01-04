"""Model prediction module

Loads trained model and generates predictions for upcoming games.
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class GoaliePredictor:
    """
    Load trained model and predict goalie saves for upcoming games
    """

    def __init__(self, model_path: Path, feature_names_path: Path):
        """
        Initialize predictor with trained model

        Args:
            model_path: Path to trained model pickle file
            feature_names_path: Path to feature names JSON file
        """
        self.model = None
        self.feature_names = None

        self._load_model(model_path, feature_names_path)

        logger.info("Predictor initialized")

    def _load_model(self, model_path: Path, feature_names_path: Path):
        """Load model and feature names from disk"""
        logger.info(f"Loading model from {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)

        logger.info(f"Model loaded successfully ({len(self.feature_names)} features)")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions

        Args:
            features: DataFrame with same features as training data

        Returns:
            Array of predicted saves
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Ensure features are in correct order
        features_ordered = features[self.feature_names]

        # Handle missing values
        features_ordered = features_ordered.fillna(0)

        # Handle infinite values
        features_ordered = features_ordered.replace([np.inf, -np.inf], 0)

        # Predict
        predictions = self.model.predict(features_ordered)

        return predictions

    def predict_single(self, feature_dict: Dict[str, float]) -> float:
        """
        Predict saves for a single goalie

        Args:
            feature_dict: Dictionary of feature name -> value

        Returns:
            Predicted number of saves
        """
        # Create DataFrame from dict
        features_df = pd.DataFrame([feature_dict])

        # Predict
        prediction = self.predict(features_df)[0]

        return prediction

    def predict_game(
        self,
        home_goalie_features: Dict[str, float],
        away_goalie_features: Dict[str, float],
        home_line: Optional[float] = None,
        away_line: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Predict saves for both goalies in a game

        Args:
            home_goalie_features: Features for home goalie
            away_goalie_features: Features for away goalie
            home_line: Betting line for home goalie (optional)
            away_line: Betting line for away goalie (optional)

        Returns:
            Dictionary with predictions and recommendations
        """
        # Predict for both goalies
        home_pred = self.predict_single(home_goalie_features)
        away_pred = self.predict_single(away_goalie_features)

        result = {
            'home': {
                'predicted_saves': home_pred,
                'betting_line': home_line,
                'difference': home_pred - home_line if home_line else None,
                'recommendation': None
            },
            'away': {
                'predicted_saves': away_pred,
                'betting_line': away_line,
                'difference': away_pred - away_line if away_line else None,
                'recommendation': None
            }
        }

        # Generate recommendations
        if home_line is not None:
            diff = home_pred - home_line
            if diff > 1.0:
                result['home']['recommendation'] = 'OVER'
            elif diff < -1.0:
                result['home']['recommendation'] = 'UNDER'
            else:
                result['home']['recommendation'] = 'NO BET (too close)'

        if away_line is not None:
            diff = away_pred - away_line
            if diff > 1.0:
                result['away']['recommendation'] = 'OVER'
            elif diff < -1.0:
                result['away']['recommendation'] = 'UNDER'
            else:
                result['away']['recommendation'] = 'NO BET (too close)'

        return result
