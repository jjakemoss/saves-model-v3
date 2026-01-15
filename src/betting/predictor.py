"""
Betting predictor - wrapper around trained XGBoost model
"""
import xgboost as xgb
import numpy as np
from pathlib import Path
from .odds_utils import calculate_ev


class BettingPredictor:
    """Make predictions using trained classifier model"""

    def __init__(self, model_path='models/trained/config_4398_ev2pct_20260115_103430/classifier_model.json', feature_order_path='models/trained/config_4398_ev2pct_20260115_103430/classifier_feature_names.json'):
        """
        Initialize predictor with trained model

        Args:
            model_path: Path to trained XGBoost model JSON file
            feature_order_path: Path to file containing exact feature order from training
        """
        self.model_path = Path(model_path)
        self.feature_order_path = Path(feature_order_path)
        self.model = None
        self.feature_order = None
        self._load_model()
        self._load_feature_order()

    def _load_model(self):
        """Load the trained XGBoost classifier"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load as Booster (core XGBoost class) for JSON format models
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))

    def _load_feature_order(self):
        """Load the expected feature order from training"""
        import json

        if not self.feature_order_path.exists():
            raise FileNotFoundError(f"Feature order file not found: {self.feature_order_path}")

        # Check if it's JSON format
        try:
            with open(self.feature_order_path, 'r') as f:
                content = f.read()
                # Try to parse as JSON first
                if content.strip().startswith('['):
                    self.feature_order = json.loads(content)
                    return
        except json.JSONDecodeError:
            pass

        # Fall back to text format
        # Format: "  1. is_home", "  2. saves_rolling_3", etc.
        with open(self.feature_order_path, 'r') as f:
            lines = f.readlines()

        self.feature_order = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Extract feature name after the number and dot
            # Example: "1. is_home" -> "is_home"
            parts = line.split('.', 1)
            if len(parts) == 2:
                feature_name = parts[1].strip()
                self.feature_order.append(feature_name)

    def predict(self, features_df, betting_line=None, line_over_odds=None, line_under_odds=None):
        """
        Generate prediction for a single game

        Args:
            features_df: pd.DataFrame with single row of features
            betting_line: Optional betting line (saves o/u) for estimating predicted saves
            line_over_odds: American odds for OVER (e.g., -115)
            line_under_odds: American odds for UNDER (e.g., -105)

        Returns:
            dict: {
                'predicted_saves': float,
                'prob_over': float (0-1),
                'confidence_pct': float (0-100),
                'confidence_bucket': str,
                'recommendation': str (OVER/UNDER/NO BET),
                'ev_over': float or None (EV for OVER side),
                'ev_under': float or None (EV for UNDER side),
                'recommended_ev': float or None (EV of recommended bet)
            }
        """
        # Remove features not used in training (data leakage and metadata)
        features_to_remove = [
            # Market-derived features (data leakage)
            'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
            'market_vig', 'impl_prob_over', 'impl_prob_under',
            'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
            'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low',
            # Metadata columns
            'game_id', 'goalie_id', 'game_date', 'over_hit',
            'odds_over_american', 'odds_under_american',
            'odds_over_decimal', 'odds_under_decimal', 'num_books',
            'team_abbrev', 'opponent_team', 'toi', 'season',
            # Actual game results (not available before game)
            'saves', 'shots_against', 'goals_against', 'save_percentage',
            'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
            'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
            'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
            'team_goals', 'team_shots', 'opp_goals', 'opp_shots', 'line_margin'
        ]
        features_cleaned = features_df.drop(columns=[col for col in features_to_remove if col in features_df.columns], errors='ignore')

        # CRITICAL: Only use features that are available in the input
        # The model was trained with 103 features, but we only have 90 base features during prediction
        # Filter feature_order to only include features we actually have
        available_features = [f for f in self.feature_order if f in features_cleaned.columns]

        # Check if we have all the base features (should be ~90)
        if len(available_features) < 85:  # Safety check: should have at least 85 features
            missing_features = [f for f in self.feature_order if f not in features_cleaned.columns and f not in features_to_remove]
            raise ValueError(f"Missing required base features: {missing_features}")

        # Reorder columns to match training order (using only available features)
        features_ordered = features_cleaned[available_features]

        # Get probability predictions using DMatrix (for Booster interface)
        dmatrix = xgb.DMatrix(features_ordered)
        prob_over = self.model.predict(dmatrix)[0]

        # Calculate confidence (distance from 0.5)
        confidence = abs(prob_over - 0.5)
        confidence_pct = confidence * 200  # Convert to 0-100 scale

        # Determine confidence bucket
        confidence_bucket = self.get_confidence_bucket(confidence)

        # Make recommendation using EV-based logic
        recommendation, ev_over, ev_under, recommended_ev = self._determine_recommendation(
            prob_over,
            line_over_odds,
            line_under_odds
        )

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
            'recommendation': recommendation,
            'ev_over': ev_over,
            'ev_under': ev_under,
            'recommended_ev': recommended_ev
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

    def _determine_recommendation(self, prob_over, line_over_odds, line_under_odds, ev_threshold=0.02):
        """
        Determine bet recommendation using Expected Value (2% minimum).

        Args:
            prob_over: Model probability of OVER
            line_over_odds: American odds for OVER (e.g., -115)
            line_under_odds: American odds for UNDER (e.g., -105)
            ev_threshold: Minimum EV required (default 0.02 = 2%)

        Returns:
            tuple: (recommendation, ev_over, ev_under, recommended_ev)

        Logic:
            1. Calculate EV for both sides (if odds provided)
            2. Recommend side with EV >= 2% AND higher EV than other side
            3. If no odds provided, fall back to probability thresholds (backwards compatible)
            4. Return NO BET if neither side meets criteria
        """
        ev_over = None
        ev_under = None
        prob_under = 1 - prob_over

        # Calculate EV if odds are provided
        if line_over_odds is not None:
            ev_over = calculate_ev(prob_over, line_over_odds)

        if line_under_odds is not None:
            ev_under = calculate_ev(prob_under, line_under_odds)

        # If we have EV calculations, use them
        if ev_over is not None or ev_under is not None:
            over_qualifies = ev_over is not None and ev_over >= ev_threshold
            under_qualifies = ev_under is not None and ev_under >= ev_threshold

            if over_qualifies and under_qualifies:
                # Both qualify - pick higher EV
                if ev_over > ev_under:
                    return 'OVER', ev_over, ev_under, ev_over
                else:
                    return 'UNDER', ev_over, ev_under, ev_under
            elif over_qualifies:
                return 'OVER', ev_over, ev_under, ev_over
            elif under_qualifies:
                return 'UNDER', ev_over, ev_under, ev_under
            else:
                # Neither qualifies
                recommended_ev = max(ev_over or -999, ev_under or -999)
                return 'NO BET', ev_over, ev_under, recommended_ev if recommended_ev > -999 else None

        # Fallback to probability thresholds if no odds provided (backwards compatibility)
        if prob_over > 0.55:
            return 'OVER', None, None, None
        elif prob_over < 0.45:
            return 'UNDER', None, None, None
        else:
            return 'NO BET', None, None, None

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
