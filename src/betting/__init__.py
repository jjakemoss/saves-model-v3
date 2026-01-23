"""
Betting module for NHL goalie saves predictions and tracking
"""
from .feature_calculator import BettingFeatureCalculator
from .predictor import BettingPredictor
from .excel_manager import BettingTracker
from .nhl_fetcher import NHLBettingData
from .metrics import calculate_performance_metrics, format_metrics_report
from .odds_utils import (
    american_to_implied_prob,
    american_to_decimal,
    calculate_ev,
    validate_american_odds,
    calculate_payout
)
from .odds_fetcher import UnderdogFetcher, PrizePicksFetcher, extract_last_name

__all__ = [
    'BettingFeatureCalculator',
    'BettingPredictor',
    'BettingTracker',
    'NHLBettingData',
    'calculate_performance_metrics',
    'format_metrics_report',
    'american_to_implied_prob',
    'american_to_decimal',
    'calculate_ev',
    'validate_american_odds',
    'calculate_payout',
    'UnderdogFetcher',
    'PrizePicksFetcher',
    'extract_last_name',
]
