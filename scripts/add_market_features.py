"""
Add market disagreement features to classification training data

These features help the model identify when it might have an edge over the market.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def add_market_features(df):
    """
    Add features that capture disagreement between betting line and goalie stats.

    Args:
        df: DataFrame with betting_line and goalie features

    Returns:
        DataFrame with new market features
    """
    logger.info("Adding market disagreement features...")

    # Only process rows with betting lines
    has_line = df['betting_line'].notna()
    logger.info(f"Processing {has_line.sum()} rows with betting lines")

    # Feature 1: Line vs Recent Average
    # How far is the betting line from goalie's rolling average?
    if 'rolling_10_saves_mean' in df.columns:
        df['line_vs_recent_avg'] = df['betting_line'] - df['rolling_10_saves_mean']
        df['line_vs_recent_avg_pct'] = (df['betting_line'] - df['rolling_10_saves_mean']) / df['rolling_10_saves_mean']

    # Feature 2: Line vs Season Average
    if 'rolling_season_saves_mean' in df.columns:
        df['line_vs_season_avg'] = df['betting_line'] - df['rolling_season_saves_mean']
        df['line_vs_season_avg_pct'] = (df['betting_line'] - df['rolling_season_saves_mean']) / df['rolling_season_saves_mean']

    # Feature 3: Line Deviation (standardized)
    # How many standard deviations is the line from the rolling mean?
    if 'rolling_10_saves_mean' in df.columns and 'rolling_10_saves_std' in df.columns:
        df['line_surprise_score'] = (df['betting_line'] - df['rolling_10_saves_mean']) / (df['rolling_10_saves_std'] + 1e-6)

    # Feature 4: Recent Line Beat Rate
    # What % of last N games went OVER their betting line?
    # (This would require historical betting lines - skip for now since we don't have it)

    # Feature 5: Market Vig (if odds available)
    if 'odds_over_american' in df.columns and 'odds_under_american' in df.columns:
        from betting.odds_utils import american_to_implied_prob

        valid_odds = df['odds_over_american'].notna() & df['odds_under_american'].notna()

        df.loc[valid_odds, 'impl_prob_over'] = df.loc[valid_odds, 'odds_over_american'].apply(american_to_implied_prob)
        df.loc[valid_odds, 'impl_prob_under'] = df.loc[valid_odds, 'odds_under_american'].apply(american_to_implied_prob)
        df.loc[valid_odds, 'market_vig'] = df.loc[valid_odds, 'impl_prob_over'] + df.loc[valid_odds, 'impl_prob_under'] - 1.0

        # Feature 6: Vig-adjusted implied probability
        # Remove vig to get "fair" probabilities
        df.loc[valid_odds, 'fair_prob_over'] = df.loc[valid_odds, 'impl_prob_over'] / (df.loc[valid_odds, 'impl_prob_over'] + df.loc[valid_odds, 'impl_prob_under'])
        df.loc[valid_odds, 'fair_prob_under'] = df.loc[valid_odds, 'impl_prob_under'] / (df.loc[valid_odds, 'impl_prob_over'] + df.loc[valid_odds, 'impl_prob_under'])

    # Feature 7: Expected saves from opponent shots
    # If line is much higher/lower than opponent's usual shots, that's suspicious
    if 'opponent_rolling_5_shots_for_mean' in df.columns:
        df['line_vs_opp_shots'] = df['betting_line'] - df['opponent_rolling_5_shots_for_mean']
        df['line_vs_opp_shots_pct'] = (df['betting_line'] - df['opponent_rolling_5_shots_for_mean']) / df['opponent_rolling_5_shots_for_mean']

    # Feature 8: Line tightness indicator
    # Is the line at a round number (25.0) or half (24.5)?
    # Half-point lines often indicate market uncertainty
    df['line_is_half'] = (df['betting_line'] % 1.0 == 0.5).astype(int)

    # Feature 9: Extreme line indicator
    # Is the line unusually high or low?
    df['line_is_extreme_high'] = (df['betting_line'] >= 28).astype(int)
    df['line_is_extreme_low'] = (df['betting_line'] <= 22).astype(int)

    logger.info(f"Added market disagreement features")

    # Count new features
    new_features = [
        'line_vs_recent_avg', 'line_vs_recent_avg_pct',
        'line_vs_season_avg', 'line_vs_season_avg_pct',
        'line_surprise_score',
        'market_vig', 'impl_prob_over', 'impl_prob_under',
        'fair_prob_over', 'fair_prob_under',
        'line_vs_opp_shots', 'line_vs_opp_shots_pct',
        'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low'
    ]

    existing_new = [f for f in new_features if f in df.columns]
    logger.info(f"Successfully added {len(existing_new)} new features: {existing_new}")

    return df


def main():
    """Main execution"""
    logger.info("="*70)
    logger.info("ADDING MARKET DISAGREEMENT FEATURES")
    logger.info("="*70)

    # Load classification training data
    input_file = Path('data/processed/classification_training_data.parquet')
    logger.info(f"Loading from {input_file}")

    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")

    # Add market features
    df = add_market_features(df)

    # Save updated data
    output_file = Path('data/processed/classification_training_data.parquet')
    df.to_parquet(output_file, index=False)
    logger.info(f"\nSaved to {output_file}")
    logger.info(f"Final shape: {len(df)} samples, {len(df.columns)} features")

    logger.info("\n" + "="*70)
    logger.info("SUCCESS")
    logger.info("="*70)


if __name__ == "__main__":
    # Add betting module to path for odds_utils
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

    main()
