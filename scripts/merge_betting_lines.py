"""
Merge betting lines with training data and create classification dataset
"""
import pandas as pd
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_betting_lines():
    """Load betting lines from JSON file"""
    lines_file = Path('data/raw/betting_lines/betting_lines.json')

    logger.info(f"Loading betting lines from {lines_file}")

    with open(lines_file, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame and expand home/away into separate rows
    records = []

    for entry in data:
        game_id = entry['game_id']
        game_date = entry['game_date']

        # Home goalie entry
        if entry.get('home_goalie_id') and entry.get('home_goalie_line') is not None:
            records.append({
                'game_id': game_id,
                'goalie_id': entry['home_goalie_id'],
                'betting_line': entry['home_goalie_line'],
                'odds_over_american': entry.get('home_odds_over_american'),
                'odds_under_american': entry.get('home_odds_under_american'),
                'odds_over_decimal': entry.get('home_odds_over_decimal'),
                'odds_under_decimal': entry.get('home_odds_under_decimal'),
                'num_books': entry.get('home_num_books'),
                'game_date': game_date
            })

        # Away goalie entry
        if entry.get('away_goalie_id') and entry.get('away_goalie_line') is not None:
            records.append({
                'game_id': game_id,
                'goalie_id': entry['away_goalie_id'],
                'betting_line': entry['away_goalie_line'],
                'odds_over_american': entry.get('away_odds_over_american'),
                'odds_under_american': entry.get('away_odds_under_american'),
                'odds_over_decimal': entry.get('away_odds_over_decimal'),
                'odds_under_decimal': entry.get('away_odds_under_decimal'),
                'num_books': entry.get('away_num_books'),
                'game_date': game_date
            })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} goalie betting lines from {len(data)} games")

    return df


def load_training_data():
    """Load clean training data"""
    training_file = Path('data/processed/clean_training_data.parquet')

    logger.info(f"Loading training data from {training_file}")
    df = pd.read_parquet(training_file)

    logger.info(f"Loaded {len(df)} training samples")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    # Show season breakdown
    season_counts = df['season'].value_counts().sort_index()
    logger.info(f"Season breakdown:\n{season_counts}")

    return df


def merge_datasets(training_df, betting_lines_df):
    """
    Merge training data with betting lines

    Args:
        training_df: Training data DataFrame
        betting_lines_df: Betting lines DataFrame

    Returns:
        Merged DataFrame with betting_line column
    """
    logger.info("\nMerging datasets...")
    logger.info(f"Training data before merge: {len(training_df)} rows")
    logger.info(f"Betting lines: {len(betting_lines_df)} rows")

    # Merge on game_id and goalie_id (include odds columns)
    merged_df = training_df.merge(
        betting_lines_df[[
            'game_id', 'goalie_id', 'betting_line',
            'odds_over_american', 'odds_under_american',
            'odds_over_decimal', 'odds_under_decimal',
            'num_books'
        ]],
        on=['game_id', 'goalie_id'],
        how='left'
    )

    logger.info(f"After merge: {len(merged_df)} rows")

    # Check how many rows have betting lines
    has_line = merged_df['betting_line'].notna()
    logger.info(f"Rows with betting lines: {has_line.sum()} ({has_line.sum()/len(merged_df)*100:.1f}%)")

    # Show breakdown by season
    logger.info("\nBetting line coverage by season:")
    for season in sorted(merged_df['season'].unique()):
        season_df = merged_df[merged_df['season'] == season]
        season_has_line = season_df['betting_line'].notna()
        logger.info(f"  {season}: {season_has_line.sum()}/{len(season_df)} ({season_has_line.sum()/len(season_df)*100:.1f}%)")

    return merged_df


def create_classification_dataset(merged_df):
    """
    Create classification dataset with over_hit target

    Args:
        merged_df: Merged DataFrame with betting lines

    Returns:
        DataFrame filtered to only rows with betting lines and over_hit target
    """
    logger.info("\nCreating classification dataset...")

    # Filter to only rows with betting lines
    clf_df = merged_df[merged_df['betting_line'].notna()].copy()

    logger.info(f"Classification dataset size: {len(clf_df)} rows")

    # Create over_hit target (1 if saves > betting_line, 0 otherwise)
    clf_df['over_hit'] = (clf_df['saves'] > clf_df['betting_line']).astype(int)

    # Calculate line_margin (how far over/under)
    clf_df['line_margin'] = clf_df['saves'] - clf_df['betting_line']

    # Statistics
    over_pct = clf_df['over_hit'].mean() * 100
    logger.info(f"\nTarget variable statistics:")
    logger.info(f"  OVER hit rate: {over_pct:.1f}%")
    logger.info(f"  UNDER hit rate: {100-over_pct:.1f}%")

    logger.info(f"\nLine margin statistics:")
    logger.info(f"  Mean: {clf_df['line_margin'].mean():.2f}")
    logger.info(f"  Std: {clf_df['line_margin'].std():.2f}")
    logger.info(f"  Min: {clf_df['line_margin'].min():.2f}")
    logger.info(f"  Max: {clf_df['line_margin'].max():.2f}")

    # Show distribution by betting line value
    logger.info(f"\nBetting line distribution:")
    line_bins = pd.cut(clf_df['betting_line'], bins=[0, 20, 25, 30, 100], labels=['<20', '20-25', '25-30', '>30'])
    for bin_label in line_bins.cat.categories:
        bin_df = clf_df[line_bins == bin_label]
        if len(bin_df) > 0:
            over_rate = bin_df['over_hit'].mean() * 100
            logger.info(f"  {bin_label} saves: {len(bin_df)} games, OVER: {over_rate:.1f}%")

    return clf_df


def save_classification_data(clf_df):
    """Save classification dataset"""
    output_file = Path('data/processed/classification_training_data.parquet')

    logger.info(f"\nSaving classification dataset to {output_file}")
    clf_df.to_parquet(output_file, index=False)

    logger.info(f"Saved {len(clf_df)} rows with {len(clf_df.columns)} columns")

    # Save a summary
    summary = {
        'total_samples': len(clf_df),
        'num_features': len(clf_df.columns),
        'over_hit_rate': float(clf_df['over_hit'].mean()),
        'date_range': {
            'min': str(clf_df['game_date'].min()),
            'max': str(clf_df['game_date'].max())
        },
        'seasons': clf_df['season'].value_counts().to_dict(),
        'columns': list(clf_df.columns)
    }

    summary_file = Path('data/processed/classification_data_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary to {summary_file}")


def main():
    """Main execution"""
    logger.info("="*70)
    logger.info("BETTING LINES MERGE - CLASSIFICATION DATASET CREATION")
    logger.info("="*70)

    # Load data
    betting_lines_df = load_betting_lines()
    training_df = load_training_data()

    # Merge datasets
    merged_df = merge_datasets(training_df, betting_lines_df)

    # Create classification dataset
    clf_df = create_classification_dataset(merged_df)

    # Save
    save_classification_data(clf_df)

    logger.info("\n" + "="*70)
    logger.info("SUCCESS - Classification dataset created")
    logger.info("="*70)
    logger.info(f"Output: data/processed/classification_training_data.parquet")
    logger.info(f"Samples: {len(clf_df)}")
    logger.info(f"Features: {len(clf_df.columns)}")
    logger.info(f"Target: over_hit (OVER={clf_df['over_hit'].sum()}, UNDER={len(clf_df)-clf_df['over_hit'].sum()})")


if __name__ == "__main__":
    main()
