"""
Build multi-book classification training data.

This script:
1. Loads base training features (training_data.parquet)
2. Parses historical odds from The-Odds-API cache (per-bookmaker lines)
3. Matches odds to training rows by game_date + team
4. Creates one row per (goalie, game, bookmaker, line) combination
5. Computes line-relative features
6. Saves as multibook_classification_training_data.parquet

Usage:
    python scripts/build_multibook_training_data.py
"""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting.odds_utils import decimal_to_american
from features.feature_engineering import compute_line_relative_features

# Full NHL team name -> abbreviation mapping
TEAM_NAME_TO_ABBREV = {
    'Anaheim Ducks': 'ANA',
    'Arizona Coyotes': 'ARI',
    'Boston Bruins': 'BOS',
    'Buffalo Sabres': 'BUF',
    'Calgary Flames': 'CGY',
    'Carolina Hurricanes': 'CAR',
    'Chicago Blackhawks': 'CHI',
    'Colorado Avalanche': 'COL',
    'Columbus Blue Jackets': 'CBJ',
    'Dallas Stars': 'DAL',
    'Detroit Red Wings': 'DET',
    'Edmonton Oilers': 'EDM',
    'Florida Panthers': 'FLA',
    'Los Angeles Kings': 'LAK',
    'Minnesota Wild': 'MIN',
    'Montreal Canadiens': 'MTL',
    'Montr\u00e9al Canadiens': 'MTL',
    'Nashville Predators': 'NSH',
    'New Jersey Devils': 'NJD',
    'New York Islanders': 'NYI',
    'New York Rangers': 'NYR',
    'Ottawa Senators': 'OTT',
    'Philadelphia Flyers': 'PHI',
    'Pittsburgh Penguins': 'PIT',
    'San Jose Sharks': 'SJS',
    'Seattle Kraken': 'SEA',
    'St. Louis Blues': 'STL',
    'St Louis Blues': 'STL',
    'Tampa Bay Lightning': 'TBL',
    'Toronto Maple Leafs': 'TOR',
    'Utah Hockey Club': 'UTA',
    'Vancouver Canucks': 'VAN',
    'Vegas Golden Knights': 'VGK',
    'Washington Capitals': 'WSH',
    'Winnipeg Jets': 'WPG',
}


def extract_last_name(full_name):
    """Extract last name from a full player name."""
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return parts[-1]
    return full_name


def parse_odds_cache(cache_dir):
    """
    Parse all odds files from The-Odds-API cache.

    Returns:
        list of dicts with keys: game_date, home_team, away_team, book_key,
        goalie_name, goalie_last_name, betting_line, odds_over_decimal,
        odds_under_decimal, odds_over_american, odds_under_american
    """
    cache_path = Path(cache_dir)
    odds_files = list(cache_path.glob('odds_*.json'))
    print(f"Found {len(odds_files)} odds files in cache")

    records = []
    parse_errors = 0

    for odds_file in odds_files:
        try:
            with open(odds_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            parse_errors += 1
            continue

        event = data.get('data', {})
        commence_time = event.get('commence_time', '')
        if not commence_time:
            continue

        game_date = commence_time[:10]  # YYYY-MM-DD
        home_team_full = event.get('home_team', '')
        away_team_full = event.get('away_team', '')

        home_abbrev = TEAM_NAME_TO_ABBREV.get(home_team_full)
        away_abbrev = TEAM_NAME_TO_ABBREV.get(away_team_full)

        if not home_abbrev or not away_abbrev:
            continue

        for bookmaker in event.get('bookmakers', []):
            book_key = bookmaker.get('key', '')

            for market in bookmaker.get('markets', []):
                if market.get('key') != 'player_total_saves':
                    continue

                # Group outcomes by player (Over/Under pairs)
                player_lines = {}
                for outcome in market.get('outcomes', []):
                    player = outcome.get('description', '')
                    if not player:
                        continue

                    if player not in player_lines:
                        player_lines[player] = {
                            'line': outcome.get('point'),
                        }

                    if outcome.get('name') == 'Over':
                        player_lines[player]['odds_over'] = outcome.get('price')
                    elif outcome.get('name') == 'Under':
                        player_lines[player]['odds_under'] = outcome.get('price')

                for player, line_data in player_lines.items():
                    line = line_data.get('line')
                    odds_over = line_data.get('odds_over')
                    odds_under = line_data.get('odds_under')

                    if line is None or odds_over is None or odds_under is None:
                        continue

                    records.append({
                        'game_date': game_date,
                        'home_team_abbrev': home_abbrev,
                        'away_team_abbrev': away_abbrev,
                        'book_key': book_key,
                        'goalie_name': player,
                        'goalie_last_name': extract_last_name(player),
                        'betting_line': line,
                        'odds_over_decimal': odds_over,
                        'odds_under_decimal': odds_under,
                        'odds_over_american': decimal_to_american(odds_over),
                        'odds_under_american': decimal_to_american(odds_under),
                    })

    if parse_errors:
        print(f"  [WARNING] {parse_errors} files failed to parse")

    print(f"  Extracted {len(records)} bookmaker-goalie line records")
    return records


def build_multibook_data(
    base_features_path='data/processed/classification_training_data.parquet',
    odds_cache_dir='data/raw/betting_lines/cache',
    output_path='data/processed/multibook_classification_training_data.parquet'
):
    """Build multi-book classification training data."""

    # Load base features
    print("\n[1/5] Loading base training features...")
    base_df = pd.read_parquet(base_features_path)
    print(f"  Loaded {len(base_df)} rows, {len(base_df.columns)} columns")
    print(f"  Date range: {base_df['game_date'].min()} to {base_df['game_date'].max()}")

    # Build lookup key: (game_date, team_abbrev) -> row index
    # Each team has one starting goalie per game in the base data
    print("\n[2/5] Building goalie lookup from base features...")
    base_df['_game_date_str'] = pd.to_datetime(base_df['game_date']).dt.strftime('%Y-%m-%d')
    base_df['_lookup_key'] = base_df['_game_date_str'] + '_' + base_df['team_abbrev']
    lookup = base_df.set_index('_lookup_key')

    # Check for duplicate keys (shouldn't happen for starting goalies)
    dupes = lookup.index.duplicated(keep='first')
    if dupes.any():
        n_dupes = dupes.sum()
        print(f"  [WARNING] {n_dupes} duplicate (date, team) entries - keeping first")
        lookup = lookup[~dupes]

    print(f"  {len(lookup)} unique (date, team) entries")

    # Parse odds cache
    print("\n[3/5] Parsing odds cache...")
    odds_records = parse_odds_cache(odds_cache_dir)
    odds_df = pd.DataFrame(odds_records)

    if len(odds_df) == 0:
        print("[ERROR] No odds records found")
        return

    # Deduplicate odds: keep one record per (game_date, book, goalie_name, line)
    # Same book might appear in multiple cache files for the same game
    odds_df = odds_df.drop_duplicates(
        subset=['game_date', 'book_key', 'goalie_name', 'betting_line'],
        keep='last'
    )
    print(f"  After dedup: {len(odds_df)} unique records")

    # Match odds to base features
    print("\n[4/5] Matching odds to base features...")
    matched_rows = []
    unmatched = 0

    for _, odds_row in odds_df.iterrows():
        game_date = odds_row['game_date']
        goalie_last = odds_row['goalie_last_name'].lower()

        # Try both home and away team keys
        for team_abbrev in [odds_row['home_team_abbrev'], odds_row['away_team_abbrev']]:
            key = f"{game_date}_{team_abbrev}"

            if key not in lookup.index:
                continue

            base_row = lookup.loc[key]

            # Verify goalie name match (last name)
            # The base data may not have goalie_name, so we skip name check
            # if the column doesn't exist - we rely on (date, team) matching
            if 'goalie_name' in base_row.index:
                base_name = str(base_row.get('goalie_name', '')).lower()
                # Check if last names match
                base_last = extract_last_name(base_name).lower() if base_name else ''
                if base_last and goalie_last and base_last != goalie_last:
                    continue

            # Build the expanded row: base features + this bookmaker's odds
            new_row = base_row.to_dict()

            # Override betting columns with this bookmaker's values
            new_row['betting_line'] = odds_row['betting_line']
            new_row['odds_over_decimal'] = odds_row['odds_over_decimal']
            new_row['odds_under_decimal'] = odds_row['odds_under_decimal']
            new_row['odds_over_american'] = odds_row['odds_over_american']
            new_row['odds_under_american'] = odds_row['odds_under_american']
            new_row['book_key'] = odds_row['book_key']
            new_row['num_books'] = 1  # Per-bookmaker row

            # Recalculate target: over_hit depends on THIS bookmaker's line
            actual_saves = new_row.get('saves')
            if pd.notna(actual_saves) and pd.notna(odds_row['betting_line']):
                new_row['over_hit'] = int(actual_saves > odds_row['betting_line'])
                new_row['line_margin'] = actual_saves - odds_row['betting_line']

            matched_rows.append(new_row)
            break  # Found match, don't check other team
        else:
            unmatched += 1

    print(f"  Matched: {len(matched_rows)} rows")
    print(f"  Unmatched: {unmatched} odds records (no base features found)")

    if not matched_rows:
        print("[ERROR] No rows matched")
        return

    # Build output DataFrame
    result_df = pd.DataFrame(matched_rows)

    # Remove lookup key
    if '_lookup_key' in result_df.columns:
        result_df = result_df.drop(columns=['_lookup_key'])

    # Compute line-relative features
    print("\n[5/5] Computing line-relative features...")
    result_df = compute_line_relative_features(result_df)

    # Sort by game_date for chronological splits
    result_df = result_df.sort_values('game_date').reset_index(drop=True)

    # Verify line-relative features exist
    lr_features = [f'line_vs_rolling_{w}' for w in [3, 5, 10]] + \
                  [f'line_z_score_{w}' for w in [3, 5, 10]]
    for feat in lr_features:
        if feat in result_df.columns:
            print(f"  {feat}: mean={result_df[feat].mean():.3f}, std={result_df[feat].std():.3f}")

    # Show stats
    print(f"\nOutput summary:")
    print(f"  Total rows: {len(result_df)}")
    print(f"  Columns: {len(result_df.columns)}")
    print(f"  Date range: {result_df['game_date'].min()} to {result_df['game_date'].max()}")
    print(f"  Unique games: {result_df['game_id'].nunique()}")
    print(f"  Unique goalies: {result_df['goalie_id'].nunique()}")
    if 'book_key' in result_df.columns:
        print(f"  Books: {result_df['book_key'].value_counts().to_dict()}")
    print(f"  Over/Under split: {result_df['over_hit'].mean():.1%} over / {1 - result_df['over_hit'].mean():.1%} under")

    # Show multi-line examples
    game_goalie_counts = result_df.groupby(['game_id', 'goalie_id']).size()
    multi_line = game_goalie_counts[game_goalie_counts > 1]
    print(f"\n  Goalie-games with multiple bookmaker lines: {len(multi_line)}")
    if len(multi_line) > 0:
        print(f"  Max lines per goalie-game: {multi_line.max()}")

        # Show example of different lines producing different over_hit labels
        diff_labels = result_df.groupby(['game_id', 'goalie_id']).agg(
            n_lines=('betting_line', 'count'),
            unique_lines=('betting_line', 'nunique'),
            unique_labels=('over_hit', 'nunique'),
        )
        diff_labels = diff_labels[diff_labels['unique_labels'] > 1]
        print(f"  Goalie-games where different lines produce different over_hit: {len(diff_labels)}")

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_file, index=False)
    print(f"\n[OK] Saved to {output_file}")


if __name__ == '__main__':
    build_multibook_data()
