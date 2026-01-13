"""
Extract historical odds from cached API responses

This script:
1. Reads all cached odds files from data/raw/betting_lines/cache/
2. Extracts betting lines and odds for each goalie
3. Averages across multiple sportsbooks
4. Converts decimal odds to American format
5. Rounds betting lines to nearest 0.5
6. Updates betting_lines.json with odds data
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def convert_decimal_to_american(decimal_odds):
    """
    Convert decimal odds to American format.

    Args:
        decimal_odds: Decimal odds (e.g., 1.87, 2.0)

    Returns:
        int: American odds (e.g., -115, +100)

    Examples:
        1.87 → -115
        2.0 → +100
        1.5 → -200
        3.0 → +200
    """
    if decimal_odds < 2.0:
        return int(-100 / (decimal_odds - 1))
    else:
        return int((decimal_odds - 1) * 100)


def round_to_nearest_half(value):
    """
    Round betting line to nearest 0.5.

    Args:
        value: Line value to round

    Returns:
        float: Rounded to nearest 0.5

    Examples:
        24.3 → 24.5
        25.7 → 25.5
        25.25 → 25.5
    """
    return round(value * 2) / 2


def normalize_goalie_name(name):
    """
    Normalize goalie name for matching.

    Args:
        name: Goalie name from betting API (e.g., "John Gibson")

    Returns:
        str: Normalized name for matching

    Examples:
        "John Gibson" → "gibson"
        "Connor Hellebuyck" → "hellebuyck"
    """
    # Take last name only, lowercase
    return name.split()[-1].lower()


def aggregate_bookmaker_odds(bookmakers, goalie_name):
    """
    Extract and average odds for specific goalie from all sportsbooks.

    Args:
        bookmakers: List of bookmaker dicts from cache file
        goalie_name: Goalie name to match (normalized)

    Returns:
        dict or None: {
            'betting_line': float (averaged, rounded to 0.5),
            'odds_over_decimal': float (average OVER odds),
            'odds_under_decimal': float (average UNDER odds),
            'num_books': int (number of sportsbooks)
        }

    Logic:
        1. Find all outcomes matching goalie_name across all bookmakers
        2. Average the 'point' values → round to nearest 0.5
        3. Average the 'price' for OVER
        4. Average the 'price' for UNDER
    """
    over_odds_list = []
    under_odds_list = []
    line_list = []

    for book in bookmakers:
        for market in book.get('markets', []):
            if market['key'] != 'player_total_saves':
                continue

            for outcome in market.get('outcomes', []):
                # Normalize description for matching
                outcome_name = normalize_goalie_name(outcome.get('description', ''))

                if outcome_name == goalie_name:
                    line_list.append(outcome['point'])

                    if outcome['name'] == 'Over':
                        over_odds_list.append(outcome['price'])
                    elif outcome['name'] == 'Under':
                        under_odds_list.append(outcome['price'])

    # Need at least one match
    if not line_list:
        return None

    # Calculate averages
    avg_line = sum(line_list) / len(line_list)
    rounded_line = round_to_nearest_half(avg_line)

    avg_over_odds = sum(over_odds_list) / len(over_odds_list) if over_odds_list else None
    avg_under_odds = sum(under_odds_list) / len(under_odds_list) if under_odds_list else None

    # Need both over and under odds
    if avg_over_odds is None or avg_under_odds is None:
        return None

    return {
        'betting_line': rounded_line,
        'odds_over_decimal': avg_over_odds,
        'odds_under_decimal': avg_under_odds,
        'num_books': len(set([b['key'] for b in bookmakers]))
    }


def normalize_team_name(team):
    """
    Normalize team name for matching between APIs.

    Args:
        team: Team name from either API

    Returns:
        str: Normalized team name

    Examples:
        "Detroit Red Wings" → "det"
        "DET" → "det"
        "Utah Hockey Club" → "uta"
    """
    # Map full names to abbreviations
    team_map = {
        'anaheim ducks': 'ana',
        'boston bruins': 'bos',
        'buffalo sabres': 'buf',
        'calgary flames': 'cgy',
        'carolina hurricanes': 'car',
        'chicago blackhawks': 'chi',
        'colorado avalanche': 'col',
        'columbus blue jackets': 'cbj',
        'dallas stars': 'dal',
        'detroit red wings': 'det',
        'edmonton oilers': 'edm',
        'florida panthers': 'fla',
        'los angeles kings': 'lak',
        'minnesota wild': 'min',
        'montreal canadiens': 'mtl',
        'nashville predators': 'nsh',
        'new jersey devils': 'njd',
        'new york islanders': 'nyi',
        'new york rangers': 'nyr',
        'ottawa senators': 'ott',
        'philadelphia flyers': 'phi',
        'pittsburgh penguins': 'pit',
        'san jose sharks': 'sjs',
        'seattle kraken': 'sea',
        'st louis blues': 'stl',
        'tampa bay lightning': 'tbl',
        'toronto maple leafs': 'tor',
        'utah hockey club': 'uta',
        'vancouver canucks': 'van',
        'vegas golden knights': 'vgk',
        'washington capitals': 'wsh',
        'winnipeg jets': 'wpg'
    }

    team_lower = team.lower()

    # Normalize accented characters (Montréal → Montreal)
    team_lower = team_lower.replace('é', 'e').replace('è', 'e').replace('ê', 'e')

    # Check if it's already an abbreviation
    if len(team) == 3:
        return team.lower()

    # Look up full name
    return team_map.get(team_lower, team_lower[:3])


def extract_odds_from_cache():
    """
    Extract odds from all cache files.

    Returns:
        dict: Mapping of (commence_time, home_team, away_team, goalie_name) → odds data
        Note: commence_time is stored as full ISO string for timezone-aware matching
    """
    cache_dir = Path('data/raw/betting_lines/cache')
    odds_files = sorted(glob.glob(str(cache_dir / 'odds_*.json')))

    logger.info(f"Found {len(odds_files)} cache files")

    odds_by_game_goalie = {}

    for odds_file in odds_files:
        try:
            with open(odds_file, 'r') as f:
                cache_data = json.load(f)

            # Extract game data
            data = cache_data.get('data', {})
            if not data:
                continue

            # Extract game commence time (keep full ISO string for matching)
            commence_time = data.get('commence_time', '')
            if not commence_time:
                continue

            # Extract teams
            home_team = normalize_team_name(data.get('home_team', ''))
            away_team = normalize_team_name(data.get('away_team', ''))

            bookmakers = data.get('bookmakers', [])
            if not bookmakers:
                continue

            # Get all unique goalie names from outcomes
            goalie_names = set()
            for book in bookmakers:
                for market in book.get('markets', []):
                    if market['key'] == 'player_total_saves':
                        for outcome in market.get('outcomes', []):
                            desc = outcome.get('description')
                            if desc:
                                goalie_names.add(desc)

            # Extract odds for each goalie
            for goalie_name in goalie_names:
                normalized_name = normalize_goalie_name(goalie_name)
                odds_data = aggregate_bookmaker_odds(bookmakers, normalized_name)

                if odds_data:
                    # Store with commence_time and teams as key
                    key = (commence_time, home_team, away_team, normalized_name)
                    odds_by_game_goalie[key] = odds_data

        except Exception as e:
            logger.warning(f"Error processing {odds_file}: {e}")
            continue

    logger.info(f"Extracted odds for {len(odds_by_game_goalie)} goalie-game combinations")

    return odds_by_game_goalie


def match_to_betting_lines(odds_data):
    """
    Match extracted odds to existing betting_lines.json entries.

    Handles timezone differences between NHL gameDate (local) and
    betting API commence_time (UTC) by checking ±1 day.

    Args:
        odds_data: Dict of (commence_time_iso, home_team, away_team, goalie_name) → odds

    Returns:
        list: Updated betting lines with odds
    """
    from datetime import datetime, timedelta

    betting_lines_file = Path('data/raw/betting_lines/betting_lines.json')

    logger.info(f"Loading existing betting lines from {betting_lines_file}")

    with open(betting_lines_file, 'r') as f:
        betting_lines = json.load(f)

    logger.info(f"Loaded {len(betting_lines)} betting line entries")

    # Update entries with odds
    updated_count = 0

    for entry in betting_lines:
        game_date = entry['game_date']  # NHL local date (YYYY-MM-DD)
        home_team = normalize_team_name(entry['home_team'])
        away_team = normalize_team_name(entry['away_team'])

        # Try to match home goalie
        if entry.get('home_goalie_name'):
            home_goalie_normalized = normalize_goalie_name(entry['home_goalie_name'])

            # Search for matching game in cache (check ±1 day for UTC/local timezone differences)
            matched_odds = None
            for (commence_time, cache_home, cache_away, cache_goalie), odds in odds_data.items():
                # Check if teams match
                if cache_home != home_team or cache_away != away_team:
                    continue

                # Check if goalie matches
                if cache_goalie != home_goalie_normalized:
                    continue

                # Check if date is within ±1 day (handles UTC/local timezone difference)
                commence_date = commence_time.split('T')[0]  # UTC date from cache
                game_dt = datetime.strptime(game_date, '%Y-%m-%d')
                commence_dt = datetime.strptime(commence_date, '%Y-%m-%d')

                day_diff = abs((commence_dt - game_dt).days)
                if day_diff <= 1:
                    matched_odds = odds
                    break

            if matched_odds:
                entry['home_goalie_line'] = matched_odds['betting_line']
                entry['home_odds_over_decimal'] = matched_odds['odds_over_decimal']
                entry['home_odds_under_decimal'] = matched_odds['odds_under_decimal']
                entry['home_odds_over_american'] = convert_decimal_to_american(matched_odds['odds_over_decimal'])
                entry['home_odds_under_american'] = convert_decimal_to_american(matched_odds['odds_under_decimal'])
                entry['home_num_books'] = matched_odds['num_books']
                updated_count += 1

        # Try to match away goalie
        if entry.get('away_goalie_name'):
            away_goalie_normalized = normalize_goalie_name(entry['away_goalie_name'])

            # Search for matching game in cache (check ±1 day for UTC/local timezone differences)
            matched_odds = None
            for (commence_time, cache_home, cache_away, cache_goalie), odds in odds_data.items():
                # Check if teams match
                if cache_home != home_team or cache_away != away_team:
                    continue

                # Check if goalie matches
                if cache_goalie != away_goalie_normalized:
                    continue

                # Check if date is within ±1 day (handles UTC/local timezone difference)
                commence_date = commence_time.split('T')[0]  # UTC date from cache
                game_dt = datetime.strptime(game_date, '%Y-%m-%d')
                commence_dt = datetime.strptime(commence_date, '%Y-%m-%d')

                day_diff = abs((commence_dt - game_dt).days)
                if day_diff <= 1:
                    matched_odds = odds
                    break

            if matched_odds:
                entry['away_goalie_line'] = matched_odds['betting_line']
                entry['away_odds_over_decimal'] = matched_odds['odds_over_decimal']
                entry['away_odds_under_decimal'] = matched_odds['odds_under_decimal']
                entry['away_odds_over_american'] = convert_decimal_to_american(matched_odds['odds_over_decimal'])
                entry['away_odds_under_american'] = convert_decimal_to_american(matched_odds['odds_under_decimal'])
                entry['away_num_books'] = matched_odds['num_books']
                updated_count += 1

    logger.info(f"Updated {updated_count} goalie entries with odds data")

    return betting_lines


def main():
    """Main extraction pipeline"""
    logger.info("Starting historical odds extraction...")

    # Extract odds from cache files
    odds_data = extract_odds_from_cache()

    # Match to existing betting lines
    updated_lines = match_to_betting_lines(odds_data)

    # Save updated betting lines
    output_file = Path('data/raw/betting_lines/betting_lines.json')
    backup_file = Path('data/raw/betting_lines/betting_lines_backup.json')

    # Backup original
    if output_file.exists():
        import shutil
        shutil.copy(output_file, backup_file)
        logger.info(f"Backed up original to {backup_file}")

    # Write updated file
    with open(output_file, 'w') as f:
        json.dump(updated_lines, f, indent=2)

    logger.info(f"Saved updated betting lines to {output_file}")

    # Show statistics
    entries_with_odds = sum(1 for e in updated_lines
                           if e.get('home_odds_over_american') or e.get('away_odds_over_american'))
    logger.info(f"\nStatistics:")
    logger.info(f"  Total entries: {len(updated_lines)}")
    logger.info(f"  Entries with odds: {entries_with_odds}")
    logger.info(f"  Coverage: {entries_with_odds/len(updated_lines)*100:.1f}%")


if __name__ == '__main__':
    main()
