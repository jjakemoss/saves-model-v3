"""
Fetch historical betting lines for all boxscores
Optimized to minimize API token usage by batching requests by date
"""
import os
import sys
import json
import requests
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv('API_KEY')
BASE_URL = "https://api.the-odds-api.com/v4"

# Output directory for betting lines
OUTPUT_DIR = Path('data/raw/betting_lines')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache directory for API responses
CACHE_DIR = Path('data/raw/betting_lines/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting (API allows up to 30 requests per second)
RATE_LIMIT_DELAY = 0.05  # 50ms between requests = ~20 requests/sec (safe margin)

# Retry configuration for rate limiting
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2  # seconds, will be exponentially increased

# API Token tracking
# Historical queries: 10 credits per market per region
# We use 1 market (player_total_saves) and 1 region (us)
# So each props fetch = 10 credits
CREDITS_PER_EVENT_FETCH = 10  # /events endpoint
CREDITS_PER_PROPS_FETCH = 10  # /odds endpoint with player_total_saves


def load_all_boxscores():
    """
    Load all boxscore JSON files and organize by game date

    Returns:
        dict: {date_str: [boxscore1, boxscore2, ...]}
    """
    print("\n" + "="*70)
    print("Loading all boxscores...")
    print("="*70)

    boxscore_dir = Path('data/raw/boxscores')
    boxscores_by_date = defaultdict(list)

    total_files = 0
    for boxscore_file in boxscore_dir.glob('*.json'):
        total_files += 1
        try:
            with open(boxscore_file, 'r') as f:
                boxscore = json.load(f)
                game_date = boxscore.get('gameDate', '')

                if game_date:
                    boxscore['_file_path'] = str(boxscore_file)
                    boxscores_by_date[game_date].append(boxscore)
        except Exception as e:
            print(f"[WARNING] Failed to load {boxscore_file}: {e}")

    print(f"[OK] Loaded {total_files} boxscore files")
    print(f"[OK] Found games across {len(boxscores_by_date)} unique dates")

    # Sort dates chronologically
    sorted_dates = sorted(boxscores_by_date.keys())

    print(f"[INFO] Date range: {sorted_dates[0]} to {sorted_dates[-1]}")

    return boxscores_by_date, sorted_dates


def filter_seasons(boxscores_by_date, seasons=['20242025']):
    """
    Filter boxscores to only include specified seasons

    Args:
        boxscores_by_date: dict of {date: [boxscores]}
        seasons: list of season IDs to include

    Returns:
        Filtered dict of {date: [boxscores]}
    """
    print("\n" + "="*70)
    print(f"Filtering to seasons: {', '.join(seasons)}")
    print("="*70)

    filtered = defaultdict(list)
    season_ints = [int(s) for s in seasons]

    total_before = sum(len(games) for games in boxscores_by_date.values())

    for date, boxscores in boxscores_by_date.items():
        for boxscore in boxscores:
            season = boxscore.get('season')
            if season in season_ints:
                filtered[date].append(boxscore)

    total_after = sum(len(games) for games in filtered.values())

    print(f"[OK] Filtered from {total_before} games to {total_after} games")
    print(f"[OK] Games across {len(filtered)} unique dates")

    return filtered


def load_existing_lines():
    """
    Load existing betting lines to avoid re-fetching

    Returns:
        set: Set of game_ids that already have lines
    """
    lines_file = OUTPUT_DIR / 'betting_lines.json'

    if not lines_file.exists():
        print("[INFO] No existing betting lines file found")
        return set()

    try:
        with open(lines_file, 'r') as f:
            existing_data = json.load(f)
            existing_game_ids = {entry['game_id'] for entry in existing_data}
            print(f"[INFO] Found existing lines for {len(existing_game_ids)} games")
            return existing_game_ids
    except Exception as e:
        print(f"[WARNING] Failed to load existing lines: {e}")
        return set()


def get_cache_key(endpoint: str, params: dict) -> str:
    """Generate a cache key for API requests"""
    # Remove API key from params for cache key
    cache_params = {k: v for k, v in params.items() if k != 'apiKey'}
    param_str = '_'.join(f"{k}={v}" for k, v in sorted(cache_params.items()))
    return f"{endpoint}_{param_str}.json".replace('/', '_').replace(':', '_')


def load_from_cache(cache_key: str):
    """Load response from cache if it exists"""
    cache_file = CACHE_DIR / cache_key
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"    [WARNING] Failed to load cache {cache_key}: {e}")
            return None
    return None


def save_to_cache(cache_key: str, data):
    """Save response to cache"""
    cache_file = CACHE_DIR / cache_key
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"    [WARNING] Failed to save cache {cache_key}: {e}")


def make_api_request(url: str, params: dict, endpoint_name: str):
    """
    Make API request with retry logic for rate limiting

    Args:
        url: API endpoint URL
        params: Request parameters
        endpoint_name: Name for logging

    Returns:
        Response data or None if failed

    Raises:
        Exception: If request fails after all retries
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_BACKOFF_BASE ** (attempt + 1)
                    print(f"    [WARNING] Rate limited (429), retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Rate limited after {MAX_RETRIES} attempts")
            else:
                # Other error
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_BACKOFF_BASE ** (attempt + 1)
                print(f"    [WARNING] Request exception: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"Request failed after {MAX_RETRIES} attempts: {e}")

    raise Exception(f"Failed to complete request after {MAX_RETRIES} attempts")


def fetch_nhl_events_for_date(date_str: str) -> list:
    """
    Fetch all NHL events (games) for a specific date
    Uses cache to avoid redundant API calls

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        List of events from the Odds API
    """
    print(f"\n  Fetching events for {date_str}...")

    params = {
        'apiKey': API_KEY,
        'date': f"{date_str}T18:00:00Z"  # 6 PM UTC (around game time)
    }

    # Check cache first
    cache_key = get_cache_key('events', params)
    cached_data = load_from_cache(cache_key)

    if cached_data is not None:
        events = cached_data.get('data', [])
        print(f"    [CACHE] Found {len(events)} events")
        return events

    # Make API request
    url = f"{BASE_URL}/historical/sports/icehockey_nhl/events"

    try:
        data = make_api_request(url, params, 'events')
        events = data.get('data', [])
        print(f"    [OK] Found {len(events)} events (Cost: {CREDITS_PER_EVENT_FETCH} credits)")

        # Save to cache
        save_to_cache(cache_key, data)

        return events
    except Exception as e:
        print(f"    [ERROR] {e}")
        raise


def fetch_goalie_props_for_event(event_id: str, date_str: str) -> dict:
    """
    Fetch goalie save props for a specific event
    Uses cache to avoid redundant API calls

    Args:
        event_id: The Odds API event ID
        date_str: Date in YYYY-MM-DD format

    Returns:
        Props data from the Odds API
    """
    params = {
        'apiKey': API_KEY,
        'date': f"{date_str}T22:00:00Z",  # Close to game time (before start)
        'regions': 'us',
        'markets': 'player_total_saves'
    }

    # Check cache first
    cache_key = get_cache_key(f'odds_{event_id}', params)
    cached_data = load_from_cache(cache_key)

    if cached_data is not None:
        print(f"    [CACHE] Props data loaded from cache")
        return cached_data

    # Make API request
    url = f"{BASE_URL}/historical/sports/icehockey_nhl/events/{event_id}/odds"

    try:
        data = make_api_request(url, params, 'props')
        print(f"    [OK] Props fetched (Cost: {CREDITS_PER_PROPS_FETCH} credits)")

        # Save to cache
        save_to_cache(cache_key, data)

        return data
    except Exception as e:
        print(f"    [ERROR] {e}")
        raise


def match_boxscore_to_event(boxscore: dict, events: list) -> dict:
    """
    Match a boxscore to an Odds API event using team abbreviations

    Args:
        boxscore: Our boxscore data
        events: List of events from Odds API

    Returns:
        Matched event or None
    """
    home_team = boxscore.get('homeTeam', {}).get('abbrev', '')
    away_team = boxscore.get('awayTeam', {}).get('abbrev', '')

    # Team name mapping (NHL abbrev to bookmaker names)
    team_name_map = {
        'NYI': ['Islanders', 'New York Islanders'],
        'CBJ': ['Blue Jackets', 'Columbus Blue Jackets'],
        'TOR': ['Maple Leafs', 'Toronto Maple Leafs'],
        'BOS': ['Bruins', 'Boston Bruins'],
        'MTL': ['Canadiens', 'Montreal Canadiens'],
        'TBL': ['Lightning', 'Tampa Bay Lightning'],
        'FLA': ['Panthers', 'Florida Panthers'],
        'DET': ['Red Wings', 'Detroit Red Wings'],
        'BUF': ['Sabres', 'Buffalo Sabres'],
        'OTT': ['Senators', 'Ottawa Senators'],
        'WSH': ['Capitals', 'Washington Capitals'],
        'CAR': ['Hurricanes', 'Carolina Hurricanes'],
        'NJD': ['Devils', 'New Jersey Devils'],
        'NYR': ['Rangers', 'New York Rangers'],
        'PHI': ['Flyers', 'Philadelphia Flyers'],
        'PIT': ['Penguins', 'Pittsburgh Penguins'],
        'CHI': ['Blackhawks', 'Chicago Blackhawks'],
        'COL': ['Avalanche', 'Colorado Avalanche'],
        'DAL': ['Stars', 'Dallas Stars'],
        'MIN': ['Wild', 'Minnesota Wild'],
        'NSH': ['Predators', 'Nashville Predators'],
        'STL': ['Blues', 'St. Louis Blues'],
        'WPG': ['Jets', 'Winnipeg Jets'],
        'ARI': ['Coyotes', 'Arizona Coyotes'],
        'CGY': ['Flames', 'Calgary Flames'],
        'EDM': ['Oilers', 'Edmonton Oilers'],
        'VAN': ['Canucks', 'Vancouver Canucks'],
        'ANA': ['Ducks', 'Anaheim Ducks'],
        'LAK': ['Kings', 'Los Angeles Kings'],
        'SJS': ['Sharks', 'San Jose Sharks'],
        'VGK': ['Golden Knights', 'Vegas Golden Knights'],
        'SEA': ['Kraken', 'Seattle Kraken']
    }

    home_names = team_name_map.get(home_team, [home_team])
    away_names = team_name_map.get(away_team, [away_team])

    for event in events:
        event_home = event.get('home_team', '')
        event_away = event.get('away_team', '')

        # Check if teams match
        home_match = any(name.lower() in event_home.lower() for name in home_names)
        away_match = any(name.lower() in event_away.lower() for name in away_names)

        if home_match and away_match:
            return event

    return None


def get_starting_goalies(boxscore: dict) -> dict:
    """
    Extract starting goalie information from boxscore

    Args:
        boxscore: Boxscore data

    Returns:
        dict: {'home': {'id': ..., 'name': ...}, 'away': {'id': ..., 'name': ...}}
    """
    starters = {'home': None, 'away': None}

    # Get home goalie
    home_goalies = boxscore.get('playerByGameStats', {}).get('homeTeam', {}).get('goalies', [])
    if home_goalies:
        starter = home_goalies[0]  # First goalie listed is typically the starter
        starters['home'] = {
            'id': starter.get('playerId'),
            'name': starter.get('name', {}).get('default', ''),
            'last_name': starter.get('name', {}).get('default', '').split()[-1]
        }

    # Get away goalie
    away_goalies = boxscore.get('playerByGameStats', {}).get('awayTeam', {}).get('goalies', [])
    if away_goalies:
        starter = away_goalies[0]  # First goalie listed is typically the starter
        starters['away'] = {
            'id': starter.get('playerId'),
            'name': starter.get('name', {}).get('default', ''),
            'last_name': starter.get('name', {}).get('default', '').split()[-1]
        }

    return starters


def extract_goalie_lines(props_data: dict, starting_goalies: dict) -> dict:
    """
    Extract betting lines for starting goalies, preferring BetMGM

    Args:
        props_data: Props response from Odds API
        starting_goalies: dict with home/away starting goalie info

    Returns:
        dict: {'home': line_value or None, 'away': line_value or None}
    """
    lines = {'home': None, 'away': None}

    bookmakers = props_data.get('data', {}).get('bookmakers', [])

    if not bookmakers:
        return lines

    # Collect all lines by goalie last name
    all_lines = defaultdict(list)

    for bookmaker in bookmakers:
        bookmaker_name = bookmaker.get('title', 'Unknown')
        markets = bookmaker.get('markets', [])

        for market in markets:
            if market.get('key') == 'player_total_saves':
                outcomes = market.get('outcomes', [])

                for outcome in outcomes:
                    player_name = outcome.get('description', '')
                    line = outcome.get('point', None)

                    if player_name and line is not None:
                        # Extract last name from player name
                        last_name = player_name.split()[-1]

                        all_lines[last_name].append({
                            'sportsbook': bookmaker_name,
                            'line': float(line),
                            'full_name': player_name
                        })

    # Match goalies to lines, preferring BetMGM
    for side in ['home', 'away']:
        if starting_goalies[side] is None:
            continue

        goalie_last_name = starting_goalies[side]['last_name']

        # Find lines for this goalie
        matching_lines = all_lines.get(goalie_last_name, [])

        if not matching_lines:
            # Try case-insensitive match
            for last_name in all_lines.keys():
                if last_name.lower() == goalie_last_name.lower():
                    matching_lines = all_lines[last_name]
                    break

        if matching_lines:
            # Prefer BetMGM
            betmgm_line = next((l for l in matching_lines if l['sportsbook'] == 'BetMGM'), None)

            if betmgm_line:
                lines[side] = betmgm_line['line']
            else:
                # Use first available book
                lines[side] = matching_lines[0]['line']

    return lines


def process_date(date_str: str, boxscores: list, existing_game_ids: set):
    """
    Process all boxscores for a single date

    Args:
        date_str: Date in YYYY-MM-DD format
        boxscores: List of boxscores for this date
        existing_game_ids: Set of game_ids already processed

    Returns:
        list: Betting line entries for this date
        int: Number of API credits used
    """
    print(f"\n{'='*70}")
    print(f"Processing date: {date_str} ({len(boxscores)} games)")
    print(f"{'='*70}")

    credits_used = 0
    betting_lines = []

    # Filter out games we've already processed
    new_boxscores = [b for b in boxscores if b.get('id') not in existing_game_ids]

    if not new_boxscores:
        print(f"  [SKIP] All games for this date already have lines")
        return betting_lines, credits_used

    print(f"  [INFO] {len(new_boxscores)} new games to process")

    # Fetch events for this date (1 API call for all games on this date)
    events = fetch_nhl_events_for_date(date_str)
    credits_used += CREDITS_PER_EVENT_FETCH

    if not events:
        print(f"  [WARNING] No events found for {date_str}")
        return betting_lines, credits_used

    time.sleep(RATE_LIMIT_DELAY)

    # Match each boxscore to an event
    for boxscore in new_boxscores:
        game_id = boxscore.get('id')
        home_team = boxscore.get('homeTeam', {}).get('abbrev', '')
        away_team = boxscore.get('awayTeam', {}).get('abbrev', '')

        print(f"\n  Processing game {game_id}: {away_team} @ {home_team}")

        # Match boxscore to event
        matched_event = match_boxscore_to_event(boxscore, events)

        if not matched_event:
            print(f"    [WARNING] Could not match game to Odds API event")
            # Store entry with NA lines
            betting_lines.append({
                'game_id': game_id,
                'game_date': date_str,
                'home_team': home_team,
                'away_team': away_team,
                'home_goalie_line': None,
                'away_goalie_line': None,
                'status': 'no_event_match'
            })
            continue

        event_id = matched_event['id']
        print(f"    [OK] Matched to event {event_id}")

        # Fetch props for this event
        props_data = fetch_goalie_props_for_event(event_id, date_str)
        credits_used += CREDITS_PER_PROPS_FETCH

        time.sleep(RATE_LIMIT_DELAY)

        if not props_data:
            print(f"    [WARNING] No props data available")
            betting_lines.append({
                'game_id': game_id,
                'game_date': date_str,
                'home_team': home_team,
                'away_team': away_team,
                'home_goalie_line': None,
                'away_goalie_line': None,
                'status': 'no_props_data'
            })
            continue

        # Get starting goalies
        starting_goalies = get_starting_goalies(boxscore)

        # Extract lines
        lines = extract_goalie_lines(props_data, starting_goalies)

        # Create entry
        entry = {
            'game_id': game_id,
            'game_date': date_str,
            'home_team': home_team,
            'away_team': away_team,
            'home_goalie_id': starting_goalies['home']['id'] if starting_goalies['home'] else None,
            'home_goalie_name': starting_goalies['home']['name'] if starting_goalies['home'] else None,
            'home_goalie_line': lines['home'],
            'away_goalie_id': starting_goalies['away']['id'] if starting_goalies['away'] else None,
            'away_goalie_name': starting_goalies['away']['name'] if starting_goalies['away'] else None,
            'away_goalie_line': lines['away'],
            'status': 'success' if (lines['home'] is not None or lines['away'] is not None) else 'no_lines_found'
        }

        betting_lines.append(entry)

        print(f"    Home: {entry['home_goalie_name']} - Line: {lines['home']}")
        print(f"    Away: {entry['away_goalie_name']} - Line: {lines['away']}")

    return betting_lines, credits_used


def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("BETTING LINES FETCHER - BULK PROCESSING")
    print("="*70)
    print(f"API Key: {'[OK] Loaded' if API_KEY else '[ERROR] Missing'}")

    if not API_KEY:
        print("\n[ERROR] API_KEY not found in environment variables")
        print("Make sure you have a .env file with API_KEY=your_key_here")
        return

    # Load all boxscores
    boxscores_by_date, sorted_dates = load_all_boxscores()

    # Filter to 2023-24 and 2024-25 seasons only
    boxscores_by_date = filter_seasons(boxscores_by_date, seasons=['20232024', '20242025'])
    sorted_dates = sorted(boxscores_by_date.keys())

    if not sorted_dates:
        print("\n[ERROR] No boxscores found for specified seasons")
        return

    # Load existing betting lines to avoid re-processing
    existing_game_ids = load_existing_lines()

    # Calculate expected API usage
    total_games = sum(len(games) for games in boxscores_by_date.values())
    games_to_process = sum(
        len([g for g in games if g.get('id') not in existing_game_ids])
        for games in boxscores_by_date.values()
    )

    # Cost estimate: 1 events fetch per date + 1 props fetch per game
    estimated_credits = (len(sorted_dates) * CREDITS_PER_EVENT_FETCH) + (games_to_process * CREDITS_PER_PROPS_FETCH)

    print("\n" + "="*70)
    print("API USAGE ESTIMATE")
    print("="*70)
    print(f"Total games: {total_games}")
    print(f"Already processed: {len(existing_game_ids)}")
    print(f"Games to process: {games_to_process}")
    print(f"Unique dates: {len(sorted_dates)}")
    print(f"Estimated API credits: {estimated_credits}")
    print(f"Monthly limit: 20,000 credits")
    print(f"Estimated usage: {estimated_credits/20000*100:.1f}% of monthly limit")

    # Ask for confirmation
    print("\n" + "="*70)
    response = input("Proceed with fetching? (yes/no): ")

    if response.lower() != 'yes':
        print("[INFO] Aborted by user")
        return

    # Process all dates
    all_betting_lines = []
    total_credits_used = 0

    for date_str in sorted_dates:
        boxscores = boxscores_by_date[date_str]

        lines, credits = process_date(date_str, boxscores, existing_game_ids)
        all_betting_lines.extend(lines)
        total_credits_used += credits

        # Update existing_game_ids to avoid reprocessing if we run again mid-execution
        existing_game_ids.update(entry['game_id'] for entry in lines)

    # Load existing lines and merge with new ones
    lines_file = OUTPUT_DIR / 'betting_lines.json'

    if lines_file.exists():
        with open(lines_file, 'r') as f:
            existing_lines = json.load(f)
    else:
        existing_lines = []

    # Merge new lines with existing (new lines override if duplicate game_id)
    existing_game_ids_set = {entry['game_id'] for entry in existing_lines}
    final_lines = existing_lines + [l for l in all_betting_lines if l['game_id'] not in existing_game_ids_set]

    # Save results
    with open(lines_file, 'w') as f:
        json.dump(final_lines, f, indent=2)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total games processed: {len(all_betting_lines)}")
    print(f"Total lines in file: {len(final_lines)}")
    print(f"API credits used this run: {total_credits_used}")
    print(f"[OK] Results saved to: {lines_file}")

    # Calculate coverage
    successful = sum(1 for entry in final_lines if entry['home_goalie_line'] is not None or entry['away_goalie_line'] is not None)
    coverage = successful / len(final_lines) * 100 if final_lines else 0

    print(f"\nCoverage: {successful}/{len(final_lines)} games ({coverage:.1f}%)")


if __name__ == "__main__":
    main()
