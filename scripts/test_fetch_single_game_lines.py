"""
Test script to fetch betting lines for a single game
Game: 2024021312 (CBJ vs NYI, 2025-04-17)
"""
import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv('API_KEY')
BASE_URL = "https://api.the-odds-api.com/v4"

# Test game details (from boxscore 2024021312.json)
TEST_GAME_ID = 2024021312
TEST_GAME_DATE = "2025-04-17"
TEST_GAME_TIME = "2025-04-17T23:00:00Z"

def load_boxscore(game_id: int) -> dict:
    """Load boxscore data for a game"""
    boxscore_path = Path(f'data/raw/boxscores/{game_id}.json')

    if not boxscore_path.exists():
        raise FileNotFoundError(f"Boxscore not found: {boxscore_path}")

    with open(boxscore_path, 'r') as f:
        return json.load(f)

def fetch_nhl_events_for_date(date_str: str) -> list:
    """
    Fetch all NHL events (games) for a specific date

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        List of events from the Odds API
    """
    print(f"\n{'='*70}")
    print(f"Fetching NHL events for {date_str}...")
    print(f"{'='*70}")

    url = f"{BASE_URL}/historical/sports/icehockey_nhl/events"
    params = {
        'apiKey': API_KEY,
        'date': f"{date_str}T18:00:00Z"  # 6 PM UTC (around game time)
    }

    print(f"Request URL: {url}")
    print(f"Parameters: date={params['date']}, regions=us")

    response = requests.get(url, params=params)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data.get('data', []))} events")
        return data.get('data', [])
    else:
        print(f"❌ Error: {response.text}")
        return []

def fetch_goalie_props_for_event(event_id: str, date_str: str) -> dict:
    """
    Fetch goalie save props for a specific event

    Args:
        event_id: The Odds API event ID
        date_str: Date in YYYY-MM-DD format

    Returns:
        Props data from the Odds API
    """
    print(f"\n{'='*70}")
    print(f"Fetching goalie props for event {event_id}...")
    print(f"{'='*70}")

    url = f"{BASE_URL}/historical/sports/icehockey_nhl/events/{event_id}/odds"
    params = {
        'apiKey': API_KEY,
        'date': f"{date_str}T22:00:00Z",  # Close to game time (before start)
        'regions': 'us',
        'markets': 'player_total_saves'
    }

    print(f"Request URL: {url}")
    print(f"Parameters: date={params['date']}, regions=us, markets=player_total_saves")

    response = requests.get(url, params=params)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Successfully fetched props data")
        return data
    else:
        print(f"❌ Error: {response.text}")
        return {}

def match_game_to_event(boxscore: dict, events: list) -> dict:
    """
    Match our NHL boxscore to an Odds API event

    Args:
        boxscore: Our boxscore data
        events: List of events from Odds API

    Returns:
        Matched event or None
    """
    print(f"\n{'='*70}")
    print(f"Matching boxscore to Odds API event...")
    print(f"{'='*70}")

    # Extract team info from boxscore
    home_team = boxscore.get('homeTeam', {}).get('abbrev', '')
    away_team = boxscore.get('awayTeam', {}).get('abbrev', '')
    game_date = boxscore.get('gameDate', '')

    print(f"Looking for: {away_team} @ {home_team} on {game_date}")

    # Common team name mappings (NHL abbrev to common names used by bookmakers)
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

    print(f"\nSearching events:")
    for event in events:
        event_home = event.get('home_team', '')
        event_away = event.get('away_team', '')
        event_date = event.get('commence_time', '')

        print(f"  Event: {event_away} @ {event_home}")
        print(f"    ID: {event.get('id')}")
        print(f"    Time: {event_date}")

        # Check if teams match
        home_match = any(name.lower() in event_home.lower() for name in home_names)
        away_match = any(name.lower() in event_away.lower() for name in away_names)

        if home_match and away_match:
            print(f"    ✅ MATCH FOUND!")
            return event

    print(f"\n❌ No matching event found")
    return None

def extract_goalie_lines(props_data: dict, boxscore: dict) -> list:
    """
    Extract goalie save lines from props data

    Args:
        props_data: Props response from Odds API
        boxscore: Our boxscore data

    Returns:
        List of goalie lines
    """
    print(f"\n{'='*70}")
    print(f"Extracting goalie save lines...")
    print(f"{'='*70}")

    lines = []

    # The structure depends on the Odds API response
    # Typically: data -> bookmakers -> markets -> outcomes

    bookmakers = props_data.get('data', {}).get('bookmakers', [])

    if not bookmakers:
        print("❌ No bookmakers found in response")
        return lines

    print(f"Found {len(bookmakers)} bookmakers")

    for bookmaker in bookmakers:
        bookmaker_name = bookmaker.get('title', 'Unknown')
        markets = bookmaker.get('markets', [])

        print(f"\n  Bookmaker: {bookmaker_name}")
        print(f"  Markets: {len(markets)}")

        for market in markets:
            market_key = market.get('key', '')

            if market_key == 'player_total_saves':
                outcomes = market.get('outcomes', [])
                print(f"    Found {len(outcomes)} player save props")

                for outcome in outcomes:
                    player_name = outcome.get('description', '')
                    line = outcome.get('point', None)
                    over_price = outcome.get('price', None)

                    if player_name and line is not None:
                        lines.append({
                            'game_id': boxscore.get('id'),
                            'game_date': boxscore.get('gameDate'),
                            'home_team': boxscore.get('homeTeam', {}).get('abbrev'),
                            'away_team': boxscore.get('awayTeam', {}).get('abbrev'),
                            'goalie_name': player_name,
                            'save_line': float(line),
                            'sportsbook': bookmaker_name,
                            'odds': over_price,
                            'timestamp': props_data.get('data', {}).get('timestamp')
                        })

                        print(f"      ✅ {player_name}: {line} saves")

    print(f"\n  Total lines extracted: {len(lines)}")
    return lines

def main():
    """Test fetching betting lines for a single game"""

    print(f"\n{'='*70}")
    print(f"NHL BETTING LINES FETCHER - SINGLE GAME TEST")
    print(f"{'='*70}")
    print(f"Game ID: {TEST_GAME_ID}")
    print(f"Date: {TEST_GAME_DATE}")
    print(f"API Key: {'✅ Loaded' if API_KEY else '❌ Missing'}")

    if not API_KEY:
        print("\n❌ ERROR: API_KEY not found in environment variables")
        print("Make sure you have a .env file with API_KEY=your_key_here")
        return

    # Step 1: Load our boxscore
    print(f"\n{'='*70}")
    print(f"Step 1: Loading boxscore...")
    print(f"{'='*70}")

    try:
        boxscore = load_boxscore(TEST_GAME_ID)
        print(f"✅ Boxscore loaded")
        print(f"   Home: {boxscore.get('homeTeam', {}).get('abbrev', 'Unknown')}")
        print(f"   Away: {boxscore.get('awayTeam', {}).get('abbrev', 'Unknown')}")
        print(f"   Date: {boxscore.get('gameDate', 'Unknown')}")
    except Exception as e:
        print(f"❌ Error loading boxscore: {e}")
        return

    # Step 2: Fetch events for that date from Odds API
    print(f"\n{'='*70}")
    print(f"Step 2: Fetching events from Odds API...")
    print(f"{'='*70}")

    events = fetch_nhl_events_for_date(TEST_GAME_DATE)

    if not events:
        print("\n❌ No events found for this date")
        print("This might mean:")
        print("  - The game date is outside the API's historical data range")
        print("  - There were no NHL games on this date")
        print("  - API request failed")
        return

    # Step 3: Match our game to an Odds API event
    print(f"\n{'='*70}")
    print(f"Step 3: Matching game to event...")
    print(f"{'='*70}")

    matched_event = match_game_to_event(boxscore, events)

    if not matched_event:
        print("\n❌ Could not match game to Odds API event")
        print("Available events:")
        for event in events:
            print(f"  - {event.get('away_team')} @ {event.get('home_team')}")
        return

    event_id = matched_event['id']

    # Step 4: Fetch goalie props for this event
    print(f"\n{'='*70}")
    print(f"Step 4: Fetching goalie props...")
    print(f"{'='*70}")

    props_data = fetch_goalie_props_for_event(event_id, TEST_GAME_DATE)

    if not props_data:
        print("\n❌ Could not fetch props data")
        return

    # Step 5: Extract goalie save lines
    print(f"\n{'='*70}")
    print(f"Step 5: Extracting goalie save lines...")
    print(f"{'='*70}")

    lines = extract_goalie_lines(props_data, boxscore)

    # Step 6: Save results
    if lines:
        print(f"\n{'='*70}")
        print(f"SUCCESS!")
        print(f"{'='*70}")
        print(f"Found {len(lines)} goalie save lines")

        for line in lines:
            print(f"\n  Goalie: {line['goalie_name']}")
            print(f"  Line: {line['save_line']} saves")
            print(f"  Sportsbook: {line['sportsbook']}")
            print(f"  Odds: {line['odds']}")

        # Save to file
        output_dir = Path('data/raw/betting_lines/test')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"test_game_{TEST_GAME_ID}_lines.json"

        with open(output_file, 'w') as f:
            json.dump({
                'game_id': TEST_GAME_ID,
                'game_date': TEST_GAME_DATE,
                'boxscore': boxscore,
                'odds_api_event': matched_event,
                'props_response': props_data,
                'extracted_lines': lines
            }, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")

    else:
        print(f"\n❌ No goalie save lines found")
        print("This might mean:")
        print("  - Sportsbooks didn't offer goalie save props for this game")
        print("  - The market wasn't available at the requested time")
        print("  - Different market name is used")

        # Save raw response for debugging
        output_dir = Path('data/raw/betting_lines/test')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"test_game_{TEST_GAME_ID}_raw_response.json"

        with open(output_file, 'w') as f:
            json.dump({
                'game_id': TEST_GAME_ID,
                'events_response': events,
                'props_response': props_data
            }, f, indent=2)

        print(f"\n📄 Raw API response saved to: {output_file}")
        print("   Review this file to see the actual API response structure")

if __name__ == "__main__":
    main()
