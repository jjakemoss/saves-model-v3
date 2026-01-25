"""
Fetch betting lines from external APIs (Underdog Fantasy, etc.)
"""
import requests
import time
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class UnderdogFetcher:
    """Fetch goalie saves lines from Underdog Fantasy API"""

    BASE_URL = "https://api.underdogfantasy.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Origin': 'https://underdogfantasy.com',
            'Referer': 'https://underdogfantasy.com/'
        })

    def get_goalie_saves(self) -> list[dict]:
        """
        Fetch all NHL goalie saves lines from Underdog.

        Returns:
            List of dicts with keys:
            - book: 'Underdog'
            - player_name: Full goalie name (e.g., 'Joseph Woll')
            - line: Saves line (e.g., 24.5)
            - line_over: American odds for OVER (e.g., -125)
            - line_under: American odds for UNDER (e.g., 102)
            - game_time: ISO timestamp of game start
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/v1/over_under_lines",
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to fetch Underdog data: {e}")
            return []

        # Build player lookup from appearances
        players = {p['id']: p for p in data.get('players', [])}
        appearances = {a['id']: a for a in data.get('appearances', [])}
        games = {g['id']: g for g in data.get('games', [])}

        lines = []

        for line_data in data.get('over_under_lines', []):
            over_under = line_data.get('over_under', {})
            title = over_under.get('title', '')

            # Filter for goalie saves only (full game, not period-specific)
            title_lower = title.lower()
            if 'saves' not in title_lower:
                continue
            # Skip period-specific lines (e.g., "1st Period Saves", "1H Saves")
            if '1st period' in title_lower or '1h ' in title_lower or ' 1h' in title_lower:
                continue

            # Get the betting line value
            stat_value = line_data.get('stat_value')
            if stat_value is None:
                continue

            # Filter out period lines by value (full game lines are typically 18+)
            if float(stat_value) < 15:
                continue

            # Parse options for odds
            options = line_data.get('options', [])
            line_over = None
            line_under = None
            player_name = None

            for opt in options:
                choice = opt.get('choice', '')
                american_price = opt.get('american_price')

                # Get player name from selection_header
                if not player_name:
                    player_name = opt.get('selection_header')

                if american_price:
                    # Convert string to int (e.g., "-125" -> -125, "+102" -> 102)
                    try:
                        odds_value = int(american_price.replace('+', ''))
                    except (ValueError, AttributeError):
                        odds_value = None

                    if choice == 'higher':
                        line_over = odds_value
                    elif choice == 'lower':
                        line_under = odds_value

            if not player_name:
                continue

            # Skip truncated/invalid player names (minimum 4 chars for last name)
            if len(player_name.strip()) < 3:
                continue

            # Try to get game time from appearances
            game_time = None
            appearance_stat = over_under.get('appearance_stat', {})
            appearance_id = appearance_stat.get('appearance_id')

            if appearance_id and appearance_id in appearances:
                appearance = appearances[appearance_id]
                match_id = appearance.get('match_id')
                if match_id and match_id in games:
                    game_time = games[match_id].get('scheduled_at')

            lines.append({
                'book': 'Underdog',
                'player_name': player_name,
                'line': float(stat_value),
                'line_over': line_over,
                'line_under': line_under,
                'game_time': game_time,
            })

        return lines


class PrizePicksFetcher:
    """Fetch goalie saves lines from PrizePicks API"""

    BASE_URL = "https://api.prizepicks.com"
    NHL_LEAGUE_ID = 8

    # Implied odds for PrizePicks projection types (3-pick flex baseline)
    IMPLIED_ODDS = {
        'standard': -120,
        'demon': -140,
        'goblin': -105
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })

    def get_goalie_saves(self) -> list[dict]:
        """
        Fetch all NHL goalie saves lines from PrizePicks.

        Returns:
            List of dicts with keys:
            - book: 'PrizePicks'
            - player_name: Full goalie name (e.g., 'Connor Hellebuyck')
            - line: Saves line (e.g., 28.5)
            - line_over: Implied American odds for OVER based on odds_type
            - line_under: None (PrizePicks only allows MORE picks)
            - odds_type: 'standard', 'demon', or 'goblin'
            - game_time: ISO timestamp of game start
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/projections",
                params={
                    'league_id': self.NHL_LEAGUE_ID,
                    'per_page': 250,
                    'single_stat': 'true'
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[WARNING] Failed to fetch PrizePicks data: {e}")
            return []

        # Check if blocked by anti-bot
        if not isinstance(data, dict) or 'data' not in data:
            print("[WARNING] PrizePicks API blocked or invalid response")
            return []

        # Build player lookup from included data
        players = {}
        for item in data.get('included', []):
            if item.get('type') == 'new_player':
                players[item['id']] = {
                    'name': item['attributes'].get('display_name', 'Unknown'),
                    'team': item['attributes'].get('team', 'Unknown')
                }

        lines = []

        for proj in data.get('data', []):
            attrs = proj.get('attributes', {})
            stat_type = attrs.get('stat_type', '')

            # Filter for goalie saves only
            if 'Goalie Saves' not in stat_type:
                continue

            # Get line value
            line_score = attrs.get('line_score')
            if line_score is None:
                continue

            # Filter out period lines by value (full game lines are typically 18+)
            if float(line_score) < 15:
                continue

            # Get player info
            player_id = proj.get('relationships', {}).get('new_player', {}).get('data', {}).get('id')
            player_info = players.get(player_id, {'name': 'Unknown', 'team': 'Unknown'})

            if player_info['name'] == 'Unknown':
                continue

            # Skip truncated/invalid player names
            if len(player_info['name'].strip()) < 3:
                continue

            # Get odds type - only include standard lines (skip demons/goblins)
            odds_type = attrs.get('odds_type', 'standard')
            if odds_type != 'standard':
                continue

            implied_odds = self.IMPLIED_ODDS.get(odds_type, -120)

            lines.append({
                'book': 'PrizePicks',
                'player_name': player_info['name'],
                'line': float(line_score),
                'line_over': implied_odds,
                'line_under': implied_odds,  # Same implied odds for both sides
                'odds_type': odds_type,
                'game_time': attrs.get('start_time'),
            })

        return lines


class TheOddsAPIFetcher:
    """Fetch goalie saves lines from The-Odds-API (BetOnline)"""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "icehockey_nhl"
    CACHE_DIR = Path("data/cache/odds_api")
    CACHE_TTL_MINUTES = 5  # Cache valid for 5 minutes

    def __init__(self, api_key: str = None):
        """
        Initialize fetcher with API key.

        Args:
            api_key: The-Odds-API key. If None, reads from .env file or environment.
        """
        self.api_key = api_key or self._load_api_key()
        self.session = requests.Session()

    def _load_api_key(self) -> str:
        """Load API key from environment or .env file"""
        # Try environment variable first
        api_key = os.environ.get('THE_ODDS_API_KEY') or os.environ.get('API_KEY')
        if api_key:
            return api_key

        # Try .env file
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('API_KEY=') or line.startswith('THE_ODDS_API_KEY='):
                        return line.split('=', 1)[1].strip().strip('"\'')

        return ''

    def _get_cache_path(self, date_str: str) -> Path:
        """Get cache file path for a specific date"""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self.CACHE_DIR / f"betonline_{date_str}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is still valid (not expired)"""
        if not cache_path.exists():
            return False

        # Check file age
        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_minutes = (datetime.now() - file_mtime).total_seconds() / 60

        return age_minutes < self.CACHE_TTL_MINUTES

    def _load_from_cache(self, cache_path: Path) -> list[dict]:
        """Load lines from cache file"""
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data.get('lines', [])
        except (json.JSONDecodeError, IOError):
            return []

    def _save_to_cache(self, cache_path: Path, lines: list[dict]):
        """Save lines to cache file"""
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'cached_at': datetime.now().isoformat(),
                    'lines': lines
                }, f, indent=2)
        except IOError as e:
            print(f"[WARNING] Failed to save cache: {e}")

    def get_goalie_saves(self, date_str: str = None) -> list[dict]:
        """
        Fetch BetOnline goalie saves lines from The-Odds-API.

        Uses event-based endpoint for player props. Cache is valid for 30 minutes.

        Args:
            date_str: Date string (YYYY-MM-DD). If None, uses today.

        Returns:
            List of dicts with keys:
            - book: 'BetOnline'
            - player_name: Full goalie name
            - line: Saves line (e.g., 28.5)
            - line_over: American odds for OVER
            - line_under: American odds for UNDER
            - game_time: ISO timestamp of game start
        """
        if not self.api_key:
            print("[WARNING] No API key for The-Odds-API. Set API_KEY in .env file.")
            return []

        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        # Check cache first
        cache_path = self._get_cache_path(date_str)
        if self._is_cache_valid(cache_path):
            cached_lines = self._load_from_cache(cache_path)
            if cached_lines:
                print(f"    [CACHE] Using cached BetOnline data ({len(cached_lines)} lines)")
                return cached_lines

        # Step 1: Get list of events for today
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            commence_from = date_obj.strftime("%Y-%m-%dT00:00:00Z")
            commence_to = date_obj.strftime("%Y-%m-%dT23:59:59Z")

            response = self.session.get(
                f"{self.BASE_URL}/sports/{self.SPORT}/events",
                params={
                    'apiKey': self.api_key,
                    'commenceTimeFrom': commence_from,
                    'commenceTimeTo': commence_to,
                },
                timeout=15
            )
            response.raise_for_status()
            events = response.json()

            remaining = response.headers.get('x-requests-remaining', 'unknown')
            print(f"    [API] Found {len(events)} events (requests remaining: {remaining})")

        except requests.exceptions.RequestException as e:
            print(f"[WARNING] Failed to fetch events: {e}")
            if cache_path.exists():
                print("    [CACHE] Using stale cache as fallback")
                return self._load_from_cache(cache_path)
            return []

        if not events:
            return []

        # Step 2: Fetch player props for each event
        all_lines = []
        for event in events:
            event_id = event.get('id')
            game_time = event.get('commence_time')

            try:
                response = self.session.get(
                    f"{self.BASE_URL}/sports/{self.SPORT}/events/{event_id}/odds",
                    params={
                        'apiKey': self.api_key,
                        'regions': 'us',
                        'markets': 'player_total_saves',
                        'bookmakers': 'betonlineag',
                        'oddsFormat': 'american'
                    },
                    timeout=15
                )
                response.raise_for_status()
                event_data = response.json()

                lines = self._parse_event_response(event_data, game_time)
                all_lines.extend(lines)

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 422:
                    print(f"    [WARNING] player_total_saves market requires Champion tier")
                    break  # No point trying other events
                continue
            except requests.exceptions.RequestException:
                continue

        remaining = response.headers.get('x-requests-remaining', 'unknown')
        if all_lines:
            print(f"    [API] Found {len(all_lines)} lines (requests remaining: {remaining})")

        # Save to cache
        self._save_to_cache(cache_path, all_lines)

        return all_lines

    def _parse_event_response(self, event_data: dict, game_time: str) -> list[dict]:
        """Parse event-level response into standard format"""
        lines = []

        for bookmaker in event_data.get('bookmakers', []):
            if bookmaker.get('key') != 'betonlineag':
                continue

            for market in bookmaker.get('markets', []):
                if market.get('key') != 'player_total_saves':
                    continue

                # Group outcomes by player (Over/Under pairs)
                player_lines = {}
                for outcome in market.get('outcomes', []):
                    player_name = outcome.get('description')
                    if not player_name:
                        continue

                    line_value = outcome.get('point')
                    if line_value is None:
                        continue

                    # Filter out period lines (full game lines typically 18+)
                    if float(line_value) < 15:
                        continue

                    side = outcome.get('name', '').lower()
                    price = outcome.get('price')

                    if player_name not in player_lines:
                        player_lines[player_name] = {
                            'line': line_value,
                            'line_over': None,
                            'line_under': None
                        }

                    if side == 'over':
                        player_lines[player_name]['line_over'] = price
                    elif side == 'under':
                        player_lines[player_name]['line_under'] = price

                # Convert to list format
                for player_name, line_data in player_lines.items():
                    # Skip if missing both odds
                    if line_data['line_over'] is None and line_data['line_under'] is None:
                        continue

                    lines.append({
                        'book': 'BetOnline',
                        'player_name': player_name,
                        'line': float(line_data['line']),
                        'line_over': line_data['line_over'],
                        'line_under': line_data['line_under'],
                        'game_time': game_time,
                    })

        return lines


def extract_last_name(full_name: str) -> str:
    """
    Extract last name from full player name.

    Examples:
        'Joseph Woll' -> 'Woll'
        'Marc-Andre Fleury' -> 'Fleury'
        'Jean-Francois Berube' -> 'Berube'
    """
    if not full_name:
        return ''
    parts = full_name.strip().split()
    return parts[-1] if parts else full_name
