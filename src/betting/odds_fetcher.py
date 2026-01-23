"""
Fetch betting lines from external APIs (Underdog Fantasy, etc.)
"""
import requests
import time
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
