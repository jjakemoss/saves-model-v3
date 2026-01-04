"""
NHL API data fetcher for betting predictions
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.api_client import NHLAPIClient


class NHLBettingData:
    """Fetch NHL game data for betting predictions"""

    def __init__(self):
        self.api = NHLAPIClient()

    def get_todays_games(self, date=None):
        """
        Fetch today's NHL schedule with game info

        Args:
            date: Date string (YYYY-MM-DD). If None, uses today

        Returns:
            list: List of game dicts with basic info
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        schedule = self.api.get_schedule_by_date(date)

        games = []
        if 'gameWeek' in schedule:
            for week in schedule['gameWeek']:
                # Only process games for the requested date
                week_date = week.get('date', '')
                if week_date != date:
                    continue

                for game in week.get('games', []):
                    # Only include regular season games
                    if game.get('gameType') != 2:
                        continue

                    game_info = {
                        'game_id': game['id'],
                        'game_date': date,
                        'home_team': game.get('homeTeam', {}).get('abbrev'),
                        'away_team': game.get('awayTeam', {}).get('abbrev'),
                        'game_state': game.get('gameState', ''),
                    }
                    games.append(game_info)

        return games

    def get_starting_goalies(self, game_id):
        """
        Try to identify starting goalies for a game

        Args:
            game_id: NHL game ID

        Returns:
            dict: {'home': {'id': ..., 'name': ...}, 'away': {...}}
        """
        try:
            # Try to get from boxscore (works for completed games and sometimes live)
            boxscore = self.api.get_boxscore(game_id)

            starters = {'home': None, 'away': None}

            # Check home goalies
            home_goalies = boxscore.get('playerByGameStats', {}).get('homeTeam', {}).get('goalies', [])
            for goalie in home_goalies:
                if goalie.get('starter', False):
                    starters['home'] = {
                        'id': goalie.get('playerId'),
                        'name': goalie.get('name', {}).get('default', 'Unknown'),
                    }
                    break

            # Check away goalies
            away_goalies = boxscore.get('playerByGameStats', {}).get('awayTeam', {}).get('goalies', [])
            for goalie in away_goalies:
                if goalie.get('starter', False):
                    starters['away'] = {
                        'id': goalie.get('playerId'),
                        'name': goalie.get('name', {}).get('default', 'Unknown'),
                    }
                    break

            return starters

        except:
            # If boxscore not available, return TBD
            return {'home': None, 'away': None}

    def get_goalie_id_from_game(self, game_id, last_name):
        """
        Find goalie ID from game roster by matching last name

        First tries current game roster, then falls back to searching recent games

        Args:
            game_id: NHL game ID
            last_name: Goalie's last name (case insensitive)

        Returns:
            int: Goalie player ID if found, None otherwise
        """
        try:
            # Try current game roster first (works if lineups announced)
            boxscore = self.api.get_boxscore(game_id)

            # Check both teams
            for team_key in ['homeTeam', 'awayTeam']:
                goalies = boxscore.get('playerByGameStats', {}).get(team_key, {}).get('goalies', [])
                for goalie in goalies:
                    full_name = goalie.get('name', {}).get('default', '')
                    # Match last name (case insensitive)
                    if last_name.lower() in full_name.lower():
                        return goalie.get('playerId')

            # If not found in current roster, search recent games (fallback)
            return self._search_goalie_in_recent_games(last_name)

        except Exception as e:
            print(f"Error looking up goalie ID: {e}")
            # Try fallback search
            return self._search_goalie_in_recent_games(last_name)

    def _search_goalie_in_recent_games(self, last_name):
        """
        Search for goalie ID by looking at recent games from this season

        Args:
            last_name: Goalie's last name (case insensitive)

        Returns:
            int: Goalie player ID if found, None otherwise
        """
        try:
            # Get goalie stats leaders to find active goalies
            season = '20252026'
            leaders = self.api.get_goalie_stats_leaders(
                season=season,
                game_type=2,
                limit=200  # Get top 200 goalies
            )

            # Search through goalies (they're in 'wins' key)
            for goalie in leaders.get('wins', []):
                goalie_last_name = goalie.get('lastName', {}).get('default', '')
                # Match last name (case insensitive)
                if last_name.lower() in goalie_last_name.lower():
                    return goalie.get('id')

            return None

        except Exception as e:
            print(f"Error searching recent games: {e}")
            return None

    def get_game_result(self, game_id):
        """
        Fetch completed game boxscore and extract goalie saves

        Args:
            game_id: NHL game ID

        Returns:
            dict: {goalie_id: saves_count} for all goalies in game
        """
        try:
            boxscore = self.api.get_boxscore(game_id)

            goalie_saves = {}

            # Extract home goalies
            home_goalies = boxscore.get('playerByGameStats', {}).get('homeTeam', {}).get('goalies', [])
            for goalie in home_goalies:
                goalie_id = goalie.get('playerId')
                saves_str = goalie.get('saveShotsAgainst', '0/0')

                # Parse "saves/shots" format
                if '/' in saves_str:
                    saves = int(saves_str.split('/')[0])
                else:
                    saves = 0

                goalie_saves[goalie_id] = saves

            # Extract away goalies
            away_goalies = boxscore.get('playerByGameStats', {}).get('awayTeam', {}).get('goalies', [])
            for goalie in away_goalies:
                goalie_id = goalie.get('playerId')
                saves_str = goalie.get('saveShotsAgainst', '0/0')

                if '/' in saves_str:
                    saves = int(saves_str.split('/')[0])
                else:
                    saves = 0

                goalie_saves[goalie_id] = saves

            return goalie_saves

        except Exception as e:
            print(f"Error fetching game result for {game_id}: {e}")
            return {}

    def get_goalie_recent_games(self, goalie_id, season='20252026', n_games=15):
        """
        Fetch last N games for a goalie

        Args:
            goalie_id: NHL player ID
            season: Season in YYYYYYYY format
            n_games: Number of recent games to fetch

        Returns:
            list: List of game dicts with goalie stats
        """
        try:
            game_log = self.api.get_player_game_log(goalie_id, season, game_type=2)

            games = game_log.get('gameLog', [])

            # Sort by date descending and take last n_games
            games_sorted = sorted(games, key=lambda x: x.get('gameDate', ''), reverse=True)
            recent_games = games_sorted[:n_games]

            return recent_games

        except Exception as e:
            print(f"Error fetching recent games for goalie {goalie_id}: {e}")
            return []
