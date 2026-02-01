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
        self._goalie_leaders_cache = None  # Cache for goalie stats leaders
        self._boxscore_cache = {}  # Cache boxscores by game_id
        self._schedule_cache = {}  # Cache team schedules by team_abbrev

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
        return self.get_goalie_id_by_name(last_name)

    def get_goalie_id_by_name(self, last_name):
        """
        Find goalie ID by last name using stats leaders API.

        This is more efficient than searching each game boxscore.

        Args:
            last_name: Goalie's last name (case insensitive)

        Returns:
            int: Goalie player ID if found, None otherwise
        """
        try:
            # Use cached leaders data if available
            if self._goalie_leaders_cache is None:
                season = '20252026'
                self._goalie_leaders_cache = self.api.get_goalie_stats_leaders(
                    season=season,
                    game_type=2,
                    limit=200
                )

            # Search through goalies (they're in 'wins' key)
            for goalie in self._goalie_leaders_cache.get('wins', []):
                goalie_last_name = goalie.get('lastName', {}).get('default', '')
                # Match last name (case insensitive)
                if last_name.lower() in goalie_last_name.lower():
                    return goalie.get('id')

            return None

        except Exception as e:
            print(f"Error searching for goalie by name: {e}")
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

    def _get_boxscore(self, game_id):
        """Fetch boxscore with in-memory caching"""
        if game_id not in self._boxscore_cache:
            try:
                self._boxscore_cache[game_id] = self.api.get_boxscore(game_id)
            except Exception as e:
                print(f"Error fetching boxscore for game {game_id}: {e}")
                return None
        return self._boxscore_cache[game_id]

    def _parse_situation_stat(self, stat_str, stat_type):
        """Parse 'saves/shots' format strings from boxscore"""
        try:
            if '/' not in str(stat_str):
                return 0
            saves_str, shots_str = str(stat_str).split('/')
            if stat_type == 'saves':
                return int(saves_str)
            else:
                return int(shots_str)
        except (ValueError, TypeError):
            return 0

    def get_goalie_boxscore_stats(self, game_id, goalie_id):
        """
        Fetch situation-specific goalie stats from a boxscore.

        Returns dict with: saves, shots_against, goals_against, save_percentage,
        even_strength_saves/shots_against/goals_against,
        power_play_saves/shots_against/goals_against,
        short_handed_saves/shots_against/goals_against
        """
        box = self._get_boxscore(game_id)
        if not box:
            return None

        # Search for the goalie in both teams
        for side in ['homeTeam', 'awayTeam']:
            goalies = box.get('playerByGameStats', {}).get(side, {}).get('goalies', [])
            for g in goalies:
                if g.get('playerId') == goalie_id:
                    es_sa_str = g.get('evenStrengthShotsAgainst', '0/0')
                    pp_sa_str = g.get('powerPlayShotsAgainst', '0/0')
                    sh_sa_str = g.get('shorthandedShotsAgainst', '0/0')

                    es_saves = self._parse_situation_stat(es_sa_str, 'saves')
                    es_shots = self._parse_situation_stat(es_sa_str, 'shots')
                    pp_saves = self._parse_situation_stat(pp_sa_str, 'saves')
                    pp_shots = self._parse_situation_stat(pp_sa_str, 'shots')
                    sh_saves = self._parse_situation_stat(sh_sa_str, 'saves')
                    sh_shots = self._parse_situation_stat(sh_sa_str, 'shots')

                    return {
                        'saves': g.get('saves', 0),
                        'shots_against': g.get('shotsAgainst', 0),
                        'goals_against': g.get('goalsAgainst', 0),
                        'save_percentage': g.get('savePctg', 0.0),
                        'even_strength_saves': es_saves,
                        'even_strength_shots_against': es_shots,
                        'even_strength_goals_against': g.get('evenStrengthGoalsAgainst', 0),
                        'power_play_saves': pp_saves,
                        'power_play_shots_against': pp_shots,
                        'power_play_goals_against': g.get('powerPlayGoalsAgainst', 0),
                        'short_handed_saves': sh_saves,
                        'short_handed_shots_against': sh_shots,
                        'short_handed_goals_against': g.get('shorthandedGoalsAgainst', 0),
                    }
        return None

    def get_team_boxscore_stats(self, game_id, team_abbrev):
        """
        Fetch team-level stats from a boxscore.

        Returns dict with: team_goals, team_shots, opp_goals, opp_shots
        """
        box = self._get_boxscore(game_id)
        if not box:
            return None

        home_abbrev = box.get('homeTeam', {}).get('abbrev', '')
        away_abbrev = box.get('awayTeam', {}).get('abbrev', '')

        if team_abbrev == home_abbrev:
            team_data = box.get('homeTeam', {})
            opp_data = box.get('awayTeam', {})
        elif team_abbrev == away_abbrev:
            team_data = box.get('awayTeam', {})
            opp_data = box.get('homeTeam', {})
        else:
            return None

        return {
            'team_goals': team_data.get('score', 0),
            'team_shots': team_data.get('sog', 0),
            'opp_goals': opp_data.get('score', 0),
            'opp_shots': opp_data.get('sog', 0),
        }

    def get_opponent_recent_stats(self, opponent_abbrev, game_date, season='20252026', n_games=10):
        """
        Fetch opponent team's recent game stats (goals scored, shots).

        Args:
            opponent_abbrev: Opponent team abbreviation
            game_date: Current game date (to exclude same-day/future games)
            season: Season string
            n_games: Number of recent games

        Returns:
            list of dicts with: opp_goals, opp_shots per game
        """
        # Fetch opponent schedule (cached)
        if opponent_abbrev not in self._schedule_cache:
            try:
                sched = self.api.get_team_season_schedule(opponent_abbrev, season)
                self._schedule_cache[opponent_abbrev] = sched.get('games', [])
            except Exception as e:
                print(f"Error fetching schedule for {opponent_abbrev}: {e}")
                return []

        all_games = self._schedule_cache[opponent_abbrev]

        # Filter to completed games before game_date
        completed = []
        for g in all_games:
            gd = g.get('gameDate', '')
            state = g.get('gameState', '')
            if state in ('OFF', 'FINAL') and gd < game_date:
                completed.append(g)

        # Sort descending by date, take most recent n_games
        completed.sort(key=lambda x: x.get('gameDate', ''), reverse=True)
        recent = completed[:n_games]

        results = []
        for g in recent:
            game_id = g.get('id')
            home = g.get('homeTeam', {})
            away = g.get('awayTeam', {})

            # Determine which side is the opponent team
            if home.get('abbrev') == opponent_abbrev:
                team_goals = home.get('score', 0)
                team_shots_from_box = None
            else:
                team_goals = away.get('score', 0)
                team_shots_from_box = None

            # Fetch boxscore for shots (schedule only has scores)
            box_stats = self.get_team_boxscore_stats(game_id, opponent_abbrev)
            if box_stats:
                team_shots = box_stats['team_shots']
            else:
                team_shots = 30  # fallback

            results.append({
                'opp_goals': team_goals if team_goals is not None else 3,
                'opp_shots': team_shots,
            })

        return results
