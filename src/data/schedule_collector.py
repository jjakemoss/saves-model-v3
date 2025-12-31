"""Schedule data collector for NHL games"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class ScheduleCollector:
    """Collect game schedules for all teams and seasons"""

    def __init__(self, api_client, data_path: str = "data/raw"):
        """
        Initialize schedule collector

        Args:
            api_client: CachedNHLAPIClient instance
            data_path: Path to store raw data
        """
        self.api = api_client
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)

        # Create schedules directory
        self.schedules_dir = self.data_path / "schedules"
        self.schedules_dir.mkdir(parents=True, exist_ok=True)

    def collect_team_schedule(self, team: str, season: str) -> Dict[str, Any]:
        """
        Collect schedule for a specific team and season

        Args:
            team: Three-letter team code (e.g., 'TOR')
            season: Season in YYYYYYYY format (e.g., '20232024')

        Returns:
            Schedule data dictionary
        """
        self.logger.info(f"Collecting schedule: {team} {season}")

        try:
            schedule_data = self.api.get_team_season_schedule(team, season)

            # Save to file
            file_path = self.schedules_dir / f"{team}_{season}.json"
            with open(file_path, 'w') as f:
                json.dump(schedule_data, f, indent=2)

            self.logger.info(f"Saved schedule: {file_path}")
            return schedule_data

        except Exception as e:
            self.logger.error(f"Failed to collect schedule for {team} {season}: {e}")
            raise

    def collect_all_teams_schedule(self, season: str, teams: List[str]) -> Dict[str, Any]:
        """
        Collect schedules for all teams in a season

        Args:
            season: Season in YYYYYYYY format
            teams: List of three-letter team codes

        Returns:
            Dictionary mapping team codes to schedule data
        """
        self.logger.info(f"Collecting schedules for {len(teams)} teams in {season}")

        schedules = {}
        for team in teams:
            try:
                schedule_data = self.collect_team_schedule(team, season)
                schedules[team] = schedule_data
            except Exception as e:
                self.logger.warning(f"Skipping {team} due to error: {e}")
                continue

        self.logger.info(f"Collected {len(schedules)} team schedules for {season}")
        return schedules

    def extract_game_ids(self, season: str, teams: List[str], game_type: int = 2) -> List[int]:
        """
        Extract all unique game IDs from team schedules

        Args:
            season: Season in YYYYYYYY format
            teams: List of team codes
            game_type: Game type to filter (1=preseason, 2=regular, 3=playoffs, default=2)

        Returns:
            List of unique game IDs
        """
        game_ids = set()

        for team in teams:
            file_path = self.schedules_dir / f"{team}_{season}.json"

            if not file_path.exists():
                self.logger.warning(f"Schedule file not found: {file_path}")
                continue

            with open(file_path, 'r') as f:
                schedule_data = json.load(f)

            # Extract game IDs from schedule, filtering by game type
            if 'games' in schedule_data:
                for game in schedule_data['games']:
                    # Filter by game type (2 = regular season)
                    if game.get('gameType') == game_type and 'id' in game:
                        game_ids.add(game['id'])

        game_ids_list = sorted(list(game_ids))
        self.logger.info(f"Extracted {len(game_ids_list)} unique game IDs from {season} (game type {game_type})")
        return game_ids_list

    def get_game_metadata(self, season: str, teams: List[str], game_type: int = 2) -> List[Dict[str, Any]]:
        """
        Extract game metadata (date, teams, home/away) from schedules

        Args:
            season: Season in YYYYYYYY format
            teams: List of team codes
            game_type: Game type to filter (1=preseason, 2=regular, 3=playoffs, default=2)

        Returns:
            List of game metadata dictionaries
        """
        games_metadata = []

        for team in teams:
            file_path = self.schedules_dir / f"{team}_{season}.json"

            if not file_path.exists():
                continue

            with open(file_path, 'r') as f:
                schedule_data = json.load(f)

            if 'games' in schedule_data:
                for game in schedule_data['games']:
                    # Filter by game type
                    if game.get('gameType') == game_type:
                        game_meta = {
                            'game_id': game.get('id'),
                            'game_date': game.get('gameDate'),
                            'season': season,
                            'game_type': game.get('gameType'),
                            'home_team': game.get('homeTeam', {}).get('abbrev'),
                            'away_team': game.get('awayTeam', {}).get('abbrev'),
                            'venue': game.get('venue', {}).get('default')
                        }
                        games_metadata.append(game_meta)

        # Remove duplicates (same game appears in both team schedules)
        unique_games = {game['game_id']: game for game in games_metadata}
        games_list = list(unique_games.values())

        self.logger.info(f"Extracted metadata for {len(games_list)} games (game type {game_type})")
        return games_list


# NHL team codes (all 32 teams)
NHL_TEAMS = [
    'ANA', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ',
    'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH',
    'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SEA', 'SJS',
    'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH', 'WPG', 'ARI'
]
