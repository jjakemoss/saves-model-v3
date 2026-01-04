"""NHL API Client with rate limiting, caching, and retry logic"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml


class NHLAPIClient:
    """
    Wrapper for NHL API endpoints with:
    - Rate limiting (10 req/sec default)
    - Automatic retry with exponential backoff
    - Request/response logging
    - Error handling
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize NHL API client with configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.base_url = config['api']['base_url']
        self.timeout = config['api']['timeout']
        self.retry_attempts = config['api']['retry_attempts']
        self.rate_limit = config['api']['rate_limit']  # requests per second
        self.retry_delay = config['api'].get('retry_delay', 1)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0 / self.rate_limit

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Setup session with retry strategy
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make HTTP GET request to NHL API

        Args:
            endpoint: API endpoint (e.g., '/v1/schedule/now')
            params: Optional query parameters

        Returns:
            JSON response as dictionary

        Raises:
            requests.exceptions.RequestException: If request fails
        """
        self._enforce_rate_limit()

        url = f"{self.base_url}{endpoint}"

        try:
            self.logger.debug(f"GET {url} with params {params}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            self.logger.info(f"Success: GET {endpoint}")
            return response.json()

        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout: GET {endpoint}")
            raise
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP Error {response.status_code}: GET {endpoint}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: GET {endpoint} - {str(e)}")
            raise

    # Schedule Endpoints

    def get_schedule_now(self) -> Dict[str, Any]:
        """Get current schedule"""
        return self._make_request("/v1/schedule/now")

    def get_schedule_by_date(self, date: str) -> Dict[str, Any]:
        """
        Get schedule for specific date

        Args:
            date: Date in YYYY-MM-DD format
        """
        return self._make_request(f"/v1/schedule/{date}")

    def get_team_season_schedule(self, team: str, season: str) -> Dict[str, Any]:
        """
        Get team's schedule for a season

        Args:
            team: Three-letter team code (e.g., 'TOR')
            season: Season in YYYYYYYY format (e.g., '20232024')
        """
        return self._make_request(f"/v1/club-schedule-season/{team}/{season}")

    # Game Endpoints

    def get_boxscore(self, game_id: int) -> Dict[str, Any]:
        """
        Get boxscore for a specific game

        Args:
            game_id: Game ID (e.g., 2023020204)
        """
        return self._make_request(f"/v1/gamecenter/{game_id}/boxscore")

    def get_play_by_play(self, game_id: int) -> Dict[str, Any]:
        """
        Get play-by-play data for a specific game (includes shot locations, xG)

        Args:
            game_id: Game ID
        """
        return self._make_request(f"/v1/gamecenter/{game_id}/play-by-play")

    def get_game_landing(self, game_id: int) -> Dict[str, Any]:
        """Get game landing page data"""
        return self._make_request(f"/v1/gamecenter/{game_id}/landing")

    # Player/Goalie Endpoints

    def get_player_game_log(self, player_id: int, season: str, game_type: int = 2) -> Dict[str, Any]:
        """
        Get player's game log for a season

        Args:
            player_id: Player ID (e.g., 8478402)
            season: Season in YYYYYYYY format
            game_type: 2 for regular season, 3 for playoffs
        """
        return self._make_request(f"/v1/player/{player_id}/game-log/{season}/{game_type}")

    def get_player_landing(self, player_id: int) -> Dict[str, Any]:
        """Get player profile/landing page"""
        return self._make_request(f"/v1/player/{player_id}/landing")

    def get_goalie_stats_leaders(self, season: str, game_type: int = 2,
                                 categories: str = "wins", limit: int = 10) -> Dict[str, Any]:
        """
        Get goalie stats leaders

        Args:
            season: Season in YYYYYYYY format
            game_type: 2 for regular season, 3 for playoffs
            categories: Stats category (e.g., 'wins', 'saves', 'savePercentage')
            limit: Number of results (-1 for all)
        """
        params = {"categories": categories, "limit": limit}
        return self._make_request(f"/v1/goalie-stats-leaders/{season}/{game_type}", params)

    # Team Endpoints

    def get_club_stats(self, team: str, season: str, game_type: int = 2) -> Dict[str, Any]:
        """
        Get team stats for a season

        Args:
            team: Three-letter team code
            season: Season in YYYYYYYY format
            game_type: 2 for regular season, 3 for playoffs
        """
        return self._make_request(f"/v1/club-stats/{team}/{season}/{game_type}")

    def get_standings(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get standings (current or for specific date)

        Args:
            date: Optional date in YYYY-MM-DD format (None for current)
        """
        if date:
            return self._make_request(f"/v1/standings/{date}")
        return self._make_request("/v1/standings/now")

    # NHL Edge Endpoints (Advanced Analytics)

    def get_goalie_shot_location_detail(self, player_id: int, season: str, game_type: int = 2) -> Dict[str, Any]:
        """
        Get goalie shot location and save % details (NHL Edge)

        Args:
            player_id: Player ID
            season: Season in YYYYYYYY format
            game_type: 2 for regular season, 3 for playoffs
        """
        return self._make_request(f"/v1/edge/goalie-shot-location-detail/{player_id}/{season}/{game_type}")

    def get_goalie_comparison(self, player_id: int, season: str, game_type: int = 2) -> Dict[str, Any]:
        """Get goalie comparison data (shot location, save %, rebound info)"""
        return self._make_request(f"/v1/edge/goalie-comparison/{player_id}/{season}/{game_type}")

    def get_goalie_5v5_detail(self, player_id: int, season: str, game_type: int = 2) -> Dict[str, Any]:
        """Get goalie 5v5 save percentage details"""
        return self._make_request(f"/v1/edge/goalie-5v5-detail/{player_id}/{season}/{game_type}")

    def get_team_comparison(self, team_id: int, season: str, game_type: int = 2) -> Dict[str, Any]:
        """
        Get team comparison data (defensive system, shot distance, etc.)

        Args:
            team_id: Team ID (numeric, e.g., 10 for TOR)
            season: Season in YYYYYYYY format
            game_type: 2 for regular season, 3 for playoffs
        """
        return self._make_request(f"/v1/edge/team-comparison/{team_id}/{season}/{game_type}")

    # Utility Endpoints

    def get_meta(self, players: Optional[str] = None, teams: Optional[str] = None) -> Dict[str, Any]:
        """
        Get meta information (team/player IDs, etc.)

        Args:
            players: Comma-separated player IDs
            teams: Comma-separated team codes
        """
        params = {}
        if players:
            params['players'] = players
        if teams:
            params['teams'] = teams
        return self._make_request("/v1/meta", params)


# Team code to ID mapping (for NHL Edge API)
TEAM_ID_MAP = {
    'ANA': 24, 'BOS': 6, 'BUF': 7, 'CGY': 20, 'CAR': 12, 'CHI': 16,
    'COL': 21, 'CBJ': 29, 'DAL': 25, 'DET': 17, 'EDM': 22, 'FLA': 13,
    'LAK': 26, 'MIN': 30, 'MTL': 8, 'NSH': 18, 'NJD': 1, 'NYI': 2,
    'NYR': 3, 'OTT': 9, 'PHI': 4, 'PIT': 5, 'SEA': 55, 'SJS': 28,
    'STL': 19, 'TBL': 14, 'TOR': 10, 'VAN': 23, 'VGK': 54, 'WSH': 15,
    'WPG': 52, 'ARI': 53
}


def get_team_id(team_code: str) -> int:
    """Convert team code to ID for Edge API"""
    return TEAM_ID_MAP.get(team_code.upper(), None)
