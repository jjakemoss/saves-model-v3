"""Main orchestration for data collection"""

import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any

from .api_client import NHLAPIClient
from .cache_manager import CacheManager, CachedNHLAPIClient
from .schedule_collector import ScheduleCollector, NHL_TEAMS
from .game_collector import GameCollector
from .player_collector import PlayerCollector


class DataCollectionOrchestrator:
    """
    Orchestrates the entire data collection process:
    1. Collect schedules for all teams
    2. Extract game IDs
    3. Collect boxscores and play-by-play
    4. Extract goalie IDs
    5. Collect goalie game logs and Edge stats
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data collection orchestrator

        Args:
            config_path: Path to configuration file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['paths']['logs'] + '/data_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize API client with caching
        self.api_client = NHLAPIClient(config_path)
        self.cache_manager = CacheManager(
            db_path=self.config['paths']['cache_db'],
            ttl_seconds=self.config['data']['cache_ttl']
        )
        self.cached_api = CachedNHLAPIClient(self.api_client, self.cache_manager)

        # Initialize collectors
        data_path = self.config['paths']['raw_data']
        self.schedule_collector = ScheduleCollector(self.cached_api, data_path)
        self.game_collector = GameCollector(self.cached_api, data_path)
        self.player_collector = PlayerCollector(self.cached_api, data_path)

        self.logger.info("Data Collection Orchestrator initialized")

    def collect_all_schedules(self, seasons: List[str], teams: List[str] = NHL_TEAMS) -> Dict[str, Any]:
        """
        Collect schedules for all teams across multiple seasons

        Args:
            seasons: List of seasons in YYYYYYYY format
            teams: List of team codes (default: all NHL teams)

        Returns:
            Summary statistics
        """
        self.logger.info(f"=" * 60)
        self.logger.info(f"PHASE 1: Collecting Schedules")
        self.logger.info(f"Seasons: {seasons}")
        self.logger.info(f"Teams: {len(teams)}")
        self.logger.info(f"=" * 60)

        total_schedules = 0
        for season in seasons:
            schedules = self.schedule_collector.collect_all_teams_schedule(season, teams)
            total_schedules += len(schedules)

        summary = {
            'total_seasons': len(seasons),
            'total_teams': len(teams),
            'total_schedules_collected': total_schedules
        }

        self.logger.info(f"Schedule collection complete: {total_schedules} schedules")
        return summary

    def collect_all_games(self, seasons: List[str], teams: List[str] = NHL_TEAMS, game_type: int = 2) -> Dict[str, Any]:
        """
        Collect boxscore and play-by-play for all games

        Args:
            seasons: List of seasons in YYYYYYYY format
            teams: List of team codes
            game_type: Game type to collect (1=preseason, 2=regular, 3=playoffs, default=2)

        Returns:
            Summary statistics
        """
        self.logger.info(f"=" * 60)
        self.logger.info(f"PHASE 2: Collecting Game Data (game type {game_type})")
        self.logger.info(f"=" * 60)

        all_game_ids = []
        for season in seasons:
            game_ids = self.schedule_collector.extract_game_ids(season, teams, game_type=game_type)
            all_game_ids.extend(game_ids)
            self.logger.info(f"{season}: {len(game_ids)} games")

        self.logger.info(f"Total games to collect: {len(all_game_ids)}")

        # Collect game data
        summary = self.game_collector.collect_games_batch(all_game_ids, batch_size=50)

        self.logger.info(f"Game collection complete")
        return summary

    def collect_all_goalie_stats(self, seasons: List[str], game_type: int = 2) -> Dict[str, Any]:
        """
        Collect goalie game logs and Edge stats

        Args:
            seasons: List of seasons in YYYYYYYY format
            game_type: 2 for regular season, 3 for playoffs

        Returns:
            Summary statistics
        """
        self.logger.info(f"=" * 60)
        self.logger.info(f"PHASE 3: Collecting Goalie Stats")
        self.logger.info(f"=" * 60)

        # Extract unique goalie IDs from collected boxscores
        boxscores_dir = Path(self.config['paths']['raw_data']) / "boxscores"
        goalie_ids = self.player_collector.extract_unique_goalie_ids(boxscores_dir)

        self.logger.info(f"Found {len(goalie_ids)} unique goalies")

        # Collect game logs and Edge stats
        summary = self.player_collector.collect_all_goalies_data(goalie_ids, seasons, game_type)

        self.logger.info(f"Goalie stats collection complete")
        return summary

    def run_full_collection(self, seasons: List[str] = None, teams: List[str] = NHL_TEAMS) -> Dict[str, Any]:
        """
        Run complete data collection pipeline

        Args:
            seasons: List of seasons (default: from config)
            teams: List of teams (default: all NHL teams)

        Returns:
            Complete summary statistics
        """
        if seasons is None:
            seasons = self.config['data']['seasons']

        self.logger.info(f"\n" + "=" * 60)
        self.logger.info(f"STARTING FULL DATA COLLECTION")
        self.logger.info(f"Seasons: {seasons}")
        self.logger.info(f"Teams: {len(teams)}")
        self.logger.info(f"=" * 60 + "\n")

        # Phase 1: Schedules
        schedule_summary = self.collect_all_schedules(seasons, teams)

        # Phase 2: Games (boxscores + play-by-play)
        game_summary = self.collect_all_games(seasons, teams)

        # Phase 3: Goalie stats
        goalie_summary = self.collect_all_goalie_stats(seasons, game_type=2)

        # Clear expired cache
        expired_count = self.cache_manager.clear_expired()
        cache_stats = self.cache_manager.get_stats()

        # Complete summary
        full_summary = {
            'schedules': schedule_summary,
            'games': game_summary,
            'goalies': goalie_summary,
            'cache': {
                'expired_cleared': expired_count,
                'stats': cache_stats
            }
        }

        self.logger.info(f"\n" + "=" * 60)
        self.logger.info(f"DATA COLLECTION COMPLETE")
        self.logger.info(f"=" * 60)
        self.logger.info(f"Schedules: {schedule_summary['total_schedules_collected']}")
        self.logger.info(f"Games (boxscores): {game_summary['successful_boxscores']}")
        self.logger.info(f"Games (play-by-play): {game_summary['successful_pbp']}")
        self.logger.info(f"Goalie game logs: {goalie_summary['successful_gamelogs']}")
        self.logger.info(f"Goalie Edge stats: {goalie_summary['successful_edge']}")
        self.logger.info(f"Cache entries: {cache_stats['active_entries']}")
        self.logger.info(f"=" * 60 + "\n")

        return full_summary
