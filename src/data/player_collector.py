"""Player and goalie stats collector"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional


class PlayerCollector:
    """Collect player/goalie statistics and game logs"""

    def __init__(self, api_client, data_path: str = "data/raw"):
        """
        Initialize player collector

        Args:
            api_client: CachedNHLAPIClient instance
            data_path: Path to store raw data
        """
        self.api = api_client
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)

        # Create directories
        self.gamelogs_dir = self.data_path / "game_logs"
        self.player_profiles_dir = self.data_path / "player_profiles"
        self.edge_stats_dir = self.data_path / "edge_stats"

        self.gamelogs_dir.mkdir(parents=True, exist_ok=True)
        self.player_profiles_dir.mkdir(parents=True, exist_ok=True)
        self.edge_stats_dir.mkdir(parents=True, exist_ok=True)

    def collect_player_game_log(self, player_id: int, season: str, game_type: int = 2) -> Optional[Dict[str, Any]]:
        """
        Collect game log for a player/goalie

        Args:
            player_id: Player ID
            season: Season in YYYYYYYY format
            game_type: 2 for regular season, 3 for playoffs

        Returns:
            Game log data or None if failed
        """
        try:
            self.logger.debug(f"Collecting game log: Player {player_id}, {season}")
            game_log = self.api.get_player_game_log(player_id, season, game_type)

            # Save to file
            file_path = self.gamelogs_dir / f"{player_id}_{season}_{game_type}.json"
            with open(file_path, 'w') as f:
                json.dump(game_log, f, indent=2)

            self.logger.info(f"Saved game log: {player_id}_{season}_{game_type}")
            return game_log

        except Exception as e:
            self.logger.error(f"Failed to collect game log for player {player_id} {season}: {e}")
            return None

    def collect_goalie_edge_stats(self, player_id: int, season: str, game_type: int = 2) -> Optional[Dict[str, Any]]:
        """
        Collect NHL Edge stats for a goalie (shot location, save %, rebounds)

        Args:
            player_id: Player ID
            season: Season in YYYYYYYY format
            game_type: 2 for regular season, 3 for playoffs

        Returns:
            Edge stats data or None if failed
        """
        try:
            self.logger.debug(f"Collecting Edge stats: Goalie {player_id}, {season}")

            # Get shot location detail
            shot_location = self.api.get_goalie_shot_location_detail(player_id, season, game_type)

            # Get goalie comparison (includes rebound info)
            comparison = self.api.get_goalie_comparison(player_id, season, game_type)

            # Get 5v5 detail
            fivev5 = self.api.get_goalie_5v5_detail(player_id, season, game_type)

            edge_data = {
                'player_id': player_id,
                'season': season,
                'game_type': game_type,
                'shot_location': shot_location,
                'comparison': comparison,
                '5v5_detail': fivev5
            }

            # Save to file
            file_path = self.edge_stats_dir / f"goalie_{player_id}_{season}_{game_type}.json"
            with open(file_path, 'w') as f:
                json.dump(edge_data, f, indent=2)

            self.logger.info(f"Saved Edge stats: goalie_{player_id}_{season}_{game_type}")
            return edge_data

        except Exception as e:
            self.logger.error(f"Failed to collect Edge stats for goalie {player_id} {season}: {e}")
            return None

    def collect_all_goalies_data(self, goalie_ids: Set[int], seasons: List[str], game_type: int = 2) -> Dict[str, Any]:
        """
        Collect game logs and Edge stats for all goalies

        Args:
            goalie_ids: Set of goalie player IDs
            seasons: List of seasons in YYYYYYYY format
            game_type: 2 for regular season, 3 for playoffs

        Returns:
            Summary statistics
        """
        total_goalies = len(goalie_ids)
        total_collections = total_goalies * len(seasons)
        successful_gamelogs = 0
        successful_edge = 0

        self.logger.info(f"Collecting data for {total_goalies} goalies across {len(seasons)} seasons")

        idx = 0
        for player_id in goalie_ids:
            for season in seasons:
                idx += 1

                # Collect game log
                game_log = self.collect_player_game_log(player_id, season, game_type)
                if game_log:
                    successful_gamelogs += 1

                # Collect Edge stats
                edge_stats = self.collect_goalie_edge_stats(player_id, season, game_type)
                if edge_stats:
                    successful_edge += 1

                # Progress update every 20 collections
                if idx % 20 == 0 or idx == total_collections:
                    self.logger.info(
                        f"Progress: {idx}/{total_collections} | "
                        f"Game logs: {successful_gamelogs} | "
                        f"Edge stats: {successful_edge}"
                    )

        summary = {
            'total_goalies': total_goalies,
            'total_seasons': len(seasons),
            'total_collections': total_collections,
            'successful_gamelogs': successful_gamelogs,
            'successful_edge': successful_edge,
            'success_rate': (successful_gamelogs + successful_edge) / (2 * total_collections) * 100 if total_collections > 0 else 0.0
        }

        self.logger.info(
            f"Goalie data collection complete: {summary['success_rate']:.1f}% success rate"
        )

        return summary

    def extract_unique_goalie_ids(self, boxscores_dir: Path) -> Set[int]:
        """
        Extract unique goalie IDs from all boxscore files

        Args:
            boxscores_dir: Path to boxscores directory

        Returns:
            Set of unique goalie player IDs
        """
        goalie_ids = set()

        boxscore_files = list(boxscores_dir.glob("*.json"))
        self.logger.info(f"Scanning {len(boxscore_files)} boxscore files for goalies")

        for file_path in boxscore_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    boxscore = json.load(f)

                # Extract goalies from playerByGameStats structure
                player_stats = boxscore.get('playerByGameStats', {})

                # Extract home team goalies
                home_goalies = player_stats.get('homeTeam', {}).get('goalies', [])
                for goalie in home_goalies:
                    if 'playerId' in goalie:
                        goalie_ids.add(goalie['playerId'])

                # Extract away team goalies
                away_goalies = player_stats.get('awayTeam', {}).get('goalies', [])
                for goalie in away_goalies:
                    if 'playerId' in goalie:
                        goalie_ids.add(goalie['playerId'])

            except Exception as e:
                self.logger.warning(f"Error reading {file_path}: {e}")
                continue

        self.logger.info(f"Found {len(goalie_ids)} unique goalies")
        return goalie_ids
