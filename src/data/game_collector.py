"""Game data collector (boxscore and play-by-play)"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class GameCollector:
    """Collect boxscore and play-by-play data for NHL games"""

    def __init__(self, api_client, data_path: str = "data/raw"):
        """
        Initialize game collector

        Args:
            api_client: CachedNHLAPIClient instance
            data_path: Path to store raw data
        """
        self.api = api_client
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)

        # Create directories
        self.boxscores_dir = self.data_path / "boxscores"
        self.pbp_dir = self.data_path / "play_by_play"
        self.boxscores_dir.mkdir(parents=True, exist_ok=True)
        self.pbp_dir.mkdir(parents=True, exist_ok=True)

    def collect_boxscore(self, game_id: int) -> Optional[Dict[str, Any]]:
        """
        Collect boxscore data for a game

        Args:
            game_id: Game ID

        Returns:
            Boxscore data or None if failed
        """
        try:
            self.logger.debug(f"Collecting boxscore: {game_id}")
            boxscore_data = self.api.get_boxscore(game_id)

            # Save to file
            file_path = self.boxscores_dir / f"{game_id}.json"
            with open(file_path, 'w') as f:
                json.dump(boxscore_data, f, indent=2)

            self.logger.info(f"Saved boxscore: {game_id}")
            return boxscore_data

        except Exception as e:
            self.logger.error(f"Failed to collect boxscore {game_id}: {e}")
            return None

    def collect_play_by_play(self, game_id: int) -> Optional[Dict[str, Any]]:
        """
        Collect play-by-play data for a game

        Args:
            game_id: Game ID

        Returns:
            Play-by-play data or None if failed
        """
        try:
            self.logger.debug(f"Collecting play-by-play: {game_id}")
            pbp_data = self.api.get_play_by_play(game_id)

            # Save to file
            file_path = self.pbp_dir / f"{game_id}.json"
            with open(file_path, 'w') as f:
                json.dump(pbp_data, f, indent=2)

            self.logger.info(f"Saved play-by-play: {game_id}")
            return pbp_data

        except Exception as e:
            self.logger.error(f"Failed to collect play-by-play {game_id}: {e}")
            return None

    def collect_game(self, game_id: int) -> Dict[str, Any]:
        """
        Collect both boxscore and play-by-play for a game

        Args:
            game_id: Game ID

        Returns:
            Dictionary with collection status
        """
        result = {
            'game_id': game_id,
            'boxscore': False,
            'play_by_play': False
        }

        boxscore = self.collect_boxscore(game_id)
        if boxscore:
            result['boxscore'] = True

        pbp = self.collect_play_by_play(game_id)
        if pbp:
            result['play_by_play'] = True

        return result

    def collect_games_batch(self, game_ids: List[int], batch_size: int = 50) -> Dict[str, Any]:
        """
        Collect data for multiple games in batches

        Args:
            game_ids: List of game IDs
            batch_size: Number of games to process between progress updates

        Returns:
            Summary statistics
        """
        total_games = len(game_ids)
        successful_boxscores = 0
        successful_pbp = 0
        failed_games = []

        self.logger.info(f"Collecting data for {total_games} games")

        for idx, game_id in enumerate(game_ids, 1):
            result = self.collect_game(game_id)

            if result['boxscore']:
                successful_boxscores += 1
            if result['play_by_play']:
                successful_pbp += 1

            if not result['boxscore'] and not result['play_by_play']:
                failed_games.append(game_id)

            # Progress update
            if idx % batch_size == 0 or idx == total_games:
                self.logger.info(
                    f"Progress: {idx}/{total_games} games | "
                    f"Boxscores: {successful_boxscores} | "
                    f"Play-by-play: {successful_pbp}"
                )

        summary = {
            'total_games': total_games,
            'successful_boxscores': successful_boxscores,
            'successful_pbp': successful_pbp,
            'failed_games': failed_games,
            'success_rate': (successful_boxscores + successful_pbp) / (2 * total_games) * 100
        }

        self.logger.info(
            f"Collection complete: {summary['success_rate']:.1f}% success rate | "
            f"{len(failed_games)} failed games"
        )

        return summary

    def extract_goalie_performance(self, game_id: int) -> List[Dict[str, Any]]:
        """
        Extract goalie performance data from boxscore

        Args:
            game_id: Game ID

        Returns:
            List of goalie performance dictionaries
        """
        file_path = self.boxscores_dir / f"{game_id}.json"

        if not file_path.exists():
            self.logger.warning(f"Boxscore not found: {game_id}")
            return []

        with open(file_path, 'r') as f:
            boxscore = json.load(f)

        goalies = []

        # Extract home team goalies
        if 'homeTeam' in boxscore and 'goalies' in boxscore['homeTeam']:
            for goalie_id in boxscore['homeTeam']['goalies']:
                # Find goalie in player stats
                if 'playerByGameStats' in boxscore:
                    home_stats = boxscore['playerByGameStats'].get('homeTeam', {}).get('goalies', [])
                    for goalie_stats in home_stats:
                        if goalie_stats.get('playerId') == goalie_id:
                            goalies.append({
                                'game_id': game_id,
                                'player_id': goalie_id,
                                'team': boxscore['homeTeam'].get('abbrev'),
                                'is_home': True,
                                'saves': goalie_stats.get('saveShotsAgainst', '').split('/')[0] if '/' in str(goalie_stats.get('saveShotsAgainst', '')) else None,
                                'shots_against': goalie_stats.get('saveShotsAgainst', '').split('/')[1] if '/' in str(goalie_stats.get('saveShotsAgainst', '')) else None,
                                'save_pct': goalie_stats.get('savePctg'),
                                'goals_against': goalie_stats.get('goalsAgainst'),
                                'toi': goalie_stats.get('toi'),
                                'decision': goalie_stats.get('decision')
                            })

        # Extract away team goalies
        if 'awayTeam' in boxscore and 'goalies' in boxscore['awayTeam']:
            for goalie_id in boxscore['awayTeam']['goalies']:
                if 'playerByGameStats' in boxscore:
                    away_stats = boxscore['playerByGameStats'].get('awayTeam', {}).get('goalies', [])
                    for goalie_stats in away_stats:
                        if goalie_stats.get('playerId') == goalie_id:
                            goalies.append({
                                'game_id': game_id,
                                'player_id': goalie_id,
                                'team': boxscore['awayTeam'].get('abbrev'),
                                'is_home': False,
                                'saves': goalie_stats.get('saveShotsAgainst', '').split('/')[0] if '/' in str(goalie_stats.get('saveShotsAgainst', '')) else None,
                                'shots_against': goalie_stats.get('saveShotsAgainst', '').split('/')[1] if '/' in str(goalie_stats.get('saveShotsAgainst', '')) else None,
                                'save_pct': goalie_stats.get('savePctg'),
                                'goals_against': goalie_stats.get('goalsAgainst'),
                                'toi': goalie_stats.get('toi'),
                                'decision': goalie_stats.get('decision')
                            })

        return goalies

    def extract_shots_data(self, game_id: int) -> List[Dict[str, Any]]:
        """
        Extract shot data from play-by-play (for xG calculation, shot quality)

        Args:
            game_id: Game ID

        Returns:
            List of shot event dictionaries
        """
        file_path = self.pbp_dir / f"{game_id}.json"

        if not file_path.exists():
            self.logger.warning(f"Play-by-play not found: {game_id}")
            return []

        with open(file_path, 'r') as f:
            pbp_data = json.load(f)

        shots = []

        if 'plays' in pbp_data:
            for play in pbp_data['plays']:
                event_type = play.get('typeDescKey')

                # Shot, goal, missed-shot, blocked-shot
                if event_type in ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']:
                    shot_data = {
                        'game_id': game_id,
                        'event_id': play.get('eventId'),
                        'period': play.get('periodDescriptor', {}).get('number'),
                        'time': play.get('timeInPeriod'),
                        'event_type': event_type,
                        'team': play.get('details', {}).get('eventOwnerTeamId'),
                        'shooter_id': play.get('details', {}).get('shootingPlayerId'),
                        'goalie_id': play.get('details', {}).get('goalieInNetId'),
                        'shot_type': play.get('details', {}).get('shotType'),
                        'x_coord': play.get('details', {}).get('xCoord'),
                        'y_coord': play.get('details', {}).get('yCoord'),
                        'zone_code': play.get('details', {}).get('zoneCode'),
                        'situation_code': play.get('situationCode')
                    }
                    shots.append(shot_data)

        return shots
