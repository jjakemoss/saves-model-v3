"""Real-time feature collection for live predictions

Fetches recent goalie/team stats from NHL API and generates features for prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from data.api_client import NHLAPIClient

logger = logging.getLogger(__name__)


class RealtimeFeatureCollector:
    """
    Collect real-time features for goalies and teams

    Generates the same 36 features used during training:
    - Rolling averages from recent games (3, 5, 10 game windows)
    - Opponent stats
    - Team stats
    - Contextual features (home/away, starter status)
    """

    def __init__(self, api_client: Optional[NHLAPIClient] = None):
        """
        Initialize feature collector

        Args:
            api_client: NHL API client (creates new one if not provided)
        """
        self.api_client = api_client or NHLAPIClient()

    def collect_goalie_features(
        self,
        goalie_name: str,
        team_abbrev: str,
        opponent_abbrev: str,
        is_home: bool,
        season: str = "20242025",
        num_recent_games: int = 15
    ) -> Dict[str, float]:
        """
        Collect all features for a goalie

        Args:
            goalie_name: Goalie's name
            team_abbrev: Team abbreviation (e.g., "MIN")
            opponent_abbrev: Opponent abbreviation (e.g., "SJ")
            is_home: Whether goalie is playing at home
            season: Season string (e.g., "20242025")
            num_recent_games: Number of recent games to fetch

        Returns:
            Dictionary of feature name -> value
        """
        logger.info(f"Collecting features for {goalie_name} ({team_abbrev})")

        # Get goalie ID by name
        goalie_id = self._find_goalie_id(goalie_name, team_abbrev, season)

        if not goalie_id:
            logger.warning(f"Could not find goalie ID for {goalie_name}")
            return self._get_default_features(is_home)

        # Fetch recent game logs
        game_logs = self._fetch_goalie_game_logs(goalie_id, season, num_recent_games)

        if not game_logs:
            logger.warning(f"No game logs found for {goalie_name}")
            return self._get_default_features(is_home)

        # Calculate rolling averages from game logs
        rolling_features = self._calculate_rolling_features(game_logs)

        # Get opponent stats
        opponent_features = self._get_opponent_features(opponent_abbrev, season)

        # Get team stats
        team_features = self._get_team_features(team_abbrev, season)

        # Combine all features
        features = {
            **rolling_features,
            **opponent_features,
            **team_features,
            'is_home': float(is_home),
            'is_starter': 1.0  # Assume starter for now
        }

        logger.info(f"Collected {len(features)} features for {goalie_name}")

        return features

    def _find_goalie_id(
        self,
        goalie_name: str,
        team_abbrev: str,
        season: str
    ) -> Optional[int]:
        """
        Find goalie ID by name

        For now, returns None (placeholder).
        Full implementation would search roster or use player search API.

        Args:
            goalie_name: Goalie's name
            team_abbrev: Team abbreviation
            season: Season string

        Returns:
            Goalie player ID or None
        """
        # Placeholder - would implement roster search here
        logger.warning("Goalie ID lookup not yet implemented")
        return None

    def _fetch_goalie_game_logs(
        self,
        goalie_id: int,
        season: str,
        num_games: int
    ) -> List[Dict]:
        """
        Fetch recent game logs for goalie

        Args:
            goalie_id: Goalie's player ID
            season: Season string
            num_games: Number of recent games to fetch

        Returns:
            List of game log dictionaries
        """
        try:
            endpoint = f"/v1/player/{goalie_id}/game-log/{season}/2"
            response = self.api_client.get(endpoint)

            if not response or 'gameLog' not in response:
                return []

            # Get most recent games
            game_logs = response['gameLog'][:num_games]

            return game_logs

        except Exception as e:
            logger.error(f"Error fetching game logs: {e}")
            return []

    def _calculate_rolling_features(
        self,
        game_logs: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate rolling average features from game logs

        Args:
            game_logs: List of recent game log dictionaries

        Returns:
            Dictionary of rolling feature values
        """
        if not game_logs:
            return {}

        # Extract stats from game logs
        saves_list = []
        shots_list = []
        goals_list = []
        sv_pct_list = []
        es_sv_pct_list = []
        pp_sv_pct_list = []

        for game in game_logs:
            goalie_stats = game.get('goalie', {})

            saves = goalie_stats.get('saves', 0)
            shots_against = goalie_stats.get('shotsAgainst', 0)
            goals_against = goalie_stats.get('goalsAgainst', 0)

            saves_list.append(saves)
            shots_list.append(shots_against)
            goals_list.append(goals_against)

            # Save percentage
            if shots_against > 0:
                sv_pct = saves / shots_against
            else:
                sv_pct = 0.900
            sv_pct_list.append(sv_pct)

            # Even strength save %
            es_shots = goalie_stats.get('evenStrengthShotsAgainst', 0)
            es_saves = goalie_stats.get('evenStrengthSaves', es_shots)
            if es_shots > 0:
                es_sv_pct = es_saves / es_shots
            else:
                es_sv_pct = 0.900
            es_sv_pct_list.append(es_sv_pct)

            # Power play save %
            pp_shots = goalie_stats.get('powerPlayShotsAgainst', 0)
            pp_saves = goalie_stats.get('powerPlaySaves', pp_shots)
            if pp_shots > 0:
                pp_sv_pct = pp_saves / pp_shots
            else:
                pp_sv_pct = 0.850
            pp_sv_pct_list.append(pp_sv_pct)

        # Calculate rolling averages for windows [3, 5, 10]
        features = {}

        # Saves rolling averages
        features['saves_rolling_3'] = self._rolling_avg(saves_list, 3)
        features['saves_rolling_5'] = self._rolling_avg(saves_list, 5)
        features['saves_rolling_10'] = self._rolling_avg(saves_list, 10)

        # Save percentage rolling averages
        features['save_percentage_rolling_3'] = self._rolling_avg(sv_pct_list, 3)
        features['save_percentage_rolling_5'] = self._rolling_avg(sv_pct_list, 5)
        features['save_percentage_rolling_10'] = self._rolling_avg(sv_pct_list, 10)

        # Shots against rolling averages
        features['shots_against_rolling_3'] = self._rolling_avg(shots_list, 3)
        features['shots_against_rolling_5'] = self._rolling_avg(shots_list, 5)
        features['shots_against_rolling_10'] = self._rolling_avg(shots_list, 10)

        # Goals against rolling averages
        features['goals_against_rolling_3'] = self._rolling_avg(goals_list, 3)
        features['goals_against_rolling_5'] = self._rolling_avg(goals_list, 5)
        features['goals_against_rolling_10'] = self._rolling_avg(goals_list, 10)

        # Even strength save % rolling averages
        features['even_strength_save_pct_rolling_3'] = self._rolling_avg(es_sv_pct_list, 3)
        features['even_strength_save_pct_rolling_5'] = self._rolling_avg(es_sv_pct_list, 5)
        features['even_strength_save_pct_rolling_10'] = self._rolling_avg(es_sv_pct_list, 10)

        # Power play save % rolling averages
        features['power_play_save_pct_rolling_3'] = self._rolling_avg(pp_sv_pct_list, 3)
        features['power_play_save_pct_rolling_5'] = self._rolling_avg(pp_sv_pct_list, 5)
        features['power_play_save_pct_rolling_10'] = self._rolling_avg(pp_sv_pct_list, 10)

        # Trend (difference between last game and 10-game average)
        if len(saves_list) > 0:
            recent_save = saves_list[0]
            avg_saves_10 = features['saves_rolling_10']
            features['saves_trend_10'] = recent_save - avg_saves_10
        else:
            features['saves_trend_10'] = 0.0

        return features

    def _rolling_avg(self, values: List[float], window: int) -> float:
        """Calculate rolling average"""
        if not values:
            return 0.0

        window_values = values[:window]

        if len(window_values) == 0:
            return 0.0

        return sum(window_values) / len(window_values)

    def _get_opponent_features(
        self,
        opponent_abbrev: str,
        season: str
    ) -> Dict[str, float]:
        """
        Get opponent team stats

        Args:
            opponent_abbrev: Opponent team abbreviation
            season: Season string

        Returns:
            Dictionary of opponent features
        """
        try:
            endpoint = f"/v1/club-stats/{opponent_abbrev}/{season}/2"
            response = self.api_client.get(endpoint)

            if not response:
                return self._get_default_opponent_features()

            # Extract relevant stats
            features = {}

            # Shots per game
            features['opp_shots'] = response.get('shotsForPerGame', 32.0)

            # Goals per game
            features['opp_goals'] = response.get('goalsForPerGame', 3.0)

            # Power play stats
            features['opp_powerplay_goals'] = response.get('powerPlayGoalsFor', 0) / max(response.get('gamesPlayed', 1), 1)
            features['opp_powerplay_opportunities'] = response.get('powerPlayOpportunities', 0) / max(response.get('gamesPlayed', 1), 1)

            return features

        except Exception as e:
            logger.error(f"Error fetching opponent stats: {e}")
            return self._get_default_opponent_features()

    def _get_team_features(
        self,
        team_abbrev: str,
        season: str
    ) -> Dict[str, float]:
        """
        Get team stats

        Args:
            team_abbrev: Team abbreviation
            season: Season string

        Returns:
            Dictionary of team features
        """
        try:
            endpoint = f"/v1/club-stats/{team_abbrev}/{season}/2"
            response = self.api_client.get(endpoint)

            if not response:
                return self._get_default_team_features()

            # Extract relevant stats
            features = {}

            features['team_shots'] = response.get('shotsForPerGame', 28.0)
            features['team_goals'] = response.get('goalsForPerGame', 2.5)
            features['team_shooting_pct'] = response.get('shootingPctg', 0.095)
            features['team_powerplay_pct'] = response.get('powerPlayPctg', 0.20)
            features['team_powerplay_goals'] = response.get('powerPlayGoalsFor', 0) / max(response.get('gamesPlayed', 1), 1)
            features['team_powerplay_opportunities'] = response.get('powerPlayOpportunities', 0) / max(response.get('gamesPlayed', 1), 1)
            features['team_faceoff_win_pct'] = response.get('faceoffWinPctg', 0.50)
            features['team_blocked_shots'] = response.get('blockedShots', 0) / max(response.get('gamesPlayed', 1), 1)
            features['team_hits'] = response.get('hits', 0) / max(response.get('gamesPlayed', 1), 1)
            features['team_pim'] = response.get('penaltyMinutes', 0) / max(response.get('gamesPlayed', 1), 1)
            features['pim'] = features['team_pim']  # Duplicate for compatibility

            return features

        except Exception as e:
            logger.error(f"Error fetching team stats: {e}")
            return self._get_default_team_features()

    def _get_default_features(self, is_home: bool) -> Dict[str, float]:
        """Get default feature values when API calls fail"""
        features = {
            **self._get_default_rolling_features(),
            **self._get_default_opponent_features(),
            **self._get_default_team_features(),
            'is_home': float(is_home),
            'is_starter': 1.0
        }
        return features

    def _get_default_rolling_features(self) -> Dict[str, float]:
        """Default rolling feature values"""
        return {
            'saves_rolling_3': 25.0,
            'saves_rolling_5': 26.0,
            'saves_rolling_10': 27.0,
            'save_percentage_rolling_3': 0.910,
            'save_percentage_rolling_5': 0.912,
            'save_percentage_rolling_10': 0.908,
            'shots_against_rolling_3': 28.0,
            'shots_against_rolling_5': 29.0,
            'shots_against_rolling_10': 30.0,
            'goals_against_rolling_3': 2.5,
            'goals_against_rolling_5': 2.6,
            'goals_against_rolling_10': 2.8,
            'even_strength_save_pct_rolling_3': 0.920,
            'even_strength_save_pct_rolling_5': 0.918,
            'even_strength_save_pct_rolling_10': 0.915,
            'power_play_save_pct_rolling_3': 0.850,
            'power_play_save_pct_rolling_5': 0.860,
            'power_play_save_pct_rolling_10': 0.855,
            'saves_trend_10': 0.5
        }

    def _get_default_opponent_features(self) -> Dict[str, float]:
        """Default opponent feature values"""
        return {
            'opp_shots': 32.0,
            'opp_goals': 3.0,
            'opp_powerplay_goals': 1.2,
            'opp_powerplay_opportunities': 3.5
        }

    def _get_default_team_features(self) -> Dict[str, float]:
        """Default team feature values"""
        return {
            'team_shots': 28.0,
            'team_goals': 2.5,
            'team_shooting_pct': 0.095,
            'team_powerplay_pct': 0.20,
            'team_powerplay_goals': 1.0,
            'team_powerplay_opportunities': 3.0,
            'team_faceoff_win_pct': 0.51,
            'team_blocked_shots': 15.0,
            'team_hits': 20.0,
            'team_pim': 8.0,
            'pim': 8.0
        }
