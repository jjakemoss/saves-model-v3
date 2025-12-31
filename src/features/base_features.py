"""Extract base features from game data"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseFeatureExtractor:
    """
    Extract base goalie performance features from boxscore data

    Features extracted:
    - saves: Total saves in game
    - shots_against: Total shots faced
    - save_percentage: Saves / shots_against
    - goals_against: Goals allowed
    - time_on_ice_seconds: Total TOI in seconds
    - game_result: W/L/OT (from team perspective)
    - is_starter: Whether goalie started vs came in relief
    - even_strength_saves: 5v5 saves
    - power_play_saves: Saves while on penalty kill
    - short_handed_saves: Saves while opponent on PP
    """

    def extract_goalie_game_features(self, boxscore_data: Dict[str, Any], goalie_id: int) -> Optional[Dict[str, Any]]:
        """
        Extract base features for a specific goalie from a boxscore

        Args:
            boxscore_data: Parsed boxscore JSON
            goalie_id: NHL player ID for the goalie

        Returns:
            Dictionary of base features, or None if goalie not found
        """
        try:
            # Find goalie in either home or away team
            goalie_stats = self._find_goalie_stats(boxscore_data, goalie_id)
            if not goalie_stats:
                logger.warning(f"Goalie {goalie_id} not found in boxscore")
                return None

            # Extract basic counting stats
            features = {
                'goalie_id': goalie_id,
                'game_id': boxscore_data.get('id'),
                'game_date': boxscore_data.get('gameDate'),
                'season': boxscore_data.get('season'),

                # Core save stats
                'saves': goalie_stats.get('saves', 0),
                'shots_against': goalie_stats.get('shotsAgainst', 0),
                'goals_against': goalie_stats.get('goalsAgainst', 0),
                'save_percentage': goalie_stats.get('savePctg', 0.0),

                # Time on ice
                'toi': goalie_stats.get('toi', '00:00'),
                'toi_seconds': self._toi_to_seconds(goalie_stats.get('toi', '00:00')),

                # Game situation stats (parse "saves/shots" format strings)
                'even_strength_saves': self._parse_situation_stat(goalie_stats.get('evenStrengthShotsAgainst', '0/0'), 'saves'),
                'even_strength_shots_against': self._parse_situation_stat(goalie_stats.get('evenStrengthShotsAgainst', '0/0'), 'shots'),
                'power_play_saves': self._parse_situation_stat(goalie_stats.get('powerPlayShotsAgainst', '0/0'), 'saves'),
                'power_play_shots_against': self._parse_situation_stat(goalie_stats.get('powerPlayShotsAgainst', '0/0'), 'shots'),
                'short_handed_saves': self._parse_situation_stat(goalie_stats.get('shorthandedShotsAgainst', '0/0'), 'saves'),
                'short_handed_shots_against': self._parse_situation_stat(goalie_stats.get('shorthandedShotsAgainst', '0/0'), 'shots'),

                # Game outcome
                'decision': goalie_stats.get('decision', ''),  # W/L/OT
                'pim': goalie_stats.get('pim', 0),
            }

            # Determine if starter (heuristic: >50% of regulation time = 40+ minutes)
            features['is_starter'] = features['toi_seconds'] > 2400

            # Calculate situational save percentages
            features['even_strength_save_pct'] = self._safe_divide(
                features['even_strength_saves'],
                features['even_strength_shots_against']
            )
            features['power_play_save_pct'] = self._safe_divide(
                features['power_play_saves'],
                features['power_play_shots_against']
            )
            features['short_handed_save_pct'] = self._safe_divide(
                features['short_handed_saves'],
                features['short_handed_shots_against']
            )

            return features

        except Exception as e:
            logger.error(f"Error extracting features for goalie {goalie_id}: {e}")
            return None

    def _find_goalie_stats(self, boxscore_data: Dict[str, Any], goalie_id: int) -> Optional[Dict[str, Any]]:
        """Find goalie stats in boxscore (checking both home and away teams)"""
        try:
            # Check home team
            home_goalies = boxscore_data.get('playerByGameStats', {}).get('homeTeam', {}).get('goalies', [])
            for goalie in home_goalies:
                if goalie.get('playerId') == goalie_id:
                    return goalie

            # Check away team
            away_goalies = boxscore_data.get('playerByGameStats', {}).get('awayTeam', {}).get('goalies', [])
            for goalie in away_goalies:
                if goalie.get('playerId') == goalie_id:
                    return goalie

            return None

        except Exception as e:
            logger.error(f"Error finding goalie stats: {e}")
            return None

    def _toi_to_seconds(self, toi_str: str) -> int:
        """Convert TOI string (MM:SS) to total seconds"""
        try:
            if not toi_str or toi_str == '00:00':
                return 0
            parts = toi_str.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            return 0
        except:
            return 0

    def _parse_situation_stat(self, stat_str: str, stat_type: str) -> int:
        """
        Parse situational stat strings like "18/18" (saves/shots format)

        Args:
            stat_str: String in "saves/shots" format (e.g., "18/20")
            stat_type: Either 'saves' or 'shots'

        Returns:
            Parsed integer value
        """
        try:
            if '/' not in str(stat_str):
                return 0
            saves_str, shots_str = str(stat_str).split('/')
            if stat_type == 'saves':
                return int(saves_str)
            else:  # shots
                return int(shots_str)
        except:
            return 0

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safely divide, returning 0 if denominator is 0"""
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def extract_team_game_features(self, boxscore_data: Dict[str, Any], team_abbrev: str) -> Dict[str, Any]:
        """
        Extract team-level features from a game

        Args:
            boxscore_data: Parsed boxscore JSON
            team_abbrev: Team abbreviation (e.g., 'MIN', 'TOR')

        Returns:
            Dictionary of team features
        """
        try:
            # Determine if home or away
            is_home = boxscore_data.get('homeTeam', {}).get('abbrev') == team_abbrev
            team_key = 'homeTeam' if is_home else 'awayTeam'
            opp_key = 'awayTeam' if is_home else 'homeTeam'

            team_data = boxscore_data.get(team_key, {})
            opp_data = boxscore_data.get(opp_key, {})

            features = {
                'game_id': boxscore_data.get('id'),
                'game_date': boxscore_data.get('gameDate'),
                'team_abbrev': team_abbrev,
                'is_home': is_home,

                # Team performance
                'team_goals': team_data.get('score', 0),
                'team_shots': team_data.get('sog', 0),
                'team_hits': team_data.get('hits', 0),
                'team_blocked_shots': team_data.get('blocks', 0),
                'team_pim': team_data.get('pim', 0),
                'team_powerplay_goals': team_data.get('powerPlayGoals', 0),
                'team_powerplay_opportunities': team_data.get('powerPlayOpportunities', 0),
                'team_faceoff_win_pct': team_data.get('faceoffWinningPctg', 0.0),

                # Opponent performance (useful for defensive metrics)
                'opp_goals': opp_data.get('score', 0),
                'opp_shots': opp_data.get('sog', 0),
                'opp_powerplay_goals': opp_data.get('powerPlayGoals', 0),
                'opp_powerplay_opportunities': opp_data.get('powerPlayOpportunities', 0),
            }

            # Calculate derived metrics
            features['team_shooting_pct'] = self._safe_divide(
                features['team_goals'],
                features['team_shots']
            )
            features['team_powerplay_pct'] = self._safe_divide(
                features['team_powerplay_goals'],
                features['team_powerplay_opportunities']
            )

            return features

        except Exception as e:
            logger.error(f"Error extracting team features: {e}")
            return {}

    def load_boxscore(self, boxscore_path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse a boxscore JSON file"""
        try:
            with open(boxscore_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading boxscore {boxscore_path}: {e}")
            return None
