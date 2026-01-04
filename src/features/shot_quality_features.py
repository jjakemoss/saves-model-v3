"""Calculate shot quality and Expected Goals (xG) features

Expected Goals is a more predictive metric than Corsi/Fenwick for save totals.
The key relationship: Saves ≈ Shots Against - Expected Goals Against (xGA)

This module extracts shot location data from play-by-play and calculates:
1. Expected Goals (xG) based on shot distance, angle, type
2. Shot danger classification (high/mid/low)
3. Rebound control metrics
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ShotQualityFeatureExtractor:
    """
    Extract shot quality and Expected Goals features from play-by-play data

    Shot danger zones (NHL standard):
    - High danger: Slot area, < 15 feet from net, good angle
    - Mid danger: Faceoff circles, 15-30 feet, moderate angle
    - Low danger: Point shots, > 30 feet or extreme angles
    """

    # Shot danger thresholds
    HIGH_DANGER_DISTANCE = 15  # feet
    MID_DANGER_DISTANCE = 30   # feet
    HIGH_DANGER_ANGLE = 45     # degrees from center

    # Expected Goals model coefficients (simplified)
    # Based on distance and shot type
    XG_BASE_RATES = {
        'wrist': 0.06,
        'slap': 0.04,
        'snap': 0.07,
        'tip': 0.12,
        'deflection': 0.12,
        'backhand': 0.05,
        'wrap-around': 0.08,
    }

    def __init__(self):
        """Initialize shot quality feature extractor"""
        pass

    def extract_shots_from_pbp(self, pbp_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all shots from play-by-play data

        Args:
            pbp_data: Play-by-play JSON data

        Returns:
            List of shot events with coordinates and metadata
        """
        shots = []

        try:
            plays = pbp_data.get('plays', [])

            for play in plays:
                event_type = play.get('typeDescKey', '')

                # Include shots and goals
                if event_type in ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']:
                    shot_info = {
                        'event_type': event_type,
                        'period': play.get('periodDescriptor', {}).get('number', 0),
                        'time_in_period': play.get('timeInPeriod', '00:00'),
                        'team_abbrev': play.get('details', {}).get('eventOwnerTeamId', ''),
                        'shot_type': play.get('details', {}).get('shotType', '').lower(),
                        'x_coord': play.get('details', {}).get('xCoord', 0),
                        'y_coord': play.get('details', {}).get('yCoord', 0),
                        'is_goal': event_type == 'goal',
                        'goalie_id': play.get('details', {}).get('goalieInNetId', None),
                    }

                    # Calculate distance and angle
                    shot_info['distance'] = self._calculate_distance(
                        shot_info['x_coord'],
                        shot_info['y_coord']
                    )
                    shot_info['angle'] = self._calculate_angle(
                        shot_info['x_coord'],
                        shot_info['y_coord']
                    )

                    # Classify danger level
                    shot_info['danger_level'] = self._classify_shot_danger(
                        shot_info['distance'],
                        shot_info['angle']
                    )

                    # Calculate xG
                    shot_info['xg'] = self._calculate_shot_xg(
                        shot_info['distance'],
                        shot_info['shot_type'],
                        shot_info['danger_level']
                    )

                    shots.append(shot_info)

        except Exception as e:
            logger.error(f"Error extracting shots from play-by-play: {e}")

        return shots

    def _calculate_distance(self, x: float, y: float) -> float:
        """
        Calculate shot distance from net in feet

        NHL rink: Net is at (89, 0) for one end, (-89, 0) for other
        We use absolute value to handle both ends
        """
        # Assume net is at x=89 (or -89), y=0
        # Distance = sqrt((x - 89)^2 + y^2)
        net_x = 89 if x > 0 else -89
        distance = np.sqrt((x - net_x)**2 + y**2)
        return distance

    def _calculate_angle(self, x: float, y: float) -> float:
        """
        Calculate shot angle from center in degrees

        Angle = 0 degrees means straight on
        Angle = 90 degrees means from the boards
        """
        net_x = 89 if x > 0 else -89

        # Handle edge case where shot is from behind net
        if abs(x) > abs(net_x):
            return 90.0

        # Calculate angle using arctangent
        angle = np.abs(np.degrees(np.arctan2(y, abs(net_x - x))))
        return angle

    def _classify_shot_danger(self, distance: float, angle: float) -> str:
        """
        Classify shot as high/mid/low danger

        Args:
            distance: Distance from net in feet
            angle: Angle from center in degrees

        Returns:
            'high', 'mid', or 'low'
        """
        if distance <= self.HIGH_DANGER_DISTANCE and angle <= self.HIGH_DANGER_ANGLE:
            return 'high'
        elif distance <= self.MID_DANGER_DISTANCE:
            return 'mid'
        else:
            return 'low'

    def _calculate_shot_xg(self, distance: float, shot_type: str, danger_level: str) -> float:
        """
        Calculate Expected Goals for a single shot

        Simplified model based on distance, shot type, and danger level

        Args:
            distance: Distance from net
            shot_type: Type of shot
            danger_level: high/mid/low

        Returns:
            Expected goal probability (0.0 to 1.0)
        """
        # Base rate from shot type
        base_xg = self.XG_BASE_RATES.get(shot_type, 0.05)

        # Distance multiplier (exponential decay)
        # Closer shots are much more dangerous
        distance_multiplier = np.exp(-distance / 30)

        # Danger level multiplier
        danger_multipliers = {
            'high': 2.5,
            'mid': 1.5,
            'low': 0.7
        }
        danger_mult = danger_multipliers.get(danger_level, 1.0)

        xg = base_xg * distance_multiplier * danger_mult

        # Cap at reasonable maximum
        return min(xg, 0.5)

    def aggregate_goalie_shot_quality(self, shots: List[Dict[str, Any]], goalie_id: int) -> Dict[str, Any]:
        """
        Aggregate shot quality metrics for a specific goalie

        Args:
            shots: List of all shots in the game
            goalie_id: ID of goalie to analyze

        Returns:
            Dictionary of aggregated shot quality features
        """
        # Filter shots faced by this goalie
        goalie_shots = [s for s in shots if s.get('goalie_id') == goalie_id]

        if not goalie_shots:
            return self._empty_shot_quality_features()

        # Count by danger level
        high_danger_shots = [s for s in goalie_shots if s['danger_level'] == 'high']
        mid_danger_shots = [s for s in goalie_shots if s['danger_level'] == 'mid']
        low_danger_shots = [s for s in goalie_shots if s['danger_level'] == 'low']

        # Count goals by danger level
        high_danger_goals = sum(1 for s in high_danger_shots if s['is_goal'])
        mid_danger_goals = sum(1 for s in mid_danger_shots if s['is_goal'])
        low_danger_goals = sum(1 for s in low_danger_shots if s['is_goal'])

        # Calculate save percentages by danger
        features = {
            'total_shots_against': len(goalie_shots),
            'total_xg_against': sum(s['xg'] for s in goalie_shots),

            # High danger
            'high_danger_shots_against': len(high_danger_shots),
            'high_danger_goals_against': high_danger_goals,
            'high_danger_saves': len(high_danger_shots) - high_danger_goals,
            'high_danger_save_pct': self._safe_save_pct(
                len(high_danger_shots) - high_danger_goals,
                len(high_danger_shots)
            ),
            'high_danger_xg_against': sum(s['xg'] for s in high_danger_shots),

            # Mid danger
            'mid_danger_shots_against': len(mid_danger_shots),
            'mid_danger_goals_against': mid_danger_goals,
            'mid_danger_saves': len(mid_danger_shots) - mid_danger_goals,
            'mid_danger_save_pct': self._safe_save_pct(
                len(mid_danger_shots) - mid_danger_goals,
                len(mid_danger_shots)
            ),

            # Low danger
            'low_danger_shots_against': len(low_danger_shots),
            'low_danger_goals_against': low_danger_goals,
            'low_danger_saves': len(low_danger_shots) - low_danger_goals,
            'low_danger_save_pct': self._safe_save_pct(
                len(low_danger_shots) - low_danger_goals,
                len(low_danger_shots)
            ),

            # Average shot distance
            'avg_shot_distance': np.mean([s['distance'] for s in goalie_shots]) if goalie_shots else 0.0,
            'avg_shot_angle': np.mean([s['angle'] for s in goalie_shots]) if goalie_shots else 0.0,
        }

        return features

    def calculate_rebound_metrics(self, shots: List[Dict[str, Any]], goalie_id: int) -> Dict[str, float]:
        """
        Calculate rebound control metrics

        A rebound is defined as a shot within 3 seconds of a previous save
        in the high-danger area

        Args:
            shots: List of shots (sorted by time)
            goalie_id: Goalie to analyze

        Returns:
            Dictionary with rebound metrics
        """
        goalie_shots = [s for s in shots if s.get('goalie_id') == goalie_id]

        if len(goalie_shots) < 2:
            return {
                'rebound_rate': 0.0,
                'dangerous_rebound_pct': 0.0,
                'rebounds_created': 0
            }

        rebounds = 0
        dangerous_rebounds = 0
        total_saves = 0

        for i in range(len(goalie_shots) - 1):
            current_shot = goalie_shots[i]
            next_shot = goalie_shots[i + 1]

            # Only count if current shot was a save
            if not current_shot['is_goal']:
                total_saves += 1

                # Check if next shot is within 3 seconds
                time_diff = self._time_difference_seconds(
                    current_shot['period'],
                    current_shot['time_in_period'],
                    next_shot['period'],
                    next_shot['time_in_period']
                )

                if 0 < time_diff <= 3:
                    rebounds += 1
                    if next_shot['danger_level'] == 'high':
                        dangerous_rebounds += 1

        rebound_rate = rebounds / total_saves if total_saves > 0 else 0.0
        dangerous_pct = dangerous_rebounds / rebounds if rebounds > 0 else 0.0

        return {
            'rebound_rate': rebound_rate,
            'dangerous_rebound_pct': dangerous_pct,
            'rebounds_created': rebounds
        }

    def _time_difference_seconds(self, period1: int, time1: str, period2: int, time2: str) -> int:
        """Calculate time difference between two events in seconds"""
        if period1 != period2:
            return 999  # Different periods, not a rebound

        # Parse MM:SS format
        try:
            m1, s1 = map(int, time1.split(':'))
            m2, s2 = map(int, time2.split(':'))

            time1_sec = m1 * 60 + s1
            time2_sec = m2 * 60 + s2

            return time2_sec - time1_sec
        except:
            return 999

    def _safe_save_pct(self, saves: int, shots: int) -> float:
        """Calculate save percentage, handling division by zero"""
        if shots == 0:
            return 0.0
        return saves / shots

    def _empty_shot_quality_features(self) -> Dict[str, Any]:
        """Return empty feature dict when no shots found"""
        return {
            'total_shots_against': 0,
            'total_xg_against': 0.0,
            'high_danger_shots_against': 0,
            'high_danger_goals_against': 0,
            'high_danger_saves': 0,
            'high_danger_save_pct': 0.0,
            'high_danger_xg_against': 0.0,
            'mid_danger_shots_against': 0,
            'mid_danger_goals_against': 0,
            'mid_danger_saves': 0,
            'mid_danger_save_pct': 0.0,
            'low_danger_shots_against': 0,
            'low_danger_goals_against': 0,
            'low_danger_saves': 0,
            'low_danger_save_pct': 0.0,
            'avg_shot_distance': 0.0,
            'avg_shot_angle': 0.0,
        }

    def load_play_by_play(self, pbp_path: Path) -> Optional[Dict[str, Any]]:
        """Load play-by-play JSON file"""
        try:
            with open(pbp_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading play-by-play {pbp_path}: {e}")
            return None
