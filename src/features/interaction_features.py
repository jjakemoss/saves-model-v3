"""Calculate interaction features

Interaction features capture non-linear relationships between variables.
These are critical for hockey analytics because the impact of one factor
often depends on the value of another.

Examples:
- Elite defense × Elite offense ≠ sum of individual effects
- Good goalie + Good defense = synergistic shot suppression
- Hot form × Well rested = amplified performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class InteractionFeatureCalculator:
    """
    Calculate multiplicative interaction features

    Key interactions:
    1. Defense × Offense: Team defense quality × Opponent offense quality
    2. Form × Rest: Recent performance × Days of rest
    3. Quality × Volume: Save % × Shots against tendency
    4. Danger × Control: High danger rate × Rebound control
    """

    def calculate_defense_offense_interactions(
        self,
        team_features: Dict[str, float],
        opponent_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate interactions between team defense and opponent offense

        Key insight: When a strong defense faces a strong offense,
        the result is not simply additive

        Args:
            team_features: Team defensive metrics
            opponent_features: Opponent offensive metrics

        Returns:
            Dictionary of interaction features
        """
        interactions = {}

        # Shot volume interaction
        # Team shot suppression × Opponent shot generation
        team_shot_suppression = team_features.get('team_shots_against_ewa_5', 0)
        opp_shot_generation = opponent_features.get('opponent_shots_per_game_ewa_5', 0)

        if team_shot_suppression > 0 and opp_shot_generation > 0:
            interactions['defense_offense_shot_volume_interaction'] = (
                (35 - team_shot_suppression) * opp_shot_generation / 1000
            )
        else:
            interactions['defense_offense_shot_volume_interaction'] = 0.0

        # Expected goals interaction
        # Team xGA suppression × Opponent xGF generation
        team_xga = team_features.get('team_expected_goals_against_ewa_5', 0)
        opp_xgf = opponent_features.get('opponent_expected_goals_for_ewa_5', 0)

        if team_xga > 0 and opp_xgf > 0:
            interactions['defense_offense_xg_interaction'] = team_xga * opp_xgf
        else:
            interactions['defense_offense_xg_interaction'] = 0.0

        # High danger interaction
        # Team HDC allowed × Opponent HDC generated
        team_hdc = team_features.get('team_high_danger_chances_against_ewa_5', 0)
        opp_hdc = opponent_features.get('opponent_high_danger_chances_ewa_5', 0)

        if team_hdc > 0 and opp_hdc > 0:
            interactions['defense_offense_hdc_interaction'] = team_hdc * opp_hdc / 100
        else:
            interactions['defense_offense_hdc_interaction'] = 0.0

        # Matchup strength ratio
        # Measures overall matchup competitiveness
        if opp_shot_generation > 0:
            interactions['defense_offense_strength_ratio'] = team_shot_suppression / opp_shot_generation
        else:
            interactions['defense_offense_strength_ratio'] = 1.0

        return interactions

    def calculate_form_rest_interactions(
        self,
        goalie_features: Dict[str, float],
        rest_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate interactions between recent form and rest

        Key insight: A hot goalie with rest performs even better (amplified)
        A cold goalie with fatigue performs even worse (compounded)

        Args:
            goalie_features: Recent goalie performance
            rest_features: Rest and fatigue metrics

        Returns:
            Dictionary of interaction features
        """
        interactions = {}

        # Recent save % × Days rest
        recent_save_pct = goalie_features.get('save_percentage_ewa_3', 0)
        days_rest = rest_features.get('days_since_last_start', 0)

        # Normalize days rest (0-7+ days → 0-1 scale)
        rest_factor = min(days_rest / 7.0, 1.0)

        interactions['form_rest_interaction'] = recent_save_pct * rest_factor

        # Recent saves × Rest (volume prediction)
        recent_saves = goalie_features.get('saves_ewa_5', 0)
        interactions['saves_rest_interaction'] = recent_saves * rest_factor

        # Fatigue penalty interaction
        # Back-to-back games significantly reduce performance
        is_b2b = rest_features.get('is_back_to_back', 0)
        is_b2b_second = rest_features.get('is_back_to_back_second_game', 0)

        # Heavy penalty for second game of back-to-back
        fatigue_penalty = 1.0 - (0.05 * is_b2b + 0.10 * is_b2b_second)
        interactions['form_fatigue_penalty'] = recent_saves * fatigue_penalty

        # Consecutive starts × Recent workload
        consecutive = rest_features.get('consecutive_starts_streak', 0)
        starts_last_7 = rest_features.get('starts_in_last_7_days', 0)

        # Workload factor (more starts = more fatigue)
        workload_factor = consecutive * starts_last_7 / 10.0
        interactions['workload_form_interaction'] = recent_save_pct * (1 - min(workload_factor, 0.3))

        return interactions

    def calculate_quality_volume_interactions(
        self,
        goalie_features: Dict[str, float],
        team_features: Dict[str, float],
        opponent_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate interactions between shot quality and shot volume

        Key insight: High save % facing high shot volume = more total saves
        Low save % facing low shot volume = fewer saves

        Args:
            goalie_features: Goalie quality metrics
            team_features: Team defensive metrics
            opponent_features: Opponent offensive metrics

        Returns:
            Dictionary of interaction features
        """
        interactions = {}

        # Goalie save % × Expected shot volume
        goalie_save_pct = goalie_features.get('save_percentage_ewa_5', 0)
        expected_shots = opponent_features.get('opponent_shots_per_game_ewa_5', 0)

        interactions['quality_volume_save_prediction'] = goalie_save_pct * expected_shots

        # Goalie high danger save % × Opponent high danger chances
        goalie_hd_save_pct = goalie_features.get('high_danger_save_pct_last_10', 0)
        opp_hd_chances = opponent_features.get('opponent_high_danger_chances_ewa_5', 0)

        interactions['hd_quality_volume_interaction'] = goalie_hd_save_pct * opp_hd_chances

        # Team defense quality × Opponent shooting skill
        team_defense = team_features.get('team_shot_suppression_index', 1.0)
        opp_shooting_pct = opponent_features.get('opponent_shooting_pct_overall_last_10', 0)

        # Good defense vs high shooting % = fewer goals, more saves
        interactions['defense_shooting_skill_interaction'] = team_defense * (1 - opp_shooting_pct)

        return interactions

    def calculate_danger_control_interactions(
        self,
        goalie_features: Dict[str, float],
        shot_quality_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate interactions between shot danger and rebound control

        Key insight: High rebound rate in high danger areas = more saves
        (but also higher risk of goals)

        Args:
            goalie_features: Goalie performance metrics
            shot_quality_features: Shot quality and rebound metrics

        Returns:
            Dictionary of interaction features
        """
        interactions = {}

        # Rebound rate × High danger shots faced
        rebound_rate = shot_quality_features.get('rebound_rate', 0)
        hd_shots = shot_quality_features.get('high_danger_shots_against', 0)

        # More rebounds in high danger areas = more save opportunities
        # (but also more dangerous)
        interactions['rebound_hd_volume_interaction'] = rebound_rate * hd_shots

        # Dangerous rebound % × High danger save %
        dangerous_rebound_pct = shot_quality_features.get('dangerous_rebound_pct', 0)
        hd_save_pct = goalie_features.get('high_danger_save_pct_last_10', 0)

        # Good high danger save % mitigates dangerous rebounds
        interactions['rebound_control_quality_interaction'] = hd_save_pct * (1 - dangerous_rebound_pct)

        return interactions

    def calculate_location_advantage_interactions(
        self,
        goalie_features: Dict[str, float],
        matchup_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate interactions between home/away and recent performance

        Key insight: Home advantage amplifies recent good form

        Args:
            goalie_features: Goalie performance
            matchup_features: Home/away and matchup context

        Returns:
            Dictionary of interaction features
        """
        interactions = {}

        # Home advantage × Recent form
        is_home = matchup_features.get('is_home', 0)
        home_advantage = matchup_features.get('goalie_home_advantage', 0)
        recent_save_pct = goalie_features.get('save_percentage_ewa_3', 0)

        # Home advantage amplifies recent form
        interactions['home_form_interaction'] = is_home * home_advantage * recent_save_pct

        # Travel fatigue × Away game
        travel_miles = matchup_features.get('travel_miles_since_last_game', 0)
        games_on_trip = matchup_features.get('games_on_current_road_trip', 0)

        # Long road trip + travel = compounded fatigue
        travel_fatigue = (travel_miles / 1000) * games_on_trip
        interactions['travel_fatigue_interaction'] = travel_fatigue

        return interactions

    def calculate_volatility_line_interactions(
        self,
        goalie_features: Dict[str, float],
        matchup_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate interactions between volatility and betting line

        Key insight: High volatility + line close to mean = unpredictable
        Low volatility + line far from mean = strong signal

        Args:
            goalie_features: Goalie performance and volatility
            matchup_features: Betting line context

        Returns:
            Dictionary of interaction features
        """
        interactions = {}

        # Volatility × Line distance from average
        volatility = matchup_features.get('saves_volatility_last_10', 0)
        line_vs_avg = matchup_features.get('line_vs_goalie_season_avg', 0)

        # High volatility when line is far from avg = uncertain
        interactions['volatility_line_uncertainty'] = volatility * abs(line_vs_avg)

        # Overline frequency × Recent form
        overline_freq = matchup_features.get('overline_frequency_last_10', 0)
        recent_saves = goalie_features.get('saves_ewa_3', 0)

        # Consistent over performer with high recent saves
        interactions['overline_consistency_interaction'] = overline_freq * recent_saves

        return interactions


def calculate_all_interaction_features(
    goalie_features: Dict[str, float],
    team_features: Dict[str, float],
    opponent_features: Dict[str, float],
    matchup_features: Dict[str, float],
    shot_quality_features: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate all interaction features

    Args:
        goalie_features: Goalie performance metrics
        team_features: Team defensive metrics
        opponent_features: Opponent offensive metrics
        matchup_features: Matchup and contextual features
        shot_quality_features: Shot quality and rebound features

    Returns:
        Dictionary with all interaction features
    """
    calculator = InteractionFeatureCalculator()

    all_interactions = {}

    # Defense × Offense
    all_interactions.update(
        calculator.calculate_defense_offense_interactions(team_features, opponent_features)
    )

    # Form × Rest
    rest_features = {
        'days_since_last_start': matchup_features.get('days_since_last_start', 0),
        'is_back_to_back': matchup_features.get('is_back_to_back', 0),
        'is_back_to_back_second_game': matchup_features.get('is_back_to_back_second_game', 0),
        'consecutive_starts_streak': matchup_features.get('consecutive_starts_streak', 0),
        'starts_in_last_7_days': matchup_features.get('starts_in_last_7_days', 0),
    }
    all_interactions.update(
        calculator.calculate_form_rest_interactions(goalie_features, rest_features)
    )

    # Quality × Volume
    all_interactions.update(
        calculator.calculate_quality_volume_interactions(
            goalie_features, team_features, opponent_features
        )
    )

    # Danger × Control
    all_interactions.update(
        calculator.calculate_danger_control_interactions(goalie_features, shot_quality_features)
    )

    # Location × Advantage
    all_interactions.update(
        calculator.calculate_location_advantage_interactions(goalie_features, matchup_features)
    )

    # Volatility × Line
    all_interactions.update(
        calculator.calculate_volatility_line_interactions(goalie_features, matchup_features)
    )

    return all_interactions
