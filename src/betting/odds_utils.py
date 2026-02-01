"""
Odds utilities for sports betting calculations

Handles American odds format conversion, Expected Value calculation,
and payout determination.
"""


def american_to_implied_prob(odds):
    """
    Convert American odds to implied probability.

    Args:
        odds: int - American odds (e.g., -110, +105)
              None is allowed and returns None

    Returns:
        float: Implied probability (0-1)
               None if odds is None

    Examples:
        >>> american_to_implied_prob(-110)
        0.5238...
        >>> american_to_implied_prob(+105)
        0.4878...
        >>> american_to_implied_prob(-150)
        0.6000
        >>> american_to_implied_prob(+200)
        0.3333...
        >>> american_to_implied_prob(None)
        None

    Formula:
        Negative odds (favorite): abs(odds) / (abs(odds) + 100)
        Positive odds (underdog): 100 / (odds + 100)
    """
    if odds is None:
        return None

    if odds < 0:
        # Favorite (e.g., -110)
        return abs(odds) / (abs(odds) + 100)
    else:
        # Underdog (e.g., +105)
        return 100 / (odds + 100)


def american_to_decimal(odds):
    """
    Convert American odds to decimal odds for payout calculation.

    Args:
        odds: int - American odds (e.g., -110, +105)
              None is allowed and returns None

    Returns:
        float: Decimal odds (e.g., 1.909, 2.05)
               None if odds is None

    Examples:
        >>> american_to_decimal(-110)
        1.909...
        >>> american_to_decimal(+105)
        2.05
        >>> american_to_decimal(-150)
        1.666...
        >>> american_to_decimal(+200)
        3.0
        >>> american_to_decimal(None)
        None

    Formula:
        Negative odds (favorite): (100 / abs(odds)) + 1
        Positive odds (underdog): (odds / 100) + 1
    """
    if odds is None:
        return None

    if odds < 0:
        # Favorite (e.g., -110 → 1.909)
        return (100 / abs(odds)) + 1
    else:
        # Underdog (e.g., +105 → 2.05)
        return (odds / 100) + 1


def decimal_to_american(decimal_odds):
    """
    Convert decimal odds to American odds.

    Args:
        decimal_odds: float - Decimal odds (e.g., 1.91, 2.05)
                      None is allowed and returns None

    Returns:
        int: American odds (e.g., -110, +105)
             None if decimal_odds is None

    Examples:
        >>> decimal_to_american(1.91)
        -110
        >>> decimal_to_american(2.05)
        +105
        >>> decimal_to_american(3.0)
        +200
        >>> decimal_to_american(1.667)
        -150
        >>> decimal_to_american(None)
        None
    """
    if decimal_odds is None:
        return None

    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))


def calculate_ev(model_prob, odds):
    """
    Calculate Expected Value as the difference between model probability
    and implied probability from the odds.

    Args:
        model_prob: float (0-1) - Model's estimated probability
        odds: int - American odds for this side
              None is allowed and returns None

    Returns:
        float: Expected Value as decimal (e.g., 0.045 = +4.5% EV)
               None if odds is None

    Examples:
        >>> calculate_ev(0.58, -115)  # Model 58%, odds imply 53.5%
        0.045...  # +4.5% EV
        >>> calculate_ev(0.58, -150)  # Model 58%, odds imply 60%
        -0.02  # -2% EV (bad bet)
        >>> calculate_ev(0.52, +120)  # Model 52%, odds imply 45.5%
        0.065...  # +6.5% EV
        >>> calculate_ev(0.58, None)
        None

    Formula:
        EV = model_prob - implied_prob

    Interpretation:
        Positive EV: Model thinks this outcome is more likely than odds suggest (good bet)
        Negative EV: Model thinks this outcome is less likely than odds suggest (bad bet)
        Zero EV: Model agrees with market (breakeven)
    """
    if odds is None:
        return None

    implied_prob = american_to_implied_prob(odds)
    return model_prob - implied_prob


def validate_american_odds(odds):
    """
    Validate that odds are in valid American format.

    Args:
        odds: any type - Value to validate

    Returns:
        tuple: (is_valid: bool, error_message: str)

    Examples:
        >>> validate_american_odds(-110)
        (True, '')
        >>> validate_american_odds(+150)
        (True, '')
        >>> validate_american_odds(-50)
        (False, 'American odds cannot be between -100 and +100')
        >>> validate_american_odds('text')
        (False, 'Odds must be numeric')
        >>> validate_american_odds(None)
        (True, '')  # None is valid (means no odds provided)

    Validation rules:
        - None is valid (means no odds entered)
        - Must be numeric (int or float)
        - Cannot be between -100 and +100 (invalid range in American odds)
        - Should be whole number (no decimals)
    """
    # None is valid (no odds provided)
    if odds is None:
        return (True, '')

    # Must be numeric
    try:
        numeric_odds = float(odds)
    except (TypeError, ValueError):
        return (False, 'Odds must be numeric')

    # Cannot be between -100 and +100 (invalid range)
    if -100 < numeric_odds < 100:
        return (False, 'American odds cannot be between -100 and +100')

    # Warn if not a whole number
    if numeric_odds != int(numeric_odds):
        return (False, 'American odds should be whole numbers')

    return (True, '')


def calculate_payout(bet_amount, odds, won):
    """
    Calculate actual profit/loss for a completed bet.

    Args:
        bet_amount: float - Amount wagered
        odds: int - American odds (e.g., -110, +105)
        won: bool - Whether the bet won

    Returns:
        float: Profit if won (positive), loss if lost (negative)

    Examples:
        >>> calculate_payout(1.0, -110, True)
        0.909...  # Win $0.909 on $1 bet at -110
        >>> calculate_payout(1.0, -110, False)
        -1.0  # Lose $1 on losing bet
        >>> calculate_payout(1.0, +105, True)
        1.05  # Win $1.05 on $1 bet at +105
        >>> calculate_payout(1.0, +105, False)
        -1.0  # Lose $1 on losing bet
        >>> calculate_payout(2.5, -150, True)
        1.666...  # Win $1.67 on $2.50 bet at -150

    Formula:
        If won: bet_amount * (decimal_odds - 1)
        If lost: -bet_amount
    """
    if won:
        decimal_odds = american_to_decimal(odds)
        return bet_amount * (decimal_odds - 1)
    else:
        return -bet_amount
