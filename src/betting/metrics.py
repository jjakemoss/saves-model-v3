"""
Performance metrics calculation for betting tracker
"""
import pandas as pd
import numpy as np


def calculate_performance_metrics(bets_df):
    """
    Calculate overall and segmented performance metrics

    Args:
        bets_df: pd.DataFrame from Bets sheet

    Returns:
        dict: Performance metrics organized by category
    """
    metrics = {}

    # Filter to actual bets (exclude NO BET and empty selections)
    actual_bets = bets_df[
        (bets_df['bet_selection'].isin(['OVER', 'UNDER'])) &
        (bets_df['result'].notna()) &
        (bets_df['result'] != '')
    ].copy()

    # Overall Performance
    metrics['overall'] = calculate_overall_metrics(actual_bets)

    # Performance by Confidence Bucket
    metrics['by_confidence'] = calculate_confidence_metrics(actual_bets)

    # Performance by Bet Type (OVER vs UNDER)
    metrics['by_bet_type'] = calculate_bet_type_metrics(actual_bets)

    # Recent Performance
    metrics['recent'] = calculate_recent_metrics(actual_bets)

    return metrics


def calculate_overall_metrics(bets_df):
    """Calculate overall betting performance"""
    if len(bets_df) == 0:
        return {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
            'win_rate': 0.0,
            'total_profit_loss': 0.0,
            'roi': 0.0,
            'units_wagered': 0.0
        }

    total_bets = len(bets_df)
    wins = len(bets_df[bets_df['result'] == 'WIN'])
    losses = len(bets_df[bets_df['result'] == 'LOSS'])
    pushes = len(bets_df[bets_df['result'] == 'PUSH'])

    # Win rate (excluding pushes)
    decided_bets = wins + losses
    win_rate = wins / decided_bets if decided_bets > 0 else 0.0

    # Profit/Loss
    total_profit_loss = bets_df['profit_loss'].sum()

    # Units wagered
    units_wagered = bets_df['bet_amount'].sum()

    # ROI
    roi = (total_profit_loss / units_wagered * 100) if units_wagered > 0 else 0.0

    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': win_rate,
        'total_profit_loss': total_profit_loss,
        'roi': roi,
        'units_wagered': units_wagered
    }


def calculate_confidence_metrics(bets_df):
    """Calculate performance by confidence bucket"""
    confidence_buckets = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75%+']

    metrics = {}
    for bucket in confidence_buckets:
        bucket_bets = bets_df[bets_df['confidence_bucket'] == bucket]
        metrics[bucket] = calculate_overall_metrics(bucket_bets)

    return metrics


def calculate_bet_type_metrics(bets_df):
    """Calculate performance by bet type (OVER vs UNDER)"""
    metrics = {}

    for bet_type in ['OVER', 'UNDER']:
        type_bets = bets_df[bets_df['bet_selection'] == bet_type]
        metrics[bet_type] = calculate_overall_metrics(type_bets)

    return metrics


def calculate_recent_metrics(bets_df, windows=[10, 20, 50]):
    """Calculate recent performance over different windows"""
    metrics = {}

    # Sort by date descending
    sorted_bets = bets_df.sort_values('game_date', ascending=False)

    for window in windows:
        recent_bets = sorted_bets.head(window)
        metrics[f'last_{window}'] = calculate_overall_metrics(recent_bets)

    return metrics


def format_metrics_report(metrics):
    """
    Format metrics dict into readable text report

    Args:
        metrics: dict from calculate_performance_metrics()

    Returns:
        str: Formatted text report
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("BETTING PERFORMANCE REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Overall Performance
    overall = metrics['overall']
    lines.append("OVERALL PERFORMANCE")
    lines.append("-" * 60)
    lines.append(f"Total Bets:        {overall['total_bets']}")
    lines.append(f"Wins:              {overall['wins']}")
    lines.append(f"Losses:            {overall['losses']}")
    lines.append(f"Pushes:            {overall['pushes']}")
    lines.append(f"Win Rate:          {overall['win_rate']:.1%}")
    lines.append(f"Total P/L:         {overall['total_profit_loss']:+.2f} units")
    lines.append(f"ROI:               {overall['roi']:+.2f}%")
    lines.append(f"Units Wagered:     {overall['units_wagered']:.2f}")
    lines.append("")

    # Performance by Confidence
    lines.append("PERFORMANCE BY CONFIDENCE LEVEL")
    lines.append("-" * 60)
    lines.append(f"{'Confidence':<15} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'P/L':<12} {'ROI':<10}")
    lines.append("-" * 60)

    for bucket, conf_metrics in metrics['by_confidence'].items():
        if conf_metrics['total_bets'] > 0:
            lines.append(
                f"{bucket:<15} "
                f"{conf_metrics['total_bets']:<8} "
                f"{conf_metrics['wins']:<8} "
                f"{conf_metrics['win_rate']:<12.1%} "
                f"{conf_metrics['total_profit_loss']:<+12.2f} "
                f"{conf_metrics['roi']:<+10.2f}%"
            )
    lines.append("")

    # Performance by Bet Type
    lines.append("PERFORMANCE BY BET TYPE")
    lines.append("-" * 60)
    lines.append(f"{'Type':<10} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'P/L':<12} {'ROI':<10}")
    lines.append("-" * 60)

    for bet_type, type_metrics in metrics['by_bet_type'].items():
        if type_metrics['total_bets'] > 0:
            lines.append(
                f"{bet_type:<10} "
                f"{type_metrics['total_bets']:<8} "
                f"{type_metrics['wins']:<8} "
                f"{type_metrics['win_rate']:<12.1%} "
                f"{type_metrics['total_profit_loss']:<+12.2f} "
                f"{type_metrics['roi']:<+10.2f}%"
            )
    lines.append("")

    # Recent Performance
    lines.append("RECENT PERFORMANCE")
    lines.append("-" * 60)
    lines.append(f"{'Window':<15} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'P/L':<12} {'ROI':<10}")
    lines.append("-" * 60)

    for window_name, window_metrics in metrics['recent'].items():
        if window_metrics['total_bets'] > 0:
            lines.append(
                f"{window_name:<15} "
                f"{window_metrics['total_bets']:<8} "
                f"{window_metrics['wins']:<8} "
                f"{window_metrics['win_rate']:<12.1%} "
                f"{window_metrics['total_profit_loss']:<+12.2f} "
                f"{window_metrics['roi']:<+10.2f}%"
            )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def get_performance_summary(bets_df):
    """
    Get quick performance summary

    Args:
        bets_df: pd.DataFrame from Bets sheet

    Returns:
        str: One-line summary
    """
    metrics = calculate_performance_metrics(bets_df)
    overall = metrics['overall']

    if overall['total_bets'] == 0:
        return "No bets recorded yet"

    return (
        f"{overall['total_bets']} bets | "
        f"{overall['wins']}-{overall['losses']}-{overall['pushes']} | "
        f"Win Rate: {overall['win_rate']:.1%} | "
        f"P/L: {overall['total_profit_loss']:+.2f}u | "
        f"ROI: {overall['roi']:+.2f}%"
    )
