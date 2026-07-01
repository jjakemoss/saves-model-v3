"""
Display betting performance dashboard

This script:
1. Reads data/betting.db
2. Calculates performance metrics
3. Displays formatted report to console
4. Regenerates the Excel snapshot (including the Summary sheet)
5. Optionally saves report to file

Usage:
    python scripts/betting_dashboard.py [--save]
"""
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting import BettingDB, calculate_performance_metrics, format_metrics_report, export_to_excel
from betting.db_manager import DEFAULT_DB_PATH
from betting.excel_export import DEFAULT_XLSX_PATH


def show_dashboard(db_path=DEFAULT_DB_PATH, tracker_file=DEFAULT_XLSX_PATH, save_report=False):
    """
    Display betting performance dashboard

    Args:
        db_path: Path to the betting database
        tracker_file: Path to the generated Excel snapshot
        save_report: If True, save report to file

    Returns:
        None
    """
    tracker = BettingDB(db_path)

    print("\nLoading betting data...")
    bets_df = tracker.get_all_bets()

    if len(bets_df) == 0:
        print("[WARNING] No betting data available yet")
        return

    # Calculate metrics
    print("Calculating performance metrics...\n")
    metrics = calculate_performance_metrics(bets_df)

    # Format report
    report = format_metrics_report(metrics)

    # Display to console
    print(report)

    # Regenerate Excel snapshot (includes an updated Summary sheet)
    print("\nRegenerating Excel snapshot...")
    export_to_excel(db_path, tracker_file)
    print(f"[OK] Excel snapshot updated: {tracker_file}")

    # Save to file if requested
    if save_report:
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f'betting_performance_{timestamp}.txt'

        with open(report_file, 'w') as f:
            f.write(report)

        print(f"\n[OK] Report saved to: {report_file}")

    # Show insights
    print_insights(metrics)


def print_insights(metrics):
    """
    Print actionable insights from metrics

    Args:
        metrics: dict from calculate_performance_metrics()
    """
    overall = metrics['overall']

    if overall['total_bets'] == 0:
        return

    print("\nKEY INSIGHTS")
    print("-" * 60)

    # Overall performance
    if overall['roi'] > 5:
        print("[OK] Strong overall ROI - strategy is profitable")
    elif overall['roi'] > 0:
        print("[OK] Positive ROI - strategy is working")
    elif overall['roi'] > -5:
        print("[WARNING] Slightly negative ROI - monitor performance")
    else:
        print("[WARNING] Negative ROI - consider strategy adjustment")

    # Sample size
    if overall['total_bets'] < 30:
        print("[WARNING] Small sample size - need more data for reliable conclusions")
    elif overall['total_bets'] < 100:
        print("[INFO] Moderate sample size - trends becoming more reliable")
    else:
        print("[OK] Large sample size - performance metrics are reliable")

    # Confidence bucket analysis
    print("\nConfidence Bucket Performance:")
    best_roi = -999
    best_bucket = None

    for bucket, conf_metrics in metrics['by_confidence'].items():
        if conf_metrics['total_bets'] >= 10:  # Only analyze buckets with enough bets
            roi = conf_metrics['roi']
            if roi > best_roi:
                best_roi = roi
                best_bucket = bucket

            win_rate = conf_metrics['win_rate']
            if roi > 10:
                print(f"  [OK] {bucket}: Excellent performance ({roi:+.1f}% ROI, {win_rate:.1%} win rate)")
            elif roi > 5:
                print(f"  [OK] {bucket}: Strong performance ({roi:+.1f}% ROI, {win_rate:.1%} win rate)")
            elif roi > 0:
                print(f"  [INFO] {bucket}: Profitable ({roi:+.1f}% ROI, {win_rate:.1%} win rate)")
            else:
                print(f"  [WARNING] {bucket}: Unprofitable ({roi:+.1f}% ROI, {win_rate:.1%} win rate)")

    if best_bucket and best_roi > 5:
        print(f"\n[INFO] Best bucket: {best_bucket} - focus bets here")

    # Recent trend
    recent = metrics['recent']
    if 'last_20' in recent and recent['last_20']['total_bets'] >= 10:
        recent_roi = recent['last_20']['roi']
        if recent_roi > overall['roi'] + 5:
            print("\n[OK] Recent performance trending UP")
        elif recent_roi < overall['roi'] - 5:
            print("\n[WARNING] Recent performance trending DOWN")

    # Bet type analysis
    over_metrics = metrics['by_bet_type'].get('OVER', {})
    under_metrics = metrics['by_bet_type'].get('UNDER', {})

    if over_metrics.get('total_bets', 0) >= 10 and under_metrics.get('total_bets', 0) >= 10:
        over_roi = over_metrics['roi']
        under_roi = under_metrics['roi']

        print(f"\nBet Type Preference:")
        if abs(over_roi - under_roi) > 10:
            if over_roi > under_roi:
                print(f"  [INFO] OVER bets performing significantly better ({over_roi:+.1f}% vs {under_roi:+.1f}%)")
            else:
                print(f"  [INFO] UNDER bets performing significantly better ({under_roi:+.1f}% vs {over_roi:+.1f}%)")

    print("")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Display betting performance dashboard'
    )
    parser.add_argument(
        '--db',
        type=str,
        default=DEFAULT_DB_PATH,
        help='Path to betting database. Default: data/betting.db'
    )
    parser.add_argument(
        '--tracker',
        type=str,
        default=DEFAULT_XLSX_PATH,
        help='Path to betting tracker Excel snapshot. Default: betting_tracker.xlsx'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save report to reports/ directory'
    )

    args = parser.parse_args()

    try:
        show_dashboard(db_path=args.db, tracker_file=args.tracker, save_report=args.save)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
