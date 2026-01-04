#!/usr/bin/env python
"""
Historical Data Collection Script

Collects 2-3 seasons of NHL data:
- Game schedules
- Boxscores and play-by-play
- Goalie game logs and Edge stats

Usage:
    python scripts/collect_historical_data.py
    python scripts/collect_historical_data.py --seasons 20222023 20232024
    python scripts/collect_historical_data.py --help
"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.collectors import DataCollectionOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Collect historical NHL data for goalie saves model"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["20222023", "20232024", "20242025", "20252026"],
        help="Seasons to collect in YYYYYYYY format (default: last 4 seasons)"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--schedules-only",
        action="store_true",
        help="Only collect schedules (skip games and goalie stats)"
    )
    parser.add_argument(
        "--games-only",
        action="store_true",
        help="Only collect games (skip schedules and goalie stats)"
    )
    parser.add_argument(
        "--goalies-only",
        action="store_true",
        help="Only collect goalie stats (skip schedules and games)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("NHL GOALIE SAVES MODEL - HISTORICAL DATA COLLECTION")
    print("=" * 70)
    print(f"Seasons: {', '.join(args.seasons)}")
    print(f"Config: {args.config}")
    print("=" * 70 + "\n")

    # Initialize orchestrator
    orchestrator = DataCollectionOrchestrator(config_path=args.config)

    # Run collection based on flags
    if args.schedules_only:
        print("Mode: Schedules Only\n")
        summary = orchestrator.collect_all_schedules(args.seasons)

    elif args.games_only:
        print("Mode: Games Only\n")
        summary = orchestrator.collect_all_games(args.seasons)

    elif args.goalies_only:
        print("Mode: Goalie Stats Only\n")
        summary = orchestrator.collect_all_goalie_stats(args.seasons)

    else:
        print("Mode: Full Collection\n")
        summary = orchestrator.run_full_collection(seasons=args.seasons)

    # Save summary to file
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Collection summary saved to: {summary_file}")
    print("\nData collection complete!")
    print("\nNext steps:")
    print("  1. Review the collection summary")
    print("  2. Run feature engineering: python scripts/engineer_features.py")
    print("  3. Train the model: python scripts/train_model.py\n")


if __name__ == "__main__":
    main()
