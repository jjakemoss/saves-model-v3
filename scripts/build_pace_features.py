"""Build pregame-safe MoneyPuck pace/xG features for clean goalie-game rows.

Inputs:
  - data/processed/clean_training_data.parquet
  - data/raw/moneypuck/team_games.parquet
  - data/raw/moneypuck/goalie_games.parquet

Output:
  - data/processed/pace_features.parquet
  - data/processed/pace_features_metadata.json

Feature computation lives in src/features/pace_features.py so future live code
can import the same definitions used by the offline experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.pace_features import (
    DEFAULT_CLEAN,
    DEFAULT_GOALIE_GAMES,
    DEFAULT_OUT,
    DEFAULT_TEAM_GAMES,
    KEY_COLUMNS,
    build_pace_features,
    feature_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pregame-safe pace/xG feature artifact."
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--team-games", type=Path, default=DEFAULT_TEAM_GAMES)
    parser.add_argument("--goalie-games", type=Path, default=DEFAULT_GOALIE_GAMES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def print_null_summary(metadata: dict, max_months: int = 12) -> None:
    rates = metadata["null_rates_by_season_month"]
    feature_cols = metadata["generated_columns"]

    print("Null rates by season-month (mean across generated features):")
    for month in sorted(rates)[:max_months]:
        month_rates = rates[month]
        mean_rate = sum(month_rates.values()) / len(feature_cols)
        max_feature, max_rate = max(month_rates.items(), key=lambda item: item[1])
        print(
            f"  {month}: mean={_format_pct(mean_rate)}, "
            f"max={max_feature} {_format_pct(max_rate)}"
        )
    if len(rates) > max_months:
        remaining = len(rates) - max_months
        print(
            f"  ... {remaining} more month(s) written in metadata JSON with exact per-feature rates"
        )


def print_summary(out_path: Path, context: pd.DataFrame, metadata: dict) -> None:
    feature_cols = feature_columns()
    print(f"Built {out_path}")
    print(f"Rows: {metadata['row_count']:,}")
    print(f"Generated feature columns: {metadata['feature_count']}")
    print("Feature names:")
    for col in feature_cols:
        print(f"  - {col}")
    print(
        "Join coverage: "
        f"team={_format_pct(metadata['coverage']['team_all_match_rate'])}, "
        f"opponent={_format_pct(metadata['coverage']['opponent_all_match_rate'])}, "
        f"goalie={_format_pct(metadata['coverage']['goalie_all_match_rate'])}"
    )
    print(f"Key uniqueness: {not context.duplicated(KEY_COLUMNS).any()}")

    top_nulls = sorted(
        metadata["null_counts"].items(), key=lambda item: item[1], reverse=True
    )[:10]
    if top_nulls:
        formatted = ", ".join(f"{col}={count:,}" for col, count in top_nulls)
        print(f"Top null counts: {formatted}")
    else:
        print("Top null counts: none")
    print_null_summary(metadata)


def main() -> None:
    args = parse_args()
    context, metadata = build_pace_features(
        clean_path=args.clean,
        team_games_path=args.team_games,
        goalie_games_path=args.goalie_games,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    context.to_parquet(args.out, index=False)

    metadata_path = args.out.with_name(f"{args.out.stem}_metadata.json")
    metadata["output_path"] = str(args.out)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=False)
        f.write("\n")

    print_summary(args.out, context, metadata)
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
