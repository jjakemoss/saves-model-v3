"""
Build pregame-safe game context features for goalie-game rows.

Inputs:
  - data/processed/clean_training_data.parquet
  - data/raw/schedules/*.json

Output:
  - data/processed/game_context_features.parquet
  - data/processed/game_context_features_metadata.json

The artifact is keyed by game_id, goalie_id, team_abbrev, opponent_team,
game_date. All season-to-date features are shifted within season/group so the
current game is excluded. Schedule features use only the schedule date/history
known before puck drop, not postgame outcomes, odds, starters, play-by-play, or
xG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CLEAN = Path("data/processed/clean_training_data.parquet")
DEFAULT_SCHEDULES_DIR = Path("data/raw/schedules")
DEFAULT_OUT = Path("data/processed/game_context_features.parquet")

KEY_COLUMNS = ["game_id", "goalie_id", "team_abbrev", "opponent_team", "game_date"]
VERIFICATION_COLUMNS = [
    "season",
    "is_home",
    "schedule_is_home",
    "schedule_opponent_team",
    "schedule_game_type",
    "team_schedule_matched",
    "opponent_schedule_matched",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pregame-safe game context feature artifact."
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--schedules-dir", type=Path, default=DEFAULT_SCHEDULES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_team_schedule_rows(schedules_dir: Path) -> pd.DataFrame:
    rows = []
    schedule_files = sorted(schedules_dir.glob("*.json"))
    if not schedule_files:
        raise FileNotFoundError(f"No schedule JSON files found in {schedules_dir}")

    for path in schedule_files:
        data = _read_json(path)
        for game in data.get("games", []):
            game_id = game.get("id")
            game_date = game.get("gameDate")
            if game_id is None or not game_date:
                continue

            away = game.get("awayTeam", {}) or {}
            home = game.get("homeTeam", {}) or {}
            away_abbrev = away.get("abbrev")
            home_abbrev = home.get("abbrev")
            if not away_abbrev or not home_abbrev:
                continue

            base = {
                "game_id": int(game_id),
                "season": game.get("season"),
                "game_date": pd.to_datetime(game_date),
                "game_type": game.get("gameType"),
                "start_time_utc": game.get("startTimeUTC"),
            }
            rows.append(
                {
                    **base,
                    "team_abbrev": away_abbrev,
                    "opponent_team": home_abbrev,
                    "is_home": 0,
                }
            )
            rows.append(
                {
                    **base,
                    "team_abbrev": home_abbrev,
                    "opponent_team": away_abbrev,
                    "is_home": 1,
                }
            )

    if not rows:
        raise ValueError(f"No games parsed from schedule JSON files in {schedules_dir}")

    schedule = pd.DataFrame(rows)
    schedule = schedule.sort_values(
        ["game_id", "team_abbrev", "game_date", "is_home"]
    ).drop_duplicates(["game_id", "team_abbrev"], keep="first")
    schedule = schedule.sort_values(
        ["season", "team_abbrev", "game_date", "game_id"]
    ).reset_index(drop=True)
    return schedule


def _games_last_four_days(dates: Iterable[pd.Timestamp]) -> list[int]:
    values = list(pd.to_datetime(pd.Series(dates)))
    counts = []
    start = 0
    for end, current in enumerate(values):
        cutoff = current - pd.Timedelta(days=3)
        while values[start] < cutoff:
            start += 1
        counts.append(end - start + 1)
    return counts


def add_schedule_rest_features(schedule: pd.DataFrame) -> pd.DataFrame:
    regular = schedule[schedule["game_type"] == 2].copy()
    regular = regular.sort_values(
        ["season", "team_abbrev", "game_date", "game_id"]
    ).reset_index(drop=True)

    group_cols = ["season", "team_abbrev"]
    regular["prev_game_date"] = regular.groupby(group_cols)["game_date"].shift(1)
    regular["days_since_prev_game"] = (
        regular["game_date"] - regular["prev_game_date"]
    ).dt.days
    regular["is_back_to_back"] = (regular["days_since_prev_game"] == 1).astype(int)
    regular["games_last_4_days"] = (
        regular.groupby(group_cols)["game_date"]
        .transform(lambda s: _games_last_four_days(s))
        .astype(int)
    )
    regular["is_3_in_4"] = (regular["games_last_4_days"] >= 3).astype(int)

    return regular[
        [
            "game_id",
            "team_abbrev",
            "opponent_team",
            "game_type",
            "is_home",
            "days_since_prev_game",
            "is_back_to_back",
            "games_last_4_days",
            "is_3_in_4",
        ]
    ].copy()


def add_prior_group_features(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    prefix: str,
    count_col: str,
) -> pd.DataFrame:
    df = df.sort_values(group_cols + ["game_date", "game_id"]).copy()
    grouped = df.groupby(group_cols, sort=False)

    df[count_col] = grouped.cumcount()
    df[f"{prefix}_avg"] = grouped[value_col].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df[f"{prefix}_std"] = grouped[value_col].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std(ddof=0)
    )
    df[f"{prefix}_ema5"] = grouped[value_col].transform(
        lambda s: s.shift(1).ewm(span=5, adjust=False, min_periods=1).mean()
    )
    return df


def add_season_to_date_context(clean: pd.DataFrame) -> pd.DataFrame:
    base = clean.sort_values(["game_date", "game_id", "team_abbrev", "goalie_id"]).copy()

    team_def = add_prior_group_features(
        base[["game_id", "game_date", "season", "team_abbrev", "shots_against"]].copy(),
        ["season", "team_abbrev"],
        "shots_against",
        "team_s2d_shots_against",
        "team_s2d_games_prior",
    )
    team_def = team_def[
        [
            "game_id",
            "team_abbrev",
            "team_s2d_games_prior",
            "team_s2d_shots_against_avg",
            "team_s2d_shots_against_std",
            "team_s2d_shots_against_ema5",
        ]
    ]

    team_off = add_prior_group_features(
        base[["game_id", "game_date", "season", "team_abbrev", "team_shots"]].copy(),
        ["season", "team_abbrev"],
        "team_shots",
        "opponent_s2d_shots_for",
        "opponent_s2d_games_prior",
    )
    team_off = team_off[
        [
            "game_id",
            "team_abbrev",
            "opponent_s2d_games_prior",
            "opponent_s2d_shots_for_avg",
            "opponent_s2d_shots_for_std",
            "opponent_s2d_shots_for_ema5",
        ]
    ].rename(columns={"team_abbrev": "opponent_team"})

    goalie = add_prior_group_features(
        base[["game_id", "game_date", "season", "goalie_id", "shots_against"]].copy(),
        ["season", "goalie_id"],
        "shots_against",
        "goalie_s2d_shots_against",
        "goalie_s2d_starts_prior",
    )
    goalie = goalie[
        [
            "game_id",
            "goalie_id",
            "goalie_s2d_starts_prior",
            "goalie_s2d_shots_against_avg",
            "goalie_s2d_shots_against_std",
            "goalie_s2d_shots_against_ema5",
        ]
    ]

    context = base[KEY_COLUMNS + ["season", "is_home"]].copy()
    context = context.merge(team_def, on=["game_id", "team_abbrev"], how="left")
    context = context.merge(team_off, on=["game_id", "opponent_team"], how="left")
    context = context.merge(goalie, on=["game_id", "goalie_id"], how="left")

    rolling_sources = base[
        [
            "game_id",
            "goalie_id",
            "shots_against_rolling_5",
            "shots_against_rolling_10",
            "team_shots_against_rolling_5",
            "team_shots_against_rolling_10",
            "opp_shots_rolling_5",
            "opp_shots_rolling_10",
        ]
    ].copy()
    context = context.merge(rolling_sources, on=["game_id", "goalie_id"], how="left")

    add_relative_roll_features(context)
    context = context.drop(
        columns=[
            "shots_against_rolling_5",
            "shots_against_rolling_10",
            "team_shots_against_rolling_5",
            "team_shots_against_rolling_10",
            "opp_shots_rolling_5",
            "opp_shots_rolling_10",
        ]
    )
    return context


def safe_z(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def add_relative_roll_features(df: pd.DataFrame) -> None:
    specs = [
        (
            "goalie_shots_against",
            "shots_against_rolling_{window}",
            "goalie_s2d_shots_against_avg",
            "goalie_s2d_shots_against_std",
        ),
        (
            "team_shots_against",
            "team_shots_against_rolling_{window}",
            "team_s2d_shots_against_avg",
            "team_s2d_shots_against_std",
        ),
        (
            "opponent_shots_for",
            "opp_shots_rolling_{window}",
            "opponent_s2d_shots_for_avg",
            "opponent_s2d_shots_for_std",
        ),
    ]

    for window in [5, 10]:
        for prefix, rolling_template, avg_col, std_col in specs:
            rolling_col = rolling_template.format(window=window)
            rel_col = f"{prefix}_roll{window}_rel_s2d"
            z_col = f"{prefix}_roll{window}_z_s2d"
            df[rel_col] = df[rolling_col] - df[avg_col]
            df[z_col] = safe_z(df[rel_col], df[std_col])


def add_schedule_context(context: pd.DataFrame, schedule_features: pd.DataFrame) -> pd.DataFrame:
    team_sched = schedule_features.rename(
        columns={
            "opponent_team": "schedule_opponent_team",
            "game_type": "schedule_game_type",
            "is_home": "schedule_is_home",
            "days_since_prev_game": "team_days_since_prev_game",
            "is_back_to_back": "team_is_back_to_back",
            "games_last_4_days": "team_games_last_4_days",
            "is_3_in_4": "team_is_3_in_4",
        }
    )
    team_cols = [
        "game_id",
        "team_abbrev",
        "schedule_opponent_team",
        "schedule_game_type",
        "schedule_is_home",
        "team_days_since_prev_game",
        "team_is_back_to_back",
        "team_games_last_4_days",
        "team_is_3_in_4",
    ]
    context = context.merge(team_sched[team_cols], on=["game_id", "team_abbrev"], how="left")
    context["team_schedule_matched"] = context["schedule_game_type"].notna().astype(int)

    opp_sched = schedule_features[
        [
            "game_id",
            "team_abbrev",
            "days_since_prev_game",
            "is_back_to_back",
            "games_last_4_days",
            "is_3_in_4",
        ]
    ].rename(
        columns={
            "team_abbrev": "opponent_team",
            "days_since_prev_game": "opponent_days_since_prev_game",
            "is_back_to_back": "opponent_is_back_to_back",
            "games_last_4_days": "opponent_games_last_4_days",
            "is_3_in_4": "opponent_is_3_in_4",
        }
    )
    opp_cols = [
        "game_id",
        "opponent_team",
        "opponent_days_since_prev_game",
        "opponent_is_back_to_back",
        "opponent_games_last_4_days",
        "opponent_is_3_in_4",
    ]
    context = context.merge(opp_sched[opp_cols], on=["game_id", "opponent_team"], how="left")
    context["opponent_schedule_matched"] = (
        context["opponent_games_last_4_days"].notna().astype(int)
    )
    return context


def build_features(clean_path: Path, schedules_dir: Path) -> tuple[pd.DataFrame, dict]:
    clean = pd.read_parquet(clean_path)
    missing = [col for col in KEY_COLUMNS + ["season", "is_home"] if col not in clean.columns]
    if missing:
        raise ValueError(f"Clean training data missing required columns: {missing}")

    clean = clean.copy()
    clean["game_date"] = pd.to_datetime(clean["game_date"])

    schedule = load_team_schedule_rows(schedules_dir)
    schedule_features = add_schedule_rest_features(schedule)

    context = add_season_to_date_context(clean)
    context = add_schedule_context(context, schedule_features)

    first_cols = KEY_COLUMNS + VERIFICATION_COLUMNS
    feature_cols = [col for col in context.columns if col not in first_cols]
    context = context[first_cols + feature_cols]
    context = context.sort_values(["game_date", "game_id", "team_abbrev", "goalie_id"])
    context = context.reset_index(drop=True)

    if context.duplicated(KEY_COLUMNS).any():
        dupes = context.loc[context.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
        raise ValueError(f"Output key is not unique. Example duplicates:\n{dupes.head()}")

    metadata = build_metadata(
        clean_path=clean_path,
        schedules_dir=schedules_dir,
        schedule_files=sorted(schedules_dir.glob("*.json")),
        context=context,
        feature_cols=feature_cols,
        clean_rows=len(clean),
    )
    return context, metadata


def build_metadata(
    clean_path: Path,
    schedules_dir: Path,
    schedule_files: list[Path],
    context: pd.DataFrame,
    feature_cols: list[str],
    clean_rows: int,
) -> dict:
    null_counts = {
        col: int(count)
        for col, count in context[feature_cols].isna().sum().items()
        if int(count) > 0
    }
    return {
        "artifact": "game_context_features",
        "input_paths": {
            "clean": str(clean_path),
            "schedules_dir": str(schedules_dir),
        },
        "schedule_file_count": len(schedule_files),
        "row_count": int(len(context)),
        "clean_input_row_count": int(clean_rows),
        "key_columns": KEY_COLUMNS,
        "verification_columns": VERIFICATION_COLUMNS,
        "generated_columns": feature_cols,
        "feature_count": len(feature_cols),
        "coverage": {
            "team_schedule_match_rate": float(context["team_schedule_matched"].mean()),
            "opponent_schedule_match_rate": float(
                context["opponent_schedule_matched"].mean()
            ),
        },
        "null_counts": null_counts,
        "leakage_notes": [
            "No current-game outcome columns are emitted.",
            "Season-to-date averages, standard deviations, counts, and EMA features use shift(1) within season/group.",
            "Schedule rest flags use regular-season schedule rows and only prior scheduled games plus the current scheduled date.",
            "No betting odds, moneyline, totals, postgame starter flags, play-by-play, or xG inputs are used.",
        ],
    }


def print_summary(out_path: Path, context: pd.DataFrame, metadata: dict) -> None:
    null_counts = metadata["null_counts"]
    top_nulls = sorted(null_counts.items(), key=lambda item: item[1], reverse=True)[:8]
    print(f"Built {out_path}")
    print(f"Rows: {metadata['row_count']:,}")
    print(f"Generated feature columns: {metadata['feature_count']}")
    print(
        "Schedule coverage: "
        f"team={metadata['coverage']['team_schedule_match_rate']:.1%}, "
        f"opponent={metadata['coverage']['opponent_schedule_match_rate']:.1%}"
    )
    print(f"Key uniqueness: {not context.duplicated(KEY_COLUMNS).any()}")
    if top_nulls:
        formatted = ", ".join(f"{col}={count:,}" for col, count in top_nulls)
        print(f"Top null counts: {formatted}")
    else:
        print("Top null counts: none")


def main() -> None:
    args = parse_args()
    context, metadata = build_features(args.clean, args.schedules_dir)

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
