"""Pregame-safe pace/xG feature computation.

The public entry point is build_pace_features(). It is intentionally importable
so future live code can reuse the exact same feature definitions as the
offline experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["game_id", "goalie_id", "team_abbrev", "opponent_team", "game_date"]

DEFAULT_CLEAN = Path("data/processed/clean_training_data.parquet")
DEFAULT_TEAM_GAMES = Path("data/raw/moneypuck/team_games.parquet")
DEFAULT_GOALIE_GAMES = Path("data/raw/moneypuck/goalie_games.parquet")
DEFAULT_OUT = Path("data/processed/pace_features.parquet")

VERIFICATION_COLUMNS = [
    "season",
    "pace_team_all_matched",
    "pace_opponent_all_matched",
    "pace_goalie_all_matched",
]

TEAM_REQUIRED_COLUMNS = [
    "team",
    "season",
    "gameId",
    "opposingTeam",
    "gameDate",
    "situation",
    "iceTime",
    "xGoalsFor",
    "scoreVenueAdjustedxGoalsFor",
    "shotAttemptsFor",
    "scoreAdjustedShotsAttemptsFor",
    "unblockedShotAttemptsFor",
    "xGoalsAgainst",
    "scoreVenueAdjustedxGoalsAgainst",
    "shotAttemptsAgainst",
    "scoreAdjustedShotsAttemptsAgainst",
    "unblockedShotAttemptsAgainst",
    "playoffGame",
]

GOALIE_REQUIRED_COLUMNS = [
    "playerId",
    "gameId",
    "season",
    "playerTeam",
    "gameDate",
    "situation",
    "xGoals",
    "ongoal",
    "rebounds",
    "highDangerShots",
]

CLEAN_REQUIRED_COLUMNS = KEY_COLUMNS + ["season"]

TEAM_PRIOR_BASELINE_ALIASES = {
    # Utah inherited Arizona's franchise history for prior-season baselines.
    "UTA": "ARI",
}


FAMILY_COLUMNS = {
    "opponent_offense_pace": [
        "opp_off_all_corsi_s2d_mean",
        "opp_off_all_corsi_roll5",
        "opp_off_all_corsi_roll10",
        "opp_off_all_corsi_ema5",
        "opp_off_all_corsi_prior_season_mean",
        "opp_off_all_fenwick_s2d_mean",
        "opp_off_all_fenwick_ema5",
        "opp_off_all_xgf_s2d_mean",
        "opp_off_all_xgf_ema5",
        "opp_off_5v5_corsi_ema5",
        "opp_off_5v5_xgf_ema5",
        "opp_off_all_score_adj_corsi_ema5",
    ],
    "team_shot_suppression": [
        "team_def_all_corsi_against_s2d_mean",
        "team_def_all_corsi_against_roll5",
        "team_def_all_corsi_against_roll10",
        "team_def_all_corsi_against_ema5",
        "team_def_all_corsi_against_prior_season_mean",
        "team_def_all_fenwick_against_s2d_mean",
        "team_def_all_fenwick_against_ema5",
        "team_def_all_xga_s2d_mean",
        "team_def_all_xga_ema5",
        "team_def_5v5_corsi_against_ema5",
        "team_def_5v5_xga_ema5",
        "team_def_all_score_adj_corsi_against_ema5",
    ],
    "combined_pace": [
        "combined_all_corsi_ema5",
        "combined_all_fenwick_ema5",
        "combined_all_xg_ema5",
        "combined_5v5_corsi_ema5",
        "combined_5v5_xg_ema5",
    ],
    "special_teams_volume": [
        "opp_pp_corsi_s2d_mean",
        "opp_pp_corsi_ema5",
        "opp_pp_xgf_s2d_mean",
        "opp_pp_xgf_ema5",
        "team_pk_corsi_against_s2d_mean",
        "team_pk_corsi_against_ema5",
        "team_pk_xga_ema5",
        "team_pk_icetime_minutes_s2d_mean",
    ],
    "goalie_workload_quality": [
        "goalie_xg_per_shot_roll10",
        "goalie_xg_per_shot_ema5",
        "goalie_high_danger_share_ema5",
        "goalie_rebound_rate_ema5",
    ],
    "league_relative_zscores": [
        "opp_off_all_corsi_ema5_prior_league_z",
        "team_def_all_corsi_against_ema5_prior_league_z",
        "combined_all_corsi_ema5_prior_league_z",
        "combined_all_xg_ema5_prior_league_z",
    ],
}


def feature_columns() -> list[str]:
    """Return generated feature columns in deterministic artifact order."""
    columns: list[str] = []
    for family_cols in FAMILY_COLUMNS.values():
        columns.extend(family_cols)
    return columns


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _team_franchise_code(team: pd.Series) -> pd.Series:
    return team.replace(TEAM_PRIOR_BASELINE_ALIASES)


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def _shifted_ema(series: pd.Series, span: int = 5) -> pd.Series:
    return series.shift(1).ewm(span=span, adjust=False, min_periods=1).mean()


def _add_prior_stats(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    prefix: str,
    *,
    s2d: bool = False,
    roll5: bool = False,
    roll10: bool = False,
    ema5: bool = False,
) -> pd.DataFrame:
    frame = frame.sort_values(group_cols + ["gameDate", "gameId"]).copy()
    grouped = frame.groupby(group_cols, sort=False)[value_col]

    if s2d:
        frame[f"{prefix}_s2d_mean"] = grouped.transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
    if roll5:
        frame[f"{prefix}_roll5"] = grouped.transform(_shifted_rolling_mean, window=5)
    if roll10:
        frame[f"{prefix}_roll10"] = grouped.transform(_shifted_rolling_mean, window=10)
    if ema5:
        frame[f"{prefix}_ema5"] = grouped.transform(_shifted_ema)

    return frame


def _add_prior_season_mean(
    base: pd.DataFrame,
    all_team_games: pd.DataFrame,
    situation: str,
    value_col: str,
    output_col: str,
) -> pd.DataFrame:
    history = all_team_games[all_team_games["situation"] == situation].copy()
    history["franchise_team"] = _team_franchise_code(history["team"])

    means = (
        history.groupby(["season", "franchise_team"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: output_col})
    )
    means["season"] = means["season"] + 1

    result = base.copy()
    result["franchise_team"] = _team_franchise_code(result["team"])
    result = result.merge(
        means,
        on=["season", "franchise_team"],
        how="left",
    )
    return result.drop(columns=["franchise_team"])


def _regular_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    _require_columns(team_games, TEAM_REQUIRED_COLUMNS, "team_games")
    team = team_games[team_games["playoffGame"] == 0].copy()
    team["gameDate"] = pd.to_datetime(team["gameDate"])
    team["season"] = team["season"].astype(int)
    team["gameId"] = team["gameId"].astype(int)
    return team.sort_values(["season", "gameDate", "gameId", "team", "situation"])


def _regular_goalie_games(goalie_games: pd.DataFrame) -> pd.DataFrame:
    _require_columns(goalie_games, GOALIE_REQUIRED_COLUMNS, "goalie_games")
    goalie = goalie_games.copy()
    goalie["gameDate"] = pd.to_datetime(goalie["gameDate"])
    goalie["season"] = goalie["season"].astype(int)
    goalie["gameId"] = goalie["gameId"].astype(int)
    goalie["playerId"] = goalie["playerId"].astype(int)
    return goalie.sort_values(["season", "gameDate", "gameId", "playerId", "situation"])


def _team_game_base(team: pd.DataFrame) -> pd.DataFrame:
    base = team[team["situation"] == "all"][
        ["gameId", "team", "season", "gameDate", "opposingTeam"]
    ].drop_duplicates(["gameId", "team"])
    if base.duplicated(["gameId", "team"]).any():
        raise ValueError("team game base has duplicate (gameId, team) rows")
    return base.sort_values(["season", "gameDate", "gameId", "team"]).reset_index(drop=True)


def _merge_metric_features(
    base: pd.DataFrame,
    team: pd.DataFrame,
    situation: str,
    value_col: str,
    prefix: str,
    *,
    s2d: bool = False,
    roll5: bool = False,
    roll10: bool = False,
    ema5: bool = False,
    prior_season: bool = False,
) -> pd.DataFrame:
    source_cols = ["gameId", "team", "season", "gameDate", "situation", value_col]
    source = team.loc[team["situation"] == situation, source_cols].copy()
    source = _add_prior_stats(
        source,
        ["season", "team"],
        value_col,
        prefix,
        s2d=s2d,
        roll5=roll5,
        roll10=roll10,
        ema5=ema5,
    )

    keep_cols = ["gameId", "team"] + [
        col
        for col in source.columns
        if col.startswith(f"{prefix}_") and col not in ["gameId", "team"]
    ]
    source = source[keep_cols].drop_duplicates(["gameId", "team"])

    if prior_season:
        prior_frame = _add_prior_season_mean(
            base[["gameId", "team", "season"]].copy(),
            team,
            situation,
            value_col,
            f"{prefix}_prior_season_mean",
        )
        source = source.merge(
            prior_frame[["gameId", "team", f"{prefix}_prior_season_mean"]],
            on=["gameId", "team"],
            how="left",
        )

    return base.merge(source, on=["gameId", "team"], how="left")


def build_team_pregame_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Build one pregame-safe row per regular-season team-game."""
    team = _regular_team_games(team_games)
    features = _team_game_base(team)

    specs = [
        ("all", "shotAttemptsFor", "off_all_corsi", True, True, True, True, True),
        ("all", "unblockedShotAttemptsFor", "off_all_fenwick", True, False, False, True, False),
        ("all", "xGoalsFor", "off_all_xgf", True, False, False, True, False),
        ("5on5", "shotAttemptsFor", "off_5v5_corsi", False, False, False, True, False),
        ("5on5", "xGoalsFor", "off_5v5_xgf", False, False, False, True, False),
        (
            "all",
            "scoreAdjustedShotsAttemptsFor",
            "off_all_score_adj_corsi",
            False,
            False,
            False,
            True,
            False,
        ),
        (
            "all",
            "shotAttemptsAgainst",
            "def_all_corsi_against",
            True,
            True,
            True,
            True,
            True,
        ),
        (
            "all",
            "unblockedShotAttemptsAgainst",
            "def_all_fenwick_against",
            True,
            False,
            False,
            True,
            False,
        ),
        ("all", "xGoalsAgainst", "def_all_xga", True, False, False, True, False),
        (
            "5on5",
            "shotAttemptsAgainst",
            "def_5v5_corsi_against",
            False,
            False,
            False,
            True,
            False,
        ),
        ("5on5", "xGoalsAgainst", "def_5v5_xga", False, False, False, True, False),
        (
            "all",
            "scoreAdjustedShotsAttemptsAgainst",
            "def_all_score_adj_corsi_against",
            False,
            False,
            False,
            True,
            False,
        ),
        ("5on4", "shotAttemptsFor", "pp_corsi", True, False, False, True, False),
        ("5on4", "xGoalsFor", "pp_xgf", True, False, False, True, False),
        (
            "4on5",
            "shotAttemptsAgainst",
            "pk_corsi_against",
            True,
            False,
            False,
            True,
            False,
        ),
        ("4on5", "xGoalsAgainst", "pk_xga", False, False, False, True, False),
    ]

    for situation, value_col, prefix, s2d, roll5, roll10, ema5, prior in specs:
        features = _merge_metric_features(
            features,
            team,
            situation,
            value_col,
            prefix,
            s2d=s2d,
            roll5=roll5,
            roll10=roll10,
            ema5=ema5,
            prior_season=prior,
        )

    team_4on5 = team[team["situation"] == "4on5"][
        ["gameId", "team", "season", "gameDate", "iceTime"]
    ].copy()
    team_4on5["pk_icetime_minutes"] = team_4on5["iceTime"] / 60.0
    team_4on5 = _add_prior_stats(
        team_4on5,
        ["season", "team"],
        "pk_icetime_minutes",
        "pk_icetime_minutes",
        s2d=True,
    )
    features = features.merge(
        team_4on5[["gameId", "team", "pk_icetime_minutes_s2d_mean"]],
        on=["gameId", "team"],
        how="left",
    )

    features = features.rename(columns={"gameId": "game_id", "team": "team_abbrev"})
    if features.duplicated(["game_id", "team_abbrev"]).any():
        raise ValueError("team pace features have duplicate (game_id, team_abbrev) rows")
    return features


def _rename_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    columns = ["game_id", "team_abbrev"] + list(mapping)
    return frame[columns].rename(columns=mapping)


def _team_offense_projection(team_features: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "off_all_corsi_s2d_mean": "opp_off_all_corsi_s2d_mean",
        "off_all_corsi_roll5": "opp_off_all_corsi_roll5",
        "off_all_corsi_roll10": "opp_off_all_corsi_roll10",
        "off_all_corsi_ema5": "opp_off_all_corsi_ema5",
        "off_all_corsi_prior_season_mean": "opp_off_all_corsi_prior_season_mean",
        "off_all_fenwick_s2d_mean": "opp_off_all_fenwick_s2d_mean",
        "off_all_fenwick_ema5": "opp_off_all_fenwick_ema5",
        "off_all_xgf_s2d_mean": "opp_off_all_xgf_s2d_mean",
        "off_all_xgf_ema5": "opp_off_all_xgf_ema5",
        "off_5v5_corsi_ema5": "opp_off_5v5_corsi_ema5",
        "off_5v5_xgf_ema5": "opp_off_5v5_xgf_ema5",
        "off_all_score_adj_corsi_ema5": "opp_off_all_score_adj_corsi_ema5",
        "pp_corsi_s2d_mean": "opp_pp_corsi_s2d_mean",
        "pp_corsi_ema5": "opp_pp_corsi_ema5",
        "pp_xgf_s2d_mean": "opp_pp_xgf_s2d_mean",
        "pp_xgf_ema5": "opp_pp_xgf_ema5",
    }
    projected = _rename_columns(team_features, mapping)
    projected["pace_opponent_all_matched"] = True
    return projected.rename(columns={"team_abbrev": "opponent_team"})


def _team_defense_projection(team_features: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "def_all_corsi_against_s2d_mean": "team_def_all_corsi_against_s2d_mean",
        "def_all_corsi_against_roll5": "team_def_all_corsi_against_roll5",
        "def_all_corsi_against_roll10": "team_def_all_corsi_against_roll10",
        "def_all_corsi_against_ema5": "team_def_all_corsi_against_ema5",
        "def_all_corsi_against_prior_season_mean": (
            "team_def_all_corsi_against_prior_season_mean"
        ),
        "def_all_fenwick_against_s2d_mean": "team_def_all_fenwick_against_s2d_mean",
        "def_all_fenwick_against_ema5": "team_def_all_fenwick_against_ema5",
        "def_all_xga_s2d_mean": "team_def_all_xga_s2d_mean",
        "def_all_xga_ema5": "team_def_all_xga_ema5",
        "def_5v5_corsi_against_ema5": "team_def_5v5_corsi_against_ema5",
        "def_5v5_xga_ema5": "team_def_5v5_xga_ema5",
        "def_all_score_adj_corsi_against_ema5": (
            "team_def_all_score_adj_corsi_against_ema5"
        ),
        "pk_corsi_against_s2d_mean": "team_pk_corsi_against_s2d_mean",
        "pk_corsi_against_ema5": "team_pk_corsi_against_ema5",
        "pk_xga_ema5": "team_pk_xga_ema5",
        "pk_icetime_minutes_s2d_mean": "team_pk_icetime_minutes_s2d_mean",
    }
    projected = _rename_columns(team_features, mapping)
    projected["pace_team_all_matched"] = True
    return projected


def build_goalie_workload_features(goalie_games: pd.DataFrame) -> pd.DataFrame:
    """Build prior-only goalie workload-quality features keyed by game/goalie."""
    goalie = _regular_goalie_games(goalie_games)
    goalie = goalie[goalie["situation"] == "all"].copy()
    goalie["goalie_xg_per_shot"] = _safe_divide(goalie["xGoals"], goalie["ongoal"])
    goalie["goalie_high_danger_share"] = _safe_divide(
        goalie["highDangerShots"], goalie["ongoal"]
    )
    goalie["goalie_rebound_rate"] = _safe_divide(goalie["rebounds"], goalie["ongoal"])

    specs = [
        ("goalie_xg_per_shot", True, True),
        ("goalie_high_danger_share", False, True),
        ("goalie_rebound_rate", False, True),
    ]
    features = goalie[
        ["gameId", "playerId", "season", "gameDate", "goalie_xg_per_shot"]
    ].copy()
    for value_col, roll10, ema5 in specs:
        source_cols = ["gameId", "playerId", "season", "gameDate", value_col]
        source = _add_prior_stats(
            goalie[source_cols].copy(),
            ["season", "playerId"],
            value_col,
            value_col,
            roll10=roll10,
            ema5=ema5,
        )
        keep_cols = ["gameId", "playerId"] + [
            col for col in source.columns if col.startswith(f"{value_col}_")
        ]
        source = source[keep_cols].drop_duplicates(["gameId", "playerId"])
        features = features.merge(source, on=["gameId", "playerId"], how="left")

    features = features.rename(columns={"gameId": "game_id", "playerId": "goalie_id"})
    features["pace_goalie_all_matched"] = True
    keep = [
        "game_id",
        "goalie_id",
        "pace_goalie_all_matched",
    ] + FAMILY_COLUMNS["goalie_workload_quality"]
    features = features[keep].drop_duplicates(["game_id", "goalie_id"])
    if features.duplicated(["game_id", "goalie_id"]).any():
        raise ValueError("goalie pace features have duplicate (game_id, goalie_id) rows")
    return features


def add_combined_pace_features(context: pd.DataFrame) -> None:
    context["combined_all_corsi_ema5"] = (
        context["opp_off_all_corsi_ema5"] + context["team_def_all_corsi_against_ema5"]
    )
    context["combined_all_fenwick_ema5"] = (
        context["opp_off_all_fenwick_ema5"]
        + context["team_def_all_fenwick_against_ema5"]
    )
    context["combined_all_xg_ema5"] = (
        context["opp_off_all_xgf_ema5"] + context["team_def_all_xga_ema5"]
    )
    context["combined_5v5_corsi_ema5"] = (
        context["opp_off_5v5_corsi_ema5"]
        + context["team_def_5v5_corsi_against_ema5"]
    )
    context["combined_5v5_xg_ema5"] = (
        context["opp_off_5v5_xgf_ema5"] + context["team_def_5v5_xga_ema5"]
    )


def _prior_league_stats_by_date(
    context: pd.DataFrame,
    source_col: str,
) -> pd.DataFrame:
    daily = (
        context.groupby(["season", "game_date"], as_index=False)[source_col]
        .agg(["count", "sum", lambda s: np.square(s.dropna()).sum()])
        .reset_index()
    )
    daily = daily.rename(columns={"<lambda_0>": "sumsq"})
    daily = daily.sort_values(["season", "game_date"])
    grouped = daily.groupby("season", sort=False)
    daily["prior_count"] = grouped["count"].cumsum() - daily["count"]
    daily["prior_sum"] = grouped["sum"].cumsum() - daily["sum"]
    daily["prior_sumsq"] = grouped["sumsq"].cumsum() - daily["sumsq"]
    daily["prior_mean"] = daily["prior_sum"] / daily["prior_count"].replace(0, np.nan)
    variance = (
        daily["prior_sumsq"] / daily["prior_count"].replace(0, np.nan)
        - np.square(daily["prior_mean"])
    )
    daily["prior_std"] = np.sqrt(variance.clip(lower=0))
    return daily[["season", "game_date", "prior_count", "prior_mean", "prior_std"]]


def _add_prior_league_zscores_returning(context: pd.DataFrame) -> pd.DataFrame:
    result = context.copy()
    zscore_sources = {
        "opp_off_all_corsi_ema5": "opp_off_all_corsi_ema5_prior_league_z",
        "team_def_all_corsi_against_ema5": (
            "team_def_all_corsi_against_ema5_prior_league_z"
        ),
        "combined_all_corsi_ema5": "combined_all_corsi_ema5_prior_league_z",
        "combined_all_xg_ema5": "combined_all_xg_ema5_prior_league_z",
    }
    for source_col, z_col in zscore_sources.items():
        stats = _prior_league_stats_by_date(result, source_col)
        result = result.merge(stats, on=["season", "game_date"], how="left")
        result[z_col] = _safe_divide(result[source_col] - result["prior_mean"], result["prior_std"])
        result = result.drop(columns=["prior_count", "prior_mean", "prior_std"])
    return result


def _load_inputs(
    clean_path: Path,
    team_games_path: Path,
    goalie_games_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = pd.read_parquet(clean_path)
    team_games = pd.read_parquet(team_games_path)
    goalie_games = pd.read_parquet(goalie_games_path)
    _require_columns(clean, CLEAN_REQUIRED_COLUMNS, "clean_training_data")
    clean = clean.copy()
    clean["game_date"] = pd.to_datetime(clean["game_date"])
    clean["game_id"] = clean["game_id"].astype(int)
    clean["goalie_id"] = clean["goalie_id"].astype(int)
    return clean, team_games, goalie_games


def build_pace_features(
    clean_path: Path = DEFAULT_CLEAN,
    team_games_path: Path = DEFAULT_TEAM_GAMES,
    goalie_games_path: Path = DEFAULT_GOALIE_GAMES,
) -> tuple[pd.DataFrame, dict]:
    """Return the pace feature artifact and metadata."""
    clean, team_games, goalie_games = _load_inputs(
        clean_path, team_games_path, goalie_games_path
    )

    team_features = build_team_pregame_features(team_games)
    goalie_features = build_goalie_workload_features(goalie_games)

    context = clean[KEY_COLUMNS + ["season"]].copy()

    team_projection = _team_defense_projection(team_features)
    opponent_projection = _team_offense_projection(team_features)
    context = context.merge(team_projection, on=["game_id", "team_abbrev"], how="left")
    context = context.merge(opponent_projection, on=["game_id", "opponent_team"], how="left")
    context = context.merge(goalie_features, on=["game_id", "goalie_id"], how="left")

    for col in [
        "pace_team_all_matched",
        "pace_opponent_all_matched",
        "pace_goalie_all_matched",
    ]:
        context[col] = context[col].eq(True)

    add_combined_pace_features(context)
    context = _add_prior_league_zscores_returning(context)

    feature_cols = feature_columns()
    missing_features = [col for col in feature_cols if col not in context.columns]
    if missing_features:
        raise ValueError(f"Feature builder did not generate columns: {missing_features}")

    context = context[KEY_COLUMNS + VERIFICATION_COLUMNS + feature_cols].copy()
    context = context.sort_values(["game_date", "game_id", "team_abbrev", "goalie_id"])
    context = context.reset_index(drop=True)

    if context.duplicated(KEY_COLUMNS).any():
        dupes = context.loc[context.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
        raise ValueError(f"Output key is not unique. Example duplicates:\n{dupes.head(10)}")

    metadata = build_metadata(
        clean_path=clean_path,
        team_games_path=team_games_path,
        goalie_games_path=goalie_games_path,
        context=context,
        clean_rows=len(clean),
        team_rows=len(team_games),
        goalie_rows=len(goalie_games),
    )
    return context, metadata


def null_rates_by_season_month(
    context: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, dict[str, float]]:
    work = context[["game_date", *feature_cols]].copy()
    work["season_month"] = pd.to_datetime(work["game_date"]).dt.to_period("M").astype(str)
    rates = work.groupby("season_month")[feature_cols].apply(lambda g: g.isna().mean())
    return {
        str(month): {col: float(value) for col, value in row.items()}
        for month, row in rates.iterrows()
    }


def build_metadata(
    clean_path: Path,
    team_games_path: Path,
    goalie_games_path: Path,
    context: pd.DataFrame,
    clean_rows: int,
    team_rows: int,
    goalie_rows: int,
) -> dict:
    feature_cols = feature_columns()
    null_counts = {
        col: int(count)
        for col, count in context[feature_cols].isna().sum().items()
        if int(count) > 0
    }
    coverage = {
        "team_all_match_rate": float(context["pace_team_all_matched"].mean()),
        "opponent_all_match_rate": float(context["pace_opponent_all_matched"].mean()),
        "goalie_all_match_rate": float(context["pace_goalie_all_matched"].mean()),
        "team_all_missing_rows": int((~context["pace_team_all_matched"]).sum()),
        "opponent_all_missing_rows": int((~context["pace_opponent_all_matched"]).sum()),
        "goalie_all_missing_rows": int((~context["pace_goalie_all_matched"]).sum()),
    }
    return {
        "artifact": "pace_features",
        "input_paths": {
            "clean": str(clean_path),
            "team_games": str(team_games_path),
            "goalie_games": str(goalie_games_path),
        },
        "row_count": int(len(context)),
        "clean_input_row_count": int(clean_rows),
        "team_games_input_row_count": int(team_rows),
        "goalie_games_input_row_count": int(goalie_rows),
        "key_columns": KEY_COLUMNS,
        "verification_columns": VERIFICATION_COLUMNS,
        "generated_columns": feature_cols,
        "feature_count": len(feature_cols),
        "family_columns": FAMILY_COLUMNS,
        "coverage": coverage,
        "null_counts": null_counts,
        "null_rates_by_season_month": null_rates_by_season_month(context, feature_cols),
        "leakage_notes": [
            "All team and goalie aggregates use regular-season MoneyPuck rows only.",
            "Rolling, EMA, and season-to-date features are shifted within season/team, season/opponent, or season/goalie groups so the current game is excluded.",
            "Prior-season baselines use the previous regular season only; Utah uses Arizona as its franchise predecessor for that baseline.",
            "League z-scores use feature distributions from earlier game dates in the same season, never same-date or future games.",
            "Early-season nulls are preserved for XGBoost rather than imputed.",
        ],
    }
