#!/usr/bin/env python3
"""Experiment 12: W1 cross-market coherence, with a hard season firewall.

This assignment implements the development stage only. The confirmation CLI
entry point is intentionally locked before any data access; its persisted guard
contract describes the one-touch protocol to use when confirmation is later
authorized explicitly.

Development usage:
    python scripts/experiment_12_w1_cross_market.py --stage development

The runner never imports an existing all-season loader. Every parquet read is a
projected pyarrow.dataset scan with a predicate applied before ``to_table()``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.optimize import brentq
from scipy.stats import poisson
from sklearn.linear_model import HuberRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "models" / "trained"
RUN_PREFIX = "experiment_12_w1_cross_market_"

CORE_SNAPSHOTS = REPO_ROOT / "data" / "processed" / "core_bettime_202607_snapshots.parquet"
SAVES_SNAPSHOTS = REPO_ROOT / "data" / "processed" / "saves_lines_snapshots.parquet"
MARKET_FEATURES = REPO_ROOT / "data" / "processed" / "market_game_features.parquet"
OUTCOMES = REPO_ROOT / "data" / "processed" / "clean_training_data.parquet"
PBP_DIR = REPO_ROOT / "data" / "raw" / "play_by_play"

DEVELOPMENT_SEASON = "2023-24"
DEVELOPMENT_SEASON_CODE = 20232024
DEVELOPMENT_START = "2023-10-10"
TRAIN_END = "2024-02-15"
VALIDATION_START = "2024-02-16"
DEVELOPMENT_END = "2024-04-18"
THRESHOLDS = (0.03, 0.05, 0.07, 0.10, 0.12)
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
MIN_SELECTED = 100
MAX_EMPTY_RATE = 0.01
DFS_BOOKS = frozenset({"underdog", "prizepicks"})
EXPECTED_SOG_BOOKS = frozenset(
    {"betmgm", "betonlineag", "bovada", "draftkings", "fanduel", "williamhill_us"}
)

TEAM_NAME_TO_ABBREV = {
    "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF", "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL", "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA", "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL", "Montréal Canadiens": "MTL", "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD", "New York Islanders": "NYI", "New York Rangers": "NYR",
    "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS", "Seattle Kraken": "SEA", "St Louis Blues": "STL",
    "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}

CORE_COLUMNS = [
    "season", "event_id", "requested_ts", "true_commence_time", "effective_gap_minutes",
    "game_date_eastern", "home_team", "away_team", "book_key", "market_key",
    "player_name_raw", "side", "line", "price_decimal",
]
SAVES_COLUMNS = [
    "event_id", "commence_time", "game_date_eastern", "home_team", "away_team",
    "requested_ts", "snapshot_pass", "book", "goalie_name_raw", "side", "line",
    "price_decimal", "goalie_id", "goalie_name_matched",
]
MARKET_COLUMNS = [
    "event_id", "game_date_eastern", "home_team", "away_team", "home_abbrev", "away_abbrev",
    "requested_ts", "book", "market", "outcome_label", "point", "price_decimal",
    "implied_prob_devig",
]
OUTCOME_COLUMNS = [
    "game_id", "game_date", "season", "goalie_id", "goalie_name", "team_abbrev",
    "opponent_team", "is_home", "saves", "team_shots",
]


class ConstructionStopped(RuntimeError):
    """Fail-closed construction error."""


@dataclass
class ReadRecord:
    input_name: str
    path: str
    columns: list[str]
    predicate: str
    predicate_applied_before_materialization: bool
    rows_loaded: int
    min_date_loaded: str | None
    max_date_loaded: str | None
    loaded_seasons: list[str]
    source_size_bytes: int
    source_mtime_utc: str
    source_schema_sha256: str
    filtered_content_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_frame_hash(frame: pd.DataFrame) -> str:
    """Hash only rows returned by a development predicate, never the source file."""
    normalized = frame.copy()
    for col in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[col]):
            normalized[col] = normalized[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    sort_cols = list(normalized.columns)
    if sort_cols and len(normalized):
        normalized = normalized.sort_values(sort_cols, kind="mergesort", na_position="first")
    payload = normalized.to_csv(
        index=False, lineterminator="\n", na_rep="<NA>", float_format="%.17g"
    ).encode("utf-8")
    return sha256_bytes(payload)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def make_logger(run_dir: Path) -> Callable[[str], None]:
    log_path = run_dir / "run_log.txt"

    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def schema_hash(path: Path) -> str:
    # Parquet footer/schema metadata only; no data column is materialized.
    return sha256_bytes(str(pq.ParquetFile(path).schema_arrow).encode("utf-8"))


def predicate_scan(
    name: str,
    path: Path,
    columns: list[str],
    predicate: ds.Expression,
    predicate_text: str,
    date_column: str,
    read_records: list[ReadRecord],
    season_column: str | None = None,
) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=columns, filter=predicate, use_threads=False)
    frame = scanner.to_table().to_pandas()
    if frame.empty:
        min_date = max_date = None
    else:
        dates = pd.to_datetime(frame[date_column], errors="raise")
        min_date = dates.min().date().isoformat()
        max_date = dates.max().date().isoformat()
    seasons = [] if season_column is None else sorted(frame[season_column].dropna().astype(str).unique())
    stat = path.stat()
    read_records.append(
        ReadRecord(
            input_name=name,
            path=str(path.relative_to(REPO_ROOT)),
            columns=columns,
            predicate=predicate_text,
            predicate_applied_before_materialization=True,
            rows_loaded=int(len(frame)),
            min_date_loaded=min_date,
            max_date_loaded=max_date,
            loaded_seasons=seasons,
            source_size_bytes=int(stat.st_size),
            source_mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            source_schema_sha256=schema_hash(path),
            filtered_content_sha256=canonical_frame_hash(frame),
        )
    )
    return frame


def assert_development_read(frame: pd.DataFrame, date_column: str, name: str) -> None:
    if frame.empty:
        raise ConstructionStopped(f"{name}: predicate returned zero development rows")
    dates = pd.to_datetime(frame[date_column], errors="raise")
    if dates.min() < pd.Timestamp(DEVELOPMENT_START) or dates.max() > pd.Timestamp(DEVELOPMENT_END):
        raise ConstructionStopped(
            f"{name}: development date assertion failed: {dates.min()}..{dates.max()}"
        )
    if "season" in frame and set(frame["season"].astype(str)) not in (
        {DEVELOPMENT_SEASON}, {str(DEVELOPMENT_SEASON_CODE)}
    ):
        raise ConstructionStopped(f"{name}: non-development season materialized")


def load_development_inputs(read_records: list[ReadRecord]) -> tuple[pd.DataFrame, ...]:
    core = predicate_scan(
        "core_sog_development", CORE_SNAPSHOTS, CORE_COLUMNS,
        (ds.field("season") == DEVELOPMENT_SEASON)
        & (ds.field("market_key") == "player_shots_on_goal")
        & (ds.field("game_date_eastern") >= DEVELOPMENT_START)
        & (ds.field("game_date_eastern") <= DEVELOPMENT_END),
        "season == '2023-24' AND market_key == 'player_shots_on_goal' AND "
        "2023-10-10 <= game_date_eastern <= 2024-04-18",
        "game_date_eastern", read_records, "season",
    )
    saves = predicate_scan(
        "saves_bettime_development", SAVES_SNAPSHOTS, SAVES_COLUMNS,
        (ds.field("game_date_eastern") >= DEVELOPMENT_START)
        & (ds.field("game_date_eastern") <= DEVELOPMENT_END)
        & (ds.field("snapshot_pass") == "bettime"),
        "2023-10-10 <= game_date_eastern <= 2024-04-18 AND snapshot_pass == 'bettime'",
        "game_date_eastern", read_records,
    )
    market = predicate_scan(
        "market_state_development", MARKET_FEATURES, MARKET_COLUMNS,
        (ds.field("game_date_eastern") >= DEVELOPMENT_START)
        & (ds.field("game_date_eastern") <= DEVELOPMENT_END),
        "2023-10-10 <= game_date_eastern <= 2024-04-18",
        "game_date_eastern", read_records,
    )
    outcomes = predicate_scan(
        "starter_outcomes_development", OUTCOMES, OUTCOME_COLUMNS,
        (ds.field("game_date") >= pd.Timestamp(DEVELOPMENT_START))
        & (ds.field("game_date") <= pd.Timestamp(DEVELOPMENT_END))
        & (ds.field("season") == DEVELOPMENT_SEASON_CODE),
        "2023-10-10 <= game_date <= 2024-04-18 AND season == 20232024",
        "game_date", read_records, "season",
    )
    for name, frame, date_col in (
        ("core SOG", core, "game_date_eastern"), ("saves", saves, "game_date_eastern"),
        ("market", market, "game_date_eastern"), ("outcomes", outcomes, "game_date"),
    ):
        assert_development_read(frame, date_col, name)
    if set(core["book_key"]) & DFS_BOOKS:
        raise ConstructionStopped("DFS book unexpectedly survived the sportsbook SOG source predicate")
    unexpected_books = set(core["book_key"]) - EXPECTED_SOG_BOOKS
    if unexpected_books:
        raise ConstructionStopped(f"unexpected 2023-24 SOG books: {sorted(unexpected_books)}")
    if (core["effective_gap_minutes"] < 10.0).any():
        raise ConstructionStopped("core development parquet contains an event inside the 10-minute drift exclusion")
    return core, saves, market, outcomes


def normalize_name(value: str) -> str:
    value = str(value).split("(", 1)[0].strip().replace("'", "").replace("’", "")
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in folded)
    return " ".join(part for part in cleaned.split() if part not in {"jr", "sr", "ii", "iii", "iv"})


def person_key(value: str) -> tuple[str, str]:
    parts = normalize_name(value).split()
    return ((parts[0][0] if parts else ""), (parts[-1] if parts else ""))


def build_nhl_games(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game_id, group in outcomes.groupby("game_id", sort=True):
        dates = pd.to_datetime(group["game_date"]).dt.date.astype(str).unique()
        home = group.loc[group["is_home"] == 1, "team_abbrev"].unique()
        away = group.loc[group["is_home"] == 0, "team_abbrev"].unique()
        if len(dates) != 1 or len(home) != 1 or len(away) != 1:
            raise ConstructionStopped(f"ambiguous NHL game metadata for game_id={game_id}")
        rows.append(
            {"game_id": int(game_id), "game_date": dates[0], "home_abbrev": home[0],
             "away_abbrev": away[0], "team_key": "|".join(sorted((home[0], away[0])))}
        )
    games = pd.DataFrame(rows)
    dup = games.duplicated(["game_date", "team_key"], keep=False)
    if dup.any():
        raise ConstructionStopped(
            "NHL date+unordered-team key is ambiguous: "
            + games.loc[dup].to_json(orient="records")
        )
    return games


def event_metadata(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    needed = ["event_id", "game_date_eastern", "home_team", "away_team"]
    meta = frame[needed].drop_duplicates()
    counts = meta.groupby("event_id").size()
    if (counts > 1).any():
        raise ConstructionStopped(f"{source}: event id has conflicting date/team metadata")
    meta = meta.rename(columns={"event_id": f"{source}_event_id", "game_date_eastern": "game_date"})
    try:
        meta["home_abbrev"] = meta["home_team"].map(TEAM_NAME_TO_ABBREV)
        meta["away_abbrev"] = meta["away_team"].map(TEAM_NAME_TO_ABBREV)
    except Exception as exc:
        raise ConstructionStopped(f"{source}: team normalization failed") from exc
    if meta[["home_abbrev", "away_abbrev"]].isna().any().any():
        unknown = sorted(set(meta.loc[meta["home_abbrev"].isna(), "home_team"]) |
                         set(meta.loc[meta["away_abbrev"].isna(), "away_team"]))
        raise ConstructionStopped(f"{source}: unknown team names: {unknown}")
    meta["team_key"] = meta.apply(
        lambda row: "|".join(sorted((row["home_abbrev"], row["away_abbrev"]))), axis=1
    )
    return meta


def map_events_to_games(frame: pd.DataFrame, games: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = event_metadata(frame, source)
    mapped = meta.merge(
        games[["game_id", "game_date", "team_key"]], on=["game_date", "team_key"],
        how="left", validate="many_to_one", indicator=True,
    )
    matched = mapped[mapped["_merge"] == "both"].drop(columns="_merge")
    unmatched = mapped[mapped["_merge"] != "both"].drop(columns="_merge")
    duplicate_games = matched.duplicated("game_id", keep=False)
    if duplicate_games.any():
        raise ConstructionStopped(
            f"{source}: multiple Odds API events map to one NHL game: "
            + matched.loc[duplicate_games].to_json(orient="records")
        )
    return matched, unmatched


def load_season_roster(
    outcomes: pd.DataFrame, log: Callable[[str], None]
) -> tuple[dict[str, set[str]], dict[tuple[str, str], set[str]], dict[str, Any]]:
    by_name: dict[str, set[str]] = defaultdict(set)
    by_person: dict[tuple[str, str], set[str]] = defaultdict(set)
    game_ids = sorted(set(outcomes["game_id"].astype(int)))
    pbp_hashes: list[dict[str, Any]] = []
    max_loaded_date = None
    for game_id in game_ids:
        if not str(game_id).startswith("2023"):
            raise ConstructionStopped(f"PBP firewall rejected non-2023 game id {game_id}")
        path = PBP_DIR / f"{game_id}.json"
        if not path.exists():
            raise ConstructionStopped(f"missing development PBP file: {path}")
        raw = path.read_bytes()
        payload = json.loads(raw)
        game_date = str(payload.get("gameDate"))
        if str(payload.get("season")) != str(DEVELOPMENT_SEASON_CODE):
            raise ConstructionStopped(f"PBP season assertion failed for {path.name}")
        if not (DEVELOPMENT_START <= game_date <= DEVELOPMENT_END):
            raise ConstructionStopped(f"PBP date assertion failed for {path.name}: {game_date}")
        max_loaded_date = max(max_loaded_date or game_date, game_date)
        teams = {
            int(payload["homeTeam"]["id"]): payload["homeTeam"]["abbrev"],
            int(payload["awayTeam"]["id"]): payload["awayTeam"]["abbrev"],
        }
        for player in payload.get("rosterSpots") or []:
            team = teams.get(int(player["teamId"]))
            if team is None:
                raise ConstructionStopped(f"unknown roster team in {path.name}")
            name = f"{player['firstName']['default']} {player['lastName']['default']}"
            norm = normalize_name(name)
            by_name[norm].add(team)
            by_person[person_key(name)].add(team)
        pbp_hashes.append({"game_id": game_id, "path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_bytes(raw)})
    aggregate_hash = sha256_bytes(
        "\n".join(f"{row['game_id']}:{row['sha256']}" for row in pbp_hashes).encode("ascii")
    )
    stats = {
        "files_loaded": len(pbp_hashes), "min_game_id": min(game_ids), "max_game_id": max(game_ids),
        "max_game_date_loaded": max_loaded_date, "season_assertion": DEVELOPMENT_SEASON_CODE,
        "file_manifest_sha256": aggregate_hash, "files": pbp_hashes,
    }
    log(f"Loaded {len(game_ids)} development-only PBP files; max date {max_loaded_date}")
    return by_name, by_person, stats


def pair_quotes(
    frame: pd.DataFrame,
    identity: list[str],
    book_col: str,
    entity_col: str,
    require_half_point: bool,
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    work = frame.copy()
    work["side_norm"] = work["side"].astype(str).str.lower()
    work = work[work["side_norm"].isin(["over", "under"])]
    missing_identity = work[identity].isna().any(axis=1)
    missing_identity_rows = int(missing_identity.sum())
    work = work[~missing_identity].copy()
    invalid_price = ~(work["price_decimal"] > 1.0)
    invalid_price_rows = int(invalid_price.sum())
    work = work[~invalid_price]
    exact_cols = identity + ["side_norm", "price_decimal"]
    before = len(work)
    work = work.drop_duplicates(exact_cols)
    exact_duplicate_rows = before - len(work)
    side_key = identity + ["side_norm"]
    conflicts = work.groupby(side_key, dropna=False)["price_decimal"].nunique()
    conflict_index = conflicts[conflicts > 1].index
    conflict_units = set(tuple(x) if isinstance(x, tuple) else (x,) for x in conflict_index)
    if conflict_units:
        mask = work[side_key].apply(tuple, axis=1).isin(conflict_units)
        conflict_rows = int(mask.sum())
        work = work[~mask]
    else:
        conflict_rows = 0
    side_counts = work.groupby(identity, dropna=False)["side_norm"].nunique()
    incomplete_units = int((side_counts != 2).sum())
    complete_keys = side_counts[side_counts == 2].reset_index()[identity]
    complete = work.merge(complete_keys, on=identity, how="inner", validate="many_to_many")
    piv = complete.pivot(index=identity, columns="side_norm", values="price_decimal").reset_index()
    piv.columns.name = None
    if piv.duplicated(identity).any():
        raise ConstructionStopped(f"pairing produced duplicate units for {entity_col}/{book_col}")
    twice = np.rint(piv["line"].astype(float) * 2).astype(int)
    is_half = np.isclose(piv["line"].astype(float) * 2, twice, atol=1e-9) & (twice % 2 == 1)
    integer_or_other = piv[~is_half].copy()
    excluded_integer_pairs = int(len(integer_or_other)) if require_half_point else 0
    if require_half_point:
        piv = piv[is_half].copy()
    over_raw = 1.0 / piv["over"]
    under_raw = 1.0 / piv["under"]
    piv["fair_prob_over"] = over_raw / (over_raw + under_raw)
    piv["fair_prob_under"] = 1.0 - piv["fair_prob_over"]
    counts = {
        "input_rows": int(len(frame)), "missing_identity_rows": missing_identity_rows,
        "invalid_price_rows": invalid_price_rows,
        "exact_duplicate_extra_rows": int(exact_duplicate_rows),
        "conflicting_price_rows_excluded": conflict_rows,
        "conflicting_side_groups_excluded": int(len(conflict_units)),
        "incomplete_two_sided_units": incomplete_units,
        "exact_paired_units_before_line_filter": int(len(piv) + excluded_integer_pairs),
        "excluded_non_half_point_pairs": excluded_integer_pairs,
        "excluded_non_half_point_quote_rows": excluded_integer_pairs * 2,
        "paired_units_retained": int(len(piv)),
    }
    return piv, counts, integer_or_other


def poisson_mean_from_over_probability(line: float, fair_over: float) -> float:
    cutoff = int(math.floor(line))
    target = float(fair_over)
    if not 0.0 < target < 1.0:
        raise ValueError("fair probability is outside (0, 1)")
    objective = lambda mean: float(poisson.sf(cutoff, mean) - target)
    return float(brentq(objective, 1e-10, 100.0, xtol=1e-12, rtol=1e-12, maxiter=200))


def attribute_sog_players(
    pairs: pd.DataFrame,
    mapping: pd.DataFrame,
    roster_by_name: dict[str, set[str]],
    roster_by_person: dict[tuple[str, str], set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    map_cols = ["core_event_id", "game_id", "game_date", "home_abbrev", "away_abbrev"]
    work = pairs.merge(mapping[map_cols], left_on="event_id", right_on="core_event_id", how="left", validate="many_to_one")
    rows = []
    for row in work.itertuples(index=False):
        event_teams = {row.home_abbrev, row.away_abbrev} if pd.notna(row.game_id) else set()
        exact = roster_by_name.get(normalize_name(row.player_name_raw), set()) & event_teams
        fallback = set()
        method = "normalized_full_name"
        teams = exact
        if not teams:
            fallback = roster_by_person.get(person_key(row.player_name_raw), set()) & event_teams
            teams = fallback
            method = "first_initial_last_name"
        status = "resolved" if len(teams) == 1 else ("ambiguous" if len(teams) > 1 else "unresolved")
        rows.append(
            {"event_id": row.event_id, "game_id": row.game_id, "game_date": row.game_date,
             "book": row.book_key, "player_name_raw": row.player_name_raw, "line": row.line,
             "price_over_decimal": row.over, "price_under_decimal": row.under,
             "fair_prob_over": row.fair_prob_over, "fair_prob_under": row.fair_prob_under,
             "home_abbrev": row.home_abbrev, "away_abbrev": row.away_abbrev,
             "assigned_team": next(iter(teams)) if len(teams) == 1 else None,
             "attribution_status": status, "attribution_method": method if teams else None}
        )
    attributed = pd.DataFrame(rows)
    resolved = attributed[attributed["attribution_status"] == "resolved"].copy()
    means = []
    failures = []
    inversion_cache: dict[tuple[float, float], float] = {}
    for row in resolved.itertuples(index=False):
        try:
            cache_key = (float(row.line), float(row.fair_prob_over))
            if cache_key not in inversion_cache:
                inversion_cache[cache_key] = poisson_mean_from_over_probability(*cache_key)
            means.append(inversion_cache[cache_key])
            failures.append(None)
        except Exception as exc:
            means.append(np.nan)
            failures.append(str(exc))
    resolved["implied_player_mean"] = means
    resolved["inversion_failure"] = failures
    attributed = attributed.merge(
        resolved[["event_id", "book", "player_name_raw", "line", "implied_player_mean", "inversion_failure"]],
        on=["event_id", "book", "player_name_raw", "line"], how="left", validate="one_to_one",
    )
    return attributed, resolved[resolved["implied_player_mean"].notna()].copy()


def build_team_projections(resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_book = (
        resolved.groupby(["event_id", "game_id", "game_date", "assigned_team", "book"], as_index=False)
        .agg(aggregate_player_mean=("implied_player_mean", "sum"), listed_player_count=("player_name_raw", "nunique"))
    )
    cross_book = (
        per_book.groupby(["event_id", "game_id", "game_date", "assigned_team"], as_index=False)
        .agg(
            aggregate_player_mean=("aggregate_player_mean", "median"),
            listed_player_count=("listed_player_count", "median"),
            n_sog_books=("book", "nunique"),
        )
        .rename(columns={"event_id": "sog_event_id", "assigned_team": "shooting_team"})
    )
    return per_book, cross_book


def actual_team_sog(outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = ["game_id", "team_abbrev", "team_shots"]
    actual = outcomes[columns].drop_duplicates()
    if actual.duplicated(["game_id", "team_abbrev"]).any():
        raise ConstructionStopped("multiple actual team-SOG values for one game-team")
    return actual.rename(columns={"team_abbrev": "shooting_team", "team_shots": "actual_team_sog"})


def fit_huber(features: pd.DataFrame, target: pd.Series) -> tuple[HuberRegressor, dict[str, Any]]:
    model = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=2000, tol=1e-8)
    model.fit(features.to_numpy(dtype=float), target.to_numpy(dtype=float))
    prediction = model.predict(features.to_numpy(dtype=float))
    params = {
        "implementation": "sklearn.linear_model.HuberRegressor",
        "epsilon": 1.35, "alpha": 0.0, "max_iter": 2000, "tol": 1e-8,
        "feature_names": list(features.columns), "intercept": float(model.intercept_),
        "coefficients": [float(value) for value in model.coef_], "scale": float(model.scale_),
        "n_iter": int(model.n_iter_), "n_rows": int(len(features)),
        "mae_in_sample": float(np.mean(np.abs(target.to_numpy() - prediction))),
    }
    return model, params


def build_market_goal_projection(
    market: pd.DataFrame,
    market_mapping: pd.DataFrame,
    core_mapping: pd.DataFrame,
    core: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    anchors = core.groupby("event_id")["requested_ts"].nunique()
    if (anchors != 1).any():
        raise ConstructionStopped("SOG core event has multiple W1 anchors")
    anchor_frame = core.groupby("event_id", as_index=False)["requested_ts"].first().rename(
        columns={"event_id": "core_event_id", "requested_ts": "w1_anchor_ts"}
    )
    games = core_mapping[["core_event_id", "game_id"]].merge(anchor_frame, on="core_event_id", validate="one_to_one")
    mm = market_mapping[["market_event_id", "game_id"]]
    work = market.merge(mm, left_on="event_id", right_on="market_event_id", how="inner", validate="many_to_one")
    work = work.merge(games, on="game_id", how="inner", validate="many_to_one")
    work["requested_dt"] = pd.to_datetime(work["requested_ts"], utc=True)
    work["anchor_dt"] = pd.to_datetime(work["w1_anchor_ts"], utc=True)
    forbidden_later_rows = int((work["requested_dt"] > work["anchor_dt"]).sum())
    safe = work[work["requested_dt"] <= work["anchor_dt"]].copy()
    latest_ts = safe.groupby("game_id")["requested_dt"].transform("max")
    latest = safe[safe["requested_dt"] == latest_ts].copy()
    if (latest["requested_dt"] > latest["anchor_dt"]).any():
        raise ConstructionStopped("market timing firewall failed")
    h2h = latest[latest["market"] == "h2h"].copy()
    h2h = h2h[h2h["outcome_label"] == h2h["home_abbrev"]]
    h2h_conflict = h2h.groupby(["game_id", "book"])["implied_prob_devig"].nunique()
    bad_h2h = set(h2h_conflict[h2h_conflict > 1].index)
    if bad_h2h:
        h2h = h2h[~h2h[["game_id", "book"]].apply(tuple, axis=1).isin(bad_h2h)]
    home_prob = h2h.drop_duplicates(["game_id", "book", "implied_prob_devig"]).groupby("game_id").agg(
        home_win_prob_devigged=("implied_prob_devig", "mean"), n_h2h_books=("book", "nunique")
    )
    totals = latest[latest["market"] == "totals"].copy()
    totals_points = totals[["game_id", "book", "point"]].drop_duplicates()
    totals_conflict = totals_points.groupby(["game_id", "book"])["point"].nunique()
    bad_totals = set(totals_conflict[totals_conflict > 1].index)
    if bad_totals:
        totals_points = totals_points[~totals_points[["game_id", "book"]].apply(tuple, axis=1).isin(bad_totals)]
    total_summary = totals_points.groupby("game_id").agg(
        consensus_total=("point", "median"), n_totals_books=("book", "nunique")
    )
    meta = latest.groupby("game_id", as_index=True).agg(
        market_event_id=("market_event_id", "first"), core_event_id=("core_event_id", "first"),
        game_date=("game_date_eastern", "first"), home_abbrev=("home_abbrev", "first"),
        away_abbrev=("away_abbrev", "first"), market_requested_ts=("requested_ts", "first"),
        w1_anchor_ts=("w1_anchor_ts", "first"),
    )
    projection = meta.join(home_prob).join(total_summary).reset_index()
    projection["home_expected_goals"] = projection["consensus_total"] * projection["home_win_prob_devigged"]
    projection["away_expected_goals"] = projection["consensus_total"] * (1.0 - projection["home_win_prob_devigged"])
    counts = {
        "rows_later_than_w1_anchor_excluded": forbidden_later_rows,
        "games_with_timing_safe_snapshot": int(latest["game_id"].nunique()),
        "h2h_book_conflict_groups_excluded": len(bad_h2h),
        "totals_book_conflict_groups_excluded": len(bad_totals),
        "games_with_complete_goals_projection": int(
            projection[["home_win_prob_devigged", "consensus_total"]].notna().all(axis=1).sum()
        ),
    }
    return projection, counts


def predict_coverage(model: HuberRegressor, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(frame[["aggregate_player_mean", "listed_player_count"]].to_numpy(dtype=float))


def build_translation_rows(
    outcomes: pd.DataFrame,
    team_projection: pd.DataFrame,
    market_goals: pd.DataFrame,
    coverage_model: HuberRegressor,
) -> pd.DataFrame:
    tp = team_projection.copy()
    tp["projected_team_sog"] = predict_coverage(coverage_model, tp)
    rows = outcomes.merge(
        tp[["game_id", "shooting_team", "sog_event_id", "projected_team_sog", "n_sog_books"]],
        left_on=["game_id", "opponent_team"], right_on=["game_id", "shooting_team"],
        how="left", validate="many_to_one",
    )
    mg = market_goals[[
        "game_id", "market_event_id", "market_requested_ts", "w1_anchor_ts", "home_abbrev",
        "away_abbrev", "home_expected_goals", "away_expected_goals",
    ]]
    rows = rows.merge(mg, on="game_id", how="left", validate="many_to_one")
    rows["market_opponent_goals"] = np.where(
        rows["is_home"] == 1, rows["away_expected_goals"], rows["home_expected_goals"]
    )
    rows["coherence_raw_saves"] = rows["projected_team_sog"] - rows["market_opponent_goals"]
    return rows


def empirical_probabilities(predicted_saves: np.ndarray, lines: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, ...]:
    realized = predicted_saves[:, None] + residuals[None, :]
    line_grid = lines[:, None]
    p_over = np.mean(realized > line_grid, axis=1)
    p_under = np.mean(realized < line_grid, axis=1)
    p_push = np.mean(np.isclose(realized, line_grid, atol=1e-12), axis=1)
    denominator = p_over + p_under
    conditional_over = np.divide(p_over, denominator, out=np.full_like(p_over, np.nan), where=denominator > 0)
    return p_over, p_under, p_push, conditional_over


def price_saves_universe(
    saves_pairs: pd.DataFrame,
    saves_mapping: pd.DataFrame,
    outcomes: pd.DataFrame,
    translation_rows: pd.DataFrame,
    translation_model: HuberRegressor,
    residuals: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    mapping = saves_mapping[["saves_event_id", "game_id"]]
    priced = saves_pairs.merge(mapping, left_on="event_id", right_on="saves_event_id", how="left", validate="many_to_one")
    outcome_cols = ["game_id", "goalie_id", "game_date", "goalie_name", "team_abbrev", "opponent_team", "saves"]
    priced = priced.merge(outcomes[outcome_cols], on=["game_id", "goalie_id"], how="left", validate="many_to_one", indicator="_goalie_match")
    goalie_unmatched = int((priced["_goalie_match"] != "both").sum())
    priced = priced[priced["_goalie_match"] == "both"].drop(columns="_goalie_match")
    tr_cols = [
        "game_id", "goalie_id", "sog_event_id", "market_event_id", "market_requested_ts",
        "w1_anchor_ts", "projected_team_sog", "market_opponent_goals", "coherence_raw_saves",
    ]
    priced = priced.merge(translation_rows[tr_cols], on=["game_id", "goalie_id"], how="left", validate="many_to_one")
    complete = priced[["coherence_raw_saves", "sog_event_id", "market_event_id"]].notna().all(axis=1)
    missing_projection = int((~complete).sum())
    priced = priced[complete].copy()
    priced["predicted_starter_saves"] = translation_model.predict(priced[["coherence_raw_saves"]].to_numpy(dtype=float))
    p_over, p_under, p_push, p_cond = empirical_probabilities(
        priced["predicted_starter_saves"].to_numpy(float), priced["line"].to_numpy(float), residuals
    )
    priced["model_prob_over"] = p_over
    priced["model_prob_under"] = p_under
    priced["model_prob_push"] = p_push
    priced["model_prob_over_conditional"] = p_cond
    priced["probability_gap_over"] = priced["model_prob_over_conditional"] - priced["fair_prob_over"]
    priced["cluster_id"] = priced["game_id"].astype(str) + ":" + priced["goalie_id"].astype(str)
    priced["split"] = np.where(
        pd.to_datetime(priced["game_date"]) <= pd.Timestamp(TRAIN_END), "development_train", "development_validation"
    )
    priced["grade"] = np.sign(priced["saves"] - priced["line"]).astype(int)
    priced["profit_over"] = np.where(priced["grade"] > 0, priced["over"] - 1.0, np.where(priced["grade"] < 0, -1.0, np.nan))
    priced["profit_under"] = np.where(priced["grade"] < 0, priced["under"] - 1.0, np.where(priced["grade"] > 0, -1.0, np.nan))
    counts = {
        "paired_saves_units": int(len(saves_pairs)), "unmatched_goalie_quote_units": goalie_unmatched,
        "quote_units_missing_sog_or_market_projection": missing_projection,
        "eligible_quote_units": int(len(priced)), "eligible_goalie_nights": int(priced["cluster_id"].nunique()),
        "eligible_push_quote_units": int((priced["grade"] == 0).sum()),
    }
    return priced, counts


def paired_cluster_bootstrap(
    universe: pd.DataFrame, side: str, threshold: float
) -> dict[str, Any]:
    profit_col = f"profit_{side}"
    nonpush = universe[universe[profit_col].notna()].copy()
    gap = nonpush["probability_gap_over"] if side == "over" else -nonpush["probability_gap_over"]
    nonpush["selected"] = gap >= threshold
    clusters = sorted(nonpush["cluster_id"].unique())
    blind = nonpush.groupby("cluster_id")[profit_col].agg(["sum", "count"]).reindex(clusters, fill_value=0)
    model = nonpush[nonpush["selected"]].groupby("cluster_id")[profit_col].agg(["sum", "count"]).reindex(clusters, fill_value=0)
    blind_profit, blind_count = blind["sum"].to_numpy(float), blind["count"].to_numpy(float)
    model_profit, model_count = model["sum"].to_numpy(float), model["count"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.full(N_BOOTSTRAP, np.nan)
    empty = 0
    batch_size = 250
    for start in range(0, N_BOOTSTRAP, batch_size):
        size = min(batch_size, N_BOOTSTRAP - start)
        sampled = rng.integers(0, len(clusters), size=(size, len(clusters)))
        mp = model_profit[sampled].sum(axis=1)
        mn = model_count[sampled].sum(axis=1)
        bp = blind_profit[sampled].sum(axis=1)
        bn = blind_count[sampled].sum(axis=1)
        valid = (mn > 0) & (bn > 0)
        empty += int((~valid).sum())
        deltas[start:start + size][valid] = mp[valid] / mn[valid] - bp[valid] / bn[valid]
    valid_delta = deltas[np.isfinite(deltas)]
    selected = nonpush[nonpush["selected"]]
    model_roi = float(selected[profit_col].mean()) if len(selected) else None
    blind_roi = float(nonpush[profit_col].mean()) if len(nonpush) else None
    lower, upper = (np.quantile(valid_delta, [0.025, 0.975]) if len(valid_delta) else (np.nan, np.nan))
    return {
        "side": side.upper(), "threshold": threshold,
        "n_eligible_nonpush_quotes": int(len(nonpush)), "n_eligible_goalie_nights": int(len(clusters)),
        "n_selected_graded_bets": int(len(selected)), "n_selected_goalie_nights": int(selected["cluster_id"].nunique()),
        "model_roi": model_roi, "blind_same_side_roi": blind_roi,
        "model_minus_blind_roi_delta": None if model_roi is None else model_roi - blind_roi,
        "bootstrap_lower_ci95": None if not np.isfinite(lower) else float(lower),
        "bootstrap_upper_ci95": None if not np.isfinite(upper) else float(upper),
        "bootstrap_empty_model_arm_resamples": empty,
        "bootstrap_empty_model_arm_rate": empty / N_BOOTSTRAP,
        "selection_eligible": bool(len(selected) >= MIN_SELECTED and empty / N_BOOTSTRAP <= MAX_EMPTY_RATE),
    }


def select_threshold(grid: pd.DataFrame, side: str) -> dict[str, Any]:
    candidate = grid[(grid["side"] == side.upper()) & grid["selection_eligible"]].copy()
    if candidate.empty:
        return {
            "side": side.upper(), "threshold": 0.05, "development_status": "INSUFFICIENT",
            "selection_reason": "No grid row had >=100 graded selected bets and <=1% empty-arm resamples.",
        }
    candidate = candidate.sort_values(
        ["bootstrap_lower_ci95", "threshold"], ascending=[False, False], kind="mergesort"
    )
    winner = candidate.iloc[0]
    return {
        "side": side.upper(), "threshold": float(winner["threshold"]),
        "development_status": "QUALIFIED", "selection_reason": "Maximum bootstrap lower CI; larger threshold breaks ties.",
        "selected_grid_row": json_ready(winner.to_dict()),
    }


def existing_frozen_runs() -> list[Path]:
    frozen = []
    for metadata_path in sorted(OUTPUT_ROOT.glob(f"{RUN_PREFIX}*/metadata.json")):
        try:
            if json.loads(metadata_path.read_text(encoding="utf-8")).get("status") == "DEVELOPMENT_FROZEN":
                frozen.append(metadata_path.parent)
        except (OSError, json.JSONDecodeError):
            continue
    return frozen


def artifact_manifest(run_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name not in {"output_manifest.json"}:
            rows.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"generated_at": utc_now(), "files": rows}


def run_development() -> Path:
    prior = existing_frozen_runs()
    if prior:
        raise ConstructionStopped(f"completed development freeze already exists: {prior}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"{RUN_PREFIX}{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log = make_logger(run_dir)
    started_at = utc_now()
    code_hash = sha256_file(Path(__file__))
    log(f"Experiment 12 development started; code_sha256={code_hash}")
    read_records: list[ReadRecord] = []
    try:
        core, saves, market, outcomes = load_development_inputs(read_records)
        log("All predicate-level input scans passed development max-date and season assertions")
        games = build_nhl_games(outcomes)
        core_map, core_unmatched = map_events_to_games(core, games, "core")
        saves_map, saves_unmatched = map_events_to_games(saves, games, "saves")
        market_map, market_unmatched = map_events_to_games(market, games, "market")
        log(
            f"Event maps: core {len(core_map)} matched/{len(core_unmatched)} unmatched; "
            f"saves {len(saves_map)}/{len(saves_unmatched)}; market {len(market_map)}/{len(market_unmatched)}"
        )
        roster_by_name, roster_by_person, pbp_audit = load_season_roster(outcomes, log)

        sog_pairs, sog_pair_counts, excluded_sog_lines = pair_quotes(
            core, ["event_id", "book_key", "player_name_raw", "line"], "book_key", "player_name_raw", True
        )
        attributed, resolved = attribute_sog_players(sog_pairs, core_map, roster_by_name, roster_by_person)
        per_book_team, team_projection = build_team_projections(resolved)
        attr_counts = attributed["attribution_status"].value_counts().to_dict()
        log(f"SOG paired units {len(sog_pairs)}; attribution {attr_counts}; team rows {len(team_projection)}")

        actual_sog = actual_team_sog(outcomes)
        coverage_rows = team_projection.merge(
            actual_sog, on=["game_id", "shooting_team"], how="left", validate="one_to_one"
        )
        coverage_rows["split"] = np.where(
            pd.to_datetime(coverage_rows["game_date"]) <= pd.Timestamp(TRAIN_END),
            "development_train", "development_validation",
        )
        coverage_train = coverage_rows[
            (coverage_rows["split"] == "development_train") & coverage_rows["actual_team_sog"].notna()
        ].copy()
        coverage_model, coverage_params = fit_huber(
            coverage_train[["aggregate_player_mean", "listed_player_count"]], coverage_train["actual_team_sog"]
        )
        coverage_rows["projected_team_sog"] = predict_coverage(coverage_model, coverage_rows)

        market_goals, market_counts = build_market_goal_projection(market, market_map, core_map, core)
        translation_rows = build_translation_rows(outcomes, team_projection, market_goals, coverage_model)
        translation_rows["split"] = np.where(
            pd.to_datetime(translation_rows["game_date"]) <= pd.Timestamp(TRAIN_END),
            "development_train", "development_validation",
        )
        translation_train = translation_rows[
            (translation_rows["split"] == "development_train") & translation_rows["coherence_raw_saves"].notna()
        ].copy()
        translation_model, translation_params = fit_huber(
            translation_train[["coherence_raw_saves"]], translation_train["saves"]
        )
        translation_rows["predicted_starter_saves"] = np.nan
        complete_translation = translation_rows["coherence_raw_saves"].notna()
        translation_rows.loc[complete_translation, "predicted_starter_saves"] = translation_model.predict(
            translation_rows.loc[complete_translation, ["coherence_raw_saves"]].to_numpy(float)
        )
        train_prediction = translation_model.predict(translation_train[["coherence_raw_saves"]].to_numpy(float))
        residuals = translation_train["saves"].to_numpy(float) - train_prediction
        residual_frame = pd.DataFrame(
            {"game_id": translation_train["game_id"].astype(int), "goalie_id": translation_train["goalie_id"].astype(int),
             "residual": residuals}
        ).sort_values(["game_id", "goalie_id"])
        residuals = residual_frame["residual"].to_numpy(float)
        log(
            f"Fits frozen on train only: coverage n={len(coverage_train)}; "
            f"starter translation/residual n={len(translation_train)}"
        )

        saves_pairs, saves_pair_counts, _ = pair_quotes(
            saves, ["event_id", "goalie_id", "book", "line"], "book", "goalie_id", False
        )
        priced, pricing_counts = price_saves_universe(
            saves_pairs, saves_map, outcomes, translation_rows, translation_model, residuals
        )
        validation = priced[priced["split"] == "development_validation"].copy()
        grid_rows = [
            paired_cluster_bootstrap(validation, side, threshold)
            for side in ("over", "under") for threshold in THRESHOLDS
        ]
        grid = pd.DataFrame(grid_rows)
        selections = {side: select_threshold(grid, side) for side in ("over", "under")}
        log(f"Thresholds frozen: OVER={selections['over']['threshold']} ({selections['over']['development_status']}), "
            f"UNDER={selections['under']['threshold']} ({selections['under']['development_status']})")

        # Persist all development universes before writing the signed freeze.
        event_maps = pd.concat(
            [core_map.assign(source="core_sog"), saves_map.assign(source="saves_bettime"),
             market_map.assign(source="market_state")], ignore_index=True, sort=False
        )
        unmatched_events = pd.concat(
            [core_unmatched.assign(source="core_sog"), saves_unmatched.assign(source="saves_bettime"),
             market_unmatched.assign(source="market_state")], ignore_index=True, sort=False
        )
        event_maps.to_csv(run_dir / "development_event_mapping.csv", index=False)
        unmatched_events.to_csv(run_dir / "development_unmatched_events.csv", index=False)
        attributed.to_parquet(run_dir / "development_sog_player_universe.parquet", index=False)
        excluded_sog_lines.to_csv(run_dir / "development_excluded_sog_lines.csv", index=False)
        per_book_team.to_parquet(run_dir / "development_event_team_book_sog.parquet", index=False)
        coverage_rows.to_csv(run_dir / "development_coverage_fit_universe.csv", index=False)
        market_goals.to_csv(run_dir / "development_market_goal_universe.csv", index=False)
        translation_rows.to_csv(run_dir / "development_starter_translation_universe.csv", index=False)
        residual_frame.to_csv(run_dir / "development_train_residuals.csv", index=False)
        priced.to_parquet(run_dir / "development_saves_quote_universe.parquet", index=False)
        grid.to_csv(run_dir / "development_threshold_sweep.csv", index=False)

        input_hashes = {
            "hash_policy": (
                "Multi-season parquet source bytes are deliberately not fully hashed because doing so would read "
                "forbidden 2024-25 data pages. Each source records size, mtime, schema hash, and a canonical SHA-256 "
                "of only the rows materialized by the development predicate. Selected 2023-24 PBP files are fully hashed."
            ),
            "parquet_predicate_reads": [asdict(record) for record in read_records],
            "play_by_play_development_manifest": pbp_audit,
        }
        write_json(run_dir / "input_checksums.json", input_hashes)
        read_audit = {
            "stage": "development", "confirmation_stage_invoked": False,
            "forbidden_season_rows_materialized": 0,
            "logical_read_contract": "pyarrow.dataset scanner filter applied before to_table",
            "development_max_allowed_date": DEVELOPMENT_END,
            "all_max_date_assertions_passed": True,
            "all_season_assertions_passed": True,
            "reads": [asdict(record) for record in read_records],
            "pbp": {key: value for key, value in pbp_audit.items() if key != "files"},
            "network_calls": 0, "betting_db_accesses": 0, "api_credits_used": 0,
        }
        write_json(run_dir / "read_audit.json", read_audit)

        coverage_counts = {
            "event_mapping": {
                "core_matched": len(core_map), "core_unmatched": len(core_unmatched),
                "saves_matched": len(saves_map), "saves_unmatched": len(saves_unmatched),
                "market_matched": len(market_map), "market_unmatched": len(market_unmatched),
                "mapping_rule": "Eastern game date plus unordered normalized team abbreviations; fail on ambiguity.",
            },
            "sog_pairing": sog_pair_counts,
            "sog_attribution": {
                "resolved": int(attr_counts.get("resolved", 0)),
                "unresolved": int(attr_counts.get("unresolved", 0)),
                "ambiguous": int(attr_counts.get("ambiguous", 0)),
                "poisson_inversion_failures": int(attributed["inversion_failure"].notna().sum()),
            },
            "sog_quote_coverage": {
                "raw_rows_by_book": core.groupby("book_key").size().astype(int).to_dict(),
                "paired_players_by_book": resolved.groupby("book").size().astype(int).to_dict(),
                "event_team_book_rows": len(per_book_team), "event_team_cross_book_rows": len(team_projection),
                "event_team_rows_by_book_count": team_projection["n_sog_books"].value_counts().sort_index().astype(int).to_dict(),
                "events_with_both_team_projections": int((team_projection.groupby("game_id")["shooting_team"].nunique() == 2).sum()),
            },
            "coverage_fit": {
                "train_rows": len(coverage_train),
                "validation_rows": int((coverage_rows["split"] == "development_validation").sum()),
                "missing_actual_team_sog": int(coverage_rows["actual_team_sog"].isna().sum()),
            },
            "market_goals": market_counts,
            "starter_translation": {
                "train_rows": len(translation_train),
                "validation_complete_rows": int(((translation_rows["split"] == "development_validation") & complete_translation).sum()),
                "all_missing_projection_rows": int((~complete_translation).sum()),
            },
            "saves_pairing": saves_pair_counts,
            "saves_pricing": pricing_counts,
            "validation": {
                "eligible_quotes": len(validation), "goalie_nights": validation["cluster_id"].nunique(),
                "pushes_excluded_per_side": int((validation["grade"] == 0).sum()),
            },
        }
        write_json(run_dir / "coverage_and_exclusion_counts.json", coverage_counts)

        frozen_recipe = {
            "experiment": 12, "stage": "DEVELOPMENT_FROZEN", "registration": "section 15.2a",
            "code_path": str(Path(__file__).relative_to(REPO_ROOT)), "code_sha256": code_hash,
            "development_dates": {
                "overall": [DEVELOPMENT_START, DEVELOPMENT_END],
                "fit": [DEVELOPMENT_START, TRAIN_END],
                "validation": [VALIDATION_START, DEVELOPMENT_END],
                "split_unit": "NHL game/event",
            },
            "event_mapping": "Eastern date plus unordered event teams; exactly one NHL game; retain source event id and game_id; fail closed on ambiguity.",
            "sog_recipe": {
                "books": "sportsbooks only; DFS excluded", "lines": "exact two-sided half-point only",
                "devig": "normalize reciprocal decimal probabilities within exact O/U pair",
                "poisson_inversion": "solve Poisson survival(floor(line), mean) == fair P(over)",
                "attribution": "2023-24 season-team PBP roster membership among event teams; actual-game presence is not a filter",
                "aggregation": "sum player means event-team-book; cross-book median mean sum and median player count",
            },
            "coverage_model": coverage_params,
            "opponent_goals": {
                "snapshot": "latest market requested_ts <= W1 SOG event anchor",
                "consensus_total": "median totals point across books",
                "home_win_probability": "mean de-vigged home h2h probability across books",
                "formula": "consensus_total * shooting_team win probability",
            },
            "starter_translation_model": translation_params,
            "residual_distribution": {
                "source": "development-train starter actual saves minus fitted starter saves",
                "n": len(residuals), "sha256": canonical_frame_hash(residual_frame),
                "artifact": "development_train_residuals.csv",
            },
            "pricing": {
                "probabilities": "empirical residual P(over), P(under), P(push)",
                "integer_line_comparison": "P(over)/(P(over)+P(under)); pushes void",
                "market_gap": "model conditional P(over) minus exact-pair de-vigged saves-market P(over)",
            },
            "betting": {
                "unit": "(saves_event_id, goalie_id, book, line)", "stake": "flat one unit",
                "payout": "quoted decimal actual price", "pushes": "excluded",
                "cluster": "NHL game_id:goalie_id", "blind_baseline": "same side on every non-push quote in identical eligible universe",
            },
            "threshold_selection": {
                "grid": list(THRESHOLDS), "criterion": "maximize goalie-night cluster bootstrap lower CI of model-minus-blind ROI delta per side",
                "minimum_selected": MIN_SELECTED, "maximum_empty_arm_rate": MAX_EMPTY_RATE,
                "tie_break": "larger threshold", "insufficient_fallback": 0.05,
                "bootstrap": {"resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
                "frozen_selections": selections,
            },
            "one_touch_confirmation_guard": {
                "status": "DESIGN_FROZEN_CONFIRMATION_NOT_AUTHORIZED_OR_IMPLEMENTED_IN_THIS_ASSIGNMENT",
                "required_resume_action": "User explicitly resumes Experiment 12 confirmation implementation/execution.",
                "pre_read_sequence": [
                    "Verify DEVELOPMENT_FROZEN metadata, code SHA-256, frozen_recipe SHA-256, and all artifact hashes.",
                    "Require the deterministic confirmation authorization token from this freeze.",
                    "Search every Experiment 12 artifact directory for any prior confirmation_touch.json; refuse if any exists, including IN_PROGRESS or COMPLETED.",
                    "Atomically create confirmation_touch.json with status IN_PROGRESS before any predicate-level 2024-25 read.",
                    "Use only predicate-level 2024-25 scans and the frozen parameters; never refit or reselect.",
                    "Atomically finalize the marker as COMPLETED only after all confirmation artifacts and manifest hashes are durable.",
                ],
                "failure_policy": "Any marker, even from a failed touch, consumes the one touch and requires a new preregistration; never delete or overwrite it.",
            },
        }
        token_basis = json.dumps(json_ready(frozen_recipe), sort_keys=True, separators=(",", ":")).encode("utf-8")
        confirmation_token = sha256_bytes(b"experiment12-confirmation:" + token_basis)
        frozen_recipe["one_touch_confirmation_guard"]["authorization_token_sha256"] = confirmation_token
        write_json(run_dir / "frozen_recipe.json", frozen_recipe)
        recipe_hash = sha256_file(run_dir / "frozen_recipe.json")
        metadata = {
            "experiment": 12, "stage": "development", "status": "DEVELOPMENT_FROZEN",
            "started_at": started_at, "completed_at": utc_now(), "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "code_sha256": code_hash, "frozen_recipe_sha256": recipe_hash,
            "confirmation_stage_invoked": False, "confirmation_rows_loaded": 0,
            "development_selections": selections,
            "interpretation": "2023-24 development evidence only; no confirmation claim.",
        }
        write_json(run_dir / "metadata.json", metadata)
        write_json(run_dir / "output_manifest.json", artifact_manifest(run_dir))
        log("DEVELOPMENT_FROZEN: confirmation was not invoked and zero 2024-25 rows were materialized")
        return run_dir
    except Exception as exc:
        stopped = {
            "experiment": 12, "stage": "development", "status": "DEVELOPMENT_STOPPED",
            "started_at": started_at, "stopped_at": utc_now(), "code_sha256": code_hash,
            "confirmation_stage_invoked": False, "confirmation_rows_loaded": 0,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "read_audit_so_far": [asdict(record) for record in read_records],
        }
        write_json(run_dir / "metadata.json", stopped)
        log(f"DEVELOPMENT_STOPPED: {type(exc).__name__}: {exc}")
        raise


def run_confirmation_locked(args: argparse.Namespace) -> None:
    # This must remain before every confirmation path/import/read until the user resumes the task.
    raise ConstructionStopped(
        "Confirmation is deliberately not implemented or authorized in this assignment. "
        "No confirmation file was opened and no one-touch marker was created."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 12 W1 cross-market runner with separate development and confirmation stages."
    )
    parser.add_argument("--stage", required=True, choices=("development", "confirmation"))
    parser.add_argument(
        "--frozen-run-dir", type=Path,
        help="Required only after explicit user authorization for a future confirmation implementation.",
    )
    parser.add_argument(
        "--confirmation-token",
        help="Required only after explicit user authorization for a future confirmation implementation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "development":
        if args.frozen_run_dir or args.confirmation_token:
            raise ConstructionStopped("confirmation-only arguments are forbidden in development")
        run_dir = run_development()
        print(f"Development frozen at {run_dir}")
        return 0
    run_confirmation_locked(args)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConstructionStopped as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        raise SystemExit(2)
