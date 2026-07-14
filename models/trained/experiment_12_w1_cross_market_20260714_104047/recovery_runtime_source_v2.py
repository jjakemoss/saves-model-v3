#!/usr/bin/env python3
"""One-touch confirmation runner for Experiment 12 W1 cross-market coherence.

The frozen development runner is imported only after its SHA-256 and complete
development manifest are verified. Importing it performs no data access. This
runner has two modes:

    --mode preflight   artifact verification plus synthetic tests; no source data
    --mode confirm     repeats preflight, creates the immutable touch marker, then
                       performs predicate-scoped 2024-25 reads exactly once

The root ``recovery_touch.json`` is immutable and remains the original
IN_PROGRESS event. Completion or failure is appended to the history JSONL and
recorded in a separate exclusive completion/failure marker, preserving history.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_RUN = REPO_ROOT / "models" / "trained" / "experiment_12_w1_cross_market_20260714_104047"
FROZEN_SCRIPT = REPO_ROOT / "scripts" / "experiment_12_w1_cross_market.py"
FROZEN_SCRIPT_SHA256 = "48ad28704e49def8b5bbf8d02f46b2f4fa370e258cd719591b48bdbf87f39f0d"
FROZEN_RECIPE_SHA256 = "3fa2aadc12b231f7260676724abf1cd85dca6e4738ec88ed00cc87a73c4ba1fc"
EXPERIMENT_GLOB = "experiment_12_w1_cross_market_*"

CORE_SNAPSHOTS = REPO_ROOT / "data" / "processed" / "core_bettime_202607_snapshots.parquet"
MARKET_FEATURES = REPO_ROOT / "data" / "processed" / "market_game_features.parquet"
OUTCOMES = REPO_ROOT / "data" / "processed" / "clean_training_data.parquet"
CLOSING = REPO_ROOT / "data" / "processed" / "multibook_classification_training_data.parquet"
PBP_DIR = REPO_ROOT / "data" / "raw" / "play_by_play"

CONFIRMATION_SEASON = "2024-25"
CONFIRMATION_SEASON_CODE = 20242025
CONFIRMATION_START = "2024-10-04"
CONFIRMATION_END = "2025-04-17"
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
THRESHOLDS = {"OVER": 0.03, "UNDER": 0.03}
DFS_BOOKS = frozenset({"underdog", "prizepicks"})

TOUCH_MARKER = FROZEN_RUN / "recovery_touch.json"
TOUCH_HISTORY = FROZEN_RUN / "recovery_touch_history.jsonl"
COMPLETION_MARKER = FROZEN_RUN / "recovery_touch_completed.json"
FAILURE_MARKER = FROZEN_RUN / "recovery_touch_failed.json"
PREFLIGHT_ARTIFACT = FROZEN_RUN / "recovery_preflight_v2.json"

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
    "Utah Hockey Club": "UTA", "Utah Mammoth": "UTA", "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}

CORE_SOG_COLUMNS = [
    "season", "event_id", "requested_ts", "true_commence_time", "effective_gap_minutes",
    "game_date_eastern", "home_team", "away_team", "book_key", "market_key",
    "player_name_raw", "side", "line", "price_decimal",
]
CORE_SAVES_COLUMNS = [
    "season", "event_id", "requested_ts", "true_commence_time", "effective_gap_minutes",
    "game_date_eastern", "home_team", "away_team", "book_key", "market_key",
    "player_name_raw", "side", "line", "price_decimal", "goalie_id", "goalie_name_matched",
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
CLOSING_COLUMNS = [
    "game_id", "game_date", "season", "goalie_id", "goalie_name", "team_abbrev",
    "opponent_team", "is_home", "saves", "betting_line", "odds_over_decimal",
    "odds_under_decimal", "book_key",
]


class PreflightFailure(RuntimeError):
    pass


class ConfirmationFailure(RuntimeError):
    pass


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
    predicate_content_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if not isinstance(value, (list, tuple, dict, set)) and pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_exclusive_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_history(payload: Any) -> None:
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")) + "\n"
    with TOUCH_HISTORY.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def make_logger(output_dir: Path) -> Callable[[str], None]:
    path = output_dir / "recovery_run_log.txt"

    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def verify_manifest() -> dict[str, Any]:
    manifest_path = FROZEN_RUN / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = FROZEN_RUN / item["path"]
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != item["bytes"] or actual_hash != item["sha256"]:
            failures.append(
                {"path": item["path"], "expected_bytes": item["bytes"], "actual_bytes": actual_size,
                 "expected_sha256": item["sha256"], "actual_sha256": actual_hash}
            )
    if failures:
        raise PreflightFailure(f"frozen output manifest mismatch: {failures}")
    return {"manifest_sha256": sha256_file(manifest_path), "files_verified": len(manifest["files"])}


def marker_paths() -> list[str]:
    root = FROZEN_RUN.parent
    return sorted(
        str(path.relative_to(REPO_ROOT))
        for directory in root.glob(EXPERIMENT_GLOB)
        for path in [directory / "recovery_touch.json"]
        if path.exists()
    )


def verify_authorization_token(recipe: dict[str, Any]) -> str:
    stored = recipe["one_touch_recovery_guard"]["authorization_token_sha256"]
    unsigned = copy.deepcopy(recipe)
    unsigned["one_touch_recovery_guard"].pop("authorization_token_sha256")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = sha256_bytes(b"experiment12-confirmation:" + canonical)
    if stored != expected:
        raise PreflightFailure(f"authorization token mismatch: stored={stored} expected={expected}")
    return stored


def load_frozen_module() -> Any:
    spec = importlib.util.spec_from_file_location("experiment_12_frozen", FROZEN_SCRIPT)
    if spec is None or spec.loader is None:
        raise PreflightFailure("cannot create frozen module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_nhl_games(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game_id, group in outcomes.groupby("game_id", sort=True):
        dates = pd.to_datetime(group["game_date"]).dt.date.astype(str).unique()
        home = group.loc[group["is_home"] == 1, "team_abbrev"].unique()
        away = group.loc[group["is_home"] == 0, "team_abbrev"].unique()
        if len(dates) != 1 or len(home) != 1 or len(away) != 1:
            raise ConfirmationFailure(f"ambiguous NHL game metadata for game_id={game_id}")
        rows.append(
            {"game_id": int(game_id), "game_date": dates[0], "home_abbrev": home[0],
             "away_abbrev": away[0], "team_key": "|".join(sorted((home[0], away[0])))}
        )
    games = pd.DataFrame(rows)
    if games.duplicated(["game_date", "team_key"], keep=False).any():
        raise ConfirmationFailure("NHL date plus unordered-team key is ambiguous")
    return games


def event_metadata(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    meta = frame[["event_id", "game_date_eastern", "home_team", "away_team"]].drop_duplicates()
    if (meta.groupby("event_id").size() > 1).any():
        raise ConfirmationFailure(f"{source}: event id has conflicting metadata")
    meta = meta.rename(columns={"event_id": f"{source}_event_id", "game_date_eastern": "game_date"})
    meta["home_abbrev"] = meta["home_team"].map(TEAM_NAME_TO_ABBREV)
    meta["away_abbrev"] = meta["away_team"].map(TEAM_NAME_TO_ABBREV)
    if meta[["home_abbrev", "away_abbrev"]].isna().any().any():
        unknown = sorted(set(meta.loc[meta["home_abbrev"].isna(), "home_team"]) |
                         set(meta.loc[meta["away_abbrev"].isna(), "away_team"]))
        raise ConfirmationFailure(f"{source}: unknown teams {unknown}")
    meta["team_key"] = meta.apply(
        lambda row: "|".join(sorted((row["home_abbrev"], row["away_abbrev"]))), axis=1
    )
    return meta


def map_events(frame: pd.DataFrame, games: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = event_metadata(frame, source)
    mapped = meta.merge(
        games[["game_id", "game_date", "team_key"]], on=["game_date", "team_key"],
        how="left", validate="many_to_one", indicator=True,
    )
    matched = mapped[mapped["_merge"] == "both"].drop(columns="_merge")
    unmatched = mapped[mapped["_merge"] != "both"].drop(columns="_merge")
    if matched.duplicated("game_id", keep=False).any():
        raise ConfirmationFailure(f"{source}: multiple Odds events map to one NHL game")
    return matched, unmatched


def primary_bootstrap(universe: pd.DataFrame, side: str, threshold: float) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    side_lower = side.lower()
    profit_col = f"profit_{side_lower}"
    nonpush = universe[universe[profit_col].notna()].copy()
    gap = nonpush["probability_gap_over"] if side == "OVER" else -nonpush["probability_gap_over"]
    nonpush["selected"] = gap >= threshold
    clusters = sorted(nonpush["cluster_id"].unique())
    blind = nonpush.groupby("cluster_id")[profit_col].agg(["sum", "count"]).reindex(clusters, fill_value=0)
    model = nonpush[nonpush["selected"]].groupby("cluster_id")[profit_col].agg(["sum", "count"]).reindex(clusters, fill_value=0)
    inputs = pd.DataFrame(
        {"side": side, "cluster_id": clusters, "blind_profit_sum": blind["sum"].to_numpy(float),
         "blind_bet_count": blind["count"].to_numpy(int), "model_profit_sum": model["sum"].to_numpy(float),
         "model_bet_count": model["count"].to_numpy(int)}
    )
    bp = inputs["blind_profit_sum"].to_numpy(float)
    bn = inputs["blind_bet_count"].to_numpy(float)
    mp = inputs["model_profit_sum"].to_numpy(float)
    mn = inputs["model_bet_count"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.full(N_BOOTSTRAP, np.nan)
    model_rois = np.full(N_BOOTSTRAP, np.nan)
    blind_rois = np.full(N_BOOTSTRAP, np.nan)
    for start in range(0, N_BOOTSTRAP, 250):
        size = min(250, N_BOOTSTRAP - start)
        sample = rng.integers(0, len(clusters), size=(size, len(clusters)))
        sampled_mp, sampled_mn = mp[sample].sum(axis=1), mn[sample].sum(axis=1)
        sampled_bp, sampled_bn = bp[sample].sum(axis=1), bn[sample].sum(axis=1)
        valid = (sampled_mn > 0) & (sampled_bn > 0)
        model_rois[start:start + size][valid] = sampled_mp[valid] / sampled_mn[valid]
        blind_rois[start:start + size][valid] = sampled_bp[valid] / sampled_bn[valid]
        values[start:start + size][valid] = model_rois[start:start + size][valid] - blind_rois[start:start + size][valid]
    selected = nonpush[nonpush["selected"]].copy()
    selected["selected_side"] = side
    selected["selected_threshold"] = threshold
    selected["selected_profit"] = selected[profit_col]
    valid_values = values[np.isfinite(values)]
    lower, upper = np.quantile(valid_values, [0.025, 0.975]) if len(valid_values) else (np.nan, np.nan)
    empty = int((~np.isfinite(values)).sum())
    n_selected = int(len(selected))
    qualified = n_selected >= 100
    unstable = qualified and empty / N_BOOTSTRAP > 0.01
    if not qualified:
        verdict = "INSUFFICIENT SAMPLE"
    elif unstable:
        verdict = "UNSTABLE"
    elif lower > 0.0:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    result = {
        "side": side, "threshold": threshold, "n_eligible_nonpush_quotes": int(len(nonpush)),
        "n_eligible_goalie_nights": int(len(clusters)), "n_selected_graded_bets": n_selected,
        "n_selected_goalie_nights": int(selected["cluster_id"].nunique()),
        "model_roi": float(selected[profit_col].mean()) if n_selected else None,
        "blind_same_side_roi": float(nonpush[profit_col].mean()) if len(nonpush) else None,
        "model_minus_blind_roi_delta": (
            float(selected[profit_col].mean() - nonpush[profit_col].mean()) if n_selected else None
        ),
        "bootstrap_lower_ci95": None if not np.isfinite(lower) else float(lower),
        "bootstrap_upper_ci95": None if not np.isfinite(upper) else float(upper),
        "bootstrap_empty_model_arm_resamples": empty,
        "bootstrap_empty_model_arm_rate": empty / N_BOOTSTRAP,
        "qualified_sample": qualified, "unstable": unstable, "verdict": verdict,
    }
    draws = pd.DataFrame(
        {"side": side, "draw": np.arange(N_BOOTSTRAP), "model_roi": model_rois,
         "blind_roi": blind_rois, "model_minus_blind_roi_delta": values,
         "empty_model_arm": ~np.isfinite(values)}
    )
    return result, selected, inputs, draws


def synthetic_preflight(frozen: Any) -> dict[str, Any]:
    synthetic_quotes = pd.DataFrame(
        [
            {"event_id": "e1", "book_key": "book", "player_name_raw": "Test Player", "line": 2.5, "side": "Over", "price_decimal": 2.0},
            {"event_id": "e1", "book_key": "book", "player_name_raw": "Test Player", "line": 2.5, "side": "Under", "price_decimal": 2.0},
            {"event_id": "e1", "book_key": "book", "player_name_raw": "Integer Player", "line": 2.0, "side": "Over", "price_decimal": 2.0},
            {"event_id": "e1", "book_key": "book", "player_name_raw": "Integer Player", "line": 2.0, "side": "Under", "price_decimal": 2.0},
        ]
    )
    paired, counts, _ = frozen.pair_quotes(
        synthetic_quotes, ["event_id", "book_key", "player_name_raw", "line"],
        "book_key", "player_name_raw", True,
    )
    if len(paired) != 1 or counts["excluded_non_half_point_pairs"] != 1:
        raise PreflightFailure("synthetic exact-pair/half-point test failed")
    mean = frozen.poisson_mean_from_over_probability(2.5, 0.5)
    if abs(float(frozen.poisson.sf(2, mean)) - 0.5) > 1e-10:
        raise PreflightFailure("synthetic Poisson inversion test failed")
    mapping = pd.DataFrame(
        [{"core_event_id": "e1", "game_id": 1, "game_date": "2024-10-04", "home_abbrev": "UTA", "away_abbrev": "CHI"}]
    )
    attributed, resolved = frozen.attribute_sog_players(
        paired, mapping, {frozen.normalize_name("Test Player"): {"UTA"}},
        {frozen.person_key("Test Player"): {"UTA"}},
    )
    if len(resolved) != 1 or attributed.iloc[0]["assigned_team"] != "UTA":
        raise PreflightFailure("synthetic season-roster attribution test failed")
    residuals = np.array([-1.0, 0.0, 1.0])
    po, pu, pp, pc = frozen.empirical_probabilities(np.array([2.0]), np.array([2.0]), residuals)
    if not (np.isclose(po[0], 1 / 3) and np.isclose(pu[0], 1 / 3) and np.isclose(pp[0], 1 / 3) and np.isclose(pc[0], 0.5)):
        raise PreflightFailure("synthetic empirical pricing test failed")
    outcomes = pd.DataFrame(
        [
            {"game_id": 1, "game_date": pd.Timestamp("2024-10-04"), "is_home": 1, "team_abbrev": "UTA"},
            {"game_id": 1, "game_date": pd.Timestamp("2024-10-04"), "is_home": 0, "team_abbrev": "CHI"},
        ]
    )
    games = build_nhl_games(outcomes)
    event = pd.DataFrame(
        [{"event_id": "e1", "game_date_eastern": "2024-10-04", "home_team": "Utah Hockey Club", "away_team": "Chicago Blackhawks"}]
    )
    mapped, unmatched = map_events(event, games, "core")
    if len(mapped) != 1 or len(unmatched) != 0 or int(mapped.iloc[0]["game_id"]) != 1:
        raise PreflightFailure("synthetic unordered-team event mapping test failed")
    universe = pd.DataFrame(
        [
            {"cluster_id": "1:1", "probability_gap_over": 0.1, "profit_over": 1.0, "profit_under": -1.0},
            {"cluster_id": "2:2", "probability_gap_over": -0.1, "profit_over": -1.0, "profit_under": 1.0},
            {"cluster_id": "3:3", "probability_gap_over": 0.0, "profit_over": 0.5, "profit_under": -1.0},
        ]
    )
    first = primary_bootstrap(universe, "OVER", 0.03)[0]
    second = primary_bootstrap(universe, "OVER", 0.03)[0]
    if first != second:
        raise PreflightFailure("synthetic bootstrap determinism test failed")
    return {
        "exact_pair_and_half_point": "PASS", "poisson_inversion": "PASS",
        "season_roster_non_dresser_rule": "PASS", "empirical_push_conditioning": "PASS",
        "utah_unordered_team_mapping": "PASS", "bootstrap_seed42_determinism": "PASS",
    }


def run_preflight(require_marker_absent: bool = True) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    if sha256_file(FROZEN_SCRIPT) != FROZEN_SCRIPT_SHA256:
        raise PreflightFailure("frozen development script hash mismatch")
    metadata = json.loads((FROZEN_RUN / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "DEVELOPMENT_FROZEN" or metadata.get("code_sha256") != FROZEN_SCRIPT_SHA256:
        raise PreflightFailure("frozen metadata status/code mismatch")
    if sha256_file(FROZEN_RUN / "frozen_recipe.json") != FROZEN_RECIPE_SHA256:
        raise PreflightFailure("frozen recipe hash mismatch")
    if metadata.get("frozen_recipe_sha256") != FROZEN_RECIPE_SHA256:
        raise PreflightFailure("metadata frozen recipe hash mismatch")
    manifest = verify_manifest()
    markers = marker_paths()
    if require_marker_absent and markers:
        raise PreflightFailure(f"prior confirmation touch marker exists: {markers}")
    recipe = json.loads((FROZEN_RUN / "frozen_recipe.json").read_text(encoding="utf-8"))
    token = verify_authorization_token(recipe)
    if recipe["threshold_selection"]["frozen_selections"]["over"]["threshold"] != 0.03:
        raise PreflightFailure("frozen OVER threshold is not 0.03")
    if recipe["threshold_selection"]["frozen_selections"]["under"]["threshold"] != 0.03:
        raise PreflightFailure("frozen UNDER threshold is not 0.03")
    frozen = load_frozen_module()
    synthetic = synthetic_preflight(frozen)
    result = {
        "status": "PREFLIGHT_PASS", "completed_at": utc_now(),
        "source_data_materialized": False, "source_data_paths_opened": [],
        "frozen_script_sha256": FROZEN_SCRIPT_SHA256,
        "frozen_recipe_sha256": FROZEN_RECIPE_SHA256,
        "recovery_runner_sha256": sha256_file(Path(__file__)),
        "authorization_token_verified": token,
        "manifest": manifest, "existing_touch_markers": markers,
        "synthetic_tests": synthetic,
    }
    return result, frozen, recipe


def canonical_frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for col in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[col]):
            normalized[col] = normalized[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    if len(normalized):
        normalized = normalized.sort_values(list(normalized.columns), kind="mergesort", na_position="first")
    payload = normalized.to_csv(
        index=False, lineterminator="\n", na_rep="<NA>", float_format="%.17g"
    ).encode("utf-8")
    return sha256_bytes(payload)


def predicate_scan(
    name: str, path: Path, columns: list[str], predicate: ds.Expression, predicate_text: str,
    date_column: str, read_records: list[ReadRecord], season_column: str | None = None,
) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    frame = dataset.scanner(columns=columns, filter=predicate, use_threads=False).to_table().to_pandas()
    dates = pd.to_datetime(frame[date_column], errors="raise") if len(frame) else pd.Series(dtype="datetime64[ns]")
    stat = path.stat()
    read_records.append(
        ReadRecord(
            input_name=name, path=str(path.relative_to(REPO_ROOT)), columns=columns,
            predicate=predicate_text, predicate_applied_before_materialization=True,
            rows_loaded=int(len(frame)),
            min_date_loaded=None if frame.empty else dates.min().date().isoformat(),
            max_date_loaded=None if frame.empty else dates.max().date().isoformat(),
            loaded_seasons=[] if season_column is None else sorted(frame[season_column].dropna().astype(str).unique()),
            source_size_bytes=int(stat.st_size),
            source_mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            source_schema_sha256=sha256_bytes(str(pq.ParquetFile(path).schema_arrow).encode("utf-8")),
            predicate_content_sha256=canonical_frame_hash(frame),
        )
    )
    if frame.empty:
        raise ConfirmationFailure(f"{name}: predicate returned zero rows")
    return frame


def load_inputs(read_records: list[ReadRecord]) -> tuple[pd.DataFrame, ...]:
    sog = predicate_scan(
        "core_2024_25_sog", CORE_SNAPSHOTS, CORE_SOG_COLUMNS,
        (ds.field("season") == CONFIRMATION_SEASON) & (ds.field("market_key") == "player_shots_on_goal"),
        "season == '2024-25' AND market_key == 'player_shots_on_goal'", "game_date_eastern", read_records, "season",
    )
    saves = predicate_scan(
        "core_2024_25_saves", CORE_SNAPSHOTS, CORE_SAVES_COLUMNS,
        (ds.field("season") == CONFIRMATION_SEASON) & (ds.field("market_key") == "player_total_saves"),
        "season == '2024-25' AND market_key == 'player_total_saves'", "game_date_eastern", read_records, "season",
    )
    market = predicate_scan(
        "market_state_2024_25", MARKET_FEATURES, MARKET_COLUMNS,
        (ds.field("game_date_eastern") >= CONFIRMATION_START) & (ds.field("game_date_eastern") <= CONFIRMATION_END),
        "2024-10-04 <= game_date_eastern <= 2025-04-17", "game_date_eastern", read_records,
    )
    outcomes = predicate_scan(
        "starter_outcomes_2024_25", OUTCOMES, OUTCOME_COLUMNS,
        ds.field("season") == CONFIRMATION_SEASON_CODE,
        "season == 20242025", "game_date", read_records, "season",
    )
    closing = predicate_scan(
        "closing_saves_2024_25", CLOSING, CLOSING_COLUMNS,
        ds.field("season") == CONFIRMATION_SEASON_CODE,
        "season == 20242025", "game_date", read_records, "season",
    )
    for name, frame, date_col in (
        ("SOG", sog, "game_date_eastern"), ("saves", saves, "game_date_eastern"),
        ("market", market, "game_date_eastern"), ("outcomes", outcomes, "game_date"),
        ("closing", closing, "game_date"),
    ):
        dates = pd.to_datetime(frame[date_col])
        if dates.min() < pd.Timestamp(CONFIRMATION_START) or dates.max() > pd.Timestamp(CONFIRMATION_END):
            raise ConfirmationFailure(f"{name}: confirmation date assertion failed: {dates.min()}..{dates.max()}")
    if set(sog["season"]) != {CONFIRMATION_SEASON} or set(saves["season"]) != {CONFIRMATION_SEASON}:
        raise ConfirmationFailure("core season assertion failed")
    if set(outcomes["season"].astype(int)) != {CONFIRMATION_SEASON_CODE} or set(closing["season"].astype(int)) != {CONFIRMATION_SEASON_CODE}:
        raise ConfirmationFailure("outcome/closing season assertion failed")
    for frame, name in ((sog, "SOG"), (saves, "saves")):
        if (frame["effective_gap_minutes"] < 10.0).any():
            raise ConfirmationFailure(f"{name}: core parquet contains a drift-excluded event")
        if "fanatics" in set(frame["book_key"]):
            raise ConfirmationFailure(f"{name}: Fanatics schema surprise")
    return sog, saves, market, outcomes, closing


def load_roster(outcomes: pd.DataFrame) -> tuple[dict[str, set[str]], dict[tuple[str, str], set[str]], dict[str, Any]]:
    by_name: dict[str, set[str]] = defaultdict(set)
    by_person: dict[tuple[str, str], set[str]] = defaultdict(set)
    file_rows = []
    max_date = None
    for game_id in sorted(set(outcomes["game_id"].astype(int))):
        if not str(game_id).startswith("2024"):
            raise ConfirmationFailure(f"PBP firewall rejected game id {game_id}")
        path = PBP_DIR / f"{game_id}.json"
        raw = path.read_bytes()
        payload = json.loads(raw)
        game_date = str(payload.get("gameDate"))
        if str(payload.get("season")) != str(CONFIRMATION_SEASON_CODE):
            raise ConfirmationFailure(f"PBP season assertion failed: {path.name}")
        if not (CONFIRMATION_START <= game_date <= CONFIRMATION_END):
            raise ConfirmationFailure(f"PBP date assertion failed: {path.name} {game_date}")
        max_date = max(max_date or game_date, game_date)
        teams = {
            int(payload["homeTeam"]["id"]): payload["homeTeam"]["abbrev"],
            int(payload["awayTeam"]["id"]): payload["awayTeam"]["abbrev"],
        }
        for player in payload.get("rosterSpots") or []:
            name = f"{player['firstName']['default']} {player['lastName']['default']}"
            team = teams[int(player["teamId"])]
            by_name[FROZEN.normalize_name(name)].add(team)
            by_person[FROZEN.person_key(name)].add(team)
        file_rows.append({"game_id": game_id, "path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_bytes(raw)})
    manifest_hash = sha256_bytes(
        "\n".join(f"{row['game_id']}:{row['sha256']}" for row in file_rows).encode("ascii")
    )
    return by_name, by_person, {
        "files_loaded": len(file_rows), "max_game_date_loaded": max_date,
        "season_assertion": CONFIRMATION_SEASON_CODE, "file_manifest_sha256": manifest_hash,
        "files": file_rows,
    }


def build_market_goals(
    market: pd.DataFrame, market_map: pd.DataFrame, core_map: pd.DataFrame, sog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    anchor_counts = sog.groupby("event_id")["requested_ts"].nunique()
    if (anchor_counts != 1).any():
        raise ConfirmationFailure("core SOG event has multiple anchors")
    anchors = sog.groupby("event_id", as_index=False)["requested_ts"].first().rename(
        columns={"event_id": "core_event_id", "requested_ts": "w1_anchor_ts"}
    )
    games = core_map[["core_event_id", "game_id"]].merge(anchors, on="core_event_id", validate="one_to_one")
    work = market.merge(
        market_map[["market_event_id", "game_id"]], left_on="event_id", right_on="market_event_id",
        how="inner", validate="many_to_one",
    ).merge(games, on="game_id", how="inner", validate="many_to_one")
    work["requested_dt"] = pd.to_datetime(work["requested_ts"], utc=True)
    work["anchor_dt"] = pd.to_datetime(work["w1_anchor_ts"], utc=True)
    later = int((work["requested_dt"] > work["anchor_dt"]).sum())
    safe = work[work["requested_dt"] <= work["anchor_dt"]].copy()
    latest = safe[safe["requested_dt"] == safe.groupby("game_id")["requested_dt"].transform("max")].copy()
    if (latest["requested_dt"] > latest["anchor_dt"]).any():
        raise ConfirmationFailure("market snapshot timing assertion failed")
    h2h = latest[(latest["market"] == "h2h") & (latest["outcome_label"] == latest["home_abbrev"])].copy()
    h2h_conflicts = h2h.groupby(["game_id", "book"])["implied_prob_devig"].nunique()
    bad_h2h = set(h2h_conflicts[h2h_conflicts > 1].index)
    if bad_h2h:
        h2h = h2h[~h2h[["game_id", "book"]].apply(tuple, axis=1).isin(bad_h2h)]
    home_prob = h2h.drop_duplicates(["game_id", "book", "implied_prob_devig"]).groupby("game_id").agg(
        home_win_prob_devigged=("implied_prob_devig", "mean"), n_h2h_books=("book", "nunique")
    )
    totals = latest[latest["market"] == "totals"][["game_id", "book", "point"]].drop_duplicates()
    total_conflicts = totals.groupby(["game_id", "book"])["point"].nunique()
    bad_totals = set(total_conflicts[total_conflicts > 1].index)
    if bad_totals:
        totals = totals[~totals[["game_id", "book"]].apply(tuple, axis=1).isin(bad_totals)]
    total_summary = totals.groupby("game_id").agg(
        consensus_total=("point", "median"), n_totals_books=("book", "nunique")
    )
    meta = latest.groupby("game_id").agg(
        market_event_id=("market_event_id", "first"), core_event_id=("core_event_id", "first"),
        game_date=("game_date_eastern", "first"), home_abbrev=("home_abbrev", "first"),
        away_abbrev=("away_abbrev", "first"), market_requested_ts=("requested_ts", "first"),
        w1_anchor_ts=("w1_anchor_ts", "first"),
    )
    result = meta.join(home_prob).join(total_summary).reset_index()
    result["home_expected_goals"] = result["consensus_total"] * result["home_win_prob_devigged"]
    result["away_expected_goals"] = result["consensus_total"] * (1.0 - result["home_win_prob_devigged"])
    counts = {
        "rows_later_than_w1_anchor_excluded": later,
        "games_with_timing_safe_snapshot": int(latest["game_id"].nunique()),
        "h2h_conflict_groups_excluded": len(bad_h2h),
        "totals_conflict_groups_excluded": len(bad_totals),
        "games_with_complete_projection": int(result[["home_expected_goals", "away_expected_goals"]].notna().all(axis=1).sum()),
    }
    return result, counts


def build_translation(
    outcomes: pd.DataFrame, team_projection: pd.DataFrame, market_goals: pd.DataFrame, recipe: dict[str, Any],
) -> pd.DataFrame:
    coverage = recipe["coverage_model"]
    tp = team_projection.copy()
    tp["projected_team_sog"] = (
        coverage["intercept"] + coverage["coefficients"][0] * tp["aggregate_player_mean"]
        + coverage["coefficients"][1] * tp["listed_player_count"]
    )
    rows = outcomes.merge(
        tp[["game_id", "shooting_team", "sog_event_id", "projected_team_sog", "n_sog_books"]],
        left_on=["game_id", "opponent_team"], right_on=["game_id", "shooting_team"],
        how="left", validate="many_to_one",
    )
    rows = rows.merge(
        market_goals[["game_id", "market_event_id", "market_requested_ts", "w1_anchor_ts",
                      "home_expected_goals", "away_expected_goals"]],
        on="game_id", how="left", validate="many_to_one",
    )
    rows["market_opponent_goals"] = np.where(
        rows["is_home"] == 1, rows["away_expected_goals"], rows["home_expected_goals"]
    )
    rows["coherence_raw_saves"] = rows["projected_team_sog"] - rows["market_opponent_goals"]
    starter = recipe["starter_translation_model"]
    rows["predicted_starter_saves"] = starter["intercept"] + starter["coefficients"][0] * rows["coherence_raw_saves"]
    return rows


def price_quotes(
    saves_pairs: pd.DataFrame, saves_map: pd.DataFrame, outcomes: pd.DataFrame,
    translation: pd.DataFrame, residuals: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    priced = saves_pairs.merge(
        saves_map[["saves_event_id", "game_id"]], left_on="event_id", right_on="saves_event_id",
        how="left", validate="many_to_one",
    )
    outcome_cols = ["game_id", "goalie_id", "game_date", "goalie_name", "team_abbrev", "opponent_team", "saves"]
    priced = priced.merge(
        outcomes[outcome_cols], on=["game_id", "goalie_id"], how="left", validate="many_to_one", indicator="_goalie"
    )
    unmatched_goalies = int((priced["_goalie"] != "both").sum())
    priced = priced[priced["_goalie"] == "both"].drop(columns="_goalie")
    projection_cols = [
        "game_id", "goalie_id", "sog_event_id", "market_event_id", "market_requested_ts",
        "w1_anchor_ts", "projected_team_sog", "market_opponent_goals", "coherence_raw_saves",
        "predicted_starter_saves",
    ]
    priced = priced.merge(translation[projection_cols], on=["game_id", "goalie_id"], how="left", validate="many_to_one")
    complete = priced[["predicted_starter_saves", "sog_event_id", "market_event_id"]].notna().all(axis=1)
    missing_projection = int((~complete).sum())
    priced = priced[complete].copy()
    po, pu, pp, pc = FROZEN.empirical_probabilities(
        priced["predicted_starter_saves"].to_numpy(float), priced["line"].to_numpy(float), residuals
    )
    priced["model_prob_over"] = po
    priced["model_prob_under"] = pu
    priced["model_prob_push"] = pp
    priced["model_prob_over_conditional"] = pc
    priced["probability_gap_over"] = pc - priced["fair_prob_over"]
    priced["cluster_id"] = priced["game_id"].astype(str) + ":" + priced["goalie_id"].astype(str)
    priced["grade"] = np.sign(priced["saves"] - priced["line"]).astype(int)
    priced["profit_over"] = np.where(priced["grade"] > 0, priced["over"] - 1.0, np.where(priced["grade"] < 0, -1.0, np.nan))
    priced["profit_under"] = np.where(priced["grade"] < 0, priced["under"] - 1.0, np.where(priced["grade"] > 0, -1.0, np.nan))
    if priced.duplicated(["event_id", "goalie_id", "book_key", "line"]).any():
        raise ConfirmationFailure("eligible quote unit is duplicated")
    return priced, {
        "paired_saves_units": len(saves_pairs), "unmatched_goalie_units": unmatched_goalies,
        "missing_projection_units": missing_projection, "eligible_quote_units": len(priced),
        "eligible_goalie_nights": priced["cluster_id"].nunique(), "push_units": int((priced["grade"] == 0).sum()),
    }


def mean_cluster_bootstrap(values: np.ndarray, cluster_ids: np.ndarray) -> dict[str, Any]:
    frame = pd.DataFrame({"value": values, "cluster": cluster_ids}).dropna()
    grouped = frame.groupby("cluster")["value"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(N_BOOTSTRAP)
    for start in range(0, N_BOOTSTRAP, 500):
        size = min(500, N_BOOTSTRAP - start)
        sample = rng.integers(0, len(grouped), size=(size, len(grouped)))
        draws[start:start + size] = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(frame["value"].mean()), "lower_ci95": float(lower), "upper_ci95": float(upper),
        "n_rows": len(frame), "n_clusters": len(grouped), "resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED,
    }


def scoring_secondary(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    usable = frame[(frame["grade"] != 0) & frame["model_prob_over_conditional"].notna()].copy()
    y = (usable["grade"] > 0).astype(float).to_numpy()
    model = np.clip(usable["model_prob_over_conditional"].to_numpy(float), 1e-12, 1 - 1e-12)
    market = np.clip(usable["fair_prob_over"].to_numpy(float), 1e-12, 1 - 1e-12)
    model_brier = (model - y) ** 2
    market_brier = (market - y) ** 2
    model_log = -(y * np.log(model) + (1 - y) * np.log(1 - model))
    market_log = -(y * np.log(market) + (1 - y) * np.log(1 - market))
    clusters = usable["cluster_id"].to_numpy(str)
    return {
        "label": label, "n_rows": len(usable), "n_clusters": usable["cluster_id"].nunique(),
        "model_brier": float(model_brier.mean()), "market_brier": float(market_brier.mean()),
        "brier_delta_model_minus_market": mean_cluster_bootstrap(model_brier - market_brier, clusters),
        "model_log_loss": float(model_log.mean()), "market_log_loss": float(market_log.mean()),
        "log_loss_delta_model_minus_market": mean_cluster_bootstrap(model_log - market_log, clusters),
    }


def build_closing_universe(
    closing: pd.DataFrame, translation: pd.DataFrame, residuals: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = closing.copy()
    missing_identity = work[["game_id", "goalie_id", "book_key", "betting_line"]].isna().any(axis=1)
    missing_rows = int(missing_identity.sum())
    work = work[~missing_identity].copy()
    before = len(work)
    work = work.drop_duplicates(
        ["game_id", "goalie_id", "book_key", "betting_line", "odds_over_decimal", "odds_under_decimal"]
    )
    exact_duplicates = before - len(work)
    key = ["game_id", "goalie_id", "book_key", "betting_line"]
    conflicts = work.groupby(key).agg(
        over_n=("odds_over_decimal", "nunique"), under_n=("odds_under_decimal", "nunique")
    )
    bad = set(conflicts[(conflicts["over_n"] > 1) | (conflicts["under_n"] > 1)].index)
    if bad:
        work = work[~work[key].apply(tuple, axis=1).isin(bad)]
    work = work.drop_duplicates(key)
    raw_over = 1.0 / work["odds_over_decimal"]
    raw_under = 1.0 / work["odds_under_decimal"]
    work["fair_prob_over"] = raw_over / (raw_over + raw_under)
    work["fair_prob_under"] = 1.0 - work["fair_prob_over"]
    projection = translation[["game_id", "goalie_id", "predicted_starter_saves"]]
    work = work.merge(projection, on=["game_id", "goalie_id"], how="left", validate="many_to_one")
    complete = work["predicted_starter_saves"].notna()
    missing_projection = int((~complete).sum())
    work = work[complete].copy()
    po, pu, pp, pc = FROZEN.empirical_probabilities(
        work["predicted_starter_saves"].to_numpy(float), work["betting_line"].to_numpy(float), residuals
    )
    work["model_prob_over"] = po
    work["model_prob_under"] = pu
    work["model_prob_push"] = pp
    work["model_prob_over_conditional"] = pc
    work["line"] = work["betting_line"]
    work["grade"] = np.sign(work["saves"] - work["line"]).astype(int)
    work["cluster_id"] = work["game_id"].astype(str) + ":" + work["goalie_id"].astype(str)
    return work, {
        "input_rows": len(closing), "missing_identity_rows": missing_rows,
        "exact_duplicate_extra_rows": exact_duplicates, "conflicting_quote_units_excluded": len(bad),
        "missing_projection_rows": missing_projection, "eligible_rows": len(work),
    }


def selected_clv(selected: pd.DataFrame, closing: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    lookup = closing[[
        "game_id", "goalie_id", "book_key", "line", "fair_prob_over", "fair_prob_under",
        "odds_over_decimal", "odds_under_decimal",
    ]].rename(
        columns={
            "fair_prob_over": "closing_fair_prob_over", "fair_prob_under": "closing_fair_prob_under",
            "odds_over_decimal": "closing_over_decimal", "odds_under_decimal": "closing_under_decimal",
        }
    )
    matched = selected.merge(lookup, on=["game_id", "goalie_id", "book_key", "line"], how="left", validate="many_to_one")
    is_over = matched["selected_side"] == "OVER"
    matched["bettime_selected_fair_prob"] = np.where(is_over, matched["fair_prob_over"], matched["fair_prob_under"])
    matched["closing_selected_fair_prob"] = np.where(is_over, matched["closing_fair_prob_over"], matched["closing_fair_prob_under"])
    matched["bettime_selected_decimal"] = np.where(is_over, matched["over"], matched["under"])
    matched["closing_selected_decimal"] = np.where(is_over, matched["closing_over_decimal"], matched["closing_under_decimal"])
    matched["clv_probability"] = matched["closing_selected_fair_prob"] - matched["bettime_selected_fair_prob"]
    matched["clv_decimal_price"] = matched["bettime_selected_decimal"] - matched["closing_selected_decimal"]
    matched_rows = matched[matched["clv_probability"].notna()].copy()
    summary = {
        "selected_bets": len(selected), "exact_same_book_line_matches": len(matched_rows),
        "match_rate": len(matched_rows) / len(selected) if len(selected) else None,
        "probability_clv": mean_cluster_bootstrap(
            matched_rows["clv_probability"].to_numpy(float), matched_rows["cluster_id"].to_numpy(str)
        ) if len(matched_rows) else None,
        "decimal_price_clv": mean_cluster_bootstrap(
            matched_rows["clv_decimal_price"].to_numpy(float), matched_rows["cluster_id"].to_numpy(str)
        ) if len(matched_rows) else None,
    }
    return matched, summary


def overall_verdict(results: dict[str, dict[str, Any]]) -> str:
    qualified = [result for result in results.values() if result["qualified_sample"]]
    if not qualified:
        return "INSUFFICIENT SAMPLE"
    if any(result["verdict"] == "UNSTABLE" for result in qualified):
        return "UNSTABLE / NO OVERALL PASS"
    if len(qualified) == 1:
        return qualified[0]["verdict"]
    verdicts = {result["verdict"] for result in qualified}
    if verdicts == {"PASS"}:
        return "PASS"
    if verdicts == {"FAIL"}:
        return "FAIL"
    return "MIXED / NO OVERALL PASS"


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "recovery_output_manifest.json":
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"generated_at": utc_now(), "files": files}


def run_confirmation(preflight: dict[str, Any], frozen: Any, recipe: dict[str, Any]) -> Path:
    global FROZEN
    FROZEN = frozen
    marker = {
        "experiment": 12, "stage": "recovery", "status": "IN_PROGRESS",
        "created_at": utc_now(), "frozen_run": str(FROZEN_RUN.relative_to(REPO_ROOT)),
        "frozen_script_sha256": FROZEN_SCRIPT_SHA256, "frozen_recipe_sha256": FROZEN_RECIPE_SHA256,
        "recovery_runner": str(Path(__file__).relative_to(REPO_ROOT)),
        "recovery_runner_sha256": sha256_file(Path(__file__)),
        "authorization_token_verified": preflight["authorization_token_verified"],
        "preflight_artifact": str(PREFLIGHT_ARTIFACT.relative_to(REPO_ROOT)),
        "history_file": str(TOUCH_HISTORY.relative_to(REPO_ROOT)),
        "immutability": "This marker is never overwritten; completion/failure is append-only history plus a separate exclusive marker.",
    }
    write_exclusive_json(TOUCH_MARKER, marker)
    append_history({"status": "IN_PROGRESS", "timestamp": marker["created_at"], "marker_sha256": sha256_file(TOUCH_MARKER)})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = FROZEN_RUN / f"recovery_{stamp}"
    output_dir.mkdir(parents=False, exist_ok=False)
    log = make_logger(output_dir)
    read_records: list[ReadRecord] = []
    started = utc_now()
    try:
        log("Immutable IN_PROGRESS marker exists; beginning the single predicate-scoped 2024-25 touch")
        sog, saves, market, outcomes, closing = load_inputs(read_records)
        log(
            f"Predicate reads: SOG={len(sog)}, saves={len(saves)}, market={len(market)}, "
            f"outcomes={len(outcomes)}, closing={len(closing)}"
        )
        games = build_nhl_games(outcomes)
        sog_map, sog_unmatched = map_events(sog, games, "core")
        saves_map, saves_unmatched = map_events(saves, games, "saves")
        market_map, market_unmatched = map_events(market, games, "market")
        roster_by_name, roster_by_person, pbp_audit = load_roster(outcomes)
        log(
            f"Event maps: SOG {len(sog_map)}/{len(sog_unmatched)} unmatched; saves "
            f"{len(saves_map)}/{len(saves_unmatched)}; market {len(market_map)}/{len(market_unmatched)}"
        )

        sog_sportsbook = sog[~sog["book_key"].isin(DFS_BOOKS)].copy()
        sog_pairs, sog_pair_counts, excluded_sog_lines = frozen.pair_quotes(
            sog_sportsbook, ["event_id", "book_key", "player_name_raw", "line"],
            "book_key", "player_name_raw", True,
        )
        attributed, resolved = frozen.attribute_sog_players(
            sog_pairs, sog_map, roster_by_name, roster_by_person
        )
        per_book_team, team_projection = frozen.build_team_projections(resolved)
        coverage = recipe["coverage_model"]
        team_projection["projected_team_sog"] = (
            coverage["intercept"] + coverage["coefficients"][0] * team_projection["aggregate_player_mean"]
            + coverage["coefficients"][1] * team_projection["listed_player_count"]
        )
        market_goals, market_counts = build_market_goals(market, market_map, sog_map, sog)
        translation = build_translation(outcomes, team_projection, market_goals, recipe)

        residual_frame = pd.read_csv(FROZEN_RUN / recipe["residual_distribution"]["artifact"], float_precision="round_trip")
        if frozen.canonical_frame_hash(residual_frame) != recipe["residual_distribution"]["sha256"]:
            raise ConfirmationFailure("frozen residual distribution content hash mismatch")
        residuals = residual_frame.sort_values(["game_id", "goalie_id"])["residual"].to_numpy(float)
        saves_sportsbook = saves[~saves["book_key"].isin(DFS_BOOKS)].copy()
        saves_pairs, saves_pair_counts, _ = frozen.pair_quotes(
            saves_sportsbook, ["event_id", "goalie_id", "book_key", "line"],
            "book_key", "goalie_id", False,
        )
        quote_universe, pricing_counts = price_quotes(
            saves_pairs, saves_map, outcomes, translation, residuals
        )
        primary_results = {}
        selected_frames, bootstrap_inputs, bootstrap_draws = [], [], []
        for side in ("OVER", "UNDER"):
            result, selected, inputs, draws = primary_bootstrap(quote_universe, side, THRESHOLDS[side])
            primary_results[side] = result
            selected_frames.append(selected)
            bootstrap_inputs.append(inputs)
            bootstrap_draws.append(draws)
            log(
                f"PRIMARY {side}: n={result['n_selected_graded_bets']} delta="
                f"{result['model_minus_blind_roi_delta']:+.6f} CI=[{result['bootstrap_lower_ci95']:+.6f},"
                f"{result['bootstrap_upper_ci95']:+.6f}] verdict={result['verdict']}"
            )
        selected_all = pd.concat(selected_frames, ignore_index=True)
        overall = overall_verdict(primary_results)

        # Registered secondaries; none feed the primary result or verdict.
        bettime_secondary = scoring_secondary(quote_universe, "2024-25 core-pass bettime")
        dev_quotes = pd.read_parquet(FROZEN_RUN / "development_saves_quote_universe.parquet")
        development_secondary = {
            "all_2023_24": scoring_secondary(dev_quotes, "2023-24 development all"),
            "validation_2023_24": scoring_secondary(
                dev_quotes[dev_quotes["split"] == "development_validation"], "2023-24 development validation"
            ),
        }
        closing_universe, closing_counts = build_closing_universe(closing, translation, residuals)
        closing_secondary = scoring_secondary(closing_universe, "2024-25 closing")
        clv_rows, clv_summary = selected_clv(selected_all, closing_universe)

        event_maps = pd.concat(
            [sog_map.assign(source="core_sog"), saves_map.assign(source="core_saves"),
             market_map.assign(source="market_state")], ignore_index=True, sort=False
        )
        unmatched = pd.concat(
            [sog_unmatched.assign(source="core_sog"), saves_unmatched.assign(source="core_saves"),
             market_unmatched.assign(source="market_state")], ignore_index=True, sort=False
        )
        event_maps.to_csv(output_dir / "recovery_event_mapping.csv", index=False)
        unmatched.to_csv(output_dir / "recovery_unmatched_events.csv", index=False)
        attributed.to_parquet(output_dir / "recovery_sog_player_universe.parquet", index=False)
        excluded_sog_lines.to_csv(output_dir / "recovery_excluded_sog_lines.csv", index=False)
        per_book_team.to_parquet(output_dir / "recovery_event_team_book_sog.parquet", index=False)
        team_projection.to_parquet(output_dir / "recovery_event_team_sog_projection.parquet", index=False)
        market_goals.to_parquet(output_dir / "recovery_market_goal_universe.parquet", index=False)
        translation.to_parquet(output_dir / "recovery_starter_translation_universe.parquet", index=False)
        quote_universe.to_parquet(output_dir / "recovery_saves_quote_universe.parquet", index=False)
        selected_all.to_parquet(output_dir / "recovery_primary_selected_rows.parquet", index=False)
        pd.concat(bootstrap_inputs, ignore_index=True).to_csv(output_dir / "recovery_bootstrap_cluster_inputs.csv", index=False)
        pd.concat(bootstrap_draws, ignore_index=True).to_parquet(output_dir / "recovery_bootstrap_draws.parquet", index=False)
        closing_universe.to_parquet(output_dir / "recovery_closing_quote_universe.parquet", index=False)
        clv_rows.to_parquet(output_dir / "recovery_selected_clv_rows.parquet", index=False)

        attr = attributed["attribution_status"].value_counts().to_dict()
        counts = {
            "event_mapping": {
                "sog_matched": len(sog_map), "sog_unmatched": len(sog_unmatched),
                "saves_matched": len(saves_map), "saves_unmatched": len(saves_unmatched),
                "market_matched": len(market_map), "market_unmatched": len(market_unmatched),
            },
            "sog_pairing": sog_pair_counts,
            "sog_attribution": {
                "resolved_quote_units": int(attr.get("resolved", 0)),
                "ambiguous_quote_units": int(attr.get("ambiguous", 0)),
                "unresolved_quote_units": int(attr.get("unresolved", 0)),
                "resolved_unique_event_players": int(resolved[["event_id", "player_name_raw"]].drop_duplicates().shape[0]),
                "unresolved_unique_event_players": int(attributed.loc[attributed["attribution_status"] == "unresolved", ["event_id", "player_name_raw"]].drop_duplicates().shape[0]),
                "ambiguous_unique_event_players": int(attributed.loc[attributed["attribution_status"] == "ambiguous", ["event_id", "player_name_raw"]].drop_duplicates().shape[0]),
                "poisson_inversion_failures": int(attributed["inversion_failure"].notna().sum()),
            },
            "sog_coverage": {
                "event_team_book_rows": len(per_book_team), "event_team_rows": len(team_projection),
                "events_with_both_teams": int((team_projection.groupby("game_id")["shooting_team"].nunique() == 2).sum()),
                "books_per_event_team": team_projection["n_sog_books"].value_counts().sort_index().astype(int).to_dict(),
            },
            "market_goals": market_counts,
            "translation": {
                "complete_rows": int(translation["predicted_starter_saves"].notna().sum()),
                "missing_rows": int(translation["predicted_starter_saves"].isna().sum()),
            },
            "saves_pairing": saves_pair_counts,
            "saves_pricing": pricing_counts,
            "closing": closing_counts,
        }
        write_json(output_dir / "recovery_join_exclusion_counts.json", counts)
        write_json(
            output_dir / "recovery_read_audit.json",
            {
                "stage": "recovery", "touch_marker_created_before_materialization": True,
                "touch_marker_sha256": sha256_file(TOUCH_MARKER),
                "predicate_level_reads_only": True, "old_21_event_fragment_loaded": False,
                "network_calls": 0, "betting_db_accesses": 0, "api_credits_used": 0,
                "reads": [asdict(record) for record in read_records],
                "pbp": {key: value for key, value in pbp_audit.items() if key != "files"},
            },
        )
        write_json(
            output_dir / "recovery_input_hashes.json",
            {
                "hash_policy": "Canonical SHA-256 of rows returned by each confirmation predicate; no all-season source hash/read.",
                "predicate_reads": [asdict(record) for record in read_records],
                "play_by_play_manifest": pbp_audit,
                "frozen_artifacts": {
                    "development_script_sha256": FROZEN_SCRIPT_SHA256,
                    "frozen_recipe_sha256": FROZEN_RECIPE_SHA256,
                    "development_output_manifest_sha256": preflight["manifest"]["manifest_sha256"],
                    "residual_csv_sha256": sha256_file(FROZEN_RUN / recipe["residual_distribution"]["artifact"]),
                },
            },
        )
        secondary = {
            "non_primary": True, "primary_unaffected_by_secondary_results": True,
            "development": development_secondary, "recovery_bettime": bettime_secondary,
            "recovery_closing": closing_secondary, "selected_same_book_exact_line_clv": clv_summary,
        }
        write_json(output_dir / "recovery_secondaries.json", secondary)
        write_json(
            output_dir / "recovery_primary_results.json",
            {"side_results": primary_results, "overall_verdict": overall,
             "semantics": "Section 15.2a/15.7 side-specific same-universe blind baseline."},
        )
        metadata = {
            "experiment": 12, "stage": "recovery", "status": "RECOVERY_COMPLETED",
            "started_at": started, "completed_at": utc_now(), "frozen_run": str(FROZEN_RUN.relative_to(REPO_ROOT)),
            "output_dir": str(output_dir.relative_to(REPO_ROOT)),
            "frozen_development_script_sha256": FROZEN_SCRIPT_SHA256,
            "frozen_recipe_sha256": FROZEN_RECIPE_SHA256,
            "recovery_runner_sha256": sha256_file(Path(__file__)),
            "touch_marker_sha256": sha256_file(TOUCH_MARKER),
            "thresholds": THRESHOLDS, "refit_performed": False, "reselection_performed": False,
            "old_21_event_fragment_loaded": False, "overall_primary_verdict": overall,
            "primary_results": primary_results,
            "secondaries_are_non_primary": True,
        }
        write_json(output_dir / "recovery_metadata.json", metadata)
        log(f"Analysis and artifact writes complete; overall primary verdict={overall}")
        write_json(output_dir / "recovery_output_manifest.json", output_manifest(output_dir))
        manifest_hash = sha256_file(output_dir / "recovery_output_manifest.json")
        # Validate the final confirmation manifest before recording completion.
        final_manifest = json.loads((output_dir / "recovery_output_manifest.json").read_text(encoding="utf-8"))
        for item in final_manifest["files"]:
            path = output_dir / item["path"]
            if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise ConfirmationFailure(f"final confirmation manifest mismatch: {item['path']}")
        completion = {
            "experiment": 12, "status": "COMPLETED", "completed_at": utc_now(),
            "touch_marker_sha256": sha256_file(TOUCH_MARKER),
            "recovery_output_dir": str(output_dir.relative_to(REPO_ROOT)),
            "recovery_manifest_sha256": manifest_hash,
            "recovery_runner_sha256": sha256_file(Path(__file__)),
            "overall_primary_verdict": overall,
        }
        write_exclusive_json(COMPLETION_MARKER, completion)
        append_history(completion)
        return output_dir
    except Exception as exc:
        failure = {
            "experiment": 12, "status": "FAILED_TOUCH_CONSUMED", "failed_at": utc_now(),
            "error_type": type(exc).__name__, "error": str(exc),
            "recovery_output_dir": str(output_dir.relative_to(REPO_ROOT)),
            "traceback": traceback.format_exc(),
        }
        try:
            write_json(output_dir / "recovery_failed_metadata.json", failure)
            write_exclusive_json(FAILURE_MARKER, failure)
            append_history(failure)
        finally:
            raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 12 one-touch confirmation runner")
    parser.add_argument("--mode", required=True, choices=("preflight", "confirm"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global FROZEN
    args = parse_args(argv)
    preflight, frozen, recipe = run_preflight(require_marker_absent=True)
    FROZEN = frozen
    if args.mode == "preflight":
        write_exclusive_json(PREFLIGHT_ARTIFACT, preflight)
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    if not PREFLIGHT_ARTIFACT.exists():
        raise PreflightFailure("persisted no-data preflight artifact is missing")
    persisted = json.loads(PREFLIGHT_ARTIFACT.read_text(encoding="utf-8"))
    if persisted.get("status") != "PREFLIGHT_PASS" or persisted.get("recovery_runner_sha256") != sha256_file(Path(__file__)):
        raise PreflightFailure("persisted preflight artifact does not match this confirmation runner")
    output_dir = run_confirmation(preflight, frozen, recipe)
    print(f"Confirmation completed at {output_dir}")
    return 0


FROZEN: Any = None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PreflightFailure, ConfirmationFailure) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        raise SystemExit(2)
