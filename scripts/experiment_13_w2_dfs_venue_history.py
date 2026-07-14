"""Experiment 13: W2 DFS venue-history census.

Section 16 fixes the chronological direction before any result is observed:
2024-25 is development and 2025-26 is confirmation.  This runner makes no
network calls and opens data/betting.db only through SQLite's read-only URI.

Run from the repository root with an environment containing pandas, numpy,
and pyarrow, for example:
  uv run --offline --no-project --with pandas --with numpy --with pyarrow \
    python scripts/experiment_13_w2_dfs_venue_history.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_SNAPSHOTS = REPO_ROOT / "data" / "processed" / "core_bettime_202607_snapshots.parquet"
NEW_SUMMARY = REPO_ROOT / "data" / "processed" / "core_bettime_202607_snapshots_summary.json"
OUTCOMES_PATH = REPO_ROOT / "data" / "processed" / "clean_training_data.parquet"
BETTING_DB = REPO_ROOT / "data" / "betting.db"
OUTPUT_ROOT = REPO_ROOT / "models" / "trained"
EXPERIMENT_PREFIX = "experiment_13_w2_dfs_venue_history_"

DEVELOPMENT_SEASON = "2024-25"
CONFIRMATION_SEASON = "2025-26"
DFS_BOOKS = {"prizepicks", "underdog"}
PRIMARY_DFS_BOOK = "prizepicks"
NON_SPORTSBOOK_BOOKS = DFS_BOOKS | {"unknown", "manual"}
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

TRACKER_COLUMNS = [
    "id",
    "game_date",
    "game_id",
    "book",
    "goalie_name",
    "goalie_id",
    "team_abbrev",
    "opponent_team",
    "is_home",
    "betting_line",
    "line_over",
    "line_under",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def canonical_row_checksum(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.to_dict("records"):
        normalized = {key: native(value) for key, value in row.items()}
        digest.update(json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            normalized = {key: native(value) for key, value in row.items()}
            handle.write(json.dumps(normalized, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: native(row.get(field)) for field in fields})


def make_logger(path: Path):
    def log(message: str) -> None:
        stamped = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(stamped)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")

    return log


def no_completed_census_exists() -> None:
    completed = []
    for candidate in OUTPUT_ROOT.glob(f"{EXPERIMENT_PREFIX}*/metadata.json"):
        try:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if metadata.get("run_status") == "completed":
            completed.append(str(candidate.parent))
    if completed:
        raise SystemExit(
            "A completed Experiment 13 census already exists. The preregistration forbids rerunning "
            f"a completed census: {completed}"
        )


def snapshot_tracker_price_extract() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Make the sole data/betting.db extract before any outcome read or grade."""
    if not BETTING_DB.exists():
        raise FileNotFoundError(f"Missing tracker database: {BETTING_DB}")
    db_uri = f"file:{BETTING_DB.as_posix()}?mode=ro"
    connection = sqlite3.connect(db_uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        query = """
            SELECT id, game_date, game_id, book, goalie_name, goalie_id,
                   team_abbrev, opponent_team, is_home, betting_line,
                   line_over, line_under
            FROM bets
            WHERE betting_line IS NOT NULL
            ORDER BY id
        """
        frame = pd.read_sql_query(query, connection)
    finally:
        connection.close()
    if list(frame.columns) != TRACKER_COLUMNS:
        raise AssertionError("Unexpected tracker extract columns; do not continue with an implicit schema mapping.")
    pin = {
        "path": str(BETTING_DB),
        "query_label": "all_tracker_price_rows_ordered_by_id",
        "outcome_columns_read": False,
        "row_count": int(len(frame)),
        "max_game_date": None if frame.empty else str(frame["game_date"].max()),
        "deterministic_row_checksum_sha256": canonical_row_checksum(frame),
        "database_file_sha256": sha256_file(BETTING_DB),
    }
    return frame, pin


def paired_new_pass_quotes(new_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    saves = new_rows[
        (new_rows["season"] == DEVELOPMENT_SEASON)
        & (new_rows["market_key"] == "player_total_saves")
    ].copy()
    fanatics_rows = int((saves["book_key"].str.lower() == "fanatics").sum())
    if fanatics_rows:
        raise RuntimeError("Section 14.5 schema surprise: Fanatics appeared in the new pass. Stop before census.")
    saves["book_key"] = saves["book_key"].str.lower()
    saves["side"] = saves["side"].str.lower()
    key = ["event_id", "goalie_id", "book_key", "line"]
    side_counts = saves.groupby(key, dropna=False)["side"].agg(lambda values: tuple(sorted(set(values))))
    valid_keys = side_counts[side_counts == ("over", "under")].index
    paired = saves.set_index(key, drop=False)
    paired = paired[paired.index.isin(valid_keys)].drop_duplicates(key).reset_index(drop=True)
    excluded_unpaired = int(len(saves) - len(paired) * 2)
    unresolved_paired_quotes = int(paired["goalie_id"].isna().sum())
    paired = paired[paired["goalie_id"].notna()].copy()
    per_book_count = paired.groupby(["event_id", "goalie_id", "book_key"], dropna=False).size()
    ambiguous_multi_line = per_book_count[per_book_count > 1]
    if not ambiguous_multi_line.empty:
        ambiguous_index = set(ambiguous_multi_line.index)
        paired_index = pd.MultiIndex.from_frame(paired[["event_id", "goalie_id", "book_key"]])
        paired = paired[~paired_index.isin(ambiguous_index)].copy()
    return paired, {
        "new_pass_saves_rows_before_pairing": int(len(saves)),
        "new_pass_paired_quote_rows": int(len(paired)),
        "new_pass_unpaired_side_rows_excluded": excluded_unpaired,
        "new_pass_unresolved_goalie_quote_rows_excluded": unresolved_paired_quotes,
        "new_pass_multi_line_goalie_book_units_excluded_fail_closed": int(len(ambiguous_multi_line)),
        "new_pass_fanatics_rows": fanatics_rows,
    }


def normalize_new_pass_census(paired: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sportsbook = paired[~paired["book_key"].isin(DFS_BOOKS)].copy()
    dfs = paired[paired["book_key"].isin(DFS_BOOKS)].copy()
    consensus = (
        sportsbook.groupby(["event_id", "goalie_id"], as_index=False)
        .agg(
            sportsbook_consensus_line=("line", "median"),
            sportsbook_quote_count=("book_key", "size"),
            sportsbook_books=("book_key", lambda values: ",".join(sorted(values))),
        )
    )
    normalized = dfs.merge(consensus, on=["event_id", "goalie_id"], how="left", validate="many_to_one")
    normalized["line_deviation"] = normalized["line"] - normalized["sportsbook_consensus_line"]
    normalized["exact_agreement"] = normalized["line_deviation"] == 0.0
    normalized["within_half_save"] = normalized["line_deviation"].abs() <= 0.5
    normalized["prospective_side"] = np.select(
        [normalized["line_deviation"] > 0, normalized["line_deviation"] < 0],
        ["UNDER", "OVER"],
        default="AGREEMENT_NO_BET",
    )
    normalized["grade_status"] = "not_graded_pending_2025_26_last_fetch_contract"
    normalized["season"] = DEVELOPMENT_SEASON
    normalized["data_source"] = "core_bettime_202607_snapshots"
    normalized["same_timestamp_evidence"] = normalized["resolved_ts"]
    required = [
        "season", "data_source", "event_id", "game_date_eastern", "goalie_id", "goalie_name_matched",
        "book_key", "line", "sportsbook_consensus_line", "sportsbook_quote_count", "sportsbook_books",
        "line_deviation", "exact_agreement", "within_half_save", "prospective_side", "grade_status",
        "same_timestamp_evidence",
    ]
    rows = [{field: native(row[field]) for field in required} for row in normalized.to_dict("records")]
    summary = {str(book): census_stats(group) for book, group in normalized.groupby("book_key")}
    return rows, summary


def normalize_tracker_census(tracker: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    tracker = tracker.copy()
    tracker["book_key"] = tracker["book"].str.lower()
    tracker["season"] = np.where(tracker["game_date"] < "2025-08-01", DEVELOPMENT_SEASON, CONFIRMATION_SEASON)
    group_cols = ["game_date", "game_id", "goalie_id", "book_key"]
    latest_index = tracker.groupby(group_cols, dropna=False)["id"].idxmax()
    latest = tracker.loc[latest_index].copy().reset_index(drop=True)
    latest = latest[latest["goalie_id"].notna()].copy()
    sportsbook = latest[~latest["book_key"].isin(NON_SPORTSBOOK_BOOKS)].copy()
    dfs = latest[latest["book_key"].isin(DFS_BOOKS)].copy()
    consensus = (
        sportsbook.groupby(["game_date", "game_id", "goalie_id"], as_index=False)
        .agg(
            sportsbook_consensus_line=("betting_line", "median"),
            sportsbook_quote_count=("book_key", "size"),
            sportsbook_books=("book_key", lambda values: ",".join(sorted(values))),
        )
    )
    normalized = dfs.merge(
        consensus, on=["game_date", "game_id", "goalie_id"], how="left", validate="many_to_one"
    )
    normalized["line_deviation"] = normalized["betting_line"] - normalized["sportsbook_consensus_line"]
    normalized["exact_agreement"] = normalized["line_deviation"] == 0.0
    normalized["within_half_save"] = normalized["line_deviation"].abs() <= 0.5
    normalized["prospective_side"] = np.select(
        [normalized["line_deviation"] > 0, normalized["line_deviation"] < 0],
        ["UNDER", "OVER"],
        default="AGREEMENT_NO_BET",
    )
    normalized["data_source"] = "betting.db"
    normalized["same_timestamp_evidence"] = "max(id) per game_date/game_id/goalie_id/book persisted append sequence"
    result = {
        "last_fetch_reconstruction": "max(id) per game_date, game_id, goalie_id, book",
        "why_allowed": "BettingDB.append_games inserts distinct quotes in SQLite AUTOINCREMENT order; record_bet and fetch_and_predict both select the current line by descending id/insertion order.",
        "limitation": "This is the latest persisted distinct append, not a wall-clock fetch timestamp or true same-second alignment. A later identical or reverted quote can be skipped by the unique key and is not recoverable.",
        "line_snapshots_historical_rows": 0,
        "tracker_rows_before_last_append_reconstruction": int(len(tracker)),
        "tracker_rows_after_last_append_reconstruction": int(len(latest)),
        "tracker_dfs_rows_after_last_append_reconstruction": int(len(dfs)),
        "named_sportsbook_rows_after_last_append_reconstruction": int(len(sportsbook)),
        "excluded_non_sportsbook_books": sorted(NON_SPORTSBOOK_BOOKS),
    }
    return normalized, result


def census_stats(frame: pd.DataFrame) -> dict[str, Any]:
    eligible = frame[frame["sportsbook_consensus_line"].notna()].copy()
    deviations = eligible[~eligible["exact_agreement"]]
    return {
        "n_goalie_night_dfs_book_rows": int(len(frame)),
        "n_with_sportsbook_consensus": int(len(eligible)),
        "exact_agreement_rate": None if eligible.empty else float(eligible["exact_agreement"].mean()),
        "within_half_save_rate": None if eligible.empty else float(eligible["within_half_save"].mean()),
        "n_deviations": int(len(deviations)),
        "mean_signed_deviation": None if deviations.empty else float(deviations["line_deviation"].mean()),
        "n_dfs_above_consensus_prospective_under": int((deviations["line_deviation"] > 0).sum()),
        "n_dfs_below_consensus_prospective_over": int((deviations["line_deviation"] < 0).sum()),
    }


def prior_reconciliation_rows(tracker_normalized: pd.DataFrame, reconstruction: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "registered_definition": (
            "one goalie-night-DFS-book row per calendar day; last fetch; same-calendar-day median across all "
            "sportsbooks; exact primary and +/-0.5 secondary"
        ),
        "registered_recompute_status": "recomputed_with_max_id_latest_persisted_append_reconstruction",
        "registered_recompute_reason": reconstruction["limitation"],
    }
    return [
        {
            "prior_claim": "2026-07-07 95.2% exact agreement",
            "original_numerator": 236,
            "original_denominator": 248,
            "original_definition": "Jan-Mar 2026 Underdog goalie-nights versus one sharp book, not all-book median",
            "what_can_be_reproduced": "The Jan-Mar 2026 Underdog question can be recomputed under the registered all-book-median definition using max(id) latest persisted append rows.",
            "what_cannot_be_reproduced": "The original scratch script and single-sharp-book pairing are not persisted, so 95.2% itself cannot be reproduced as the original computation.",
            **common,
        },
        {
            "prior_claim": "2026-07-13 90.1% exact agreement",
            "original_numerator": 265,
            "original_denominator": 294,
            "original_definition": "Underdog rows versus an unspecified sportsbook consensus over an unspecified window",
            "what_can_be_reproduced": "The full persisted tracker universe can be recomputed under the registered definition using max(id) latest persisted append rows.",
            "what_cannot_be_reproduced": "The scratch script and its original window, aggregation, and row-level dedup are not persisted, so 90.1% itself cannot be reproduced as the original computation.",
            **common,
        },
        {
            "prior_claim": "2026-07-13 PrizePicks 78.1% exact agreement",
            "original_numerator": 50,
            "original_denominator": 64,
            "original_definition": "PrizePicks tracker rows under the exploratory recon's unspecified consensus",
            "what_can_be_reproduced": "The full persisted PrizePicks tracker universe can be recomputed under the registered definition using max(id) latest persisted append rows.",
            "what_cannot_be_reproduced": "The original 64-row selection and consensus are not persisted, so 78.1% itself cannot be reproduced as the original computation.",
            **common,
        },
    ]


def snapshot_tracker_outcome_extract() -> tuple[pd.DataFrame, dict[str, Any]]:
    db_uri = f"file:{BETTING_DB.as_posix()}?mode=ro"
    connection = sqlite3.connect(db_uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        frame = pd.read_sql_query(
            "SELECT id, game_date, game_id, goalie_id, actual_saves FROM bets ORDER BY id", connection
        )
    finally:
        connection.close()
    return frame, {
        "path": str(BETTING_DB),
        "query_label": "all_tracker_outcome_rows_ordered_by_id",
        "row_count": int(len(frame)),
        "max_game_date": None if frame.empty else str(frame["game_date"].max()),
        "deterministic_row_checksum_sha256": canonical_row_checksum(frame),
        "database_file_sha256": sha256_file(BETTING_DB),
    }


def cluster_bootstrap_mean_ci(values: np.ndarray, cluster_ids: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids, dtype=object)
    mask = ~np.isnan(values)
    values, cluster_ids = values[mask], cluster_ids[mask]
    if len(values) == 0:
        return {"mean": None, "lower": None, "upper": None, "n_bets": 0, "n_clusters": 0}
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_sum = np.zeros(len(clusters))
    cluster_count = np.zeros(len(clusters))
    np.add.at(cluster_sum, inverse, values)
    np.add.at(cluster_count, inverse, 1)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    draws = rng.randint(0, len(clusters), size=(N_BOOTSTRAP, len(clusters)))
    counts = np.apply_along_axis(lambda draw: np.bincount(draw, minlength=len(clusters)), 1, draws)
    means = (counts @ cluster_sum) / (counts @ cluster_count)
    return {
        "mean": float(values.mean()),
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
        "n_bets": int(len(values)),
        "n_clusters": int(len(clusters)),
    }


def attach_and_grade(new_rows: list[dict[str, Any]], tracker: pd.DataFrame, log) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Open and pin outcomes only after all price-level reconstruction choices are fixed."""
    tracker_outcomes, tracker_outcome_pin = snapshot_tracker_outcome_extract()
    clean = pd.read_parquet(OUTCOMES_PATH, columns=["game_date", "game_id", "goalie_id", "saves"])
    clean["game_date"] = pd.to_datetime(clean["game_date"]).dt.strftime("%Y-%m-%d")
    clean = clean[["game_date", "game_id", "goalie_id", "saves"]].drop_duplicates(["game_date", "goalie_id"])

    development = pd.DataFrame(new_rows)
    development = development.rename(columns={"game_date_eastern": "game_date", "line": "dfs_line"})
    development = development.merge(clean[["game_date", "goalie_id", "saves"]], on=["game_date", "goalie_id"], how="left")
    development["actual_saves"] = development["saves"]
    development["cluster_id"] = development["event_id"].astype(str) + ":" + development["goalie_id"].astype(str)

    confirmation = tracker.copy()
    confirmation = confirmation.rename(columns={"betting_line": "dfs_line"})
    confirmation = confirmation.merge(tracker_outcomes[["id", "actual_saves"]], on="id", how="left", validate="one_to_one")
    confirmation["event_id"] = None
    confirmation["cluster_id"] = confirmation["game_id"].astype(str) + ":" + confirmation["goalie_id"].astype(str)
    confirmation["goalie_name_matched"] = confirmation["goalie_name"]

    all_rows = pd.concat([development, confirmation], ignore_index=True, sort=False)
    all_rows["is_push"] = all_rows["actual_saves"] == all_rows["dfs_line"]
    all_rows["grade"] = None
    all_rows["profit_even_money_approximation"] = np.nan
    bet_mask = (
        (all_rows["book_key"] == PRIMARY_DFS_BOOK)
        & all_rows["sportsbook_consensus_line"].notna()
        & (all_rows["prospective_side"] != "AGREEMENT_NO_BET")
        & all_rows["actual_saves"].notna()
        & ~all_rows["is_push"]
    )
    under_win = (all_rows["prospective_side"] == "UNDER") & (all_rows["actual_saves"] < all_rows["dfs_line"])
    over_win = (all_rows["prospective_side"] == "OVER") & (all_rows["actual_saves"] > all_rows["dfs_line"])
    all_rows.loc[bet_mask & (under_win | over_win), "grade"] = "WIN"
    all_rows.loc[bet_mask & ~(under_win | over_win), "grade"] = "LOSS"
    all_rows.loc[all_rows["grade"] == "WIN", "profit_even_money_approximation"] = 1.0
    all_rows.loc[all_rows["grade"] == "LOSS", "profit_even_money_approximation"] = -1.0
    log(
        "Snapshot-pinned the tracker outcome extract before grading: "
        f"rows={tracker_outcome_pin['row_count']}, max_game_date={tracker_outcome_pin['max_game_date']}, "
        f"row_checksum={tracker_outcome_pin['deterministic_row_checksum_sha256']}."
    )
    return all_rows, tracker_outcome_pin, {
        "path": str(OUTCOMES_PATH), "bytes": OUTCOMES_PATH.stat().st_size, "sha256": sha256_file(OUTCOMES_PATH)
    }


def graded_summary(rows: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    prize = rows[(rows["book_key"] == PRIMARY_DFS_BOOK) & rows["sportsbook_consensus_line"].notna()].copy()
    for season in (DEVELOPMENT_SEASON, CONFIRMATION_SEASON):
        season_rows = prize[prize["season"] == season]
        population: dict[str, Any] = {"census": census_stats(season_rows)}
        for label, subset in {
            "agreeing_population": season_rows[season_rows["exact_agreement"]],
            "deviating_population": season_rows[~season_rows["exact_agreement"]],
        }.items():
            gradeable = subset[subset["actual_saves"].notna() & ~subset["is_push"]]
            base_under = None if gradeable.empty else float((gradeable["actual_saves"] < gradeable["dfs_line"]).mean())
            population[label] = {
                "n_rows": int(len(subset)),
                "n_gradeable_nonpush": int(len(gradeable)),
                "under_at_dfs_line_rate": base_under,
            }
        deviations = season_rows[season_rows["grade"].notna()].copy()
        if deviations.empty:
            population["deviation_outcome_grade"] = {"n_bets": 0, "hit_rate": None, "even_money_profit_per_bet": None, "cluster_bootstrap_ci95": {"mean": None, "lower": None, "upper": None, "n_bets": 0, "n_clusters": 0}}
        else:
            ci = cluster_bootstrap_mean_ci(
                deviations["profit_even_money_approximation"].to_numpy(), deviations["cluster_id"].to_numpy(),
            )
            population["deviation_outcome_grade"] = {
                "n_bets": int(len(deviations)),
                "hit_rate": float((deviations["grade"] == "WIN").mean()),
                "even_money_profit_per_bet": float(deviations["profit_even_money_approximation"].mean()),
                "cluster_bootstrap_ci95": ci,
            }
        result[season] = population
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 13's preregistered W2 DFS venue-history census.")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    no_completed_census_exists()
    output_dir = OUTPUT_ROOT / f"{EXPERIMENT_PREFIX}{args.run_id}"
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    log = make_logger(output_dir / "run_log.txt")
    log("Experiment 13 started. Chronological direction is fixed: 2024-25 development -> 2025-26 confirmation.")
    log("No network/API calls are implemented. data/betting.db will be opened once through a read-only URI before any outcome data is read.")

    for path in (NEW_SNAPSHOTS, NEW_SUMMARY, OUTCOMES_PATH, BETTING_DB):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    tracker, tracker_pin = snapshot_tracker_price_extract()
    input_checksums = {
        "core_bettime_snapshots": {"path": str(NEW_SNAPSHOTS), "bytes": NEW_SNAPSHOTS.stat().st_size, "sha256": sha256_file(NEW_SNAPSHOTS)},
        "core_bettime_summary": {"path": str(NEW_SUMMARY), "bytes": NEW_SUMMARY.stat().st_size, "sha256": sha256_file(NEW_SUMMARY)},
        "betting_db_price_extract": tracker_pin,
        "execution_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    write_json(output_dir / "input_checksums.json", input_checksums)
    log(
        "Snapshot-pinned the sole tracker price extract before outcome grading: "
        f"rows={tracker_pin['row_count']}, max_game_date={tracker_pin['max_game_date']}, "
        f"row_checksum={tracker_pin['deterministic_row_checksum_sha256']}."
    )

    new_rows = pd.read_parquet(NEW_SNAPSHOTS)
    paired, ingestion = paired_new_pass_quotes(new_rows)
    normalized_new, new_summary = normalize_new_pass_census(paired)
    tracker_normalized, reconstruction = normalize_tracker_census(tracker)
    tracker_normalized["grade_status"] = "pending_outcome_grading"
    tracker_normalized["line"] = tracker_normalized["betting_line"]
    tracker_normalized["game_date_eastern"] = tracker_normalized["game_date"]
    tracker_normalized["goalie_name_matched"] = tracker_normalized["goalie_name"]
    tracker_fields = [
        "season", "data_source", "game_id", "game_date_eastern", "goalie_id", "goalie_name_matched", "book_key",
        "line", "sportsbook_consensus_line", "sportsbook_quote_count", "sportsbook_books", "line_deviation",
        "exact_agreement", "within_half_save", "prospective_side", "grade_status", "same_timestamp_evidence", "id",
    ]
    tracker_rows = [{field: native(row[field]) for field in tracker_fields} for row in tracker_normalized.to_dict("records")]
    normalized_rows = normalized_new + tracker_rows
    write_jsonl(output_dir / "normalized_census_rows.jsonl", normalized_rows)
    reconciliation = prior_reconciliation_rows(tracker_normalized, reconstruction)
    jan_mar = tracker_normalized[
        (tracker_normalized["book_key"] == "underdog")
        & (tracker_normalized["game_date"] >= "2026-01-01")
        & (tracker_normalized["game_date"] <= "2026-03-31")
    ]
    full_underdog = tracker_normalized[tracker_normalized["book_key"] == "underdog"]
    full_prizepicks = tracker_normalized[tracker_normalized["book_key"] == "prizepicks"]
    for row, recomputed in zip(reconciliation, (jan_mar, full_underdog, full_prizepicks)):
        stats = census_stats(recomputed)
        row["registered_recompute_n"] = stats["n_with_sportsbook_consensus"]
        row["registered_recompute_exact_agreement_rate"] = stats["exact_agreement_rate"]
        row["registered_recompute_within_half_save_rate"] = stats["within_half_save_rate"]
    reconciliation_fields = [
        "prior_claim", "original_numerator", "original_denominator", "original_definition", "registered_definition",
        "registered_recompute_status", "registered_recompute_reason", "registered_recompute_n",
        "registered_recompute_exact_agreement_rate", "registered_recompute_within_half_save_rate",
        "what_can_be_reproduced", "what_cannot_be_reproduced",
    ]
    write_csv(output_dir / "prior_reconciliation_rows.csv", reconciliation, reconciliation_fields)

    all_graded, tracker_outcome_pin, clean_outcome_pin = attach_and_grade(normalized_new, tracker_normalized, log)
    input_checksums["betting_db_outcome_extract"] = tracker_outcome_pin
    input_checksums["clean_training_data_outcomes"] = clean_outcome_pin
    write_json(output_dir / "input_checksums.json", input_checksums)

    graded_fields = [
        "season", "event_id", "game_id", "goalie_id", "book_key", "dfs_line", "sportsbook_consensus_line",
        "line_deviation", "prospective_side", "actual_saves", "grade", "profit_even_money_approximation",
    ]
    graded_deviations = all_graded[(all_graded["book_key"] == PRIMARY_DFS_BOOK) & all_graded["grade"].notna()]
    write_csv(output_dir / "graded_deviation_rows.csv", graded_deviations.to_dict("records"), graded_fields)
    outcome_summary = graded_summary(all_graded)
    development_ci = outcome_summary[DEVELOPMENT_SEASON]["deviation_outcome_grade"]["cluster_bootstrap_ci95"]
    confirmation_ci = outcome_summary[CONFIRMATION_SEASON]["deviation_outcome_grade"]["cluster_bootstrap_ci95"]
    chronological_holds = all(
        ci["lower"] is not None and ci["lower"] > 0 for ci in (development_ci, confirmation_ci)
    )

    summary = {
        "chronological_direction": f"{DEVELOPMENT_SEASON} development -> {CONFIRMATION_SEASON} confirmation",
        "ingestion": ingestion,
        "development_2024_25_census": new_summary,
        "confirmation_2025_26_census": {
            book: census_stats(tracker_normalized[tracker_normalized["book_key"] == book])
            for book in sorted(DFS_BOOKS)
        },
        "prior_reconciliation": reconciliation,
        "tracker_last_fetch_reconstruction": reconstruction,
        "outcome_grading": {
            "performed": True,
            "prizepicks_only": True,
            "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "performed": True},
            "real_roi_reported": False,
            "even_money_approximation_reported": True,
            "even_money_disclosure": "Profit per bet assumes +1 for a win and -1 for a loss. It is not PrizePicks payout economics and is not reported as ROI.",
            "by_season_and_population": outcome_summary,
        },
        "underdog": "Descriptive only. No Underdog outcome grading, CI, or ROI is performed.",
    }
    write_json(output_dir / "census_summary.json", summary)

    exact_verdict = (
        "CHRONOLOGICAL BAR CLEARED: development and confirmation PrizePicks deviation-grade cluster CIs both clear zero."
        if chronological_holds else
        "CENSUS NULL FOR EDGE LANGUAGE: the PrizePicks deviation-grade result does not clear the fixed 2024-25 development -> 2025-26 confirmation bar."
    )
    exact_consequence = (
        "The deviation-selection mechanism is only a development candidate for a filter stacked on model EV and requires its own preregistration before any confirmatory touch; it is not a standalone strategy."
        if chronological_holds else
        "DFS venue staleness is not pursued further this cycle. This is a census finding, not a bettable-edge claim."
    )
    metadata = {
        "experiment": "Experiment 13 - W2 DFS venue-history census",
        "registration": "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 16",
        "run_status": "completed",
        "completed_census": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "chronological_direction_fixed_before_results": {
            "development": DEVELOPMENT_SEASON,
            "confirmation": CONFIRMATION_SEASON,
        },
        "agreement_definition": {
            "unit": "one goalie-night-DFS-book quote per calendar day",
            "dedup": "max(id) latest persisted distinct append per calendar day/book; established production insertion-order reconstruction",
            "comparator": "same-calendar-day median line across all sportsbooks",
            "primary_tolerance": "exact",
            "secondary_tolerance": "+/-0.5 saves",
        },
        "exact_verdict": exact_verdict,
        "exact_consequence": exact_consequence,
        "limitations": [
            "Pre-migration line_snapshots has zero rows, so true same-second matching is unavailable.",
            reconstruction["limitation"],
            "The original 95.2%, 90.1%, and 78.1% scripts/definitions were not persisted, so their exact original computations cannot be reproduced.",
            "PrizePicks and Underdog tracker prices are placeholder values and are never used for ROI.",
        ],
        "artifacts": {
            "normalized_census_rows": "normalized_census_rows.jsonl",
            "prior_reconciliation_rows": "prior_reconciliation_rows.csv",
            "graded_deviation_rows": "graded_deviation_rows.csv",
            "input_checksums": "input_checksums.json",
            "run_log": "run_log.txt",
            "summary": "census_summary.json",
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    log(exact_verdict)
    log(exact_consequence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
