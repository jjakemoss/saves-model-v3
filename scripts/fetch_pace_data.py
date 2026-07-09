"""
Fetch and normalize MoneyPuck pace/xG data for the pace/xG ingestion
experiment (see docs/OFFSEASON_OPTIMIZATION_PLAN.md section 3.14,
"Component 1 -- ingestion, scripts/fetch_pace_data.py").

Downloads (idempotent -- skipped if already cached unless --force):
  - MoneyPuck team game-by-game file (all seasons 2008-present, one row per
    team per situation per game):
    https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv
  - MoneyPuck per-season goalie game logs (one zip per season):
    https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/goalies/{YEAR}.zip

Normalizes both into:
  - data/raw/moneypuck/team_games.parquet   (season >= 2021, all situations,
    regular season + playoffs, playoffGame flag preserved)
  - data/raw/moneypuck/goalie_games.parquet (seasons 2021-2025, all
    situations)

Season 2021 (2021-22) is fetched only to supply prior-season baselines for
the clean training frame's 2022-23 rows; the clean frame itself starts at
season 2022-23.

Cross-validates MoneyPuck shot-attempt totals (situation="all", regular
season only) against an independent source, the NHL stats API "team
realtime" report, before anything downstream is allowed to trust this data.
Primary metric is Corsi (MoneyPuck shotAttemptsFor vs NHL
totalShotAttempts); for seasons where the NHL report's totalShotAttempts is
entirely null (verified true for 20212022, populated from 20222023 onward)
it falls back to Fenwick (MoneyPuck unblockedShotAttemptsFor vs NHL
shots + missedShots) with the same tolerance and pass bars.
Hard-fails (nonzero exit) if the cross-validation bar is missed.

Also reports how well team_games.parquet / goalie_games.parquet cover the
repo's existing training keys (data/processed/clean_training_data.parquet),
so join gaps are visible immediately rather than discovered later in the
Component 2 feature builder.

MoneyPuck permits pulling these download files with attribution (per ToS
review recorded in the offseason plan, 2026-07-09).

Usage:
    python scripts/fetch_pace_data.py
    python scripts/fetch_pace_data.py --seasons 2021 2022 2023 2024 2025 --force
    python scripts/fetch_pace_data.py --raw-dir data/raw/moneypuck
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.parse
import zipfile
from pathlib import Path

import pandas as pd
import requests

DEFAULT_RAW_DIR = Path("data/raw/moneypuck")
DEFAULT_SEASONS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_CLEAN = Path("data/processed/clean_training_data.parquet")

TEAM_CSV_URL = (
    "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"
)
GOALIE_ZIP_URL_TEMPLATE = (
    "https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/goalies/{year}.zip"
)
NHL_REALTIME_URL_TEMPLATE = (
    "https://api.nhle.com/stats/rest/en/team/realtime"
    "?isAggregate=false&isGame=true&limit=-1&cayenneExp={cayenne}"
)

# A generic browser UA. The NHL stats API 403s generic non-browser-looking
# clients (verified 2026-07-09); MoneyPuck/peter-tanner do not appear to
# require it but there is no downside to sending it everywhere.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
    "saves-model-v3-pace-ingestion/1.0"
)
REQUEST_HEADERS = {"User-Agent": USER_AGENT, "Accept": "*/*"}

# Pace-relevant columns kept from the MoneyPuck team file, per the
# Component 1 spec: attempts (Corsi), fenwick (unblocked attempts), xG
# for+against, score-adjusted variants, danger-tier xG, iceTime, and the
# join/identity columns (gameId, gameDate, team, opposingTeam, home_or_away,
# situation, season, playoffGame).
TEAM_COLUMNS = [
    "gameId",
    "season",
    "gameDate",
    "team",
    "opposingTeam",
    "home_or_away",
    "situation",
    "playoffGame",
    "iceTime",
    "shotAttemptsFor",
    "shotAttemptsAgainst",
    "unblockedShotAttemptsFor",
    "unblockedShotAttemptsAgainst",
    "xGoalsFor",
    "xGoalsAgainst",
    "scoreAdjustedShotsAttemptsFor",
    "scoreAdjustedShotsAttemptsAgainst",
    "scoreAdjustedUnblockedShotAttemptsFor",
    "scoreAdjustedUnblockedShotAttemptsAgainst",
    "scoreVenueAdjustedxGoalsFor",
    "scoreVenueAdjustedxGoalsAgainst",
    "lowDangerxGoalsFor",
    "mediumDangerxGoalsFor",
    "highDangerxGoalsFor",
    "lowDangerxGoalsAgainst",
    "mediumDangerxGoalsAgainst",
    "highDangerxGoalsAgainst",
]

# Cross-validation thresholds, from the Component 1 spec.
DIFF_TOLERANCE = 2
DIFF_PASS_RATE_BAR = 0.99
JOIN_COVERAGE_BAR = 0.995
TRAINING_KEY_COVERAGE_WARN_BAR = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and normalize MoneyPuck team/goalie game logs, "
            "cross-validate against the NHL stats API, and report training-"
            "key join coverage."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONS,
        help=(
            "Season-start years to fetch goalie zips / cross-validate "
            "(default: 2021 2022 2023 2024 2025)."
        ),
    )
    parser.add_argument(
        "--clean",
        type=Path,
        default=DEFAULT_CLEAN,
        help="Path to clean_training_data.parquet for the training-key coverage check.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if a cached copy already exists.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_file(url: str, dest_path: Path, force: bool = False, timeout: int = 120) -> Path:
    """Stream-download url to dest_path. Idempotent unless force=True.

    Retries once on failure (two attempts total), per spec.
    """
    if dest_path.exists() and not force:
        size = dest_path.stat().st_size
        print(
            f"[INFO] Cached file exists, skipping download: {dest_path} "
            f"({size:,} bytes). Pass --force to re-download."
        )
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".tmp")

    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            print(f"[INFO] Downloading {url} (attempt {attempt}/2) -> {dest_path}")
            t0 = time.time()
            with requests.get(url, headers=REQUEST_HEADERS, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                expected = resp.headers.get("Content-Length")
                written = 0
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        written += len(chunk)
            tmp_path.replace(dest_path)
            elapsed = time.time() - t0
            expected_note = f", expected {int(expected):,}" if expected else ""
            print(
                f"[INFO] Downloaded {dest_path} ({written:,} bytes{expected_note}) "
                f"in {elapsed:.1f}s"
            )
            return dest_path
        except Exception as exc:  # noqa: BLE001 - report and retry once
            last_exc = exc
            print(f"[WARNING] Download attempt {attempt}/2 failed for {url}: {exc}")
            if tmp_path.exists():
                tmp_path.unlink()

    raise RuntimeError(f"[ERROR] Failed to download {url} after 2 attempts: {last_exc}")


def fetch_json(url: str, timeout: int = 60) -> dict:
    """GET url and return parsed JSON. Retries once on failure."""
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - report and retry once
            last_exc = exc
            print(f"[WARNING] Fetch attempt {attempt}/2 failed for {url}: {exc}")
    raise RuntimeError(f"[ERROR] Failed to fetch {url} after 2 attempts: {last_exc}")


def download_team_csv(raw_dir: Path, force: bool) -> Path:
    dest = raw_dir / "all_teams.csv"
    return download_file(TEAM_CSV_URL, dest, force=force)


def download_goalie_zip(raw_dir: Path, year: int, force: bool) -> Path:
    url = GOALIE_ZIP_URL_TEMPLATE.format(year=year)
    dest = raw_dir / f"goalies_{year}.zip"
    return download_file(url, dest, force=force)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def build_team_games(csv_path: Path, min_season: int = 2023) -> pd.DataFrame:
    print(f"[INFO] Reading {csv_path} (columns limited to pace-relevant set)...")
    df = pd.read_csv(csv_path, usecols=TEAM_COLUMNS)
    print(f"[INFO] Read {len(df):,} raw rows across all seasons/situations.")

    missing = [c for c in TEAM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Team CSV missing expected columns: {missing}")

    df = df[df["season"] >= min_season].copy()
    df["gameDate"] = pd.to_datetime(df["gameDate"].astype(str), format="%Y%m%d")
    df = df.sort_values(["season", "gameId", "team", "situation"]).reset_index(drop=True)

    dup_key = ["gameId", "team", "situation"]
    if df.duplicated(dup_key).any():
        dupes = df.loc[df.duplicated(dup_key, keep=False), dup_key]
        raise ValueError(
            f"[ERROR] team_games has duplicate (gameId, team, situation) rows:\n{dupes.head(10)}"
        )

    print(f"[INFO] Filtered to season >= {min_season}: {len(df):,} rows.")
    return df


def build_goalie_games(raw_dir: Path, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for year in seasons:
        zip_path = raw_dir / f"goalies_{year}.zip"
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError(
                    f"[ERROR] Expected exactly one CSV in {zip_path}, found {names}"
                )
            with zf.open(names[0]) as f:
                season_df = pd.read_csv(io.BytesIO(f.read()))
        season_df["gameDate"] = pd.to_datetime(
            season_df["gameDate"].astype(str), format="%Y%m%d"
        )
        print(f"[INFO] Read {len(season_df):,} rows from {zip_path.name} (season {year}).")
        frames.append(season_df)

    goalie_games = pd.concat(frames, ignore_index=True)
    goalie_games = goalie_games.sort_values(
        ["season", "gameId", "playerId", "situation"]
    ).reset_index(drop=True)

    dup_key = ["gameId", "playerId", "situation"]
    if goalie_games.duplicated(dup_key).any():
        dupes = goalie_games.loc[
            goalie_games.duplicated(dup_key, keep=False), dup_key
        ]
        raise ValueError(
            f"[ERROR] goalie_games has duplicate (gameId, playerId, situation) rows:\n{dupes.head(10)}"
        )

    game_types = goalie_games["gameId"].astype(str).str.slice(4, 6).unique()
    print(
        f"[INFO] goalie_games total rows: {len(goalie_games):,}. "
        f"gameId type codes present: {sorted(game_types)} "
        "('02' = regular season, '03' = playoffs; MoneyPuck's per-season "
        "goalie zip files were found to contain regular-season games only)."
    )
    return goalie_games


# ---------------------------------------------------------------------------
# Cross-validation against the NHL stats API
# ---------------------------------------------------------------------------


def fetch_nhl_realtime(season_year: int) -> pd.DataFrame:
    season_id = f"{season_year}{season_year + 1}"
    cayenne = f"gameTypeId=2 and seasonId={season_id}"
    url = NHL_REALTIME_URL_TEMPLATE.format(cayenne=urllib.parse.quote(cayenne))
    print(f"[INFO] Fetching NHL realtime report for season {season_id}...")
    payload = fetch_json(url)
    rows = payload.get("data", [])
    df = pd.DataFrame(rows)
    print(f"[INFO] NHL realtime report for {season_id}: {len(df):,} team-game rows.")
    return df


def resolve_nhl_team_abbrev(nhl_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve each NHL realtime row's own team abbreviation.

    The NHL realtime report carries teamId/teamFullName and
    opponentTeamAbbrev per row, but NOT the row's own team abbreviation.
    Since gameTypeId=2 games are always exactly two teams, the trick is to
    self-join each gameId's rows and take the *other* row's
    opponentTeamAbbrev as this row's own abbreviation -- e.g. if PIT's row
    for gameId X says opponentTeamAbbrev=LAK, then LAK's row for the same
    gameId is the one whose team_abbrev is being resolved as PIT (taken from
    PIT's own opponentTeamAbbrev field on LAK's row). This is derived
    entirely from the season's own data, so it is automatically robust to
    franchise abbreviation changes (e.g. ARI -> UTA) without a hardcoded
    teamId -> abbrev table.
    """
    game_sizes = nhl_df.groupby("gameId").size()
    bad_games = game_sizes[game_sizes != 2]
    if len(bad_games) > 0:
        print(
            f"[WARNING] {len(bad_games)} gameId(s) do not have exactly 2 team rows "
            f"in the NHL realtime report; they will not be resolvable/joinable. "
            f"Examples: {bad_games.head(5).to_dict()}"
        )
    valid_games = game_sizes[game_sizes == 2].index
    df = nhl_df[nhl_df["gameId"].isin(valid_games)].copy()

    merged = df.merge(
        df[["gameId", "teamId", "opponentTeamAbbrev"]],
        on="gameId",
        suffixes=("", "_other"),
    )
    merged = merged[merged["teamId"] != merged["teamId_other"]]
    merged = merged.drop_duplicates(["gameId", "teamId"])
    merged["team_abbrev"] = merged["opponentTeamAbbrev_other"]
    merged = merged.drop(columns=["teamId_other", "opponentTeamAbbrev_other"])

    # Sanity check: within one season, each teamId should resolve to exactly
    # one abbreviation (franchise renames happen between seasons, not within
    # one, and this function is always called per-season).
    per_team = merged.groupby("teamId")["team_abbrev"].nunique()
    inconsistent = per_team[per_team > 1]
    if len(inconsistent) > 0:
        raise ValueError(
            f"[ERROR] teamId(s) resolved to more than one abbreviation within "
            f"a single season: {inconsistent.to_dict()}"
        )
    return merged


def cross_validate(team_games: pd.DataFrame, seasons: list[int]) -> None:
    print("\n[INFO] === Cross-validation vs NHL stats API (team realtime report) ===")

    per_season_results = {}
    all_matched_diffs = []
    all_nhl_rows = 0
    all_matched_rows = 0

    for year in seasons:
        season_id = f"{year}{year + 1}"
        nhl_raw = fetch_nhl_realtime(year)
        nhl = resolve_nhl_team_abbrev(nhl_raw)

        # Metric selection. Primary is Corsi (shotAttemptsFor vs
        # totalShotAttempts). The NHL realtime report's totalShotAttempts is
        # entirely null for seasonId=20212022 (populated from 20222023
        # onward, verified 2026-07-09); for such seasons fall back to a
        # Fenwick comparison: unblockedShotAttemptsFor vs shots+missedShots.
        # A season with PARTIALLY null totalShotAttempts is unexpected and
        # hard-fails rather than silently mixing metrics.
        if "totalShotAttempts" not in nhl.columns or nhl["totalShotAttempts"].isna().all():
            metric = "fenwick"
            print(
                f"[INFO] Season {season_id}: totalShotAttempts unavailable from NHL API; "
                "cross-validating Fenwick (unblockedShotAttemptsFor vs shots+missedShots) instead."
            )
        elif nhl["totalShotAttempts"].isna().any():
            n_null = int(nhl["totalShotAttempts"].isna().sum())
            print(
                f"[ERROR] Season {season_id}: totalShotAttempts is partially null "
                f"({n_null:,}/{len(nhl):,} rows) in the NHL realtime report. This is "
                "unexpected; refusing to silently mix Corsi and Fenwick metrics."
            )
            raise SystemExit(1)
        else:
            metric = "corsi"

        mp_season = team_games[
            (team_games["season"] == year)
            & (team_games["situation"] == "all")
            & (team_games["playoffGame"] == 0)
        ][["gameId", "team", "shotAttemptsFor", "unblockedShotAttemptsFor"]]

        merged = nhl.merge(
            mp_season,
            left_on=["gameId", "team_abbrev"],
            right_on=["gameId", "team"],
            how="left",
            indicator=True,
        )
        n_nhl = len(nhl)
        matched = merged[merged["_merge"] == "both"].copy()
        n_matched = len(matched)
        coverage = n_matched / n_nhl if n_nhl else float("nan")

        if metric == "corsi":
            metric_desc = "|MoneyPuck shotAttemptsFor - NHL totalShotAttempts|"
            metric_cols = ["shotAttemptsFor", "totalShotAttempts"]
            nhl_value = matched["totalShotAttempts"]
            mp_value = matched["shotAttemptsFor"]
        else:
            metric_desc = "|MoneyPuck unblockedShotAttemptsFor - NHL (shots + missedShots)|"
            metric_cols = ["unblockedShotAttemptsFor", "shots", "missedShots"]
            nhl_value = matched["shots"] + matched["missedShots"]
            mp_value = matched["unblockedShotAttemptsFor"]
        matched["abs_diff"] = (mp_value - nhl_value).abs()

        # Guard against the silent-NaN failure mode: a null input column
        # would otherwise make the within-tolerance rate silently evaluate
        # to 0% (or the diff distribution print empty). Fail loudly instead.
        if n_matched and matched["abs_diff"].isna().any():
            null_cols = [c for c in metric_cols if matched[c].isna().any()]
            n_nan = int(matched["abs_diff"].isna().sum())
            print(
                f"[ERROR] Season {season_id}: abs_diff is NaN on {n_nan:,}/{n_matched:,} "
                f"matched rows ({metric} metric). Null input columns: {null_cols}."
            )
            raise SystemExit(1)

        within_tol = (matched["abs_diff"] <= DIFF_TOLERANCE).mean() if n_matched else float("nan")

        unmatched = merged[merged["_merge"] == "left_only"]
        unmatched_abbrevs = sorted(unmatched["team_abbrev"].unique()) if len(unmatched) else []

        print(f"\n[INFO] --- Season {season_id} (metric: {metric}) ---")
        print(f"[INFO] NHL team-game rows: {n_nhl:,}; matched to MoneyPuck: {n_matched:,} ({coverage:.2%})")
        if unmatched_abbrevs:
            print(f"[WARNING] Unmatched NHL rows involve team_abbrev values: {unmatched_abbrevs}")
        print(f"[INFO] {metric_desc} <= {DIFF_TOLERANCE}: {within_tol:.2%} of matched rows")
        print("[INFO] Diff distribution (value counts of absolute diff):")
        print(matched["abs_diff"].value_counts().sort_index().to_string())

        per_season_results[year] = {
            "n_nhl_rows": n_nhl,
            "n_matched": n_matched,
            "coverage": coverage,
            "within_tol_rate": within_tol,
            "metric": metric,
        }
        all_matched_diffs.append(matched["abs_diff"])
        all_nhl_rows += n_nhl
        all_matched_rows += n_matched

    combined_diffs = pd.concat(all_matched_diffs, ignore_index=True)
    combined_coverage = all_matched_rows / all_nhl_rows if all_nhl_rows else float("nan")
    combined_within_tol = (combined_diffs <= DIFF_TOLERANCE).mean() if len(combined_diffs) else float("nan")

    metrics_used = ", ".join(
        f"{year}{year + 1}={res['metric']}" for year, res in sorted(per_season_results.items())
    )
    print("\n[INFO] --- Combined across all seasons ---")
    print(f"[INFO] Metric used per season: {metrics_used}")
    print(f"[INFO] Join coverage: {all_matched_rows:,}/{all_nhl_rows:,} = {combined_coverage:.2%}")
    print(f"[INFO] Within-tolerance rate: {combined_within_tol:.2%} (bar: >= {DIFF_PASS_RATE_BAR:.0%})")
    print("[INFO] Combined diff distribution (value counts of absolute diff):")
    print(combined_diffs.value_counts().sort_index().to_string())

    failures = []
    if combined_coverage < JOIN_COVERAGE_BAR:
        failures.append(
            f"combined join coverage {combined_coverage:.2%} < required {JOIN_COVERAGE_BAR:.1%}"
        )
    if combined_within_tol < DIFF_PASS_RATE_BAR:
        failures.append(
            f"combined within-tolerance rate {combined_within_tol:.2%} < required {DIFF_PASS_RATE_BAR:.0%}"
        )
    for year, res in per_season_results.items():
        if res["coverage"] < JOIN_COVERAGE_BAR:
            failures.append(
                f"season {year} join coverage {res['coverage']:.2%} < required {JOIN_COVERAGE_BAR:.1%}"
            )
        if res["within_tol_rate"] < DIFF_PASS_RATE_BAR:
            failures.append(
                f"season {year} within-tolerance rate {res['within_tol_rate']:.2%} < required {DIFF_PASS_RATE_BAR:.0%}"
            )

    if failures:
        print("\n[ERROR] Cross-validation FAILED:")
        for f in failures:
            print(f"[ERROR]   - {f}")
        raise SystemExit(1)

    print("\n[INFO] Cross-validation PASSED (combined and every individual season clear both bars).")


# ---------------------------------------------------------------------------
# Training-key coverage
# ---------------------------------------------------------------------------


def report_training_key_coverage(
    clean_path: Path,
    team_games: pd.DataFrame,
    goalie_games: pd.DataFrame,
    seasons: list[int],
) -> None:
    print("\n[INFO] === Training-key join coverage vs clean_training_data.parquet ===")
    if not clean_path.exists():
        print(f"[WARNING] {clean_path} not found; skipping training-key coverage check.")
        return

    clean = pd.read_parquet(clean_path)
    clean_seasons = sorted(clean["season"].unique().tolist())
    print(f"[INFO] {clean_path}: {len(clean):,} rows, seasons present: {clean_seasons}")

    team_all = team_games[
        (team_games["situation"] == "all") & (team_games["playoffGame"] == 0)
    ][["gameId", "team"]].drop_duplicates()
    team_keys = set(map(tuple, team_all.itertuples(index=False, name=None)))

    goalie_all = goalie_games[goalie_games["situation"] == "all"][
        ["gameId", "playerId"]
    ].drop_duplicates()
    goalie_keys = set(map(tuple, goalie_all.itertuples(index=False, name=None)))

    clean_team_keys = clean[["game_id", "team_abbrev"]].drop_duplicates()
    clean_goalie_keys = clean[["game_id", "goalie_id"]].drop_duplicates()

    def _covered(row_keys: pd.DataFrame, ref_keys: set) -> pd.Series:
        return pd.Series(
            list(map(tuple, row_keys.itertuples(index=False, name=None)))
        ).apply(lambda k: k in ref_keys)

    team_covered = _covered(clean_team_keys, team_keys)
    goalie_covered = _covered(clean_goalie_keys, goalie_keys)

    team_rate_overall = team_covered.mean()
    goalie_rate_overall = goalie_covered.mean()
    print(
        f"[INFO] team_games coverage of clean (game_id, team_abbrev) keys: "
        f"{team_covered.sum():,}/{len(team_covered):,} = {team_rate_overall:.2%} (all clean seasons)"
    )
    print(
        f"[INFO] goalie_games coverage of clean (game_id, goalie_id) keys: "
        f"{goalie_covered.sum():,}/{len(goalie_covered):,} = {goalie_rate_overall:.2%} (all clean seasons)"
    )

    # Only clean seasons whose start year is among the fetched MoneyPuck
    # seasons can possibly match (earlier fetched seasons, e.g. 2021, exist
    # only to supply prior-season baselines and have no clean rows). Report
    # coverage restricted to the seasons actually in scope so a shortfall
    # from the deliberate fetch cutoff isn't mistaken for a join bug.
    fetched = set(seasons)
    in_scope_seasons = {s for s in clean_seasons if int(str(s)[:4]) in fetched}
    if in_scope_seasons:
        clean_in_scope = clean[clean["season"].isin(in_scope_seasons)]
        team_keys_scope = clean_in_scope[["game_id", "team_abbrev"]].drop_duplicates()
        goalie_keys_scope = clean_in_scope[["game_id", "goalie_id"]].drop_duplicates()
        team_covered_scope = _covered(team_keys_scope, team_keys)
        goalie_covered_scope = _covered(goalie_keys_scope, goalie_keys)
        team_rate_scope = team_covered_scope.mean()
        goalie_rate_scope = goalie_covered_scope.mean()
        print(
            f"[INFO] Restricted to clean seasons within the fetched MoneyPuck scope "
            f"({sorted(in_scope_seasons)}):"
        )
        print(
            f"[INFO]   team_games coverage: {team_covered_scope.sum():,}/{len(team_covered_scope):,} = {team_rate_scope:.2%}"
        )
        print(
            f"[INFO]   goalie_games coverage: {goalie_covered_scope.sum():,}/{len(goalie_covered_scope):,} = {goalie_rate_scope:.2%}"
        )

        if team_rate_scope < TRAINING_KEY_COVERAGE_WARN_BAR:
            missing = team_keys_scope[~team_covered_scope.values]
            print(
                f"[WARNING] team_games in-scope coverage {team_rate_scope:.2%} is below "
                f"{TRAINING_KEY_COVERAGE_WARN_BAR:.0%}. Example missing (game_id, team_abbrev) keys:"
            )
            print(missing.head(10).to_string(index=False))
        if goalie_rate_scope < TRAINING_KEY_COVERAGE_WARN_BAR:
            missing = goalie_keys_scope[~goalie_covered_scope.values]
            print(
                f"[WARNING] goalie_games in-scope coverage {goalie_rate_scope:.2%} is below "
                f"{TRAINING_KEY_COVERAGE_WARN_BAR:.0%}. Example missing (game_id, goalie_id) keys:"
            )
            print(missing.head(10).to_string(index=False))
    else:
        print(
            "[WARNING] No clean seasons fall within the fetched MoneyPuck scope; "
            "cannot compute in-scope coverage."
        )


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_row_counts(team_games: pd.DataFrame, goalie_games: pd.DataFrame) -> None:
    print("\n[INFO] === Row counts per season/situation ===")
    print("[INFO] team_games.parquet (season, situation, playoffGame -> row count):")
    print(
        team_games.groupby(["season", "situation", "playoffGame"]).size().to_string()
    )
    print("\n[INFO] goalie_games.parquet (season, situation -> row count):")
    print(goalie_games.groupby(["season", "situation"]).size().to_string())


def ensure_gitignore(raw_dir: Path) -> None:
    """Verify data/raw/moneypuck/ is not committable; add an entry if needed."""
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        print("[WARNING] No .gitignore found at repo root; skipping gitignore check.")
        return

    contents = gitignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in contents.splitlines()]

    # data/raw/ (whole-directory ignore) already covers data/raw/moneypuck/.
    if "data/raw/" in lines or "data/raw" in lines:
        print(
            "[INFO] .gitignore already ignores data/raw/ wholesale, which covers "
            "data/raw/moneypuck/ (verified with `git check-ignore`). No change needed."
        )
        return

    entry = "data/raw/moneypuck/"
    if entry in lines or f"/{entry}" in lines:
        print(f"[INFO] .gitignore already has an explicit entry for {entry}.")
        return

    print(f"[WARNING] Adding missing .gitignore entry: {entry}")
    with gitignore_path.open("a", encoding="utf-8") as f:
        f.write(f"\n{entry}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    seasons: list[int] = sorted(args.seasons)

    print(f"[INFO] Raw data directory: {raw_dir}")
    print(f"[INFO] Seasons: {seasons}")
    print(f"[INFO] force={args.force}")

    ensure_gitignore(raw_dir)

    # --- Download ---
    print("\n[INFO] === Download ===")
    team_csv_path = download_team_csv(raw_dir, force=args.force)
    for year in seasons:
        download_goalie_zip(raw_dir, year, force=args.force)

    # --- Normalize ---
    print("\n[INFO] === Normalize ===")
    team_games = build_team_games(team_csv_path, min_season=min(seasons))
    goalie_games = build_goalie_games(raw_dir, seasons)

    team_out = raw_dir / "team_games.parquet"
    goalie_out = raw_dir / "goalie_games.parquet"
    team_games.to_parquet(team_out, index=False)
    goalie_games.to_parquet(goalie_out, index=False)
    print(f"[INFO] Wrote {team_out} ({len(team_games):,} rows)")
    print(f"[INFO] Wrote {goalie_out} ({len(goalie_games):,} rows)")

    print_row_counts(team_games, goalie_games)

    # --- Cross-validation (mandatory, hard-fails on miss) ---
    cross_validate(team_games, seasons)

    # --- Training-key coverage ---
    report_training_key_coverage(args.clean, team_games, goalie_games, seasons)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
