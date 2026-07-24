"""
Walk-forward (rolling-origin) validation of the production classifier RECIPE.

Implements docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 21 (starts at
line 5415), registered 2026-07-24. Read that section in full before touching
this script -- it is locked, and this script implements it literally, with no
"improvements", no extra EV thresholds, and no re-tuning.

Question under test: does the production classifier recipe's backtested
+23.31% ROI (models/trained/tuned_v1_20260201_155204/) survive an honest
forward-in-time out-of-sample test on the two additional seasons that have
since been folded into the training data? A market-parity/null result is a
fully legitimate outcome (section 21.1) -- this script does not search for a
better-looking answer, and its PASS/FAIL bar (section 21.4) is applied
literally with no softening.

SINGLE-RUN DISCIPLINE (section 21.6): this evaluation is meant to run EXACTLY
ONCE without --smoke. No re-carving folds, no re-running a fold after seeing
its result, no threshold re-sweep. --smoke mode (see below) is exempt: it
only ever touches 2023-24 data, which is a training window in both origins
and never a test fold, so repeated smoke runs touch no test data and are not
subject to the single-run discipline.

WHAT THIS SCRIPT DOES
----------------------
1. Loads data/processed/multibook_classification_training_data.parquet,
   drops the 12 market-derived columns, sorts chronologically, filters to
   rows with non-null odds on both sides, and reproduces the frozen 114
   engineered features -- mirroring scripts/tune_hyperparameters.py lines
   112-158 EXACTLY (that script trained the production model). The engineered-
   feature function (add_all_engineered_features) is COPIED verbatim from
   tune_hyperparameters.py rather than imported, because that module runs a
   full 168-evaluation hyperparameter search at import time and cannot safely
   be imported from another script. A runtime AST-equality guard
   (verify_engineered_features_verbatim) proves the copy is faithful and
   hard-fails if it has drifted.
2. Hard-fails unless the derived 114-feature list exactly matches
   models/trained/tuned_v1_20260201_155204/classifier_feature_names.json
   (same names, same order) -- section 21.2 freezes the feature set to that
   file.
3. Carves two expanding-window folds by game_date (section 21.2):
     Origin 1: train = 2023-24,            test = 2024-25
     Origin 2: train = 2023-24 + 2024-25,  test = 2025-26
4. Retrains the frozen recipe's hyperparameters (read at runtime from
   classifier_metadata.json, guarded against the registered section-21.2
   values) per origin, on the ENTIRE training window -- no validation
   holdout, no early stopping (n_estimators is fixed by the frozen recipe).
5. Grades bets via ClassifierTrainer.evaluate_profitability (imported, never
   reimplemented) at ev_threshold=0.12, then reconstructs a per-game bet/
   profit vector by calling that SAME function once per game_id (see
   per_game_bet_totals below) so a GAME-level (not row-level) bootstrap is
   possible without modifying the shared trainer file. Reconciliation is
   hard-asserted: per-game bets/profits must sum exactly to the whole-fold
   result.
6. Runs a 10,000-iteration, seed-42, percentile-method 95% bootstrap CI on
   ROI resampled at the game level, per fold and pooled across both test
   folds (section 21.3; row-level bootstrap is explicitly forbidden by
   section 21.6 item 3 -- this data has ~4 correlated book-line rows per
   goalie-game).
7. Reports secondary diagnostics (AUC, log-loss, Brier, a calibration table,
   a degeneracy flag, and the market's own devigged-probability log-loss/
   Brier for comparison) -- these are diagnostic only, never gating.
8. Applies the section 21.4 PASS/FAIL bar literally: PASS requires BOTH a
   pooled game-level bootstrap 95% CI lower bound > 0 AND a positive ROI
   point estimate in each origin individually. Anything else is
   MARKET-PARITY / FAIL. No alternative thresholds, no "close" framing.

DISCLOSED DEVIATIONS FROM THE REGISTRATION TEXT
-------------------------------------------------
- Section 21.2's registered pre-filter row/game counts (2023-24: 7,607 rows /
  1,123 games; 2024-25: 7,463 rows / 1,291 games; 2025-26: 5,729 rows / 1,107
  games) were counted BEFORE the odds-non-null filter this script applies
  (mirroring tune_hyperparameters.py, which is the recipe under test). This
  script reports BOTH the pre-filter and post-filter row/game counts per
  season so any reduction is visible rather than silently absorbed. As of the
  2026-07-24 snapshot of the parquet, every row already has non-null odds on
  both sides, so the filter is empirically a no-op on this data (verified by
  hand before writing this script) -- but the check and the disclosure are
  kept unconditional so a future data refresh that reintroduces nulls doesn't
  silently change the evaluated population without being reported.
- The registered figures label counts as "games"; this script reports both
  distinct game_id counts ("games", the bootstrap's resampling unit per
  section 21.3) and distinct (game_id, goalie_id) counts ("goalie-games",
  matching section 21.8's parenthetical) so the two are never conflated.

SELF-TEST (--smoke)
--------------------
Uses ONLY 2023-24 rows (never a test fold in the registered folds), split
internally by date into a smoke-train/smoke-test pair, trains with the frozen
recipe's own hyperparameters (no n_estimators override -- a weakened model
can place zero bets, which would leave the reconciliation and bootstrap code
paths unexercised and make the self-test pass vacuously), and exercises the
full downstream path (evaluate_profitability, per-game reconciliation,
bootstrap, diagnostics, artifact writing) end to end. Hard-fails if the smoke
fold places zero bets, since a self-test that can pass without exercising
what it exists to test is worse than none. Its numbers are otherwise
meaningless -- it exists only to prove the code is wired correctly before the
real (non-smoke) run, which is a separate, deliberately gated decision.

REAL RUN GUARD (--confirm-single-run)
---------------------------------------
The non-smoke path additionally requires --confirm-single-run. Without it,
the script refuses to load any data or train anything and exits non-zero,
citing section 21.6's single-run discipline -- this guards against an
accidental bare invocation (shell-history recall, tab-completion) consuming
the one registered run.

Usage:
    python scripts/experiment_walk_forward_classifier.py --smoke
    python scripts/experiment_walk_forward_classifier.py --confirm-single-run   # the real, single registered run -- see section 21.6
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Plain sys.path imports (not importlib), matching tune_hyperparameters.py
# lines 32-39 and the sibling experiment_rolling_origin.py convention.
from models.classifier_trainer import ClassifierTrainer  # noqa: E402
from betting.tracking_db import devig_prob  # noqa: E402

import logging  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DATA_PATH = REPO_ROOT / "data" / "processed" / "multibook_classification_training_data.parquet"
PROD_MODEL_DIR = REPO_ROOT / "models" / "trained" / "tuned_v1_20260201_155204"
FEATURE_NAMES_PATH = PROD_MODEL_DIR / "classifier_feature_names.json"
PROD_METADATA_PATH = PROD_MODEL_DIR / "classifier_metadata.json"
TUNE_SCRIPT_PATH = REPO_ROOT / "scripts" / "tune_hyperparameters.py"
OUTPUT_ROOT = REPO_ROOT / "models" / "trained"

EV_THRESHOLD = 0.12
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42
XGB_SEED = 42

# Section 21.2: "market-derived" columns dropped before feature selection --
# byte-identical to tune_hyperparameters.py lines 115-121.
MARKET_FEATURES = [
    'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
    'market_vig', 'impl_prob_over', 'impl_prob_under',
    'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
    'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low'
]

# Byte-identical to tune_hyperparameters.py lines 129-142.
EXCLUDED = [
    'game_id', 'goalie_id', 'game_date', 'over_hit',
    'odds_over_american', 'odds_under_american',
    'odds_over_decimal', 'odds_under_decimal', 'num_books',
    'team_abbrev', 'opponent_team', 'toi', 'season',
    'saves', 'shots_against', 'goals_against', 'save_percentage',
    'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
    'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
    'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
    'team_goals', 'team_shots', 'opp_goals', 'opp_shots', 'line_margin',
    'book_key', 'decision', 'team_id', 'goalie_name',
    'saves_margin', 'over_line',
    '_game_date_str', '_lookup_key',
]

# Byte-identical to tune_hyperparameters.py lines 153-156.
FORBIDDEN_COLS = [
    'goalie_name', 'book_key', 'team_abbrev', 'opponent_team',
    'season', 'game_id', 'goalie_id', 'game_date',
]

# Section 21.2: season assignment by game_date, frozen.
SEASON_BOUNDS = {
    '2023-24': (pd.Timestamp('2023-08-01'), pd.Timestamp('2024-07-31')),
    '2024-25': (pd.Timestamp('2024-08-01'), pd.Timestamp('2025-07-31')),
    '2025-26': (pd.Timestamp('2025-08-01'), pd.Timestamp('2026-07-31')),
}

# Section 21.2's registered PRE-FILTER row/game figures -- logged for
# comparison only, never asserted against (the filter is part of the frozen
# recipe and is allowed to change the population; see module docstring).
REGISTERED_PRE_FILTER = {
    '2023-24': {'rows': 7607, 'games': 1123},
    '2024-25': {'rows': 7463, 'games': 1291},
    '2025-26': {'rows': 5729, 'games': 1107},
}

# Section 21.2: the frozen "Random #30" hyperparameters, verbatim from
# classifier_metadata.json. Read at runtime from that file (never
# hardcoded into the fit call) -- this dict exists ONLY as a guard that the
# file hasn't drifted from the registered values.
EXPECTED_HYPERPARAMETERS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_child_weight': 30,
    'gamma': 2.0,
    'reg_alpha': 20,
    'reg_lambda': 60,
    'n_estimators': 600,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
}


# ---------------------------------------------------------------------------
# Engineered features -- COPIED VERBATIM from scripts/tune_hyperparameters.py
# (lines 51-105 as of this writing), NOT imported. tune_hyperparameters.py
# executes a full 168-evaluation hyperparameter search at module import time
# (it is a script, not a library), so importing it here would run that search
# as a side effect. The repo's existing convention for this exact problem
# (see optimize_features.py, calibrate_model.py, experiment_market_anchor.py)
# is to copy this function rather than import it. verify_engineered_features_
# verbatim() below is a REAL runtime check (AST comparison against
# tune_hyperparameters.py's source, ignoring only the docstring and
# whitespace/indentation) that hard-fails if this copy has drifted.
# ---------------------------------------------------------------------------


def add_all_engineered_features(df):
    """Reproduce the 18 engineered features from optimize_features.py."""
    df = df.copy()

    # Interaction features
    for w in [3, 5, 10]:
        sr = f'saves_rolling_{w}'
        sar = f'shots_against_rolling_{w}'
        if sr in df.columns and sar in df.columns:
            df[f'save_efficiency_{w}'] = df[sr] / df[sar].clip(lower=1)

    for w in [5, 10]:
        es = f'even_strength_saves_rolling_{w}'
        sr = f'saves_rolling_{w}'
        if es in df.columns and sr in df.columns:
            df[f'es_saves_proportion_{w}'] = df[es] / df[sr].clip(lower=1)

    if 'opp_shots_rolling_5' in df.columns and 'team_shots_against_rolling_5' in df.columns:
        df['opp_vs_team_shots_5'] = df['opp_shots_rolling_5'] - df['team_shots_against_rolling_5']
    if 'opp_shots_rolling_10' in df.columns and 'team_shots_against_rolling_10' in df.columns:
        df['opp_vs_team_shots_10'] = df['opp_shots_rolling_10'] - df['team_shots_against_rolling_10']

    # Volatility features
    for w in [5, 10]:
        mean_col = f'saves_rolling_{w}'
        std_col = f'saves_rolling_std_{w}'
        if mean_col in df.columns and std_col in df.columns:
            df[f'saves_cv_{w}'] = df[std_col] / df[mean_col].clip(lower=1)
        if std_col in df.columns and 'betting_line' in df.columns:
            df[f'volatility_vs_line_{w}'] = df[std_col] / df['betting_line'].clip(lower=1)

    # Trend/momentum features
    for stat in ['saves', 'shots_against', 'goals_against']:
        short = f'{stat}_rolling_3'
        long = f'{stat}_rolling_10'
        if short in df.columns and long in df.columns:
            df[f'{stat}_momentum'] = df[short] - df[long]

    sp_short = 'save_percentage_rolling_3'
    sp_long = 'save_percentage_rolling_10'
    if sp_short in df.columns and sp_long in df.columns:
        df['save_pct_momentum'] = df[sp_short] - df[sp_long]

    # Matchup context features
    if 'opp_shots_rolling_5' in df.columns and 'shots_against_rolling_5' in df.columns:
        df['expected_workload_diff'] = df['opp_shots_rolling_5'] - df['shots_against_rolling_5']

    if 'opp_shots_rolling_5' in df.columns and 'opp_goals_rolling_5' in df.columns:
        opp_saves_implied = df['opp_shots_rolling_5'] - df['opp_goals_rolling_5']
        df['line_vs_opp_implied_saves'] = df['betting_line'] - opp_saves_implied

    if 'goalie_days_rest' in df.columns and 'saves_rolling_5' in df.columns:
        df['rest_x_performance'] = df['goalie_days_rest'].clip(upper=7) * df['saves_rolling_5']

    return df


def _find_top_level_function(source_text: str, func_name: str) -> ast.FunctionDef:
    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise AssertionError(f"Could not find top-level function '{func_name}' in given source.")


def _function_body_without_docstring(node: ast.FunctionDef) -> list:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            body = body[1:]
    return body


def _normalized_ast_dump(node: ast.FunctionDef) -> str:
    # ast.dump (include_attributes=False, the default) already ignores line/
    # column offsets, i.e. whitespace/indentation. Stripping the docstring
    # statement is the only additional normalization applied, per the task's
    # instruction to normalize "only the docstring/leading whitespace".
    body = _function_body_without_docstring(node)
    return "\n".join(ast.dump(stmt) for stmt in body)


def verify_engineered_features_verbatim(log) -> None:
    """Hard-fail unless this module's add_all_engineered_features is a
    faithful copy of tune_hyperparameters.py's version (AST-equal after
    stripping the docstring)."""
    if not TUNE_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Cannot verify verbatim copy: {TUNE_SCRIPT_PATH} not found.")
    reference_source = TUNE_SCRIPT_PATH.read_text(encoding="utf-8")
    reference_node = _find_top_level_function(reference_source, "add_all_engineered_features")

    local_source = textwrap.dedent(inspect.getsource(add_all_engineered_features))
    local_node = _find_top_level_function(local_source, "add_all_engineered_features")

    reference_dump = _normalized_ast_dump(reference_node)
    local_dump = _normalized_ast_dump(local_node)

    if reference_dump != local_dump:
        raise AssertionError(
            "VERBATIM-COPY GUARD FAILED: this script's local copy of "
            "add_all_engineered_features does not AST-match "
            f"scripts/tune_hyperparameters.py's version (compared after "
            "stripping the docstring; whitespace/indentation is already "
            "ignored by ast.dump). Refusing to proceed -- the feature "
            "recipe under test would not be the frozen production recipe."
        )
    log(
        "Verbatim-copy guard PASSED: local add_all_engineered_features is "
        "AST-identical to scripts/tune_hyperparameters.py's version "
        "(docstring and whitespace/indentation ignored by construction)."
    )


# ---------------------------------------------------------------------------
# Logging convention (matches scripts/experiment_rolling_origin.py's
# make_logger: print + accumulate + flush to run_log.txt).
# ---------------------------------------------------------------------------


def make_logger(log_path: Path):
    log_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(str(msg))

    def flush_log() -> None:
        log_path.write_text("\n".join(log_lines), encoding="utf-8")

    return log, flush_log


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(log) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        commit = result.stdout.strip()
        log(f"Script git commit: {commit}")
        return commit
    except Exception as exc:  # noqa: BLE001
        log(f"Could not determine git commit (non-fatal): {exc}")
        return None


# ---------------------------------------------------------------------------
# Data + features (mirrors tune_hyperparameters.py lines 112-158 exactly)
# ---------------------------------------------------------------------------


def assign_season_labels(df: pd.DataFrame, log, context_label: str) -> np.ndarray:
    """Assign each row's game_date to one of the three registered season
    windows (section 21.2). Hard-fails if any row falls outside all three."""
    dates = df["game_date"].values
    labels = np.full(len(df), None, dtype=object)
    for season, (start, end) in SEASON_BOUNDS.items():
        mask = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
        labels[mask] = season

    unassigned_mask = pd.isna(labels)
    n_unassigned = int(unassigned_mask.sum())
    if n_unassigned:
        bad_dates = pd.Series(dates[unassigned_mask])
        raise AssertionError(
            f"[{context_label}] {n_unassigned} rows have game_date outside all three "
            f"registered season windows (2023-08-01..2026-07-31 combined); offending "
            f"date range {bad_dates.min()}..{bad_dates.max()}. Investigate before proceeding."
        )
    log(f"[{context_label}] season assignment: all {len(df)} rows assigned to a registered season window.")
    return labels


def load_and_prepare_data(log):
    """Mirrors tune_hyperparameters.py lines 112-158 exactly: load, drop
    market-derived columns, sort chronologically, filter to rows with
    non-null odds on both sides, add engineered features, build feature_cols.
    Reports pre-filter and post-filter row/game counts per season so the
    odds-filter's effect (if any) is disclosed rather than absorbed."""
    log(f"Loading {DATA_PATH}")
    df_raw = pd.read_parquet(DATA_PATH)
    df_raw["game_date"] = pd.to_datetime(df_raw["game_date"])
    log(f"Loaded {len(df_raw)} raw rows, {len(df_raw.columns)} columns.")

    before_cols = set(df_raw.columns)
    df_raw = df_raw.drop(columns=[c for c in MARKET_FEATURES if c in df_raw.columns], errors="ignore")
    dropped_market_cols = sorted(before_cols - set(df_raw.columns))
    log(f"Dropped market-derived columns ({len(dropped_market_cols)}): {dropped_market_cols}")

    df_raw = df_raw.sort_values("game_date").reset_index(drop=True)

    # Pre-filter counts -- section 21.2's registered figures were counted
    # at this stage, BEFORE the odds-non-null filter below.
    pre_filter_labels = assign_season_labels(df_raw, log, "pre-filter")
    pre_filter_counts = {}
    for season in SEASON_BOUNDS:
        mask = pre_filter_labels == season
        pre_filter_counts[season] = {
            "rows": int(mask.sum()),
            "games": int(df_raw.loc[mask, "game_id"].nunique()),
            "goalie_games": int(df_raw.loc[mask, ["game_id", "goalie_id"]].drop_duplicates().shape[0]),
        }
        registered = REGISTERED_PRE_FILTER[season]
        log(
            f"[{season}] PRE-FILTER: {pre_filter_counts[season]['rows']} rows, "
            f"{pre_filter_counts[season]['games']} games, "
            f"{pre_filter_counts[season]['goalie_games']} goalie-games "
            f"(section 21.2 registered figure: {registered['rows']} rows / {registered['games']} games)"
        )

    n_before_odds_filter = len(df_raw)
    df_raw = df_raw[
        df_raw["odds_over_american"].notna() & df_raw["odds_under_american"].notna()
    ].reset_index(drop=True)
    n_after_odds_filter = len(df_raw)
    n_dropped_by_odds_filter = n_before_odds_filter - n_after_odds_filter
    log(
        f"Odds-non-null filter (both odds_over_american and odds_under_american required): "
        f"{n_before_odds_filter} -> {n_after_odds_filter} rows "
        f"({n_dropped_by_odds_filter} dropped)."
    )
    if n_dropped_by_odds_filter == 0:
        log(
            "NOTE: the odds-non-null filter dropped 0 rows on this snapshot of the parquet "
            "-- every row already has non-null odds on both sides. The pre-filter and "
            "post-filter counts below are therefore expected to be identical for this run; "
            "the filter is still applied and both counts are still reported unconditionally "
            "(module docstring)."
        )

    df = add_all_engineered_features(df_raw)
    log(f"Added engineered features: {len(df)} rows, {len(df.columns)} columns.")

    post_filter_labels = assign_season_labels(df, log, "post-filter")
    post_filter_counts = {}
    for season in SEASON_BOUNDS:
        mask = post_filter_labels == season
        post_filter_counts[season] = {
            "rows": int(mask.sum()),
            "games": int(df.loc[mask, "game_id"].nunique()),
            "goalie_games": int(df.loc[mask, ["game_id", "goalie_id"]].drop_duplicates().shape[0]),
        }
        log(f"[{season}] POST-FILTER: {post_filter_counts[season]}")

    feature_cols = [c for c in df.columns if c not in EXCLUDED]
    log(f"Derived feature_cols: {len(feature_cols)}")
    assert len(feature_cols) == 114, (
        f"Expected exactly 114 feature columns, got {len(feature_cols)}. "
        f"Feature set must not change for this evaluation (section 21.2)."
    )
    leaked = [c for c in FORBIDDEN_COLS if c in feature_cols]
    assert not leaked, f"Forbidden metadata/identifier columns leaked into feature_cols: {leaked}"

    return df, feature_cols, post_filter_labels, pre_filter_counts, post_filter_counts


def verify_frozen_feature_list(feature_cols: list[str], log) -> None:
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(f"Missing frozen feature list: {FEATURE_NAMES_PATH}")
    frozen = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    if feature_cols != frozen:
        only_derived = [c for c in feature_cols if c not in frozen]
        only_frozen = [c for c in frozen if c not in feature_cols]
        raise AssertionError(
            "FROZEN FEATURE-LIST GUARD FAILED: derived feature_cols does not exactly match "
            f"{FEATURE_NAMES_PATH} (same 114 names, same order required by section 21.2).\n"
            f"In derived but not frozen: {only_derived}\n"
            f"In frozen but not derived: {only_frozen}\n"
            f"(If both lists are empty here, the mismatch is purely in ORDER.)"
        )
    log(
        f"Frozen feature-list guard PASSED: derived feature_cols exactly matches "
        f"{FEATURE_NAMES_PATH.relative_to(REPO_ROOT)} (114 names, same order)."
    )


def verify_frozen_hyperparameters(hyperparameters: dict, log) -> None:
    mismatches = []
    for key, expected in EXPECTED_HYPERPARAMETERS.items():
        actual = hyperparameters.get(key)
        if actual != expected:
            mismatches.append((key, expected, actual))
    if mismatches:
        raise AssertionError(
            f"HYPERPARAMETER GUARD FAILED: {PROD_METADATA_PATH} hyperparameters do not match "
            f"section 21.2's frozen recipe. Mismatches (key, expected, actual): {mismatches}"
        )
    log(
        f"Hyperparameter guard PASSED: {hyperparameters} read from "
        f"{PROD_METADATA_PATH.relative_to(REPO_ROOT)} exactly matches section 21.2."
    )


# ---------------------------------------------------------------------------
# Fold carving (expanding window, section 21.2)
# ---------------------------------------------------------------------------


def carve_origin(
    df: pd.DataFrame, season_labels: np.ndarray, train_seasons: list[str], test_season: str, log, label: str,
):
    train_mask = np.isin(season_labels, train_seasons)
    test_mask = season_labels == test_season
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise AssertionError(f"[{label}] Empty train or test fold after carving; investigate.")

    train_dates = df["game_date"].values[train_idx]
    test_dates = df["game_date"].values[test_idx]
    if train_dates.max() >= test_dates.min():
        raise AssertionError(
            f"[{label}] TRAIN/TEST DATE OVERLAP: train max date {train_dates.max()} >= "
            f"test min date {test_dates.min()}. Folds must not overlap (section 21.2)."
        )

    log(
        f"[{label}] train seasons={train_seasons}: {len(train_idx)} rows, "
        f"{pd.Timestamp(train_dates.min()).date()}..{pd.Timestamp(train_dates.max()).date()}; "
        f"test season={test_season}: {len(test_idx)} rows, "
        f"{pd.Timestamp(test_dates.min()).date()}..{pd.Timestamp(test_dates.max()).date()}"
    )
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Per-game bet reconstruction (section 21.3). evaluate_profitability's loop
# is strictly per-row with no cross-row state (verified by reading
# src/models/classifier_trainer.py -- each iteration only reads
# df_split.iloc[i] and the model's already-computed predict_proba output),
# so calling it once per game with that game's own row positions as
# split_idx must reproduce exactly the same per-row bet decisions as the
# whole-fold call, and per-game totals must sum exactly to the whole-fold
# total. This is hard-asserted by the caller, not assumed.
# ---------------------------------------------------------------------------


def per_game_bet_totals(trainer: ClassifierTrainer, X_full: np.ndarray, y_full: np.ndarray,
                         df_full: pd.DataFrame, fold_idx: np.ndarray, log, label: str):
    ct_logger = logging.getLogger("models.classifier_trainer")
    original_level = ct_logger.level
    ct_logger.setLevel(logging.ERROR)  # suppress per-call INFO noise for the game loop
    # evaluate_profitability (src/models/classifier_trainer.py) does an
    # unconditional sys.path.insert(0, .../src) on EVERY call. Called once
    # per game (~1,000+ times per fold), that would otherwise accumulate
    # thousands of duplicate sys.path entries and slow down every later
    # import in the process. Snapshot and restore around the whole loop
    # (not per-call) since we can't fix the shared file.
    original_sys_path = list(sys.path)
    try:
        game_ids_in_fold = df_full["game_id"].values[fold_idx]
        unique_games = np.unique(game_ids_in_fold)
        bets_per_game = np.zeros(len(unique_games), dtype=np.int64)
        wins_per_game = np.zeros(len(unique_games), dtype=np.int64)
        profit_per_game = np.zeros(len(unique_games), dtype=np.float64)

        for gi, gid in enumerate(unique_games):
            game_positions = fold_idx[game_ids_in_fold == gid]
            m = trainer.evaluate_profitability(
                X_full[game_positions], y_full[game_positions], df_full, game_positions,
                dataset_name=f"{label}_game_{gid}", ev_threshold=EV_THRESHOLD,
            )
            bets_per_game[gi] = m["total_bets"]
            wins_per_game[gi] = m["wins"]
            profit_per_game[gi] = m["total_profit"]
    finally:
        ct_logger.setLevel(original_level)
        sys.path[:] = original_sys_path

    log(
        f"[{label}] Per-game reconstruction: {len(unique_games)} distinct games, "
        f"{int(bets_per_game.sum())} total bets, {float(profit_per_game.sum()):+.4f} total profit units."
    )
    return unique_games, bets_per_game, wins_per_game, profit_per_game


def reconcile_per_game_totals(whole_fold: dict, bets_per_game: np.ndarray, profit_per_game: np.ndarray,
                               log, label: str) -> None:
    sum_bets = int(bets_per_game.sum())
    sum_profit = float(profit_per_game.sum())
    whole_bets = int(whole_fold["total_bets"])
    whole_profit = float(whole_fold["total_profit"])

    if sum_bets != whole_bets:
        raise AssertionError(
            f"[{label}] RECONCILIATION FAILED: sum of per-game total_bets ({sum_bets}) != "
            f"whole-fold total_bets ({whole_bets})."
        )
    profit_diff = abs(sum_profit - whole_profit)
    if profit_diff > 1e-9:
        raise AssertionError(
            f"[{label}] RECONCILIATION FAILED: sum of per-game total_profit ({sum_profit}) != "
            f"whole-fold total_profit ({whole_profit}); |diff|={profit_diff} > 1e-9."
        )
    log(
        f"[{label}] Reconciliation PASSED: sum(per-game bets)={sum_bets} == whole-fold "
        f"bets={whole_bets}; sum(per-game profit)={sum_profit:+.6f} == whole-fold "
        f"profit={whole_profit:+.6f} (|diff|={profit_diff:.2e} <= 1e-9)."
    )


# ---------------------------------------------------------------------------
# Game-level bootstrap (section 21.3; row-level is forbidden by 21.6 item 3)
# ---------------------------------------------------------------------------


def bootstrap_game_level_roi(bets_per_game: np.ndarray, profit_per_game: np.ndarray, seed: int,
                              n_resamples: int, ci_pct: float = 95.0, chunk_size: int = 1000) -> dict:
    n_games = len(bets_per_game)
    if n_games == 0:
        return {"lower": None, "upper": None, "n_games": 0, "n_resamples_used": 0, "n_resamples_requested": n_resamples}

    rng = np.random.RandomState(seed)
    roi_values = []
    remaining = n_resamples
    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        idx = rng.randint(0, n_games, size=(this_chunk, n_games))
        chunk_bets = bets_per_game[idx].sum(axis=1)
        chunk_profit = profit_per_game[idx].sum(axis=1)
        valid = chunk_bets > 0
        if valid.any():
            roi_values.append(chunk_profit[valid] / chunk_bets[valid] * 100.0)
        remaining -= this_chunk

    if roi_values:
        roi_arr = np.concatenate(roi_values)
    else:
        roi_arr = np.array([])

    n_used = int(len(roi_arr))
    n_skipped = n_resamples - n_used
    if n_used == 0:
        return {"lower": None, "upper": None, "n_games": n_games, "n_resamples_used": 0, "n_resamples_requested": n_resamples}

    alpha = (100.0 - ci_pct) / 2.0
    lower = float(np.percentile(roi_arr, alpha))
    upper = float(np.percentile(roi_arr, 100.0 - alpha))
    return {
        "lower": lower,
        "upper": upper,
        "n_games": int(n_games),
        "n_resamples_used": n_used,
        "n_resamples_requested": n_resamples,
        "n_resamples_skipped_zero_bets": int(n_skipped),
    }


# ---------------------------------------------------------------------------
# Secondary diagnostics (section 21.3 -- reported, not gating)
# ---------------------------------------------------------------------------


def compute_calibration_table(proba: np.ndarray, y_true: np.ndarray) -> list[dict]:
    bin_edges = np.linspace(0.0, 1.0, 11)
    table = []
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (proba >= lo) & (proba < hi) if i < 9 else (proba >= lo) & (proba <= hi)
        n = int(mask.sum())
        table.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "count": n,
            "mean_predicted_p_over": float(proba[mask].mean()) if n else None,
            "realized_over_rate": float(y_true[mask].mean()) if n else None,
        })
    return table


def compute_market_devig(df: pd.DataFrame, idx: np.ndarray):
    """De-vigged market implied OVER probability, reusing
    betting.tracking_db.devig_prob (which itself reuses
    betting.odds_utils.american_to_implied_prob) -- never averages American
    odds directly, per docs/HISTORICAL_DATA_ANALYSIS.md section 1."""
    odds_over = df["odds_over_american"].values[idx].astype(float)
    odds_under = df["odds_under_american"].values[idx].astype(float)
    p_over = np.full(len(idx), np.nan)
    p_under = np.full(len(idx), np.nan)
    n_fail = 0
    for i in range(len(idx)):
        po, pu = devig_prob(odds_over[i], odds_under[i])
        if po is None:
            n_fail += 1
            continue
        p_over[i] = po
        p_under[i] = pu
    return p_over, p_under, n_fail


# ---------------------------------------------------------------------------
# Per-origin (or per-smoke-split) runner
# ---------------------------------------------------------------------------


def run_fold(label: str, df: pd.DataFrame, feature_cols: list[str], X_full: np.ndarray, y_full: np.ndarray,
             train_idx: np.ndarray, test_idx: np.ndarray, hyperparameters: dict, output_dir: Path, log,
             n_estimators_override: int | None = None) -> dict:
    log("\n" + "=" * 80)
    log(f"FOLD {label}: train n={len(train_idx)}, test n={len(test_idx)}")
    log("=" * 80)

    fit_hyperparameters = dict(hyperparameters)
    if n_estimators_override is not None:
        log(
            f"[{label}] SMOKE OVERRIDE: n_estimators {fit_hyperparameters.get('n_estimators')} -> "
            f"{n_estimators_override} (speed only; production recipe otherwise unchanged)."
        )
        fit_hyperparameters["n_estimators"] = n_estimators_override

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        random_state=XGB_SEED,
        n_jobs=-1,
        verbosity=0,
        **fit_hyperparameters,
    )
    t0 = time.time()
    model.fit(X_full[train_idx], y_full[train_idx])
    log(f"[{label}] Model fit on entire training window in {time.time() - t0:.1f}s (no validation holdout, no early stopping).")

    model_path = output_dir / f"{label}_model.json"
    model.get_booster().save_model(str(model_path))
    log(f"[{label}] Saved fitted booster to {model_path}")

    trainer = ClassifierTrainer()
    trainer.model = model
    trainer.feature_names = feature_cols

    whole_fold = trainer.evaluate_profitability(
        X_full[test_idx], y_full[test_idx], df, test_idx, dataset_name=label, ev_threshold=EV_THRESHOLD,
    )
    bet_rate_pct = whole_fold["total_bets"] / len(test_idx) * 100.0
    degenerate = bet_rate_pct < 5.0 or bet_rate_pct > 50.0
    log(
        f"[{label}] WHOLE-FOLD: {whole_fold['total_bets']} bets ({bet_rate_pct:.2f}% of {len(test_idx)} "
        f"candidate lines), ROI={whole_fold['roi']:+.2f}%, profit={whole_fold['total_profit']:+.2f} units"
        + (" -- DEGENERATE BET VOLUME FLAG" if degenerate else "")
    )

    unique_games, bets_per_game, wins_per_game, profit_per_game = per_game_bet_totals(
        trainer, X_full, y_full, df, test_idx, log, label,
    )
    reconcile_per_game_totals(whole_fold, bets_per_game, profit_per_game, log, label)

    ci = bootstrap_game_level_roi(bets_per_game, profit_per_game, BOOTSTRAP_SEED, N_BOOTSTRAP)
    if ci["lower"] is not None:
        log(f"[{label}] Game-level bootstrap 95% CI on ROI: [{ci['lower']:+.2f}%, {ci['upper']:+.2f}%] (n_games={ci['n_games']})")
    else:
        log(f"[{label}] Game-level bootstrap: no resample had any bets -- CI undefined.")

    proba_all = model.predict_proba(X_full[test_idx])[:, 1]
    y_test = y_full[test_idx]
    auc = float(roc_auc_score(y_test, proba_all))
    logloss = float(log_loss(y_test, proba_all))
    brier = float(brier_score_loss(y_test, proba_all))
    calibration_table = compute_calibration_table(proba_all, y_test)
    log(f"[{label}] Diagnostics (all {len(test_idx)} test rows): AUC={auc:.4f}, log-loss={logloss:.4f}, Brier={brier:.4f}")

    p_over_mkt, p_under_mkt, n_devig_fail = compute_market_devig(df, test_idx)
    market_mask = ~np.isnan(p_over_mkt)
    if market_mask.all():
        market_logloss = float(log_loss(y_test, p_over_mkt))
        market_brier = float(brier_score_loss(y_test, p_over_mkt))
    elif market_mask.any():
        market_logloss = float(log_loss(y_test[market_mask], p_over_mkt[market_mask]))
        market_brier = float(brier_score_loss(y_test[market_mask], p_over_mkt[market_mask]))
    else:
        market_logloss = None
        market_brier = None
    market_logloss_str = f"{market_logloss:.4f}" if market_logloss is not None else "None"
    market_brier_str = f"{market_brier:.4f}" if market_brier is not None else "None"
    log(
        f"[{label}] Market devig comparison ({int(market_mask.sum())}/{len(test_idx)} rows devigged, "
        f"{n_devig_fail} failures): market log-loss={market_logloss_str}, market Brier={market_brier_str} "
        f"vs model log-loss={logloss:.4f}, model Brier={brier:.4f}"
    )

    return {
        "label": label,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "hyperparameters_used": fit_hyperparameters,
        "model_path": str(model_path),
        "whole_fold": whole_fold,
        "unique_games": unique_games,
        "bets_per_game": bets_per_game,
        "wins_per_game": wins_per_game,
        "profit_per_game": profit_per_game,
        "bet_rate_pct": bet_rate_pct,
        "degenerate_bet_volume": degenerate,
        "bootstrap_ci_95": ci,
        "diagnostics": {
            "auc": auc,
            "log_loss": logloss,
            "brier": brier,
            "market_log_loss": market_logloss,
            "market_brier": market_brier,
            "market_devig_failures": n_devig_fail,
            "market_devig_coverage": int(market_mask.sum()),
            "calibration_table": calibration_table,
        },
    }


def fold_result_to_json_safe(result: dict) -> dict:
    """Drop numpy arrays not meant for direct JSON embedding, replacing them
    with the per-game table as a list of dicts."""
    out = dict(result)
    unique_games = out.pop("unique_games")
    bets_per_game = out.pop("bets_per_game")
    wins_per_game = out.pop("wins_per_game")
    profit_per_game = out.pop("profit_per_game")
    out["per_game"] = [
        {
            "game_id": int(unique_games[i]),
            "bets": int(bets_per_game[i]),
            "wins": int(wins_per_game[i]),
            "profit": float(profit_per_game[i]),
        }
        for i in range(len(unique_games))
    ]
    return out


# ---------------------------------------------------------------------------
# Pooling and PASS/FAIL verdict (sections 21.3, 21.4)
# ---------------------------------------------------------------------------


def pool_folds(fold_results: list[dict], log) -> dict:
    all_games = np.concatenate([r["unique_games"] for r in fold_results])
    all_bets = np.concatenate([r["bets_per_game"] for r in fold_results])
    all_profit = np.concatenate([r["profit_per_game"] for r in fold_results])

    if len(np.unique(all_games)) != len(all_games):
        raise AssertionError(
            "POOLING FAILED: game_id collision across test folds -- the same game_id appears in "
            "more than one fold's test set, which should be impossible given non-overlapping "
            "season windows. Investigate before pooling."
        )

    total_bets = int(sum(r["whole_fold"]["total_bets"] for r in fold_results))
    total_profit = float(sum(r["whole_fold"]["total_profit"] for r in fold_results))
    point_estimate_roi = (total_profit / total_bets * 100.0) if total_bets else None

    ci = bootstrap_game_level_roi(all_bets, all_profit, BOOTSTRAP_SEED, N_BOOTSTRAP)

    log("\n" + "=" * 80)
    log("POOLED (across all test folds)")
    log("=" * 80)
    log(f"Pooled games: {len(all_games)}, pooled bets: {total_bets}, pooled profit: {total_profit:+.2f} units")
    if point_estimate_roi is not None:
        log(f"Pooled point-estimate ROI: {point_estimate_roi:+.2f}%")
    else:
        log("Pooled point-estimate ROI: undefined (zero bets).")
    if ci["lower"] is not None:
        log(f"Pooled game-level bootstrap 95% CI: [{ci['lower']:+.2f}%, {ci['upper']:+.2f}%]")
    else:
        log("Pooled game-level bootstrap 95% CI: undefined (no resample had any bets).")

    return {
        "n_games": int(len(all_games)),
        "total_bets": total_bets,
        "total_profit": total_profit,
        "point_estimate_roi": point_estimate_roi,
        "bootstrap_ci_95": ci,
    }


def compute_verdict(fold_results: list[dict], pooled: dict, log, meaningless: bool = False) -> tuple[str, dict]:
    ci_lower = pooled["bootstrap_ci_95"]["lower"]
    cond_a = ci_lower is not None and ci_lower > 0.0
    per_origin_roi = {r["label"]: r["whole_fold"]["roi"] for r in fold_results}
    cond_b = len(per_origin_roi) > 0 and all(roi > 0.0 for roi in per_origin_roi.values())

    verdict = "PASS" if (cond_a and cond_b) else "MARKET-PARITY / FAIL"
    detail = {
        "condition_a_pooled_ci_lower_gt_0": cond_a,
        "condition_b_positive_roi_each_fold": cond_b,
        "pooled_ci_lower": ci_lower,
        "per_fold_roi": per_origin_roi,
    }

    prefix = "[MEANINGLESS -- SMOKE TEST] " if meaningless else ""
    log("\n" + "=" * 80)
    log(f"{prefix}VERDICT (section 21.4, applied literally): {verdict}")
    log("=" * 80)
    log(f"{prefix}Condition (a) pooled 95% CI lower bound > 0: {cond_a} (lower={ci_lower})")
    log(f"{prefix}Condition (b) positive ROI point estimate in EACH fold: {cond_b} ({per_origin_roi})")
    if meaningless:
        log(
            "[MEANINGLESS -- SMOKE TEST] This verdict is computed on 2023-24-only internal "
            "wiring-test data, NOT the registered folds. It proves the verdict logic runs "
            "end to end and means nothing about the recipe's real out-of-sample edge."
        )

    return verdict, detail


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward validation of the production classifier recipe "
            "(docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 21)."
        )
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Run a fast internal wiring self-test using ONLY 2023-24 data (never a test "
            "fold in the registered folds), using the frozen recipe's own n_estimators "
            "(no override -- a weakened model can place zero bets and leave the "
            "reconciliation/bootstrap code paths unexercised). Exercises the full "
            "downstream path (evaluate_profitability, per-game reconciliation, bootstrap, "
            "diagnostics, artifact writing) end to end, and hard-fails if it places zero "
            "bets. Produces meaningless numbers."
        ),
    )
    parser.add_argument(
        "--confirm-single-run", action="store_true",
        help=(
            "Required to run the REAL walk-forward evaluation (i.e. without --smoke). "
            "Section 21.6 registers this evaluation to run EXACTLY ONCE: no re-carving "
            "folds, no re-running a fold after seeing its result, no threshold re-sweep. "
            "This flag exists so an accidental bare invocation (shell-history recall, "
            "tab-completion) cannot consume that single run. Ignored (and unnecessary) "
            "with --smoke."
        ),
    )
    args = parser.parse_args()

    if not args.smoke and not args.confirm_single_run:
        print(
            "REFUSING TO RUN: this is the real (non-smoke) walk-forward evaluation, and "
            "section 21.6 of docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md registers it to "
            "run EXACTLY ONCE -- no re-carving folds, no re-running a fold after seeing "
            "its result, no threshold re-sweep. Refusing to load data or train anything "
            "without an explicit --confirm-single-run flag, so a stray shell-history "
            "recall or tab-completion accident cannot consume that single run.\n"
            "  - For a wiring self-test (2023-24 data only, never a registered test "
            "fold): python scripts/experiment_walk_forward_classifier.py --smoke\n"
            "  - For the real, single registered run (only once you intend exactly "
            "that): python scripts/experiment_walk_forward_classifier.py --confirm-single-run"
        )
        return 1

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_smoke" if args.smoke else ""
    output_dir = OUTPUT_ROOT / f"experiment_walk_forward_classifier_{timestamp}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log, flush_log = make_logger(log_path)

    metadata: dict = {"timestamp": datetime.now().isoformat(), "smoke_mode": args.smoke}

    try:
        log("=" * 80)
        log("WALK-FORWARD VALIDATION OF THE PRODUCTION CLASSIFIER RECIPE")
        log("Pre-registration: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 21")
        log("=" * 80)
        log(f"Output directory: {output_dir}")

        if args.smoke:
            log("")
            log("#" * 80)
            log("# SMOKE TEST MODE -- WIRING SELF-TEST ONLY.")
            log("# Uses ONLY 2023-24 data (a training window in BOTH registered origins, and")
            log("# NEVER a registered test fold), with the frozen recipe's own n_estimators")
            log("# (no override -- hard-fails below if it places zero bets).")
            log("# EVERY NUMBER BELOW IS MEANINGLESS. NO REGISTERED TEST FOLD IS TOUCHED.")
            log("#" * 80)
        else:
            log("")
            log("#" * 80)
            log("# REAL WALK-FORWARD EVALUATION -- touches 2024-25 and 2025-26 test data.")
            log("# Section 21.6 single-run discipline applies: no re-carving folds, no")
            log("# re-running a fold after seeing its result, no threshold re-sweep.")
            log("#" * 80)

        for path in (DATA_PATH, PROD_MODEL_DIR, FEATURE_NAMES_PATH, PROD_METADATA_PATH, TUNE_SCRIPT_PATH):
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required input: {path}")

        git_commit = get_git_commit(log)

        log("\n" + "=" * 80)
        log("GUARDS: verbatim engineered-features copy, frozen feature list, frozen hyperparameters")
        log("=" * 80)
        verify_engineered_features_verbatim(log)

        checksums = {
            "multibook_classification_training_data.parquet": sha256_of_file(DATA_PATH),
            "classifier_feature_names.json": sha256_of_file(FEATURE_NAMES_PATH),
            "classifier_metadata.json": sha256_of_file(PROD_METADATA_PATH),
        }
        log(f"Input checksums: {checksums}")
        (output_dir / "input_checksums.json").write_text(
            json.dumps(checksums, indent=2), encoding="utf-8"
        )

        prod_metadata = json.loads(PROD_METADATA_PATH.read_text(encoding="utf-8"))
        hyperparameters = prod_metadata["hyperparameters"]
        verify_frozen_hyperparameters(hyperparameters, log)

        log("\n" + "=" * 80)
        log("DATA + FEATURES (mirrors tune_hyperparameters.py lines 112-158)")
        log("=" * 80)
        df, feature_cols, season_labels, pre_filter_counts, post_filter_counts = load_and_prepare_data(log)
        verify_frozen_feature_list(feature_cols, log)

        X_full = df[feature_cols].values
        y_full = df["over_hit"].values

        if args.smoke:
            # --- SMOKE: internal date split within 2023-24 only ---
            log("\n" + "=" * 80)
            log("SMOKE FOLD: internal date split within 2023-24 ONLY")
            log("=" * 80)
            season_positions = np.where(season_labels == "2023-24")[0]
            season_dates = df["game_date"].values[season_positions]
            split_i = int(len(season_positions) * 0.7)
            cutoff_date = season_dates[split_i]
            smoke_train_idx = season_positions[season_dates < cutoff_date]
            smoke_test_idx = season_positions[season_dates >= cutoff_date]
            log(
                f"2023-24 has {len(season_positions)} rows; internal cutoff date="
                f"{pd.Timestamp(cutoff_date).date()} -> smoke-train n={len(smoke_train_idx)}, "
                f"smoke-test n={len(smoke_test_idx)} (NOT a registered test fold)."
            )

            fold_result = run_fold(
                "smoke_fold", df, feature_cols, X_full, y_full, smoke_train_idx, smoke_test_idx,
                hyperparameters, output_dir, log,
            )

            # A self-test that can pass without exercising the two code paths
            # it exists to prove (per-game reconciliation, game-level
            # bootstrap) is worse than no self-test at all: both paths handle
            # a zero-bet fold "correctly" but trivially (0 == 0; bootstrap's
            # "no resample had any bets" branch). Hard-fail here so a
            # vacuous smoke run cannot be mistaken for a passing one.
            smoke_bets = fold_result["whole_fold"]["total_bets"]
            if smoke_bets == 0:
                raise AssertionError(
                    "SMOKE TEST VACUOUS: 0 bets placed, so the reconciliation and bootstrap "
                    "paths were not exercised; the self-test cannot pass without them. This "
                    "is a wiring-test failure, not a comment on the recipe's edge (2023-24 "
                    "internal split only -- no registered test fold was touched)."
                )
            log(f"Smoke-vacuity check PASSED: {smoke_bets} bets placed, reconciliation and bootstrap were genuinely exercised.")

            fold_results = [fold_result]
            pooled = pool_folds(fold_results, log)
            verdict, verdict_detail = compute_verdict(fold_results, pooled, log, meaningless=True)

            metadata["smoke_fold"] = fold_result_to_json_safe(fold_result)
            metadata["pooled"] = pooled
            metadata["verdict"] = verdict
            metadata["verdict_detail"] = verdict_detail
            (output_dir / "smoke_fold_results.json").write_text(
                json.dumps(fold_result_to_json_safe(fold_result), indent=2, default=_json_default), encoding="utf-8"
            )
            (output_dir / "pooled_results.json").write_text(
                json.dumps(pooled, indent=2, default=_json_default), encoding="utf-8"
            )

        else:
            # --- REAL: Origin 1 and Origin 2, section 21.2 ---
            log("\n" + "=" * 80)
            log("ORIGIN 1: train = 2023-24, test = 2024-25")
            log("=" * 80)
            train_idx_1, test_idx_1 = carve_origin(df, season_labels, ["2023-24"], "2024-25", log, "Origin 1")
            origin_1_result = run_fold(
                "origin_1", df, feature_cols, X_full, y_full, train_idx_1, test_idx_1,
                hyperparameters, output_dir, log,
            )

            log("\n" + "=" * 80)
            log("ORIGIN 2: train = 2023-24 + 2024-25, test = 2025-26")
            log("=" * 80)
            train_idx_2, test_idx_2 = carve_origin(
                df, season_labels, ["2023-24", "2024-25"], "2025-26", log, "Origin 2",
            )
            origin_2_result = run_fold(
                "origin_2", df, feature_cols, X_full, y_full, train_idx_2, test_idx_2,
                hyperparameters, output_dir, log,
            )

            # Each test season touched exactly once as a TEST fold. Derived
            # from what ACTUALLY ran (season_labels at the test positions
            # carved by carve_origin above), not a hardcoded literal -- a
            # literal would assert properties of itself, not of the carving.
            test_season_labels_1 = np.unique(season_labels[test_idx_1])
            test_season_labels_2 = np.unique(season_labels[test_idx_2])
            if len(test_season_labels_1) != 1:
                raise AssertionError(
                    f"FOLD DISCIPLINE VIOLATION: Origin 1's test rows span "
                    f"{len(test_season_labels_1)} distinct season labels {list(test_season_labels_1)}, "
                    "expected exactly 1 (section 21.2 test folds are single-season)."
                )
            if len(test_season_labels_2) != 1:
                raise AssertionError(
                    f"FOLD DISCIPLINE VIOLATION: Origin 2's test rows span "
                    f"{len(test_season_labels_2)} distinct season labels {list(test_season_labels_2)}, "
                    "expected exactly 1 (section 21.2 test folds are single-season)."
                )
            test_season_1 = str(test_season_labels_1[0])
            test_season_2 = str(test_season_labels_2[0])
            if test_season_1 == test_season_2:
                raise AssertionError(
                    f"FOLD DISCIPLINE VIOLATION: both origins tested the same season "
                    f"({test_season_1}) -- each test season must be evaluated exactly once "
                    "(section 21.2)."
                )
            if "2023-24" in (test_season_1, test_season_2):
                raise AssertionError(
                    f"FOLD DISCIPLINE VIOLATION: 2023-24 was used as a TEST season "
                    f"(Origin 1 test={test_season_1}, Origin 2 test={test_season_2}); "
                    "2023-24 must only ever be a training window (section 21.2)."
                )
            log(
                f"\nFold discipline check PASSED (derived from actual test-fold season labels, "
                f"not a hardcoded literal): Origin 1 tested {test_season_1} only, Origin 2 tested "
                f"{test_season_2} only, the two differ, and neither is 2023-24."
            )

            fold_results = [origin_1_result, origin_2_result]
            pooled = pool_folds(fold_results, log)
            verdict, verdict_detail = compute_verdict(fold_results, pooled, log, meaningless=False)

            metadata["origin_1"] = fold_result_to_json_safe(origin_1_result)
            metadata["origin_2"] = fold_result_to_json_safe(origin_2_result)
            metadata["pooled"] = pooled
            metadata["verdict"] = verdict
            metadata["verdict_detail"] = verdict_detail

            (output_dir / "origin_1_results.json").write_text(
                json.dumps(fold_result_to_json_safe(origin_1_result), indent=2, default=_json_default), encoding="utf-8"
            )
            (output_dir / "origin_2_results.json").write_text(
                json.dumps(fold_result_to_json_safe(origin_2_result), indent=2, default=_json_default), encoding="utf-8"
            )
            (output_dir / "pooled_results.json").write_text(
                json.dumps(pooled, indent=2, default=_json_default), encoding="utf-8"
            )

        metadata["git_commit"] = git_commit
        metadata["input_checksums"] = checksums
        metadata["ev_threshold"] = EV_THRESHOLD
        metadata["hyperparameters"] = hyperparameters
        metadata["xgb_seed"] = XGB_SEED
        metadata["bootstrap_seed"] = BOOTSTRAP_SEED
        metadata["n_bootstrap_resamples"] = N_BOOTSTRAP
        metadata["feature_cols"] = feature_cols
        metadata["pre_filter_counts_by_season"] = pre_filter_counts
        metadata["post_filter_counts_by_season"] = post_filter_counts
        metadata["registered_pre_filter_figures"] = REGISTERED_PRE_FILTER
        metadata["season_bounds"] = {
            k: [str(v[0].date()), str(v[1].date())] for k, v in SEASON_BOUNDS.items()
        }

        elapsed = time.time() - start_time
        metadata["wall_clock_seconds"] = elapsed
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
        log(f"\nSaved metadata to: {metadata_path}")
        log(f"Wall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

        if args.smoke:
            log("\n" + "#" * 80)
            log("# SMOKE TEST COMPLETE. Every number above is meaningless (2023-24-only")
            log("# internal split). No registered test fold (2024-25 / 2025-26) was touched.")
            log("#" * 80)
        else:
            log("\n" + "=" * 80)
            log("WALK-FORWARD EVALUATION COMPLETE")
            log("=" * 80)
        flush_log()
    except Exception:
        flush_log()
        raise

    print(f"Saved run log to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
