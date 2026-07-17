"""
Experiment 15 -- W3 saves-market microstructure feature block (juice skew).

Contract: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 18 (18.1-18.8),
registered 2026-07-14 (amended same day to the {Origin B, Origin C} gating
set, Origin A reclassified as a registered placebo/negative control, 18.3a).
Read section 18 in full before touching this script. This is a direct
sibling of Experiment 5 (scripts/experiment_market_state_features.py,
section 6) -- same variant mechanics, same script lineage -- extended with a
second, book-agnostic feature family (juice_* / price-shape) instead of the
market-game-state one. Where this docstring and section 18 disagree, section
18 wins; this file is the implementation, not a re-registration.

ARCHITECTURE (mirrors the template, extended from 2 to 4 variants):

Three rolling origins (18.3), fold boundaries carved via
experiment_rolling_origin.carve_origin_split/season_date_range/
date_range_test_idx, unchanged:
  Origin A (PLACEBO, 18.3a): pool 2022-10-07..2023-04-14, test = 2023-24.
  Origin B (GATING):         pool 2022-10-07..2024-04-18, test = 2024-25.
  Origin C (GATING):         pool 2022-10-07..2025-04-17, test = 2025-26.

Four shots-model variants per origin, one shared save-rate model per origin
(trained once on the 104-column no-pace-control list, reused by all four
variants -- literally trained once, not four times, mirroring section 6's
"shared by both variants" extended to four):
  no_pace_control                                -- 104 cols.
  control_plus_microstructure                    -- 104 + 6 juice_* + juice_matched = 111.
  control_plus_market_state                      -- 104 + 7 mkt_* + mkt_matched = 112.
  control_plus_market_state_plus_microstructure  -- 112 + 7 = 119.
juice_* features enter the SHOTS-AGAINST model ONLY (18.3, mirroring where
mkt_* went -- verified in 18.8 against experiment_market_state_
20260710_213106/metadata.json before this run).

MANDATORY WIRING GATE (18.3, "before any juice_* quote is loaded"). Read
literally and applied conservatively in this implementation: not only is no
juice_* FEATURE computed before the gate passes, no raw microstructure quote
archive (data/processed/saves_lines_snapshots.parquet's bettime pass,
data/processed/core_bettime_202607_snapshots.parquet) is even opened before
the gate passes -- including for BETTIME grading-frame construction, which
this experiment also needs (18.4/18.5's bettime secondaries). This is a
stricter reading than strictly required (the registered gate targets are all
closing-pass/workload/alpha values, per 18.3's own text), chosen because it
removes any interpretive risk at zero cost: the two wiring-gate variants
(no_pace_control, control_plus_market_state) are trained ONCE, pre-gate, on
the CLOSING pass only; once the gate passes and the bettime archives are
opened, their bettime-pass predictions are obtained WITHOUT a second
training by reloading the already-saved shots-model JSON (bit-identical
round-trip, sanity-checked against the original in-memory closing-pass
p_over to 1e-9 before being trusted -- exactly the reload pattern
scripts/experiment_market_state_origin_c.py already established for its own
train-fitted-dispersion sensitivity pass) and re-pricing against the new
bettime frame. This keeps the total shots-model-training count at exactly
12 (4 variants x 3 origins, each trained exactly once), matching the task's
own stated expectation, while respecting "no microstructure quote loaded
before the gate" literally rather than narrowly.

Gate targets (18.3, values re-stated here for a single source of truth --
cross-checked against the frozen metadata.json files at runtime, not
hand-copied blind):
  Origin A/B: experiment_market_state_20260710_213106/metadata.json's
    brier_vs_control_closing and shots_mae_delta_vs_control (mean to 1e-4,
    exact n_bets/n_clusters), plus both variants' val-fitted alphas.
  Origin C: experiment_market_state_origin_c_20260713_140706/metadata.json's
    P1 val_fitted_headline (mean to 1e-4, exact n_bets/n_clusters/
    n_push_excluded), both variants' workload bias/MAE, both variants'
    val-fitted alphas.
If ANY check fails: sys.exit(1), artifacts preserved, no juice_* quote is
loaded (STOP-AND-REPORT, 18.3/18.6).

FEATURE CONSTRUCTION (18.2). Six features + juice_matched indicator, built
ONLY from bettime player_total_saves-market quotes, book-agnostic,
proportional (multiplicative) de-vig per book/line/two-sided pair (17.2's
method, restated as binding in 18.2), modal line L* = the line quoted by the
most qualifying (two-sided, non-DFS) books, ties broken by the LOWEST
numeric line. DFS venues (prizepicks, underdog) excluded from the book
universe at every step; one-sided quotes contribute nothing anywhere. NaN is
never imputed (XGBoost native missing-value handling): features 1,2,3,6 are
NaN wherever juice_matched==0 (feature 6 additionally NaN wherever
saves_rolling_5 is NaN, inherited from that existing column); juice_n_books
is 0 (a real count) when unmatched; juice_line_dispersion is NaN when
unmatched (no line data to spread over). The "0.0 for a single qualifying
book/line" convention (features 3 and 5) falls out of ddof=0 population
std's own math for an n=1 sample -- no special-case branch is written for it
(verified empirically before this run: every observed single-book/
single-line goalie-night in this archive already lands on exactly 0.0).

Section 17.2's max-resolved_ts within-pass dedup rule is applied uniformly
to every bettime archive this block reads (2023-24 and 2025-26 from
saves_lines_snapshots.parquet, 2024-25 from core_bettime_202607_
snapshots.parquet), by reference per 18.4/18.8. Verified empirically before
this run (read-only, no feature value computed) that applying this dedup
changes NONE of 18.8's four registered coverage percentages -- the
dedup only resolves which duplicate ROW is kept inside an already-qualifying
group, it does not change which goalie-nights qualify, on this data. The
coverage-reconciliation step below (18.5/18.8) hard-stops if a fresh
computation disagrees with the registered figures regardless.

Feature sources (18.4), by season:
  2023-24 bettime: data/processed/saves_lines_snapshots.parquet,
    snapshot_pass=="bettime", season window 2023-10-10..2024-04-18.
  2024-25 bettime: data/processed/core_bettime_202607_snapshots.parquet,
    pass_name=="combined-2024-25", market_key=="player_total_saves" only
    (the pre-existing 258-row saves_lines_snapshots.parquet 2024-25 bettime
    fragment is TOTALLY EXCLUDED, per 17.4/14.3a, reused by reference).
  2025-26 bettime: data/processed/saves_lines_snapshots.parquet,
    snapshot_pass=="bettime", season window 2025-10-07..2026-04-19 (Origin
    C's test-fold feature source under the 2026-07-14 amendment).
Outcomes/goalie-nights: data/processed/clean_training_data.parquet ONLY.
data/betting.db is never opened anywhere in this script, reads included.

STATISTICS (18.5). PRIMARY = paired shots |error| delta and paired
per-quote Brier delta (CLOSING pass), control_plus_microstructure minus
no_pace_control, on all three origins; PASS requires CI95 upper bound < 0 on
EITHER metric on BOTH Origin B and Origin C independently (Experiment 5's
own passes_on_both_origins/overall_pass=any(...) logic, re-based onto the
amended gating set). SECONDARY = the same two metrics for
control_plus_market_state_plus_microstructure minus the retrained
control_plus_market_state, reported unconditionally (no gate), read to
decide REDUNDANT vs ADDITIVE. BETTIME is an unconditional secondary
universe on both comparisons, all three origins. Origin A's copies of both
comparisons are the 18.3a PLACEBO readout: both variants have zero real
juice_* training exposure there (the bettime archive starts 2023-11-02,
seven months after Origin A's pool ends), so a CI95 excluding zero in either
direction is registered as a pipeline-defect red flag, not a finding, and
triggers a mandatory STOP-AND-INVESTIGATE before any Origin B/C verdict
language is used (18.3a/18.7).

Cluster bootstrap: 10,000 resamples, seed 42, goalie-night clusters
(f"{game_id}_{goalie_id}"), via clv_audit_pace_policy.cluster_bootstrap_
mean_ci -- the template's own unmodified bootstrap engine, used for every
statistic in this script (wiring gate AND the new PRIMARY/SECONDARY
statistics) for internal consistency, per this task's explicit instruction.
The wiring gate calls experiment_market_state_features.paired_brier_delta_
vs_variant / paired_shots_mae_delta and experiment_market_state_origin_c.
paired_brier_delta_p1 VERBATIM (not a copy) to give the strongest possible
reproduction guarantee. The new PRIMARY/SECONDARY statistics use a small
local wrapper (paired_brier_delta_generic / paired_shots_mae_delta_generic)
that reproduces the identical delta construction but (a) avoids emsf's
wrapper's hardcoded "control_plus_market_state - no_pace_control" log
string, which would be misleading for this script's other variant pairs
(mirrors the precedent already set by fit_dispersion's own hardcoded "TRAIN
residuals" string, module docstring point 6 of the template), and (b)
exposes the raw per-row delta array for the bootstrap-cluster-inputs
artifact. This wrapper's correctness is cross-checked at runtime against
emsf's own function on the Origin B wiring-gate statistic (must match to
1e-9) before being trusted for anything else -- see verify_generic_
wrapper_matches_template below.

No betting-policy ROI is computed or selected anywhere in this script;
run_shots_variant's internal policy_roi block (fixed EV threshold 0.05,
reused unchanged from experiment_rolling_origin.py) is carried only because
join_and_price/grade_bets are reused unchanged, exactly mirroring how
Experiment 5 itself reported it as an unconditional secondary, never a gate.

Zero network calls. No writes to any existing file. No modification of any
pre-existing models/trained/ directory. No touching data/betting.db.

Usage:
    python scripts/experiment_15_w3_microstructure.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Reuse the existing machinery unchanged -- fold construction, betting-frame
# construction, the SHOTS_CONFIGS/SAVE_RATE_CONFIGS grid, val-fitted
# dispersion, the cluster bootstrap, and the whole per-variant train+
# evaluate loop (run_shots_variant). Nothing below reimplements any of this.
import experiment_rolling_origin as ero  # noqa: E402
import clv_audit_pace_policy as clv  # noqa: E402
import experiment_market_state_features as emsf  # noqa: E402
import experiment_market_state_origin_c as emsc  # noqa: E402
import experiment_11_frozen_origin_b_p2 as e11  # noqa: E402
from experiments.distributional_saves import (  # noqa: E402
    SavesDistribution,
    build_betting_frame,
    compute_distribution_predictions,
    join_and_price,
    load_modeling_frame,
    train_save_rate_model,
)
from experiments.harness import (  # noqa: E402
    betting_metrics_bundle,
    fold_wide_auc_brier,
    grade_bets,
)

make_logger = ero.make_logger

# ---------------------------------------------------------------------------
# Paths. Pre-gate paths never touch a raw microstructure-quote archive.
# Post-gate paths are opened only after the wiring gate has passed.
# ---------------------------------------------------------------------------

DATA_PATH_CLEAN = REPO_ROOT / "data" / "processed" / "clean_training_data.parquet"
DATA_PATH_CONTEXT = REPO_ROOT / "data" / "processed" / "game_context_features.parquet"
DATA_PATH_MULTIBOOK = REPO_ROOT / "data" / "processed" / "multibook_classification_training_data.parquet"
MARKET_PATH = REPO_ROOT / "data" / "processed" / "market_game_features.parquet"
FROZEN_AB_METADATA = REPO_ROOT / "models" / "trained" / "experiment_market_state_20260710_213106" / "metadata.json"
FROZEN_C_METADATA = (
    REPO_ROOT / "models" / "trained" / "experiment_market_state_origin_c_20260713_140706" / "metadata.json"
)

CLOSING_FRAME_A_PATH = ero.CLOSING_FRAME_PATH  # data/processed/multibook_frame_2023_24.parquet (pre-built cache)
BETTIME_FRAME_A_PATH = ero.BETTIME_FRAME_PATH  # data/processed/multibook_frame_2023_24_bettime.parquet (pre-built)

# Post-gate only -- never opened before the wiring gate passes.
SNAPSHOTS_PATH = REPO_ROOT / "data" / "processed" / "saves_lines_snapshots.parquet"
CORE_BETTIME_PATH = REPO_ROOT / "data" / "processed" / "core_bettime_202607_snapshots.parquet"

OUTPUT_ROOT = REPO_ROOT / "models" / "trained"

SEASON_2025_26 = 20252026

N_BOOTSTRAP_RESAMPLES = ero.N_BOOTSTRAP_RESAMPLES  # 10000
BOOTSTRAP_SEED = ero.BOOTSTRAP_SEED  # 42
FIXED_EV_THRESHOLD = ero.FIXED_EV_THRESHOLD  # 0.05, unused for any policy decision here
ORIGIN_CAP = ero.ORIGIN_CAP  # 90
COVERAGE_FLOOR_PCT = 50.0  # 18.5's coverage-insufficient floor for the gating origins

GATE_TOLERANCE = 1e-4

DFS_BOOKS = {"prizepicks", "underdog"}

JUICE_FEATURE_COLS = [
    "juice_p_under_consensus",
    "juice_overround_median",
    "juice_p_under_dispersion",
    "juice_n_books",
    "juice_line_dispersion",
    "juice_line_minus_baseline",
]
JUICE_INDICATOR_COL = "juice_matched"
ALL_JUICE_COLS = JUICE_FEATURE_COLS + [JUICE_INDICATOR_COL]

VARIANT_NAMES_15 = (
    "no_pace_control",
    "control_plus_microstructure",
    "control_plus_market_state",
    "control_plus_market_state_plus_microstructure",
)
GATE_VARIANTS = ("no_pace_control", "control_plus_market_state")
MICROSTRUCTURE_VARIANTS = ("control_plus_microstructure", "control_plus_market_state_plus_microstructure")

# ---------------------------------------------------------------------------
# Registered wiring-gate reference values (section 18.3), re-stated here for
# a single source of truth. Cross-checked against the frozen metadata.json
# files at runtime (not hand-copied blind) inside the wiring-gate section.
# ---------------------------------------------------------------------------

WIRING_GATE_REFS = {
    "origin_a_brier_closing": {"mean": 0.000537819408872642, "n_bets": 8880, "n_clusters": 2298},
    "origin_b_brier_closing": {"mean": -0.0041404240194266384, "n_bets": 7463, "n_clusters": 2510},
    "origin_a_shots_mae": {"mean": 0.009711408033603576, "n": 2624},
    "origin_b_shots_mae": {"mean": -0.07375802354114812, "n": 2624},
    "origin_a_alpha": {"no_pace_control": 0.033265130249687844, "control_plus_market_state": 0.03312151062457906},
    "origin_b_alpha": {"no_pace_control": 0.02852173299997726, "control_plus_market_state": 0.026775577916660614},
    "origin_c_alpha": {"no_pace_control": 0.027718386433224374, "control_plus_market_state": 0.026939644644863918},
    "origin_c_p1": {
        "mean": -0.003111099251412182,
        "n_bets": 5729,
        "n_clusters": 2070,
        "n_push_excluded": 0,
    },
    "origin_c_workload": {
        "no_pace_control": {"bias": 0.23460506447931614, "mae": 5.407952885075313},
        "control_plus_market_state": {"bias": 0.42204831794994635, "mae": 5.3598914618899185},
    },
}

# Registered coverage-reconciliation targets (18.5/18.8), (n_matched, n_total).
COVERAGE_REFS = {
    "origin_b_train": (1349, 4528),
    "origin_b_val": (625, 720),
    "origin_c_train": (3520, 7134),
    "origin_c_val": (597, 738),
    "origin_a_test": (1974, 2624),
    "origin_b_test_nondfs": (2143, 2624),
    "origin_c_test": (1394, 2624),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 17.2 dedup, reused by reference for every bettime archive (18.4/18.8).
# ---------------------------------------------------------------------------


def dedup_max_resolved_ts(df: pd.DataFrame, key_cols: list[str], log, label: str) -> pd.DataFrame:
    """Section 17.2's registered dedup rule, restated as binding for 18.2's
    feature construction (18.4/18.8, "applies uniformly ... by reference,
    generalized"): within a single snapshot_pass, for rows sharing key_cols,
    keep only the row with the MAXIMUM resolved_ts; ties broken by maximum
    requested_ts; any remaining tie broken deterministically by original row
    order. Verified pre-execution (read-only) that applying this dedup does
    not change any of 18.8's four registered coverage percentages on this
    archive -- it only resolves which duplicate row survives inside an
    already-qualifying group."""
    df = df.reset_index(drop=True).copy()
    n_before = len(df)
    df["_orig_order"] = np.arange(n_before)
    df["_resolved_ts_parsed"] = pd.to_datetime(df["resolved_ts"], utc=True, errors="coerce")
    df["_requested_ts_parsed"] = pd.to_datetime(df["requested_ts"], utc=True, errors="coerce")

    n_nat = int(df["_resolved_ts_parsed"].isna().sum())
    if n_nat:
        log(f"[{label}] WARNING: {n_nat} rows have an unparseable resolved_ts (treated as earliest, never preferred).")

    group_sizes = df.groupby(key_cols, sort=False).size()
    n_groups = int(len(group_sizes))
    n_dup_groups = int((group_sizes > 1).sum())

    # Residual-tie count (17.2: "any remaining tie broken deterministically by
    # original row order, with the tie count logged"): rows tied on BOTH max
    # resolved_ts and max requested_ts within their group.
    grp_max_res = df.groupby(key_cols)["_resolved_ts_parsed"].transform("max")
    at_max_res = df[df["_resolved_ts_parsed"] == grp_max_res]
    grp_max_req = at_max_res.groupby(key_cols)["_requested_ts_parsed"].transform("max")
    at_max_both = at_max_res[at_max_res["_requested_ts_parsed"] == grp_max_req]
    n_residual_tie_groups = int((at_max_both.groupby(key_cols).size() > 1).sum())

    # na_position="first" so NaT timestamps sort earliest and can never win
    # the tail(1) max-timestamp selection over a real timestamp.
    df_sorted = df.sort_values(
        key_cols + ["_resolved_ts_parsed", "_requested_ts_parsed", "_orig_order"], na_position="first"
    )
    deduped = df_sorted.groupby(key_cols, as_index=False, sort=False).tail(1)
    deduped = deduped.drop(columns=["_orig_order", "_resolved_ts_parsed", "_requested_ts_parsed"]).reset_index(
        drop=True
    )
    n_after = len(deduped)
    log(
        f"[{label}] 17.2 dedup on natural key {key_cols}: {n_before} rows / {n_groups} groups; "
        f"{n_dup_groups} groups have >1 row; kept 1 row/group (max resolved_ts, tie->max requested_ts, "
        f"tie->original row order; residual-tie groups broken by row order: {n_residual_tie_groups}) "
        f"-> {n_after} rows ({n_before - n_after} dropped)."
    )
    return deduped


# ---------------------------------------------------------------------------
# 18.2 feature construction: paired (two-sided) quote table per season
# source, then a single combined modal-line / at-modal aggregation pass.
# ---------------------------------------------------------------------------


def build_paired_quotes(
    raw: pd.DataFrame,
    key_cols: list[str],
    book_col: str,
    game_lookup: dict,
    log,
    label: str,
) -> pd.DataFrame:
    """One season source -> paired (two-sided) book/line quote rows:
    dedup (17.2) -> DFS exclusion (18.2) -> goalie_id resolved rows only (no
    name-fallback, matching 18.8's own inventory-check convention, verified
    to reproduce the registered coverage figures exactly) -> attach_game_id
    ((goalie_id, game_date_eastern) +/-1-day, clv_audit_pace_policy) ->
    pivot_both_sides (book contributes only when BOTH sides of the exact
    same line are present, 18.2's fail-closed rule)."""
    df = dedup_max_resolved_ts(raw, key_cols, log, label)

    n_dfs = int(df[book_col].isin(DFS_BOOKS).sum())
    df = df[~df[book_col].isin(DFS_BOOKS)].copy()
    log(f"[{label}] DFS exclusion ({sorted(DFS_BOOKS)}): {n_dfs} rows dropped -> {len(df)} rows.")

    n_null_goalie = int(df["goalie_id"].isna().sum())
    df = df[df["goalie_id"].notna()].copy()
    df["goalie_id"] = df["goalie_id"].astype(int)
    log(f"[{label}] null goalie_id dropped: {n_null_goalie} -> {len(df)} rows.")

    df, n_unmatched = clv.attach_game_id(df, game_lookup)
    log(f"[{label}] attach_game_id ((goalie_id, game_date_eastern) +/-1 day): {n_unmatched} unmatched dropped -> {len(df)} rows.")

    group_cols = ["event_id", "game_id", "goalie_id", book_col, "line"]
    wide = clv.pivot_both_sides(df, group_cols)
    wide = wide.rename(columns={book_col: "book"})
    wide["source"] = label
    log(f"[{label}] paired (two-sided) quote rows: {len(wide)}.")
    return wide[["game_id", "goalie_id", "book", "line", "price_decimal_over", "price_decimal_under", "source"]]


def build_juice_quote_universe(clean_full: pd.DataFrame, log) -> pd.DataFrame:
    """POST-GATE ONLY. Opens the two raw microstructure-quote archives and
    builds the combined paired-quote table across all three season sources
    (18.4): 2023-24 and 2025-26 from saves_lines_snapshots.parquet
    (snapshot_pass=="bettime"), 2024-25 from core_bettime_202607_snapshots.
    parquet (pass_name=="combined-2024-25", market_key=="player_total_saves"
    only -- the pre-existing 258-row 2024-25 bettime fragment inside
    saves_lines_snapshots.parquet is TOTALLY EXCLUDED, per 17.4/14.3a)."""
    game_lookup_df = clean_full[["goalie_id", "game_date", "game_id"]].copy()
    game_lookup_df["date_str"] = pd.to_datetime(game_lookup_df["game_date"]).dt.strftime("%Y-%m-%d")
    dup_keys = int(game_lookup_df.duplicated(subset=["goalie_id", "date_str"]).sum())
    if dup_keys:
        raise AssertionError(f"{dup_keys} duplicate (goalie_id, date) keys; game_id lookup would be ambiguous.")
    game_lookup = dict(zip(zip(game_lookup_df["goalie_id"], game_lookup_df["date_str"]), game_lookup_df["game_id"]))

    log("\n" + "=" * 80)
    log("POST-GATE: opening saves_lines_snapshots.parquet and core_bettime_202607_snapshots.parquet")
    log("=" * 80)
    snapshots_all = pd.read_parquet(SNAPSHOTS_PATH)
    log(f"saves_lines_snapshots.parquet: {len(snapshots_all)} total rows.")
    bt_all = snapshots_all[snapshots_all["snapshot_pass"] == "bettime"].copy()

    bt_23_24 = bt_all[
        (bt_all["game_date_eastern"] >= "2023-10-10") & (bt_all["game_date_eastern"] <= "2024-04-18")
    ].copy()
    log(f"\n2023-24 bettime window raw rows: {len(bt_23_24)}, events: {bt_23_24['event_id'].nunique()}.")
    paired_23_24 = build_paired_quotes(
        bt_23_24, ["event_id", "goalie_name_raw", "book", "side"], "book", game_lookup, log, "2023-24 bettime"
    )

    bt_25_26 = bt_all[
        (bt_all["game_date_eastern"] >= "2025-10-07") & (bt_all["game_date_eastern"] <= "2026-04-19")
    ].copy()
    log(f"\n2025-26 bettime window raw rows: {len(bt_25_26)}, events: {bt_25_26['event_id'].nunique()}.")
    paired_25_26 = build_paired_quotes(
        bt_25_26, ["event_id", "goalie_name_raw", "book", "side"], "book", game_lookup, log, "2025-26 bettime"
    )

    core_all = pd.read_parquet(CORE_BETTIME_PATH)
    log(f"\ncore_bettime_202607_snapshots.parquet: {len(core_all)} total rows.")
    saves_24_25 = core_all[
        (core_all["pass_name"] == "combined-2024-25") & (core_all["market_key"] == "player_total_saves")
    ].copy()
    log(f"2024-25 player_total_saves rows (combined-2024-25 pass): {len(saves_24_25)}, "
        f"events: {saves_24_25['event_id'].nunique()}.")
    paired_24_25 = build_paired_quotes(
        saves_24_25, ["event_id", "player_name_raw", "book_key", "side"], "book_key", game_lookup, log, "2024-25 bettime"
    )

    paired_all = pd.concat([paired_23_24, paired_24_25, paired_25_26], ignore_index=True)
    dup_book = int(paired_all.duplicated(subset=["game_id", "goalie_id", "book"]).sum())
    if dup_book:
        raise AssertionError(
            f"{dup_book} (game_id, goalie_id, book) rows are duplicated across the combined paired-quote "
            "table -- a book should contribute at most one paired row per goalie-night by construction "
            "(dedup collapses (event,goalie,book,side) to one row; pivot_both_sides then requires a "
            "matching line on both sides). Investigate before computing any juice_* feature."
        )
    log(f"\nCombined paired bettime quote universe (all three seasons): {len(paired_all)} rows, "
        f"{paired_all[['game_id', 'goalie_id']].drop_duplicates().shape[0]} distinct goalie-nights.")
    return paired_all


def compute_juice_features(paired_all: pd.DataFrame, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """18.2's registered feature math, applied to the combined paired-quote
    table. Returns (per-goalie-night feature table, paired_all annotated
    with per-quote de-vig columns + an is_at_modal flag for the quote-
    universe artifact)."""
    df = paired_all.copy()
    df["raw_p_over"] = 1.0 / df["price_decimal_over"]
    df["raw_p_under"] = 1.0 / df["price_decimal_under"]
    df["overround"] = df["raw_p_over"] + df["raw_p_under"]
    df["p_over_devigged"] = df["raw_p_over"] / df["overround"]
    df["p_under_devigged"] = df["raw_p_under"] / df["overround"]

    n_inf = int(np.isinf(df[["overround", "p_over_devigged", "p_under_devigged"]].values).sum())
    if n_inf:
        raise AssertionError(f"{n_inf} infinite de-vig values -- a price_decimal of 0 or less slipped through.")

    # juice_n_books, juice_line_dispersion: ANY line, all qualifying books.
    n_books = df.groupby(["game_id", "goalie_id"])["book"].nunique().rename("juice_n_books")
    line_disp = (
        df.groupby(["game_id", "goalie_id"])["line"]
        .apply(lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0)))
        .rename("juice_line_dispersion")
    )

    # Modal line L*: most qualifying books; ties broken by the LOWEST line.
    line_counts = df.groupby(["game_id", "goalie_id", "line"]).size().reset_index(name="n_at_line")
    line_counts_sorted = line_counts.sort_values(
        ["game_id", "goalie_id", "n_at_line", "line"], ascending=[True, True, False, True]
    )
    modal = line_counts_sorted.drop_duplicates(subset=["game_id", "goalie_id"], keep="first")[
        ["game_id", "goalie_id", "line"]
    ].rename(columns={"line": "juice_modal_line"})

    df = df.merge(modal, on=["game_id", "goalie_id"], how="left")
    df["is_at_modal"] = df["line"] == df["juice_modal_line"]
    at_modal = df[df["is_at_modal"]]

    at_modal_agg = at_modal.groupby(["game_id", "goalie_id"]).agg(
        juice_p_under_consensus=("p_under_devigged", "median"),
        juice_overround_median=("overround", "median"),
        juice_p_under_dispersion=("p_under_devigged", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
    )

    feat = n_books.to_frame().join(line_disp, how="outer").join(modal.set_index(["game_id", "goalie_id"]), how="left")
    feat = feat.join(at_modal_agg, how="left").reset_index()
    feat["juice_matched"] = (feat["juice_n_books"] > 0).astype(int)

    n_single_book = int((feat["juice_n_books"] == 1).sum())
    single_ok = bool(
        (feat.loc[feat["juice_n_books"] == 1, "juice_p_under_dispersion"] == 0.0).all()
        and (feat.loc[feat["juice_n_books"] == 1, "juice_line_dispersion"] == 0.0).all()
    )
    log(
        f"\nJuice feature table: {len(feat)} matched goalie-nights. juice_n_books distribution:\n"
        f"{feat['juice_n_books'].value_counts().sort_index().to_dict()}"
    )
    log(
        f"Single-qualifying-book nights: {n_single_book}; dispersion features land on exactly 0.0 for all "
        f"of them (the registered '0.0 not NaN for n=1' convention, verified to fall out of ddof=0 std's "
        f"own math with no special-case code): {single_ok}."
    )
    if n_single_book and not single_ok:
        raise AssertionError("Single-book dispersion values are not all exactly 0.0 -- investigate.")

    return feat, df


def attach_juice_features(df_full: pd.DataFrame, juice_feat: pd.DataFrame, log) -> pd.DataFrame:
    """Left-join juice_* onto the modeling frame (ADD COLUMNS ONLY -- row
    count and order are asserted unchanged, so train/val/test index arrays
    computed before this call remain valid afterward). Missing-data routing
    per 18.2: features 1,2,3,6 NaN wherever juice_matched==0 (6 additionally
    NaN wherever saves_rolling_5 is NaN, inherited automatically from plain
    float subtraction); juice_n_books is 0 (not NaN) when unmatched;
    juice_line_dispersion is NaN when unmatched. Nothing is ever imputed."""
    overlap = set(ALL_JUICE_COLS) & set(df_full.columns)
    if overlap:
        raise ValueError(f"juice_* feature names collide with modeling-frame columns: {sorted(overlap)}")

    before = len(df_full)
    key = df_full[["game_id", "goalie_id"]].reset_index(drop=True)
    merged = key.merge(juice_feat, on=["game_id", "goalie_id"], how="left")
    if len(merged) != before:
        raise AssertionError("juice_* merge changed modeling-frame row count.")
    if not (merged["game_id"].values == df_full["game_id"].values).all():
        raise AssertionError("juice_* merge changed modeling-frame row order (game_id mismatch).")
    if not (merged["goalie_id"].values == df_full["goalie_id"].values).all():
        raise AssertionError("juice_* merge changed modeling-frame row order (goalie_id mismatch).")

    out = df_full.copy()
    out["juice_n_books"] = merged["juice_n_books"].fillna(0).astype(int).values
    out["juice_line_dispersion"] = merged["juice_line_dispersion"].values
    out["juice_p_under_consensus"] = merged["juice_p_under_consensus"].values
    out["juice_overround_median"] = merged["juice_overround_median"].values
    out["juice_p_under_dispersion"] = merged["juice_p_under_dispersion"].values
    out["juice_matched"] = merged["juice_matched"].fillna(0).astype(int).values
    out["juice_line_minus_baseline"] = merged["juice_modal_line"].values - out["saves_rolling_5"].values

    matched = out["juice_matched"].values.astype(bool)
    log(
        f"\njuice_* attach: {int(matched.sum())}/{len(out)} rows matched "
        f"({matched.mean() * 100:.2f}% overall across ALL seasons combined; see the per-origin/split "
        "coverage-reconciliation section for the numbers that actually determine trainability)."
    )
    null_counts = {c: int(v) for c, v in out[JUICE_FEATURE_COLS].isna().sum().items() if int(v) > 0}
    log(f"juice_* NaN counts (retained for XGBoost native missing-value handling): {null_counts}")
    return out


# ---------------------------------------------------------------------------
# Generic paired-delta cluster-bootstrap wrappers. Same construction, same
# bootstrap engine (clv.cluster_bootstrap_mean_ci), as
# emsf.paired_brier_delta_vs_variant / emsf.paired_shots_mae_delta -- written
# locally only to avoid those functions' hardcoded "control_plus_market_
# state - no_pace_control" log text (misleading for this script's other
# variant pairs) and to expose the raw per-row delta array for the
# bootstrap-cluster-inputs artifact. Cross-checked against the template's
# own functions before being trusted (see verify_generic_wrapper_matches_
# template).
# ---------------------------------------------------------------------------


def paired_brier_delta_generic(df_bet, p_over_base, matched_base, p_over_new, matched_new, log, label):
    both_matched = matched_base & matched_new
    y = (df_bet["saves"].values.astype(float) > df_bet["betting_line"].values.astype(float)).astype(float)
    sq_base = (p_over_base - y) ** 2
    sq_new = (p_over_new - y) ** 2
    delta = np.where(both_matched, sq_new - sq_base, np.nan)
    cluster_ids = np.array(
        [f"{int(g)}_{int(o)}" for g, o in zip(df_bet["game_id"].values, df_bet["goalie_id"].values)], dtype=object
    )
    stat = clv.cluster_bootstrap_mean_ci(
        delta, cluster_ids, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, ci_pct=95.0
    )
    log(
        f"[{label}] paired Brier delta (new - base): mean={stat['mean']} 95% CI=[{stat['lower']}, {stat['upper']}] "
        f"n_rows={stat['n_bets']} n_clusters={stat['n_clusters']} (negative = new variant better)"
    )
    return stat, delta, cluster_ids


def paired_shots_mae_delta_generic(df_full, test_idx, mu_base, mu_new, log, label):
    actual = df_full["shots_against"].values[test_idx].astype(float)
    delta = np.abs(mu_new - actual) - np.abs(mu_base - actual)
    cluster_ids = np.array(
        [
            f"{int(g)}_{int(o)}"
            for g, o in zip(df_full["game_id"].values[test_idx], df_full["goalie_id"].values[test_idx])
        ],
        dtype=object,
    )
    stat = clv.cluster_bootstrap_mean_ci(
        delta, cluster_ids, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, ci_pct=95.0
    )
    log(
        f"[{label}] paired shots |error| delta (new - base): mean={stat['mean']:+.6f} "
        f"95% CI=[{stat['lower']:+.6f}, {stat['upper']:+.6f}] n={stat['n_bets']} (negative = new variant better)"
    )
    return stat, delta, cluster_ids


def verify_generic_wrapper_matches_template(df_bet, p_over_base, matched_base, p_over_new, matched_new, log) -> bool:
    """Cross-check paired_brier_delta_generic against emsf.paired_brier_
    delta_vs_variant (the template's own function) on identical inputs.
    Both must call the identical clv.cluster_bootstrap_mean_ci with the
    identical delta array, so mean/n_bets/n_clusters must match exactly."""
    stat_template = emsf.paired_brier_delta_vs_variant(
        df_bet, p_over_base, matched_base, p_over_new, matched_new, log, "generic-wrapper cross-check (template fn)"
    )
    stat_generic, _delta, _cid = paired_brier_delta_generic(
        df_bet, p_over_base, matched_base, p_over_new, matched_new, log, "generic-wrapper cross-check (generic fn)"
    )
    ok = (
        stat_template["mean"] == stat_generic["mean"]
        and stat_template["n_bets"] == stat_generic["n_bets"]
        and stat_template["n_clusters"] == stat_generic["n_clusters"]
        and stat_template["lower"] == stat_generic["lower"]
        and stat_template["upper"] == stat_generic["upper"]
    )
    log(f"Generic-wrapper cross-check vs template function: EXACT MATCH={ok}")
    return ok


# ---------------------------------------------------------------------------
# Bettime-pass extension of an already-trained gate variant, without a
# second shots-model training (reload-and-reprice, mirrors experiment_
# market_state_origin_c.py's own train-fitted-dispersion reload pattern).
# ---------------------------------------------------------------------------


def evaluate_additional_pass(label_base, variant_name, dist_preds_test, dist, pass_name, df_bet, origin_label, log):
    """Reproduces the per-pass body of emsf.run_shots_variant's evaluation
    loop verbatim (same reused functions, same order), applied to an
    already-trained variant's dist_preds_test for a pass that was not yet
    available when run_shots_variant was originally called."""
    label = f"{label_base} TEST {pass_name}"
    p_over, p_under, p_push, matched, cov = join_and_price(df_bet, dist_preds_test, dist, log, label)

    auc, brier_val = fold_wide_auc_brier(
        p_over, matched, df_bet["saves"].values, df_bet["betting_line"].values,
        df_bet["game_id"].values, df_bet["goalie_id"].values, log, label,
    )
    brier_delta_stat, market_p_over_arr, market_p_under_arr = ero.paired_brier_delta(df_bet, p_over, matched, log, label)

    bet_results = grade_bets(
        p_over, p_under, df_bet["saves"].values.astype(float), df_bet["betting_line"].values.astype(float),
        df_bet["odds_over_american"].astype(float).values, df_bet["odds_under_american"].astype(float).values,
        df_bet["game_id"].values, df_bet["goalie_id"].values, FIXED_EV_THRESHOLD, matched, log, label,
    )
    bundle = betting_metrics_bundle(bet_results, df_bet["game_id"].values, df_bet["goalie_id"].values, len(df_bet))
    log(
        f"[{label}] {bundle['summary']['bets']} bets, {bundle['summary']['bet_rate']:.1f}% bet rate, "
        f"{bundle['summary']['hit_rate']:.1f}% hit rate, {bundle['summary']['roi']:+.2f}% ROI"
    )

    keep_pos = emsf.dedup_positions(df_bet, matched)
    y_full = (df_bet["saves"].values.astype(float) > df_bet["betting_line"].values.astype(float)).astype(float)
    toi_full = emsf.parse_toi_minutes(df_bet["toi"].values) if "toi" in df_bet.columns else np.full(len(df_bet), np.nan)
    p_over_dd, y_dd, toi_dd = p_over[keep_pos], y_full[keep_pos], toi_full[keep_pos]
    side_cal = emsf.side_calibration(p_over_dd, y_dd, log, label)
    tail_cal = emsf.lower_tail_calibration(p_over_dd, y_dd, toi_dd, log, label)

    row_df = ero.build_row_predictions(
        df_bet, p_over, p_under, matched, market_p_over_arr, market_p_under_arr, FIXED_EV_THRESHOLD, origin_label, pass_name,
    )
    row_df["variant"] = variant_name

    pass_result = {
        "join_coverage_pct": cov,
        "fold_wide_auc": auc,
        "fold_wide_brier": brier_val,
        "paired_brier_delta_vs_market": brier_delta_stat,
        "policy_roi": bundle,
        "side_calibration": side_cal,
        "lower_tail_calibration_toi_lt_50": tail_cal,
        "ev_threshold": FIXED_EV_THRESHOLD,
        "n_goalie_nights_deduped": int(len(keep_pos)),
    }
    return pass_result, (p_over, matched), row_df


def extend_variant_with_bettime(
    origin_label, variant_name, result_json, probs_entry, rate_model, no_pace_cols, shots_cols,
    df_full, test_idx, df_bet_closing, df_bet_bettime, log,
):
    label_base = f"origin_{origin_label.lower()}_{variant_name}"
    shots_path = result_json["shots_model"]["model_path"]
    alpha = result_json["dispersion"]["alpha"]
    shots_model_reloaded = emsc.reload_shots_model(shots_path)
    dist = SavesDistribution(ORIGIN_CAP)

    dist_preds_test = compute_distribution_predictions(
        df_full, test_idx, shots_model_reloaded, rate_model, alpha, shots_cols, no_pace_cols, dist, log,
        f"{label_base} reload for bettime extension",
    )

    p_over_closing_check, _pu, _pp, matched_closing_check, _cov = join_and_price(
        df_bet_closing, dist_preds_test, dist, log, f"{label_base} reload sanity (closing)"
    )
    p_over_closing_orig, matched_closing_orig = probs_entry["closing"]
    emsc.assert_allclose(p_over_closing_check, p_over_closing_orig, 1e-9, f"{label_base} closing reload sanity")
    if not (matched_closing_check == matched_closing_orig).all():
        raise AssertionError(f"[{label_base}] reload closing matched-mask mismatch.")
    log(f"[{label_base}] reload sanity check PASSED (closing p_over matches original in-memory training to 1e-9).")

    pass_result, (p_over_bt, matched_bt), row_df = evaluate_additional_pass(
        label_base, variant_name, dist_preds_test, dist, "bettime", df_bet_bettime, origin_label, log,
    )
    result_json["price_passes"]["bettime"] = pass_result
    probs_entry["bettime"] = (p_over_bt, matched_bt)
    return row_df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_15_w3_microstructure_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "run_log.txt"
    log, flush_log = make_logger(log_path)

    metadata: dict = {
        "timestamp": utc_now_iso(),
        "registration": "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 18 (Experiment 15 -- W3 microstructure)",
        "network_calls": False,
        "betting_db_touched": False,
        "deviations_from_registration": [],
        "judgment_calls": [],
    }
    input_checksums: dict = {}

    try:
        log("=" * 80)
        log("EXPERIMENT 15 -- W3 SAVES-MARKET MICROSTRUCTURE FEATURE BLOCK (JUICE SKEW)")
        log("docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 18")
        log("=" * 80)
        log(f"Output directory: {output_dir}")

        pre_gate_paths = {
            "clean_training_data": DATA_PATH_CLEAN,
            "game_context_features": DATA_PATH_CONTEXT,
            "multibook_classification_training_data": DATA_PATH_MULTIBOOK,
            "market_game_features": MARKET_PATH,
            "multibook_frame_2023_24": CLOSING_FRAME_A_PATH,
            "frozen_ab_metadata": FROZEN_AB_METADATA,
            "frozen_c_metadata": FROZEN_C_METADATA,
        }
        for name, path in pre_gate_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required input: {path}")
        # Existence-only check for the post-gate archives (NOT opened/read --
        # no quote is loaded), so a missing file cannot crash the run after
        # an hour of training.
        for path in (SNAPSHOTS_PATH, CORE_BETTIME_PATH, BETTIME_FRAME_A_PATH):
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required post-gate input (existence check only): {path}")
        input_checksums["pre_gate"] = {name: sha256_file(Path(p)) for name, p in pre_gate_paths.items()}
        log("Pre-gate input checksums computed (SHA-256). No microstructure-quote archive opened yet "
            "(post-gate archives existence-checked only).")

        frozen_ab = json.loads(FROZEN_AB_METADATA.read_text(encoding="utf-8"))
        frozen_c = json.loads(FROZEN_C_METADATA.read_text(encoding="utf-8"))

        # =====================================================================
        # PHASE 0: no-pace control + market-state modeling frame (NOT juice).
        # =====================================================================
        log("\n" + "=" * 80)
        log("PHASE 0: build no-pace-control + market-state modeling frame (juice_* NOT loaded yet)")
        log("=" * 80)
        events, market_stats = emsf.build_market_state_events(MARKET_PATH, log)
        frame = load_modeling_frame(DATA_PATH_CLEAN, DATA_PATH_CONTEXT, log)
        df_full = emsf.attach_market_state_features(frame.df, events, log)

        no_pace_cols = frame.base_feature_cols + frame.engineered_cols
        market_shots_cols = no_pace_cols + emsf.ALL_MARKET_COLS
        if no_pace_cols != frozen_ab["feature_sets"]["no_pace_control"]:
            raise AssertionError("no_pace_control feature list does not match the frozen Experiment 5 recipe.")
        if emsf.MARKET_FEATURE_COLS != frozen_ab["feature_sets"]["market_feature_cols"]:
            raise AssertionError("market_feature_cols does not match the frozen Experiment 5 recipe.")
        log(f"no_pace_control: {len(no_pace_cols)} cols (feature-identity check vs frozen Experiment 5 recipe: PASSED).")
        log(f"control_plus_market_state: {len(market_shots_cols)} cols.")

        clean_full = pd.read_parquet(DATA_PATH_CLEAN)
        clean_full["game_date"] = pd.to_datetime(clean_full["game_date"])

        # ---- fold boundaries, cross-checked against the frozen artifacts ----
        pool_min_a, pool_max_a = ero.season_date_range(df_full, [ero.SEASON_2022_23])
        train_idx_a, val_idx_a, boundaries_a = ero.carve_origin_split(df_full, pool_min_a, pool_max_a, ero.VAL_WINDOW_DAYS, log, "Origin A")
        test_min_a, test_max_a = ero.season_date_range(df_full, [ero.SEASON_2023_24])
        test_idx_a = ero.date_range_test_idx(df_full, test_min_a, test_max_a, log, "Origin A")

        pool_min_b, pool_max_b = ero.season_date_range(df_full, [ero.SEASON_2022_23, ero.SEASON_2023_24])
        train_idx_b, val_idx_b, boundaries_b = ero.carve_origin_split(df_full, pool_min_b, pool_max_b, ero.VAL_WINDOW_DAYS, log, "Origin B")
        test_min_b, test_max_b = ero.season_date_range(df_full, [ero.SEASON_2024_25])
        test_idx_b = ero.date_range_test_idx(df_full, test_min_b, test_max_b, log, "Origin B")

        pool_min_c, pool_max_c = ero.season_date_range(df_full, [ero.SEASON_2022_23, ero.SEASON_2023_24, ero.SEASON_2024_25])
        train_idx_c, val_idx_c, boundaries_c = ero.carve_origin_split(df_full, pool_min_c, pool_max_c, ero.VAL_WINDOW_DAYS, log, "Origin C")
        test_min_c, test_max_c = ero.season_date_range(df_full, [SEASON_2025_26])
        test_idx_c = ero.date_range_test_idx(df_full, test_min_c, test_max_c, log, "Origin C")
        if SEASON_2025_26 in set(df_full["season"].values[train_idx_c]) or SEASON_2025_26 in set(df_full["season"].values[val_idx_c]):
            raise AssertionError("2025-26 rows leaked into Origin C train/val.")

        def _filtered(frozen_fb: dict, boundaries: dict) -> dict:
            # Frozen artifacts store test_season/test_rows alongside the carve
            # output; compare only the keys carve_origin_split itself produces.
            return {k: v for k, v in frozen_fb.items() if k in boundaries}

        fold_check = {
            "origin_a": boundaries_a == _filtered(frozen_ab["origin_a"]["fold_boundaries"], boundaries_a) and len(test_idx_a) == frozen_ab["origin_a"]["fold_boundaries"]["test_rows"],
            "origin_b": boundaries_b == _filtered(frozen_ab["origin_b"]["fold_boundaries"], boundaries_b) and len(test_idx_b) == frozen_ab["origin_b"]["fold_boundaries"]["test_rows"],
            "origin_c": boundaries_c == _filtered(frozen_c["origin_c_fold_boundaries"], boundaries_c) and len(test_idx_c) == frozen_c["origin_c_fold_boundaries"]["test_rows"],
        }
        log(f"\nFold-boundary cross-check vs frozen artifacts (18.3): {fold_check}")
        if not all(fold_check.values()):
            raise AssertionError(f"Fold boundaries do not match the registered frozen artifacts: {fold_check}")
        metadata["fold_boundaries"] = {
            "origin_a": {**boundaries_a, "test_season": ero.SEASON_2023_24, "test_rows": int(len(test_idx_a))},
            "origin_b": {**boundaries_b, "test_season": ero.SEASON_2024_25, "test_rows": int(len(test_idx_b))},
            "origin_c": {**boundaries_c, "test_season": SEASON_2025_26, "test_rows": int(len(test_idx_c))},
            "cross_check_vs_frozen_artifacts_passed": True,
        }
        flush_log()

        # ---- CLOSING price frames only (pre-gate) ----
        df_bet_a_closing = build_betting_frame(CLOSING_FRAME_A_PATH, log)
        df_bet_multibook_full = build_betting_frame(DATA_PATH_MULTIBOOK, log)
        df_bet_b_closing = df_bet_multibook_full[df_bet_multibook_full["season"] == ero.SEASON_2024_25].reset_index(drop=True)
        df_bet_c_closing = df_bet_multibook_full[df_bet_multibook_full["season"] == SEASON_2025_26].reset_index(drop=True)
        log(f"\nOrigin A closing frame: {len(df_bet_a_closing)} rows. Origin B closing frame: {len(df_bet_b_closing)} rows. "
            f"Origin C closing frame: {len(df_bet_c_closing)} rows.")

        origin_specs = {
            "A": {"train_idx": train_idx_a, "val_idx": val_idx_a, "test_idx": test_idx_a, "closing": df_bet_a_closing},
            "B": {"train_idx": train_idx_b, "val_idx": val_idx_b, "test_idx": test_idx_b, "closing": df_bet_b_closing},
            "C": {"train_idx": train_idx_c, "val_idx": val_idx_c, "test_idx": test_idx_c, "closing": df_bet_c_closing},
        }

        # =====================================================================
        # PHASE 1: train rate models (shared per origin) + the two wiring-gate
        # shots variants (no_pace_control, control_plus_market_state), CLOSING
        # PASS ONLY. This is exactly the first 6 of the 12 registered shots-
        # model trainings.
        # =====================================================================
        log("\n" + "=" * 80)
        log("PHASE 1: rate models + wiring-gate shots variants (CLOSING pass only, juice_* still not loaded)")
        log("=" * 80)

        rate_models: dict = {}
        result_store: dict = {origin: {} for origin in ("A", "B", "C")}
        probs_store: dict = {origin: {} for origin in ("A", "B", "C")}
        mu_store: dict = {origin: {} for origin in ("A", "B", "C")}
        row_frames: dict = {origin: [] for origin in ("A", "B", "C")}

        for origin_label, spec in origin_specs.items():
            log("\n" + "-" * 80)
            log(f"Origin {origin_label}: shared save-rate model")
            log("-" * 80)
            rate_model, rate_winner, rate_evals = train_save_rate_model(
                df_full, spec["train_idx"], spec["val_idx"], no_pace_cols, log, f"origin_{origin_label.lower()}_shared_save_rate"
            )
            rate_path = output_dir / f"origin_{origin_label.lower()}_shared_save_rate_model.json"
            rate_model.get_booster().save_model(str(rate_path))
            rate_models[origin_label] = {"model": rate_model, "winner": rate_winner, "evals": rate_evals, "path": rate_path}

            for variant_name in GATE_VARIANTS:
                shots_cols = no_pace_cols if variant_name == "no_pace_control" else market_shots_cols
                result_json, probs_by_pass, row_predictions, mu_test = emsf.run_shots_variant(
                    origin_label, variant_name, df_full, spec["train_idx"], spec["val_idx"], spec["test_idx"],
                    shots_cols, no_pace_cols, rate_model, {"closing": spec["closing"]}, output_dir, log,
                )
                result_store[origin_label][variant_name] = result_json
                probs_store[origin_label][variant_name] = probs_by_pass
                mu_store[origin_label][variant_name] = mu_test
                row_frames[origin_label].append(row_predictions)
                flush_log()

        # =====================================================================
        # WIRING GATE (18.3): reproduce the frozen recorded values before any
        # juice_* quote is loaded.
        # =====================================================================
        log("\n" + "=" * 80)
        log("WIRING GATE (18.3): reproduce recorded Experiment 5 / Experiment 8 values")
        log("=" * 80)

        gate: dict = {}

        # --- cross-check the frozen reference values themselves against the metadata files ---
        assert frozen_ab["origin_a"]["brier_vs_control_closing"]["mean"] == WIRING_GATE_REFS["origin_a_brier_closing"]["mean"]
        assert frozen_ab["origin_b"]["brier_vs_control_closing"]["mean"] == WIRING_GATE_REFS["origin_b_brier_closing"]["mean"]
        assert frozen_ab["origin_a"]["shots_mae_delta_vs_control"]["mean"] == WIRING_GATE_REFS["origin_a_shots_mae"]["mean"]
        assert frozen_ab["origin_b"]["shots_mae_delta_vs_control"]["mean"] == WIRING_GATE_REFS["origin_b_shots_mae"]["mean"]
        assert frozen_c["P1"]["val_fitted_headline"]["mean"] == WIRING_GATE_REFS["origin_c_p1"]["mean"]
        log("Hardcoded WIRING_GATE_REFS constants verified against the frozen metadata.json files: MATCH.")

        # --- Origin A / B: closing Brier delta + shots MAE delta (control_plus_market_state - no_pace_control) ---
        for origin_label in ("A", "B"):
            base_p_over, base_matched = probs_store[origin_label]["no_pace_control"]["closing"]
            new_p_over, new_matched = probs_store[origin_label]["control_plus_market_state"]["closing"]
            df_bet_closing = origin_specs[origin_label]["closing"]

            brier_stat = emsf.paired_brier_delta_vs_variant(
                df_bet_closing, base_p_over, base_matched, new_p_over, new_matched, log,
                f"WIRING GATE origin_{origin_label} closing",
            )
            mae_stat = emsf.paired_shots_mae_delta(
                df_full, origin_specs[origin_label]["test_idx"],
                mu_store[origin_label]["no_pace_control"], mu_store[origin_label]["control_plus_market_state"],
                log, f"WIRING GATE origin_{origin_label} shots MAE",
            )

            ref_brier = WIRING_GATE_REFS[f"origin_{origin_label.lower()}_brier_closing"]
            ref_mae = WIRING_GATE_REFS[f"origin_{origin_label.lower()}_shots_mae"]
            brier_diff = abs(brier_stat["mean"] - ref_brier["mean"])
            mae_diff = abs(mae_stat["mean"] - ref_mae["mean"])
            brier_ok = brier_diff <= GATE_TOLERANCE and brier_stat["n_bets"] == ref_brier["n_bets"] and brier_stat["n_clusters"] == ref_brier["n_clusters"]
            mae_ok = mae_diff <= GATE_TOLERANCE and mae_stat["n_bets"] == ref_mae["n"]

            ref_alpha = WIRING_GATE_REFS[f"origin_{origin_label.lower()}_alpha"]
            alpha_checks = {}
            for v in GATE_VARIANTS:
                observed_alpha = result_store[origin_label][v]["dispersion"]["alpha"]
                diff = abs(observed_alpha - ref_alpha[v])
                alpha_checks[v] = {"expected": ref_alpha[v], "observed": observed_alpha, "diff": diff, "passed": diff <= GATE_TOLERANCE}

            gate[f"origin_{origin_label.lower()}"] = {
                "brier_closing": {"expected": ref_brier, "observed": {"mean": brier_stat["mean"], "n_bets": brier_stat["n_bets"], "n_clusters": brier_stat["n_clusters"]}, "abs_diff": brier_diff, "passed": brier_ok},
                "shots_mae": {"expected": ref_mae, "observed": {"mean": mae_stat["mean"], "n_bets": mae_stat["n_bets"]}, "abs_diff": mae_diff, "passed": mae_ok},
                "alphas": alpha_checks,
                "passed": bool(brier_ok and mae_ok and all(c["passed"] for c in alpha_checks.values())),
            }
            log(f"\nOrigin {origin_label} gate: brier_ok={brier_ok} mae_ok={mae_ok} alphas_ok={ {v: c['passed'] for v, c in alpha_checks.items()} }")

        # --- Origin C: P1 val-fitted headline + workload bias/MAE + alphas ---
        base_p_over_c, base_matched_c = probs_store["C"]["no_pace_control"]["closing"]
        new_p_over_c, new_matched_c = probs_store["C"]["control_plus_market_state"]["closing"]
        df_bet_c_closing_gate = origin_specs["C"]["closing"]
        p1_stat = emsc.paired_brier_delta_p1(
            df_bet_c_closing_gate, base_p_over_c, base_matched_c, new_p_over_c, new_matched_c, log,
            "WIRING GATE origin_C P1 (val-fitted headline)",
        )
        ref_p1 = WIRING_GATE_REFS["origin_c_p1"]
        p1_diff = abs(p1_stat["mean"] - ref_p1["mean"])
        p1_ok = (
            p1_diff <= GATE_TOLERANCE
            and p1_stat["n_bets"] == ref_p1["n_bets"]
            and p1_stat["n_clusters"] == ref_p1["n_clusters"]
            and p1_stat["n_push_excluded"] == ref_p1["n_push_excluded"]
        )

        ref_workload = WIRING_GATE_REFS["origin_c_workload"]
        workload_checks = {}
        for v in GATE_VARIANTS:
            wl = result_store["C"][v]["workload_shots_against_test"]
            bias_diff = abs(wl["mean_bias"] - ref_workload[v]["bias"])
            mae_diff_wl = abs(wl["mae"] - ref_workload[v]["mae"])
            workload_checks[v] = {
                "expected": ref_workload[v],
                "observed": {"bias": wl["mean_bias"], "mae": wl["mae"]},
                "bias_diff": bias_diff, "mae_diff": mae_diff_wl,
                "passed": bias_diff <= GATE_TOLERANCE and mae_diff_wl <= GATE_TOLERANCE,
            }

        ref_alpha_c = WIRING_GATE_REFS["origin_c_alpha"]
        alpha_checks_c = {}
        for v in GATE_VARIANTS:
            observed_alpha = result_store["C"][v]["dispersion"]["alpha"]
            diff = abs(observed_alpha - ref_alpha_c[v])
            alpha_checks_c[v] = {"expected": ref_alpha_c[v], "observed": observed_alpha, "diff": diff, "passed": diff <= GATE_TOLERANCE}

        gate["origin_c"] = {
            "p1_val_fitted_headline": {"expected": ref_p1, "observed": {"mean": p1_stat["mean"], "n_bets": p1_stat["n_bets"], "n_clusters": p1_stat["n_clusters"], "n_push_excluded": p1_stat["n_push_excluded"]}, "abs_diff": p1_diff, "passed": p1_ok},
            "workload": workload_checks,
            "alphas": alpha_checks_c,
            "passed": bool(p1_ok and all(c["passed"] for c in workload_checks.values()) and all(c["passed"] for c in alpha_checks_c.values())),
        }
        log(f"\nOrigin C gate: p1_ok={p1_ok} workload_ok={ {v: c['passed'] for v, c in workload_checks.items()} } alphas_ok={ {v: c['passed'] for v, c in alpha_checks_c.items()} }")

        overall_gate_passed = bool(gate["origin_a"]["passed"] and gate["origin_b"]["passed"] and gate["origin_c"]["passed"])
        gate["overall_passed"] = overall_gate_passed
        metadata["wiring_gate"] = gate
        log(f"\n{'=' * 80}\nWIRING GATE OVERALL: {'PASSED' if overall_gate_passed else 'FAILED'}\n{'=' * 80}")
        flush_log()

        if not overall_gate_passed:
            log("WIRING GATE FAILED. STOPPING per 18.3/18.6. No juice_* quote has been loaded.")
            metadata["stopped_at"] = "wiring_gate"
            metadata["input_checksums_partial"] = input_checksums
            metadata_path = output_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            log(f"Saved metadata (wiring-gate-failure record) to: {metadata_path}")
            flush_log()
            print(f"Saved run log to: {log_path}")
            return 1

        # generic-wrapper cross-check, using Origin B's already-computed gate inputs
        cross_check_ok = verify_generic_wrapper_matches_template(
            origin_specs["B"]["closing"],
            *probs_store["B"]["no_pace_control"]["closing"],
            *probs_store["B"]["control_plus_market_state"]["closing"],
            log,
        )
        if not cross_check_ok:
            raise AssertionError(
                "paired_brier_delta_generic does not exactly match emsf.paired_brier_delta_vs_variant on "
                "identical inputs -- refusing to use it for the PRIMARY/SECONDARY statistics."
            )
        metadata["generic_wrapper_cross_check_passed"] = True

        # =====================================================================
        # PHASE 2 (POST-GATE): load juice_* quotes, compute features, attach.
        # =====================================================================
        paired_all = build_juice_quote_universe(clean_full, log)
        juice_feat, paired_annotated = compute_juice_features(paired_all, log)
        df_full = attach_juice_features(df_full, juice_feat, log)

        microstructure_shots_cols = no_pace_cols + ALL_JUICE_COLS
        both_shots_cols = market_shots_cols + ALL_JUICE_COLS
        if len(microstructure_shots_cols) != 111:
            raise AssertionError(f"control_plus_microstructure expected 111 cols, got {len(microstructure_shots_cols)}.")
        if len(both_shots_cols) != 119:
            raise AssertionError(f"control_plus_market_state_plus_microstructure expected 119 cols, got {len(both_shots_cols)}.")
        log(f"\ncontrol_plus_microstructure: {len(microstructure_shots_cols)} cols. "
            f"control_plus_market_state_plus_microstructure: {len(both_shots_cols)} cols.")

        metadata["feature_sets"] = {
            "no_pace_control": no_pace_cols,
            "juice_feature_cols": JUICE_FEATURE_COLS,
            "juice_indicator_col": JUICE_INDICATOR_COL,
            "market_feature_cols": emsf.MARKET_FEATURE_COLS,
            "market_indicator_col": emsf.MARKET_INDICATOR_COL,
            "control_plus_microstructure_n": len(microstructure_shots_cols),
            "control_plus_market_state_n": len(market_shots_cols),
            "control_plus_market_state_plus_microstructure_n": len(both_shots_cols),
        }

        input_checksums["post_gate"] = {
            "saves_lines_snapshots": sha256_file(SNAPSHOTS_PATH),
            "core_bettime_202607_snapshots": sha256_file(CORE_BETTIME_PATH),
            "multibook_frame_2023_24_bettime": sha256_file(BETTIME_FRAME_A_PATH) if BETTIME_FRAME_A_PATH.exists() else None,
        }
        metadata["input_checksums"] = input_checksums
        flush_log()

        # =====================================================================
        # COVERAGE RECONCILIATION (18.5/18.8) -- hard-stop on mismatch.
        # =====================================================================
        log("\n" + "=" * 80)
        log("COVERAGE RECONCILIATION (18.5/18.8)")
        log("=" * 80)
        matched_nights = set(
            zip(paired_all["game_id"].tolist(), paired_all["goalie_id"].tolist())
        )

        def coverage_for_idx(idx):
            # NOTE: must index df_full (the frame the fold index arrays were
            # carved on), never the raw clean parquet, whose on-disk row order
            # is not guaranteed to match load_modeling_frame's date-sorted
            # order.
            sub = df_full.iloc[idx]
            keys = list(zip((int(g) for g in sub["game_id"]), (int(o) for o in sub["goalie_id"])))
            n_matched = sum(1 for k in keys if k in matched_nights)
            return n_matched, len(keys)

        coverage_observed = {
            "origin_b_train": coverage_for_idx(train_idx_b),
            "origin_b_val": coverage_for_idx(val_idx_b),
            "origin_c_train": coverage_for_idx(train_idx_c),
            "origin_c_val": coverage_for_idx(val_idx_c),
            "origin_a_test": coverage_for_idx(test_idx_a),
            "origin_b_test_nondfs": coverage_for_idx(test_idx_b),
            "origin_c_test": coverage_for_idx(test_idx_c),
        }
        coverage_check = {}
        for key, (n_obs, n_total_obs) in coverage_observed.items():
            n_ref, n_total_ref = COVERAGE_REFS[key]
            passed = n_obs == n_ref and n_total_obs == n_total_ref
            pct_obs = n_obs / n_total_obs * 100 if n_total_obs else 0.0
            coverage_check[key] = {
                "observed": {"n_matched": n_obs, "n_total": n_total_obs, "pct": pct_obs},
                "expected": {"n_matched": n_ref, "n_total": n_total_ref, "pct": n_ref / n_total_ref * 100},
                "passed": passed,
            }
            log(f"  {key}: observed {n_obs}/{n_total_obs} ({pct_obs:.2f}%)  expected {n_ref}/{n_total_ref} ({n_ref / n_total_ref * 100:.2f}%)  -> {'MATCH' if passed else 'MISMATCH'}")

        coverage_overall_passed = all(c["passed"] for c in coverage_check.values())
        coverage_check["overall_passed"] = coverage_overall_passed
        metadata["coverage_reconciliation"] = coverage_check
        log(f"\nCOVERAGE RECONCILIATION OVERALL: {'PASSED' if coverage_overall_passed else 'FAILED'}")

        if not coverage_overall_passed:
            log("COVERAGE RECONCILIATION FAILED. STOPPING per this task's item 5 (hard-stop on mismatch).")
            metadata["stopped_at"] = "coverage_reconciliation"
            metadata_path = output_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            log(f"Saved metadata (coverage-reconciliation-failure record) to: {metadata_path}")
            flush_log()
            juice_feat.to_parquet(output_dir / "juice_goalie_night_features.parquet", index=False)
            paired_annotated.to_parquet(output_dir / "juice_bettime_quote_universe.parquet", index=False)
            print(f"Saved run log to: {log_path}")
            return 1

        # 18.5 50%-coverage floor for the GATING origins' TEST folds.
        origin_b_test_pct = coverage_check["origin_b_test_nondfs"]["observed"]["pct"]
        origin_c_test_pct = coverage_check["origin_c_test"]["observed"]["pct"]
        origin_b_sufficient = origin_b_test_pct >= COVERAGE_FLOOR_PCT
        origin_c_sufficient = origin_c_test_pct >= COVERAGE_FLOOR_PCT
        log(f"\n18.5 coverage floor ({COVERAGE_FLOOR_PCT}%): Origin B test {origin_b_test_pct:.2f}% -> "
            f"{'SUFFICIENT' if origin_b_sufficient else 'COVERAGE-INSUFFICIENT'}; Origin C test {origin_c_test_pct:.2f}% -> "
            f"{'SUFFICIENT' if origin_c_sufficient else 'COVERAGE-INSUFFICIENT'}.")
        metadata["coverage_floor_check"] = {
            "origin_b_sufficient": origin_b_sufficient, "origin_c_sufficient": origin_c_sufficient,
            "floor_pct": COVERAGE_FLOOR_PCT,
        }
        flush_log()

        # =====================================================================
        # PHASE 2b: extend the two gate variants with the bettime pass
        # (reload-and-reprice, no retraining) now that the archives are open.
        # =====================================================================
        log("\n" + "=" * 80)
        log("PHASE 2b: extend wiring-gate variants with the BETTIME pass (reload, no retraining)")
        log("=" * 80)

        df_bet_a_bettime = build_betting_frame(BETTIME_FRAME_A_PATH, log)

        snapshots_all_for_bettime = pd.read_parquet(SNAPSHOTS_PATH)
        base_2025_26 = clean_full[clean_full["season"] == SEASON_2025_26].reset_index(drop=True)
        df_bet_c_bettime = ero.build_season_multibook_frame(base_2025_26, snapshots_all_for_bettime, "bettime", log)

        core_all_for_bettime = pd.read_parquet(CORE_BETTIME_PATH)
        new_norm_b = e11.normalize_new_snapshots(core_all_for_bettime)
        base_2024_25 = clean_full[clean_full["season"] == ero.SEASON_2024_25].reset_index(drop=True)
        df_bet_b_bettime = ero.build_season_multibook_frame(base_2024_25, new_norm_b, "bettime", log)

        origin_specs["A"]["bettime"] = df_bet_a_bettime
        origin_specs["B"]["bettime"] = df_bet_b_bettime
        origin_specs["C"]["bettime"] = df_bet_c_bettime

        # Log-only comparison vs Experiment 8's recorded Origin C bettime
        # frame (5,763 rows / 1,370 clusters, 18.8) -- not a registered gate.
        c_bt_nights = int(df_bet_c_bettime[["game_id", "goalie_id"]].drop_duplicates().shape[0])
        log(f"\nOrigin C bettime frame: {len(df_bet_c_bettime)} rows / {c_bt_nights} goalie-night clusters "
            f"(Experiment 8's recorded frame was 5,763 rows / 1,370 clusters; log-only comparison, no gate).")

        # Persist the grading quote universes (18.5's closing/bettime frames)
        # for independent verification.
        frame_stats = {}
        for origin_label in ("A", "B", "C"):
            frame_stats[origin_label] = {}
            for pass_name in ("closing", "bettime"):
                df_q = origin_specs[origin_label][pass_name]
                q_path = output_dir / f"origin_{origin_label.lower()}_{pass_name}_quote_universe.parquet"
                df_q.to_parquet(q_path, index=False)
                frame_stats[origin_label][pass_name] = {
                    "rows": int(len(df_q)),
                    "goalie_nights": int(df_q[["game_id", "goalie_id"]].drop_duplicates().shape[0]),
                    "books": {str(k): int(v) for k, v in df_q["book_key"].value_counts().items()} if "book_key" in df_q.columns else None,
                    "path": str(q_path),
                }
                log(f"Saved Origin {origin_label} {pass_name} quote universe ({len(df_q)} rows) to: {q_path}")
        metadata["grading_quote_universes"] = frame_stats

        for origin_label in ("A", "B", "C"):
            for variant_name in GATE_VARIANTS:
                shots_cols = no_pace_cols if variant_name == "no_pace_control" else market_shots_cols
                extra_row_df = extend_variant_with_bettime(
                    origin_label, variant_name, result_store[origin_label][variant_name], probs_store[origin_label][variant_name],
                    rate_models[origin_label]["model"], no_pace_cols, shots_cols, df_full, origin_specs[origin_label]["test_idx"],
                    origin_specs[origin_label]["closing"], origin_specs[origin_label]["bettime"], log,
                )
                row_frames[origin_label].append(extra_row_df)
                flush_log()

        # =====================================================================
        # PHASE 3: train the two microstructure shots variants (both passes
        # available now). The other 6 of the 12 registered shots trainings.
        # =====================================================================
        log("\n" + "=" * 80)
        log("PHASE 3: train control_plus_microstructure and control_plus_market_state_plus_microstructure")
        log("=" * 80)

        variant_shots_cols = {
            "no_pace_control": no_pace_cols,
            "control_plus_microstructure": microstructure_shots_cols,
            "control_plus_market_state": market_shots_cols,
            "control_plus_market_state_plus_microstructure": both_shots_cols,
        }

        for origin_label in ("A", "B", "C"):
            price_frames = {"closing": origin_specs[origin_label]["closing"], "bettime": origin_specs[origin_label]["bettime"]}
            for variant_name in MICROSTRUCTURE_VARIANTS:
                shots_cols = variant_shots_cols[variant_name]
                result_json, probs_by_pass, row_predictions, mu_test = emsf.run_shots_variant(
                    origin_label, variant_name, df_full, origin_specs[origin_label]["train_idx"], origin_specs[origin_label]["val_idx"],
                    origin_specs[origin_label]["test_idx"], shots_cols, no_pace_cols, rate_models[origin_label]["model"],
                    price_frames, output_dir, log,
                )
                result_store[origin_label][variant_name] = result_json
                probs_store[origin_label][variant_name] = probs_by_pass
                mu_store[origin_label][variant_name] = mu_test
                row_frames[origin_label].append(row_predictions)
                flush_log()

        # Save per-origin combined predictions (all 4 variants, both passes).
        predictions_paths = {}
        for origin_label in ("A", "B", "C"):
            preds = pd.concat(row_frames[origin_label], ignore_index=True)
            path = output_dir / f"origin_{origin_label.lower()}_test_predictions.parquet"
            preds.to_parquet(path, index=False)
            predictions_paths[origin_label] = str(path)
            log(f"Saved {len(preds)} per-row test predictions (4 variants x passes) for Origin {origin_label} to: {path}")

        juice_feat.to_parquet(output_dir / "juice_goalie_night_features.parquet", index=False)
        paired_annotated.to_parquet(output_dir / "juice_bettime_quote_universe.parquet", index=False)
        log(f"Saved juice goalie-night feature table and bettime quote universe artifacts.")
        flush_log()

        # =====================================================================
        # PHASE 4: PRIMARY and SECONDARY statistics (18.5), all 3 origins,
        # closing PRIMARY + bettime unconditional secondary.
        # =====================================================================
        log("\n" + "=" * 80)
        log("PHASE 4: PRIMARY (microstructure - no_pace) and SECONDARY (both - market_state) statistics")
        log("=" * 80)

        bootstrap_input_rows = []

        def run_comparison(origin_label, base_variant, new_variant, comparison_name):
            entry = {}
            for pass_name in ("closing", "bettime"):
                df_bet = origin_specs[origin_label][pass_name]
                base_p_over, base_matched = probs_store[origin_label][base_variant][pass_name]
                new_p_over, new_matched = probs_store[origin_label][new_variant][pass_name]
                label = f"origin_{origin_label} {comparison_name} {pass_name} ({new_variant} - {base_variant})"
                stat, delta, cluster_ids = paired_brier_delta_generic(df_bet, base_p_over, base_matched, new_p_over, new_matched, log, label)
                entry[f"brier_delta_{pass_name}"] = stat
                q_game = df_bet["game_id"].values
                q_goalie = df_bet["goalie_id"].values
                q_book = df_bet["book_key"].values if "book_key" in df_bet.columns else np.array([None] * len(df_bet), dtype=object)
                q_line = df_bet["betting_line"].values.astype(float)
                for i in range(len(delta)):
                    if not np.isnan(delta[i]):
                        bootstrap_input_rows.append({
                            "origin": origin_label, "comparison": comparison_name, "metric": "brier",
                            "pass": pass_name, "cluster_id": cluster_ids[i], "delta": float(delta[i]),
                            "game_id": int(q_game[i]), "goalie_id": int(q_goalie[i]),
                            "book": q_book[i], "betting_line": float(q_line[i]),
                        })
            mae_label = f"origin_{origin_label} {comparison_name} shots MAE ({new_variant} - {base_variant})"
            mae_stat, mae_delta, mae_cluster_ids = paired_shots_mae_delta_generic(
                df_full, origin_specs[origin_label]["test_idx"], mu_store[origin_label][base_variant],
                mu_store[origin_label][new_variant], log, mae_label,
            )
            entry["shots_mae_delta"] = mae_stat
            t_idx = origin_specs[origin_label]["test_idx"]
            t_game = df_full["game_id"].values[t_idx]
            t_goalie = df_full["goalie_id"].values[t_idx]
            for i in range(len(mae_delta)):
                bootstrap_input_rows.append({
                    "origin": origin_label, "comparison": comparison_name, "metric": "shots_mae",
                    "pass": "test_fold", "cluster_id": mae_cluster_ids[i], "delta": float(mae_delta[i]),
                    "game_id": int(t_game[i]), "goalie_id": int(t_goalie[i]),
                    "book": None, "betting_line": None,
                })
            return entry

        primary_stats = {o: run_comparison(o, "no_pace_control", "control_plus_microstructure", "PRIMARY") for o in ("A", "B", "C")}
        secondary_stats = {o: run_comparison(o, "control_plus_market_state", "control_plus_market_state_plus_microstructure", "SECONDARY") for o in ("A", "B", "C")}

        bootstrap_inputs_df = pd.DataFrame(bootstrap_input_rows)
        bootstrap_inputs_path = output_dir / "bootstrap_cluster_inputs.parquet"
        bootstrap_inputs_df.to_parquet(bootstrap_inputs_path, index=False)
        log(f"\nSaved {len(bootstrap_inputs_df)} bootstrap cluster-input rows to: {bootstrap_inputs_path}")

        metadata["primary_control_plus_microstructure_minus_no_pace_control"] = primary_stats
        metadata["secondary_both_minus_control_plus_market_state"] = secondary_stats
        flush_log()

        # =====================================================================
        # PLACEBO READOUT (18.3a) -- mandatory STOP-AND-INVESTIGATE check.
        # =====================================================================
        log("\n" + "=" * 80)
        log("PLACEBO READOUT (18.3a) -- Origin A")
        log("=" * 80)

        def excludes_zero(stat) -> bool:
            return stat["lower"] is not None and stat["upper"] is not None and (stat["lower"] > 0 or stat["upper"] < 0)

        placebo_flags = {
            "primary_brier_closing": excludes_zero(primary_stats["A"]["brier_delta_closing"]),
            "primary_shots_mae": excludes_zero(primary_stats["A"]["shots_mae_delta"]),
            "secondary_brier_closing": excludes_zero(secondary_stats["A"]["brier_delta_closing"]),
            "secondary_shots_mae": excludes_zero(secondary_stats["A"]["shots_mae_delta"]),
        }
        placebo_anomaly = any(placebo_flags.values())
        log(f"Origin A placebo CI95-excludes-zero flags: {placebo_flags}")
        log(f"PLACEBO ANOMALY: {placebo_anomaly}")
        metadata["placebo_readout"] = {
            "origin_a_primary": primary_stats["A"], "origin_a_secondary": secondary_stats["A"],
            "excludes_zero_flags": placebo_flags, "anomaly": placebo_anomaly,
        }
        flush_log()

        # =====================================================================
        # VERDICT (18.5/18.7).
        # =====================================================================
        log("\n" + "=" * 80)
        log("VERDICT (18.5/18.7)")
        log("=" * 80)

        def improvement(stat) -> bool:
            # 18.5's PASS direction: CI95 upper bound strictly below zero
            # (delta = new - base; negative = new variant better). One-sided
            # by registration -- a CI entirely ABOVE zero is a significant
            # DEGRADATION, not a clearance. excludes_zero (two-sided) is used
            # only for the 18.3a placebo-anomaly rule, which flags either
            # direction.
            return stat["upper"] is not None and stat["upper"] < 0

        primary_pass_per_metric = {}
        for metric_key in ("brier_delta_closing", "shots_mae_delta"):
            b_clears = improvement(primary_stats["B"][metric_key]) if origin_b_sufficient else None
            c_clears = improvement(primary_stats["C"][metric_key]) if origin_c_sufficient else None
            passes_both = bool(origin_b_sufficient and origin_c_sufficient and b_clears and c_clears)
            primary_pass_per_metric[metric_key] = {
                "origin_b_clears": b_clears, "origin_c_clears": c_clears, "passes_both_gating_origins": passes_both,
            }
            log(f"  PRIMARY {metric_key}: origin_B clears={b_clears}  origin_C clears={c_clears}  passes_both={passes_both}")

        primary_overall_pass = any(m["passes_both_gating_origins"] for m in primary_pass_per_metric.values())
        origin_b_clears_any = origin_b_sufficient and any(
            improvement(primary_stats["B"][k]) for k in ("brier_delta_closing", "shots_mae_delta")
        )
        origin_c_clears_any = origin_c_sufficient and any(
            improvement(primary_stats["C"][k]) for k in ("brier_delta_closing", "shots_mae_delta")
        )

        secondary_pass_per_metric = {}
        for metric_key in ("brier_delta_closing", "shots_mae_delta"):
            b_clears = improvement(secondary_stats["B"][metric_key]) if origin_b_sufficient else None
            c_clears = improvement(secondary_stats["C"][metric_key]) if origin_c_sufficient else None
            passes_both = bool(origin_b_sufficient and origin_c_sufficient and b_clears and c_clears)
            secondary_pass_per_metric[metric_key] = {
                "origin_b_clears": b_clears, "origin_c_clears": c_clears, "passes_both_gating_origins": passes_both,
            }
        secondary_overall_additive = any(m["passes_both_gating_origins"] for m in secondary_pass_per_metric.values())

        if placebo_anomaly:
            verdict = "PLACEBO ANOMALY -- STOP: no PASS/FAIL verdict issued; investigation required before any Origin B/C verdict language is used (18.3a/18.7)."
        elif not origin_b_sufficient and not origin_c_sufficient:
            verdict = "INSUFFICIENT SAMPLE: both gating origins are below the 50% test-fold coverage floor (18.5/18.7)."
        elif not origin_b_sufficient or not origin_c_sufficient:
            verdict = "INSUFFICIENT SAMPLE: one gating origin is below the 50% test-fold coverage floor; a single surviving origin cannot satisfy the two-origin-agreement bar (18.7)."
        elif primary_overall_pass:
            verdict = (
                "PRIMARY PASS: control_plus_microstructure beats no_pace_control with CI95 entirely below zero on "
                "at least one metric on BOTH Origin B and Origin C. Promoted to candidate status for the 2026-27 "
                "model rebuild, queued for a separate future betting-policy registration (NOT promoted to live "
                "betting by this registration alone). SECONDARY comparison read below to decide REDUNDANT vs ADDITIVE."
            )
        elif origin_b_clears_any != origin_c_clears_any:
            verdict = "ONE-OF-TWO: clears the bar on exactly one of Origin B / Origin C. Recorded as NOT REPLICATED, CLOSED (18.7) -- a single-origin result cannot promote the block."
        else:
            verdict = "PRIMARY FAIL on both gating origins: the juice-skew feature lead is CLOSED this cycle (18.7); it does not reopen without a new architecture or a genuinely new season of bettime coverage."

        if placebo_anomaly:
            secondary_readout = "Not read (placebo anomaly stands; no verdict language used)."
        elif primary_overall_pass:
            secondary_readout = "ADDITIVE (a further CI-excluding-zero gain survives on top of market-state, on at least one metric on both gating origins)." if secondary_overall_additive else "REDUNDANT (no further CI-excluding-zero gain over control_plus_market_state on either metric on both gating origins)."
        else:
            secondary_readout = "Not gating (PRIMARY did not pass); reported for completeness only."

        log(f"\nFINAL VERDICT: {verdict}")
        log(f"SECONDARY READOUT: {secondary_readout}")

        metadata["pre_registered_pass_bar"] = {
            "primary": primary_pass_per_metric,
            # Under an open 18.3a placebo anomaly the computed per-metric CI
            # facts stand as artifacts but no overall verdict boolean is
            # recorded (18.7: "carry no verdict language while the anomaly is
            # open").
            "primary_overall_pass": None if placebo_anomaly else primary_overall_pass,
            "secondary": secondary_pass_per_metric,
            "secondary_overall_additive": None if placebo_anomaly else secondary_overall_additive,
        }
        metadata["final_verdict"] = verdict
        metadata["secondary_readout"] = secondary_readout
        metadata["market_source_stats"] = market_stats
        metadata["origin_results"] = {
            origin_label: {
                "variants": result_store[origin_label],
                "rate_model": {
                    "winner": rate_models[origin_label]["winner"],
                    "val_evaluations": rate_models[origin_label]["evals"],
                    "model_path": str(rate_models[origin_label]["path"]),
                    "feature_cols_n": len(no_pace_cols),
                    "shared_across_all_four_variants": True,
                },
            }
            for origin_label in ("A", "B", "C")
        }
        metadata["fixed_ev_threshold"] = FIXED_EV_THRESHOLD
        metadata["origin_cap"] = ORIGIN_CAP
        metadata["failed_prior_run_directories"] = []
        metadata["predictions_paths"] = predictions_paths
        metadata["bootstrap_cluster_inputs_path"] = str(bootstrap_inputs_path)
        metadata["juice_goalie_night_features_path"] = str(output_dir / "juice_goalie_night_features.parquet")
        metadata["juice_bettime_quote_universe_path"] = str(output_dir / "juice_bettime_quote_universe.parquet")

        metadata["judgment_calls"] = [
            "Deferred loading of ALL bettime-pass archives (saves_lines_snapshots.parquet's bettime rows, "
            "core_bettime_202607_snapshots.parquet, and the pre-built multibook_frame_2023_24_bettime.parquet "
            "cache) until after the wiring gate passed -- a stricter reading than the registered gate targets "
            "strictly require (all of which are closing-pass/workload/alpha values), chosen to remove "
            "interpretive risk on '18.3: do not load any juice_* quote' at zero training cost. Achieved via a "
            "reload-and-reprice pattern (mirrors experiment_market_state_origin_c.py's own train-fitted-"
            "dispersion reload) so the two gate variants are trained exactly once each, keeping the total at "
            "the expected 12 shots-model trainings (4 variants x 3 origins), never 18.",
            "juice_n_books computed via nunique() on 'book' rather than row count, defensively equivalent by "
            "construction (verified empirically: zero goalie-nights have more than one paired row from the "
            "same book, since the 17.2 dedup key excludes 'line', so a book contributes at most one Over/Under "
            "pair per goalie-night regardless of any in-pass line move).",
            "17.2's max-resolved_ts dedup natural key uses (event_id, player_name_raw, book_key, side) for "
            "core_bettime_202607_snapshots.parquet (that parquet's own column names for the same fields), vs "
            "(event_id, goalie_name_raw, book, side) for saves_lines_snapshots.parquet, per 18.8's own usage "
            "of each parquet's column names.",
            "The dispersion features' '0.0 for a single qualifying book/line' convention (18.2 features 3 and "
            "5) is not special-cased in code -- it falls out of ddof=0 population std's own math for an n=1 "
            "sample, verified to land on exactly 0.0 for every single-book/single-line goalie-night in this "
            "archive before trusting it.",
            "Coverage reconciliation (18.5/18.8) uses no goalie_id name-based fallback (matching 18.8's own "
            "inventory-check convention exactly, not 14.5 rule 1's fuller fallback), verified pre-execution to "
            "reproduce all four registered coverage percentages exactly bit-for-bit including the 17.2 dedup.",
            "The new PRIMARY/SECONDARY statistics (Phase 4) use a small local paired-delta wrapper "
            "(paired_brier_delta_generic / paired_shots_mae_delta_generic) rather than emsf.paired_brier_"
            "delta_vs_variant / paired_shots_mae_delta directly, to avoid those functions' hardcoded "
            "'control_plus_market_state - no_pace_control' log text (which would misdescribe this script's "
            "other variant pairs) and to expose raw per-row delta arrays for the bootstrap-cluster-inputs "
            "artifact. The wrapper's construction is byte-identical to the template's own (same masking, same "
            "clv.cluster_bootstrap_mean_ci call with the same n_resamples/seed) and is cross-checked at "
            "runtime against the template's own function on Origin B's wiring-gate inputs (exact match on "
            "mean/lower/upper/n_bets/n_clusters) before being trusted for anything else. The wiring gate "
            "itself calls the template's own functions verbatim, never the generic wrapper.",
            "ONE-OF-TWO verdict logic (18.7) is computed as 'clears the CI-upper-bound-below-zero bar on at "
            "least one metric' per origin (origin_b_clears_any XOR origin_c_clears_any), since 18.7's own text "
            "('clears the bar on exactly one of Origin B / Origin C, on whichever metric') does not pin down "
            "whether the same metric must be the one that clears on the clearing origin; this reading treats "
            "origin-level clearance as 'passes on some metric'.",
            "Origin C's BETTIME grading universe is built by passing the raw saves_lines_snapshots.parquet "
            "into experiment_rolling_origin.build_season_multibook_frame exactly as Experiment 8's own script "
            "did (that function's internal clean_bettime_pass applies its own earliest-requested_ts-per-event "
            "cleaning), NOT by pre-applying the 17.2 max-resolved_ts dedup. 18.4's wording ('built from the "
            "deduped 2025-26 snapshot rows via the same build_season_multibook_frame/pivot_both_sides path as "
            "the other origins') is ambiguous between the two, but its own parenthetical anchors the "
            "construction to Experiment 8's recorded frame (5,763 rows / 1,370 clusters), and 'the same path "
            "as the other origins' matches Experiment 8/11's raw-input calls -- so the frozen precedent's "
            "literal construction wins. The run log records the resulting frame size next to Experiment 8's "
            "recorded figures for the reader to verify the interpretation empirically. The 17.2 dedup governs "
            "the FEATURE construction (18.2) exactly as registered; this call affects only the SECONDARY, "
            "non-gating bettime grading universe.",
        ]

        elapsed = time.time() - start_time
        metadata["wall_clock_seconds"] = elapsed
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        log(f"\nSaved metadata to: {metadata_path}")

        input_checksums_path = output_dir / "input_checksums.json"
        input_checksums_path.write_text(json.dumps(input_checksums, indent=2), encoding="utf-8")
        log(f"Saved input checksums to: {input_checksums_path}")

        log(f"Wall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        log("\n" + "=" * 80)
        log("EXPERIMENT 15 COMPLETE")
        log("=" * 80)
        flush_log()
    except Exception:
        flush_log()
        raise

    print(f"Saved run log to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
