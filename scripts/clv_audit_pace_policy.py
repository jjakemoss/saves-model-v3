"""
CLV audit of the frozen pace_shots policy.

Contract: docs/OFFSEASON_OPTIMIZATION_PLAN.md section 3.15, "CLV audit of the
frozen pace_shots policy" bullet (interpretation rules pre-registered there,
before this data existed), read together with section 3.14's Component 3
verification block for the statistical standard (goalie-night cluster
bootstrap).

This script re-prices the already-selected, already-frozen pace_shots policy
(models/trained/experiment_pace_distributional_20260709_100802/) against the
bet-time pass of data/processed/saves_lines_snapshots.parquet, and measures
how those selected bets moved against the closing line. Nothing is retrained,
reselected, or tuned. The EV threshold (0.05) and the selection code
(experiments.harness.decide_bet) are reused unchanged from the frozen
experiment.

Two-stage design:

  Stage 0 (integrity gate): reload the frozen pace_shots shots/save-rate
  models, rebuild the distributional predictions on the clean_training_data
  test fold (game_date >= 2025-12-04), and reproduce the original
  experiment's own test-fold betting evaluation on
  multibook_classification_training_data.parquet. This must reproduce the
  recorded numbers exactly (616 bets, +9.02% ROI, Brier 0.24904, 508 OVER /
  108 UNDER) before anything below is interpreted. If it does not reproduce,
  the script stops.

  Stage 1 (the audit): using the SAME dist_preds_test object (identical mu/q/
  pmf per test-fold goalie-night) built in Stage 0, price every bet-time row
  in saves_lines_snapshots.parquet for the test-fold window
  (2025-12-04..2026-04-16), select bets with the frozen decide_bet() logic,
  and compute probability CLV and price CLV for each selected bet against
  the closing pass. Report goalie-night cluster bootstrap 95% CIs.

A note on "de-vigged" bet selection: the audit's plain-English design says
"edge = model prob minus de-vigged market prob > 0.05". The actual frozen
selection code (experiments.harness.decide_bet -> calculate_ev) does NOT
de-vig for selection -- it compares the model probability to the RAW
(vig-inclusive) single-side implied probability of that one book's price,
per side, and picks OVER if ev_over >= threshold and ev_over > ev_under,
else UNDER if ev_under >= threshold. This script replicates the literal code,
not the paraphrase, per the task instruction to read the code and replicate
it exactly. De-vigging IS used, exactly as specified, for the CLV computation
itself (step 2 of the audit design) -- that is a separate, later computation
on top of the already-selected bets.

Usage:
    python scripts/clv_audit_pace_policy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import experiment_pace_distributional as epd  # noqa: E402
from experiments.distributional_saves import (  # noqa: E402
    CAP,
    SavesDistribution,
    compute_distribution_predictions,
    fit_dispersion,
    join_and_price,
)
from experiments.harness import decide_bet, split_by_date  # noqa: E402
from betting.odds_utils import decimal_to_american  # noqa: E402


ARTIFACT_DIR = REPO_ROOT / "models" / "trained" / "experiment_pace_distributional_20260709_100802"
SNAPSHOTS_PATH = REPO_ROOT / "data" / "processed" / "saves_lines_snapshots.parquet"
OUTPUT_BETS_PATH = REPO_ROOT / "data" / "processed" / "clv_audit_bets.parquet"

TEST_WINDOW_START = "2025-12-04"
TEST_WINDOW_END = "2026-04-16"

CONTROL_GATE_EXPECTED = {
    "bets": 616,
    "roi_rounded": 9.02,
    "brier_rounded": 0.24904,
    "over_bets": 508,
    "under_bets": 108,
}

N_BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
CI_PCT = 95.0


def log(msg: str = "") -> None:
    print(msg)


# ---------------------------------------------------------------------------
# Stage 0: reload the frozen pace_shots models and reproduce the original
# experiment's own test-fold result (integrity gate).
# ---------------------------------------------------------------------------


def load_frozen_pace_shots():
    """Rebuild the pace_shots modeling frame, reload the frozen shots/
    save-rate models, recompute dispersion alpha, and build test-fold
    distributional predictions (mu, q, pmf per test-fold goalie-night).
    Returns (frame, clean_split, dist, dist_preds_test, shots_cols, rate_cols,
    alpha, metadata_result)."""
    metadata_path = ARTIFACT_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing frozen experiment metadata: {metadata_path}")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    result = metadata["results"]["pace_shots"]

    log("=" * 80)
    log("STAGE 0: reload frozen pace_shots policy and rebuild test-fold predictions")
    log("=" * 80)

    frame = epd.load_pace_modeling_frame(
        epd.DATA_PATH_CLEAN, epd.DATA_PATH_CONTEXT, epd.DATA_PATH_PACE, epd.DATA_PATH_PACE_METADATA, log,
    )
    clean_split = split_by_date(frame.df, log, "clean_training_data")

    variant = next(v for v in epd.VARIANTS if v.name == "pace_shots")
    shots_cols, rate_cols = epd.feature_cols_for_variant(frame, variant)

    if shots_cols != result["shots_feature_cols"]:
        raise AssertionError(
            "Reconstructed pace_shots shots feature list does not match the frozen "
            "artifact's recorded feature list. Cannot proceed."
        )
    if rate_cols != result["rate_feature_cols"]:
        raise AssertionError(
            "Reconstructed pace_shots save-rate feature list does not match the frozen "
            "artifact's recorded feature list. Cannot proceed."
        )
    log(f"Feature identity check passed: {len(shots_cols)} shots cols, {len(rate_cols)} rate cols match artifact.")

    shots_model = xgb.XGBRegressor()
    shots_model.load_model(str(ARTIFACT_DIR / "pace_shots_shots_model.json"))
    rate_model = xgb.XGBRegressor()
    rate_model.load_model(str(ARTIFACT_DIR / "pace_shots_save_rate_model.json"))
    log("Loaded frozen pace_shots shots_model and save_rate_model from artifact JSON (no retraining).")

    alpha_recomputed, method, diag = fit_dispersion(
        shots_model, frame.df, clean_split.train_idx, shots_cols, log, "pace_shots_reload",
    )
    alpha_frozen = float(result["dispersion"]["alpha"])
    alpha_diff = abs(alpha_recomputed - alpha_frozen)
    log(f"Dispersion alpha: frozen artifact={alpha_frozen:.10f}, recomputed={alpha_recomputed:.10f}, diff={alpha_diff:.2e}")
    if alpha_diff > 1e-6:
        raise AssertionError(
            f"Recomputed dispersion alpha ({alpha_recomputed}) does not match the frozen "
            f"artifact's recorded alpha ({alpha_frozen}). The reload does not reproduce the "
            "frozen policy; stopping rather than improvising."
        )
    alpha = alpha_frozen

    dist = SavesDistribution(CAP)
    dist_preds_test = compute_distribution_predictions(
        frame.df, clean_split.test_idx, shots_model, rate_model, alpha, shots_cols, rate_cols, dist, log,
        "pace_shots TEST (reload)",
    )

    return frame, clean_split, dist, dist_preds_test, shots_cols, rate_cols, alpha, result


def run_integrity_gate(dist, dist_preds_test) -> dict:
    """Reproduce the original experiment's own test-fold betting evaluation
    on multibook_classification_training_data.parquet using the reloaded
    frozen models. Hard-stops if it does not reproduce the recorded numbers."""
    log("\n--- Integrity gate: reproduce original pace_shots TEST result on multibook data ---")
    df_bet = epd.build_betting_frame(epd.DATA_PATH_MULTIBOOK, log)
    bet_split = split_by_date(df_bet, log, "multibook_classification_training_data")
    df_bet_test = df_bet.iloc[bet_split.test_idx].reset_index(drop=True)

    p_over_test, p_under_test, p_push_test, matched_test, cov_test = join_and_price(
        df_bet_test, dist_preds_test, dist, log, "pace_shots TEST (reload) betting frame",
    )
    test_bundle, test_auc, test_brier = epd.evaluate_test_once(
        df_bet_test, p_over_test, p_under_test, matched_test, 0.05, log, "pace_shots TEST (reload)",
    )

    summary = test_bundle["summary"]
    side = test_bundle["side_breakdown"]
    observed = {
        "bets": int(summary["bets"]),
        "roi_rounded": round(float(summary["roi"]), 2),
        "brier_rounded": round(float(test_brier), 5),
        "over_bets": int(side["OVER"]["bets"]),
        "under_bets": int(side["UNDER"]["bets"]),
    }
    log(f"Observed: {observed}")
    log(f"Expected: {CONTROL_GATE_EXPECTED}")

    if observed != CONTROL_GATE_EXPECTED:
        raise AssertionError(
            "INTEGRITY GATE FAILED: reloaded frozen pace_shots artifacts do not reproduce "
            f"the experiment's own recorded test-fold result. Expected {CONTROL_GATE_EXPECTED}, "
            f"observed {observed}. Stopping per task instructions rather than improvising."
        )
    log("INTEGRITY GATE PASSED: reloaded artifacts exactly reproduce the frozen experiment's "
        "recorded test-fold result (616 bets, +9.02% ROI, Brier 0.24904, 508 OVER / 108 UNDER).")

    return {
        "df_bet_test": df_bet_test,
        "p_over_test": p_over_test,
        "p_under_test": p_under_test,
        "matched_test": matched_test,
        "observed": observed,
    }


# ---------------------------------------------------------------------------
# Stage 1: build the bet universe from saves_lines_snapshots.parquet.
# ---------------------------------------------------------------------------


def load_snapshot_window() -> pd.DataFrame:
    df = pd.read_parquet(SNAPSHOTS_PATH)
    window = df[
        (df["game_date_eastern"] >= TEST_WINDOW_START) & (df["game_date_eastern"] <= TEST_WINDOW_END)
    ].copy()
    log(f"\nSnapshots parquet: {len(df)} total rows; test-fold window "
        f"{TEST_WINDOW_START}..{TEST_WINDOW_END}: {len(window)} rows.")
    pass_counts = window["snapshot_pass"].value_counts().to_dict()
    log(f"  snapshot_pass breakdown: {pass_counts}")
    return window


def clean_bettime_pass(window: pd.DataFrame) -> pd.DataFrame:
    """Bet-time pass, cleaned to one snapshot per event (the earliest
    requested_ts, matching the pre-registered 22:30Z-or-commence-minus-30min
    anchor) and de-duplicated of exact-duplicate rows.

    A minority of events (68 in this window) carry two distinct bet-time
    requested_ts values -- almost certainly from a schedule-time correction
    between fetch runs producing two different computed anchors for the same
    event, since compute_bettime_ts(commence_time) is deterministic and the
    pre-registered anchor rule always resolves to the EARLIER candidate
    (min(22:30Z, commence-30min)). Taking the earliest requested_ts per event
    recovers the intended anchor and discards the later (closer-to-commence,
    sometimes only ~7 minutes before puck drop) duplicate, which would
    understate true bet-time-to-close movement if kept."""
    bt = window[window["snapshot_pass"] == "bettime"].copy()
    n_null_goalie = int(bt["goalie_id"].isna().sum())
    bt = bt[bt["goalie_id"].notna()].copy()
    bt["goalie_id"] = bt["goalie_id"].astype(int)
    log(f"\nBet-time pass: {len(bt) + n_null_goalie} rows, {n_null_goalie} dropped for unmatched goalie_id.")

    min_ts = bt.groupby("event_id")["requested_ts"].transform("min")
    n_dup_events = bt.loc[bt["requested_ts"] != min_ts, "event_id"].nunique()
    n_dup_rows = int((bt["requested_ts"] != min_ts).sum())
    bt = bt[bt["requested_ts"] == min_ts].copy()
    log(f"  Dropped {n_dup_rows} rows from {n_dup_events} events with a later duplicate bet-time "
        "snapshot (kept the earliest requested_ts per event).")

    before = len(bt)
    bt = bt.drop_duplicates(
        subset=["event_id", "requested_ts", "book", "goalie_id", "side", "line", "price_decimal"]
    )
    log(f"  Dropped {before - len(bt)} exact-duplicate rows (same event/book/goalie/side/line/price).")

    return bt


def clean_closing_pass(window: pd.DataFrame) -> pd.DataFrame:
    cl = window[window["snapshot_pass"] == "closing"].copy()
    n_null_goalie = int(cl["goalie_id"].isna().sum())
    cl = cl[cl["goalie_id"].notna()].copy()
    cl["goalie_id"] = cl["goalie_id"].astype(int)
    log(f"\nClosing pass: {len(cl) + n_null_goalie} rows, {n_null_goalie} dropped for unmatched goalie_id.")

    before = len(cl)
    cl = cl.drop_duplicates(
        subset=["event_id", "requested_ts", "book", "goalie_id", "side", "line", "price_decimal"]
    )
    log(f"  Dropped {before - len(cl)} exact-duplicate rows.")
    return cl


def attach_game_id(df: pd.DataFrame, game_lookup: dict) -> tuple[pd.DataFrame, int]:
    """Attach game_id via (goalie_id, game_date_eastern), with a +/-1 day
    fallback for UTC/Eastern date-boundary edge cases (empirically unused on
    this window -- exact match rate was 100% -- but kept as a documented
    safety net rather than assuming that always holds)."""
    df = df.copy()

    def _lookup_game_id(goalie_id, date_str) -> object:
        for offset in (0, -1, 1):
            d = (pd.Timestamp(date_str) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            key = (int(goalie_id), d)
            if key in game_lookup:
                return game_lookup[key]
        return None

    df["game_id"] = [
        _lookup_game_id(g, d) for g, d in zip(df["goalie_id"], df["game_date_eastern"])
    ]
    n_unmatched = int(df["game_id"].isna().sum())
    df = df[df["game_id"].notna()].copy()
    df["game_id"] = df["game_id"].astype(int)
    return df, n_unmatched


def pivot_both_sides(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Pivot side (Over/Under) into price_decimal_over/price_decimal_under
    columns, keeping only groups where both sides are present. Handles the
    documented rare same-key conflicting-price rows (isolated to bovada
    elsewhere in this archive) by taking the first value and logging a
    count."""
    key_with_side = group_cols + ["side"]
    conflict_check = df.groupby(key_with_side)["price_decimal"].nunique()
    n_conflicts = int((conflict_check > 1).sum())
    if n_conflicts:
        log(f"  WARNING: {n_conflicts} (key, side) groups have conflicting prices; taking first.")

    wide = df.pivot_table(index=group_cols, columns="side", values="price_decimal", aggfunc="first")
    wide = wide.reset_index()
    for col in ("Over", "Under"):
        if col not in wide.columns:
            wide[col] = np.nan
    n_total = len(wide)
    wide = wide.dropna(subset=["Over", "Under"]).copy()
    log(f"  Both-sides pivot: {len(wide)}/{n_total} (book, goalie, line) groups have both Over and Under.")
    wide = wide.rename(columns={"Over": "price_decimal_over", "Under": "price_decimal_under"})
    return wide


def build_bet_candidates(bt: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "event_id", "game_id", "goalie_id", "goalie_name_matched", "book", "line",
        "commence_time", "game_date_eastern",
    ]
    candidates = pivot_both_sides(bt, group_cols)
    return candidates


def attach_model_probabilities(candidates: pd.DataFrame, dist, dist_preds_test) -> pd.DataFrame:
    price_input = candidates[["game_id", "goalie_id", "line"]].rename(columns={"line": "betting_line"})
    p_over, p_under, p_push, matched, coverage = join_and_price(
        price_input, dist_preds_test, dist, log, "CLV audit bet-time candidates",
    )
    candidates = candidates.copy()
    candidates["model_p_over"] = p_over
    candidates["model_p_under"] = p_under
    candidates["model_p_push"] = p_push
    candidates["matched_to_pmf"] = matched
    n_unmatched = int((~matched).sum())
    if n_unmatched:
        log(f"  WARNING: {n_unmatched} candidates did not match a test-fold goalie-night pmf; dropping.")
    candidates = candidates[candidates["matched_to_pmf"]].reset_index(drop=True)
    return candidates


# ---------------------------------------------------------------------------
# Bet selection: literal reuse of experiments.harness.decide_bet.
# ---------------------------------------------------------------------------


def select_bets(candidates: pd.DataFrame, ev_threshold: float) -> pd.DataFrame:
    rows = []
    n_decimal_only_flip = 0
    for r in candidates.itertuples(index=False):
        odds_over_american = decimal_to_american(r.price_decimal_over)
        odds_under_american = decimal_to_american(r.price_decimal_under)
        bet_side, ev = decide_bet(
            r.model_p_over, r.model_p_under, odds_over_american, odds_under_american, ev_threshold,
        )

        # Diagnostic-only cross-check: the same selection using the exact
        # decimal-implied probability (1/price) instead of the American-
        # rounded round trip. Never used for the official selection -- see
        # module docstring on why American-rounded is the primary path.
        ev_over_decimal = r.model_p_over - (1.0 / r.price_decimal_over)
        ev_under_decimal = r.model_p_under - (1.0 / r.price_decimal_under)
        if ev_over_decimal >= ev_threshold and ev_over_decimal > ev_under_decimal:
            bet_side_decimal = "OVER"
        elif ev_under_decimal >= ev_threshold:
            bet_side_decimal = "UNDER"
        else:
            bet_side_decimal = None
        if bet_side_decimal != bet_side:
            n_decimal_only_flip += 1

        if bet_side is None:
            continue

        rows.append({
            "event_id": r.event_id,
            "game_id": r.game_id,
            "goalie_id": r.goalie_id,
            "goalie_name_matched": r.goalie_name_matched,
            "book": r.book,
            "line": r.line,
            "commence_time": r.commence_time,
            "game_date_eastern": r.game_date_eastern,
            "month": str(r.game_date_eastern)[:7],
            "price_decimal_over": r.price_decimal_over,
            "price_decimal_under": r.price_decimal_under,
            "model_p_over": r.model_p_over,
            "model_p_under": r.model_p_under,
            "bet_side": bet_side,
            "ev": ev,
        })

    log(f"\nSelection (frozen decide_bet, EV threshold={ev_threshold}): "
        f"{len(rows)} bets selected out of {len(candidates)} candidates.")
    log(f"  Diagnostic: {n_decimal_only_flip} candidates would flip bet/no-bet or side "
        "if selection used the exact decimal-implied probability instead of American-rounded "
        "odds (American-rounded is the primary/official selection -- see module docstring).")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLV computation.
# ---------------------------------------------------------------------------


def devig_pair_decimal(price_side, price_other):
    """Additive-normalization de-vig, applied directly to decimal odds
    (1/price is the implied probability -- identical math to
    src/betting/tracking_db.py's devig_prob on American odds, without an
    unnecessary decimal->American round trip)."""
    if price_side is None or price_other is None:
        return None, None
    if pd.isna(price_side) or pd.isna(price_other):
        return None, None
    p_side = 1.0 / price_side
    p_other = 1.0 / price_other
    total = p_side + p_other
    if total <= 0:
        return None, None
    return p_side / total, p_other / total


def build_closing_consensus(cl: pd.DataFrame) -> pd.DataFrame:
    """Per (event_id, goalie_id, line): mean de-vigged closing probability of
    Over and of Under, averaged across every book that quotes BOTH sides of
    that exact line at closing. De-vigged probabilities are averaged, never
    raw odds (the odds-averaging bug)."""
    group_cols = ["event_id", "goalie_id", "line"]
    per_book = pivot_both_sides(cl, group_cols + ["book"])
    novig = per_book.apply(
        lambda r: devig_pair_decimal(r["price_decimal_over"], r["price_decimal_under"]), axis=1,
    )
    per_book["novig_over"] = [t[0] for t in novig]
    per_book["novig_under"] = [t[1] for t in novig]

    consensus = per_book.groupby(group_cols).agg(
        consensus_prob_over=("novig_over", "mean"),
        consensus_prob_under=("novig_under", "mean"),
        n_closing_books=("book", "nunique"),
    ).reset_index()
    log(f"\nClosing consensus table: {len(consensus)} distinct (event, goalie, line) combinations "
        f"with a de-vigged closing consensus, from {len(per_book)} book-level closing quotes.")
    return consensus


def build_closing_same_book_lookup(cl: pd.DataFrame) -> dict:
    """(event_id, book, goalie_id, line, side) -> closing price_decimal, for
    the same-book same-line price CLV comparison."""
    key_cols = ["event_id", "book", "goalie_id", "line", "side"]
    dup = cl.groupby(key_cols)["price_decimal"].nunique()
    n_conflict = int((dup > 1).sum())
    if n_conflict:
        log(f"  WARNING: {n_conflict} same-book closing (event,book,goalie,line,side) groups have "
            "conflicting prices; taking first.")
    first = cl.drop_duplicates(subset=key_cols, keep="first")
    return {
        (row.event_id, row.book, row.goalie_id, row.line, row.side): row.price_decimal
        for row in first[key_cols + ["price_decimal"]].itertuples(index=False)
    }


def compute_clv(bets: pd.DataFrame, consensus: pd.DataFrame, same_book_closing: dict) -> pd.DataFrame:
    bets = bets.copy()

    consensus_idx = consensus.set_index(["event_id", "goalie_id", "line"])

    def _bettime_novig(row):
        price_side = row["price_decimal_over"] if row["bet_side"] == "OVER" else row["price_decimal_under"]
        price_other = row["price_decimal_under"] if row["bet_side"] == "OVER" else row["price_decimal_over"]
        novig_side, _ = devig_pair_decimal(price_side, price_other)
        return novig_side

    def _consensus_closing(row):
        key = (row["event_id"], row["goalie_id"], row["line"])
        if key not in consensus_idx.index:
            return None
        rec = consensus_idx.loc[key]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        return rec["consensus_prob_over"] if row["bet_side"] == "OVER" else rec["consensus_prob_under"]

    def _same_book_close_price(row):
        snap_side = "Over" if row["bet_side"] == "OVER" else "Under"
        key = (row["event_id"], row["book"], row["goalie_id"], row["line"], snap_side)
        return same_book_closing.get(key)

    bets["bettime_novig_prob"] = bets.apply(_bettime_novig, axis=1)
    bets["closing_consensus_prob"] = bets.apply(_consensus_closing, axis=1)
    bets["clv_prob"] = bets["closing_consensus_prob"] - bets["bettime_novig_prob"]

    bets["bettime_price_chosen_side"] = np.where(
        bets["bet_side"] == "OVER", bets["price_decimal_over"], bets["price_decimal_under"]
    )
    bets["closing_price_same_book"] = bets.apply(_same_book_close_price, axis=1)
    bets["clv_price"] = bets["bettime_price_chosen_side"] - bets["closing_price_same_book"]

    bets["cluster_id"] = bets["event_id"].astype(str) + "_" + bets["goalie_id"].astype(str)
    return bets


def attach_actuals(bets: pd.DataFrame, frame) -> pd.DataFrame:
    outcomes = frame.df[["game_id", "goalie_id", "saves"]].drop_duplicates(subset=["game_id", "goalie_id"])
    bets = bets.merge(outcomes, on=["game_id", "goalie_id"], how="left")

    def _result(row):
        if pd.isna(row["saves"]):
            return None
        if row["saves"] == row["line"]:
            return "PUSH"
        over_hit = row["saves"] > row["line"]
        won = over_hit if row["bet_side"] == "OVER" else not over_hit
        return "WIN" if won else "LOSS"

    bets["result"] = bets.apply(_result, axis=1)
    return bets


# ---------------------------------------------------------------------------
# Cluster bootstrap for an arbitrary per-bet metric.
# ---------------------------------------------------------------------------


def cluster_bootstrap_mean_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci_pct: float = CI_PCT,
) -> dict:
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids, dtype=object)
    mask = ~np.isnan(values)
    values = values[mask]
    cluster_ids = cluster_ids[mask]

    if len(values) == 0:
        return {"mean": None, "lower": None, "upper": None, "n_bets": 0, "n_clusters": 0}

    unique_clusters, inv = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    cluster_sum = np.zeros(n_clusters)
    cluster_count = np.zeros(n_clusters)
    np.add.at(cluster_sum, inv, values)
    np.add.at(cluster_count, inv, 1)

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_resamples)
    for b in range(n_resamples):
        draw = rng.randint(0, n_clusters, size=n_clusters)
        counts = np.bincount(draw, minlength=n_clusters)
        total_val = np.dot(counts, cluster_sum)
        total_n = np.dot(counts, cluster_count)
        boot_means[b] = total_val / total_n if total_n > 0 else np.nan

    valid = boot_means[~np.isnan(boot_means)]
    alpha = (100.0 - ci_pct) / 2.0
    return {
        "mean": float(values.mean()),
        "lower": float(np.percentile(valid, alpha)),
        "upper": float(np.percentile(valid, 100.0 - alpha)),
        "n_bets": int(len(values)),
        "n_clusters": int(n_clusters),
    }


def fmt_ci(stat: dict, decimals: int = 4) -> str:
    if stat["n_bets"] == 0:
        return "n/a (0 bets)"
    return (
        f"mean={stat['mean']:+.{decimals}f}  95% CI=[{stat['lower']:+.{decimals}f}, "
        f"{stat['upper']:+.{decimals}f}]  n_bets={stat['n_bets']}  n_clusters={stat['n_clusters']}"
    )


# ---------------------------------------------------------------------------
# Sanity check: agreement between the reload-integrity-gate pricing path and
# the new snapshot-audit pricing path, on shared (game_id, goalie_id, line).
# ---------------------------------------------------------------------------


def agreement_check(gate_result: dict, candidates: pd.DataFrame) -> None:
    log("\n--- Sanity check: model-probability agreement on shared goalie-night x line ---")
    df_bet_test = gate_result["df_bet_test"].copy()
    df_bet_test["model_p_over_gate"] = gate_result["p_over_test"]
    df_bet_test = df_bet_test[gate_result["matched_test"]]
    gate_keyed = df_bet_test[["game_id", "goalie_id", "betting_line", "model_p_over_gate"]].rename(
        columns={"betting_line": "line"}
    )

    merged = candidates.merge(gate_keyed, on=["game_id", "goalie_id", "line"], how="inner")
    n_shared_rows = len(merged)
    n_shared_nights = merged[["game_id", "goalie_id"]].drop_duplicates().shape[0]
    n_gate_nights = gate_keyed[["game_id", "goalie_id"]].drop_duplicates().shape[0]
    n_audit_nights = candidates[["game_id", "goalie_id"]].drop_duplicates().shape[0]

    log(f"  Original test-fold reproduction: {n_gate_nights} distinct goalie-nights (multibook rows).")
    log(f"  New snapshot-audit universe: {n_audit_nights} distinct goalie-nights (bet-time rows).")
    log(f"  Rows sharing an identical (game_id, goalie_id, line): {n_shared_rows}, "
        f"covering {n_shared_nights} distinct goalie-nights.")

    if n_shared_rows == 0:
        log("  No shared (goalie-night, line) rows to compare -- skipping correlation.")
        return

    diff = (merged["model_p_over"] - merged["model_p_over_gate"]).abs()
    corr = merged["model_p_over"].corr(merged["model_p_over_gate"])
    log(f"  model_p_over agreement: correlation={corr:.6f}, max abs diff={diff.max():.2e}, "
        f"mean abs diff={diff.mean():.2e}")
    if diff.max() > 1e-9:
        log("  NOTE: nonzero difference found -- investigate before trusting the audit pipeline.")
    else:
        log("  Exact agreement (both pricing paths call the identical pmf/price_line code on the "
            "identical dist_preds_test object) -- no wiring bug detected in the new pricing path.")


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def print_report(bets: pd.DataFrame) -> None:
    log("\n" + "=" * 80)
    log("CLV AUDIT REPORT: frozen pace_shots policy, bet-time vs closing")
    log("=" * 80)

    n_bets = len(bets)
    n_nights = bets[["event_id", "goalie_id"]].drop_duplicates().shape[0]
    log(f"Selected bets: {n_bets}  (goalie-nights with a bet: {n_nights})")
    log(f"Side split: OVER={int((bets['bet_side'] == 'OVER').sum())}  "
        f"UNDER={int((bets['bet_side'] == 'UNDER').sum())}")

    n_prob_covered = int(bets["clv_prob"].notna().sum())
    n_price_covered = int(bets["clv_price"].notna().sum())
    log(f"\nProbability CLV coverage: {n_prob_covered}/{n_bets} "
        f"({n_prob_covered / n_bets * 100:.1f}%) had a de-vigged closing consensus at the bet's line.")
    log(f"Price CLV coverage: {n_price_covered}/{n_bets} "
        f"({n_price_covered / n_bets * 100:.1f}%) had the same book quoting the same line at close.")

    log("\n--- Headline: probability CLV (primary pre-registered metric) ---")
    prob_stat = cluster_bootstrap_mean_ci(bets["clv_prob"].values, bets["cluster_id"].values)
    log(f"  {fmt_ci(prob_stat)}")

    log("\n--- Price CLV (secondary, same-book same-line) ---")
    price_stat = cluster_bootstrap_mean_ci(bets["clv_price"].values, bets["cluster_id"].values)
    log(f"  {fmt_ci(price_stat)}")

    log("\n--- Interpretation (pre-registered in section 3.15) ---")
    if prob_stat["n_bets"] == 0:
        log("  No bets with matchable probability CLV -- cannot interpret.")
    elif prob_stat["lower"] > 0:
        log("  Probability CLV 95% CI is entirely ABOVE zero: STRONG CONFIRMATION. "
            "Justifies opening-season shadow + token live stakes.")
    elif prob_stat["upper"] < 0:
        log("  Probability CLV 95% CI is entirely BELOW zero: treat the +9.02% fold ROI as LUCK.")
    else:
        log("  Probability CLV 95% CI SPANS ZERO: UNRESOLVED. In-season shadow run remains the arbiter.")

    log("\n--- Descriptive breakdown by month ---")
    for month, g in bets.groupby("month"):
        p = cluster_bootstrap_mean_ci(g["clv_prob"].values, g["cluster_id"].values)
        pr = cluster_bootstrap_mean_ci(g["clv_price"].values, g["cluster_id"].values)
        log(f"  {month}  n={len(g):4d}  prob_clv: {fmt_ci(p)}")
        log(f"  {'':7s}       price_clv: {fmt_ci(pr)}")

    log("\n--- Descriptive breakdown by side ---")
    for side, g in bets.groupby("bet_side"):
        p = cluster_bootstrap_mean_ci(g["clv_prob"].values, g["cluster_id"].values)
        pr = cluster_bootstrap_mean_ci(g["clv_price"].values, g["cluster_id"].values)
        log(f"  {side:6s} n={len(g):4d}  prob_clv: {fmt_ci(p)}")
        log(f"  {'':7s}       price_clv: {fmt_ci(pr)}")

    if "result" in bets.columns:
        graded = bets[bets["result"].isin(["WIN", "LOSS"])]
        if len(graded):
            hit_rate = (graded["result"] == "WIN").mean() * 100
            log(f"\n(Descriptive only, not requested by the audit design) hit rate on graded bets: "
                f"{hit_rate:.1f}% ({len(graded)} graded, "
                f"{int((bets['result'] == 'PUSH').sum())} pushes, "
                f"{int(bets['result'].isna().sum())} ungraded/future).")


def main() -> int:
    for path in (SNAPSHOTS_PATH, ARTIFACT_DIR / "metadata.json", epd.DATA_PATH_MULTIBOOK):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    frame, clean_split, dist, dist_preds_test, shots_cols, rate_cols, alpha, frozen_result = (
        load_frozen_pace_shots()
    )
    gate_result = run_integrity_gate(dist, dist_preds_test)
    ev_threshold = float(frozen_result["test_single_touch"]["threshold"])
    if ev_threshold != 0.05:
        raise AssertionError(f"Frozen threshold expected 0.05, artifact says {ev_threshold}.")

    log("\n" + "=" * 80)
    log("STAGE 1: CLV audit on saves_lines_snapshots.parquet bet-time pass")
    log("=" * 80)

    window = load_snapshot_window()
    bt = clean_bettime_pass(window)
    cl = clean_closing_pass(window)

    game_lookup_df = frame.df[["goalie_id", "game_date", "game_id"]].copy()
    game_lookup_df["date_str"] = game_lookup_df["game_date"].dt.strftime("%Y-%m-%d")
    dup_keys = game_lookup_df.duplicated(subset=["goalie_id", "date_str"]).sum()
    if dup_keys:
        raise AssertionError(f"clean_training_data has {dup_keys} duplicate (goalie_id, date) keys; "
                              "game_id lookup would be ambiguous.")
    game_lookup = dict(zip(zip(game_lookup_df["goalie_id"], game_lookup_df["date_str"]), game_lookup_df["game_id"]))

    bt, n_unmatched_game_id = attach_game_id(bt, game_lookup)
    log(f"\nAttached game_id to bet-time rows via (goalie_id, game_date_eastern): "
        f"{n_unmatched_game_id} rows could not be matched (dropped).")

    candidates = build_bet_candidates(bt)
    candidates = attach_model_probabilities(candidates, dist, dist_preds_test)

    agreement_check(gate_result, candidates)

    bets = select_bets(candidates, ev_threshold)
    if len(bets) == 0:
        raise RuntimeError("No bets were selected -- cannot audit CLV on an empty bet set.")

    consensus = build_closing_consensus(cl)
    same_book_closing = build_closing_same_book_lookup(cl)
    bets = compute_clv(bets, consensus, same_book_closing)
    bets = attach_actuals(bets, frame)

    print_report(bets)

    OUTPUT_BETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bets.to_parquet(OUTPUT_BETS_PATH, index=False)
    log(f"\nSaved {len(bets)} selected bets with CLV inputs/outputs to: {OUTPUT_BETS_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
