"""
Component G: cross-line outlier pricing (development phase only).

Contract: docs/BREAKTHROUGH_MODEL_PLAN.md sections 1a, 3.4, 4.6, 4.7, 6.1-6.4
and 7 (Gate C), read together with docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md
section 1 (shared conventions) and section 8 (Experiment 7 -- this task).
This script implements the DEVELOPMENT phase only: develop and lock a
shape-translation method + gap threshold on the 2023-24 season ONLY. It does
not touch 2024-25 or 2025-26 in any form. The single 2024-25-closing-pass
confirmatory touch (pre-reg section 8.2/8.5) happens in a separate run,
strictly after the pre-registration is locked and the lead reviewer approves
the policy locked here (pre-reg section 0.2).

Background (plan section 3.4, verified): roughly 11% of bet-time
goalie-nights (16-18% at closing) have books posting saves lines a full save
or more apart; one save of line is worth 5-6 probability points. Same-line
price dispersion across books is negligible (~0.35-0.55 probability points
against a typical ~3.5-point half-vig), so "stale book" here means stale
ACROSS LINES. The strategy needs no edge over the market consensus -- only
over a single book whose line has not caught up.

Pre-registration alignments (pre-reg section 8, checked item by item):
  - 8.3: the temporal bettime-to-close CLV diagnostic (does a flagged
    bettime outlier move toward the consensus by close?) is produced here as
    a DEVELOPMENT-stage deliverable, net of the 2023-24 unconditional
    bettime-to-close drift baseline. The reference measurement for that
    baseline (-0.006 percentage points, cluster 95% CI [-0.023, +0.010] --
    statistically zero for this season) is re-derived independently below
    and compared against the reference; the subtraction is applied anyway
    per plan section 4.6's mandatory rule.
  - 8.6: the NB dispersion alpha used for shape translation comes from
    2023-24 development data / prior-season outcomes ONLY -- NEVER from the
    frozen production artifact (whose training window postdates this
    evaluation window). Experiment 3's validation-fitted alpha convention
    had not concluded at implementation time, so the pre-reg's explicit
    fallback branch ("a value fit on 2023-24 training/validation data
    only") is used. An independent verification pass found held-out-fitted
    alpha ~0.030 on a 2022-23-trained model; the candidate dispersions
    below are checked against that reference and any large discrepancy is
    investigated in the log, not waved through.

  IMPLEMENTATION CORRECTION (found and fixed before any result was
  reported): an earlier version of this script chose the market-calibrated
  dispersion by minimizing MAE between the shape-translated fair
  probability and off-modal-line books' own realized de-vigged prices. That
  metric is CIRCULAR for this specific purpose: it is minimized by the
  FLATTEST possible distribution (translated probabilities barely move
  across lines), because a flat shape trivially "explains away" any
  cross-line disagreement as consistent with the consensus -- which is
  exactly the mispricing this experiment is trying to detect and bet
  against, not explain away. It selected sigma=11.5 at the edge of the
  search grid (MAE was still falling at the boundary) and produced a nearly
  flat translation, which in turn produced almost no flagged bets. Fixed by
  switching both dispersion calibration and shape/family locking to an
  OUTCOME-based Brier score (fair probability at each book's own posted
  line vs the ACTUAL game result), computed on 2023-24 development rows --
  legitimate per pre-reg section 0.1 ("2023-24 ... outcomes have already
  been viewed ... new modeling against them is development") and consistent
  with how every other experiment in this project selects between
  candidates (paired Brier delta vs. market). This metric cannot degenerate
  to "flat wins" because a flat/uninformative distribution scores WORSE, not
  better, against real binary outcomes at real lines.
  - 8.4: the book-concentration diagnostic and the venue-accessible
    (BetOnline) vs research-only book split are both reported. NOTE: the
    2023-24 snapshot archive contains ZERO betonlineag rows (BetOnline
    coverage starts in 2024-25), so the venue-accessible subset is empty by
    construction during development; the pre-reg 8.5 gate's
    venue-accessible ROI can only be scored on the later 2024-25 touch.
  - 8.5/8.6: this script's final output states the locked threshold +
    translation method with a written rationale, per the freeze rule. The
    numeric pass/fail gate itself is NOT scored here -- it belongs to the
    2024-25 single touch.

Conventions reused from scripts/clv_audit_pace_policy.py (read in full; its
helpers are mechanically copied rather than imported because parallel agents
may be editing that script concurrently, and importing it drags in the
frozen-artifact reload machinery this experiment must not touch):
  - De-vig: per-book two-way additive normalization on decimal odds
    (p_side_raw / (p_over_raw + p_under_raw)); consensus = mean of per-book
    DE-VIGGED probabilities. Raw odds are never averaged across books (the
    odds-averaging bug, docs/HISTORICAL_DATA_ANALYSIS.md section 1).
  - Timestamps in saves_lines_snapshots.parquet are ISO strings; they are
    compared as strings, never parsed to pandas Timestamps.
  - Bet-time pass dedup: keep only the earliest requested_ts per event
    (schedule-correction artifact), then drop exact-duplicate rows. Applied
    to both passes (closing shows the same rare artifact).
  - EV/selection: EV = fair_prob - RAW (vig-inclusive) implied probability
    of the book's own price, via American-rounded odds
    (betting.odds_utils.decimal_to_american + calculate_ev), matching
    experiments.harness.decide_bet's literal selection code. A
    decimal-precise diagnostic gap is also carried.
  - Goalie-night cluster bootstrap: 10,000 resamples, seed 42, 95% CI,
    cluster = event_id + "_" + goalie_id (event_id and game_id are 1:1 per
    night in this archive, so this matches pre-reg section 1's
    game_id-based cluster definition).

HARD FENCE: this script only ever loads/filters/counts/describes 2023-24
rows. Every loader filters immediately after reading the parquet and never
returns an unfiltered frame; ALLOWED_DEV_SEASON_LABELS is the single point
of truth and is asserted on every load. Prior-outcome shape fitting uses
2022-23-and-earlier rows only (stricter than the fence -- it never even
reaches 2023-24 outcomes).

Usage:
    python scripts/experiment_cross_line_pricing.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.distributional_saves import CAP, SavesDistribution  # noqa: E402
from betting.odds_utils import calculate_ev, decimal_to_american  # noqa: E402


# ---------------------------------------------------------------------------
# Paths and hard-fenced constants
# ---------------------------------------------------------------------------

SNAPSHOTS_PATH = REPO_ROOT / "data" / "processed" / "saves_lines_snapshots.parquet"
CLEAN_PATH = REPO_ROOT / "data" / "processed" / "clean_training_data.parquet"

# Single point of truth for the season fence. Every loader asserts against
# this set. Do not add "2024-25" or "2025-26" here during development.
ALLOWED_DEV_SEASON_LABELS = {"2023-24"}
DEV_SEASON_LABEL = "2023-24"
DEV_SEASON_CODE = 20232024  # clean_training_data.parquet's "season" int code
PRIOR_SEASON_CODE_MAX = 20222023  # prior-outcome shape fitting uses <= this

# Same literal boundaries as scripts/build_odds_snapshots.py (Eastern-date
# based, inclusive).
SEASON_BOUNDS = [
    ("2023-24", "2023-08-01", "2024-07-31"),
    ("2024-25", "2024-08-01", "2025-07-31"),
    ("2025-26", "2025-08-01", "2026-07-31"),
]

# Pre-reg section 0.3 / plan section 1a: BetOnline is the only sportsbook in
# this archive the user can execute at. Underdog/PrizePicks are not
# sportsbook feeds and do not appear in this archive at all.
EXECUTABLE_BOOKS = {"betonlineag"}

CAP_N = CAP  # 70, reused from distributional_saves

# Chronological within-development split used for dispersion calibration vs
# shape-comparison fairness: dev-outcome-fitted candidates are fit on rows
# dated before this Eastern date (string comparison for the snapshots side;
# Timestamp comparison for clean_training_data's game_date, which is not
# subject to the ISO-string gotcha), and ALL candidates are LOCKED using
# outcome Brier score computed on rows on/after it. Declared before any
# calibration result was inspected.
DEV_CALIB_SPLIT_DATE = "2024-01-15"

# Pre-declared dispersion grid for the outcome-Brier-calibrated candidates.
# Brackets both the held-out-fitted reference (~0.030) and the prior-outcome
# proxy fit (~0.100, see fit_prior_shape_params).
NB2_ALPHA_GRID = [0.010, 0.020, 0.030, 0.040, 0.055, 0.075, 0.100]
NORMAL_SIGMA_GRID = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]

# Pre-declared gap-threshold grid (probability points of EV net of the
# book's own vig), swept on 2023-24 development data only (plan 4.7 step 3).
GAP_THRESHOLD_GRID = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]

# Pre-declared locking rule (declared before inspecting sweep results):
# among thresholds with at least MIN_FLAGGED graded bet-time bets, prefer
# the SMALLEST threshold whose probability-CLV-net-of-drift cluster 95% CI
# is entirely above zero (broadest coverage among mechanism-confirmed
# thresholds; CLV is the mechanism metric per plan 6.3 -- "ROI remains
# secondary"). If none qualifies, fall back to the threshold maximizing mean
# net CLV among thresholds meeting the minimum size, reported as unresolved.
MIN_FLAGGED_BETTIME_BETS_FOR_LOCK = 20

# Pre-reg section 8.4's already-measured 2023-24 unconditional
# bettime-to-close drift baseline, quoted in PROBABILITY units (the pre-reg
# states it in percentage points: -0.006% [-0.023%, +0.010%]). Re-derived
# independently below and compared against this reference.
DRIFT_REFERENCE = {"mean": -0.00006, "lower": -0.00023, "upper": +0.00010}

N_BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
CI_PCT = 95.0


# ---------------------------------------------------------------------------
# Logging (prints to stdout and accumulates for run_log.txt)
# ---------------------------------------------------------------------------


class Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str = "") -> None:
        print(msg)
        self.lines.append(str(msg))

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers mechanically copied from scripts/clv_audit_pace_policy.py (see
# module docstring for why copied rather than imported)
# ---------------------------------------------------------------------------


def devig_pair_decimal(price_side, price_other):
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


def pivot_both_sides(df: pd.DataFrame, group_cols: list[str]) -> tuple:
    key_with_side = group_cols + ["side"]
    conflict_check = df.groupby(key_with_side)["price_decimal"].nunique()
    n_conflicts = int((conflict_check > 1).sum())

    wide = df.pivot_table(index=group_cols, columns="side", values="price_decimal", aggfunc="first")
    wide = wide.reset_index()
    for col in ("Over", "Under"):
        if col not in wide.columns:
            wide[col] = np.nan
    n_total = len(wide)
    wide = wide.dropna(subset=["Over", "Under"]).copy()
    wide = wide.rename(columns={"Over": "price_decimal_over", "Under": "price_decimal_under"})
    return wide, n_conflicts, n_total


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
        return "n/a (0 obs)"
    return (
        f"mean={stat['mean']:+.{decimals}f}  95% CI=[{stat['lower']:+.{decimals}f}, "
        f"{stat['upper']:+.{decimals}f}]  n={stat['n_bets']}  n_clusters={stat['n_clusters']}"
    )


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Season-fenced loaders. Every loader filters immediately after
# pd.read_parquet and never returns an unfiltered frame.
# ---------------------------------------------------------------------------


def load_snapshots_for_dev_season(season_label: str, log: Callable[[str], None]) -> pd.DataFrame:
    assert season_label in ALLOWED_DEV_SEASON_LABELS, (
        f"REFUSING to load snapshots for season={season_label!r}: not in the fenced "
        f"development-season allowlist {ALLOWED_DEV_SEASON_LABELS}."
    )
    bounds = {label: (start, end) for label, start, end in SEASON_BOUNDS}
    start, end = bounds[season_label]

    df = pd.read_parquet(SNAPSHOTS_PATH)
    df = df[(df["game_date_eastern"] >= start) & (df["game_date_eastern"] <= end)].copy()
    # Defensive re-check, as string comparisons (documented gotcha: these
    # timestamp/date columns are ISO strings, never parse to Timestamps).
    assert (df["game_date_eastern"] >= start).all() and (df["game_date_eastern"] <= end).all()

    df = df[df["goalie_id"].notna()].copy()
    df["goalie_id"] = df["goalie_id"].astype(int)
    log(
        f"Loaded saves_lines_snapshots.parquet, filtered to season={season_label} "
        f"({start}..{end}) with matched goalie_id: {len(df)} rows."
    )
    log(f"  snapshot_pass breakdown: {df['snapshot_pass'].value_counts().to_dict()}")
    log(f"  book breakdown: {df['book'].value_counts().to_dict()}")
    if not set(df["book"].unique()) & EXECUTABLE_BOOKS:
        log(
            "  NOTE (pre-reg 8.4 coverage-at-bettable-books): ZERO rows from executable books "
            f"({sorted(EXECUTABLE_BOOKS)}) in the 2023-24 archive -- the venue-accessible subset "
            "is empty by construction this season; it becomes scoreable on the 2024-25 touch."
        )
    return df


def load_prior_shape_data(log: Callable[[str], None]) -> pd.DataFrame:
    """PRIOR-only outcomes (season <= 2022-23) from clean_training_data.parquet,
    used exclusively for the prior-outcome dispersion candidates. Never
    touches 2023-24 or later rows."""
    df = pd.read_parquet(CLEAN_PATH)
    df = df[df["season"] <= PRIOR_SEASON_CODE_MAX].copy()
    log(
        f"Loaded clean_training_data.parquet, filtered to season<={PRIOR_SEASON_CODE_MAX} "
        f"(prior-outcome shape fitting only): {len(df)} rows, "
        f"{df['game_date'].min().date()}..{df['game_date'].max().date()}."
    )
    return df


def load_dev_season_clean(season_label: str, log: Callable[[str], None]) -> pd.DataFrame:
    """2023-24-only rows of clean_training_data.parquet, for the game_id
    lookup and actual-saves outcome join."""
    assert season_label in ALLOWED_DEV_SEASON_LABELS
    df = pd.read_parquet(CLEAN_PATH)
    df = df[df["season"] == DEV_SEASON_CODE].copy()
    log(
        f"Loaded clean_training_data.parquet, filtered to season=={DEV_SEASON_CODE} "
        f"(2023-24 outcomes/game_id join only): {len(df)} rows, "
        f"{df['game_date'].min().date()}..{df['game_date'].max().date()}."
    )
    return df


# ---------------------------------------------------------------------------
# Distribution shapes. The NB2 pmf machinery is reused from
# SavesDistribution.shots_pmf (generic NB2/Poisson over a nonnegative count,
# nothing shots-specific); Normal is the pre-declared alternative family.
# ---------------------------------------------------------------------------


@dataclass
class Shape:
    name: str
    family: str
    param_desc: str
    param_value: float
    param_provenance: str
    price_over: Callable[[float, float], tuple]  # (mu, line) -> (p_over, p_under, p_push)
    invert_mu: Callable[[float, float], float]  # (line, p_over_target) -> mu


def make_nb2_shape(dist: SavesDistribution, alpha: float, name: str, provenance: str) -> Shape:
    cache: dict = {}

    def price_over(mu: float, line: float) -> tuple:
        pmf = dist.shots_pmf(mu, alpha)
        return dist.price_line(pmf, line)

    def invert_mu(line: float, p_target: float) -> float:
        p_target = min(max(p_target, 0.002), 0.998)
        key = (round(float(line), 2), round(float(p_target), 8))
        if key in cache:
            return cache[key]

        def f(mu: float) -> float:
            return price_over(mu, line)[0] - p_target

        lo, hi = 0.1, float(CAP_N) - 0.1
        if f(lo) > 0 or f(hi) < 0:
            raise ValueError(f"NB2 invert_mu bracket failed for line={line}, p_target={p_target}")
        result = brentq(f, lo, hi, xtol=1e-6, rtol=1e-8, maxiter=100)
        cache[key] = result
        return result

    return Shape(
        name=name, family="negative_binomial", param_desc=f"alpha={alpha:.6f}", param_value=alpha,
        param_provenance=provenance, price_over=price_over, invert_mu=invert_mu,
    )


def make_normal_shape(sigma: float, name: str, provenance: str) -> Shape:
    def price_over(mu: float, line: float) -> tuple:
        p_under = float(norm.cdf((line - mu) / sigma))
        return 1.0 - p_under, p_under, 0.0

    def invert_mu(line: float, p_target: float) -> float:
        p_target = min(max(p_target, 0.002), 0.998)
        return float(line - sigma * norm.ppf(1.0 - p_target))

    return Shape(
        name=name, family="normal", param_desc=f"sigma={sigma:.6f}", param_value=sigma,
        param_provenance=provenance, price_over=price_over, invert_mu=invert_mu,
    )


def fit_prior_shape_params(prior_df: pd.DataFrame, log: Callable[[str], None]) -> dict:
    """Prior-outcome dispersion candidates, fit from 2022-23-and-earlier
    outcomes only, using each start's prior-only rolling average
    (saves_rolling_10, an existing leak-free feature) as the local mu proxy.
    NB2 alpha uses the method-of-moments formula from
    experiments.distributional_saves.fit_dispersion (Var = mu + alpha*mu^2).

    ALPHA-PROVENANCE INVESTIGATION (required by pre-reg 8.6 alignment): this
    proxy fit is expected to land WELL ABOVE the held-out-fitted reference
    (~0.030 from a 2022-23-trained model) because the rolling average's own
    prediction error folds into the residual variance -- the proxy's
    residual variance is the model's conditional variance PLUS the variance
    of (true conditional mean - rolling average). A market-implied
    cross-check: plan 3.4's verified "one save of line is worth 5-6
    probability points" means the market's own distribution has ~0.05-0.06
    pmf mass at the central line, implying variance ~1/(2*pi*mass^2) ~=
    44-64, i.e. alpha ~= (var - mu)/mu^2 ~= 0.025-0.053 at mu ~= 26.6 --
    consistent with the ~0.030 reference and NOT with the proxy fit. That
    is exactly why the market-calibrated candidates below exist: the
    translation task needs the MARKET's conditional dispersion, and the
    proxy fit is kept only as a pre-declared conservative (wide) candidate.
    """
    mu_proxy_col = "saves_rolling_10"
    mu = prior_df[mu_proxy_col].values.astype(float)
    y = prior_df["saves"].values.astype(float)
    resid = y - mu

    train_mean = float(np.mean(mu))
    train_var = float(np.mean(resid**2))
    mean_mu2 = float(np.mean(mu**2))
    alpha = max((train_var - train_mean) / mean_mu2, 1e-6)
    sigma = float(np.sqrt(train_var))

    log("\n--- Prior-outcome (2022-23 and earlier) dispersion fit ---")
    log(f"  mu proxy column: {mu_proxy_col};  n rows: {len(prior_df)}")
    log(f"  mean(mu proxy) = {train_mean:.4f};  mean(resid^2) = {train_var:.4f}")
    log(f"  NB2 alpha (proxy fit) = {alpha:.6f};  Normal sigma (proxy fit) = {sigma:.6f}")
    log("  Reference held-out-fitted alpha on a 2022-23-trained MODEL: ~0.030.")
    peak_mass_proxy = 1.0 / np.sqrt(2 * np.pi * (train_mean + alpha * train_mean**2))
    log(
        f"  Investigation: proxy alpha implies ~{peak_mass_proxy * 100:.1f} probability points per "
        "save of line at the center; the market's verified value is 5-6 points (plan 3.4), "
        "implying alpha ~0.025-0.053. The proxy fit is inflated by the rolling average's own "
        "prediction error (variance decomposition in fit_prior_shape_params docstring); it is "
        "retained only as the pre-declared conservative-wide candidate, and market-calibrated "
        "candidates bracketing the ~0.030 reference are fit below."
    )

    return {
        "mu_proxy_col": mu_proxy_col,
        "n_prior_rows": int(len(prior_df)),
        "train_mean_mu": train_mean,
        "train_var_resid2": train_var,
        "nb2_alpha_proxy": alpha,
        "normal_sigma_proxy": sigma,
        "reference_heldout_alpha": 0.030,
        "implied_points_per_save_at_center_proxy": float(peak_mass_proxy * 100),
    }


def fit_dev_season_shape_params(dev_clean_df: pd.DataFrame, split_date: str, log: Callable[[str], None]) -> dict:
    """Pre-reg 8.6 fallback branch: 'a value fit on 2023-24 training/
    validation data only'. Same method-of-moments procedure as
    fit_prior_shape_params, applied to the 2023-24 CALIBRATION half
    (game_date < split_date) instead of 2022-23. This is outcome data from
    the development season itself, which pre-reg section 0.1 explicitly
    permits ('2023-24 ... outcomes have already been viewed ... new
    modeling against them is development'). Directly comparable to the
    pre-reg's ~0.030 held-out-fitted reference."""
    calib = dev_clean_df[dev_clean_df["game_date"] < pd.Timestamp(split_date)].copy()
    mu_proxy_col = "saves_rolling_10"
    mu = calib[mu_proxy_col].values.astype(float)
    y = calib["saves"].values.astype(float)
    resid = y - mu

    train_mean = float(np.mean(mu))
    train_var = float(np.mean(resid**2))
    mean_mu2 = float(np.mean(mu**2))
    alpha = max((train_var - train_mean) / mean_mu2, 1e-6)
    sigma = float(np.sqrt(train_var))

    log(f"\n--- 2023-24 dev-outcome (< {split_date}) dispersion fit (pre-reg 8.6 fallback branch) ---")
    log(f"  mu proxy column: {mu_proxy_col};  n rows: {len(calib)}")
    log(f"  mean(mu proxy) = {train_mean:.4f};  mean(resid^2) = {train_var:.4f}")
    log(f"  NB2 alpha (2023-24 dev fit) = {alpha:.6f};  Normal sigma (2023-24 dev fit) = {sigma:.6f}")
    log(f"  Reference held-out-fitted alpha on a 2022-23-trained MODEL: ~0.030 -- "
        f"{'agrees' if abs(alpha - 0.030) <= 0.025 else 'DISAGREES (same proxy-inflation mechanism as the prior-outcome fit)'}.")

    return {
        "mu_proxy_col": mu_proxy_col,
        "n_calib_rows": int(len(calib)),
        "split_date": split_date,
        "train_mean_mu": train_mean,
        "train_var_resid2": train_var,
        "nb2_alpha_dev": alpha,
        "normal_sigma_dev": sigma,
    }


# ---------------------------------------------------------------------------
# Per-book-quote table construction
# ---------------------------------------------------------------------------


def clean_pass(window: pd.DataFrame, pass_name: str, log: Callable[[str], None]) -> pd.DataFrame:
    """One snapshot per event for the given pass: earliest requested_ts per
    event, then exact-duplicate drop (clv_audit_pace_policy convention)."""
    p = window[window["snapshot_pass"] == pass_name].copy()
    n_raw = len(p)
    min_ts = p.groupby("event_id")["requested_ts"].transform("min")
    n_dup_rows = int((p["requested_ts"] != min_ts).sum())
    p = p[p["requested_ts"] == min_ts].copy()
    before = len(p)
    p = p.drop_duplicates(subset=["event_id", "requested_ts", "book", "goalie_id", "side", "line", "price_decimal"])
    log(
        f"[{pass_name}] {n_raw} raw rows -> dropped {n_dup_rows} later-duplicate-snapshot rows -> "
        f"dropped {before - len(p)} exact-duplicate rows -> {len(p)} rows."
    )
    return p


def build_book_quote_table(clean_pass_df: pd.DataFrame, log: Callable[[str], None], label: str) -> pd.DataFrame:
    """One row per (event_id, goalie_id, book, line): de-vigged and raw
    implied probabilities for both sides, plus per-night line-spread
    context."""
    group_cols = ["event_id", "goalie_id", "book", "line", "commence_time", "game_date_eastern"]
    wide, n_conflicts, n_total = pivot_both_sides(clean_pass_df, group_cols)
    if n_conflicts:
        log(f"  [{label}] WARNING: {n_conflicts} (key, side) groups have conflicting prices; took first.")
    log(f"  [{label}] Both-sides pivot: {len(wide)}/{n_total} (event, goalie, book, line) groups have both sides.")

    # After the pass-level dedup, no book posts more than one line per
    # goalie-night in this season (verified 0 cases in both passes); assert
    # so a future data change cannot silently break the book-level LOO.
    n_multi = int((wide.groupby(["event_id", "goalie_id", "book"])["line"].nunique() > 1).sum())
    assert n_multi == 0, (
        f"[{label}] {n_multi} (event, goalie, book) groups post multiple lines; the book-level "
        "LOO consensus assumes one line per book per night -- investigate before proceeding."
    )

    novig = wide.apply(lambda r: devig_pair_decimal(r["price_decimal_over"], r["price_decimal_under"]), axis=1)
    wide["devig_prob_over"] = [t[0] for t in novig]
    wide["devig_prob_under"] = [t[1] for t in novig]
    wide["raw_implied_prob_over"] = 1.0 / wide["price_decimal_over"]
    wide["raw_implied_prob_under"] = 1.0 / wide["price_decimal_under"]

    grp = wide.groupby(["event_id", "goalie_id"])
    wide["n_books_this_night"] = grp["book"].transform("nunique")
    wide["modal_line_this_night"] = grp["line"].transform(lambda s: float(s.mode().min()))
    wide["line_spread_this_night"] = grp["line"].transform("max") - grp["line"].transform("min")
    wide["is_off_modal_line"] = wide["line"] != wide["modal_line_this_night"]

    return wide.reset_index(drop=True)


def report_candidate_pool(quotes: dict, log: Callable[[str], None]) -> dict:
    log("\n--- Candidate pool: goalie-nights with a full-save-or-more line spread (2023-24) ---")
    out = {}
    for pass_name, qt in quotes.items():
        nights = qt.drop_duplicates(subset=["event_id", "goalie_id"])
        n_total = len(nights)
        n_full = int((nights["line_spread_this_night"] >= 1.0).sum())
        pct = n_full / n_total * 100 if n_total else 0.0
        log(
            f"  [{pass_name}] {n_total} goalie-nights, {n_full} ({pct:.1f}%) with >=1.0 save spread. "
            "Plan 3.4/4.7 expectation: ~150-200 bettime nights (~11%), 16-18% at closing."
        )
        out[pass_name] = {"n_goalie_nights": n_total, "n_full_save_spread": n_full, "pct_full_save_spread": pct}
    return out


# ---------------------------------------------------------------------------
# Book-level leave-one-out mu consensus
# ---------------------------------------------------------------------------


def compute_mu_and_loo(quote_table: pd.DataFrame, shape: Shape, log, label: str) -> pd.DataFrame:
    """Invert each book's (line, devig_prob_over) into an implied
    distribution mean mu_own under `shape`; the leave-one-out consensus for
    a row is the mean of the OTHER books' mu_own within the same
    goalie-night (book-level LOO -- each book contributes exactly one line
    per night, asserted upstream). Rows whose own inversion failed can
    still receive a peer consensus. Single-book nights get loo_mu = NaN
    (no peer to compare against -> never flaggable)."""
    qt = quote_table.copy()
    mu_own = np.full(len(qt), np.nan)
    n_failed = 0
    lines_arr = qt["line"].values.astype(float)
    probs_arr = qt["devig_prob_over"].values.astype(float)
    for i in range(len(qt)):
        try:
            mu_own[i] = shape.invert_mu(lines_arr[i], probs_arr[i])
        except ValueError:
            n_failed += 1
    qt["mu_own"] = mu_own
    if n_failed:
        log(f"  [{label}/{shape.name}] {n_failed} rows failed mu inversion; own-mu left NaN.")

    grp = qt.groupby(["event_id", "goalie_id"])["mu_own"]
    night_sum = grp.transform(lambda s: np.nansum(s.values))
    night_n = grp.transform(lambda s: float(np.sum(~np.isnan(s.values))))
    own_valid = ~qt["mu_own"].isna()
    peer_sum = night_sum - qt["mu_own"].fillna(0.0)
    peer_n = night_n - own_valid.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = peer_sum / peer_n
    qt["loo_mu"] = np.where(peer_n >= 1, loo, np.nan)
    log(f"  [{label}/{shape.name}] {int(qt['loo_mu'].notna().sum())}/{len(qt)} rows have a valid LOO peer consensus mu.")
    return qt


# ---------------------------------------------------------------------------
# Translation accuracy (outcome-independent market self-consistency):
# translate the LOO peer consensus to each OFF-MODAL-LINE book quote's own
# line and compare with that book's realized de-vigged OVER probability.
# ---------------------------------------------------------------------------


def translation_mae(
    quotes: dict, shape: Shape, log, date_from: str | None = None, date_before: str | None = None
) -> dict:
    per_pass = {}
    all_err: list[float] = []
    for pass_name, qt in quotes.items():
        sub = qt
        if date_from is not None:
            sub = sub[sub["game_date_eastern"] >= date_from]
        if date_before is not None:
            sub = sub[sub["game_date_eastern"] < date_before]
        # Only nights with >=2 distinct lines contain off-modal quotes; the
        # LOO must be computed on the full night's book set though.
        multi = sub.groupby(["event_id", "goalie_id"])["line"].transform("nunique") >= 2
        sub = sub[multi]
        if len(sub) == 0:
            per_pass[pass_name] = {"n_off_modal_quotes_evaluated": 0, "mae": None}
            continue
        with_loo = compute_mu_and_loo(sub, shape, log, f"mae/{pass_name}")
        om = with_loo[with_loo["is_off_modal_line"] & with_loo["loo_mu"].notna()]
        fair = np.array([shape.price_over(m, l)[0] for m, l in zip(om["loo_mu"], om["line"])])
        abs_err = np.abs(fair - om["devig_prob_over"].values)
        per_pass[pass_name] = {
            "n_off_modal_quotes_evaluated": int(len(abs_err)),
            "mae": float(np.mean(abs_err)) if len(abs_err) else None,
        }
        all_err.extend(abs_err.tolist())
    pooled = float(np.mean(all_err)) if all_err else None
    return {
        "shape": shape.name, "family": shape.family, "param": shape.param_desc,
        "provenance": shape.param_provenance, "per_pass": per_pass,
        "pooled_mae": pooled, "pooled_n": len(all_err),
    }


def outcome_brier(
    quotes: dict, shape: Shape, log, date_from: str | None = None, date_before: str | None = None,
    off_modal_only: bool = False,
) -> dict:
    """OUTCOME-based Brier score of the shape-translated fair OVER
    probability (from the book-level LOO peer consensus mu, at each book's
    OWN posted line) against the ACTUAL game result. Non-degenerate: unlike
    market self-consistency, a flat/wide shape does not trivially win here
    -- it scores WORSE against real binary outcomes. LOO is computed on the
    date-filtered (whole-night) set first, then row-level filters
    (off_modal_only, matched outcome, non-push) are applied for scoring,
    exactly mirroring translation_mae's night-preserving filter order."""
    per_pass = {}
    all_sq: list[float] = []
    for pass_name, qt in quotes.items():
        sub = qt
        if date_from is not None:
            sub = sub[sub["game_date_eastern"] >= date_from]
        if date_before is not None:
            sub = sub[sub["game_date_eastern"] < date_before]
        if len(sub) == 0:
            per_pass[pass_name] = {"n": 0, "brier": None}
            continue
        with_loo = compute_mu_and_loo(sub, shape, log, f"brier/{pass_name}")
        scoreable = with_loo[
            with_loo["loo_mu"].notna() & with_loo["saves_actual"].notna() & (with_loo["saves_actual"] != with_loo["line"])
        ]
        if off_modal_only:
            scoreable = scoreable[scoreable["is_off_modal_line"]]
        if len(scoreable) == 0:
            per_pass[pass_name] = {"n": 0, "brier": None}
            continue
        fair = np.array([shape.price_over(m, l)[0] for m, l in zip(scoreable["loo_mu"], scoreable["line"])])
        actual = (scoreable["saves_actual"].values > scoreable["line"].values).astype(float)
        sq_err = (fair - actual) ** 2
        per_pass[pass_name] = {"n": int(len(sq_err)), "brier": float(np.mean(sq_err))}
        all_sq.extend(sq_err.tolist())
    pooled = float(np.mean(all_sq)) if all_sq else None
    return {
        "shape": shape.name, "family": shape.family, "param": shape.param_desc,
        "provenance": shape.param_provenance, "per_pass": per_pass,
        "pooled_brier": pooled, "pooled_n": len(all_sq),
    }


def calibrate_dispersion_on_outcomes(quotes: dict, dist: SavesDistribution, log) -> dict:
    """Grid-search the dispersion parameter minimizing OUTCOME Brier score
    (all book quotes, not only off-modal) on the CALIBRATION half of the
    development season (game_date_eastern < DEV_CALIB_SPLIT_DATE, both
    passes pooled). Outcome-based -- see outcome_brier docstring for why
    this does not degenerate to the widest candidate."""
    log("\n--- Outcome-Brier dispersion grid search (2023-24 first half, "
        f"< {DEV_CALIB_SPLIT_DATE}) ---")
    nb_results = []
    for a in NB2_ALPHA_GRID:
        shape = make_nb2_shape(dist, a, f"nb2_grid_{a}", "grid")
        r = outcome_brier(quotes, shape, lambda m: None, date_before=DEV_CALIB_SPLIT_DATE)
        nb_results.append({"alpha": a, "calib_brier": r["pooled_brier"], "n": r["pooled_n"]})
        log(f"  NB2 alpha={a:.3f}: calib Brier={r['pooled_brier']:.5f} (n={r['pooled_n']})")
    best_alpha = min(nb_results, key=lambda r: r["calib_brier"])["alpha"]

    norm_results = []
    for s in NORMAL_SIGMA_GRID:
        shape = make_normal_shape(s, f"normal_grid_{s}", "grid")
        r = outcome_brier(quotes, shape, lambda m: None, date_before=DEV_CALIB_SPLIT_DATE)
        norm_results.append({"sigma": s, "calib_brier": r["pooled_brier"], "n": r["pooled_n"]})
        log(f"  Normal sigma={s:.1f}: calib Brier={r['pooled_brier']:.5f} (n={r['pooled_n']})")
    best_sigma = min(norm_results, key=lambda r: r["calib_brier"])["sigma"]

    log(f"  Outcome-calibrated winners: NB2 alpha={best_alpha:.3f}, Normal sigma={best_sigma:.1f}")
    log(f"  Reference held-out-fitted alpha ~0.030: outcome-calibrated alpha "
        f"{'agrees with' if abs(best_alpha - 0.030) <= 0.025 else 'differs from'} the reference.")
    return {
        "nb2_grid": nb_results, "normal_grid": norm_results,
        "best_alpha": best_alpha, "best_sigma": best_sigma,
        "calib_split_date": DEV_CALIB_SPLIT_DATE,
    }


def lock_translation_shape(quotes: dict, candidates: list[Shape], log) -> tuple[Shape, dict]:
    """Compare all pre-declared candidates on the EVALUATION half of the
    development season (>= DEV_CALIB_SPLIT_DATE), which none of the
    dispersion parameters was fit on. PRIMARY locking metric: outcome
    Brier score (all book quotes) -- lower wins. Off-modal-only Brier and
    the market-self-consistency MAE are also computed and reported as
    secondary diagnostics for the winner, not used for selection."""
    log("\n" + "=" * 80)
    log("TRANSLATION-METHOD COMPARISON (2023-24 second half, outcome Brier)")
    log("=" * 80)
    log(
        "PRIMARY metric: Brier score of the shape-translated LOO peer-consensus fair OVER "
        "probability (at each book's own posted line) against the ACTUAL game result, on quotes "
        f"dated >= {DEV_CALIB_SPLIT_DATE} (the half no dispersion parameter was fit on), ALL book "
        "quotes (not only off-modal). Lower wins. This is outcome-based and cannot be gamed by a "
        "flat shape (see module docstring correction note)."
    )
    evals = []
    for shape in candidates:
        r = outcome_brier(quotes, shape, log, date_from=DEV_CALIB_SPLIT_DATE)
        evals.append(r)
        log(f"\n  {shape.name} ({shape.param_desc}, provenance: {shape.param_provenance}):")
        log(f"    pooled eval-half outcome Brier={r['pooled_brier']:.5f}  n={r['pooled_n']}")
        for pn, v in r["per_pass"].items():
            log(f"    [{pn}] Brier={v['brier']:.5f}  n={v['n']}")

    by_name = {s.name: s for s in candidates}
    winner_eval = min(evals, key=lambda e: e["pooled_brier"])
    winner = by_name[winner_eval["shape"]]
    log(
        f"\nLOCKED translation shape: {winner.name} ({winner.param_desc}) -- lowest eval-half "
        f"pooled outcome Brier ({winner_eval['pooled_brier']:.5f}) among {len(candidates)} pre-declared candidates."
    )

    log("\n--- Secondary diagnostics for the locked shape (not used for selection) ---")
    off_modal_brier = outcome_brier(quotes, winner, log, date_from=DEV_CALIB_SPLIT_DATE, off_modal_only=True)
    log(f"  Off-modal-line-only outcome Brier (eval half): {off_modal_brier['pooled_brier']} "
        f"(n={off_modal_brier['pooled_n']})")
    mkt_mae = translation_mae(quotes, winner, log, date_from=DEV_CALIB_SPLIT_DATE)
    log(f"  Market self-consistency MAE (eval half, off-modal only, NOT the selection criterion -- "
        f"see module docstring correction note): {mkt_mae['pooled_mae']} (n={mkt_mae['pooled_n']})")

    return winner, {
        "candidates": evals, "locked_shape": winner.name,
        "locked_shape_off_modal_only_brier": off_modal_brier,
        "locked_shape_market_self_consistency_mae_diagnostic": mkt_mae,
    }


# ---------------------------------------------------------------------------
# Attach translated fair probabilities and gap columns
# ---------------------------------------------------------------------------


def attach_translation_and_gap(qt_with_loo: pd.DataFrame, shape: Shape, log, label: str) -> pd.DataFrame:
    qt = qt_with_loo.copy()
    have_loo = qt["loo_mu"].notna().values
    fair_over = np.full(len(qt), np.nan)
    fair_under = np.full(len(qt), np.nan)
    fair_push = np.full(len(qt), np.nan)
    loo_arr = qt["loo_mu"].values.astype(float)
    line_arr = qt["line"].values.astype(float)
    for i in np.where(have_loo)[0]:
        fair_over[i], fair_under[i], fair_push[i] = shape.price_over(loo_arr[i], line_arr[i])
    qt["fair_prob_over"] = fair_over
    qt["fair_prob_under"] = fair_under
    qt["fair_prob_push"] = fair_push

    # Primary gap: American-rounded EV convention (harness.decide_bet's
    # literal code path). Diagnostic: exact decimal-implied gap.
    american_over = qt["price_decimal_over"].apply(decimal_to_american)
    american_under = qt["price_decimal_under"].apply(decimal_to_american)
    qt["gap_over"] = [
        calculate_ev(fo, ao) if not np.isnan(fo) else np.nan for fo, ao in zip(fair_over, american_over)
    ]
    qt["gap_under"] = [
        calculate_ev(fu, au) if not np.isnan(fu) else np.nan for fu, au in zip(fair_under, american_under)
    ]
    qt["gap_over_decimal_diag"] = qt["fair_prob_over"] - qt["raw_implied_prob_over"]
    qt["gap_under_decimal_diag"] = qt["fair_prob_under"] - qt["raw_implied_prob_under"]
    log(f"  [{label}] Attached translated fair probabilities + gaps for {int(have_loo.sum())}/{len(qt)} rows.")
    return qt


def select_bet_for_row(gap_over: float, gap_under: float, threshold: float):
    if np.isnan(gap_over) or np.isnan(gap_under):
        return None
    if gap_over >= threshold and gap_over > gap_under:
        return "OVER"
    if gap_under >= threshold:
        return "UNDER"
    return None


def flag_bets(qt: pd.DataFrame, threshold: float) -> pd.DataFrame:
    side = [select_bet_for_row(go, gu, threshold) for go, gu in zip(qt["gap_over"], qt["gap_under"])]
    out = qt.copy()
    out["bet_side"] = side
    out["gap"] = np.where(
        out["bet_side"] == "OVER", out["gap_over"], np.where(out["bet_side"] == "UNDER", out["gap_under"], np.nan)
    )
    return out[out["bet_side"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Outcome join (clv_audit_pace_policy's (goalie_id, date) convention with
# +/-1 day fallback), grading, profit
# ---------------------------------------------------------------------------


def build_game_lookup(dev_clean_df: pd.DataFrame) -> dict:
    lk = dev_clean_df[["goalie_id", "game_date", "game_id", "saves"]].copy()
    lk["date_str"] = lk["game_date"].dt.strftime("%Y-%m-%d")
    dup = lk.duplicated(subset=["goalie_id", "date_str"]).sum()
    assert dup == 0, f"clean_training_data (2023-24) has {dup} duplicate (goalie_id, date) keys."
    return {
        (int(g), d): (int(gid), float(sv))
        for g, d, gid, sv in zip(lk["goalie_id"], lk["date_str"], lk["game_id"], lk["saves"])
    }


def attach_game_id_and_saves(df: pd.DataFrame, lookup: dict, log, label: str, drop_unmatched: bool = True) -> pd.DataFrame:
    def _lookup(goalie_id, date_str):
        for offset in (0, -1, 1):
            d = (pd.Timestamp(date_str) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            key = (int(goalie_id), d)
            if key in lookup:
                return lookup[key]
        return (None, None)

    results = [_lookup(g, d) for g, d in zip(df["goalie_id"], df["game_date_eastern"])]
    df = df.copy()
    df["game_id"] = [r[0] for r in results]
    df["saves_actual"] = [r[1] for r in results]
    n_unmatched = int(df["game_id"].isna().sum())
    if drop_unmatched:
        df = df[df["game_id"].notna()].copy()
        log(f"  [{label}] game_id/outcome join: {n_unmatched} rows unmatched (dropped), {len(df)} remain.")
    else:
        log(f"  [{label}] game_id/outcome join: {n_unmatched}/{len(df)} rows unmatched (kept, saves_actual=NaN; "
            "still usable as LOO peers / for CLV, just not for outcome scoring).")
    df["game_id"] = df["game_id"].astype("Int64")
    return df


def filter_graded(df: pd.DataFrame, log, label: str) -> pd.DataFrame:
    """Drop flagged bets with no matched outcome (kept upstream as LOO peers
    but not gradeable). Mirrors the old drop-on-join behavior for the final
    ROI/CLV grading step."""
    if len(df) == 0:
        return df
    n_before = len(df)
    out = df[df["saves_actual"].notna()].copy()
    n_dropped = n_before - len(out)
    if n_dropped:
        log(f"  [{label}] dropped {n_dropped}/{n_before} flagged rows with no matched outcome.")
    return out


def grade_result(saves_actual: float, line: float, bet_side: str) -> str:
    if saves_actual == line:
        return "PUSH"
    over_hit = saves_actual > line
    won = over_hit if bet_side == "OVER" else not over_hit
    return "WIN" if won else "LOSS"


def attach_profit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df) == 0:
        for c in ("result", "profit", "cluster_id"):
            df[c] = pd.Series(dtype=object)
        return df
    df["result"] = [grade_result(sv, ln, bs) for sv, ln, bs in zip(df["saves_actual"], df["line"], df["bet_side"])]
    price_chosen = np.where(df["bet_side"] == "OVER", df["price_decimal_over"], df["price_decimal_under"])
    df["profit"] = np.where(df["result"] == "WIN", price_chosen - 1.0, np.where(df["result"] == "LOSS", -1.0, 0.0))
    df["cluster_id"] = df["event_id"].astype(str) + "_" + df["goalie_id"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Closing consensus (per exact line) and the mandatory drift baseline
# ---------------------------------------------------------------------------


def build_closing_consensus_by_line(closing_qt: pd.DataFrame, log) -> pd.DataFrame:
    consensus = closing_qt.groupby(["event_id", "goalie_id", "line"]).agg(
        consensus_prob_over=("devig_prob_over", "mean"),
        consensus_prob_under=("devig_prob_under", "mean"),
        n_closing_books=("book", "nunique"),
    ).reset_index()
    log(
        f"\nClosing consensus table (per exact event/goalie/line): {len(consensus)} entries "
        f"from {len(closing_qt)} closing book-quote rows."
    )
    return consensus


def compute_drift_baseline(bettime_qt: pd.DataFrame, closing_consensus: pd.DataFrame, log) -> dict:
    """Mandatory per plan 4.6 / pre-reg 8.4: unconditional bettime-to-close
    drift, computed on ALL 2023-24 bettime book quotes (not only flagged),
    matched to the closing consensus at the SAME exact line, split by side.
    Compared against the pre-reg's already-measured reference."""
    merged = bettime_qt.merge(closing_consensus, on=["event_id", "goalie_id", "line"], how="left")
    merged["cluster_id"] = merged["event_id"].astype(str) + "_" + merged["goalie_id"].astype(str)
    merged["drift_over"] = merged["consensus_prob_over"] - merged["devig_prob_over"]
    merged["drift_under"] = merged["consensus_prob_under"] - merged["devig_prob_under"]

    n_matched = int(merged["consensus_prob_over"].notna().sum())
    log("\n--- Unconditional bettime-to-close drift baseline (ALL 2023-24 bettime quotes) ---")
    log(f"  {n_matched}/{len(merged)} bettime book quotes matched a closing consensus at their exact line.")

    over_stat = cluster_bootstrap_mean_ci(merged["drift_over"].values, merged["cluster_id"].values)
    under_stat = cluster_bootstrap_mean_ci(merged["drift_under"].values, merged["cluster_id"].values)
    log(f"  Drift OVER  (closing consensus - bettime devig): {fmt_ci(over_stat, 5)}")
    log(f"  Drift UNDER (closing consensus - bettime devig): {fmt_ci(under_stat, 5)}")
    log(
        f"  Pre-reg 8.4 reference (OVER drift): mean={DRIFT_REFERENCE['mean']:+.5f} "
        f"CI=[{DRIFT_REFERENCE['lower']:+.5f}, {DRIFT_REFERENCE['upper']:+.5f}] -- "
        "statistically zero for this season; the net-of-drift subtraction below is applied "
        "anyway per the mandatory rule (plan 4.6), and any material disagreement between this "
        "run's measurement and the reference is a red flag to investigate."
    )
    agree = (
        over_stat["mean"] is not None
        and abs(over_stat["mean"] - DRIFT_REFERENCE["mean"]) < 0.001
    )
    log(f"  Agreement with reference within 0.1 probability points: {'YES' if agree else 'NO -- INVESTIGATE'}")

    return {
        "over": over_stat,
        "under": under_stat,
        "n_matched": n_matched,
        "n_total_bettime_quotes": len(merged),
        "reference": DRIFT_REFERENCE,
        "agrees_with_reference_within_0p001": bool(agree),
    }


def attach_clv(flagged_bettime: pd.DataFrame, closing_consensus: pd.DataFrame, drift_baseline: dict, log) -> pd.DataFrame:
    df = flagged_bettime.merge(closing_consensus, on=["event_id", "goalie_id", "line"], how="left")
    df["closing_consensus_prob_chosen_side"] = np.where(
        df["bet_side"] == "OVER", df["consensus_prob_over"], df["consensus_prob_under"]
    )
    df["bettime_devig_prob_chosen_side"] = np.where(
        df["bet_side"] == "OVER", df["devig_prob_over"], df["devig_prob_under"]
    )
    df["clv_prob"] = df["closing_consensus_prob_chosen_side"] - df["bettime_devig_prob_chosen_side"]
    drift_scalar = np.where(
        df["bet_side"] == "OVER", drift_baseline["over"]["mean"], drift_baseline["under"]["mean"]
    )
    df["clv_prob_net_of_drift"] = df["clv_prob"] - drift_scalar
    n_cov = int(df["clv_prob"].notna().sum())
    log(f"  CLV coverage: {n_cov}/{len(df)} flagged bettime bets had a closing consensus at their exact line.")
    return df


# ---------------------------------------------------------------------------
# Breakdown reporting
# ---------------------------------------------------------------------------


def summarize_roi(df: pd.DataFrame) -> dict:
    graded = df[df["result"].isin(["WIN", "LOSS"])] if len(df) else df
    n_bets = len(graded)
    n_push = int((df["result"] == "PUSH").sum()) if len(df) else 0
    if n_bets == 0:
        return {"n_bets": 0, "n_push": n_push, "hit_rate": None,
                "roi_ci": cluster_bootstrap_mean_ci(np.array([]), np.array([]))}
    hit_rate = float((graded["result"] == "WIN").mean() * 100)
    roi_ci = cluster_bootstrap_mean_ci(graded["profit"].values * 100, graded["cluster_id"].values)
    return {"n_bets": n_bets, "n_push": n_push, "hit_rate": hit_rate, "roi_ci": roi_ci}


def report_breakdowns(df: pd.DataFrame, log, label: str) -> dict:
    out = {}
    log(f"\n--- [{label}] Overall ---")
    overall = summarize_roi(df)
    log(f"  n_bets={overall['n_bets']}  n_push={overall['n_push']}  hit_rate={overall['hit_rate']}")
    log(f"  ROI (%): {fmt_ci(overall['roi_ci'], 2)}")
    out["overall"] = overall

    log(f"\n--- [{label}] By side (mandatory split, plan 4.6) ---")
    out["by_side"] = {}
    for side in ("OVER", "UNDER"):
        g = df[df["bet_side"] == side] if len(df) else df
        s = summarize_roi(g)
        log(f"  {side}: n_bets={s['n_bets']}  hit_rate={s['hit_rate']}  ROI: {fmt_ci(s['roi_ci'], 2)}")
        out["by_side"][side] = s

    log(f"\n--- [{label}] By book (pre-reg 8.4 concentration diagnostic) ---")
    out["by_book"] = {}
    if len(df):
        for book, g in df.groupby("book"):
            s = summarize_roi(g)
            share = len(g) / len(df) * 100
            log(f"  {book}: n_flagged={len(g)} ({share:.1f}% of flagged)  n_graded={s['n_bets']}  ROI: {fmt_ci(s['roi_ci'], 2)}")
            out["by_book"][book] = {**s, "n_flagged": len(g), "pct_share_of_flagged": share}
        max_share = max(v["pct_share_of_flagged"] for v in out["by_book"].values())
        log(f"  Max single-book share of flagged bets: {max_share:.1f}% "
            f"(pre-reg 8.5(c) bar for the eventual test: no more than roughly half).")
        out["max_single_book_share_pct"] = max_share

    log(f"\n--- [{label}] By line-spread-size bucket ---")
    out["by_spread_bucket"] = {}
    for lo, hi, bl in [(-0.01, 0.01, "0 (same line)"), (0.01, 0.51, "0.5"), (0.51, 1.01, "1.0"),
                       (1.01, 1.51, "1.5"), (1.51, 100, "2.0+")]:
        g = df[(df["line_spread_this_night"] > lo) & (df["line_spread_this_night"] <= hi)] if len(df) else df
        s = summarize_roi(g)
        log(f"  spread={bl}: n_flagged={len(g)}  n_graded={s['n_bets']}  ROI: {fmt_ci(s['roi_ci'], 2)}")
        out["by_spread_bucket"][bl] = {**s, "n_flagged": len(g)}

    log(f"\n--- [{label}] Executable (BetOnline) vs research-only books (plan 1a / pre-reg 0.3) ---")
    out["executable_vs_research"] = {}
    for name, g in (
        ("executable_betonline", df[df["book"].isin(EXECUTABLE_BOOKS)] if len(df) else df),
        ("research_only", df[~df["book"].isin(EXECUTABLE_BOOKS)] if len(df) else df),
    ):
        s = summarize_roi(g)
        log(f"  {name}: n_flagged={len(g)}  n_graded={s['n_bets']}  ROI: {fmt_ci(s['roi_ci'], 2)}")
        out["executable_vs_research"][name] = {**s, "n_flagged": len(g)}
    if len(df) and len(df[df["book"].isin(EXECUTABLE_BOOKS)]) == 0:
        log("  NOTE: 0 executable-book rows exist in 2023-24 (BetOnline coverage starts 2024-25); "
            "this cut is empty by construction during development, not because the filter fired.")

    return out


# ---------------------------------------------------------------------------
# Threshold sweep and locking
# ---------------------------------------------------------------------------


def run_threshold_sweep(
    bettime_qt, closing_qt, closing_consensus, drift_baseline, log
) -> tuple[list[dict], dict, str]:
    log("\n" + "=" * 80)
    log("THRESHOLD SWEEP (2023-24 development data only)")
    log("=" * 80)
    log(f"Pre-declared grid: {GAP_THRESHOLD_GRID}")
    log(
        f"Pre-declared locking rule: among thresholds with >= {MIN_FLAGGED_BETTIME_BETS_FOR_LOCK} graded "
        "bettime bets, prefer the SMALLEST threshold whose probability-CLV-net-of-drift cluster 95% CI "
        "is entirely above zero. If none qualifies, fall back to the threshold maximizing mean net CLV "
        "among eligible thresholds, reported as unresolved (not rounded up to an edge)."
    )

    sweep = []
    for thr in GAP_THRESHOLD_GRID:
        log(f"\n--- threshold={thr:.2f} ---")
        bt_flagged = attach_profit(filter_graded(flag_bets(bettime_qt, thr), log, f"bt thr={thr:.2f}"))
        cl_flagged = attach_profit(filter_graded(flag_bets(closing_qt, thr), log, f"cl thr={thr:.2f}"))
        if len(bt_flagged):
            bt_flagged = attach_clv(bt_flagged, closing_consensus, drift_baseline, log)

        bt_roi = summarize_roi(bt_flagged)
        cl_roi = summarize_roi(cl_flagged)
        if len(bt_flagged):
            clv_net = cluster_bootstrap_mean_ci(bt_flagged["clv_prob_net_of_drift"].values, bt_flagged["cluster_id"].values)
            clv_raw = cluster_bootstrap_mean_ci(bt_flagged["clv_prob"].values, bt_flagged["cluster_id"].values)
        else:
            clv_net = clv_raw = cluster_bootstrap_mean_ci(np.array([]), np.array([]))

        n_over_bt = int((bt_flagged["bet_side"] == "OVER").sum()) if len(bt_flagged) else 0
        n_under_bt = int((bt_flagged["bet_side"] == "UNDER").sum()) if len(bt_flagged) else 0
        log(
            f"  bettime: n_flagged={len(bt_flagged)} (OVER={n_over_bt}/UNDER={n_under_bt})  "
            f"ROI: {fmt_ci(bt_roi['roi_ci'], 2)}"
        )
        log(f"           CLV net of drift: {fmt_ci(clv_net, 5)}")
        log(f"  closing: n_flagged={len(cl_flagged)}  ROI: {fmt_ci(cl_roi['roi_ci'], 2)}")

        sweep.append({
            "threshold": thr,
            "bettime": {"n_flagged": len(bt_flagged), "n_over": n_over_bt, "n_under": n_under_bt,
                        "roi": bt_roi, "clv_raw": clv_raw, "clv_net_of_drift": clv_net},
            "closing": {"n_flagged": len(cl_flagged), "roi": cl_roi},
        })

    eligible = [s for s in sweep if s["bettime"]["roi"]["n_bets"] >= MIN_FLAGGED_BETTIME_BETS_FOR_LOCK]
    confirmed = [
        s for s in eligible
        if s["bettime"]["clv_net_of_drift"]["lower"] is not None and s["bettime"]["clv_net_of_drift"]["lower"] > 0
    ]

    if confirmed:
        locked = min(confirmed, key=lambda s: s["threshold"])
        rationale = (
            f"Smallest threshold ({locked['threshold']:.2f}) among those with >= "
            f"{MIN_FLAGGED_BETTIME_BETS_FOR_LOCK} graded bettime bets whose probability-CLV-net-of-drift "
            "cluster 95% CI sits entirely above zero -- the flagged-outlier-converges-to-consensus "
            "mechanism is confirmed on development data at this threshold, and the smallest such "
            "threshold maximizes deployable coverage."
        )
        status = "confirmed_on_development"
    elif eligible:
        locked = max(eligible, key=lambda s: (s["bettime"]["clv_net_of_drift"]["mean"] or -999))
        rationale = (
            f"No threshold's net-CLV CI cleared zero with >= {MIN_FLAGGED_BETTIME_BETS_FOR_LOCK} bets. "
            f"Fell back to the threshold ({locked['threshold']:.2f}) maximizing mean net CLV among "
            "eligible thresholds. UNRESOLVED / statistically weak -- reported plainly."
        )
        status = "unresolved"
    else:
        locked = max(sweep, key=lambda s: s["bettime"]["roi"]["n_bets"])
        rationale = (
            f"No threshold produced >= {MIN_FLAGGED_BETTIME_BETS_FOR_LOCK} graded bettime bets. Fell back "
            f"to the most-flagging threshold ({locked['threshold']:.2f}). INSUFFICIENT SAMPLE -- not a "
            "usable policy lock."
        )
        status = "insufficient_sample"

    log(f"\nLOCKED threshold: {locked['threshold']:.2f}  status={status}")
    log(f"Rationale: {rationale}")
    return sweep, locked, f"{status}: {rationale}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    logger = Logger()
    log = logger.log
    t0 = datetime.now()

    log("=" * 80)
    log("Component G: cross-line outlier pricing -- DEVELOPMENT PHASE (2023-24 ONLY)")
    log("Pre-registration: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md sections 1 and 8")
    log("=" * 80)
    log(f"Season fence: ALLOWED_DEV_SEASON_LABELS={ALLOWED_DEV_SEASON_LABELS}. This run touches ONLY "
        "2023-24 (plus 2022-23-and-earlier outcomes for prior shape fitting). 2024-25/2025-26 are "
        "never loaded, filtered to, counted, or described. The 2024-25 closing-pass single touch "
        "is a separate later run, after lead-reviewer approval of the policy locked here.")

    for p in (SNAPSHOTS_PATH, CLEAN_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    # -- Load, season-fenced ------------------------------------------------
    snaps = load_snapshots_for_dev_season(DEV_SEASON_LABEL, log)
    prior_df = load_prior_shape_data(log)
    dev_clean = load_dev_season_clean(DEV_SEASON_LABEL, log)
    game_lookup = build_game_lookup(dev_clean)

    # -- Per-book-quote tables, with outcomes attached (kept, not dropped, so
    # unmatched rows still serve as LOO peers) -------------------------------
    log("\n" + "=" * 80)
    log("BUILDING PER-BOOK-QUOTE TABLES")
    log("=" * 80)
    quotes = {}
    for pass_name in ("bettime", "closing"):
        cleaned = clean_pass(snaps, pass_name, log)
        qt = build_book_quote_table(cleaned, log, pass_name)
        qt = attach_game_id_and_saves(qt, game_lookup, log, f"{pass_name} (full table)", drop_unmatched=False)
        quotes[pass_name] = qt

    candidate_pool = report_candidate_pool(quotes, log)

    # -- Dispersion candidates: prior-season (2022-23), 2023-24-dev proxy fit,
    # and 2023-24-dev outcome-Brier-grid-calibrated -- all outcome-grounded,
    # none from the frozen production artifact (pre-reg 8.6) -----------------
    shape_params = fit_prior_shape_params(prior_df, log)
    dev_shape_params = fit_dev_season_shape_params(dev_clean, DEV_CALIB_SPLIT_DATE, log)
    dist = SavesDistribution(CAP_N)
    outcome_calib = calibrate_dispersion_on_outcomes(quotes, dist, log)

    candidates = [
        make_nb2_shape(dist, shape_params["nb2_alpha_proxy"], "nb2_prior_outcomes",
                       "moment-matched on 2022-23-and-earlier outcomes with saves_rolling_10 mu proxy"),
        make_normal_shape(shape_params["normal_sigma_proxy"], "normal_prior_outcomes",
                          "residual std on 2022-23-and-earlier outcomes with saves_rolling_10 mu proxy"),
        make_nb2_shape(dist, dev_shape_params["nb2_alpha_dev"], "nb2_dev_proxy",
                       f"moment-matched on 2023-24 first-half (< {DEV_CALIB_SPLIT_DATE}) outcomes, saves_rolling_10 mu proxy"),
        make_normal_shape(dev_shape_params["normal_sigma_dev"], "normal_dev_proxy",
                          f"residual std on 2023-24 first-half (< {DEV_CALIB_SPLIT_DATE}) outcomes, saves_rolling_10 mu proxy"),
        make_nb2_shape(dist, outcome_calib["best_alpha"], "nb2_outcome_calibrated",
                       f"grid-searched minimizing outcome Brier on 2023-24 first-half (< {DEV_CALIB_SPLIT_DATE})"),
        make_normal_shape(outcome_calib["best_sigma"], "normal_outcome_calibrated",
                          f"grid-searched minimizing outcome Brier on 2023-24 first-half (< {DEV_CALIB_SPLIT_DATE})"),
    ]
    log("\nPre-declared translation candidates: " + ", ".join(f"{c.name} ({c.param_desc})" for c in candidates))
    log("Alpha provenance note (pre-reg 8.6): NO dispersion parameter comes from the frozen "
        "production artifact. Experiment 3's validation-fitted alpha was not yet available, so the "
        "pre-reg fallback branch (fit on 2023-24 development / prior-outcome data only) is used.")

    # -- Lock translation method (eval half of development season) -----------
    locked_shape, shape_comparison = lock_translation_shape(quotes, candidates, log)

    # -- Attach LOO mu + translated fair probabilities under locked shape ----
    log("\n" + "=" * 80)
    log(f"ATTACHING TRANSLATED FAIR PROBABILITIES UNDER LOCKED SHAPE: {locked_shape.name}")
    log("=" * 80)
    if "dev_proxy" in locked_shape.name or "outcome_calibrated" in locked_shape.name:
        log("NOTE: the locked shape's dispersion was fit on the season's first half; full-season "
            "flagging below therefore includes the calibration half in-sample for the shape. The "
            "genuinely out-of-sample check is the later 2024-25 touch.")
    for pass_name in quotes:
        with_loo = compute_mu_and_loo(quotes[pass_name], locked_shape, log, pass_name)
        quotes[pass_name] = attach_translation_and_gap(with_loo, locked_shape, log, pass_name)

    # -- Closing consensus + mandatory drift baseline -------------------------
    closing_consensus = build_closing_consensus_by_line(quotes["closing"], log)
    drift_baseline = compute_drift_baseline(quotes["bettime"], closing_consensus, log)

    # -- Threshold sweep + lock -----------------------------------------------
    sweep, locked, lock_rationale = run_threshold_sweep(
        quotes["bettime"], quotes["closing"], closing_consensus, drift_baseline, log
    )
    locked_threshold = locked["threshold"]

    # -- Final grading at the locked threshold --------------------------------
    log("\n" + "=" * 80)
    log(f"FINAL GRADING AT LOCKED THRESHOLD={locked_threshold:.2f} (shape={locked_shape.name})")
    log("=" * 80)
    bt_final = attach_profit(filter_graded(flag_bets(quotes["bettime"], locked_threshold), log, "bettime FINAL"))
    cl_final = attach_profit(filter_graded(flag_bets(quotes["closing"], locked_threshold), log, "closing FINAL"))
    bt_final = attach_clv(bt_final, closing_consensus, drift_baseline, log)

    bt_breakdowns = report_breakdowns(bt_final, log, "bettime FINAL")
    cl_breakdowns = report_breakdowns(cl_final, log, "closing FINAL")

    log("\n--- [bettime FINAL] Temporal CLV toward close (pre-reg 8.3 development diagnostic), "
        "raw and net of drift, by side ---")
    clv_breakdown = {}
    for side in ("OVER", "UNDER"):
        g = bt_final[bt_final["bet_side"] == side]
        raw = cluster_bootstrap_mean_ci(g["clv_prob"].values, g["cluster_id"].values)
        net = cluster_bootstrap_mean_ci(g["clv_prob_net_of_drift"].values, g["cluster_id"].values)
        log(f"  {side}: raw CLV: {fmt_ci(raw, 5)}")
        log(f"  {side}: net CLV: {fmt_ci(net, 5)}")
        clv_breakdown[side] = {"raw": raw, "net_of_drift": net}
    raw_all = cluster_bootstrap_mean_ci(bt_final["clv_prob"].values, bt_final["cluster_id"].values)
    net_all = cluster_bootstrap_mean_ci(bt_final["clv_prob_net_of_drift"].values, bt_final["cluster_id"].values)
    log(f"  OVERALL: raw CLV: {fmt_ci(raw_all, 5)}")
    log(f"  OVERALL: net CLV: {fmt_ci(net_all, 5)}")
    clv_breakdown["overall"] = {"raw": raw_all, "net_of_drift": net_all}

    # -- CLV by book (is the movement all one book converging?) ---------------
    log("\n--- [bettime FINAL] CLV net of drift, by book ---")
    clv_by_book = {}
    if len(bt_final):
        for book, g in bt_final.groupby("book"):
            net = cluster_bootstrap_mean_ci(g["clv_prob_net_of_drift"].values, g["cluster_id"].values)
            log(f"  {book}: {fmt_ci(net, 5)}")
            clv_by_book[book] = net

    # -- Persist outputs -------------------------------------------------------
    ts = t0.strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "models" / "trained" / f"experiment_cross_line_pricing_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)

    bt_out = bt_final.copy()
    bt_out["snapshot_pass"] = "bettime"
    cl_out = cl_final.copy()
    cl_out["snapshot_pass"] = "closing"
    for col in ("clv_prob", "clv_prob_net_of_drift", "closing_consensus_prob_chosen_side",
                "bettime_devig_prob_chosen_side", "consensus_prob_over", "consensus_prob_under",
                "n_closing_books"):
        if col not in cl_out.columns:
            cl_out[col] = np.nan
    common_cols = [c for c in bt_out.columns if c in cl_out.columns]
    flagged_all = pd.concat([bt_out[common_cols], cl_out[common_cols]], ignore_index=True)
    flagged_path = out_dir / "flagged_bets.parquet"
    flagged_all.to_parquet(flagged_path, index=False)
    log(f"\nSaved {len(flagged_all)} flagged bets (locked threshold={locked_threshold:.2f}, both passes) to {flagged_path}")

    metadata = {
        "timestamp": ts,
        "wall_clock_seconds": (datetime.now() - t0).total_seconds(),
        "preregistration": {
            "document": "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md",
            "sections": ["1 (shared conventions)", "8 (Experiment 7)"],
            "alignment": {
                "8.3_temporal_clv_diagnostic": "Produced below as a development-stage deliverable, "
                                                "net of the 2023-24 unconditional drift baseline.",
                "8.6_alpha_provenance": "No dispersion parameter from the frozen production artifact; "
                                         "candidates fit on prior-season outcomes and 2023-24 first-half "
                                         "outcome data only (pre-reg fallback branch: 'a value fit on "
                                         "2023-24 training/validation data only').",
                "8.4_book_concentration_and_venue_split": "Both reported in final_grading breakdowns.",
                "8.5_gate": "NOT scored here -- belongs to the later 2024-25 closing-pass single touch. "
                             "Its venue-accessible ROI cut is empty in 2023-24 (no BetOnline rows).",
            },
        },
        "season_fence": {
            "development_season": DEV_SEASON_LABEL,
            "allowed_dev_season_labels": sorted(ALLOWED_DEV_SEASON_LABELS),
            "touched_2024_25_or_2025_26": False,
            "note": "This script never loaded, filtered to, counted, or described any row outside "
                    "season=2023-24 (shape fitting additionally used 2022-23-and-earlier outcomes, "
                    "an even earlier window). The single 2024-25 closing-pass confirmatory touch is "
                    "NOT run by this script and awaits lead-reviewer approval of this locked policy.",
        },
        "data_paths": {"snapshots": str(SNAPSHOTS_PATH), "clean_training_data": str(CLEAN_PATH)},
        "candidate_pool": candidate_pool,
        "prior_shape_fit": shape_params,
        "dev_season_proxy_shape_fit": dev_shape_params,
        "outcome_calibrated_dispersion": outcome_calib,
        "shape_comparison": shape_comparison,
        "locked_policy": {
            "translation_shape": locked_shape.name,
            "translation_family": locked_shape.family,
            "shape_param": locked_shape.param_desc,
            "shape_param_value": locked_shape.param_value,
            "shape_param_provenance": locked_shape.param_provenance,
            "gap_threshold": locked_threshold,
            "gap_threshold_grid": GAP_THRESHOLD_GRID,
            "lock_rationale": lock_rationale,
            "min_flagged_bettime_bets_for_lock": MIN_FLAGGED_BETTIME_BETS_FOR_LOCK,
            "gap_convention": "fair_prob (shape-translated LOO cross-book consensus) minus RAW "
                              "vig-inclusive implied prob of the book's own price, American-rounded "
                              "(harness.decide_bet convention); a gap of x means +x EV net of vig.",
        },
        "threshold_sweep": sweep,
        "drift_baseline": drift_baseline,
        "final_grading": {
            "bettime": bt_breakdowns,
            "closing": cl_breakdowns,
            "bettime_temporal_clv": clv_breakdown,
            "bettime_clv_net_by_book": clv_by_book,
        },
        "executable_books": sorted(EXECUTABLE_BOOKS),
        "open_methodological_choices": [
            "Consensus construction: book-level leave-one-out in mu-space (invert each book's own "
            "quote to an implied distribution mean under the locked shape, average the OTHER books' "
            "means, translate back to the target book's own line). Chosen over a "
            "per-line-consensus-then-reference-line approach for symmetry and to avoid an arbitrary "
            "reference-line pick; verified that no book posts >1 line per goalie-night after dedup.",
            "Dispersion candidates (6 total, 3 provenances x 2 families): (1) prior-outcome proxy fits "
            "(2022-23-and-earlier, saves_rolling_10 mu proxy), (2) 2023-24-first-half proxy fits (same "
            "method, pre-reg 8.6 fallback branch), (3) 2023-24-first-half grid search minimizing "
            "OUTCOME Brier score. Shape family + dispersion LOCKED on the season's SECOND half using "
            "outcome Brier (fair probability at each book's own line vs the actual game result), not "
            "development ROI or CLV. An earlier version of this script locked on market "
            "self-consistency (MAE vs off-modal books' own realized prices) and was found to be "
            "circular -- it is minimized by the flattest possible shape, which erases the exact "
            "cross-line mispricing this experiment targets. See module docstring correction note. "
            "Market self-consistency MAE is still computed and reported for the winner, as a "
            "secondary diagnostic only.",
            "Threshold locking metric: probability CLV net of drift (mechanism confirmation), not "
            "development ROI, per plan 6.3's 'ROI remains secondary'. ROI is reported across the "
            "full sweep as the tradeoff curve.",
            "Drift baseline granularity: every individual bettime book quote is one drift "
            "observation (not deduplicated per line); goalie-night cluster bootstrap handles the "
            "non-independence.",
            "CLV is graded only for the bettime pass (against the later closing consensus). "
            "Closing-pass flagged bets get outcomes/ROI only -- no later snapshot exists (pre-reg "
            "8.3's interpretive point).",
            "Integer (X.0) lines exist in 46/33,253 2023-24 rows (0.14%); the Normal family "
            "approximates their push probability as 0, the NB2 family computes it exactly.",
        ],
        "caveats": [
            "BetOnline (betonlineag) has ZERO rows in the 2023-24 snapshot archive; the "
            "venue-accessible cut is empty by construction during development and only becomes "
            "scoreable on the 2024-25 touch. Underdog/PrizePicks are absent from this archive "
            "entirely (not sportsbook feeds).",
            "Development-phase-only numbers: threshold and shape were locked on 2023-24, so every "
            "ROI/CLV below is hypothesis support (plan 6.1 tier 1), not confirmed edge. No 2024-25 "
            "or 2025-26 row was touched in any form.",
            "The bettime pass sits ~15min-5.2h before puck drop (plan 3.1) -- late-window quotes, "
            "not true openers; 'stale' here means stale within that late window.",
        ],
    }

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(jsonable(metadata), indent=2), encoding="utf-8")
    log(f"Saved metadata to {metadata_path}")

    run_log_path = out_dir / "run_log.txt"
    logger.write(run_log_path)
    print(f"Saved run log to {run_log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
