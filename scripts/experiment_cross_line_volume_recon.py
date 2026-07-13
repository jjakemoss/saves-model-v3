"""Experiment 10: outcome-blind Component G executable-volume reconnaissance.

Binding registration: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 13.
This script opens one row-bearing input and persists aggregate metadata only.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import gammaln


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "saves_lines_snapshots.parquet"
REGISTERED_SHA256 = "E81CA4EA01B1DFB3F69068782B75E18B4D174BEF82DD34D8E341B59CBC94ED56"
READ_COLUMNS = [
    "event_id",
    "commence_time",
    "game_date_eastern",
    "requested_ts",
    "resolved_ts",
    "snapshot_pass",
    "book",
    "goalie_id",
    "side",
    "line",
    "price_decimal",
]
EXACT_DUPLICATE_KEY = [
    "event_id",
    "requested_ts",
    "book",
    "goalie_id",
    "side",
    "line",
    "price_decimal",
]
PAIR_KEY = [
    "event_id",
    "goalie_id",
    "book",
    "line",
    "commence_time",
    "game_date_eastern",
]
QUOTE_SIDE_KEY = PAIR_KEY + ["side"]
NIGHT_KEY = ["event_id", "goalie_id"]
TARGET_BOOK = "betonlineag"
SEASON_START = "2025-08-01"
SEASON_END = "2026-07-31"
SNAPSHOT_PASS = "bettime"
NB2_ALPHA = 0.100000
SUPPORT_CAP = 70
THRESHOLDS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
MIN_ARITHMETIC_COUNT = 20
FORBIDDEN_LOADED_COLUMNS = {
    "saves",
    "shots_against",
    "goals_against",
    "toi",
    "result",
    "grade",
    "outcome",
    "profit",
    "roi",
    "clv",
}


class Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str = "") -> None:
        print(message)
        self.lines.append(str(message))

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


class NB2Translator:
    """Experiment 7's capped NB2 pricing and inversion conventions."""

    def __init__(self, alpha: float = NB2_ALPHA, cap: int = SUPPORT_CAP) -> None:
        self.alpha = alpha
        self.cap = cap
        self.n_arr = np.arange(cap + 1, dtype=np.float64)
        self.cache: dict[tuple[float, float], float] = {}

    def pmf(self, mu: float) -> np.ndarray:
        mu = max(mu, 1e-6)
        n = self.n_arr
        r = 1.0 / self.alpha
        p = r / (r + mu)
        logpmf = (
            gammaln(n + r)
            - gammaln(r)
            - gammaln(n + 1)
            + r * np.log(p)
            + n * np.log(1 - p)
        )
        return np.exp(logpmf)

    def price_line(self, mu: float, line: float) -> tuple[float, float, float]:
        pmf = self.pmf(mu)
        p_over = float(pmf[self.n_arr > line].sum())
        p_under = float(pmf[self.n_arr < line].sum())
        p_push = float(pmf[self.n_arr == line].sum())
        return p_over, p_under, p_push

    def invert_mu(self, line: float, p_target: float) -> float:
        p_target = min(max(p_target, 0.002), 0.998)
        key = (round(float(line), 2), round(float(p_target), 8))
        if key in self.cache:
            return self.cache[key]

        def objective(mu: float) -> float:
            return self.price_line(mu, line)[0] - p_target

        lo, hi = 0.1, float(self.cap) - 0.1
        if objective(lo) > 0 or objective(hi) < 0:
            raise ValueError(f"NB2 inversion bracket failed for line={line}, p_target={p_target}")
        value = brentq(objective, lo, hi, xtol=1e-6, rtol=1e-8, maxiter=100)
        self.cache[key] = value
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def devig_pair_decimal(price_over: float, price_under: float) -> tuple[float, float]:
    p_over = 1.0 / price_over
    p_under = 1.0 / price_under
    total = p_over + p_under
    if total <= 0:
        raise AssertionError("Paired quote has a non-positive implied-probability total")
    return p_over / total, p_under / total


def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def american_implied_probability(odds: int) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def select_side(gap_over: float, gap_under: float, threshold: float) -> str | None:
    if np.isnan(gap_over) or np.isnan(gap_under):
        return None
    if gap_over >= threshold and gap_over > gap_under:
        return "OVER"
    if gap_under >= threshold:
        return "UNDER"
    return None


def int_dict(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.items()}


def numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {"n": 0, "min": None, "q25": None, "median": None, "mean": None, "q75": None, "max": None}
    return {
        "n": int(len(clean)),
        "min": float(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "q75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def main() -> int:
    started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger()
    log = logger.log

    log("=" * 80)
    log("EXPERIMENT 10: COMPONENT G EXECUTABLE-VOLUME RECONNAISSANCE")
    log("Binding registration: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 13")
    log("=" * 80)

    assert INPUT_PATH.resolve() == (REPO_ROOT / "data" / "processed" / "saves_lines_snapshots.parquet").resolve()
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Missing registered input: {INPUT_PATH}")
    input_hash = sha256_file(INPUT_PATH)
    assert input_hash == REGISTERED_SHA256, (
        f"Input SHA-256 mismatch: expected {REGISTERED_SHA256}, got {input_hash}"
    )
    log(f"Registered input: {INPUT_PATH.resolve()}")
    log(f"SHA-256: {input_hash} (MATCH)")
    log(f"READ_COLUMNS ({len(READ_COLUMNS)}): {READ_COLUMNS}")

    # Binding row read: exactly the registered columns from the sole row-bearing input.
    raw = pd.read_parquet(INPUT_PATH, columns=READ_COLUMNS)
    loaded_columns_exact = list(raw.columns) == READ_COLUMNS
    assert loaded_columns_exact
    loaded_lower = {str(column).lower() for column in raw.columns}
    assert loaded_lower.isdisjoint(FORBIDDEN_LOADED_COLUMNS)
    n_loaded = len(raw)

    date_text = raw["game_date_eastern"].astype("string")
    population_mask = (
        date_text.between(SEASON_START, SEASON_END, inclusive="both")
        & raw["snapshot_pass"].eq(SNAPSHOT_PASS)
    )
    population = raw.loc[population_mask].copy()
    del raw
    assert population["snapshot_pass"].eq(SNAPSHOT_PASS).all()
    assert population["game_date_eastern"].astype("string").between(
        SEASON_START, SEASON_END, inclusive="both"
    ).all()
    n_population = len(population)
    missing_goalie_population = int(population["goalie_id"].isna().sum())
    critical_population_columns = [
        "event_id",
        "commence_time",
        "game_date_eastern",
        "requested_ts",
        "book",
        "side",
        "line",
        "price_decimal",
    ]
    missing_critical_values = {
        column: int(population[column].isna().sum()) for column in critical_population_columns
    }
    assert not any(missing_critical_values.values()), (
        f"Missing values in critical population columns: {missing_critical_values}"
    )
    unmerged_alias_rows = int(population["book"].eq("betonline").sum())
    assert unmerged_alias_rows == 0, (
        "The separate betonline alias appeared in the registered snapshot input; refusing to "
        "merge it with or use it alongside canonical betonlineag"
    )

    earliest_ts = population.groupby("event_id", dropna=False)["requested_ts"].transform("min")
    earliest = population.loc[population["requested_ts"].eq(earliest_ts)].copy()
    n_later_snapshot_rows_removed = n_population - len(earliest)
    assert earliest.groupby("event_id", dropna=False)["requested_ts"].nunique(dropna=False).le(1).all()

    n_before_exact_dedup = len(earliest)
    deduped = earliest.drop_duplicates(subset=EXACT_DUPLICATE_KEY, keep="first").copy()
    n_exact_duplicates_removed = n_before_exact_dedup - len(deduped)
    assert not deduped.duplicated(subset=EXACT_DUPLICATE_KEY, keep=False).any()
    missing_goalie_after_cleanup = int(deduped["goalie_id"].isna().sum())

    side_values = set(deduped["side"].dropna().astype(str).unique())
    assert side_values.issubset({"Over", "Under"}), f"Unexpected side labels: {sorted(side_values)}"
    conflict_counts = deduped.groupby(QUOTE_SIDE_KEY, dropna=False)["price_decimal"].nunique(dropna=False)
    n_conflicting_quote_sides = int((conflict_counts > 1).sum())
    assert n_conflicting_quote_sides == 0, (
        f"Found {n_conflicting_quote_sides} quote-key/side groups with conflicting prices"
    )

    pairable = deduped.loc[deduped["goalie_id"].notna()].copy()
    pairable["goalie_id"] = pairable["goalie_id"].astype("int64")
    duplicate_quote_sides = pairable.duplicated(subset=QUOTE_SIDE_KEY, keep=False)
    assert not duplicate_quote_sides.any(), "Duplicate quote-key/side rows remain after exact deduplication"

    wide = pairable.pivot(index=PAIR_KEY, columns="side", values="price_decimal").reset_index()
    wide.columns.name = None
    for side in ("Over", "Under"):
        if side not in wide.columns:
            wide[side] = np.nan
    n_quote_groups_before_both_sides = len(wide)
    paired = wide.dropna(subset=["Over", "Under"]).rename(
        columns={"Over": "price_decimal_over", "Under": "price_decimal_under"}
    ).copy()
    n_unpaired_quote_groups = n_quote_groups_before_both_sides - len(paired)
    prices = paired[["price_decimal_over", "price_decimal_under"]].astype(float)
    assert np.isfinite(prices.to_numpy()).all()
    assert (prices.to_numpy() > 1.0).all(), "Decimal prices must be finite and greater than 1"

    multi_line = paired.groupby(NIGHT_KEY + ["book"])["line"].nunique(dropna=False)
    n_multi_line_book_nights = int((multi_line > 1).sum())
    assert n_multi_line_book_nights == 0, (
        f"Found {n_multi_line_book_nights} (event, goalie, book) groups with multiple paired lines"
    )

    devigged = [
        devig_pair_decimal(over, under)
        for over, under in zip(paired["price_decimal_over"], paired["price_decimal_under"])
    ]
    paired["devig_prob_over"] = [item[0] for item in devigged]
    paired["devig_prob_under"] = [item[1] for item in devigged]

    frequency = (
        paired.groupby(NIGHT_KEY + ["line"], dropna=False)["book"]
        .nunique()
        .rename("distinct_book_count")
        .reset_index()
    )
    frequency["max_distinct_book_count"] = frequency.groupby(NIGHT_KEY)["distinct_book_count"].transform("max")
    modal_lines = frequency.loc[
        frequency["distinct_book_count"].eq(frequency["max_distinct_book_count"]),
        NIGHT_KEY + ["line"],
    ].copy()
    modal_lines["is_modal_line"] = True
    paired = paired.merge(modal_lines, on=NIGHT_KEY + ["line"], how="left", validate="many_to_one")
    paired["is_modal_line"] = paired["is_modal_line"].eq(True)

    night_context = paired.groupby(NIGHT_KEY).agg(
        n_books=("book", "nunique"),
        min_line=("line", "min"),
        max_line=("line", "max"),
    ).reset_index()
    night_context["line_spread"] = night_context["max_line"] - night_context["min_line"]
    modal_counts = modal_lines.groupby(NIGHT_KEY).size().rename("n_modal_lines").reset_index()
    night_context = night_context.merge(modal_counts, on=NIGHT_KEY, how="left", validate="one_to_one")
    paired = paired.merge(
        night_context[NIGHT_KEY + ["n_books", "line_spread", "n_modal_lines"]],
        on=NIGHT_KEY,
        how="left",
        validate="many_to_one",
    )

    translator = NB2Translator()
    implied_mu = np.full(len(paired), np.nan)
    inversion_failures = 0
    for position, (line, probability) in enumerate(zip(paired["line"], paired["devig_prob_over"])):
        try:
            implied_mu[position] = translator.invert_mu(float(line), float(probability))
        except ValueError:
            inversion_failures += 1
    paired["implied_mu"] = implied_mu

    valid_mu = paired.loc[paired["implied_mu"].notna(), NIGHT_KEY + ["book", "implied_mu"]].copy()
    peer_aggregates = valid_mu.groupby(NIGHT_KEY).agg(
        valid_mu_sum=("implied_mu", "sum"),
        valid_mu_count=("implied_mu", "size"),
    ).reset_index()
    paired = paired.merge(peer_aggregates, on=NIGHT_KEY, how="left", validate="many_to_one")
    own_valid = paired["implied_mu"].notna().astype(int)
    paired["peer_mu_count"] = paired["valid_mu_count"].fillna(0).astype(int) - own_valid
    peer_sum = paired["valid_mu_sum"].fillna(0.0) - paired["implied_mu"].fillna(0.0)
    paired["loo_mu"] = np.where(paired["peer_mu_count"].ge(1), peer_sum / paired["peer_mu_count"], np.nan)

    target = paired.loc[paired["book"].eq(TARGET_BOOK)].copy()
    assert not target.duplicated(subset=NIGHT_KEY, keep=False).any()
    target["non_target_peer_count"] = target["peer_mu_count"].astype(int)
    assert target["non_target_peer_count"].le(target["n_books"] - 1).all()
    target["strictly_off_modal"] = ~target["is_modal_line"]
    target["peer_eligible"] = target["non_target_peer_count"].ge(1) & target["loo_mu"].notna()

    fair_over = np.full(len(target), np.nan)
    fair_under = np.full(len(target), np.nan)
    fair_push = np.full(len(target), np.nan)
    for position, (eligible, mu, line) in enumerate(
        zip(target["peer_eligible"], target["loo_mu"], target["line"])
    ):
        if eligible:
            fair_over[position], fair_under[position], fair_push[position] = translator.price_line(
                float(mu), float(line)
            )
    target["fair_prob_over"] = fair_over
    target["fair_prob_under"] = fair_under
    target["fair_prob_push"] = fair_push
    target["gap_over"] = [
        fair - american_implied_probability(decimal_to_american(float(price))) if not np.isnan(fair) else np.nan
        for fair, price in zip(target["fair_prob_over"], target["price_decimal_over"])
    ]
    target["gap_under"] = [
        fair - american_implied_probability(decimal_to_american(float(price))) if not np.isnan(fair) else np.nan
        for fair, price in zip(target["fair_prob_under"], target["price_decimal_under"])
    ]

    eligible_target = target.loc[target["strictly_off_modal"] & target["peer_eligible"]].copy()
    threshold_counts: dict[str, dict[str, int]] = {}
    threshold_month_counts: dict[str, dict[str, dict[str, int]]] = {}
    selected_at_primary = pd.DataFrame()
    for threshold in THRESHOLDS:
        selected_side = [
            select_side(gap_over, gap_under, threshold)
            for gap_over, gap_under in zip(eligible_target["gap_over"], eligible_target["gap_under"])
        ]
        selected = eligible_target.assign(selected_side=selected_side)
        selected = selected.loc[selected["selected_side"].notna()].copy()
        assert not selected.duplicated(subset=NIGHT_KEY, keep=False).any()
        counts = selected["selected_side"].value_counts()
        threshold_counts[f"{threshold:.2f}"] = {
            "total": int(len(selected)),
            "OVER": int(counts.get("OVER", 0)),
            "UNDER": int(counts.get("UNDER", 0)),
        }
        selected_month = selected["game_date_eastern"].astype("string").str.slice(0, 7)
        month_side_counts = selected.assign(month=selected_month).groupby(
            ["month", "selected_side"]
        ).size()
        threshold_month_counts[f"{threshold:.2f}"] = {
            str(month): {
                "total": int(month_side_counts.loc[month].sum()),
                "OVER": int(month_side_counts.get((month, "OVER"), 0)),
                "UNDER": int(month_side_counts.get((month, "UNDER"), 0)),
            }
            for month in sorted(selected_month.dropna().unique().tolist())
        }
        if math.isclose(threshold, 0.02):
            selected_at_primary = selected

    primary_count = int(len(selected_at_primary))
    verdict = "ARITHMETICALLY_FEASIBLE" if primary_count >= MIN_ARITHMETIC_COUNT else "TOO_SPARSE"
    assert verdict in {"ARITHMETICALLY_FEASIBLE", "TOO_SPARSE"}

    target["month"] = target["game_date_eastern"].astype("string").str.slice(0, 7)
    selected_at_primary = selected_at_primary.copy()
    selected_at_primary["month"] = selected_at_primary["game_date_eastern"].astype("string").str.slice(0, 7)
    all_months = sorted(target["month"].dropna().unique().tolist())
    monthly_counts: dict[str, dict[str, int]] = {}
    for month in all_months:
        month_target = target.loc[target["month"].eq(month)]
        month_primary = selected_at_primary.loc[selected_at_primary["month"].eq(month)]
        side_counts = month_primary["selected_side"].value_counts()
        monthly_counts[str(month)] = {
            "paired_target_quotes": int(len(month_target)),
            "strictly_off_modal": int(month_target["strictly_off_modal"].sum()),
            "strictly_off_modal_with_peer": int(
                (month_target["strictly_off_modal"] & month_target["peer_eligible"]).sum()
            ),
            "threshold_0.02_total": int(len(month_primary)),
            "threshold_0.02_OVER": int(side_counts.get("OVER", 0)),
            "threshold_0.02_UNDER": int(side_counts.get("UNDER", 0)),
            "thresholds": {
                threshold: counts_by_month.get(str(month), {"total": 0, "OVER": 0, "UNDER": 0})
                for threshold, counts_by_month in threshold_month_counts.items()
            },
        }

    full_peer_hist = int_dict(target["non_target_peer_count"].value_counts().sort_index())
    off_modal_peer_hist = int_dict(
        target.loc[target["strictly_off_modal"], "non_target_peer_count"].value_counts().sort_index()
    )
    peer_presence = valid_mu.loc[valid_mu["book"].ne(TARGET_BOOK)].merge(
        target[NIGHT_KEY + ["strictly_off_modal"]],
        on=NIGHT_KEY,
        how="inner",
        validate="many_to_one",
    )
    primary_peer_presence = valid_mu.loc[valid_mu["book"].ne(TARGET_BOOK)].merge(
        selected_at_primary[NIGHT_KEY],
        on=NIGHT_KEY,
        how="inner",
        validate="many_to_one",
    )
    peer_summary = {
        "paired_target_quotes_histogram": full_peer_hist,
        "strictly_off_modal_histogram": off_modal_peer_hist,
        "paired_target_nights_valid_peer_presence_by_book": int_dict(
            peer_presence["book"].value_counts().sort_index()
        ),
        "strictly_off_modal_valid_peer_presence_by_book": int_dict(
            peer_presence.loc[peer_presence["strictly_off_modal"], "book"].value_counts().sort_index()
        ),
        "threshold_0.02_valid_peer_presence_by_book": int_dict(
            primary_peer_presence["book"].value_counts().sort_index()
        ),
        "paired_target_quotes_summary": numeric_summary(target["non_target_peer_count"]),
        "strictly_off_modal_summary": numeric_summary(
            target.loc[target["strictly_off_modal"], "non_target_peer_count"]
        ),
        "threshold_0.02_summary": numeric_summary(selected_at_primary["non_target_peer_count"]),
    }
    line_spread_summary = {
        "paired_target_quotes": numeric_summary(target["line_spread"]),
        "strictly_off_modal": numeric_summary(target.loc[target["strictly_off_modal"], "line_spread"]),
        "strictly_off_modal_with_peer": numeric_summary(
            target.loc[target["strictly_off_modal"] & target["peer_eligible"], "line_spread"]
        ),
        "threshold_0.02": numeric_summary(selected_at_primary["line_spread"]),
    }

    n_modal_tie_nights = int((night_context["n_modal_lines"] > 1).sum())
    target_modal_tie_nights = int((target["n_modal_lines"] > 1).sum())
    qa = {
        "registered_input_path_exact": True,
        "registered_sha256_exact": input_hash == REGISTERED_SHA256,
        "loaded_columns_exact": loaded_columns_exact,
        "forbidden_columns_absent": loaded_lower.isdisjoint(FORBIDDEN_LOADED_COLUMNS),
        "season_fence_exact": True,
        "bettime_only": True,
        "earliest_requested_ts_per_event": True,
        "exact_duplicate_key_clear": True,
        "conflicting_quote_side_prices": n_conflicting_quote_sides,
        "unexpected_side_labels": sorted(side_values - {"Over", "Under"}),
        "missing_critical_population_values": missing_critical_values,
        "unmerged_betonline_alias_rows": unmerged_alias_rows,
        "multi_line_paired_book_nights": n_multi_line_book_nights,
        "target_duplicate_goalie_nights": int(target.duplicated(subset=NIGHT_KEY).sum()),
        "modal_set_includes_all_tied_maxima": True,
        "target_line_outside_entire_modal_set_required": True,
        "non_target_peer_required": True,
        "line_spread_minimum_applied": False,
        "candidate_rows_persisted": False,
        "threshold_selected_from_counts": False,
    }

    denominators = {
        "loaded_rows": n_loaded,
        "season_and_bettime_rows": n_population,
        "later_snapshot_rows_removed": n_later_snapshot_rows_removed,
        "earliest_snapshot_rows_before_exact_dedup": n_before_exact_dedup,
        "exact_duplicates_removed": n_exact_duplicates_removed,
        "rows_after_exact_dedup": int(len(deduped)),
        "missing_goalie_rows_in_population": missing_goalie_population,
        "missing_goalie_rows_after_cleanup": missing_goalie_after_cleanup,
        "unmerged_betonline_alias_rows": unmerged_alias_rows,
        "quote_groups_before_both_sides_requirement": n_quote_groups_before_both_sides,
        "unpaired_quote_groups": n_unpaired_quote_groups,
        "all_paired_book_quotes": int(len(paired)),
        "all_paired_goalie_nights": int(len(night_context)),
        "target_paired_quotes": int(len(target)),
        "target_paired_goalie_nights": int(target[NIGHT_KEY].drop_duplicates().shape[0]),
        "target_strictly_off_modal": int(target["strictly_off_modal"].sum()),
        "target_strictly_off_modal_with_non_target_peer": int(
            (target["strictly_off_modal"] & target["peer_eligible"]).sum()
        ),
        "all_mu_inversion_failures": int(inversion_failures),
        "all_modal_tie_goalie_nights": n_modal_tie_nights,
        "target_modal_tie_goalie_nights": target_modal_tie_nights,
    }

    elapsed = time.perf_counter() - started
    metadata = {
        "experiment": "Experiment 10 - Component G executable-volume reconnaissance",
        "registration": "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 13",
        "timestamp": timestamp,
        "wall_clock_seconds": elapsed,
        "primary_verdict": verdict,
        "primary_gate": {
            "threshold": 0.02,
            "minimum_count": MIN_ARITHMETIC_COUNT,
            "unique_strictly_off_modal_target_goalie_nights": primary_count,
        },
        "input": {
            "absolute_path": str(INPUT_PATH.resolve()),
            "repo_relative_path": "data/processed/saves_lines_snapshots.parquet",
            "sha256": input_hash,
            "registered_sha256": REGISTERED_SHA256,
            "read_columns": READ_COLUMNS,
            "read_call_contract": "pd.read_parquet(INPUT_PATH, columns=READ_COLUMNS)",
            "row_bearing_input_count": 1,
        },
        "population": {
            "season_start_inclusive": SEASON_START,
            "season_end_inclusive": SEASON_END,
            "snapshot_pass": SNAPSHOT_PASS,
            "target_book": TARGET_BOOK,
            "target_aliases_merged": False,
        },
        "frozen_scorer": {
            "family": "NB2",
            "alpha": NB2_ALPHA,
            "support_cap": SUPPORT_CAP,
            "pmf_cap_renormalized": False,
            "de_vig": "paired-price additive normalization",
            "inversion": "brentq [0.1, 69.9], target clip [0.002, 0.998], xtol=1e-6, rtol=1e-8",
            "consensus": "leave-one-book-out arithmetic mean of peer implied means",
            "gap": "translated fair probability minus American-rounded raw vig-inclusive implied probability",
            "side_rule": "OVER when gap_over >= t and gap_over > gap_under; otherwise UNDER when gap_under >= t",
            "threshold_grid": THRESHOLDS,
        },
        "outcome_firewall": {
            "no_outcome_columns_loaded": True,
            "forbidden_loaded_columns": sorted(FORBIDDEN_LOADED_COLUMNS),
            "loaded_columns_disjoint_from_forbidden": True,
            "closing_snapshot_loaded": False,
            "other_row_bearing_input_opened": False,
            "candidate_identifiers_persisted": False,
        },
        "denominators": denominators,
        "qa": qa,
        "threshold_counts": threshold_counts,
        "monthly_aggregate_counts": monthly_counts,
        "peer_book_counts": peer_summary,
        "line_spread_summaries": line_spread_summary,
        "persistence": {
            "aggregate_only": True,
            "files": ["metadata.json", "run_log.txt"],
        },
    }

    log("")
    log("OUTCOME FIREWALL")
    log(f"  one row-bearing input: {INPUT_PATH.resolve()}")
    log(f"  exact loaded columns: {READ_COLUMNS}")
    log("  season=2025-08-01..2026-07-31; snapshot_pass=bettime; target=betonlineag")
    log("  no outcome columns loaded; no closing snapshot loaded; aggregate-only persistence")
    log("")
    log("DENOMINATORS AND CLEANUP")
    for key, value in denominators.items():
        log(f"  {key}: {value}")
    log("")
    log("THRESHOLD COUNTS")
    for threshold, counts in threshold_counts.items():
        log(f"  t={threshold}: total={counts['total']} OVER={counts['OVER']} UNDER={counts['UNDER']}")
    log("")
    log("MONTHLY AGGREGATE COUNTS")
    for month, counts in monthly_counts.items():
        log(f"  {month}: {counts}")
    log("")
    log(f"PEER-BOOK COUNTS: {peer_summary}")
    log(f"LINE-SPREAD SUMMARIES: {line_spread_summary}")
    log(f"QA ASSERTIONS: {qa}")
    log("")
    log("=" * 80)
    log(f"PRIMARY VERDICT: {verdict}")
    log(f"t=0.02 unique strictly off-modal {TARGET_BOOK} goalie-nights: {primary_count}")
    log("=" * 80)

    output_dir = REPO_ROOT / "models" / "trained" / f"experiment_cross_line_volume_recon_{timestamp}"
    assert not output_dir.exists(), f"Output directory already exists: {output_dir}"
    output_dir.mkdir(parents=False)
    metadata_path = output_dir / "metadata.json"
    run_log_path = output_dir / "run_log.txt"
    metadata_path.write_text(json.dumps(jsonable(metadata), indent=2) + "\n", encoding="utf-8")
    logger.write(run_log_path)
    log(f"Saved aggregate metadata: {metadata_path}")
    log(f"Saved run log: {run_log_path}")
    # Re-write so the two saved-path lines are included in the persisted log.
    logger.write(run_log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
