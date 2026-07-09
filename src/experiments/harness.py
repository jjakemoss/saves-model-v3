"""Reusable evaluation helpers for honest chronological experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from betting.odds_utils import calculate_ev, calculate_payout


TRAIN_END_EXCLUSIVE = "2025-10-16"
VAL_START = "2025-10-16"
VAL_END_INCLUSIVE = "2025-12-03"
TEST_START = "2025-12-04"
EV_THRESHOLDS = [0.05, 0.10, 0.12, 0.15]


@dataclass(frozen=True)
class DateSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    boundaries: dict


def split_by_date(df: pd.DataFrame, log: Callable[[str], None], label: str) -> DateSplit:
    dates = pd.to_datetime(df["game_date"])
    train_mask = dates < pd.Timestamp(TRAIN_END_EXCLUSIVE)
    val_mask = (dates >= pd.Timestamp(VAL_START)) & (dates <= pd.Timestamp(VAL_END_INCLUSIVE))
    test_mask = dates >= pd.Timestamp(TEST_START)

    assert (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int) <= 1).all(), (
        f"[{label}] A row satisfies more than one fold date condition."
    )
    assert (train_mask | val_mask | test_mask).all(), (
        f"[{label}] A row satisfies none of the fold date conditions."
    )
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == len(df)

    train_dates = set(dates.loc[train_mask])
    val_dates = set(dates.loc[val_mask])
    test_dates = set(dates.loc[test_mask])
    assert train_dates.isdisjoint(val_dates), f"[{label}] Train/val date overlap."
    assert train_dates.isdisjoint(test_dates), f"[{label}] Train/test date overlap."
    assert val_dates.isdisjoint(test_dates), f"[{label}] Val/test date overlap."

    train_idx = np.where(train_mask.values)[0]
    val_idx = np.where(val_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]

    boundaries = {
        "train": {
            "start": str(dates.iloc[train_idx].min().date()),
            "end": str(dates.iloc[train_idx].max().date()),
            "rows": int(len(train_idx)),
        },
        "val": {
            "start": str(dates.iloc[val_idx].min().date()),
            "end": str(dates.iloc[val_idx].max().date()),
            "rows": int(len(val_idx)),
        },
        "test": {
            "start": str(dates.iloc[test_idx].min().date()),
            "end": str(dates.iloc[test_idx].max().date()),
            "rows": int(len(test_idx)),
        },
    }

    log(f"\n[{label}] Fold boundaries (date-based split):")
    for name in ("train", "val", "test"):
        b = boundaries[name]
        log(f"  {name:<5}: {b['start']} to {b['end']}  (n={b['rows']})")
    log(f"  [{label}] Fold date-disjointness verified.")

    return DateSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, boundaries=boundaries)


def decide_bet(
    p_over: float,
    p_under: float,
    odds_over: float,
    odds_under: float,
    ev_threshold: float,
) -> tuple[str | None, float | None]:
    ev_over = calculate_ev(p_over, odds_over)
    ev_under = calculate_ev(p_under, odds_under)
    if ev_over >= ev_threshold and ev_over > ev_under:
        return "OVER", ev_over
    if ev_under >= ev_threshold:
        return "UNDER", ev_under
    return None, None


def grade_bets(
    p_over_arr: np.ndarray,
    p_under_arr: np.ndarray,
    saves_actual: np.ndarray,
    lines: np.ndarray,
    odds_over: np.ndarray,
    odds_under: np.ndarray,
    game_id: np.ndarray,
    goalie_id: np.ndarray,
    ev_threshold: float,
    matched_mask: np.ndarray,
    log: Callable[[str], None] | None = None,
    label: str | None = None,
) -> list[dict]:
    results = []
    n_push = 0
    for i in range(len(p_over_arr)):
        if not matched_mask[i]:
            continue
        bet, ev = decide_bet(
            float(p_over_arr[i]),
            float(p_under_arr[i]),
            float(odds_over[i]),
            float(odds_under[i]),
            ev_threshold,
        )
        if bet is None:
            continue

        actual = float(saves_actual[i])
        line = float(lines[i])
        if actual == line:
            n_push += 1
            continue
        if bet == "OVER":
            won = actual > line
            profit = calculate_payout(1.0, odds_over[i], won)
        else:
            won = actual < line
            profit = calculate_payout(1.0, odds_under[i], won)
        results.append(
            {
                "local_idx": int(i),
                "bet": bet,
                "profit": float(profit),
                "won": bool(won),
                "ev": float(ev),
                "cluster_id": f"{int(game_id[i])}_{int(goalie_id[i])}",
            }
        )

    if log is not None and n_push:
        log(f"[{label}] {n_push} push bet(s) excluded from grading (actual saves == line).")
    return results


def summarize_bets(results: list[dict], fold_size: int) -> dict:
    n_bets = len(results)
    if n_bets == 0:
        return {"bets": 0, "bet_rate": 0.0, "hit_rate": 0.0, "roi": 0.0, "profit": 0.0}
    wins = sum(r["won"] for r in results)
    profit = sum(r["profit"] for r in results)
    return {
        "bets": n_bets,
        "bet_rate": n_bets / fold_size * 100,
        "hit_rate": wins / n_bets * 100,
        "roi": profit / n_bets * 100,
        "profit": profit,
    }


def side_breakdown(results: list[dict]) -> dict:
    breakdown = {}
    for side in ("OVER", "UNDER"):
        side_bets = [r for r in results if r["bet"] == side]
        n_side = len(side_bets)
        if n_side == 0:
            breakdown[side] = {"bets": 0, "hit_rate": 0.0, "roi": 0.0, "profit": 0.0}
            continue
        wins = sum(r["won"] for r in side_bets)
        profit = sum(r["profit"] for r in side_bets)
        breakdown[side] = {
            "bets": n_side,
            "hit_rate": wins / n_side * 100,
            "roi": profit / n_side * 100,
            "profit": profit,
        }
    return breakdown


def bootstrap_roi_ci(
    results: list[dict],
    n_resamples: int = 10000,
    seed: int = 42,
    ci_pct: float = 95.0,
) -> dict:
    profits = np.asarray([r["profit"] for r in results], dtype=float)
    n_bets = len(profits)
    if n_bets == 0:
        return {"lower": 0.0, "upper": 0.0, "n_bets": 0}
    rng = np.random.RandomState(seed)
    resample_idx = rng.randint(0, n_bets, size=(n_resamples, n_bets))
    boot_rois = profits[resample_idx].mean(axis=1) * 100
    alpha = (100.0 - ci_pct) / 2.0
    return {
        "lower": float(np.percentile(boot_rois, alpha)),
        "upper": float(np.percentile(boot_rois, 100.0 - alpha)),
        "n_bets": int(n_bets),
    }


def cluster_bootstrap_roi_ci(
    results: list[dict],
    n_resamples: int = 10000,
    seed: int = 42,
    ci_pct: float = 95.0,
) -> dict:
    profits = np.asarray([r["profit"] for r in results], dtype=float)
    cluster_ids = np.asarray([r["cluster_id"] for r in results], dtype=object)
    if len(profits) == 0:
        return {"lower": 0.0, "upper": 0.0, "n_clusters": 0}

    unique_clusters, inv = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    cluster_sum = np.zeros(n_clusters)
    cluster_count = np.zeros(n_clusters)
    np.add.at(cluster_sum, inv, profits)
    np.add.at(cluster_count, inv, 1)

    rng = np.random.RandomState(seed)
    boot_rois = np.empty(n_resamples)
    for b in range(n_resamples):
        draw = rng.randint(0, n_clusters, size=n_clusters)
        counts = np.bincount(draw, minlength=n_clusters)
        total_profit = np.dot(counts, cluster_sum)
        total_bets = np.dot(counts, cluster_count)
        boot_rois[b] = (total_profit / total_bets) * 100 if total_bets > 0 else 0.0

    alpha = (100.0 - ci_pct) / 2.0
    return {
        "lower": float(np.percentile(boot_rois, alpha)),
        "upper": float(np.percentile(boot_rois, 100.0 - alpha)),
        "n_clusters": int(n_clusters),
    }


def dedup_one_per_goalie_night(
    prob: np.ndarray,
    y_true: np.ndarray,
    game_id: np.ndarray,
    goalie_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d = pd.DataFrame(
        {
            "prob": np.asarray(prob),
            "y": np.asarray(y_true),
            "game_id": np.asarray(game_id),
            "goalie_id": np.asarray(goalie_id),
        }
    )
    d = d.drop_duplicates(subset=["game_id", "goalie_id"], keep="first")
    return d["prob"].values, d["y"].values


def auc_both(prob: np.ndarray, y_true: np.ndarray, game_id: np.ndarray, goalie_id: np.ndarray) -> dict:
    row_auc = roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else float("nan")
    dedup_prob, dedup_y = dedup_one_per_goalie_night(prob, y_true, game_id, goalie_id)
    night_auc = roc_auc_score(dedup_y, dedup_prob) if len(set(dedup_y)) > 1 else float("nan")
    return {
        "row_level": float(row_auc),
        "one_per_goalie_night": float(night_auc),
        "n_rows": int(len(prob)),
        "n_goalie_nights": int(len(dedup_prob)),
    }


def brier(prob: np.ndarray, y_true: np.ndarray) -> float:
    return float(brier_score_loss(y_true, prob))


def goalie_night_count(game_id: np.ndarray, goalie_id: np.ndarray) -> int:
    return len(set(zip(np.asarray(game_id).tolist(), np.asarray(goalie_id).tolist())))


def bet_goalie_night_count(results: list[dict]) -> int:
    return len(set(r["cluster_id"] for r in results))


def select_threshold_from_val(
    evaluations: list[dict],
    log: Callable[[str], None],
) -> tuple[dict, str | None]:
    in_range = [e for e in evaluations if 15 <= e["summary"]["bet_rate"] <= 35]
    deviation = None
    if not in_range:
        in_range = [e for e in evaluations if 10 <= e["summary"]["bet_rate"] <= 40]
        deviation = "No EV threshold landed in the pre-registered 15-35% val bet-rate band; widened to 10-40%."
        log(f"WARNING: {deviation}")
        if not in_range:
            in_range = evaluations
            deviation += " Even the widened 10-40% band was empty; fell back to ALL thresholds."
            log(f"WARNING: {deviation}")
    return max(in_range, key=lambda e: e["summary"]["roi"]), deviation


def evaluate_threshold_sweep(
    df_bet_val: pd.DataFrame,
    p_over_val: np.ndarray,
    p_under_val: np.ndarray,
    matched_val: np.ndarray,
    log: Callable[[str], None],
) -> tuple[list[dict], dict, str | None]:
    saves_val = df_bet_val["saves"].values.astype(float)
    lines_val = df_bet_val["betting_line"].values.astype(float)
    odds_over_val = df_bet_val["odds_over_american"].astype(float).values
    odds_under_val = df_bet_val["odds_under_american"].astype(float).values
    game_id_val = df_bet_val["game_id"].values
    goalie_id_val = df_bet_val["goalie_id"].values
    n_val = len(df_bet_val)

    log("\n--- VAL EV threshold sweep (probabilities fixed; only threshold varies) ---")
    log(f"{'thresh':>7} {'bets':>6} {'bet_rate':>9} {'hit_rate':>9} {'roi':>9}")
    evaluations = []
    for thresh in EV_THRESHOLDS:
        results = grade_bets(
            p_over_val,
            p_under_val,
            saves_val,
            lines_val,
            odds_over_val,
            odds_under_val,
            game_id_val,
            goalie_id_val,
            thresh,
            matched_val,
            log,
            "VAL",
        )
        summary = summarize_bets(results, n_val)
        evaluations.append({"threshold": thresh, "summary": summary, "results": results})
        log(
            f"{thresh:>7.2f} {summary['bets']:>6} {summary['bet_rate']:>8.1f}% "
            f"{summary['hit_rate']:>8.1f}% {summary['roi']:>+8.2f}%"
        )

    winner, deviation = select_threshold_from_val(evaluations, log)
    log(
        f"\nVAL sweep winner: EV threshold={winner['threshold']:.2f}  "
        f"bets={winner['summary']['bets']}  bet_rate={winner['summary']['bet_rate']:.1f}%  "
        f"roi={winner['summary']['roi']:+.2f}%"
    )
    return evaluations, winner, deviation


def betting_metrics_bundle(
    results: list[dict],
    game_id: np.ndarray,
    goalie_id: np.ndarray,
    fold_size: int,
) -> dict:
    return {
        "summary": summarize_bets(results, fold_size),
        "roi_ci_row": bootstrap_roi_ci(results),
        "roi_ci_cluster": cluster_bootstrap_roi_ci(results),
        "side_breakdown": side_breakdown(results),
        "goalie_nights_total": goalie_night_count(game_id, goalie_id),
        "goalie_nights_bet": bet_goalie_night_count(results),
    }


def fold_wide_auc_brier(
    p_over: np.ndarray,
    matched: np.ndarray,
    saves: np.ndarray,
    lines: np.ndarray,
    game_id: np.ndarray,
    goalie_id: np.ndarray,
    log: Callable[[str], None],
    label: str,
) -> tuple[dict, float]:
    y = (saves[matched] > lines[matched]).astype(int)
    prob = p_over[matched]
    gid = game_id[matched]
    gaid = goalie_id[matched]
    auc = auc_both(prob, y, gid, gaid)
    b = brier(prob, y)
    log(
        f"[{label}] fold-wide AUC row-level={auc['row_level']:.4f} "
        f"one-per-goalie-night={auc['one_per_goalie_night']:.4f}  Brier={b:.5f}  (n={len(prob)})"
    )
    return auc, b
