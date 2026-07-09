"""Betting policy helpers for distributional saves experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    family: str
    min_true_ev: float | None = None
    min_prob_edge: float | None = None
    min_conditional_edge: float | None = None
    book_filter: tuple[str, ...] | None = None
    line_shop: bool = False


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(value_float):
        return None
    return value_float


def american_profit_per_dollar(odds: float) -> float | None:
    odds_float = _finite_float(odds)
    if odds_float is None or -100 < odds_float < 100:
        return None
    if odds_float < 0:
        return 100.0 / abs(odds_float)
    return odds_float / 100.0


def american_implied_probability(odds: float) -> float | None:
    odds_float = _finite_float(odds)
    if odds_float is None or -100 < odds_float < 100:
        return None
    if odds_float < 0:
        return abs(odds_float) / (abs(odds_float) + 100.0)
    return 100.0 / (odds_float + 100.0)


def probability_edge(model_prob: float, odds: float) -> float | None:
    if not _finite_probability(model_prob):
        return None
    implied = american_implied_probability(odds)
    if implied is None:
        return None
    return float(model_prob) - float(implied)


def conditional_novig_edge(
    p_side: float,
    p_other_side: float,
    side_odds: float,
    other_side_odds: float,
) -> float | None:
    if not _finite_probability(p_side) or not _finite_probability(p_other_side):
        return None
    non_push = float(p_side) + float(p_other_side)
    if non_push <= 0:
        return None
    side_implied = american_implied_probability(side_odds)
    other_implied = american_implied_probability(other_side_odds)
    if side_implied is None or other_implied is None:
        return None
    vig_sum = side_implied + other_implied
    if vig_sum <= 0:
        return None
    model_conditional = float(p_side) / non_push
    market_fair = side_implied / vig_sum
    return model_conditional - market_fair


def true_expected_profit(
    p_win: float,
    p_loss: float,
    odds: float,
    p_push: float = 0.0,
) -> float | None:
    if not _finite_probability(p_win) or not _finite_probability(p_loss) or not _finite_probability(p_push):
        return None
    payout_profit = american_profit_per_dollar(odds)
    if payout_profit is None:
        return None
    return float(p_win) * payout_profit - float(p_loss) + float(p_push) * 0.0


def side_values(
    p_over: float,
    p_under: float,
    p_push: float,
    odds_over: float,
    odds_under: float,
) -> dict:
    p_over_float = _probability_or_none(p_over)
    p_under_float = _probability_or_none(p_under)
    p_push_float = _probability_or_none(p_push)
    prob_total = None
    if p_over_float is not None and p_under_float is not None and p_push_float is not None:
        prob_total = p_over_float + p_under_float + p_push_float
    return {
        "p_over": p_over_float,
        "p_under": p_under_float,
        "p_push": p_push_float,
        "prob_total": prob_total,
        "odds_over": _finite_float(odds_over),
        "odds_under": _finite_float(odds_under),
        "profit_per_dollar_over": american_profit_per_dollar(odds_over),
        "profit_per_dollar_under": american_profit_per_dollar(odds_under),
        "implied_prob_over": american_implied_probability(odds_over),
        "implied_prob_under": american_implied_probability(odds_under),
        "true_ev_over": true_expected_profit(p_over, p_under, odds_over, p_push),
        "true_ev_under": true_expected_profit(p_under, p_over, odds_under, p_push),
        "prob_edge_over": probability_edge(p_over, odds_over),
        "prob_edge_under": probability_edge(p_under, odds_under),
        "conditional_edge_over": conditional_novig_edge(p_over, p_under, odds_over, odds_under),
        "conditional_edge_under": conditional_novig_edge(p_under, p_over, odds_under, odds_over),
    }


def decide_bet(
    p_over: float,
    p_under: float,
    p_push: float,
    odds_over: float,
    odds_under: float,
    policy: PolicyConfig,
) -> dict | None:
    values = side_values(p_over, p_under, p_push, odds_over, odds_under)
    candidates = []
    for side in ("OVER", "UNDER"):
        true_ev = values[f"true_ev_{side.lower()}"]
        prob_edge = values[f"prob_edge_{side.lower()}"]
        conditional_edge = values[f"conditional_edge_{side.lower()}"]
        if true_ev is None or prob_edge is None or conditional_edge is None:
            continue
        if policy.family == "probability_edge":
            score = prob_edge
            threshold = policy.min_prob_edge
        elif policy.family in {"true_ev", "true_ev_plus_conditional_edge"}:
            score = true_ev
            threshold = policy.min_true_ev
        else:
            raise ValueError(f"Unknown policy family: {policy.family}")

        if threshold is not None and score < threshold:
            continue
        if policy.family == "true_ev_plus_conditional_edge":
            assert policy.min_conditional_edge is not None
            if conditional_edge < policy.min_conditional_edge:
                continue
        candidates.append(
            {
                "bet": side,
                "score": float(score),
                "true_ev": float(true_ev),
                "prob_edge": float(prob_edge),
                "conditional_edge": float(conditional_edge),
                "p_side": float(p_over if side == "OVER" else p_under),
                "p_opposite": float(p_under if side == "OVER" else p_over),
                "p_push": float(p_push),
            }
        )

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c["score"], c["true_ev"], c["prob_edge"]), reverse=True)
    return candidates[0]


def grade_policy_bets(
    df_bet: pd.DataFrame,
    p_over_arr: np.ndarray,
    p_under_arr: np.ndarray,
    p_push_arr: np.ndarray,
    matched_mask: np.ndarray,
    policy: PolicyConfig,
) -> list[dict]:
    rows = []
    for i in range(len(df_bet)):
        if not matched_mask[i]:
            continue
        row = df_bet.iloc[i]
        if policy.book_filter is not None and str(row["book_key"]) not in policy.book_filter:
            continue
        decision = decide_bet(
            float(p_over_arr[i]),
            float(p_under_arr[i]),
            float(p_push_arr[i]),
            float(row["odds_over_american"]),
            float(row["odds_under_american"]),
            policy,
        )
        if decision is None:
            continue
        side = decision["bet"]
        actual = float(row["saves"])
        line = float(row["betting_line"])
        odds = float(row["odds_over_american"] if side == "OVER" else row["odds_under_american"])
        profit = _realized_profit(side, actual, line, odds)
        rows.append(
            {
                "local_idx": int(i),
                "bet": side,
                "profit": float(profit),
                "outcome": "PUSH" if actual == line else ("WIN" if profit > 0 else "LOSS"),
                "won": bool(profit > 0),
                "pushed": bool(actual == line),
                "score": float(decision["score"]),
                "true_ev": float(decision["true_ev"]),
                "prob_edge": float(decision["prob_edge"]),
                "conditional_edge": float(decision["conditional_edge"]),
                "p_side": float(decision["p_side"]),
                "p_opposite": float(decision["p_opposite"]),
                "p_push": float(decision["p_push"]),
                "book_key": str(row["book_key"]),
                "line": line,
                "odds": odds,
                "cluster_id": f"{int(row['game_id'])}_{int(row['goalie_id'])}",
            }
        )
    if policy.line_shop:
        return _line_shop_best_per_cluster(rows)
    return rows


def summarize_policy_bets(results: list[dict], denominator: int) -> dict:
    n_bets = len(results)
    if n_bets == 0:
        return {
            "bets": 0,
            "bet_rate": 0.0,
            "win_rate": 0.0,
            "push_rate": 0.0,
            "roi": 0.0,
            "profit": 0.0,
            "avg_true_ev": 0.0,
            "avg_prob_edge": 0.0,
            "avg_conditional_edge": 0.0,
        }
    wins = sum(r["outcome"] == "WIN" for r in results)
    pushes = sum(r["outcome"] == "PUSH" for r in results)
    profit = sum(r["profit"] for r in results)
    return {
        "bets": n_bets,
        "bet_rate": n_bets / denominator * 100 if denominator else 0.0,
        "win_rate": wins / n_bets * 100,
        "push_rate": pushes / n_bets * 100,
        "roi": profit / n_bets * 100,
        "profit": profit,
        "avg_true_ev": float(np.mean([r["true_ev"] for r in results])),
        "avg_prob_edge": float(np.mean([r["prob_edge"] for r in results])),
        "avg_conditional_edge": float(np.mean([r["conditional_edge"] for r in results])),
    }


def side_breakdown_policy(results: list[dict]) -> dict:
    breakdown = {}
    for side in ("OVER", "UNDER"):
        side_results = [r for r in results if r["bet"] == side]
        breakdown[side] = summarize_policy_bets(side_results, len(side_results))
    return breakdown


def policy_denominator(df_bet: pd.DataFrame, policy: PolicyConfig) -> int:
    if policy.book_filter is not None:
        df_bet = df_bet[df_bet["book_key"].astype(str).isin(policy.book_filter)]
    if policy.line_shop:
        return int(df_bet[["game_id", "goalie_id"]].drop_duplicates().shape[0])
    return int(len(df_bet))


def make_policy_grid() -> list[PolicyConfig]:
    policies: list[PolicyConfig] = []
    for min_edge in (0.05, 0.10, 0.12, 0.15):
        policies.append(
            PolicyConfig(
                name=f"old_prob_edge_{min_edge:.2f}",
                family="probability_edge",
                min_prob_edge=min_edge,
            )
        )
    for min_ev in (0.00, 0.02, 0.04, 0.06):
        policies.append(
            PolicyConfig(
                name=f"true_ev_{min_ev:.2f}",
                family="true_ev",
                min_true_ev=min_ev,
            )
        )
    for min_ev in (0.00, 0.02, 0.04):
        for min_edge in (0.00, 0.02, 0.04):
            policies.append(
                PolicyConfig(
                    name=f"true_ev_{min_ev:.2f}_cond_edge_{min_edge:.2f}",
                    family="true_ev_plus_conditional_edge",
                    min_true_ev=min_ev,
                    min_conditional_edge=min_edge,
                )
            )
    for min_ev in (0.00, 0.02, 0.04):
        policies.append(
            PolicyConfig(
                name=f"line_shop_true_ev_{min_ev:.2f}",
                family="true_ev",
                min_true_ev=min_ev,
                line_shop=True,
            )
        )
    return policies


def serialize_policy(policy: PolicyConfig) -> dict:
    return {
        "name": policy.name,
        "family": policy.family,
        "min_true_ev": policy.min_true_ev,
        "min_prob_edge": policy.min_prob_edge,
        "min_conditional_edge": policy.min_conditional_edge,
        "book_filter": list(policy.book_filter) if policy.book_filter is not None else None,
        "line_shop": policy.line_shop,
    }


def _line_shop_best_per_cluster(results: Iterable[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for row in results:
        current = best.get(row["cluster_id"])
        if current is None or (row["true_ev"], row["score"], row["conditional_edge"]) > (
            current["true_ev"],
            current["score"],
            current["conditional_edge"],
        ):
            best[row["cluster_id"]] = row
    return list(best.values())


def _realized_profit(side: str, actual: float, line: float, odds: float) -> float:
    if actual == line:
        return 0.0
    won = actual > line if side == "OVER" else actual < line
    if not won:
        return -1.0
    payout_profit = american_profit_per_dollar(odds)
    if payout_profit is None:
        raise ValueError(f"Invalid odds for realized payout: {odds}")
    return payout_profit


def _valid_american_odds(odds: float) -> bool:
    odds_float = _finite_float(odds)
    return odds_float is not None and (odds_float <= -100 or odds_float >= 100)


def _probability_or_none(prob: float) -> float | None:
    prob_float = _finite_float(prob)
    if prob_float is None or prob_float < 0.0 or prob_float > 1.0:
        return None
    return prob_float


def _finite_probability(prob: float) -> bool:
    return _probability_or_none(prob) is not None
