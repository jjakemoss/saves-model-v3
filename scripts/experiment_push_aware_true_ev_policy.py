"""
Push-aware true-EV policy audit for the game-context distributional model.

This script reuses the saved 2026-07-08 game-context distributional artifact
and asks a narrower question than another feature retrain:

  Does selecting bets by true expected profit, with pushes treated as refunds,
  change the honest policy result compared with the repo's legacy
  probability-edge rule?

Important protocol caveat: the source test fold has already been inspected in
the game-context experiment. A positive result here would be a post-hoc policy
sensitivity finding, not independent proof of a tradable edge.

Usage:
    python scripts/experiment_push_aware_true_ev_policy.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.distributional_saves import (  # noqa: E402
    CAP,
    SavesDistribution,
    build_betting_frame,
    compute_distribution_predictions,
    join_and_price,
    load_modeling_frame,
)
from experiments.harness import (  # noqa: E402
    betting_metrics_bundle,
    evaluate_threshold_sweep,
    grade_bets,
    split_by_date,
)
from experiments.policies import (  # noqa: E402
    PolicyConfig,
    grade_policy_bets,
    make_policy_grid,
    policy_denominator,
    serialize_policy,
    side_breakdown_policy,
    summarize_policy_bets,
)


DATA_PATH_CLEAN = Path("data/processed/clean_training_data.parquet")
DATA_PATH_MULTIBOOK = Path("data/processed/multibook_classification_training_data.parquet")
DATA_PATH_CONTEXT = Path("data/processed/game_context_features.parquet")
SOURCE_ARTIFACT = Path("models/trained/experiment_game_context_distributional_20260708_210813")
OUTPUT_ROOT = Path("models/trained")
BET_RATE_BAND_PRIMARY = (15.0, 35.0)
BET_RATE_BAND_FALLBACK = (10.0, 40.0)
VARIANTS = ("control", "context_shots", "context_both")


def make_logger(log_path: Path):
    log_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(str(msg))

    def flush_log() -> None:
        log_path.write_text("\n".join(log_lines), encoding="utf-8")

    return log, flush_log


def load_source_metadata() -> dict:
    metadata_path = SOURCE_ARTIFACT / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing source artifact metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_xgb_regressor(path: Path) -> xgb.XGBRegressor:
    if not path.exists():
        raise FileNotFoundError(f"Missing saved model: {path}")
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def replay_legacy_policy(
    variant_name: str,
    source_variant: dict,
    df_bet_val,
    df_bet_test,
    p_over_val,
    p_under_val,
    matched_val,
    p_over_test,
    p_under_test,
    matched_test,
    log,
) -> dict:
    log("\n--- Legacy probability-edge replay audit ---")
    val_evaluations, val_winner, selection_deviation = evaluate_threshold_sweep(
        df_bet_val,
        p_over_val,
        p_under_val,
        matched_val,
        log,
    )
    expected_threshold = float(source_variant["val_winner"]["threshold"])
    expected_test = source_variant["test_single_touch"]["summary"]
    if abs(float(val_winner["threshold"]) - expected_threshold) > 1e-12:
        raise AssertionError(
            f"{variant_name}: legacy replay selected threshold {val_winner['threshold']} "
            f"but source artifact selected {expected_threshold}."
        )

    test_results = grade_bets(
        p_over_test,
        p_under_test,
        df_bet_test["saves"].values.astype(float),
        df_bet_test["betting_line"].values.astype(float),
        df_bet_test["odds_over_american"].astype(float).values,
        df_bet_test["odds_under_american"].astype(float).values,
        df_bet_test["game_id"].values,
        df_bet_test["goalie_id"].values,
        expected_threshold,
        matched_test,
        log,
        f"{variant_name} TEST legacy replay",
    )
    test_bundle = betting_metrics_bundle(
        test_results,
        df_bet_test["game_id"].values,
        df_bet_test["goalie_id"].values,
        len(df_bet_test),
    )
    observed = test_bundle["summary"]
    if int(observed["bets"]) != int(expected_test["bets"]):
        raise AssertionError(
            f"{variant_name}: legacy replay bet count {observed['bets']} "
            f"does not match source {expected_test['bets']}."
        )
    if abs(float(observed["roi"]) - float(expected_test["roi"])) > 1e-9:
        raise AssertionError(
            f"{variant_name}: legacy replay ROI {observed['roi']} "
            f"does not match source {expected_test['roi']}."
        )
    log(
        f"Legacy replay matched source artifact: threshold={expected_threshold:.2f}, "
        f"test bets={observed['bets']}, ROI={observed['roi']:+.2f}%."
    )
    return {
        "val_evaluations": [{"threshold": e["threshold"], "summary": e["summary"]} for e in val_evaluations],
        "val_winner": {"threshold": val_winner["threshold"], "summary": val_winner["summary"]},
        "selection_deviation": selection_deviation,
        "test_replay": test_bundle,
        "matched_source_artifact": True,
    }


def select_policy_from_val(evaluations: list[dict], log) -> tuple[dict, str | None]:
    in_range = [e for e in evaluations if BET_RATE_BAND_PRIMARY[0] <= e["summary"]["bet_rate"] <= BET_RATE_BAND_PRIMARY[1]]
    deviation = None
    if not in_range:
        in_range = [
            e
            for e in evaluations
            if BET_RATE_BAND_FALLBACK[0] <= e["summary"]["bet_rate"] <= BET_RATE_BAND_FALLBACK[1]
        ]
        deviation = (
            "No policy landed in the pre-registered 15-35% val bet-rate band; "
            "widened to 10-40%."
        )
        log(f"WARNING: {deviation}")
        if not in_range:
            in_range = evaluations
            deviation += " Even the widened 10-40% band was empty; fell back to all policies."
            log(f"WARNING: {deviation}")
    return max(in_range, key=lambda e: e["summary"]["roi"]), deviation


def policy_metrics(results: list[dict], denominator: int) -> dict:
    from experiments.harness import bet_goalie_night_count, bootstrap_roi_ci, cluster_bootstrap_roi_ci

    return {
        "summary": summarize_policy_bets(results, denominator),
        "roi_ci_row": bootstrap_roi_ci(results),
        "roi_ci_cluster": cluster_bootstrap_roi_ci(results),
        "side_breakdown": side_breakdown_policy(results),
        "goalie_nights_bet": bet_goalie_night_count(results),
    }


def evaluate_policy_grid(
    df_bet,
    p_over,
    p_under,
    p_push,
    matched,
    policies: list[PolicyConfig],
    log,
    label: str,
) -> list[dict]:
    evaluations = []
    log("\n--- VAL policy sweep ---")
    log(
        f"{'policy':<34} {'bets':>6} {'bet_rate':>9} {'win':>8} "
        f"{'push':>8} {'roi':>9} {'avg_ev':>9} {'cond':>9}"
    )
    for policy in policies:
        denominator = policy_denominator(df_bet, policy)
        results = grade_policy_bets(df_bet, p_over, p_under, p_push, matched, policy)
        summary = summarize_policy_bets(results, denominator)
        evaluations.append(
            {
                "policy": serialize_policy(policy),
                "summary": summary,
                "results": results,
                "denominator": denominator,
            }
        )
        log(
            f"{policy.name:<34} {summary['bets']:>6} {summary['bet_rate']:>8.1f}% "
            f"{summary['win_rate']:>7.1f}% {summary['push_rate']:>7.1f}% "
            f"{summary['roi']:>+8.2f}% {summary['avg_true_ev']:>+8.3f} "
            f"{summary['avg_conditional_edge']:>+8.3f}"
        )
    if label:
        log(f"Completed {label} policy sweep across {len(policies)} policies.")
    return evaluations


def _policy_from_serialized(data: dict) -> PolicyConfig:
    book_filter = tuple(data["book_filter"]) if data.get("book_filter") is not None else None
    return PolicyConfig(
        name=data["name"],
        family=data["family"],
        min_true_ev=data.get("min_true_ev"),
        min_prob_edge=data.get("min_prob_edge"),
        min_conditional_edge=data.get("min_conditional_edge"),
        book_filter=book_filter,
        line_shop=bool(data.get("line_shop", False)),
    )


def _probability_sum_check(p_over, p_under, p_push, matched, log, label: str) -> dict:
    sums = p_over[matched] + p_under[matched] + p_push[matched]
    diag = {
        "min": float(np.min(sums)),
        "max": float(np.max(sums)),
        "mean": float(np.mean(sums)),
        "rows_below_0_999": int((sums < 0.999).sum()),
        "rows_above_1_001": int((sums > 1.001).sum()),
    }
    log(
        f"[{label}] p_over+p_under+p_push: min={diag['min']:.6f}, "
        f"max={diag['max']:.6f}, mean={diag['mean']:.6f}, "
        f"below .999={diag['rows_below_0_999']}, above 1.001={diag['rows_above_1_001']}."
    )
    if diag["rows_below_0_999"] or diag["rows_above_1_001"]:
        raise AssertionError(f"{label}: probability sums outside expected tolerance.")
    return diag


def run_variant(
    variant_name: str,
    source_metadata: dict,
    frame,
    clean_split,
    df_bet_val,
    df_bet_test,
    policies: list[PolicyConfig],
    log,
) -> dict:
    log("\n" + "=" * 80)
    log(f"VARIANT: {variant_name}")
    log("=" * 80)
    source_variant = source_metadata["results"][variant_name]
    shots_model = load_xgb_regressor(Path(source_variant["shots_model"]["model_path"]))
    rate_model = load_xgb_regressor(Path(source_variant["save_rate_model"]["model_path"]))
    shots_cols = source_variant["shots_feature_cols"]
    rate_cols = source_variant["rate_feature_cols"]
    alpha = float(source_variant["dispersion"]["alpha"])
    dist = SavesDistribution(CAP)

    dist_preds_val = compute_distribution_predictions(
        frame.df,
        clean_split.val_idx,
        shots_model,
        rate_model,
        alpha,
        shots_cols,
        rate_cols,
        dist,
        log,
        f"{variant_name} VAL clean_training_data",
    )
    p_over_val, p_under_val, p_push_val, matched_val, cov_val = join_and_price(
        df_bet_val,
        dist_preds_val,
        dist,
        log,
        f"{variant_name} VAL betting frame",
    )
    prob_sum_val = _probability_sum_check(p_over_val, p_under_val, p_push_val, matched_val, log, f"{variant_name} VAL")

    dist_preds_test = compute_distribution_predictions(
        frame.df,
        clean_split.test_idx,
        shots_model,
        rate_model,
        alpha,
        shots_cols,
        rate_cols,
        dist,
        log,
        f"{variant_name} TEST clean_training_data",
    )
    p_over_test, p_under_test, p_push_test, matched_test, cov_test = join_and_price(
        df_bet_test,
        dist_preds_test,
        dist,
        log,
        f"{variant_name} TEST betting frame",
    )
    prob_sum_test = _probability_sum_check(
        p_over_test, p_under_test, p_push_test, matched_test, log, f"{variant_name} TEST"
    )

    legacy_replay = replay_legacy_policy(
        variant_name,
        source_variant,
        df_bet_val,
        df_bet_test,
        p_over_val,
        p_under_val,
        matched_val,
        p_over_test,
        p_under_test,
        matched_test,
        log,
    )

    policy_evaluations = evaluate_policy_grid(
        df_bet_val,
        p_over_val,
        p_under_val,
        p_push_val,
        matched_val,
        policies,
        log,
        f"{variant_name} VAL",
    )
    winner, selection_deviation = select_policy_from_val(policy_evaluations, log)
    selected_policy = _policy_from_serialized(winner["policy"])
    log(
        f"\nVAL policy winner: {selected_policy.name}  bets={winner['summary']['bets']}  "
        f"bet_rate={winner['summary']['bet_rate']:.1f}%  ROI={winner['summary']['roi']:+.2f}%"
    )

    log("\n" + "-" * 80)
    log(f"SINGLE TEST TOUCH: {variant_name}, selected policy={selected_policy.name}")
    log("-" * 80)
    test_denominator = policy_denominator(df_bet_test, selected_policy)
    test_results = grade_policy_bets(
        df_bet_test,
        p_over_test,
        p_under_test,
        p_push_test,
        matched_test,
        selected_policy,
    )
    val_bundle = policy_metrics(winner["results"], winner["denominator"])
    test_bundle = policy_metrics(test_results, test_denominator)
    test_summary = test_bundle["summary"]
    cluster_ci = test_bundle["roi_ci_cluster"]
    side = test_bundle["side_breakdown"]
    log(
        f"{variant_name} TEST: {test_summary['bets']} bets, "
        f"{test_summary['bet_rate']:.1f}% bet rate, {test_summary['win_rate']:.1f}% win rate, "
        f"{test_summary['push_rate']:.1f}% push rate, {test_summary['roi']:+.2f}% ROI"
    )
    log(
        f"{variant_name} TEST ROI 95% CI (cluster): "
        f"[{cluster_ci['lower']:+.2f}%, {cluster_ci['upper']:+.2f}%] "
        f"(n_clusters={cluster_ci['n_clusters']})"
    )
    log(
        f"{variant_name} TEST side breakdown: OVER {side['OVER']['bets']} bets "
        f"({side['OVER']['roi']:+.2f}%), UNDER {side['UNDER']['bets']} bets "
        f"({side['UNDER']['roi']:+.2f}%)"
    )

    return {
        "variant": variant_name,
        "source_variant_description": source_variant["description"],
        "model_paths": {
            "shots": source_variant["shots_model"]["model_path"],
            "save_rate": source_variant["save_rate_model"]["model_path"],
        },
        "feature_counts": {"shots": len(shots_cols), "save_rate": len(rate_cols)},
        "dispersion_alpha": alpha,
        "join_coverage": {"val_pct": cov_val, "test_pct": cov_test},
        "probability_sum_check": {"val": prob_sum_val, "test": prob_sum_test},
        "legacy_probability_edge_replay": legacy_replay,
        "val_policy_sweep": [
            {
                "policy": e["policy"],
                "summary": e["summary"],
                "denominator": e["denominator"],
            }
            for e in policy_evaluations
        ],
        "val_winner": {
            "policy": winner["policy"],
            "summary": winner["summary"],
            "denominator": winner["denominator"],
            **val_bundle,
        },
        "selection_deviation": selection_deviation,
        "test_single_touch": {
            "policy": serialize_policy(selected_policy),
            "denominator": test_denominator,
            **test_bundle,
        },
        "test_bets": test_results,
        "touch_count_audit": {
            "legacy_replay_test_evaluations": 1,
            "policy_val_evaluations": len(policy_evaluations),
            "policy_test_evaluations": 1,
        },
    }


def main() -> int:
    start = time.time()
    source_metadata = load_source_metadata()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_push_aware_true_ev_policy_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log, flush_log = make_logger(log_path)

    try:
        log("=" * 80)
        log("PUSH-AWARE TRUE-EV POLICY AUDIT")
        log(f"Run timestamp: {datetime.now().isoformat()}")
        log("=" * 80)
        log(f"Source artifact: {SOURCE_ARTIFACT}")
        log(f"Output directory: {output_dir}")
        log(
            "Protocol caveat: this reuses an already-touched test fold. Positive ROI here "
            "would be post-hoc policy sensitivity, not independent edge proof."
        )

        frame = load_modeling_frame(DATA_PATH_CLEAN, DATA_PATH_CONTEXT, log)
        clean_split = split_by_date(frame.df, log, "clean_training_data")
        df_bet = build_betting_frame(DATA_PATH_MULTIBOOK, log)
        bet_split = split_by_date(df_bet, log, "multibook_classification_training_data")
        df_bet_val = df_bet.iloc[bet_split.val_idx].reset_index(drop=True)
        df_bet_test = df_bet.iloc[bet_split.test_idx].reset_index(drop=True)

        policies = make_policy_grid()
        log(f"Pre-registered policy grid: {len(policies)} policies.")
        for policy in policies:
            log(f"  - {policy.name}: {serialize_policy(policy)}")

        results = {}
        for variant_name in VARIANTS:
            results[variant_name] = run_variant(
                variant_name,
                source_metadata,
                frame,
                clean_split,
                df_bet_val,
                df_bet_test,
                policies,
                log,
            )
            flush_log()

        log("\n" + "=" * 80)
        log("HEAD-TO-HEAD POLICY SUMMARY")
        log("=" * 80)
        log(
            f"{'variant':<16} {'selected_policy':<34} {'roi':>9} {'bets':>6} "
            f"{'push':>7} {'cluster_ci':>25}"
        )
        for variant_name in VARIANTS:
            row = results[variant_name]["test_single_touch"]
            summary = row["summary"]
            ci = row["roi_ci_cluster"]
            log(
                f"{variant_name:<16} {row['policy']['name']:<34} "
                f"{summary['roi']:>+8.2f}% {summary['bets']:>6} "
                f"{summary['push_rate']:>6.1f}% "
                f"[{ci['lower']:+.2f}%, {ci['upper']:+.2f}%]"
            )
        log("No result above is independent proof of edge because this is a post-hoc policy audit.")

        elapsed = time.time() - start
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "wall_clock_seconds": elapsed,
            "source_artifact": str(SOURCE_ARTIFACT),
            "source_artifact_timestamp": source_metadata.get("timestamp"),
            "data_paths": {
                "clean": str(DATA_PATH_CLEAN),
                "multibook": str(DATA_PATH_MULTIBOOK),
                "context": str(DATA_PATH_CONTEXT),
            },
            "cap": CAP,
            "policy_grid": [serialize_policy(p) for p in policies],
            "bet_rate_bands": {
                "primary": BET_RATE_BAND_PRIMARY,
                "fallback": BET_RATE_BAND_FALLBACK,
            },
            "fold_boundaries_clean_training_data": clean_split.boundaries,
            "fold_boundaries_multibook": bet_split.boundaries,
            "protocol_notes": [
                "This policy audit reuses the already-inspected game-context distributional test fold.",
                "Legacy probability-edge replay must match the source artifact before true-EV policies are interpreted.",
                "True expected profit per $1 stake uses P(win)*profit_if_win - P(loss); P(push) contributes 0.",
                "Conditional no-vig edge guardrails compare P(side)/(P(over)+P(under)) to market fair probability.",
                "Policy selection is validation-only; each variant gets one selected-policy test evaluation.",
                "Cluster bootstrap CI is the primary uncertainty check.",
            ],
            "results": results,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        log(f"\nSaved metadata to: {metadata_path}")
        log(f"Wall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        log("\n" + "=" * 80)
        log("POLICY AUDIT COMPLETE")
        log("=" * 80)
        flush_log()
    except Exception:
        flush_log()
        raise

    print(f"Saved run log to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
