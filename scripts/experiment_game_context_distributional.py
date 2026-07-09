"""
Roadmap item 8 follow-up: game-context distributional saves experiment.

This compares the no-context distributional prototype against pre-registered
game-context variants. The primary question is whether context helps the
shots-against submodel, where the previous distributional experiment found
nearly all useful signal lives. This script is an honest experiment harness,
not an optimizer for a positive result.

Variants:
  control:       same feature setup as scripts/experiment_distributional.py
  context_shots: context columns added to shots model only; save-rate matches control
  context_both:  context columns added to shots and save-rate; secondary diagnostic

Protocol:
  train = game_date < 2025-10-16
  val   = 2025-10-16 through 2025-12-03
  test  = game_date >= 2025-12-04

  Submodel config selection is validation-only. EV threshold selection is
  validation-only over [0.05, 0.10, 0.12, 0.15], filtered to the 15-35% val
  bet-rate band with the documented fallback. Each variant gets exactly one
  test evaluation after validation selection.

Artifacts:
  models/trained/experiment_game_context_distributional_{timestamp}/
    run_log.txt
    metadata.json
    {variant}_shots_model.json
    {variant}_save_rate_model.json

Usage:
    python scripts/experiment_game_context_distributional.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.distributional_saves import (  # noqa: E402
    CAP,
    SAVE_RATE_CONFIGS,
    SHOTS_CONFIGS,
    SavesDistribution,
    build_betting_frame,
    compute_distribution_predictions,
    fit_dispersion,
    intrinsic_quality_metrics,
    join_and_price,
    load_modeling_frame,
    top_features,
    train_save_rate_model,
    train_shots_model,
)
from experiments.harness import (  # noqa: E402
    EV_THRESHOLDS,
    betting_metrics_bundle,
    evaluate_threshold_sweep,
    fold_wide_auc_brier,
    grade_bets,
    split_by_date,
)


DATA_PATH_CLEAN = Path("data/processed/clean_training_data.parquet")
DATA_PATH_MULTIBOOK = Path("data/processed/multibook_classification_training_data.parquet")
DATA_PATH_CONTEXT = Path("data/processed/game_context_features.parquet")
OUTPUT_ROOT = Path("models/trained")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    shots_use_context: bool
    rate_use_context: bool
    primary: bool


VARIANTS = [
    VariantSpec(
        name="control",
        description="No-context distributional setup from scripts/experiment_distributional.py.",
        shots_use_context=False,
        rate_use_context=False,
        primary=True,
    ),
    VariantSpec(
        name="context_shots",
        description="Context columns added to shots-against model only; save-rate model matches control.",
        shots_use_context=True,
        rate_use_context=False,
        primary=True,
    ),
    VariantSpec(
        name="context_both",
        description="Context columns added to shots-against and save-rate models; secondary diagnostic.",
        shots_use_context=True,
        rate_use_context=True,
        primary=False,
    ),
]
assert len(VARIANTS) <= 3
assert len(SHOTS_CONFIGS) <= 6
assert len(SAVE_RATE_CONFIGS) <= 6
assert EV_THRESHOLDS == [0.05, 0.10, 0.12, 0.15]


def make_logger(log_path: Path):
    log_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(str(msg))

    def flush_log() -> None:
        log_path.write_text("\n".join(log_lines), encoding="utf-8")

    return log, flush_log


def feature_cols_for_variant(frame, variant: VariantSpec) -> tuple[list[str], list[str]]:
    base = frame.base_feature_cols + frame.engineered_cols
    shots_cols = base + frame.context_cols if variant.shots_use_context else list(base)
    rate_cols = base + frame.context_cols if variant.rate_use_context else list(base)
    return shots_cols, rate_cols


def evaluate_test_once(
    df_bet_test,
    p_over_test,
    p_under_test,
    matched_test,
    threshold,
    log,
    label,
) -> tuple[dict, dict, float]:
    test_auc, test_brier = fold_wide_auc_brier(
        p_over_test,
        matched_test,
        df_bet_test["saves"].values,
        df_bet_test["betting_line"].values,
        df_bet_test["game_id"].values,
        df_bet_test["goalie_id"].values,
        log,
        label,
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
        threshold,
        matched_test,
        log,
        label,
    )
    bundle = betting_metrics_bundle(
        test_results,
        df_bet_test["game_id"].values,
        df_bet_test["goalie_id"].values,
        len(df_bet_test),
    )
    return {**bundle, "auc": test_auc, "brier": test_brier}, test_auc, test_brier


def run_variant(
    variant: VariantSpec,
    frame,
    clean_split,
    df_bet_val,
    df_bet_test,
    output_dir: Path,
    log,
) -> dict:
    log("\n" + "=" * 80)
    log(f"VARIANT: {variant.name}")
    log("=" * 80)
    log(variant.description)

    shots_cols, rate_cols = feature_cols_for_variant(frame, variant)
    log(f"Shots feature count: {len(shots_cols)}")
    log(f"Save-rate feature count: {len(rate_cols)}")

    shots_model, shots_winner, shots_evals = train_shots_model(
        frame.df,
        clean_split.train_idx,
        clean_split.val_idx,
        shots_cols,
        log,
        variant.name,
    )
    alpha, dispersion_method, dispersion_diag = fit_dispersion(
        shots_model,
        frame.df,
        clean_split.train_idx,
        shots_cols,
        log,
        variant.name,
    )
    rate_model, rate_winner, rate_evals = train_save_rate_model(
        frame.df,
        clean_split.train_idx,
        clean_split.val_idx,
        rate_cols,
        log,
        variant.name,
    )

    shots_path = output_dir / f"{variant.name}_shots_model.json"
    rate_path = output_dir / f"{variant.name}_save_rate_model.json"
    shots_model.get_booster().save_model(str(shots_path))
    rate_model.get_booster().save_model(str(rate_path))
    log(f"Saved {variant.name} shots model to: {shots_path}")
    log(f"Saved {variant.name} save-rate model to: {rate_path}")

    shots_top10 = top_features(shots_model, 10)
    rate_top10 = top_features(rate_model, 10)
    log(f"{variant.name} shots top-10 features by gain: {shots_top10}")
    log(f"{variant.name} save-rate top-10 features by gain: {rate_top10}")

    dist = SavesDistribution(CAP)

    # Validation pricing and selection only. Test is deliberately not priced
    # until after the validation winner has been chosen.
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
        f"{variant.name} VAL clean_training_data",
    )
    intrinsic_val = intrinsic_quality_metrics(
        frame.df,
        clean_split.val_idx,
        dist_preds_val,
        dist,
        log,
        variant.name,
    )
    p_over_val, p_under_val, p_push_val, matched_val, cov_val = join_and_price(
        df_bet_val,
        dist_preds_val,
        dist,
        log,
        f"{variant.name} VAL betting frame",
    )
    val_auc, val_brier = fold_wide_auc_brier(
        p_over_val,
        matched_val,
        df_bet_val["saves"].values,
        df_bet_val["betting_line"].values,
        df_bet_val["game_id"].values,
        df_bet_val["goalie_id"].values,
        log,
        f"{variant.name} VAL",
    )
    val_evaluations, val_winner, selection_deviation = evaluate_threshold_sweep(
        df_bet_val,
        p_over_val,
        p_under_val,
        matched_val,
        log,
    )
    winner_threshold = val_winner["threshold"]
    val_bundle = betting_metrics_bundle(
        val_winner["results"],
        df_bet_val["game_id"].values,
        df_bet_val["goalie_id"].values,
        len(df_bet_val),
    )
    log(
        f"{variant.name} VAL winner ROI 95% CI (row): "
        f"[{val_bundle['roi_ci_row']['lower']:+.2f}%, {val_bundle['roi_ci_row']['upper']:+.2f}%]"
    )
    log(
        f"{variant.name} VAL winner ROI 95% CI (cluster): "
        f"[{val_bundle['roi_ci_cluster']['lower']:+.2f}%, {val_bundle['roi_ci_cluster']['upper']:+.2f}%] "
        f"(n_clusters={val_bundle['roi_ci_cluster']['n_clusters']})"
    )

    log("\n" + "-" * 80)
    log(f"SINGLE TEST TOUCH: {variant.name}, EV threshold={winner_threshold:.2f}")
    log("-" * 80)
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
        f"{variant.name} TEST clean_training_data",
    )
    p_over_test, p_under_test, p_push_test, matched_test, cov_test = join_and_price(
        df_bet_test,
        dist_preds_test,
        dist,
        log,
        f"{variant.name} TEST betting frame",
    )
    test_bundle, test_auc, test_brier = evaluate_test_once(
        df_bet_test,
        p_over_test,
        p_under_test,
        matched_test,
        winner_threshold,
        log,
        f"{variant.name} TEST",
    )
    log(
        f"{variant.name} TEST: {test_bundle['summary']['bets']} bets, "
        f"{test_bundle['summary']['bet_rate']:.1f}% bet rate, "
        f"{test_bundle['summary']['hit_rate']:.1f}% hit rate, "
        f"{test_bundle['summary']['roi']:+.2f}% ROI"
    )
    log(
        f"{variant.name} TEST ROI 95% CI (row): "
        f"[{test_bundle['roi_ci_row']['lower']:+.2f}%, {test_bundle['roi_ci_row']['upper']:+.2f}%]"
    )
    log(
        f"{variant.name} TEST ROI 95% CI (cluster): "
        f"[{test_bundle['roi_ci_cluster']['lower']:+.2f}%, {test_bundle['roi_ci_cluster']['upper']:+.2f}%] "
        f"(n_clusters={test_bundle['roi_ci_cluster']['n_clusters']})"
    )
    log(
        f"{variant.name} TEST side breakdown: OVER "
        f"{test_bundle['side_breakdown']['OVER']['bets']} bets "
        f"({test_bundle['side_breakdown']['OVER']['roi']:+.2f}%), UNDER "
        f"{test_bundle['side_breakdown']['UNDER']['bets']} bets "
        f"({test_bundle['side_breakdown']['UNDER']['roi']:+.2f}%)"
    )
    log(
        f"{variant.name} TEST goalie-nights: {test_bundle['goalie_nights_total']} total, "
        f"{test_bundle['goalie_nights_bet']} with a bet"
    )
    log(
        f"{variant.name} TEST AUC row-level={test_auc['row_level']:.4f} "
        f"one-per-goalie-night={test_auc['one_per_goalie_night']:.4f}; "
        f"Brier={test_brier:.5f}"
    )

    return {
        "variant": variant.name,
        "description": variant.description,
        "primary": variant.primary,
        "shots_use_context": variant.shots_use_context,
        "rate_use_context": variant.rate_use_context,
        "shots_feature_count": len(shots_cols),
        "rate_feature_count": len(rate_cols),
        "shots_feature_cols": shots_cols,
        "rate_feature_cols": rate_cols,
        "shots_model": {
            "configs": [{"name": n, "params": c} for n, c in SHOTS_CONFIGS],
            "val_evaluations": shots_evals,
            "winner": shots_winner,
            "top10_features": shots_top10,
            "model_path": str(shots_path),
        },
        "dispersion": {"alpha": alpha, "method": dispersion_method, "diagnostics": dispersion_diag},
        "save_rate_model": {
            "configs": [{"name": n, "params": c} for n, c in SAVE_RATE_CONFIGS],
            "val_evaluations": rate_evals,
            "winner": rate_winner,
            "top10_features": rate_top10,
            "model_path": str(rate_path),
        },
        "intrinsic_quality_val": intrinsic_val,
        "join_coverage": {"val_pct": cov_val, "test_pct": cov_test},
        "val_fold_wide_auc": val_auc,
        "val_fold_wide_brier": val_brier,
        "val_betting_sweep": [
            {"threshold": e["threshold"], "summary": e["summary"]} for e in val_evaluations
        ],
        "val_winner": {
            "threshold": winner_threshold,
            "summary": val_winner["summary"],
            **val_bundle,
        },
        "selection_deviation": selection_deviation,
        "test_single_touch": {
            "threshold": winner_threshold,
            **test_bundle,
        },
        "touch_count_audit": {
            "shots_model_val_evaluations": len(shots_evals),
            "save_rate_model_val_evaluations": len(rate_evals),
            "betting_val_evaluations": len(val_evaluations),
            "betting_test_evaluations": 1,
        },
    }


def main() -> int:
    start_time = time.time()

    for path in (DATA_PATH_CLEAN, DATA_PATH_MULTIBOOK):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input parquet: {path}")
    if not DATA_PATH_CONTEXT.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH_CONTEXT}. Run scripts/build_game_context_features.py first, "
            "then rerun this experiment."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_game_context_distributional_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log, flush_log = make_logger(log_path)

    try:
        log("=" * 80)
        log("GAME-CONTEXT DISTRIBUTIONAL SAVES EXPERIMENT")
        log(f"Run timestamp: {datetime.now().isoformat()}")
        log("=" * 80)
        log(f"Output directory: {output_dir}")

        frame = load_modeling_frame(DATA_PATH_CLEAN, DATA_PATH_CONTEXT, log)
        clean_split = split_by_date(frame.df, log, "clean_training_data")

        df_bet = build_betting_frame(DATA_PATH_MULTIBOOK, log)
        bet_split = split_by_date(df_bet, log, "multibook_classification_training_data")
        df_bet_val = df_bet.iloc[bet_split.val_idx].reset_index(drop=True)
        df_bet_test = df_bet.iloc[bet_split.test_idx].reset_index(drop=True)

        results = {}
        for variant in VARIANTS:
            results[variant.name] = run_variant(
                variant,
                frame,
                clean_split,
                df_bet_val,
                df_bet_test,
                output_dir,
                log,
            )
            flush_log()

        log("\n" + "=" * 80)
        log("HEAD-TO-HEAD SUMMARY (single-touch test results)")
        log("=" * 80)
        log(f"{'variant':<16} {'role':<9} {'roi':>9} {'bets':>6} {'auc_row':>9} {'auc_night':>10} {'brier':>9}")
        for variant in VARIANTS:
            row = results[variant.name]["test_single_touch"]
            role = "primary" if variant.primary else "secondary"
            log(
                f"{variant.name:<16} {role:<9} {row['summary']['roi']:>+8.2f}% "
                f"{row['summary']['bets']:>6} {row['auc']['row_level']:>9.4f} "
                f"{row['auc']['one_per_goalie_night']:>10.4f} {row['brier']:>9.5f}"
            )
        log("Do not interpret any point estimate as an edge without the CI/chronological context above.")

        elapsed = time.time() - start_time
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "wall_clock_seconds": elapsed,
            "data_paths": {
                "clean": str(DATA_PATH_CLEAN),
                "multibook": str(DATA_PATH_MULTIBOOK),
                "context": str(DATA_PATH_CONTEXT),
            },
            "cap": CAP,
            "ev_thresholds": EV_THRESHOLDS,
            "variant_specs": [variant.__dict__ for variant in VARIANTS],
            "context": {
                "join_keys": frame.context_join_keys,
                "coverage_pct": frame.context_coverage_pct,
                "raw_feature_cols": frame.context_raw_cols,
                "prefixed_feature_cols": frame.context_cols,
                "null_counts": frame.context_null_counts,
            },
            "fold_boundaries_clean_training_data": clean_split.boundaries,
            "fold_boundaries_multibook": bet_split.boundaries,
            "results": results,
            "protocol_notes": [
                "Feature variants are pre-registered in script code.",
                "Submodel selection used validation only.",
                "EV threshold selection used validation only.",
                "Each variant has exactly one test-fold betting evaluation after validation selection.",
                "context_both is secondary and should not override the primary control vs context_shots comparison.",
            ],
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        log(f"\nSaved metadata to: {metadata_path}")
        log(f"Wall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        log("\n" + "=" * 80)
        log("EXPERIMENT COMPLETE")
        log("=" * 80)
        flush_log()
    except Exception:
        flush_log()
        raise

    print(f"Saved run log to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
