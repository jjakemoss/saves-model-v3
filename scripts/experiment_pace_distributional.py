"""
Roadmap item 8 / section 3.14 Component 3: pace/xG distributional experiment.

This is the pre-registered pace follow-up to the game-context distributional
experiment. It reuses the existing distributional harness unchanged and only
adds the Component 2 pace feature artifact to the modeling frame.

Variants:
  control:              exact rerun of the 3.12 no-context control
  pace_shots:           pace families 1-4 and 6 in shots model only
  pace_context_shots:   pace families 1-4 and 6 plus game-context in shots
  pace_both:            pace+context in shots, goalie workload family 5 in rate

Protocol:
  train = game_date < 2025-10-16
  val   = 2025-10-16 through 2025-12-03
  test  = game_date >= 2025-12-04

  Submodel config selection is validation-only. Betting policy selection uses
  the same probability-edge EV threshold grid as section 3.12:
  [0.05, 0.10, 0.12, 0.15]. Each variant gets exactly one test evaluation
  after validation selection.

Artifacts:
  models/trained/experiment_pace_distributional_{timestamp}/
    run_log.txt
    metadata.json
    {variant}_shots_model.json
    {variant}_save_rate_model.json

Usage:
    python scripts/experiment_pace_distributional.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
DATA_PATH_PACE = Path("data/processed/pace_features.parquet")
DATA_PATH_PACE_METADATA = Path("data/processed/pace_features_metadata.json")
OUTPUT_ROOT = Path("models/trained")
MARKET_BRIER_BENCHMARK = 0.24961

CONTROL_EXPECTED = {
    "brier_rounded": 0.25487,
    "roi_rounded": 1.06,
    "bets": 888,
}

PACE_SHOTS_FAMILIES = [
    "opponent_offense_pace",
    "team_shot_suppression",
    "combined_pace",
    "special_teams_volume",
    "league_relative_zscores",
]
PACE_RATE_FAMILIES = ["goalie_workload_quality"]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    shots_use_context: bool
    shots_use_pace: bool
    rate_use_goalie_workload: bool
    primary: bool


VARIANTS = [
    VariantSpec(
        name="control",
        description="Exact no-context distributional setup from section 3.12 control.",
        shots_use_context=False,
        shots_use_pace=False,
        rate_use_goalie_workload=False,
        primary=True,
    ),
    VariantSpec(
        name="pace_shots",
        description="Pace families 1-4 and 6 added to the shots-against model only.",
        shots_use_context=False,
        shots_use_pace=True,
        rate_use_goalie_workload=False,
        primary=True,
    ),
    VariantSpec(
        name="pace_context_shots",
        description="Pace families 1-4 and 6 plus game-context features in the shots-against model.",
        shots_use_context=True,
        shots_use_pace=True,
        rate_use_goalie_workload=False,
        primary=True,
    ),
    VariantSpec(
        name="pace_both",
        description="Pace+context in shots; goalie workload-quality pace family in save-rate model.",
        shots_use_context=True,
        shots_use_pace=True,
        rate_use_goalie_workload=True,
        primary=False,
    ),
]
assert EV_THRESHOLDS == [0.05, 0.10, 0.12, 0.15]
assert len(SHOTS_CONFIGS) <= 6
assert len(SAVE_RATE_CONFIGS) <= 6


@dataclass(frozen=True)
class PaceModelingFrame:
    df: pd.DataFrame
    base_feature_cols: list[str]
    engineered_cols: list[str]
    context_cols: list[str]
    context_raw_cols: list[str]
    context_null_counts: dict
    context_join_keys: list[str]
    context_coverage_pct: float
    pace_join_keys: list[str]
    pace_cols: list[str]
    pace_shots_cols: list[str]
    pace_goalie_workload_cols: list[str]
    pace_family_columns: dict[str, list[str]]
    pace_null_counts: dict
    pace_coverage: dict
    pace_metadata: dict


def make_logger(log_path: Path):
    log_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(str(msg))

    def flush_log() -> None:
        log_path.write_text("\n".join(log_lines), encoding="utf-8")

    return log, flush_log


def load_pace_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/build_pace_features.py before Component 3."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _pace_family_cols(metadata: dict, families: list[str]) -> list[str]:
    family_columns = metadata.get("family_columns")
    if not isinstance(family_columns, dict):
        raise ValueError("pace metadata missing family_columns.")

    cols: list[str] = []
    for family in families:
        family_cols = family_columns.get(family)
        if not family_cols:
            raise ValueError(f"pace metadata missing expected family: {family}")
        cols.extend(family_cols)
    return cols


def load_pace_modeling_frame(
    clean_path: Path,
    context_path: Path,
    pace_path: Path,
    pace_metadata_path: Path,
    log,
) -> PaceModelingFrame:
    frame = load_modeling_frame(clean_path, context_path, log)

    if not pace_path.exists():
        raise FileNotFoundError(
            f"Missing {pace_path}. Run scripts/build_pace_features.py before Component 3."
        )
    metadata = load_pace_metadata(pace_metadata_path)
    pace_join_keys = list(metadata.get("key_columns", []))
    pace_cols = list(metadata.get("generated_columns", []))
    if not pace_join_keys or not pace_cols:
        raise ValueError("pace metadata must include key_columns and generated_columns.")

    pace_df = pd.read_parquet(pace_path)
    log(f"Raw pace_features.parquet: {len(pace_df)} rows, {len(pace_df.columns)} columns.")
    missing_keys = [c for c in pace_join_keys if c not in pace_df.columns or c not in frame.df.columns]
    missing_features = [c for c in pace_cols if c not in pace_df.columns]
    if missing_keys:
        raise ValueError(f"Cannot merge pace features; missing join key(s): {missing_keys}")
    if missing_features:
        raise ValueError(f"pace_features.parquet missing generated columns: {missing_features}")
    if pace_df.duplicated(pace_join_keys).any():
        dupes = pace_df.loc[pace_df.duplicated(pace_join_keys, keep=False), pace_join_keys]
        raise ValueError(f"pace_features.parquet has duplicate keys. Examples:\n{dupes.head(10)}")

    df = frame.df.copy()
    if "game_date" in pace_join_keys:
        df["game_date"] = pd.to_datetime(df["game_date"])
        pace_df = pace_df.copy()
        pace_df["game_date"] = pd.to_datetime(pace_df["game_date"])

    overlapping_features = sorted(set(pace_cols) & set(df.columns))
    if overlapping_features:
        raise ValueError(f"Pace feature names collide with modeling frame columns: {overlapping_features}")

    keep_cols = pace_join_keys + [c for c in metadata.get("verification_columns", []) if c in pace_df.columns] + pace_cols
    keep_cols = list(dict.fromkeys(keep_cols))
    before = len(df)
    df = df.merge(pace_df[keep_cols], how="left", on=pace_join_keys, indicator="_pace_merge")
    if len(df) != before:
        raise AssertionError("Pace merge changed clean-training row count.")

    matched = df["_pace_merge"].eq("both")
    pace_coverage_pct = float(matched.mean() * 100) if len(df) else 0.0
    log(
        f"Pace join keys: {pace_join_keys}; coverage "
        f"{int(matched.sum())}/{len(df)} ({pace_coverage_pct:.2f}%)."
    )
    if pace_coverage_pct < 99.0:
        raise AssertionError("Pace feature coverage below 99%; investigate before running experiment.")
    df = df.drop(columns=["_pace_merge"])

    pace_shots_cols = _pace_family_cols(metadata, PACE_SHOTS_FAMILIES)
    pace_goalie_workload_cols = _pace_family_cols(metadata, PACE_RATE_FAMILIES)
    missing_variant_cols = [
        c for c in pace_shots_cols + pace_goalie_workload_cols if c not in df.columns
    ]
    if missing_variant_cols:
        raise ValueError(f"Merged frame missing pace variant columns: {missing_variant_cols}")

    for label, cols in {
        "pace all": pace_cols,
        "pace shots families 1-4,6": pace_shots_cols,
        "pace goalie workload family 5": pace_goalie_workload_cols,
    }.items():
        mat = df[cols].values.astype(np.float64)
        n_inf = int(np.isinf(mat).sum())
        if n_inf:
            raise AssertionError(f"{label} feature matrix contains {n_inf} infinite values.")

    pace_null_counts = {
        c: int(v) for c, v in df[pace_cols].isna().sum().items() if int(v) > 0
    }
    if pace_null_counts:
        top_nulls = dict(sorted(pace_null_counts.items(), key=lambda kv: kv[1], reverse=True)[:12])
        log(
            "Pace feature NaNs retained for XGBoost missing-value handling "
            f"(top nonzero null counts: {top_nulls})."
        )

    coverage = {}
    for col in ("pace_team_all_matched", "pace_opponent_all_matched", "pace_goalie_all_matched"):
        if col in df.columns:
            coverage[col] = {
                "match_rate_pct": float(df[col].astype(bool).mean() * 100),
                "missing_rows": int((~df[col].astype(bool)).sum()),
            }
    if coverage:
        log(f"Pace source coverage flags: {coverage}")

    log(f"Pace shots feature columns: {len(pace_shots_cols)} -> {pace_shots_cols}")
    log(f"Pace goalie-workload feature columns: {len(pace_goalie_workload_cols)} -> {pace_goalie_workload_cols}")

    return PaceModelingFrame(
        df=df,
        base_feature_cols=frame.base_feature_cols,
        engineered_cols=frame.engineered_cols,
        context_cols=frame.context_cols,
        context_raw_cols=frame.context_raw_cols,
        context_null_counts=frame.context_null_counts,
        context_join_keys=frame.context_join_keys,
        context_coverage_pct=frame.context_coverage_pct,
        pace_join_keys=pace_join_keys,
        pace_cols=pace_cols,
        pace_shots_cols=pace_shots_cols,
        pace_goalie_workload_cols=pace_goalie_workload_cols,
        pace_family_columns=metadata["family_columns"],
        pace_null_counts=pace_null_counts,
        pace_coverage=coverage,
        pace_metadata=metadata,
    )


def feature_cols_for_variant(frame: PaceModelingFrame, variant: VariantSpec) -> tuple[list[str], list[str]]:
    base = frame.base_feature_cols + frame.engineered_cols
    shots_cols = list(base)
    if variant.shots_use_context:
        shots_cols += frame.context_cols
    if variant.shots_use_pace:
        shots_cols += frame.pace_shots_cols

    rate_cols = list(base)
    if variant.rate_use_goalie_workload:
        rate_cols += frame.pace_goalie_workload_cols

    if len(shots_cols) != len(set(shots_cols)):
        raise ValueError(f"{variant.name} shots feature list contains duplicates.")
    if len(rate_cols) != len(set(rate_cols)):
        raise ValueError(f"{variant.name} save-rate feature list contains duplicates.")
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


def assert_control_gate(result: dict, log) -> None:
    summary = result["test_single_touch"]["summary"]
    brier = float(result["test_single_touch"]["brier"])
    checks = {
        "brier_rounded": round(brier, 5),
        "roi_rounded": round(float(summary["roi"]), 2),
        "bets": int(summary["bets"]),
    }
    if checks != CONTROL_EXPECTED:
        raise AssertionError(
            "Control integrity gate failed. Expected "
            f"{CONTROL_EXPECTED}, observed {checks}. Do not interpret pace variants."
        )
    log(
        "CONTROL GATE PASSED: reproduced section 3.12 control "
        f"Brier={brier:.5f}, ROI={summary['roi']:+.2f}%, bets={summary['bets']}."
    )


def run_variant(
    variant: VariantSpec,
    frame: PaceModelingFrame,
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
        "shots_use_pace": variant.shots_use_pace,
        "rate_use_goalie_workload": variant.rate_use_goalie_workload,
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

    for path in (DATA_PATH_CLEAN, DATA_PATH_MULTIBOOK, DATA_PATH_CONTEXT, DATA_PATH_PACE, DATA_PATH_PACE_METADATA):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_pace_distributional_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log, flush_log = make_logger(log_path)

    try:
        log("=" * 80)
        log("PACE/XG DISTRIBUTIONAL SAVES EXPERIMENT")
        log(f"Run timestamp: {datetime.now().isoformat()}")
        log("=" * 80)
        log(f"Output directory: {output_dir}")
        log(f"Market Brier benchmark from section 3.14: {MARKET_BRIER_BENCHMARK:.5f}")

        frame = load_pace_modeling_frame(
            DATA_PATH_CLEAN,
            DATA_PATH_CONTEXT,
            DATA_PATH_PACE,
            DATA_PATH_PACE_METADATA,
            log,
        )
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
            if variant.name == "control":
                assert_control_gate(results[variant.name], log)
            flush_log()

        log("\n" + "=" * 80)
        log("HEAD-TO-HEAD SUMMARY (single-touch test results)")
        log("=" * 80)
        log(
            f"{'variant':<20} {'role':<9} {'roi':>9} {'bets':>6} "
            f"{'auc_row':>9} {'auc_night':>10} {'brier':>9} {'vs_mkt':>9}"
        )
        for variant in VARIANTS:
            row = results[variant.name]["test_single_touch"]
            role = "primary" if variant.primary else "secondary"
            brier = float(row["brier"])
            log(
                f"{variant.name:<20} {role:<9} {row['summary']['roi']:>+8.2f}% "
                f"{row['summary']['bets']:>6} {row['auc']['row_level']:>9.4f} "
                f"{row['auc']['one_per_goalie_night']:>10.4f} {brier:>9.5f} "
                f"{(brier - MARKET_BRIER_BENCHMARK):>+9.5f}"
            )
        log("vs_mkt is model Brier minus the de-vigged market Brier; lower is better.")
        log("Do not interpret any point estimate as an edge without the CI/chronological context above.")

        elapsed = time.time() - start_time
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "wall_clock_seconds": elapsed,
            "data_paths": {
                "clean": str(DATA_PATH_CLEAN),
                "multibook": str(DATA_PATH_MULTIBOOK),
                "context": str(DATA_PATH_CONTEXT),
                "pace": str(DATA_PATH_PACE),
                "pace_metadata": str(DATA_PATH_PACE_METADATA),
            },
            "cap": CAP,
            "market_brier_benchmark": MARKET_BRIER_BENCHMARK,
            "control_expected_gate": CONTROL_EXPECTED,
            "ev_thresholds": EV_THRESHOLDS,
            "variant_specs": [variant.__dict__ for variant in VARIANTS],
            "context": {
                "join_keys": frame.context_join_keys,
                "coverage_pct": frame.context_coverage_pct,
                "raw_feature_cols": frame.context_raw_cols,
                "prefixed_feature_cols": frame.context_cols,
                "null_counts": frame.context_null_counts,
            },
            "pace": {
                "join_keys": frame.pace_join_keys,
                "generated_feature_cols": frame.pace_cols,
                "shots_feature_cols": frame.pace_shots_cols,
                "goalie_workload_feature_cols": frame.pace_goalie_workload_cols,
                "family_columns": frame.pace_family_columns,
                "null_counts": frame.pace_null_counts,
                "coverage": frame.pace_coverage,
                "source_metadata": frame.pace_metadata,
            },
            "fold_boundaries_clean_training_data": clean_split.boundaries,
            "fold_boundaries_multibook": bet_split.boundaries,
            "results": results,
            "protocol_notes": [
                "Component 3 uses the pre-registered date folds from section 3.14.",
                "The control variant must reproduce section 3.12 before pace variants are interpreted.",
                "Submodel selection used validation only.",
                "Betting threshold selection used the section 3.12 EV threshold grid on validation only.",
                "Each variant has exactly one test-fold betting evaluation after validation selection.",
                "pace_both is secondary and should not override the primary control vs pace_shots/pace_context_shots comparison.",
                "The Dec 2025-Apr 2026 test fold is worn from prior experiments; positive results require next-season confirmation.",
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
