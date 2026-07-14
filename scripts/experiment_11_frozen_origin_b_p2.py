"""Experiment 11: frozen-Origin-B 2024-25 bettime P2 re-test.

Implements section 14 of docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md.
The closing wiring gate intentionally runs before this script opens the new
core_bettime_202607 parquet.  This is inference only: all three Origin B
XGBoost JSON artifacts and their recorded configurations are reloaded.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import clv_audit_pace_policy as clv  # noqa: E402
import experiment_market_state_features as emsf  # noqa: E402
import experiment_rolling_origin as ero  # noqa: E402
from experiment_market_state_origin_c import (  # noqa: E402
    add_devig_cols,
    attach_clv,
    build_closing_consensus,
    build_p2_universe,
    compute_drift_baseline,
    flag_bets_full_policy,
    p2_paired_bootstrap,
    side_breakdown_for_subset,
)
from experiment_pace_distributional import make_logger  # noqa: E402
from experiments.distributional_saves import (  # noqa: E402
    SavesDistribution,
    build_betting_frame,
    compute_distribution_predictions,
    fit_dispersion,
    join_and_price,
    load_modeling_frame,
)


FROZEN_DIR = REPO_ROOT / "models" / "trained" / "experiment_market_state_20260710_213106"
FROZEN_METADATA = FROZEN_DIR / "metadata.json"
MODEL_PATHS = {
    "control_plus_market_state": FROZEN_DIR / "origin_b_control_plus_market_state_shots_model.json",
    "no_pace_control": FROZEN_DIR / "origin_b_no_pace_control_shots_model.json",
    "shared_save_rate": FROZEN_DIR / "origin_b_shared_save_rate_model.json",
}
DATA_CLEAN = REPO_ROOT / "data" / "processed" / "clean_training_data.parquet"
DATA_CONTEXT = REPO_ROOT / "data" / "processed" / "game_context_features.parquet"
DATA_MULTIBOOK = REPO_ROOT / "data" / "processed" / "multibook_classification_training_data.parquet"
DATA_MARKET = REPO_ROOT / "data" / "processed" / "market_game_features.parquet"
DATA_NEW_SNAPSHOTS = REPO_ROOT / "data" / "processed" / "core_bettime_202607_snapshots.parquet"
NEW_SUMMARY = REPO_ROOT / "data" / "processed" / "core_bettime_202607_snapshots_summary.json"
OUTPUT_ROOT = REPO_ROOT / "models" / "trained"
STOPPED_GATE_ATTEMPT = OUTPUT_ROOT / "experiment_11_frozen_origin_b_p2_20260714_085614"

SEASON = 20242025
EXPECTED_GATE_MEAN = -0.0041404240194266384
EXPECTED_GATE_N_BETS = 7463
EXPECTED_GATE_N_CLUSTERS = 2510
GATE_TOLERANCE = 1e-4
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
EV_THRESHOLD = 0.05
ORIGIN_CAP = 90
P2_MIN_MODEL_ARM_BETS = 100
P2_MAX_EMPTY_RESAMPLE_PCT = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(output_dir: Path, metadata: dict) -> None:
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def load_json_model(path: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def assert_parity(a: np.ndarray, b: np.ndarray, label: str) -> float:
    diff = float(np.nanmax(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))) if len(a) else 0.0
    if diff >= 1e-9:
        raise AssertionError(f"{label}: frozen JSON reload parity {diff} is not < 1e-9.")
    return diff


def input_checksums(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")
        result[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def frozen_recipe(meta: dict) -> dict:
    origin = meta["origin_b"]
    variants = origin["variants"]
    market = variants["control_plus_market_state"]
    control = variants["no_pace_control"]
    rate = origin["rate_model"]
    recipe = {
        "market_state_feature_cols": market["shots_feature_cols"],
        "control_feature_cols": control["shots_feature_cols"],
        "rate_feature_cols": rate["feature_cols"],
        "market_alpha": market["dispersion"]["alpha"],
        "control_alpha": control["dispersion"]["alpha"],
        "market_config": market["shots_model"]["winner"],
        "control_config": control["shots_model"]["winner"],
        "rate_config": rate["winner"],
        "fold_boundaries": origin["fold_boundaries"],
    }
    if meta["fixed_ev_threshold"] != EV_THRESHOLD or meta["origin_cap"] != ORIGIN_CAP:
        raise AssertionError("Frozen metadata threshold or Origin cap does not match the registered Experiment 11 recipe.")
    return recipe


def build_gate_state(log, recipe: dict) -> dict:
    """The mandatory, self-contained closing gate.  It does not reference
    DATA_NEW_SNAPSHOTS in any form."""
    log("Loading only frozen-gate inputs; new bettime snapshots remain unopened.")
    frame = load_modeling_frame(DATA_CLEAN, DATA_CONTEXT, log)
    events, market_stats = emsf.build_market_state_events(DATA_MARKET, log)
    df_full = emsf.attach_market_state_features(frame.df, events, log)
    control_cols = recipe["control_feature_cols"]
    market_cols = recipe["market_state_feature_cols"]
    reconstructed_control = frame.base_feature_cols + frame.engineered_cols
    reconstructed_market = reconstructed_control + emsf.ALL_MARKET_COLS
    if reconstructed_control != control_cols or reconstructed_market != market_cols:
        raise AssertionError("Frozen 104/112-column feature lists do not match the registered Origin B recipe.")
    if recipe["rate_feature_cols"] != control_cols:
        raise AssertionError("Frozen shared rate-model feature list differs from the registered no-pace feature list.")

    clean = pd.read_parquet(DATA_CLEAN)
    clean["game_date"] = pd.to_datetime(clean["game_date"])
    pool_min, pool_max = ero.season_date_range(clean, [ero.SEASON_2022_23, ero.SEASON_2023_24])
    train_idx, val_idx, boundaries = ero.carve_origin_split(
        df_full, pool_min, pool_max, ero.VAL_WINDOW_DAYS, log, "Experiment 11 Origin B gate"
    )
    test_min, test_max = ero.season_date_range(clean, [SEASON])
    test_idx = ero.date_range_test_idx(df_full, test_min, test_max, log, "Experiment 11 Origin B gate")
    if boundaries != recipe["fold_boundaries"] | {"test_season": SEASON, "test_rows": 2624}:
        # The persisted boundary object omits no fields here except the test values added below.
        expected = recipe["fold_boundaries"]
        for key in ("pool_date_range", "train", "val", "val_window_days"):
            if boundaries[key] != expected[key]:
                raise AssertionError(f"Frozen fold boundary mismatch for {key}: {boundaries[key]} != {expected[key]}")
    if len(test_idx) != recipe["fold_boundaries"]["test_rows"]:
        raise AssertionError("Origin B test-fold row count differs from frozen metadata.")

    closing = build_betting_frame(DATA_MULTIBOOK, log)
    closing = closing[closing["season"] == SEASON].reset_index(drop=True)
    rate = load_json_model(MODEL_PATHS["shared_save_rate"])
    control = load_json_model(MODEL_PATHS["no_pace_control"])
    market = load_json_model(MODEL_PATHS["control_plus_market_state"])
    dist = SavesDistribution(ORIGIN_CAP)
    predictions: dict[str, dict] = {}
    rows: list[pd.DataFrame] = []
    for variant, model, cols, alpha in (
        ("no_pace_control", control, control_cols, recipe["control_alpha"]),
        ("control_plus_market_state", market, market_cols, recipe["market_alpha"]),
    ):
        pred = compute_distribution_predictions(
            df_full, test_idx, model, rate, alpha, cols, control_cols, dist, log, f"gate {variant}"
        )
        p_over, p_under, p_push, matched, coverage = join_and_price(
            closing, pred, dist, log, f"gate {variant} closing"
        )
        predictions[variant] = {
            "model": model, "cols": cols, "alpha": alpha, "dist": pred,
            "p_over_closing": p_over, "p_under_closing": p_under,
            "p_push_closing": p_push, "matched_closing": matched, "coverage_closing": coverage,
        }
        rows.append(pd.DataFrame({
            "variant": variant,
            "game_id": df_full["game_id"].values[test_idx],
            "goalie_id": df_full["goalie_id"].values[test_idx],
            "mu_shots_against": pred["mu"], "save_rate": pred["q"], "alpha": alpha,
        }))

    gate = emsf.paired_brier_delta_vs_variant(
        closing,
        predictions["no_pace_control"]["p_over_closing"], predictions["no_pace_control"]["matched_closing"],
        predictions["control_plus_market_state"]["p_over_closing"], predictions["control_plus_market_state"]["matched_closing"],
        log, "Experiment 11 frozen-Origin-B closing wiring gate",
    )
    mean_diff = abs(gate["mean"] - EXPECTED_GATE_MEAN)
    passed = (
        mean_diff <= GATE_TOLERANCE
        and gate["n_bets"] == EXPECTED_GATE_N_BETS
        and gate["n_clusters"] == EXPECTED_GATE_N_CLUSTERS
    )
    log(f"Gate expected: mean={EXPECTED_GATE_MEAN}, n_bets={EXPECTED_GATE_N_BETS}, n_clusters={EXPECTED_GATE_N_CLUSTERS}")
    log(f"Gate observed: mean={gate['mean']}, n_bets={gate['n_bets']}, n_clusters={gate['n_clusters']}")
    log(f"Gate absolute mean difference={mean_diff:.12f}; tolerance={GATE_TOLERANCE}; passed={passed}")
    return {
        "frame": df_full, "clean": clean, "test_idx": test_idx, "train_idx": train_idx,
        "closing": closing, "dist": dist, "recipe": recipe, "predictions": predictions,
        "gate": gate, "gate_mean_abs_diff": mean_diff, "gate_passed": passed,
        "market_source_stats": market_stats, "fold_boundaries": {**boundaries, "test_season": SEASON, "test_rows": int(len(test_idx))},
        "prediction_rows": pd.concat(rows, ignore_index=True),
    }


def normalize_new_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "event_id", "snapshot_pass", "market_key", "book_key", "goalie_id", "side", "line", "price_decimal", "true_commence_time", "game_date_eastern"}
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"New snapshot schema missing required columns: {sorted(missing)}")
    out = df[(df["season"] == "2024-25") & (df["snapshot_pass"] == "bettime") & (df["market_key"] == "player_total_saves")].copy()
    out = out.rename(columns={"book_key": "book", "true_commence_time": "commence_time"})
    out["source"] = "new_core_bettime_202607"
    return out


def one_touch_already_completed() -> bool:
    for metadata_path in OUTPUT_ROOT.glob("experiment_11_frozen_origin_b_p2_*/metadata.json"):
        try:
            if json.loads(metadata_path.read_text(encoding="utf-8")).get("p2_touch_completed"):
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False


def sign_flip(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and a != 0 and b != 0 and (a > 0) != (b > 0)


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"experiment_11_frozen_origin_b_p2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log, flush_log = make_logger(output_dir / "run_log.txt")
    metadata: dict = {
        "timestamp": datetime.now().isoformat(),
        "registration": "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 14 (Experiment 11)",
        "network_calls": False, "betting_db_touched": False, "deviations_from_registration": [],
        "p2_touch_completed": False,
        "notes": [
            "Section 14.3a controls the bettime source population: core_bettime_202607 snapshots only.",
            "The old saves_lines_snapshots.parquet fragment contributes zero Experiment 11 quotes and is never loaded.",
            "The prior gate-only stopped attempt did not calculate model probabilities, prices, outcome grading, bootstrap, or secondaries; this is the first P2 touch.",
        ],
    }
    started = time.time()
    try:
        log("EXPERIMENT 11 -- FROZEN-ORIGIN-B P2 RE-TEST (SECTION 14)")
        if one_touch_already_completed():
            raise RuntimeError("A completed Experiment 11 P2 touch already exists; refusing a forbidden re-run.")

        # New parquet intentionally absent from this checksum/input phase and every gate function above.
        metadata["gate_input_checksums"] = input_checksums({
            "frozen_metadata": FROZEN_METADATA, "market_state_shots": MODEL_PATHS["control_plus_market_state"],
            "no_pace_control_shots": MODEL_PATHS["no_pace_control"], "shared_save_rate": MODEL_PATHS["shared_save_rate"],
            "clean_training": DATA_CLEAN, "game_context": DATA_CONTEXT, "market_game_features": DATA_MARKET,
            "multibook_closing": DATA_MULTIBOOK,
        })
        frozen_meta = json.loads(FROZEN_METADATA.read_text(encoding="utf-8"))
        recipe = frozen_recipe(frozen_meta)
        metadata["frozen_recipe"] = recipe

        log("STEP 1: mandatory closing wiring gate. New core bettime parquet has not been opened.")
        state = build_gate_state(log, recipe)
        metadata["wiring_gate"] = {
            "expected": {"mean": EXPECTED_GATE_MEAN, "n_bets": EXPECTED_GATE_N_BETS, "n_clusters": EXPECTED_GATE_N_CLUSTERS},
            "observed": state["gate"], "mean_abs_diff": state["gate_mean_abs_diff"], "tolerance": GATE_TOLERANCE,
            "passed": state["gate_passed"], "fold_boundaries": state["fold_boundaries"],
        }
        state["prediction_rows"].to_parquet(output_dir / "gate_predictions.parquet", index=False)
        if not state["gate_passed"]:
            metadata["stopped_at"] = "wiring_gate_before_new_parquet_load"
            metadata["exact_verdict"] = "WIRING_GATE_FAILED"
            write_metadata(output_dir, metadata)
            log("WIRING GATE FAILED. STOPPING BEFORE NEW PARQUET LOAD.")
            flush_log()
            return 1

        log("WIRING GATE PASSED. STEP 2: opening the section 14.3a new-pass-only bettime population exactly once.")
        post_gate_paths = {"new_core_bettime_snapshots": DATA_NEW_SNAPSHOTS, "new_snapshot_summary": NEW_SUMMARY}
        metadata["post_gate_input_checksums"] = input_checksums(post_gate_paths)
        if not STOPPED_GATE_ATTEMPT.exists():
            raise FileNotFoundError(f"Required stopped gate-only audit artifact is missing: {STOPPED_GATE_ATTEMPT}")
        metadata["section_14_3a_clarification"] = {
            "source_population": "data/processed/core_bettime_202607_snapshots.parquet only",
            "old_fragment_quotes_used": 0,
            "stopped_attempt": {
                "path": str(STOPPED_GATE_ATTEMPT),
                "metadata_sha256": sha256(STOPPED_GATE_ATTEMPT / "metadata.json"),
                "exact_verdict": "REGISTRATION_AMBIGUITY",
                "p2_touch_completed": False,
            },
        }
        metadata["deviations_from_registration"].append(
            "Section 14.3a clarification, recorded before any P2 statistic: the new core_bettime_202607 pass is the entire bettime population; the old 21-event fragment is not loaded or appended."
        )
        summary = json.loads(NEW_SUMMARY.read_text(encoding="utf-8"))
        new_raw = pd.read_parquet(DATA_NEW_SNAPSHOTS)
        new_norm = normalize_new_snapshots(new_raw)
        base_2024_25 = state["clean"][state["clean"]["season"] == SEASON].reset_index(drop=True)
        betonline_events = int(new_norm.loc[new_norm["book"] == "betonlineag", "event_id"].nunique())
        if betonline_events != 1050:
            raise AssertionError(f"Section 14.3a primary source has {betonline_events} BetOnline saves events, expected 1050.")
        if (new_norm["book"] == "fanatics").any():
            raise AssertionError("Fanatics appeared in the new-only source after the registered schema gate.")
        ingestion = {
            "source_population": "new_core_bettime_202607_only",
            "new_rows": int(len(new_norm)),
            "new_events": int(new_norm["event_id"].nunique()),
            "betonlineag_events": betonline_events,
            "old_fragment_rows_used": 0,
            "old_fragment_loaded": False,
            "registered_overlap_event_ids_not_merged": int(len(summary["overlap_with_existing_2024_25_bettime_archive"]["overlapping_event_ids"])),
        }
        log(f"14.3a source population: {len(new_norm)} new-pass saves rows, {ingestion['new_events']} events, {betonline_events} BetOnline events; old fragment not loaded.")
        metadata["ingestion"] = ingestion
        new_norm.to_parquet(output_dir / "bettime_snapshot_rows_new_pass_only.parquet", index=False)

        log("STEP 3: reuse build_season_multibook_frame / pivot_both_sides for the paired 2024-25 quote frame.")
        bettime = ero.build_season_multibook_frame(base_2024_25, new_norm, "bettime", log)
        books = sorted(bettime["book_key"].unique().tolist())
        if "fanatics" in books:
            raise AssertionError("Fanatics appeared after the registered schema gate; stopping before P2.")
        if "betonline" in books:
            raise AssertionError("Unexpected 'betonline' key in new bettime frame; P2 venue definition is ambiguous.")
        bettime.to_parquet(output_dir / "bettime_frame_allbooks.parquet", index=False)
        metadata["bettime_frame"] = {
            "rows": int(len(bettime)), "events": int(bettime["event_id"].nunique()),
            "goalie_nights": int(bettime[["game_id", "goalie_id"]].drop_duplicates().shape[0]), "books": books,
        }

        # Reload the market-state JSON independently for parity, then produce the registered
        # val-fitted headline and train-fitted-dispersion sensitivity predictions.
        log("STEP 4: frozen market-state inference; no training and no threshold change.")
        market_primary = load_json_model(MODEL_PATHS["control_plus_market_state"])
        market_parity = load_json_model(MODEL_PATHS["control_plus_market_state"])
        rate_primary = load_json_model(MODEL_PATHS["shared_save_rate"])
        rate_parity = load_json_model(MODEL_PATHS["shared_save_rate"])
        dist = state["dist"]
        val_pred = compute_distribution_predictions(
            state["frame"], state["test_idx"], market_primary, rate_primary, recipe["market_alpha"],
            recipe["market_state_feature_cols"], recipe["control_feature_cols"], dist, log, "P2 val-fitted frozen inference",
        )
        parity_pred = compute_distribution_predictions(
            state["frame"], state["test_idx"], market_parity, rate_parity, recipe["market_alpha"],
            recipe["market_state_feature_cols"], recipe["control_feature_cols"], dist, log, "P2 JSON reload parity",
        )
        parity_mu = assert_parity(val_pred["mu"], parity_pred["mu"], "shots means")
        parity_q = assert_parity(val_pred["q"], parity_pred["q"], "save rates")
        p_over_val, p_under_val, p_push_val, matched_val, coverage_val = join_and_price(
            bettime, val_pred, dist, log, "P2 val-fitted bettime"
        )
        alpha_train, alpha_train_method, alpha_train_diag = fit_dispersion(
            market_primary, state["frame"], state["train_idx"], recipe["market_state_feature_cols"], log,
            "P2 train-fitted dispersion sensitivity",
        )
        train_pred = compute_distribution_predictions(
            state["frame"], state["test_idx"], market_primary, rate_primary, alpha_train,
            recipe["market_state_feature_cols"], recipe["control_feature_cols"], dist, log, "P2 train-fitted sensitivity",
        )
        p_over_train, p_under_train, p_push_train, matched_train, coverage_train = join_and_price(
            bettime, train_pred, dist, log, "P2 train-fitted bettime"
        )
        if not np.array_equal(matched_val, matched_train):
            raise AssertionError("Val/train dispersion passes have different prediction joins; stopping before P2.")
        metadata["reload_parity"] = {"max_abs_mu_diff": parity_mu, "max_abs_q_diff": parity_q, "assertion": "both < 1e-9"}
        metadata["dispersion"] = {
            "headline_val_fitted_alpha": recipe["market_alpha"], "train_fitted_alpha": alpha_train,
            "train_fitted_method": alpha_train_method, "train_fitted_diagnostics": alpha_train_diag,
        }

        prediction_frame = bettime[["event_id", "game_id", "goalie_id", "book_key", "betting_line", "saves"]].copy()
        prediction_frame["p_over_val_fitted"] = p_over_val
        prediction_frame["p_under_val_fitted"] = p_under_val
        prediction_frame["p_push_val_fitted"] = p_push_val
        prediction_frame["p_over_train_fitted"] = p_over_train
        prediction_frame["p_under_train_fitted"] = p_under_train
        prediction_frame["p_push_train_fitted"] = p_push_train
        prediction_frame["matched"] = matched_val
        prediction_frame.to_parquet(output_dir / "bettime_predictions.parquet", index=False)

        log("STEP 5: one registered P2 touch and all pre-registered secondaries.")
        universe_val_all = build_p2_universe(bettime, p_under_val, matched_val, EV_THRESHOLD, log, "P2 headline all books")
        universe_val_bo = universe_val_all[universe_val_all["book_key"] == "betonlineag"].reset_index(drop=True)
        universe_train_all = build_p2_universe(bettime, p_under_train, matched_train, EV_THRESHOLD, log, "P2 train sensitivity all books")
        universe_train_bo = universe_train_all[universe_train_all["book_key"] == "betonlineag"].reset_index(drop=True)
        p2_primary = p2_paired_bootstrap(universe_val_bo["cluster_id"], universe_val_bo["profit_if_under"], universe_val_bo["is_model_arm"], N_BOOTSTRAP, BOOTSTRAP_SEED)
        p2_allbooks = p2_paired_bootstrap(universe_val_all["cluster_id"], universe_val_all["profit_if_under"], universe_val_all["is_model_arm"], N_BOOTSTRAP, BOOTSTRAP_SEED)
        p2_train = p2_paired_bootstrap(universe_train_bo["cluster_id"], universe_train_bo["profit_if_under"], universe_train_bo["is_model_arm"], N_BOOTSTRAP, BOOTSTRAP_SEED)
        universe_val_bo.to_parquet(output_dir / "p2_primary_betonlineag_universe.parquet", index=False)
        universe_val_all.to_parquet(output_dir / "p2_allbooks_universe.parquet", index=False)

        if p2_primary["n_model_arm_bets"] < P2_MIN_MODEL_ARM_BETS:
            p2_verdict = "INSUFFICIENT_SAMPLE"
        elif p2_primary["pct_empty_model_arm_resamples"] > P2_MAX_EMPTY_RESAMPLE_PCT:
            p2_verdict = "UNSTABLE"
        elif p2_primary["lower"] is not None and p2_primary["lower"] > 0:
            p2_verdict = "PASS"
        else:
            p2_verdict = "FAIL"
        fragile = sign_flip(p2_primary["mean"], p2_train["mean"])
        exact_verdict = "DISPERSION-FRAGILE" if fragile else p2_verdict
        log(f"P2 PRIMARY BetOnline: {p2_primary}; verdict={p2_verdict}; dispersion_fragile={fragile}.")
        log(f"P2 SECONDARY all-books: {p2_allbooks}")
        log(f"P2 train-fitted sensitivity: {p2_train}")

        # Registered OVER/UNDER ROI splits at the frozen threshold, for both passes and cuts.
        bo_bettime_mask = (bettime["book_key"] == "betonlineag").values
        bo_closing_mask = state["closing"]["book_key"].isin({"betonlineag", "betonline"}).values
        p_over_close = state["predictions"]["control_plus_market_state"]["p_over_closing"]
        p_under_close = state["predictions"]["control_plus_market_state"]["p_under_closing"]
        matched_close = state["predictions"]["control_plus_market_state"]["matched_closing"]
        secondaries = {
            "selection_over_blind_allbooks": p2_allbooks,
            "over_under_roi_bettime_allbooks": side_breakdown_for_subset(bettime, p_over_val, p_under_val, matched_val, np.ones(len(bettime), dtype=bool), EV_THRESHOLD, log, "bettime all books"),
            "over_under_roi_bettime_betonlineag": side_breakdown_for_subset(bettime, p_over_val, p_under_val, matched_val, bo_bettime_mask, EV_THRESHOLD, log, "bettime betonlineag"),
            "over_under_roi_closing_allbooks": side_breakdown_for_subset(state["closing"], p_over_close, p_under_close, matched_close, np.ones(len(state["closing"]), dtype=bool), EV_THRESHOLD, log, "closing all books"),
            "over_under_roi_closing_betonline": side_breakdown_for_subset(state["closing"], p_over_close, p_under_close, matched_close, bo_closing_mask, EV_THRESHOLD, log, "closing BetOnline"),
            "shots_model_frozen_artifact": frozen_meta["origin_b"]["variants"]["control_plus_market_state"]["workload_shots_against_test"],
            "join_coverage_by_book": {str(book): float(matched_val[bettime["book_key"].values == book].mean() * 100) for book in books},
            "join_coverage_bettime_allbooks": coverage_val,
            "join_coverage_bettime_train_sensitivity": coverage_train,
        }
        closing_devig = add_devig_cols(state["closing"], log, "2024-25 closing")
        bettime_devig = add_devig_cols(bettime, log, "2024-25 bettime")
        consensus = build_closing_consensus(closing_devig, log)
        drift = compute_drift_baseline(bettime_devig, consensus, log)
        flagged = flag_bets_full_policy(bettime, p_over_val, p_under_val, matched_val, EV_THRESHOLD, log, "2024-25 model flags")
        flagged = attach_clv(add_devig_cols(flagged, log, "flagged bettime"), consensus, drift, log)
        flagged.to_parquet(output_dir / "flagged_bets_with_clv.parquet", index=False)
        def clv_stat(frame: pd.DataFrame) -> dict:
            ids = frame["game_id"].astype(str) + "_" + frame["goalie_id"].astype(str)
            return clv.cluster_bootstrap_mean_ci(frame["clv_prob_net_of_drift"].values, ids.values, N_BOOTSTRAP, BOOTSTRAP_SEED, 95.0)
        secondaries["clv_net_of_drift"] = {
            "unconditional_drift_baseline": drift,
            "allbooks": clv_stat(flagged), "betonlineag": clv_stat(flagged[flagged["book_key"] == "betonlineag"]),
            "n_flagged_allbooks": int(len(flagged)), "n_flagged_betonlineag": int((flagged["book_key"] == "betonlineag").sum()),
        }

        metadata["p2_touch_completed"] = True
        metadata["P2_primary_betonlineag"] = p2_primary
        metadata["P2_train_fitted_dispersion_sensitivity"] = p2_train
        metadata["dispersion_fragile"] = fragile
        metadata["secondaries"] = secondaries
        metadata["exact_verdict"] = exact_verdict
        metadata["consequence"] = {
            "PASS": "Bettime UNDER-selection is promoted only to 2026-27 shadow-candidacy consideration; viewed 2024-25 is not proof of edge.",
            "FAIL": "The Origin B UNDER-selection effect is not demonstrated at the executable venue in either available season and drops from candidacy.",
            "UNSTABLE": "Report as a wiring/sample-structure finding, not a verdict on the mechanism.",
            "INSUFFICIENT_SAMPLE": "Report all-books secondary and closing context; neither promote nor close the mechanism.",
            "DISPERSION-FRAGILE": "P2 sign flipped under train-fitted dispersion; no pass.",
        }[exact_verdict]
        metadata["wall_clock_seconds"] = time.time() - started
        write_metadata(output_dir, metadata)
        log(f"EXACT VERDICT: {exact_verdict}")
        log(f"Artifacts and metadata saved under {output_dir}")
        flush_log()
        return 0
    except Exception as exc:
        metadata["stopped_at"] = metadata.get("stopped_at", "exception")
        metadata["exception"] = f"{type(exc).__name__}: {exc}"
        metadata["exact_verdict"] = "REGISTRATION_AMBIGUITY" if "REGISTRATION_AMBIGUITY" in str(exc) else "RUN_ABORTED"
        metadata["wall_clock_seconds"] = time.time() - started
        write_metadata(output_dir, metadata)
        log(f"STOPPED: {metadata['exception']}")
        flush_log()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
