"""Experiment 9: fixed-offset funnel plus fixed-weight exposure mixture.

Binding registration: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 12.
This is an offline-only run. It reads existing local parquet inputs and writes
only a new models/trained/experiment_fixed_offset_mixture_<timestamp>/ tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import clv_audit_pace_policy as clv  # noqa: E402
import experiment_exposure_mixture as emix  # noqa: E402
import experiment_pace_distributional as epd  # noqa: E402
import experiment_rolling_origin as ero  # noqa: E402
import experiment_season_funnel as esf  # noqa: E402
from experiments import distributional_saves as ds  # noqa: E402
from experiments import harness as hn  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "models" / "trained"
CLOSING_A_PATH = REPO_ROOT / "data" / "processed" / "multibook_frame_2023_24.parquet"
BETTIME_A_PATH = REPO_ROOT / "data" / "processed" / "multibook_frame_2023_24_bettime.parquet"
FROZEN_CONTROL_METADATA = (
    REPO_ROOT / "models" / "trained" / "experiment_season_funnel_20260713_124419" / "metadata.json"
)

SEASON_2022_23 = 20222023
SEASON_2023_24 = 20232024
SEASON_2024_25 = 20242025
VAL_WINDOW_DAYS = 49
ORIGIN_CAP = 90
FIXED_EV_THRESHOLD = 0.05
N_BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
PIT_SEED = 123

RATE_TARGET_TOI_FLOOR_MIN = 10.0
RATE_TARGET_WINSOR_PCT = 99.5
OFFSET_FALLBACK_RATE60 = 30.0
EARLY_TOI_THRESHOLD_MIN = 50.0
TOI_BIN_LAPLACE = 1.0
LOWER_TAIL_KS = [5, 10, 15, 20, 25, 30]

WIRING_EXPECTED = {
    "A": {"shots_bias": 0.4419625, "closing_brier": 0.2552057011},
    "B": {"shots_bias": 0.0308310, "closing_brier": 0.2512895068},
}
WIRING_BIAS_TOL = 1e-4
WIRING_BRIER_TOL = 1e-6

OFFSET_FORMULA = (
    "A=opp_off_all_corsi_ema5*team_def_all_corsi_against_ema5/prior_league_corsi; "
    "F=A*opp_unblocked_frac*team_unblocked_frac/prior_league_unblocked_frac; "
    "c=prior_league_sog/prior_league_fenwick; "
    "r_opp=(opp_shots_rolling_10/opp_off_all_fenwick_ema5)/c; "
    "r_team=(team_shots_against_rolling_10/team_def_all_fenwick_against_ema5)/c; "
    "lambda0_60=F*c*sqrt(clip(r_opp,0.5,2.0)*clip(r_team,0.5,2.0)); "
    "non-finite lambda0_60 uses fixed 30.0 SOG/60 fallback"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit new artifact directory. Defaults to the registered timestamped name.",
    )
    return parser.parse_args()


def make_output_dir(explicit: Path | None) -> Path:
    if explicit is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_ROOT / f"experiment_fixed_offset_mixture_{timestamp}"
    else:
        path = explicit if explicit.is_absolute() else REPO_ROOT / explicit
    expected_prefix = "experiment_fixed_offset_mixture_"
    if path.parent.resolve() != OUTPUT_ROOT.resolve() or not path.name.startswith(expected_prefix):
        raise ValueError(f"Output must be a new {OUTPUT_ROOT / (expected_prefix + '<timestamp>')} directory.")
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_logger(path: Path):
    lines: list[str] = []

    def log(message: str = "") -> None:
        print(message, flush=True)
        lines.append(str(message))

    def flush() -> None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return log, flush


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [native(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def add_fixed_offset(frame_df: pd.DataFrame, log) -> pd.DataFrame:
    """Build section 12.2's strictly prior-date deterministic SOG/60 rate."""
    needed = [
        "opp_off_all_corsi_ema5",
        "team_def_all_corsi_against_ema5",
        "opp_off_all_fenwick_ema5",
        "team_def_all_fenwick_against_ema5",
        "opp_shots_rolling_10",
        "team_shots_against_rolling_10",
        "opp_shots",
    ]
    missing = [c for c in needed if c not in frame_df.columns]
    if missing:
        raise AssertionError(f"Fixed-offset inputs missing: {missing}")

    df = frame_df.copy()
    prior = esf.attach_prior_league_stats(df, "opp_off_all_corsi_ema5")
    df["_offset_prior_league_corsi"] = prior["prior_mean"].values
    df["_offset_attempts"] = safe_div(
        df["opp_off_all_corsi_ema5"] * df["team_def_all_corsi_against_ema5"],
        df["_offset_prior_league_corsi"],
    )
    df["_offset_opp_unblocked_frac"] = safe_div(
        df["opp_off_all_fenwick_ema5"], df["opp_off_all_corsi_ema5"]
    )
    df["_offset_team_unblocked_frac"] = safe_div(
        df["team_def_all_fenwick_against_ema5"], df["team_def_all_corsi_against_ema5"]
    )
    prior = esf.attach_prior_league_stats(df, "_offset_opp_unblocked_frac")
    df["_offset_prior_league_unblocked_frac"] = prior["prior_mean"].values
    df["_offset_fenwick"] = safe_div(
        df["_offset_attempts"]
        * df["_offset_opp_unblocked_frac"]
        * df["_offset_team_unblocked_frac"],
        df["_offset_prior_league_unblocked_frac"],
    )

    prior_sog = esf.attach_prior_league_stats(df, "opp_shots")
    prior_fenwick = esf.attach_prior_league_stats(df, "opp_off_all_fenwick_ema5")
    df["_offset_prior_league_sog"] = prior_sog["prior_mean"].values
    df["_offset_prior_league_fenwick"] = prior_fenwick["prior_mean"].values
    df["_offset_conversion"] = safe_div(
        df["_offset_prior_league_sog"], df["_offset_prior_league_fenwick"]
    )
    df["_offset_r_opp"] = safe_div(
        safe_div(df["opp_shots_rolling_10"], df["opp_off_all_fenwick_ema5"]),
        df["_offset_conversion"],
    )
    df["_offset_r_team"] = safe_div(
        safe_div(df["team_shots_against_rolling_10"], df["team_def_all_fenwick_against_ema5"]),
        df["_offset_conversion"],
    )
    raw = (
        df["_offset_fenwick"]
        * df["_offset_conversion"]
        * np.sqrt(df["_offset_r_opp"].clip(0.5, 2.0) * df["_offset_r_team"].clip(0.5, 2.0))
    )
    raw = raw.replace([np.inf, -np.inf], np.nan)
    df["_offset_lambda0_60_raw"] = raw
    df["_offset_fallback"] = raw.isna()
    df["_offset_lambda0_60"] = raw.fillna(OFFSET_FALLBACK_RATE60)
    df["_offset_z"] = np.log(np.maximum(df["_offset_lambda0_60"].values.astype(float), 1e-3))

    nonpositive = int((df["_offset_lambda0_60_raw"].dropna() <= 0).sum())
    log(
        f"Fixed offset built for {len(df)} rows: raw finite={int(raw.notna().sum())}, "
        f"fallback={int(df['_offset_fallback'].sum())}, finite nonpositive={nonpositive}."
    )
    log(f"Offset formula: {OFFSET_FORMULA}")
    return df


def fold_offset_stats(df: pd.DataFrame, idx: np.ndarray) -> dict:
    raw = df["_offset_lambda0_60_raw"].values[idx].astype(float)
    fallback = df["_offset_fallback"].values[idx].astype(bool)
    finite = np.isfinite(raw)
    return {
        "rows": int(len(idx)),
        "raw_finite_count": int(finite.sum()),
        "raw_coverage_pct": float(finite.mean() * 100.0),
        "fallback_count": int(fallback.sum()),
        "fallback_pct": float(fallback.mean() * 100.0),
        "raw_mean_finite": float(np.mean(raw[finite])) if finite.any() else None,
        "used_mean": float(df["_offset_lambda0_60"].values[idx].astype(float).mean()),
    }


class MarginAudit:
    def __init__(self) -> None:
        self.fit_calls: list[dict] = []
        self.predict_calls: list[dict] = []

    @staticmethod
    def validate(margin: np.ndarray, n: int, context: str) -> np.ndarray:
        if margin is None:
            raise AssertionError(f"[{context}] candidate base_margin is mandatory.")
        arr = np.asarray(margin, dtype=np.float32)
        if len(arr) != n or not np.isfinite(arr).all():
            raise AssertionError(f"[{context}] invalid base_margin shape or values.")
        return arr

    def record_fit(self, train_margin, n_train, val_margin, n_val, context: str) -> tuple[np.ndarray, np.ndarray]:
        z_train = self.validate(train_margin, n_train, context + " train")
        z_val = self.validate(val_margin, n_val, context + " val")
        self.fit_calls.append({"context": context, "n_train": n_train, "n_val": n_val})
        return z_train, z_val

    def predict(self, model, X: pd.DataFrame, margin: np.ndarray, context: str) -> np.ndarray:
        z = self.validate(margin, len(X), context)
        self.predict_calls.append({"context": context, "rows": len(X)})
        return np.asarray(model.predict(X, base_margin=z), dtype=float)


def train_offset_rate60_model(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_cols: list[str],
    audit: MarginAudit,
    log,
    label: str,
) -> tuple[xgb.XGBRegressor, dict, list[dict], float]:
    ds.assert_feature_matrix_clean(df, feature_cols, f"{label} residual X", allow_nan=True)
    if len(feature_cols) != 104:
        raise AssertionError(f"{label}: expected exact 104-column residual X, got {len(feature_cols)}.")
    forbidden_named = [
        c for c in feature_cols
        if c.startswith(("fnl_", "pace_", "ctx_", "_offset_")) or c in {"toi", "_toi_minutes"}
    ]
    if forbidden_named:
        raise AssertionError(f"{label}: forbidden columns entered residual X: {forbidden_named}")

    toi = df["_toi_minutes"].values.astype(float)
    rate60_all = df["shots_against"].values.astype(float) / (np.maximum(toi, RATE_TARGET_TOI_FLOOR_MIN) / 60.0)
    y_train_raw = rate60_all[train_idx]
    winsor_cap = float(np.percentile(y_train_raw, RATE_TARGET_WINSOR_PCT))
    y_train = np.minimum(y_train_raw, winsor_cap)
    y_val = rate60_all[val_idx]
    X_train = df[feature_cols].iloc[train_idx].astype(np.float32)
    X_val = df[feature_cols].iloc[val_idx].astype(np.float32)
    z_train = df["_offset_z"].values[train_idx]
    z_val = df["_offset_z"].values[val_idx]

    log(
        f"\n--- {label}: offset Poisson rate60 residual model ---\n"
        f"Residual X={len(feature_cols)} columns; target TOI floor={RATE_TARGET_TOI_FLOOR_MIN}; "
        f"train p{RATE_TARGET_WINSOR_PCT} cap={winsor_cap:.6f}; "
        f"clipped={int((y_train_raw > winsor_cap).sum())}/{len(y_train_raw)}."
    )
    evaluations: list[dict] = []
    models: dict[str, xgb.XGBRegressor] = {}
    for name, cfg in ds.SHOTS_CONFIGS:
        params = dict(objective="count:poisson", random_state=42, n_jobs=-1, verbosity=0, **cfg)
        model = xgb.XGBRegressor(**params)
        zt, zv = audit.record_fit(z_train, len(X_train), z_val, len(X_val), f"{label} config={name}")
        model.fit(
            X_train,
            y_train,
            base_margin=zt,
            eval_set=[(X_val, y_val)],
            base_margin_eval_set=[zv],
            verbose=False,
        )
        pred_val = np.clip(audit.predict(model, X_val, z_val, f"{label} config={name} val selection"), 1e-3, None)
        val_mae = ds.mae(y_val, pred_val)
        evaluations.append({"config": name, "hyperparams": cfg, "val_rate60_mae": val_mae})
        models[name] = model
        log(f"  {name:<20} val_rate60_mae={val_mae:.6f}")

    winner = min(evaluations, key=lambda item: item["val_rate60_mae"])
    log(f"{label} winner={winner['config']} val_rate60_mae={winner['val_rate60_mae']:.6f}")
    return models[winner["config"]], winner, evaluations, winsor_cap


def build_toi_distribution(train_toi: np.ndarray, early: bool, log, label: str) -> list[tuple[float, float]]:
    if early:
        selected = train_toi[train_toi < EARLY_TOI_THRESHOLD_MIN]
        lo, hi, width = 0.0, 50.0, 5.0
    else:
        selected = train_toi[train_toi >= EARLY_TOI_THRESHOLD_MIN]
        lo, hi, width = 50.0, 66.0, 1.0
    if np.any(selected < lo) or np.any(selected >= hi):
        raise AssertionError(f"{label}: observed TOI falls outside registered [{lo},{hi}) bins.")
    return emix.build_toi_bin_distribution(
        selected, width, lo, hi, log, label, laplace=TOI_BIN_LAPLACE
    )


def mixture_mean_minutes(pi: float, early_bins, normal_bins) -> float:
    early_mean = sum(float(w) * float(t) for w, t in early_bins)
    normal_mean = sum(float(w) * float(t) for w, t in normal_bins)
    return float(pi * early_mean + (1.0 - pi) * normal_mean)


def build_candidate_pmfs(
    rate60: np.ndarray,
    q: np.ndarray,
    alpha: float,
    pi: float,
    early_bins,
    normal_bins,
    dist: ds.SavesDistribution,
    label: str,
) -> tuple[dict, dict]:
    n = len(rate60)
    tbar = mixture_mean_minutes(pi, early_bins, normal_bins)
    mu_mean_matched = np.asarray(rate60, dtype=float) * tbar / 60.0
    single_pmf = np.zeros((n, dist.cap + 1), dtype=float)
    mixture_pmf = np.zeros((n, dist.cap + 1), dtype=float)
    for i in range(n):
        single_pmf[i], _ = dist.saves_pmf(mu_mean_matched[i], alpha, q[i])
        p = np.zeros(dist.cap + 1, dtype=float)
        for weight, minutes in early_bins:
            body, _ = dist.saves_pmf(rate60[i] * minutes / 60.0, alpha, q[i])
            p += pi * weight * body
        for weight, minutes in normal_bins:
            body, _ = dist.saves_pmf(rate60[i] * minutes / 60.0, alpha, q[i])
            p += (1.0 - pi) * weight * body
        mixture_pmf[i] = p

    for arm, pmf in (("single", single_pmf), ("mixture", mixture_pmf)):
        sums = pmf.sum(axis=1)
        if np.any(sums < 0.999):
            raise AssertionError(f"{label} {arm}: PMF cap truncation too large; min sum={sums.min()}.")
    support = np.arange(dist.cap + 1, dtype=float)
    expected_saves_gap = np.max(np.abs(single_pmf @ support - mixture_pmf @ support))
    if expected_saves_gap > 0.0025:
        raise AssertionError(
            f"{label}: PMF expected-saves mean matching failed; max gap={expected_saves_gap}."
        )
    common = {"mu": mu_mean_matched, "q": q, "rate60": rate60, "tbar": tbar}
    return ({**common, "pmf": single_pmf}, {**common, "pmf": mixture_pmf})


def attach_lookup(preds: dict, df: pd.DataFrame, idx: np.ndarray) -> dict:
    keys = list(zip(df["game_id"].values[idx].astype(int), df["goalie_id"].values[idx].astype(int)))
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate goalie-night keys in test predictions.")
    preds = dict(preds)
    preds.update({"keys": keys, "lookup": {key: i for i, key in enumerate(keys)}, "idx": idx})
    return preds


def distribution_metrics(pmf: np.ndarray, saves: np.ndarray, log, label: str) -> dict:
    saves_c = np.clip(np.asarray(saves, dtype=int), 0, pmf.shape[1] - 1)
    rows = np.arange(len(saves_c))
    negative_log_score = float(np.mean(-np.log(np.clip(pmf[rows, saves_c], 1e-12, None))))
    coverage_full = emix.coverage_pit_metrics(pmf, saves_c, pmf.shape[1] - 1, seed=PIT_SEED)
    coverage = emix.strip_private(coverage_full)
    lower_rows = emix.lower_tail_calibration(
        pmf, saves_c, pmf.shape[1] - 1, LOWER_TAIL_KS, log, label
    )
    lower_l = float(sum(abs(row["diff"]) for row in lower_rows))
    return {
        "full_distribution_negative_log_score": negative_log_score,
        "coverage": coverage,
        "lower_tail": lower_rows,
        "lower_tail_L": lower_l,
    }


def paired_brier_models(
    df_bet: pd.DataFrame,
    p_a: np.ndarray,
    matched_a: np.ndarray,
    p_b: np.ndarray,
    matched_b: np.ndarray,
    log,
    label: str,
) -> dict:
    if not np.array_equal(matched_a, matched_b):
        raise AssertionError(f"{label}: joined quote rows differ between arms.")
    mask = matched_a & np.isfinite(p_a) & np.isfinite(p_b)
    y = (df_bet["saves"].values[mask] > df_bet["betting_line"].values[mask]).astype(float)
    sq_a = (p_a[mask] - y) ** 2
    sq_b = (p_b[mask] - y) ** 2
    delta = sq_a - sq_b
    clusters = np.asarray(
        [
            f"{int(g)}_{int(k)}"
            for g, k in zip(df_bet["game_id"].values[mask], df_bet["goalie_id"].values[mask])
        ],
        dtype=object,
    )
    stat = clv.cluster_bootstrap_mean_ci(
        delta, clusters, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, ci_pct=95.0
    )
    result = {
        "model_a_brier": float(sq_a.mean()),
        "model_b_brier": float(sq_b.mean()),
        "delta_mean": float(stat["mean"]),
        "delta_ci95_lower": float(stat["lower"]),
        "delta_ci95_upper": float(stat["upper"]),
        "n_rows": int(stat["n_bets"]),
        "n_clusters": int(stat["n_clusters"]),
    }
    log(
        f"[{label}] paired Brier delta={result['delta_mean']:+.8f} "
        f"CI95=[{result['delta_ci95_lower']:+.8f},{result['delta_ci95_upper']:+.8f}] "
        f"n={result['n_rows']} clusters={result['n_clusters']}"
    )
    return result


def price_arm(
    df_bet: pd.DataFrame,
    preds: dict,
    dist: ds.SavesDistribution,
    origin: str,
    pass_name: str,
    arm_name: str,
    log,
) -> tuple[dict, pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    label = f"origin_{origin} {arm_name} {pass_name}"
    p_over, p_under, p_push, matched, coverage = ds.join_and_price(df_bet, preds, dist, log, label)
    auc, brier = hn.fold_wide_auc_brier(
        p_over,
        matched,
        df_bet["saves"].values,
        df_bet["betting_line"].values,
        df_bet["game_id"].values,
        df_bet["goalie_id"].values,
        log,
        label,
    )
    market_delta, market_p_over, market_p_under = ero.paired_brier_delta(df_bet, p_over, matched, log, label)
    bets = hn.grade_bets(
        p_over,
        p_under,
        df_bet["saves"].values.astype(float),
        df_bet["betting_line"].values.astype(float),
        df_bet["odds_over_american"].astype(float).values,
        df_bet["odds_under_american"].astype(float).values,
        df_bet["game_id"].values,
        df_bet["goalie_id"].values,
        FIXED_EV_THRESHOLD,
        matched,
        log,
        label,
    )
    bundle = hn.betting_metrics_bundle(
        bets, df_bet["game_id"].values, df_bet["goalie_id"].values, len(df_bet)
    )
    side = emix.side_calibration(
        p_over,
        matched,
        df_bet["saves"].values.astype(float),
        df_bet["betting_line"].values.astype(float),
        log,
        label,
    )
    rows = ero.build_row_predictions(
        df_bet,
        p_over,
        p_under,
        matched,
        market_p_over,
        market_p_under,
        FIXED_EV_THRESHOLD,
        origin,
        pass_name,
    )
    rows["arm"] = arm_name
    rows["model_p_push"] = p_push[matched]
    result = {
        "join_coverage_pct": float(coverage),
        "fold_wide_auc": auc,
        "fold_wide_brier": float(brier),
        "paired_brier_vs_devigged_market": market_delta,
        "side_calibration": side,
        "fixed_0_05_policy": bundle,
    }
    return result, rows, (p_over, p_under, p_push, matched)


def fit_baseline_origin(
    origin: str,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: list[str],
    price_frames: dict[str, pd.DataFrame],
    output_dir: Path,
    log,
) -> dict:
    log("\n" + "=" * 80)
    log(f"WIRING PREP ORIGIN {origin}: exact no-pace control only")
    log("=" * 80)
    shots_model, shots_winner, shots_evals = ds.train_shots_model(
        df, train_idx, val_idx, feature_cols, log, f"origin_{origin} no_pace_control"
    )
    save_model, save_winner, save_evals = ds.train_save_rate_model(
        df, train_idx, val_idx, feature_cols, log, f"origin_{origin} shared_save_rate"
    )
    X_val = df[feature_cols].iloc[val_idx].astype(np.float32)
    mu_val = np.clip(shots_model.predict(X_val), 1e-3, None)
    alpha_val, alpha_val_method, alpha_val_diag = emix.fit_dispersion_oos(
        mu_val,
        df["shots_against"].values[val_idx].astype(float),
        log,
        f"origin_{origin} no_pace_control validation alpha",
    )
    alpha_train, alpha_train_method, alpha_train_diag = ds.fit_dispersion(
        shots_model, df, train_idx, feature_cols, log, f"origin_{origin} no_pace_control train alpha"
    )
    shots_path = output_dir / f"origin_{origin.lower()}_no_pace_control_shots_model.json"
    save_path = output_dir / f"origin_{origin.lower()}_shared_save_rate_model.json"
    shots_model.get_booster().save_model(shots_path)
    save_model.get_booster().save_model(save_path)

    dist = ds.SavesDistribution(ORIGIN_CAP)
    test_preds = ds.compute_distribution_predictions(
        df,
        test_idx,
        shots_model,
        save_model,
        alpha_val,
        feature_cols,
        feature_cols,
        dist,
        log,
        f"origin_{origin} no_pace_control TEST wiring",
    )
    shots_actual = df["shots_against"].values[test_idx].astype(float)
    bias = float(np.mean(test_preds["mu"] - shots_actual))
    closing_result, closing_rows, closing_probs = price_arm(
        price_frames["closing"], test_preds, dist, origin, "closing", "no_pace_control", log
    )
    expected = WIRING_EXPECTED[origin]
    observed_brier = closing_result["fold_wide_brier"]
    bias_ok = abs(bias - expected["shots_bias"]) <= WIRING_BIAS_TOL
    brier_ok = abs(observed_brier - expected["closing_brier"]) <= WIRING_BRIER_TOL
    log(
        f"WIRING ORIGIN {origin}: bias={bias:+.10f} expected={expected['shots_bias']:+.10f} "
        f"tol={WIRING_BIAS_TOL} -> {'PASS' if bias_ok else 'FAIL'}"
    )
    log(
        f"WIRING ORIGIN {origin}: closing Brier={observed_brier:.10f} "
        f"expected={expected['closing_brier']:.10f} tol={WIRING_BRIER_TOL} "
        f"-> {'PASS' if brier_ok else 'FAIL'}"
    )
    return {
        "passed": bool(bias_ok and brier_ok),
        "gate": {
            "expected": expected,
            "observed": {"shots_bias": bias, "closing_brier": observed_brier},
            "tolerances": {"shots_bias": WIRING_BIAS_TOL, "closing_brier": WIRING_BRIER_TOL},
            "passed": bool(bias_ok and brier_ok),
        },
        "shots_model": shots_model,
        "save_model": save_model,
        "test_preds": test_preds,
        "dist": dist,
        "closing_result": closing_result,
        "closing_rows": closing_rows,
        "closing_probs": closing_probs,
        "model_metadata": {
            "shots": {"winner": shots_winner, "evaluations": shots_evals, "path": shots_path},
            "save_rate": {"winner": save_winner, "evaluations": save_evals, "path": save_path},
            "validation_alpha": {
                "alpha": alpha_val,
                "method": alpha_val_method,
                "diagnostics": alpha_val_diag,
            },
            "train_alpha": {
                "alpha": alpha_train,
                "method": alpha_train_method,
                "diagnostics": alpha_train_diag,
            },
        },
    }


def persist_goalie_night_predictions(
    path: Path,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    baseline: dict,
    arms: dict[str, dict],
    pi: float,
    tbar: float,
    alpha_primary: float,
    alpha_train: float,
) -> None:
    out = pd.DataFrame(
        {
            "game_id": df["game_id"].values[test_idx].astype(int),
            "goalie_id": df["goalie_id"].values[test_idx].astype(int),
            "game_date": df["game_date"].values[test_idx],
            "actual_saves": df["saves"].values[test_idx].astype(int),
            "actual_shots_against": df["shots_against"].values[test_idx].astype(float),
            "actual_toi_min_postgame_only": df["_toi_minutes"].values[test_idx].astype(float),
            "actual_toi_lt50_postgame_only": (df["_toi_minutes"].values[test_idx] < 50.0),
            "offset_lambda0_60_raw": df["_offset_lambda0_60_raw"].values[test_idx].astype(float),
            "offset_fallback_used": df["_offset_fallback"].values[test_idx].astype(bool),
            "offset_lambda0_60_used": df["_offset_lambda0_60"].values[test_idx].astype(float),
            "offset_base_margin_z": df["_offset_z"].values[test_idx].astype(float),
            "lambda_hat_60": arms["fixed_single_primary"]["rate60"],
            "mean_matched_shots_mu": arms["fixed_single_primary"]["mu"],
            "shared_save_rate_q": arms["fixed_single_primary"]["q"],
            "train_fold_pi": pi,
            "train_fold_tbar": tbar,
            "validation_alpha_primary": alpha_primary,
            "train_alpha_sensitivity": alpha_train,
            "no_pace_control_mu": baseline["mu"],
            "no_pace_control_q": baseline["q"],
        }
    )
    all_pmfs = {"no_pace_control": baseline["pmf"], **{name: pred["pmf"] for name, pred in arms.items()}}
    for arm_name, pmf in all_pmfs.items():
        for s in range(pmf.shape[1]):
            out[f"pmf_{arm_name}_{s}"] = pmf[:, s]
    out.to_parquet(path, index=False, compression="zstd")


def run_candidate_origin(
    origin: str,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: list[str],
    price_frames: dict[str, pd.DataFrame],
    baseline_cache: dict,
    output_dir: Path,
    log,
) -> dict:
    log("\n" + "=" * 80)
    log(f"ORIGIN {origin}: fixed-offset candidate authorized after BOTH wiring gates")
    log("=" * 80)
    audit = MarginAudit()
    model, winner, evaluations, winsor_cap = train_offset_rate60_model(
        df, train_idx, val_idx, feature_cols, audit, log, f"origin_{origin} fixed_offset"
    )
    model_path = output_dir / f"origin_{origin.lower()}_fixed_offset_rate60_model.json"
    model.get_booster().save_model(model_path)

    X_train = df[feature_cols].iloc[train_idx].astype(np.float32)
    X_val = df[feature_cols].iloc[val_idx].astype(np.float32)
    X_test = df[feature_cols].iloc[test_idx].astype(np.float32)
    z_train = df["_offset_z"].values[train_idx]
    z_val = df["_offset_z"].values[val_idx]
    z_test = df["_offset_z"].values[test_idx]
    rate60_train = np.clip(audit.predict(model, X_train, z_train, f"origin_{origin} train alpha"), 1e-3, None)
    rate60_val = np.clip(audit.predict(model, X_val, z_val, f"origin_{origin} validation alpha"), 1e-3, None)

    train_toi = df["_toi_minutes"].values[train_idx].astype(float)
    val_toi = df["_toi_minutes"].values[val_idx].astype(float)
    alpha_train, method_train, diag_train = emix.fit_dispersion_oos(
        rate60_train * train_toi / 60.0,
        df["shots_against"].values[train_idx].astype(float),
        log,
        f"origin_{origin} train-alpha sensitivity",
    )
    alpha_primary, method_primary, diag_primary = emix.fit_dispersion_oos(
        rate60_val * val_toi / 60.0,
        df["shots_against"].values[val_idx].astype(float),
        log,
        f"origin_{origin} PRIMARY validation alpha",
    )

    pi = float(np.mean(train_toi < EARLY_TOI_THRESHOLD_MIN))
    early_bins = build_toi_distribution(train_toi, True, log, f"origin_{origin} early")
    normal_bins = build_toi_distribution(train_toi, False, log, f"origin_{origin} normal")
    tbar = mixture_mean_minutes(pi, early_bins, normal_bins)
    log(f"origin_{origin}: train pi={pi:.10f}; mixture-weighted Tbar={tbar:.10f} minutes.")

    # This is the first candidate TEST prediction call. Both wiring gates are
    # checked in main before this function can be entered.
    rate60_test = np.clip(audit.predict(model, X_test, z_test, f"origin_{origin} TEST"), 1e-3, None)
    q_test = np.clip(
        baseline_cache["save_model"].predict(X_test),
        1e-6,
        1.0 - 1e-6,
    )
    dist = baseline_cache["dist"]
    single_primary, mixture_primary = build_candidate_pmfs(
        rate60_test, q_test, alpha_primary, pi, early_bins, normal_bins, dist, f"origin_{origin} primary"
    )
    single_train, mixture_train = build_candidate_pmfs(
        rate60_test, q_test, alpha_train, pi, early_bins, normal_bins, dist, f"origin_{origin} train-alpha"
    )
    arms = {
        "fixed_single_primary": attach_lookup(single_primary, df, test_idx),
        "fixed_mixture_primary": attach_lookup(mixture_primary, df, test_idx),
        "fixed_single_train_alpha": attach_lookup(single_train, df, test_idx),
        "fixed_mixture_train_alpha": attach_lookup(mixture_train, df, test_idx),
    }
    if not np.array_equal(arms["fixed_single_primary"]["mu"], arms["fixed_mixture_primary"]["mu"]):
        raise AssertionError("Mean matching failed between primary single and mixture arms.")

    saves_test = df["saves"].values[test_idx].astype(int)
    shots_test = df["shots_against"].values[test_idx].astype(float)
    intrinsic: dict[str, dict] = {}
    for arm_name, pred in arms.items():
        intrinsic[arm_name] = distribution_metrics(pred["pmf"], saves_test, log, f"origin_{origin} {arm_name}")
    intrinsic["no_pace_control"] = distribution_metrics(
        baseline_cache["test_preds"]["pmf"], saves_test, log, f"origin_{origin} no_pace_control"
    )

    mu = arms["fixed_mixture_primary"]["mu"]
    shots_bias = float(np.mean(mu - shots_test))
    shots_mae = float(np.mean(np.abs(mu - shots_test)))

    early_mask = df["_toi_minutes"].values[test_idx].astype(float) < EARLY_TOI_THRESHOLD_MIN
    early_diagnostics = {
        "label": "postgame_only_not_used_in_predictions",
        "n": int(early_mask.sum()),
        "rate_pct": float(early_mask.mean() * 100.0),
        "shots_bias": float(np.mean(mu[early_mask] - shots_test[early_mask])),
        "shots_mae": float(np.mean(np.abs(mu[early_mask] - shots_test[early_mask]))),
        "distribution_metrics": {
            arm_name: distribution_metrics(
                pred["pmf"][early_mask], saves_test[early_mask], log, f"origin_{origin} {arm_name} TOI<50 postgame"
            )
            for arm_name, pred in arms.items()
        },
    }

    pricing: dict[str, dict] = {}
    quote_rows: list[pd.DataFrame] = []
    pricing_probs: dict[str, dict] = {}
    priced_arms = {"no_pace_control": baseline_cache["test_preds"], **arms}
    for pass_name, df_bet in price_frames.items():
        pricing[pass_name] = {}
        pricing_probs[pass_name] = {}
        for arm_name, pred in priced_arms.items():
            result, rows, probs = price_arm(df_bet, pred, dist, origin, pass_name, arm_name, log)
            pricing[pass_name][arm_name] = result
            pricing_probs[pass_name][arm_name] = probs
            quote_rows.append(rows)

    candidate_probs = pricing_probs["closing"]["fixed_mixture_primary"]
    baseline_probs = pricing_probs["closing"]["no_pace_control"]
    brier_vs_control = paired_brier_models(
        price_frames["closing"],
        candidate_probs[0],
        candidate_probs[3],
        baseline_probs[0],
        baseline_probs[3],
        log,
        f"origin_{origin} P2 fixed_mixture_primary-minus-no_pace_control",
    )

    d_single = intrinsic["fixed_single_primary"]["coverage"]["summed_coverage_deviation"]
    d_mixture = intrinsic["fixed_mixture_primary"]["coverage"]["summed_coverage_deviation"]
    l_single = intrinsic["fixed_single_primary"]["lower_tail_L"]
    l_mixture = intrinsic["fixed_mixture_primary"]["lower_tail_L"]
    bet_rate = pricing["closing"]["fixed_mixture_primary"]["fixed_0_05_policy"]["summary"]["bet_rate"]
    pass_bars = {
        "P1_shots_level": {
            "value_abs_bias": abs(shots_bias),
            "signed_bias": shots_bias,
            "bar": "abs(bias) < 0.5",
            "passed": bool(abs(shots_bias) < 0.5),
        },
        "P2_probability_accuracy": {
            **brier_vs_control,
            "bar": "candidate-minus-control Brier < 0",
            "passed": bool(brier_vs_control["delta_mean"] < 0.0),
            "statistically_weak": bool(
                brier_vs_control["delta_mean"] < 0.0
                and brier_vs_control["delta_ci95_lower"]
                <= 0.0
                <= brier_vs_control["delta_ci95_upper"]
            ),
        },
        "P3_central_coverage": {
            "D_single": d_single,
            "D_mixture": d_mixture,
            "bar": "D_mixture <= D_single",
            "passed": bool(d_mixture <= d_single),
        },
        "P4_lower_tail": {
            "L_single": l_single,
            "L_mixture": l_mixture,
            "bar": "L_mixture < L_single",
            "passed": bool(l_mixture < l_single),
        },
        "P5_no_extreme_edge_inflation": {
            "closing_bet_rate_pct": bet_rate,
            "bar": "15 <= bet_rate_pct <= 45",
            "passed": bool(15.0 <= bet_rate <= 45.0),
        },
    }
    pass_bars["origin_all_passed"] = bool(all(v["passed"] for k, v in pass_bars.items() if k.startswith("P")))

    quote_path = output_dir / f"origin_{origin.lower()}_quote_predictions.parquet"
    pd.concat(quote_rows, ignore_index=True).to_parquet(quote_path, index=False, compression="zstd")
    goalie_path = output_dir / f"origin_{origin.lower()}_goalie_night_predictions.parquet"
    persist_goalie_night_predictions(
        goalie_path,
        df,
        test_idx,
        baseline_cache["test_preds"],
        arms,
        pi,
        tbar,
        alpha_primary,
        alpha_train,
    )

    # Reload verification also uses the mandatory base margin.
    reloaded = xgb.XGBRegressor()
    reloaded.load_model(model_path)
    reload_pred = np.clip(
        audit.predict(reloaded, X_test, z_test, f"origin_{origin} TEST reload verification"), 1e-3, None
    )
    reload_max_abs_diff = float(np.max(np.abs(reload_pred - rate60_test)))
    if reload_max_abs_diff > 1e-10:
        raise AssertionError(f"origin_{origin}: reloaded offset model differs by {reload_max_abs_diff}.")

    return {
        "origin": origin,
        "test_rows": int(len(test_idx)),
        "shots_bias_test": shots_bias,
        "shots_mae_test": shots_mae,
        "pass_bars": pass_bars,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "offset_coverage": {
            "train": fold_offset_stats(df, train_idx),
            "validation": fold_offset_stats(df, val_idx),
            "test": fold_offset_stats(df, test_idx),
        },
        "rate60_model": {
            "winner": winner,
            "evaluations": evaluations,
            "winsor_cap": winsor_cap,
            "path": model_path,
            "reload_max_abs_diff": reload_max_abs_diff,
        },
        "base_margin_audit": {
            "fit_calls": audit.fit_calls,
            "predict_calls": audit.predict_calls,
            "all_candidate_calls_wired": True,
        },
        "exposure": {
            "pi_train": pi,
            "early_bins": early_bins,
            "normal_bins": normal_bins,
            "Tbar": tbar,
            "early_mean_minutes": sum(w * t for w, t in early_bins),
            "normal_mean_minutes": sum(w * t for w, t in normal_bins),
        },
        "dispersion": {
            "primary_validation_alpha": {
                "alpha": alpha_primary,
                "method": method_primary,
                "diagnostics": diag_primary,
            },
            "sensitivity_train_alpha": {
                "alpha": alpha_train,
                "method": method_train,
                "diagnostics": diag_train,
            },
        },
        "intrinsic_test": intrinsic,
        "pricing": pricing,
        "test_toi_lt50_postgame_only": early_diagnostics,
        "prediction_artifacts": {"goalie_night": goalie_path, "quote_rows": quote_path},
    }


def main() -> int:
    args = parse_args()
    start = time.time()
    output_dir = make_output_dir(args.output_dir)
    log_path = output_dir / "run_log.txt"
    log, flush = make_logger(log_path)
    metadata: dict = {
        "timestamp": datetime.now().isoformat(),
        "registration": "docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 12 (Experiment 9, plan steps 6b/6c)",
        "output_dir": output_dir,
        "network_or_odds_api_calls": 0,
        "deviations_from_registration": [],
        "completed": False,
    }
    try:
        log("=" * 80)
        log("EXPERIMENT 9 -- FIXED-OFFSET FUNNEL PLUS FIXED-WEIGHT EXPOSURE MIXTURE")
        log("Binding registration: docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md section 12")
        log("ZERO NETWORK / ZERO ODDS API CREDITS")
        log("=" * 80)
        log(f"Output directory: {output_dir}")
        required = [
            epd.DATA_PATH_CLEAN,
            epd.DATA_PATH_CONTEXT,
            epd.DATA_PATH_PACE,
            epd.DATA_PATH_PACE_METADATA,
            epd.DATA_PATH_MULTIBOOK,
            CLOSING_A_PATH,
            BETTIME_A_PATH,
            FROZEN_CONTROL_METADATA,
        ]
        for path in required:
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required local input: {path}")

        frame = epd.load_pace_modeling_frame(
            epd.DATA_PATH_CLEAN,
            epd.DATA_PATH_CONTEXT,
            epd.DATA_PATH_PACE,
            epd.DATA_PATH_PACE_METADATA,
            log,
        )
        clean = pd.read_parquet(epd.DATA_PATH_CLEAN)
        clean["game_date"] = pd.to_datetime(clean["game_date"])
        df = esf.add_season_column(frame.df, clean, log)
        df["_toi_minutes"] = df["toi"].map(emix.toi_to_minutes).astype(float)
        original_keys = list(zip(df["game_id"].tolist(), df["goalie_id"].tolist()))
        df = add_fixed_offset(df, log)
        if original_keys != list(zip(df["game_id"].tolist(), df["goalie_id"].tolist())):
            raise AssertionError("Offset construction changed row order.")

        feature_cols = list(frame.base_feature_cols) + list(frame.engineered_cols)
        frozen = json.loads(FROZEN_CONTROL_METADATA.read_text(encoding="utf-8"))
        frozen_a = frozen["results"]["A"]["no_pace_control"]["shots_feature_cols"]
        frozen_b = frozen["results"]["B"]["no_pace_control"]["shots_feature_cols"]
        if len(feature_cols) != 104 or feature_cols != frozen_a or feature_cols != frozen_b:
            raise AssertionError("Exact no-pace 104-column residual X identity gate failed.")
        log("FEATURE GATE PASSED: residual X is bit-for-list identical to frozen 104-column no-pace X.")

        df_bet_a_closing = epd.build_betting_frame(CLOSING_A_PATH, log)
        df_bet_a_bettime = epd.build_betting_frame(BETTIME_A_PATH, log)
        df_bet_all = epd.build_betting_frame(epd.DATA_PATH_MULTIBOOK, log)
        df_bet_b_closing = df_bet_all[df_bet_all["season"] == SEASON_2024_25].reset_index(drop=True)

        origins: dict[str, dict] = {}
        for origin, train_seasons, test_season, price_frames in (
            (
                "A",
                [SEASON_2022_23],
                SEASON_2023_24,
                {"closing": df_bet_a_closing, "bettime": df_bet_a_bettime},
            ),
            (
                "B",
                [SEASON_2022_23, SEASON_2023_24],
                SEASON_2024_25,
                {"closing": df_bet_b_closing},
            ),
        ):
            pool_min, pool_max = ero.season_date_range(clean, train_seasons)
            train_idx, val_idx, boundaries = ero.carve_origin_split(
                df, pool_min, pool_max, VAL_WINDOW_DAYS, log, f"Origin {origin}"
            )
            test_min, test_max = ero.season_date_range(clean, [test_season])
            test_idx = ero.date_range_test_idx(df, test_min, test_max, log, f"Origin {origin}")
            if len(test_idx) != 2624:
                raise AssertionError(f"Origin {origin}: expected 2,624 test starts, got {len(test_idx)}.")
            origins[origin] = {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "boundaries": {**boundaries, "test_season": test_season, "test_rows": len(test_idx)},
                "price_frames": price_frames,
            }

        # Mandatory gate: both no-pace reproductions complete before any
        # candidate model is trained or any candidate test prediction exists.
        baseline_cache: dict[str, dict] = {}
        for origin in ("A", "B"):
            o = origins[origin]
            baseline_cache[origin] = fit_baseline_origin(
                origin,
                df,
                o["train_idx"],
                o["val_idx"],
                o["test_idx"],
                feature_cols,
                o["price_frames"],
                output_dir,
                log,
            )
            flush()
        wiring_passed = all(baseline_cache[o]["passed"] for o in ("A", "B"))
        metadata["wiring_gate"] = {
            "candidate_test_predictions_before_gate": 0,
            "origins": {o: baseline_cache[o]["gate"] for o in ("A", "B")},
            "passed": wiring_passed,
        }
        if not wiring_passed:
            metadata["stopped_at"] = "mandatory_no_pace_wiring_gate"
            raise AssertionError("Mandatory no-pace wiring gate failed. Candidate not run.")
        log("\nMANDATORY NO-PACE WIRING GATE PASSED FOR BOTH ORIGINS.")
        log("Candidate training and TEST prediction are now authorized.")
        flush()

        results = {}
        for origin in ("A", "B"):
            o = origins[origin]
            results[origin] = run_candidate_origin(
                origin,
                df,
                o["train_idx"],
                o["val_idx"],
                o["test_idx"],
                feature_cols,
                o["price_frames"],
                baseline_cache[origin],
                output_dir,
                log,
            )
            flush()

        combined_pass = bool(results["A"]["pass_bars"]["origin_all_passed"] and results["B"]["pass_bars"]["origin_all_passed"])
        fixed_offset_partial = bool(
            all(
                results[o]["pass_bars"][p]["passed"]
                for o in ("A", "B")
                for p in ("P1_shots_level", "P2_probability_accuracy")
            )
            and not combined_pass
        )
        metadata.update(
            {
                "completed": True,
                "fold_boundaries": {o: origins[o]["boundaries"] for o in origins},
                "feature_identity": {"count": len(feature_cols), "columns": feature_cols, "passed": True},
                "design": {
                    "offset_formula": OFFSET_FORMULA,
                    "offset_fallback_rate60": OFFSET_FALLBACK_RATE60,
                    "target_toi_floor_minutes": RATE_TARGET_TOI_FLOOR_MIN,
                    "train_target_winsor_percentile": RATE_TARGET_WINSOR_PCT,
                    "early_definition": "TOI < 50 minutes",
                    "early_bins": "5-minute bins on [0,50)",
                    "normal_bins": "1-minute bins on [50,66)",
                    "toi_bin_laplace": TOI_BIN_LAPLACE,
                    "origin_cap": ORIGIN_CAP,
                    "lower_tail_k": LOWER_TAIL_KS,
                    "fixed_ev_threshold": FIXED_EV_THRESHOLD,
                    "bootstrap": {
                        "resamples": N_BOOTSTRAP_RESAMPLES,
                        "seed": BOOTSTRAP_SEED,
                        "cluster": "game_id_goalie_id",
                    },
                    "primary_alpha": "one validation-residual alpha per origin, composed with actual validation TOI, shared by single and mixture",
                    "sensitivity_alpha": "one train-residual alpha per origin, composed with actual train TOI, shared by single and mixture",
                    "test_toi_use": "postgame diagnostics only; forbidden from all predictions",
                    "network_policy": "local parquet/model reads only; no network client imported or called",
                },
                "results": results,
                "overall_verdict": {
                    "combined_6b_6c_gate_a_passed": combined_pass,
                    "fixed_offset_mean_model_partial_mechanism_result": fixed_offset_partial,
                    "consequence": (
                        "Full pass: architecture evidence only; authorizes low-credit Gate-B probes and future-season shadow candidate."
                        if combined_pass
                        else "Failure: purchases remain blocked; move queue per section 12.7."
                    ),
                },
                "wall_clock_seconds": time.time() - start,
            }
        )
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(native(metadata), indent=2), encoding="utf-8")

        log("\n" + "=" * 80)
        log("REGISTERED P1-P5 READOUT (FIRST VALID COMPLETED RUN)")
        log("=" * 80)
        for origin in ("A", "B"):
            log(f"Origin {origin}: {json.dumps(native(results[origin]['pass_bars']), sort_keys=True)}")
        log(f"OVERALL COMBINED VERDICT: {'PASS' if combined_pass else 'FAIL'}")
        log(f"Metadata: {metadata_path}")
        log(f"Wall clock: {metadata['wall_clock_seconds']:.1f}s")
        log("No alternative candidate rerun is authorized after this readout.")
        flush()
        return 0
    except Exception as exc:
        metadata["completed"] = False
        metadata["error"] = repr(exc)
        metadata["traceback"] = traceback.format_exc()
        metadata["wall_clock_seconds"] = time.time() - start
        (output_dir / "metadata.json").write_text(json.dumps(native(metadata), indent=2), encoding="utf-8")
        log("\nRUN FAILED BEFORE A VALID COMPLETED P1-P5 READOUT.")
        log(traceback.format_exc())
        flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
