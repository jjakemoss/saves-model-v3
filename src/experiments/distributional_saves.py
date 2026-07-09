"""Import-safe helpers for distributional goalie saves experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import gammaln


CAP = 70
NAIVE_BASELINE_COL = "shots_against_rolling_5"

BASE_SHOTS = dict(
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=400,
    reg_lambda=1.0,
)
SHOTS_CONFIGS = [
    ("base", dict(BASE_SHOTS)),
    ("depth2", {**BASE_SHOTS, "max_depth": 2}),
    ("depth4_mcw20", {**BASE_SHOTS, "max_depth": 4, "min_child_weight": 20}),
    ("more_trees_lowlr", {**BASE_SHOTS, "n_estimators": 800, "learning_rate": 0.03}),
    ("shallow_highreg", {**BASE_SHOTS, "max_depth": 2, "min_child_weight": 30, "reg_lambda": 5.0}),
    ("deep_reg", {**BASE_SHOTS, "max_depth": 5, "min_child_weight": 30, "n_estimators": 300}),
]

BASE_RATE = dict(
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=400,
    reg_lambda=1.0,
)
SAVE_RATE_CONFIGS = [
    ("base", dict(BASE_RATE)),
    ("depth2", {**BASE_RATE, "max_depth": 2}),
    ("depth4_mcw20", {**BASE_RATE, "max_depth": 4, "min_child_weight": 20}),
    ("more_trees_lowlr", {**BASE_RATE, "n_estimators": 800, "learning_rate": 0.03}),
    ("shallow_highreg", {**BASE_RATE, "max_depth": 2, "min_child_weight": 30, "reg_lambda": 5.0}),
    ("deep_reg", {**BASE_RATE, "max_depth": 5, "min_child_weight": 30, "n_estimators": 300}),
]
assert len(SHOTS_CONFIGS) <= 6
assert len(SAVE_RATE_CONFIGS) <= 6

IDENTIFIER_COLS = [
    "game_id",
    "game_date",
    "season",
    "goalie_id",
    "goalie_name",
    "team_abbrev",
    "opponent_team",
]
CURRENT_GAME_OUTCOME_COLS = [
    "saves",
    "shots_against",
    "goals_against",
    "save_percentage",
    "toi",
    "decision",
    "team_goals",
    "team_shots",
    "opp_goals",
    "opp_shots",
    "even_strength_saves",
    "even_strength_shots_against",
    "even_strength_goals_against",
    "power_play_saves",
    "power_play_shots_against",
    "power_play_goals_against",
    "short_handed_saves",
    "short_handed_shots_against",
    "short_handed_goals_against",
]
FORBIDDEN_FEATURE_COLS = set(IDENTIFIER_COLS) | set(CURRENT_GAME_OUTCOME_COLS) | {"betting_line", "over_hit"}
CONTEXT_JOIN_CANDIDATES = [
    ["game_id", "goalie_id"],
    ["game_id", "team_abbrev"],
    ["game_id"],
]
CONTEXT_NON_FEATURE_COLS = set(IDENTIFIER_COLS) | set(CURRENT_GAME_OUTCOME_COLS) | {
    "betting_line",
    "over_hit",
    "odds_over_american",
    "odds_under_american",
    "odds_over_decimal",
    "odds_under_decimal",
    "book_key",
    "num_books",
    # Context-artifact QA/verification columns, not pre-registered model inputs.
    "is_home",
    "schedule_is_home",
    "schedule_opponent_team",
    "schedule_game_type",
    "team_schedule_matched",
    "opponent_schedule_matched",
}
CONTAMINATED_MARKET_COLS = [
    "line_vs_recent_avg",
    "line_vs_season_avg",
    "line_surprise_score",
    "market_vig",
    "impl_prob_over",
    "impl_prob_under",
    "fair_prob_over",
    "fair_prob_under",
    "line_vs_opp_shots",
    "line_is_half",
    "line_is_extreme_high",
    "line_is_extreme_low",
]


@dataclass(frozen=True)
class ModelingFrame:
    df: pd.DataFrame
    base_feature_cols: list[str]
    engineered_cols: list[str]
    context_cols: list[str]
    context_raw_cols: list[str]
    context_null_counts: dict
    context_join_keys: list[str]
    context_coverage_pct: float


def add_engineered_features_no_line(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for w in [3, 5, 10]:
        sr = f"saves_rolling_{w}"
        sar = f"shots_against_rolling_{w}"
        if sr in df.columns and sar in df.columns:
            df[f"save_efficiency_{w}"] = df[sr] / df[sar].clip(lower=1)

    for w in [5, 10]:
        es = f"even_strength_saves_rolling_{w}"
        sr = f"saves_rolling_{w}"
        if es in df.columns and sr in df.columns:
            df[f"es_saves_proportion_{w}"] = df[es] / df[sr].clip(lower=1)

    if "opp_shots_rolling_5" in df.columns and "team_shots_against_rolling_5" in df.columns:
        df["opp_vs_team_shots_5"] = df["opp_shots_rolling_5"] - df["team_shots_against_rolling_5"]
    if "opp_shots_rolling_10" in df.columns and "team_shots_against_rolling_10" in df.columns:
        df["opp_vs_team_shots_10"] = df["opp_shots_rolling_10"] - df["team_shots_against_rolling_10"]

    for w in [5, 10]:
        mean_col = f"saves_rolling_{w}"
        std_col = f"saves_rolling_std_{w}"
        if mean_col in df.columns and std_col in df.columns:
            df[f"saves_cv_{w}"] = df[std_col] / df[mean_col].clip(lower=1)

    for stat in ["saves", "shots_against", "goals_against"]:
        short = f"{stat}_rolling_3"
        long = f"{stat}_rolling_10"
        if short in df.columns and long in df.columns:
            df[f"{stat}_momentum"] = df[short] - df[long]

    sp_short = "save_percentage_rolling_3"
    sp_long = "save_percentage_rolling_10"
    if sp_short in df.columns and sp_long in df.columns:
        df["save_pct_momentum"] = df[sp_short] - df[sp_long]

    if "opp_shots_rolling_5" in df.columns and "shots_against_rolling_5" in df.columns:
        df["expected_workload_diff"] = df["opp_shots_rolling_5"] - df["shots_against_rolling_5"]

    if "goalie_days_rest" in df.columns and "saves_rolling_5" in df.columns:
        df["rest_x_performance"] = df["goalie_days_rest"].clip(upper=7) * df["saves_rolling_5"]

    return df


def _choose_context_join_keys(context_df: pd.DataFrame, clean_df: pd.DataFrame) -> list[str]:
    for keys in CONTEXT_JOIN_CANDIDATES:
        if all(k in context_df.columns and k in clean_df.columns for k in keys):
            if not context_df.duplicated(subset=keys).any():
                return keys
    raise ValueError(
        "Could not find a unique context join key. Expected unique rows by "
        "(game_id, goalie_id), (game_id, team_abbrev), or game_id."
    )


def _select_context_feature_cols(context_df: pd.DataFrame, join_keys: list[str]) -> list[str]:
    numeric_cols = [
        c
        for c in context_df.columns
        if c not in join_keys
        and c not in CONTEXT_NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(context_df[c])
    ]
    forbidden = set(numeric_cols) & FORBIDDEN_FEATURE_COLS
    assert not forbidden, f"Context raw feature set leaks forbidden columns: {sorted(forbidden)}"
    if not numeric_cols:
        raise ValueError("game_context_features.parquet has no numeric context feature columns after exclusions.")
    return numeric_cols


def load_modeling_frame(
    clean_path: Path,
    context_path: Path,
    log: Callable[[str], None],
) -> ModelingFrame:
    if not context_path.exists():
        raise FileNotFoundError(
            f"Missing {context_path}. Run scripts/build_game_context_features.py first, "
            "then rerun scripts/experiment_game_context_distributional.py."
        )

    df = pd.read_parquet(clean_path)
    log(f"Raw clean_training_data.parquet: {len(df)} rows, {len(df.columns)} columns.")
    df = df.sort_values("game_date").reset_index(drop=True)
    assert "betting_line" not in df.columns, (
        "clean_training_data.parquet unexpectedly has betting_line; distributional training assumes no odds line."
    )

    base_feature_cols = [c for c in df.columns if c not in IDENTIFIER_COLS and c not in CURRENT_GAME_OUTCOME_COLS]
    log(f"Base pre-game/context features: {len(base_feature_cols)}")

    df = add_engineered_features_no_line(df)
    engineered_cols = [
        c
        for c in df.columns
        if c not in base_feature_cols and c not in IDENTIFIER_COLS and c not in CURRENT_GAME_OUTCOME_COLS
    ]
    log(f"Engineered features (line-dependent ones omitted): {len(engineered_cols)} -> {engineered_cols}")

    context_df = pd.read_parquet(context_path)
    log(f"Raw game_context_features.parquet: {len(context_df)} rows, {len(context_df.columns)} columns.")
    context_join_keys = _choose_context_join_keys(context_df, df)
    context_raw_cols = _select_context_feature_cols(context_df, context_join_keys)
    context_prefixed = context_df[context_join_keys + context_raw_cols].copy()
    rename_map = {c: f"ctx_{c}" for c in context_raw_cols}
    context_prefixed = context_prefixed.rename(columns=rename_map)
    context_cols = [rename_map[c] for c in context_raw_cols]

    before_rows = len(df)
    df = df.merge(context_prefixed, how="left", on=context_join_keys, indicator="_context_merge")
    assert len(df) == before_rows, "Context merge changed clean-training row count."
    matched = (df["_context_merge"] == "both").values
    context_coverage_pct = float(matched.mean() * 100) if len(df) else 0.0
    log(
        f"Context join keys: {context_join_keys}; coverage "
        f"{int(matched.sum())}/{len(df)} ({context_coverage_pct:.2f}%)."
    )
    df = df.drop(columns=["_context_merge"])

    log(f"Context feature columns: {len(context_cols)} -> {context_cols}")

    base_cols = base_feature_cols + engineered_cols
    base_mat = df[base_cols].values.astype(np.float64)
    assert np.isinf(base_mat).sum() == 0, "base_plus_engineered feature matrix contains infinite values."
    assert np.isnan(base_mat).sum() == 0, "base_plus_engineered feature matrix contains NaN values."

    for name, cols in {"base_plus_engineered": base_cols, "context": context_cols}.items():
        leaked = set(cols) & FORBIDDEN_FEATURE_COLS
        assert not leaked, f"{name} feature set leaks forbidden columns: {sorted(leaked)}"
        mat = df[cols].values.astype(np.float64)
        assert np.isinf(mat).sum() == 0, f"{name} feature matrix contains infinite values."

    context_null_counts = {
        c: int(v) for c, v in df[context_cols].isna().sum().items() if int(v) > 0
    }
    if context_null_counts:
        log(
            "Context feature NaNs retained for XGBoost missing-value handling "
            f"(nonzero null counts: {context_null_counts})."
        )

    n_zero_shots = int((df["shots_against"] == 0).sum())
    if n_zero_shots:
        log(
            f"Note: {n_zero_shots} row(s) with shots_against==0; valid for shots model, "
            "excluded from save-rate training."
        )

    return ModelingFrame(
        df=df,
        base_feature_cols=base_feature_cols,
        engineered_cols=engineered_cols,
        context_cols=context_cols,
        context_raw_cols=context_raw_cols,
        context_null_counts=context_null_counts,
        context_join_keys=context_join_keys,
        context_coverage_pct=context_coverage_pct,
    )


def build_betting_frame(multibook_path: Path, log: Callable[[str], None]) -> pd.DataFrame:
    df = pd.read_parquet(multibook_path)
    log(f"\nRaw multibook_classification_training_data.parquet: {len(df)} rows, {len(df.columns)} columns.")
    df = df.drop(columns=[c for c in CONTAMINATED_MARKET_COLS if c in df.columns], errors="ignore")
    df = df[df["odds_over_american"].notna() & df["odds_under_american"].notna()].copy()
    df = df.sort_values("game_date").reset_index(drop=True)
    log(f"After both-side-odds filter + chronological sort: {len(df)} rows.")
    return df


def assert_feature_matrix_clean(df: pd.DataFrame, cols: list[str], label: str, allow_nan: bool = False) -> None:
    leaked = set(cols) & FORBIDDEN_FEATURE_COLS
    assert not leaked, f"{label} feature set leaks forbidden columns: {sorted(leaked)}"
    mat = df[cols].values.astype(np.float64)
    n_inf = int(np.isinf(mat).sum())
    n_nan = int(np.isnan(mat).sum())
    assert n_inf == 0, f"{label} feature matrix contains {n_inf} infinite values."
    if not allow_nan:
        assert n_nan == 0, f"{label} feature matrix contains {n_nan} NaN values."


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)
    term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    dev = 2 * (term - (y_true - y_pred))
    return float(np.mean(dev))


def weighted_logloss(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    eps = 1e-7
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    weights = np.asarray(weights, dtype=float)
    ll = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return float(np.average(ll, weights=weights))


def top_features(model, k: int = 10) -> list[dict]:
    gain = model.get_booster().get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"feature": f, "gain": float(g)} for f, g in top]


def train_shots_model(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_cols: list[str],
    log: Callable[[str], None],
    label: str,
) -> tuple[xgb.XGBRegressor, dict, list[dict]]:
    assert_feature_matrix_clean(df, feature_cols, f"{label} shots", allow_nan=True)
    X_train = df[feature_cols].iloc[train_idx].astype(np.float32)
    y_train = df["shots_against"].values[train_idx].astype(float)
    X_val = df[feature_cols].iloc[val_idx].astype(np.float32)
    y_val = df["shots_against"].values[val_idx].astype(float)

    log(f"\n--- {label}: shots model (count:poisson), selected on VAL MAE ---")
    log(f"{'config':<20} {'val_mae':>9} {'val_poisson_dev':>16}")
    evaluations = []
    models = {}
    for name, cfg in SHOTS_CONFIGS:
        params = dict(objective="count:poisson", random_state=42, n_jobs=-1, verbosity=0, **cfg)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred_val = np.clip(model.predict(X_val), 1e-3, None)
        val_mae = mae(y_val, pred_val)
        val_dev = poisson_deviance(y_val, pred_val)
        evaluations.append({"config": name, "hyperparams": cfg, "val_mae": val_mae, "val_poisson_deviance": val_dev})
        models[name] = model
        log(f"{name:<20} {val_mae:>9.4f} {val_dev:>16.4f}")

    winner_entry = min(evaluations, key=lambda e: (e["val_mae"], e["val_poisson_deviance"]))
    winner_model = models[winner_entry["config"]]
    log(
        f"{label} shots winner: {winner_entry['config']}  "
        f"val_mae={winner_entry['val_mae']:.4f}  "
        f"val_poisson_deviance={winner_entry['val_poisson_deviance']:.4f}"
    )
    return winner_model, winner_entry, evaluations


def train_save_rate_model(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_cols: list[str],
    log: Callable[[str], None],
    label: str,
) -> tuple[xgb.XGBRegressor, dict, list[dict]]:
    assert_feature_matrix_clean(df, feature_cols, f"{label} save_rate", allow_nan=True)
    shots = df["shots_against"].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        save_rate = df["saves"].values.astype(float) / shots
    valid = shots > 0

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(len(df), dtype=bool)
    val_mask[val_idx] = True
    train_use = train_mask & valid
    val_use = val_mask & valid

    X_train = df[feature_cols].loc[train_use].astype(np.float32)
    y_train = save_rate[train_use]
    w_train = shots[train_use]
    X_val = df[feature_cols].loc[val_use].astype(np.float32)
    y_val = save_rate[val_use]
    w_val = shots[val_use]

    log(f"\n--- {label}: save-rate model (binary:logistic), selected on VAL weighted log-loss ---")
    log(f"{'config':<20} {'val_weighted_logloss':>21}")
    evaluations = []
    models = {}
    for name, cfg in SAVE_RATE_CONFIGS:
        params = dict(objective="binary:logistic", random_state=42, n_jobs=-1, verbosity=0, **cfg)
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False,
        )
        pred_val = np.clip(model.predict(X_val), 1e-6, 1 - 1e-6)
        val_ll = weighted_logloss(y_val, pred_val, w_val)
        evaluations.append({"config": name, "hyperparams": cfg, "val_weighted_logloss": val_ll})
        models[name] = model
        log(f"{name:<20} {val_ll:>21.5f}")

    winner_entry = min(evaluations, key=lambda e: e["val_weighted_logloss"])
    winner_model = models[winner_entry["config"]]
    log(
        f"{label} save-rate winner: {winner_entry['config']}  "
        f"val_weighted_logloss={winner_entry['val_weighted_logloss']:.5f}"
    )
    return winner_model, winner_entry, evaluations


def fit_dispersion(
    shots_model: xgb.XGBRegressor,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    feature_cols: list[str],
    log: Callable[[str], None],
    label: str,
) -> tuple[float, str, dict]:
    X_train = df[feature_cols].iloc[train_idx].astype(np.float32)
    y_train = df["shots_against"].values[train_idx].astype(float)
    mu_train = np.clip(shots_model.predict(X_train), 1e-3, None)
    resid = y_train - mu_train

    train_mean = float(np.mean(mu_train))
    train_var = float(np.mean(resid**2))
    log(f"\n--- {label}: dispersion fit on TRAIN residuals ---")
    log(f"  mean(predicted mu) = {train_mean:.4f}")
    log(f"  mean(residual^2)   = {train_var:.4f}")

    if train_var <= train_mean:
        log("  TRAIN variance <= TRAIN mean: falling back to Poisson (alpha=0).")
        return 0.0, "poisson_fallback", {"train_mean": train_mean, "train_var": train_var}

    mean_mu2 = float(np.mean(mu_train**2))
    alpha = max((train_var - train_mean) / mean_mu2, 1e-6)
    log(f"  NB2 dispersion alpha = {alpha:.6f}  (Var = mean + alpha*mean^2)")
    return alpha, "negative_binomial", {"train_mean": train_mean, "train_var": train_var, "mean_mu2": mean_mu2}


class SavesDistribution:
    def __init__(self, cap: int = CAP):
        self.cap = cap
        self.n_arr = np.arange(cap + 1).astype(np.float64)
        n_idx = self.n_arr.reshape(-1, 1)
        s_idx = self.n_arr.reshape(1, -1)
        diff = n_idx - s_idx
        self.valid = diff >= 0
        self.diff_safe = np.where(self.valid, diff, 0)
        self.s_idx = s_idx
        self.log_C = gammaln(n_idx + 1) - gammaln(s_idx + 1) - gammaln(self.diff_safe + 1)

    def shots_pmf(self, mu: float, alpha: float) -> np.ndarray:
        mu = max(mu, 1e-6)
        n = self.n_arr
        if alpha <= 0:
            logpmf = n * np.log(mu) - mu - gammaln(n + 1)
        else:
            r = 1.0 / alpha
            p = r / (r + mu)
            logpmf = gammaln(n + r) - gammaln(r) - gammaln(n + 1) + r * np.log(p) + n * np.log(1 - p)
        return np.exp(logpmf)

    def saves_pmf(self, mu: float, alpha: float, q: float) -> tuple[np.ndarray, np.ndarray]:
        shots_pmf = self.shots_pmf(mu, alpha)
        q = min(max(q, 1e-6), 1 - 1e-6)
        logB = self.log_C + self.s_idx * np.log(q) + self.diff_safe * np.log(1 - q)
        B = np.where(self.valid, np.exp(logB), 0.0)
        saves_pmf = shots_pmf @ B
        return saves_pmf, shots_pmf

    def price_line(self, saves_pmf: np.ndarray, line: float) -> tuple[float, float, float]:
        s = self.n_arr
        p_over = float(saves_pmf[s > line].sum())
        p_under = float(saves_pmf[s < line].sum())
        p_push = float(saves_pmf[s == line].sum())
        return p_over, p_under, p_push


def compute_distribution_predictions(
    df: pd.DataFrame,
    idx: np.ndarray,
    shots_model: xgb.XGBRegressor,
    rate_model: xgb.XGBRegressor,
    alpha: float,
    shots_feature_cols: list[str],
    rate_feature_cols: list[str],
    dist: SavesDistribution,
    log: Callable[[str], None],
    label: str,
) -> dict:
    X_shots = df[shots_feature_cols].iloc[idx].astype(np.float32)
    X_rate = df[rate_feature_cols].iloc[idx].astype(np.float32)
    mu_arr = np.clip(shots_model.predict(X_shots), 1e-3, None)
    q_arr = np.clip(rate_model.predict(X_rate), 1e-6, 1 - 1e-6)

    n = len(idx)
    pmf_arr = np.zeros((n, dist.cap + 1))
    for i in range(n):
        saves_pmf, _ = dist.saves_pmf(float(mu_arr[i]), alpha, float(q_arr[i]))
        pmf_arr[i] = saves_pmf

    sums = pmf_arr.sum(axis=1)
    bad = np.where(sums < 0.999)[0]
    log(
        f"[{label}] pmf normalization check: n={n}, min_sum={sums.min():.6f}, "
        f"max_sum={sums.max():.6f}, rows below 0.999: {len(bad)}"
    )
    assert len(bad) == 0, f"[{label}] {len(bad)} rows have pmf sum < 0.999 (cap={dist.cap} may be too small)."

    game_ids = df["game_id"].values[idx]
    goalie_ids = df["goalie_id"].values[idx]
    keys = list(zip(game_ids.tolist(), goalie_ids.tolist()))
    assert len(set(keys)) == len(keys), f"[{label}] duplicate (game_id, goalie_id) keys in predictions."
    lookup = {k: i for i, k in enumerate(keys)}
    return {"mu": mu_arr, "q": q_arr, "pmf": pmf_arr, "keys": keys, "lookup": lookup, "idx": idx}


def join_and_price(
    df_bet_fold: pd.DataFrame,
    dist_preds: dict,
    dist: SavesDistribution,
    log: Callable[[str], None],
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    game_ids = df_bet_fold["game_id"].values
    goalie_ids = df_bet_fold["goalie_id"].values
    lines = df_bet_fold["betting_line"].values.astype(float)
    n = len(df_bet_fold)
    p_over = np.full(n, np.nan)
    p_under = np.full(n, np.nan)
    p_push = np.full(n, np.nan)
    matched = np.zeros(n, dtype=bool)
    for i in range(n):
        key = (int(game_ids[i]), int(goalie_ids[i]))
        pi = dist_preds["lookup"].get(key)
        if pi is None:
            continue
        matched[i] = True
        po, pu, pp = dist.price_line(dist_preds["pmf"][pi], lines[i])
        p_over[i], p_under[i], p_push[i] = po, pu, pp

    coverage = matched.mean() * 100 if n else 0.0
    log(f"[{label}] join coverage: {int(matched.sum())}/{n} rows matched ({coverage:.2f}%).")
    if coverage < 95.0:
        log("  WARNING: join coverage below 95%; investigate before trusting betting numbers.")
    return p_over, p_under, p_push, matched, coverage


def intrinsic_quality_metrics(
    df: pd.DataFrame,
    val_idx: np.ndarray,
    dist_preds_val: dict,
    dist: SavesDistribution,
    log: Callable[[str], None],
    label: str,
) -> dict:
    y_val_shots = df["shots_against"].values[val_idx].astype(float)
    mu_val = dist_preds_val["mu"]
    naive_pred = df[NAIVE_BASELINE_COL].values[val_idx].astype(float)

    model_mae = mae(y_val_shots, mu_val)
    naive_mae = mae(y_val_shots, naive_pred)
    log(f"\n--- {label}: intrinsic distribution quality on VAL ---")
    log(f"Shots MAE: model={model_mae:.4f}  naive({NAIVE_BASELINE_COL})={naive_mae:.4f}")

    saves_actual = df["saves"].values[val_idx].astype(int)
    saves_actual_c = np.clip(saves_actual, 0, dist.cap)
    pmf_arr = dist_preds_val["pmf"]
    n = len(val_idx)
    row_idx = np.arange(n)
    p_actual = np.clip(pmf_arr[row_idx, saves_actual_c], 1e-12, None)
    mean_pmf_logloss = float(np.mean(-np.log(p_actual)))

    cdf = np.cumsum(pmf_arr, axis=1)
    cov50 = 0
    cov80 = 0
    for i in range(n):
        row_cdf = cdf[i]
        lo50 = np.searchsorted(row_cdf, 0.25, side="left")
        hi50 = np.searchsorted(row_cdf, 0.75, side="left")
        lo80 = np.searchsorted(row_cdf, 0.10, side="left")
        hi80 = np.searchsorted(row_cdf, 0.90, side="left")
        if lo50 <= saves_actual_c[i] <= hi50:
            cov50 += 1
        if lo80 <= saves_actual_c[i] <= hi80:
            cov80 += 1

    cov50_pct = cov50 / n * 100
    cov80_pct = cov80 / n * 100
    rng = np.random.RandomState(123)
    pit_vals = np.zeros(n)
    for i in range(n):
        y = saves_actual_c[i]
        cdf_y = cdf[i, y]
        cdf_y_minus1 = cdf[i, y - 1] if y > 0 else 0.0
        pit_vals[i] = cdf_y_minus1 + rng.uniform() * (cdf_y - cdf_y_minus1)
    hist, _ = np.histogram(pit_vals, bins=10, range=(0, 1))
    freqs = (hist / n).tolist()
    log(f"Mean pmf log-loss at actual saves value: {mean_pmf_logloss:.4f}")
    log(f"Central 50% coverage: {cov50_pct:.1f}%; central 80% coverage: {cov80_pct:.1f}%")
    log(f"PIT histogram frequencies: {[round(f, 3) for f in freqs]}")

    return {
        "naive_baseline_col": NAIVE_BASELINE_COL,
        "shots_mae_model": model_mae,
        "shots_mae_naive": naive_mae,
        "mean_pmf_logloss_at_actual": mean_pmf_logloss,
        "coverage_50pct_nominal_actual": cov50_pct,
        "coverage_80pct_nominal_actual": cov80_pct,
        "pit_histogram_10bins": freqs,
        "pit_bin_freq_min": min(freqs),
        "pit_bin_freq_max": max(freqs),
        "n_val_rows": n,
    }
