"""
Probability calibration layer for the clean retrain
(models/trained/tuned_v2_clean_20260707_212023/).

Roadmap item 4, docs/OFFSEASON_OPTIMIZATION_PLAN.md sections 3.2 and 5.
This is the decisive test of whether the model knows anything the market
does not: the raw model bets a third of all lines claiming +22% average
probability edge and resolves at 49% (section 3.1). Calibration either
finds an honest +EV pocket that survives, or it does not -- and either
answer is reported plainly, per CLAUDE.md.

PROTOCOL (pre-registered, followed exactly, no peeking at test until
step 6):

  1. Chronological data pipeline rebuilt to match scripts/tune_hyperparameters.py
     EXACTLY (copied below, not imported -- that module has no __main__
     guard and runs a ~30 minute tuning search on import).
  2. Raw P(over) predicted on ALL folds ONCE with the saved booster. Test
     predictions may be computed but test outcomes/odds are not inspected
     before step 6.
  3. Validation fold split chronologically in half: CAL-FIT (first half)
     fits two calibrators (isotonic regression on the raw probability;
     Platt scaling = sklearn LogisticRegression on the raw margin/logit,
     i.e. the pre-sigmoid score -- the classic Platt-scaling input).
     CAL-SELECT (second half) picks the winner by Brier score.
  4. Diagnostics on CAL-SELECT before any test contact: reliability
     tables, probability-compression quantiles, AUC (model vs market
     vig-free implied probability -- the single most important number),
     and a cheap model+market blend check.
  5. Threshold sweep on CAL-SELECT with CALIBRATED probabilities, two
     arms (both-sides, UNDER-only). Policies are pre-registered (highest
     CAL-SELECT ROI subject to >=50 bets, else the lowest threshold)
     BEFORE touching test.
  6. SINGLE TEST TOUCH: the two pre-registered policies evaluated on the
     test fold exactly once, with row-level and cluster (goalie-night)
     bootstrap 95% CIs.
  7. Artifacts saved: calibrator.pkl, calibration_metadata.json, this
     script, and a run log.

Do NOT modify: src/betting/predictor.py, src/betting/feature_calculator.py,
scripts/tune_hyperparameters.py, or anything in
models/trained/tuned_v1_20260201_155204/ (the live production model).
This script only reads the clean model at
models/trained/tuned_v2_clean_20260707_212023/ and writes calibration
artifacts alongside it. Nothing here is wired into production.

Usage:
    python scripts/calibrate_model.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from betting.odds_utils import calculate_ev, calculate_payout, american_to_implied_prob

MODEL_DIR = Path('models/trained/tuned_v2_clean_20260707_212023')
DATA_PATH = Path('data/processed/multibook_classification_training_data.parquet')

EV_THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
MIN_BETS_FOR_POLICY = 50
FALLBACK_THRESHOLD = 0.02


# ============================================================
# Section 1: exact copy of scripts/tune_hyperparameters.py's data
# pipeline. Copied, not imported. Any change here must be mirrored by
# hand if tune_hyperparameters.py ever changes -- that is intentional,
# since importing it executes a full tuning run.
# ============================================================

def add_all_engineered_features(df):
    """Reproduce the 18 engineered features from optimize_features.py
    (verbatim copy of scripts/tune_hyperparameters.py's function)."""
    df = df.copy()

    # Interaction features
    for w in [3, 5, 10]:
        sr = f'saves_rolling_{w}'
        sar = f'shots_against_rolling_{w}'
        if sr in df.columns and sar in df.columns:
            df[f'save_efficiency_{w}'] = df[sr] / df[sar].clip(lower=1)

    for w in [5, 10]:
        es = f'even_strength_saves_rolling_{w}'
        sr = f'saves_rolling_{w}'
        if es in df.columns and sr in df.columns:
            df[f'es_saves_proportion_{w}'] = df[es] / df[sr].clip(lower=1)

    if 'opp_shots_rolling_5' in df.columns and 'team_shots_against_rolling_5' in df.columns:
        df['opp_vs_team_shots_5'] = df['opp_shots_rolling_5'] - df['team_shots_against_rolling_5']
    if 'opp_shots_rolling_10' in df.columns and 'team_shots_against_rolling_10' in df.columns:
        df['opp_vs_team_shots_10'] = df['opp_shots_rolling_10'] - df['team_shots_against_rolling_10']

    # Volatility features
    for w in [5, 10]:
        mean_col = f'saves_rolling_{w}'
        std_col = f'saves_rolling_std_{w}'
        if mean_col in df.columns and std_col in df.columns:
            df[f'saves_cv_{w}'] = df[std_col] / df[mean_col].clip(lower=1)
        if std_col in df.columns and 'betting_line' in df.columns:
            df[f'volatility_vs_line_{w}'] = df[std_col] / df['betting_line'].clip(lower=1)

    # Trend/momentum features
    for stat in ['saves', 'shots_against', 'goals_against']:
        short = f'{stat}_rolling_3'
        long = f'{stat}_rolling_10'
        if short in df.columns and long in df.columns:
            df[f'{stat}_momentum'] = df[short] - df[long]

    sp_short = 'save_percentage_rolling_3'
    sp_long = 'save_percentage_rolling_10'
    if sp_short in df.columns and sp_long in df.columns:
        df['save_pct_momentum'] = df[sp_short] - df[sp_long]

    # Matchup context features
    if 'opp_shots_rolling_5' in df.columns and 'shots_against_rolling_5' in df.columns:
        df['expected_workload_diff'] = df['opp_shots_rolling_5'] - df['shots_against_rolling_5']

    if 'opp_shots_rolling_5' in df.columns and 'opp_goals_rolling_5' in df.columns:
        opp_saves_implied = df['opp_shots_rolling_5'] - df['opp_goals_rolling_5']
        df['line_vs_opp_implied_saves'] = df['betting_line'] - opp_saves_implied

    if 'goalie_days_rest' in df.columns and 'saves_rolling_5' in df.columns:
        df['rest_x_performance'] = df['goalie_days_rest'].clip(upper=7) * df['saves_rolling_5']

    return df


def build_modeling_frame():
    """Rebuild the exact modeling frame scripts/tune_hyperparameters.py
    used to train the clean model: drop market-feature columns, sort
    chronologically, filter to rows with both-side odds, add the 18
    engineered features. Returns (df, feature_cols)."""
    df_raw = pd.read_parquet(DATA_PATH)

    market_features = [
        'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
        'market_vig', 'impl_prob_over', 'impl_prob_under',
        'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
        'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low'
    ]
    df_raw = df_raw.drop(columns=[c for c in market_features if c in df_raw.columns], errors='ignore')
    df_raw = df_raw.sort_values('game_date').reset_index(drop=True)
    df_raw = df_raw[df_raw['odds_over_american'].notna() & df_raw['odds_under_american'].notna()].reset_index(drop=True)

    df = add_all_engineered_features(df_raw)

    EXCLUDED = [
        'game_id', 'goalie_id', 'game_date', 'over_hit',
        'odds_over_american', 'odds_under_american',
        'odds_over_decimal', 'odds_under_decimal', 'num_books',
        'team_abbrev', 'opponent_team', 'toi', 'season',
        'saves', 'shots_against', 'goals_against', 'save_percentage',
        'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
        'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
        'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
        'team_goals', 'team_shots', 'opp_goals', 'opp_shots', 'line_margin',
        'book_key', 'decision', 'team_id', 'goalie_name',
        'saves_margin', 'over_line',
        '_game_date_str', '_lookup_key',
    ]
    feature_cols = [c for c in df.columns if c not in EXCLUDED]

    assert len(df) == 13192, f"Expected 13192 rows, got {len(df)}. Training data has changed."
    assert len(feature_cols) == 114, f"Expected 114 feature columns, got {len(feature_cols)}."

    with open(MODEL_DIR / 'classifier_feature_names.json') as f:
        saved_feature_names = json.load(f)
    assert feature_cols == saved_feature_names, (
        "Feature column order does not match the saved model's classifier_feature_names.json. "
        "The booster's raw predictions would be meaningless with mismatched column order."
    )

    return df, feature_cols


# ============================================================
# Section 2: helpers -- bootstrap CIs, reliability tables, bet grading
# ============================================================

def bootstrap_roi_ci(profits, n_resamples=10000, seed=42, ci_pct=95.0):
    """Row-level percentile-method bootstrap CI on ROI, resampling
    per-bet profits with replacement."""
    profits = np.asarray(profits, dtype=float)
    n_bets = len(profits)
    if n_bets == 0:
        return {'lower': 0.0, 'upper': 0.0, 'n_bets': 0}
    rng = np.random.RandomState(seed)
    resample_idx = rng.randint(0, n_bets, size=(n_resamples, n_bets))
    boot_rois = profits[resample_idx].mean(axis=1) * 100
    alpha = (100.0 - ci_pct) / 2.0
    return {
        'lower': float(np.percentile(boot_rois, alpha)),
        'upper': float(np.percentile(boot_rois, 100.0 - alpha)),
        'n_bets': int(n_bets),
    }


def cluster_bootstrap_roi_ci(profits, cluster_ids, n_resamples=10000, seed=42, ci_pct=95.0):
    """Cluster (goalie-night) bootstrap CI on ROI: resamples whole
    clusters (unique game_id/goalie_id pairs) with replacement rather
    than individual rows, since multibook rows on the same goalie-night
    are correlated and a row-level CI overstates precision."""
    profits = np.asarray(profits, dtype=float)
    cluster_ids = np.asarray(cluster_ids, dtype=object)
    unique_clusters, inv = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    if n_clusters == 0:
        return {'lower': 0.0, 'upper': 0.0, 'n_clusters': 0}

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
        'lower': float(np.percentile(boot_rois, alpha)),
        'upper': float(np.percentile(boot_rois, 100.0 - alpha)),
        'n_clusters': int(n_clusters),
    }


def reliability_table(prob, y, n_bins=10):
    """Decile reliability table: bucket prob into n_bins quantile bins,
    report bin range, n, mean predicted prob, and actual over rate."""
    d = pd.DataFrame({'prob': np.asarray(prob), 'y': np.asarray(y)})
    try:
        d['bin'] = pd.qcut(d['prob'], q=n_bins, duplicates='drop')
    except ValueError:
        d['bin'] = pd.cut(d['prob'], bins=n_bins)
    grouped = d.groupby('bin', observed=True).agg(
        n=('y', 'size'), mean_pred=('prob', 'mean'), actual_rate=('y', 'mean')
    ).reset_index()
    grouped['bin'] = grouped['bin'].astype(str)
    return grouped


def print_reliability_table(label, table):
    print(f"\n  {label}")
    print(f"  {'bin':<22} {'n':>6} {'mean_pred':>10} {'actual_rate':>12}")
    for _, row in table.iterrows():
        print(f"  {row['bin']:<22} {row['n']:>6.0f} {row['mean_pred']:>10.4f} {row['actual_rate']:>12.4f}")


def prob_quantiles(arr):
    qs = [1, 5, 25, 50, 75, 95, 99]
    return {f'p{q}': float(np.percentile(arr, q)) for q in qs}


def decide_bet(p_over, p_under, odds_over, odds_under, ev_threshold, arm):
    """arm='both' replicates ClassifierTrainer.evaluate_profitability's
    decision logic exactly (bet OVER if it clears threshold and beats
    UNDER's EV, else bet UNDER if it clears threshold). arm='under_only'
    never bets OVER."""
    ev_over = calculate_ev(p_over, odds_over)
    ev_under = calculate_ev(p_under, odds_under)
    if arm == 'both':
        if ev_over >= ev_threshold and ev_over > ev_under:
            return 'OVER', ev_over
        elif ev_under >= ev_threshold:
            return 'UNDER', ev_under
        return None, None
    elif arm == 'under_only':
        if ev_under >= ev_threshold:
            return 'UNDER', ev_under
        return None, None
    raise ValueError(f"Unknown arm: {arm}")


def grade_bets(idx, prob_over_full, y_full, odds_over_full, odds_under_full,
               game_id_full, goalie_id_full, ev_threshold, arm):
    """Grade bets over a fold's index array using calibrated probabilities.
    Returns a list of per-bet dicts (bet, profit, won, ev, cluster_id)."""
    results = []
    for i in idx:
        p_over = prob_over_full[i]
        p_under = 1 - p_over
        odds_over = odds_over_full[i]
        odds_under = odds_under_full[i]
        bet, ev = decide_bet(p_over, p_under, odds_over, odds_under, ev_threshold, arm)
        if bet is None:
            continue
        actual_over = y_full[i]
        if bet == 'OVER':
            won = (actual_over == 1)
            profit = calculate_payout(1.0, odds_over, won)
        else:
            won = (actual_over == 0)
            profit = calculate_payout(1.0, odds_under, won)
        results.append({
            'row_idx': int(i),
            'bet': bet,
            'profit': float(profit),
            'won': bool(won),
            'ev': float(ev),
            # String, not tuple: a list of equal-length tuples gets collapsed
            # into a 2D array by np.asarray (even with dtype=object), which
            # breaks np.unique(..., return_inverse=True) downstream.
            'cluster_id': f"{int(game_id_full[i])}_{int(goalie_id_full[i])}",
        })
    return results


def summarize_bets(results, fold_size):
    n_bets = len(results)
    if n_bets == 0:
        return {'bets': 0, 'bet_rate': 0.0, 'hit_rate': 0.0, 'roi': 0.0, 'profit': 0.0}
    wins = sum(r['won'] for r in results)
    profit = sum(r['profit'] for r in results)
    return {
        'bets': n_bets,
        'bet_rate': n_bets / fold_size * 100,
        'hit_rate': wins / n_bets * 100,
        'roi': profit / n_bets * 100,
        'profit': profit,
    }


def side_breakdown(results):
    breakdown = {}
    for side in ('OVER', 'UNDER'):
        side_bets = [r for r in results if r['bet'] == side]
        n_side = len(side_bets)
        if n_side == 0:
            breakdown[side] = {'bets': 0, 'hit_rate': 0.0, 'roi': 0.0, 'profit': 0.0}
            continue
        wins = sum(r['won'] for r in side_bets)
        profit = sum(r['profit'] for r in side_bets)
        breakdown[side] = {
            'bets': n_side,
            'hit_rate': wins / n_side * 100,
            'roi': profit / n_side * 100,
            'profit': profit,
        }
    return breakdown


def apply_calibrator(calibrator_name, sk_model, raw_prob, raw_margin):
    """Apply the chosen calibrator to raw predictions. Isotonic operates
    on the raw post-sigmoid probability; Platt operates on the raw
    pre-sigmoid margin (the classic Platt-scaling input).

    NOTE: deliberately a plain function, not a method on a custom class.
    A custom class instance saved via joblib from a script run as
    __main__ pickles with module '__main__' and cannot be unpickled
    from any other importing context (verified the hard way while
    building this script). Saving the bare sklearn object plus a
    calibrator_type string in the metadata sidesteps that entirely --
    both IsotonicRegression and LogisticRegression are pickled from
    their real sklearn module paths."""
    if calibrator_name == 'isotonic':
        return sk_model.predict(np.asarray(raw_prob))
    elif calibrator_name == 'platt':
        m = np.asarray(raw_margin).reshape(-1, 1)
        return sk_model.predict_proba(m)[:, 1]
    raise ValueError(f"Unknown calibrator_name: {calibrator_name}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("PROBABILITY CALIBRATION -- tuned_v2_clean_20260707_212023")
    print(f"Run timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    df, feature_cols = build_modeling_frame()
    n = len(df)
    print(f"\nModeling frame: {n} rows, {len(feature_cols)} features (verified against saved model).")

    # Fold boundaries -- identical to tune_hyperparameters.py's 60/20/20
    # chronological split, and pre-registered in the task brief.
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    assert train_end == 7915 and val_end == 10553, (
        f"Fold boundaries drifted from pre-registered indices: train_end={train_end}, val_end={val_end}"
    )
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    cal_fit_end = train_end + (val_end - train_end) // 2  # 7915 + 1319 = 9234
    cal_fit_idx = np.arange(train_end, cal_fit_end)
    cal_select_idx = np.arange(cal_fit_end, val_end)
    assert cal_fit_idx[0] == 7915 and cal_fit_idx[-1] == 9233
    assert cal_select_idx[0] == 9234 and cal_select_idx[-1] == 10552

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"  CAL-FIT:    rows 7915..9234  (n={len(cal_fit_idx)})")
    print(f"  CAL-SELECT: rows 9234..10553 (n={len(cal_select_idx)})")

    # ------------------------------------------------------------
    # Step 2: raw predictions on ALL folds, ONCE. Test predictions may
    # be computed here but test outcomes/odds are not inspected until
    # step 6.
    # ------------------------------------------------------------
    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / 'classifier_model.json'))
    X_all = df[feature_cols].values.astype(np.float32)
    dmat = xgb.DMatrix(X_all)
    raw_prob = booster.predict(dmat)                          # P(over), post-sigmoid
    raw_margin = booster.predict(dmat, output_margin=True)    # pre-sigmoid score (Platt input)

    y = df['over_hit'].values.astype(int)
    odds_over = df['odds_over_american'].values.astype(float)
    odds_under = df['odds_under_american'].values.astype(float)
    game_id = df['game_id'].values
    goalie_id = df['goalie_id'].values

    impl_over = np.array([american_to_implied_prob(o) for o in odds_over])
    impl_under = np.array([american_to_implied_prob(o) for o in odds_under])
    market_prob = impl_over / (impl_over + impl_under)  # vig-free implied P(over)

    print("\nRaw probabilities computed for all folds (train/val/test).")
    print("Test outcomes/odds will not be inspected until step 6.")

    # ------------------------------------------------------------
    # Step 3: fit calibrators on CAL-FIT, select on CAL-SELECT by Brier.
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: CALIBRATOR FIT (CAL-FIT) + SELECTION (CAL-SELECT)")
    print("=" * 80)

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_prob[cal_fit_idx], y[cal_fit_idx])

    platt = LogisticRegression()
    platt.fit(raw_margin[cal_fit_idx].reshape(-1, 1), y[cal_fit_idx])

    iso_calselect = apply_calibrator('isotonic', iso, raw_prob[cal_select_idx], None)
    platt_calselect = apply_calibrator('platt', platt, None, raw_margin[cal_select_idx])

    y_calselect = y[cal_select_idx]

    def _clip(p):
        return np.clip(p, 1e-6, 1 - 1e-6)

    brier_raw = brier_score_loss(y_calselect, raw_prob[cal_select_idx])
    brier_iso = brier_score_loss(y_calselect, iso_calselect)
    brier_platt = brier_score_loss(y_calselect, platt_calselect)

    ll_raw = log_loss(y_calselect, _clip(raw_prob[cal_select_idx]), labels=[0, 1])
    ll_iso = log_loss(y_calselect, _clip(iso_calselect), labels=[0, 1])
    ll_platt = log_loss(y_calselect, _clip(platt_calselect), labels=[0, 1])

    print(f"\n{'Calibrator':<12} {'Brier':>10} {'LogLoss':>10}")
    print(f"{'Raw':<12} {brier_raw:>10.5f} {ll_raw:>10.5f}")
    print(f"{'Isotonic':<12} {brier_iso:>10.5f} {ll_iso:>10.5f}")
    print(f"{'Platt':<12} {brier_platt:>10.5f} {ll_platt:>10.5f}")

    if brier_iso <= brier_platt:
        chosen_sk_model = iso
        calibrator_name = 'isotonic'
    else:
        chosen_sk_model = platt
        calibrator_name = 'platt'
    print(f"\nChosen calibrator (lower CAL-SELECT Brier): {calibrator_name}")

    calibrated_prob = apply_calibrator(calibrator_name, chosen_sk_model, raw_prob, raw_margin)  # full-length, all folds

    # ------------------------------------------------------------
    # Step 4: diagnostics on CAL-SELECT (pre-test).
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: DIAGNOSTICS ON CAL-SELECT (before any test contact)")
    print("=" * 80)

    # 4a. Reliability tables
    print("\n--- 4a. Reliability tables (CAL-SELECT) ---")
    raw_reliability = reliability_table(raw_prob[cal_select_idx], y_calselect)
    cal_reliability = reliability_table(calibrated_prob[cal_select_idx], y_calselect)
    print_reliability_table("RAW P(over) deciles:", raw_reliability)
    print_reliability_table(f"CALIBRATED ({calibrator_name}) P(over) deciles:", cal_reliability)

    raw_high = raw_prob[cal_select_idx] >= 0.6
    cal_high = calibrated_prob[cal_select_idx] >= 0.6
    raw_high_actual = y_calselect[raw_high].mean() if raw_high.sum() > 0 else float('nan')
    cal_high_actual = y_calselect[cal_high].mean() if cal_high.sum() > 0 else float('nan')
    print(f"\n  Inversion check (doc 1.3): P(over)>=0.6 bucket actual over-rate")
    print(f"    Raw:        n={int(raw_high.sum())}, actual over rate={raw_high_actual*100:.1f}%")
    print(f"    Calibrated: n={int(cal_high.sum())}, actual over rate={cal_high_actual*100:.1f}%")

    # 4b. Probability compression
    print("\n--- 4b. Probability compression (quantiles, CAL-SELECT) ---")
    raw_q = prob_quantiles(raw_prob[cal_select_idx])
    cal_q = prob_quantiles(calibrated_prob[cal_select_idx])
    print(f"  {'quantile':<8} {'raw':>10} {'calibrated':>12}")
    for q in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  {q:>6}% {raw_q[f'p{q}']:>11.4f} {cal_q[f'p{q}']:>12.4f}")

    # 4c. Discrimination (AUC) -- model vs market
    print("\n--- 4c. Discrimination: AUC (calibration cannot fix this) ---")
    auc_model_calselect = roc_auc_score(y_calselect, raw_prob[cal_select_idx])
    auc_market_calselect = roc_auc_score(y_calselect, market_prob[cal_select_idx])
    auc_model_calfit = roc_auc_score(y[cal_fit_idx], raw_prob[cal_fit_idx])
    print(f"  Model AUC on CAL-SELECT:          {auc_model_calselect:.4f}")
    print(f"  Market (vig-free) AUC on CAL-SELECT: {auc_market_calselect:.4f}")
    print(f"  Model AUC on CAL-FIT (reference):  {auc_model_calfit:.4f}")
    if auc_model_calselect <= auc_market_calselect:
        print("  --> Model AUC <= market AUC: no calibration can create an edge over this market.")
    else:
        print("  --> Model AUC > market AUC: there is at least discrimination headroom over the market.")

    # 4d. Blend check
    print("\n--- 4d. Blend check: calibrated model prob + market prob -> logistic regression ---")
    calibrated_calfit = apply_calibrator(calibrator_name, chosen_sk_model, raw_prob[cal_fit_idx], raw_margin[cal_fit_idx])
    blend_X_fit = np.column_stack([calibrated_calfit, market_prob[cal_fit_idx]])
    blend_y_fit = y[cal_fit_idx]
    blend_model = LogisticRegression()
    blend_model.fit(blend_X_fit, blend_y_fit)

    blend_X_select = np.column_stack([calibrated_prob[cal_select_idx], market_prob[cal_select_idx]])
    blend_pred_select = blend_model.predict_proba(blend_X_select)[:, 1]
    auc_blend = roc_auc_score(y_calselect, blend_pred_select)
    brier_blend = brier_score_loss(y_calselect, blend_pred_select)
    blend_coef = blend_model.coef_[0]
    blend_intercept = blend_model.intercept_[0]

    print(f"  Blend coefficients: model={blend_coef[0]:+.4f}, market={blend_coef[1]:+.4f}, intercept={blend_intercept:+.4f}")
    print(f"  Blend AUC on CAL-SELECT:   {auc_blend:.4f}  (model alone: {auc_model_calselect:.4f}, market alone: {auc_market_calselect:.4f})")
    print(f"  Blend Brier on CAL-SELECT: {brier_blend:.5f}  (calibrated model alone: {brier_iso if calibrator_name=='isotonic' else brier_platt:.5f})")

    # ------------------------------------------------------------
    # Step 5: threshold sweep on CAL-SELECT using CALIBRATED probabilities.
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: THRESHOLD SWEEP ON CAL-SELECT (calibrated probabilities)")
    print("=" * 80)

    fold_size_calselect = len(cal_select_idx)
    sweep = {'both': [], 'under_only': []}
    for arm in ('both', 'under_only'):
        print(f"\n--- Arm: {arm} ---")
        print(f"  {'thresh':>7} {'bets':>6} {'bet_rate':>9} {'hit_rate':>9} {'roi':>9}")
        for thresh in EV_THRESHOLDS:
            results = grade_bets(
                cal_select_idx, calibrated_prob, y, odds_over, odds_under,
                game_id, goalie_id, thresh, arm
            )
            summary = summarize_bets(results, fold_size_calselect)
            summary['threshold'] = thresh
            summary['results'] = results
            sweep[arm].append(summary)
            print(f"  {thresh:>6.2f} {summary['bets']:>6} {summary['bet_rate']:>8.1f}% {summary['hit_rate']:>8.1f}% {summary['roi']:>+8.2f}%")

    # Pre-register policy per arm from CAL-SELECT results only.
    def select_policy(arm_sweep):
        eligible = [r for r in arm_sweep if r['bets'] >= MIN_BETS_FOR_POLICY]
        if eligible:
            best = max(eligible, key=lambda r: r['roi'])
            return best['threshold'], best
        fallback = next(r for r in arm_sweep if r['threshold'] == FALLBACK_THRESHOLD)
        return FALLBACK_THRESHOLD, fallback

    policy_a_thresh, policy_a_calselect = select_policy(sweep['both'])
    policy_b_thresh, policy_b_calselect = select_policy(sweep['under_only'])

    print("\n" + "=" * 80)
    print("PRE-REGISTERED POLICIES (chosen from CAL-SELECT only, before touching test)")
    print("=" * 80)
    print(f"  Policy A (both sides):  EV threshold = {policy_a_thresh:.2f}  "
          f"(CAL-SELECT: {policy_a_calselect['bets']} bets, {policy_a_calselect['roi']:+.2f}% ROI)")
    print(f"  Policy B (UNDER only):  EV threshold = {policy_b_thresh:.2f}  "
          f"(CAL-SELECT: {policy_b_calselect['bets']} bets, {policy_b_calselect['roi']:+.2f}% ROI)")

    # ------------------------------------------------------------
    # Step 6: SINGLE TEST TOUCH.
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6: SINGLE TEST-FOLD TOUCH")
    print("=" * 80)

    fold_size_test = len(test_idx)
    test_report = {}
    for policy_name, arm, thresh in [('A_both', 'both', policy_a_thresh), ('B_under_only', 'under_only', policy_b_thresh)]:
        results = grade_bets(
            test_idx, calibrated_prob, y, odds_over, odds_under,
            game_id, goalie_id, thresh, arm
        )
        summary = summarize_bets(results, fold_size_test)
        profits = [r['profit'] for r in results]
        cluster_ids = [r['cluster_id'] for r in results]
        row_ci = bootstrap_roi_ci(profits, n_resamples=10000, seed=42, ci_pct=95.0)
        cluster_ci = cluster_bootstrap_roi_ci(profits, cluster_ids, n_resamples=10000, seed=42, ci_pct=95.0)
        unique_nights = len(set(cluster_ids))

        print(f"\n--- Policy {policy_name} (arm={arm}, threshold={thresh:.2f}) ---")
        print(f"  Bets: {summary['bets']} | Bet rate: {summary['bet_rate']:.1f}% | Hit rate: {summary['hit_rate']:.1f}% | ROI: {summary['roi']:+.2f}%")
        print(f"  Row-level bootstrap 95% CI:     [{row_ci['lower']:+.2f}%, {row_ci['upper']:+.2f}%]")
        print(f"  Cluster (goalie-night) bootstrap 95% CI: [{cluster_ci['lower']:+.2f}%, {cluster_ci['upper']:+.2f}%]  (n_clusters={cluster_ci['n_clusters']})")
        print(f"  Unique goalie-nights among bets: {unique_nights}")

        breakdown = None
        if policy_name == 'A_both':
            breakdown = side_breakdown(results)
            print("  OVER/UNDER breakdown:")
            for side in ('OVER', 'UNDER'):
                b = breakdown[side]
                print(f"    {side}: {b['bets']} bets | hit rate {b['hit_rate']:.1f}% | ROI {b['roi']:+.2f}%")

        test_report[policy_name] = {
            'arm': arm,
            'threshold': thresh,
            'summary': summary,
            'row_bootstrap_ci_95': row_ci,
            'cluster_bootstrap_ci_95': cluster_ci,
            'unique_goalie_nights': unique_nights,
            'side_breakdown': breakdown,
        }

    # Test-fold reliability + Brier, calibrated vs raw (same single pass)
    print("\n--- Test-fold reliability + Brier (calibrated vs raw, single touch) ---")
    y_test = y[test_idx]
    brier_raw_test = brier_score_loss(y_test, raw_prob[test_idx])
    brier_cal_test = brier_score_loss(y_test, calibrated_prob[test_idx])
    print(f"  Brier raw (test):        {brier_raw_test:.5f}")
    print(f"  Brier calibrated (test): {brier_cal_test:.5f}")
    raw_reliability_test = reliability_table(raw_prob[test_idx], y_test)
    cal_reliability_test = reliability_table(calibrated_prob[test_idx], y_test)
    print_reliability_table("RAW P(over) deciles (TEST):", raw_reliability_test)
    print_reliability_table(f"CALIBRATED ({calibrator_name}) P(over) deciles (TEST):", cal_reliability_test)

    # ------------------------------------------------------------
    # Step 7: save artifacts
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 7: SAVING ARTIFACTS")
    print("=" * 80)

    calibrator_path = MODEL_DIR / 'calibrator.pkl'
    joblib.dump(chosen_sk_model, calibrator_path)
    print(f"  Saved calibrator to: {calibrator_path}")
    print(f"  Calibrator type: {calibrator_name} (see calibration_metadata.json 'calibrator_type' / 'calibrator_usage')")

    def table_to_records(t):
        return t.to_dict(orient='records')

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'source_model_dir': str(MODEL_DIR),
        'calibrator_type': calibrator_name,
        'calibrator_usage': (
            "calibrator.pkl holds the bare sklearn object (IsotonicRegression or "
            "LogisticRegression), joblib-loadable from any context. If calibrator_type "
            "== 'isotonic': calibrated_prob = calibrator.predict(raw_prob) where raw_prob "
            "is the booster's sigmoid output. If calibrator_type == 'platt': "
            "calibrated_prob = calibrator.predict_proba(raw_margin.reshape(-1,1))[:,1] "
            "where raw_margin is booster.predict(dmat, output_margin=True) -- the "
            "pre-sigmoid score, NOT raw_prob."
        ),
        'fold_boundaries': {
            'train': [0, int(train_end)],
            'val': [int(train_end), int(val_end)],
            'test': [int(val_end), int(n)],
            'cal_fit': [int(cal_fit_idx[0]), int(cal_fit_end)],
            'cal_select': [int(cal_fit_end), int(val_end)],
        },
        'calibration_fit': {
            'brier_calselect': {'raw': brier_raw, 'isotonic': brier_iso, 'platt': brier_platt},
            'logloss_calselect': {'raw': ll_raw, 'isotonic': ll_iso, 'platt': ll_platt},
        },
        'reliability_calselect': {
            'raw': table_to_records(raw_reliability),
            'calibrated': table_to_records(cal_reliability),
            'inversion_check_p_over_0.6': {
                'raw': {'n': int(raw_high.sum()), 'actual_over_rate': float(raw_high_actual)},
                'calibrated': {'n': int(cal_high.sum()), 'actual_over_rate': float(cal_high_actual)},
            },
        },
        'reliability_test': {
            'raw': table_to_records(raw_reliability_test),
            'calibrated': table_to_records(cal_reliability_test),
            'brier_raw': brier_raw_test,
            'brier_calibrated': brier_cal_test,
        },
        'probability_quantiles_calselect': {'raw': raw_q, 'calibrated': cal_q},
        'auc': {
            'model_calselect': auc_model_calselect,
            'market_vigfree_calselect': auc_market_calselect,
            'model_calfit_reference': auc_model_calfit,
            'blend_calselect': auc_blend,
        },
        'blend_check': {
            'coefficients': {'model': float(blend_coef[0]), 'market': float(blend_coef[1])},
            'intercept': float(blend_intercept),
            'auc_calselect': auc_blend,
            'brier_calselect': brier_blend,
        },
        'threshold_sweep_calselect': {
            arm: [
                {k: v for k, v in r.items() if k != 'results'} for r in sweep[arm]
            ] for arm in ('both', 'under_only')
        },
        'preregistered_policies': {
            'A_both': {'threshold': policy_a_thresh, 'calselect_summary': {k: v for k, v in policy_a_calselect.items() if k != 'results'}},
            'B_under_only': {'threshold': policy_b_thresh, 'calselect_summary': {k: v for k, v in policy_b_calselect.items() if k != 'results'}},
        },
        'test_results_single_touch': test_report,
    }

    metadata_path = MODEL_DIR / 'calibration_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved metadata to: {metadata_path}")

    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
