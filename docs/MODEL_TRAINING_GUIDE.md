# NHL Goalie Saves Model — Training Guide

This document is a complete account of how the current production model came to exist: the data it's built on, the exact pipeline that produced it, every hyperparameter and feature decision, the bugs that were found and fixed along the way, and what's still fragile about reproducing it from scratch. Everything below was verified directly against the code, `git log`, and the model metadata files in this repo as of 2026-07-02 — not copied from an earlier version of this doc, which had drifted out of sync with the actual pipeline in several places (noted inline where relevant).

## Table of Contents

1. [Current Production Model](#1-current-production-model)
2. [The Story: How This Model Came To Be](#2-the-story-how-this-model-came-to-be)
3. [Data Collection](#3-data-collection)
4. [Feature Engineering — The Live Pipeline](#4-feature-engineering--the-live-pipeline)
5. [Feature Engineering — The Dead Pipeline (Read This Before You Touch `src/features/`)](#5-feature-engineering--the-dead-pipeline-read-this-before-you-touch-srcfeatures)
6. [The 114 Production Features, In Full](#6-the-114-production-features-in-full)
7. [Multi-Book Training Data](#7-multi-book-training-data)
8. [Model Training](#8-model-training)
9. [Why 114 Features and High Regularization — The Actual Experiment](#9-why-114-features-and-high-regularization--the-actual-experiment)
10. [Hyperparameter Tuning](#10-hyperparameter-tuning)
11. [Model Evaluation Methodology](#11-model-evaluation-methodology)
12. [Bugs Found and Fixed Along The Way](#12-bugs-found-and-fixed-along-the-way)
13. [Full Model Generation History](#13-full-model-generation-history)
14. [Known Gaps If You Need To Retrain From Scratch](#14-known-gaps-if-you-need-to-retrain-from-scratch)
15. [Model Deployment](#15-model-deployment)
16. [Quick Reference Commands](#16-quick-reference-commands)
17. [Appendix: Directory Structure](#17-appendix-directory-structure)

---

## 1. Current Production Model

- **Location**: `models/trained/tuned_v1_20260201_155204/`
- **Config name** (from its own metadata): `"Tuned V1 (Random #30)"` — the 30th of 40 randomly sampled hyperparameter configurations in a search, not a hand-picked config
- **Trained**: 2026-02-01 15:52:04
- **Algorithm**: XGBoost binary classifier (`binary:logistic`), saved via the `Booster` interface (not `XGBClassifier` pickle) for portability
- **Features**: 114 (confirmed by counting `classifier_feature_names.json` — 90 base + 6 line-relative + 18 engineered, see [§6](#6-the-114-production-features-in-full))
- **EV threshold**: 12% minimum edge over the market to recommend a bet
- **Referenced from**: `src/betting/predictor.py:13` and `src/betting/feature_calculator.py:18` both hardcode this exact path as the default

**Hyperparameters** (verbatim from `classifier_metadata.json`):

```json
{
  "max_depth": 6,
  "learning_rate": 0.05,
  "min_child_weight": 30,
  "gamma": 2.0,
  "reg_alpha": 20,
  "reg_lambda": 60,
  "n_estimators": 600,
  "subsample": 0.7,
  "colsample_bytree": 0.8
}
```

**Performance** (verbatim from `classifier_metadata.json`):

| Metric | Validation | Test | Combined |
|---|---|---|---|
| ROI | +27.05% | +20.45% | **+23.31%** |
| Bets placed | 191 (17.65% of lines) | 250 (23.11% of lines) | 441 |

> **DO NOT CITE +23.31% AS EVIDENCE OF EDGE (annotation added 2026-07-24).** A
> preregistered walk-forward validation of this exact recipe — same 114 features,
> same hyperparameters, same 12% EV threshold, retrained per fold and evaluated
> forward-in-time on two seasons it had never seen — returned **pooled OOS ROI
> -7.72%** over 3,258 bets, game-level 95% CI **[-13.48%, -2.16%]**, with **AUC
> below 0.5 on both folds** (no out-of-sample discrimination) and calibration
> worse than the market's own devigged line. The number in this table does not
> reproduce out-of-sample; it reverses sign. See
> [HISTORICAL_DATA_ANALYSIS.md section 10](HISTORICAL_DATA_ANALYSIS.md) for the
> authoritative read and PREREGISTRATION section 21.9 for the full result. Note
> also that this table's figure was already retired once, in
> [OFFSEASON_OPTIMIZATION_PLAN.md](OFFSEASON_OPTIMIZATION_PLAN.md) section 1.4, as
> an optimistically biased maximum over 168 draws.

This is a backtested number — see [§11](#11-model-evaluation-methodology) for exactly what "ROI" means here and what it doesn't guarantee about live performance.

---

## 2. The Story: How This Model Came To Be

This section exists because the honest history of this model is not a straight line — it went through a regression phase, a leakage bug, a false "all clear" audit that was wrong, a full pipeline rewrite, a second (subtler) leakage near-miss, a structural bug where the model was functionally ignoring the betting line, and only then arrived at the architecture described in the rest of this doc. If you're going to retrain or extend this model, the mistakes below are exactly the ones you're most likely to repeat.

Everything here is reconstructed from `git log` on this repository (435 commits total, but the model work starts 2025-12-31 — everything before that is unrelated content from the repo this project was forked from). Where a commit message is quoted, it's verbatim.

### Phase 0 — Regression, and the first leakage bug (Dec 31, 2025 – Jan 1, 2026)

The model started as an **XGBoost regressor** predicting a raw save count (`target_rmse: 5.0`, `target_mae: 3.5` in the original `config/config.yaml`), not a classifier predicting OVER/UNDER probability. The idea was: predict a number, then compare it to the betting line.

On Jan 1, commit **`a35e285 "feat: leak"`** — the commit message says it plainly — added `corsi_fenwick_features.py` and computed Corsi/Fenwick and game-outcome stats **from the current game itself**, not shifted to exclude it, and fed them straight into the model. RMSE dropped from 6.66 to 1.41 overnight, and `team_corsi_against`/`team_fenwick_against` claimed 41%/26% of feature importance — the classic signature of a leaked feature that's essentially telling the model the answer. It was partially fixed minutes later in **`8019144 "feat: better"`** (excluded the worst offenders from the trainer's feature list), but the underlying rolling-window computation for Corsi/Fenwick was *still* leaky — it wasn't properly fixed until **`8d89b56 "feat: still only at 5.3"`**, which rewrote the rolling logic to be per-team and `.shift(1)`-based.

**Lesson embedded in the current code**: this is why every rolling feature in this codebase — in both the live and dead pipelines — uses the same defensive pattern: sort chronologically, then `.shift(1)` before windowing, every single time, with no exceptions. See [§4](#4-feature-engineering--the-live-pipeline) for exactly where this appears today.

### Phase 1 — A classifier, a false "all clear," and a full rewrite (Jan 3–4, 2026)

Commit **`c8b7922 "feat: now 54%!"`** (Jan 3) introduced `classifier_trainer.py` and, alongside it, a document called `MODEL_AUDIT_REPORT.md` (since deleted — recoverable via `git show c8b7922:MODEL_AUDIT_REPORT.md`) which claimed **97.9% test accuracy** with **100% OVER precision / 100% UNDER recall**, and concluded, verbatim: *"VERDICT: ✅ MODEL IS LEGITIMATE - Performance is valid and will likely replicate in real betting scenarios."*

That conclusion was wrong. The very next commit, **`5370f06 "feat: still ~50%"`**, made the same day, shows the real picture once the feature set was reworked: **87.6% train accuracy / 0.97 AUC vs. 52.7% validation accuracy / 0.52 AUC** — essentially a coin flip on unseen data, with a massive train/validation gap that's the textbook signature of leakage or severe overfitting on a contaminated split.

The response was a full rewrite, not a patch. Commit **`c75c061 "feat: add the betting script and excel file- need to fix the prediction stuff"`** (Jan 4, 15:21) deleted `MODEL_AUDIT_REPORT.md` — confirmed via `git log --diff-filter=D -- MODEL_AUDIT_REPORT.md`, though the commit's own message is about unrelated betting-script work, not the audit report; deleting it appears to have been a side effect of a broader cleanup in the same commit, not something called out in the message.

`scripts/create_clean_features.py` was added the same day, about an hour later, as part of commit **`9c209a0 "feat: switch to categorical model (#1)"`** (Jan 4, 16:34) — a squash-merge PR whose body is a real-time log of the whole pivot (`feat: leak` → `feat: better` → `feat: still only at 5.3` → `feat: now 54%!` → `feat: still ~50%` → `feat: all the training now looks good!` → `feat: add the betting script and excel file...` → `feat: betting tracker is ready!` → `feat: clean up`, all as sub-bullets of one PR). Its docstring states its purpose outright:

> *"This script creates a SIMPLE, CLEAN feature set with NO data leakage... No complex merge operations that create `_x`/`_y` columns... Easy to verify correctness."*

`create_clean_features.py` is the actual ancestor of today's production feature pipeline — it was deleted 11 days later (§4) and has since been recovered and restored to `scripts/`. The same `9c209a0` PR also formally merged the regression→classifier pivot, deleted the interim planning docs (`CLASSIFICATION_MIGRATION_GUIDE.md`, `QUICKSTART.md`, `AUTOMATION.md`, `BETTING_LINES_REQUIREMENTS.md`), and added the first version of the Excel betting tracker.

**Lesson**: a suspiciously good accuracy number on a betting model is a bug report, not a result. The "audit" that declared the model clean was itself the thing that needed auditing.

### Phase 2 — From accuracy to money: EV-based training (Jan 11–13, 2026)

Getting the classifier working was necessary but not sufficient — a model can have great accuracy and still lose money if it's most confident on the bets with the worst odds. Commit **`8c77154 "Train on value (#2)"`** (Jan 12) is where the training loop stopped optimizing for accuracy and started optimizing for what actually matters: backtested ROI using real historical American odds. This PR added `evaluate_profitability()` (still the only live method used from `classifier_trainer.py` today — see [§11](#11-model-evaluation-methodology)) and `calculate_sample_weights()`, a market-vig-based weighting scheme (sharp lines weighted 1.5x, soft/high-vig lines 0.8x) that is **not used** by the current production model (`use_sample_weights: False` throughout).

The same PR briefly introduced market-derived features like `line_vs_recent_avg` and `market_vig` in `scripts/add_market_features.py` — and the accompanying tuning script comments `# Remove market-derived features (data leakage)` right before dropping them, meaning this near-miss was caught and self-corrected within the same PR rather than shipping. Those excluded-market-feature names are still hardcoded in three separate scripts today (see [§8](#8-model-training)) as a direct legacy of this catch.

Commit **`59cf24a "feat: switched to a better model"`** (Jan 13) shipped **Config #4398**: 90 features, single betting book, 2% EV threshold, `+1.62% combined ROI (699 bets)` (commonly quoted elsewhere, including the training script's own print statement and the README, as "+1.60%" — both are rounding the same underlying `1.6176...%`; see §13). Its docstring is explicit about why 4398 beat the competing candidate, Config #5419: *"More volume than Config #5419 @ 4% (699 vs 581 bets)."* Full metrics for both are in [§13](#13-full-model-generation-history).

### Phase 3 — Diagnosing a model that was ignoring the line (Jan 31, 2026)

By Jan 31 — after roughly two and a half weeks of Config #4398 running in production — commit **`8a5c9d5 "Create MODEL_ISSUES_AND_FIXES.md"`** (deleted since; recoverable via `git show 8a5c9d5:docs/MODEL_ISSUES_AND_FIXES.md`) documented a real structural flaw: **the model's P(over) was barely sensitive to the actual betting line.** On real data from that day, the same goalie showed nearly identical predicted probability against two *different* books' lines (e.g. Bobrovsky at 51.2% against both a 21.5 and a 20.5 line) — the model was reading the goalie's recent form and largely ignoring what number it was actually being asked to bet against.

The root cause: `betting_line` is highly correlated with the rolling-average features the model already saw (sportsbooks set lines close to recent form), and the training data at that point had exactly one row per goalie-game — so the model never had a chance to learn "same goalie, different line, different answer" in isolation. In practice this produced a systematic bias that *looked* like insight but was just mean-reversion: OVER got recommended disproportionately for goalies with low lines, UNDER for goalies with high lines, regardless of whether the specific number made sense.

The doc laid out five candidate fixes (two-stage regression, line-relative features, multi-book data augmentation, a hybrid, or a quick heuristic patch) and recommended the two-stage approach. **What actually shipped the next day was the other two options — line-relative features plus multi-book training data** — because together they attacked both halves of the root cause: multi-book data gives the model multiple (goalie, different-line) pairs to learn from, and line-relative features (`line_vs_rolling_*`, `line_z_score_*`) make the line's relationship to recent form an explicit, trainable input rather than something the model has to infer indirectly.

### Phase 4 — Multibook, engineered features, and hyperparameter tuning: three model generations in one day (Feb 1, 2026)

This is the day the model reached its current architecture. All of the following happened Feb 1, in order:

**10:04am** — the first model trained on the new multi-book data (`scripts/build_multibook_training_data.py`, producing one training row per (goalie, game, bookmaker, line) instead of one per game, plus the 6 new line-relative features) — **Multibook V1** — was evaluated at the old 2% EV threshold and got **+7.32% combined ROI (1,571 bets)**. The corresponding commit, `f64f6ea "add line adjustments (#4)"`, landed about 20 minutes later at 10:26am — the script was evidently run from an uncommitted working tree first, then committed once the result looked good.

**11:25am** — the same Multibook V1 architecture was re-evaluated at a 12% EV threshold and got **+9.16% combined ROI (715 bets)** — a meaningfully better ROI on roughly half the volume. *(Note: the previous version of this doc, and the README, both stated "Multibook V1: 12% EV, +7.32% ROI" — that conflates the two runs. The +7.32% figure belongs to the 2%-threshold run; +9.16% is the actual number for the 12%-threshold config that went on to ship. Verified directly from `models/archive/multibook_v1_20260201_100441/classifier_metadata.json` (`ev_threshold: 0.02`, `combined_roi: 7.32`) vs. `models/archive/multibook_v1_20260201_112500/classifier_metadata.json` (`ev_threshold: 0.12`, `combined_roi: 9.16`).)*

**3:26pm** — commit **`8f93fe6 "feat: huge bug fix"`** — a 304-line rewrite of `src/betting/feature_calculator.py`, the **live inference-time** feature builder (separate module from the training pipeline). This fixed three problems specific to computing features for a *live* prediction rather than from historical boxscores: the NHL game-log API has no `saves` field (it was being derived incorrectly), situation-specific stats (even-strength/power-play/short-handed) need to come from the boxscore endpoint rather than the game log (which lacks them entirely), and the rolling computation needed to be rebuilt around `nhl_fetcher.get_goalie_boxscore_stats()`. This was a production-correctness bug, not a training-data leak — but it means any live predictions made before this fix landed were built on wrong feature values.

**3:38pm** — `scripts/optimize_features.py` added 18 hand-engineered features on top of the 96-feature multibook set (interaction, volatility, momentum, and matchup-context features — full formulas in [§6](#6-the-114-production-features-in-full)) and tested 8 different configurations to decide what to keep. The winner, **"Engineered + high reg"** — all 18 engineered features plus higher regularization (`reg_alpha=20, reg_lambda=60, gamma=2.0`) — became **Optimized V1**: 114 features, **+15.24% combined ROI (451 bets)**. See [§9](#9-why-114-features-and-high-regularization--the-actual-experiment) for exactly what the other 7 configurations were and why they lost.

**3:52pm** — `scripts/tune_hyperparameters.py` ran a randomized search (40 sampled configs + 2 seeded baselines, × 4 EV thresholds = 168 evaluations) over the 114-feature set from the previous step. The winner, config **"Random #30"**, became **Tuned V1** — the model still in production today: **+23.31% combined ROI (441 bets)**. Full search space in [§10](#10-hyperparameter-tuning).

In the space of about five and a half hours, ROI on this backtest went from +9.16% to +15.24% to +23.31%, driven by (in order) better features, then better regularization.

### Phase 5 — Stable production, then two rounds of cleanup (Feb 2 – Jul 2, 2026)

From Feb 2 through mid-June, the model itself didn't change — commits in this window are entirely about the daily betting-tracker workflow (odds fetching, GitHub Actions automation, tracker bug fixes), not the training pipeline.

Two cleanup passes happened after that, both of which are directly relevant if you're trying to reproduce the pipeline from scratch (see [§14](#14-known-gaps-if-you-need-to-retrain-from-scratch)):

- **Jan 15, 2026** (`8b1f3ab "feat: removed unnecessary scripts"`) — deleted `scripts/create_clean_features.py` and `scripts/extract_historical_odds.py` once their one-time output was already sitting in `data/processed/`. This is the origin of the reproducibility gap described in §14 — the scripts that produced today's production feature schema are gone from the working tree, though still recoverable from git.
- **Jun 17, 2026** (`b706811 "feat: remove outdated scripts and files"`) — a much larger cleanup that removed the entire old regression-era pipeline (`src/models/trainer.py`, `predictor.py`, `evaluator.py`, `scripts/retrain_model.py`, `scripts/train_production_4398.py`, and others), all the intermediate `models/trained/` directories for superseded model generations (moved to `models/archive/`, not deleted — see [§13](#13-full-model-generation-history)), and stale config files (`config/feature_config.yaml`, `config/model_config.yaml`).

---

## 3. Data Collection

```bash
python scripts/collect_historical_data.py --seasons 20222023 20232024 20242025 20252026
```

This calls `DataCollectionOrchestrator` (`src/data/collectors.py`), which fetches and caches:

| Data | Source | Destination |
|---|---|---|
| Schedules | NHL API `/v1/schedule/{date}` | `data/raw/` (via `CachedNHLAPIClient`) |
| Boxscores | NHL API `/v1/gamecenter/{gameId}/boxscore` | `data/raw/boxscores/{game_id}.json` |
| Play-by-play | NHL API | `data/raw/play_by_play/{game_id}.json` (5,248 files present — only consumed by the dead pipeline, see §5) |
| Goalie game logs | NHL API `/v1/player/{playerId}/game-log/{season}/{gameType}` | via cache |

**Known discrepancy**: `config/config.yaml`'s `data.seasons` list is `["20222023", "20232024", "20242025"]` — it's missing `20252026`, even though the script's own CLI default and this doc's own commands include it. If you run the script with no `--seasons` flag, you get the CLI default (correct); if some other caller reads `config.yaml`'s value directly, it'll silently miss the current season.

---

## 4. Feature Engineering — The Live Pipeline

**This is the pipeline that actually produced the 114 production features.** Read this section, not §5, if you're trying to understand how today's model's features are built.

### Update: the missing scripts have been recovered

An earlier version of this doc reported four scripts as permanently deleted with no surviving producer for `classification_training_data.parquet`. All four have since been recovered from git history and restored to `scripts/` — the pipeline below is now fully reproducible end to end. Recovery commands, for the record:

```bash
git show 8b1f3ab~1:scripts/create_clean_features.py > scripts/create_clean_features.py
git show 8b1f3ab~1:scripts/extract_historical_odds.py > scripts/extract_historical_odds.py
git show 8b1f3ab~1:scripts/merge_betting_lines.py > scripts/merge_betting_lines.py
git show 8b1f3ab~1:scripts/add_market_features.py > scripts/add_market_features.py
```

All four were deleted in the same commit, `8b1f3ab "feat: removed unnecessary scripts"` (Jan 15, 2026), once their one-time output already existed in `data/processed/`. Two of them (`create_clean_features.py`, `extract_historical_odds.py`) were recovered first; reading their actual content revealed they don't produce `classification_training_data.parquet` at all — `create_clean_features.py` only gets you to `clean_training_data.parquet` (goalie/game rolling features, no odds), and `extract_historical_odds.py` only enriches `data/raw/betting_lines/betting_lines.json` with odds, it doesn't touch any parquet file. The two scripts that actually close the gap are `merge_betting_lines.py` (merges the two into `classification_training_data.parquet`, computing `over_hit`/`line_margin`) and `add_market_features.py` (enriches that same file in place with market-derived columns). All four file paths were checked directly against each other's inputs/outputs to confirm the chain is genuinely unbroken — see below.

### The actual, current data flow

```
data/raw/boxscores/*.json
        │
        ▼
scripts/create_clean_features.py
        │
        ▼
data/processed/clean_training_data.parquet   (113 columns — goalie/game rolling features, target = saves)
        │
        ▼
data/raw/betting_lines/cache/*.json (The-Odds-API cache)
        │
        ▼
scripts/extract_historical_odds.py   → updates data/raw/betting_lines/betting_lines.json in place with odds
        │
        ▼
scripts/merge_betting_lines.py   ← reads clean_training_data.parquet + betting_lines.json
        │  computes over_hit (saves > betting_line) and line_margin
        ▼
data/processed/classification_training_data.parquet   (129 columns)
        │
        ▼
scripts/add_market_features.py   ← reads and re-writes the same file, adding market-derived columns
        │  (line_vs_recent_avg, market_vig, fair_prob_over/under, etc. — later excluded before training, see §8)
        ▼
data/processed/classification_training_data.parquet   (enriched, same path)
        │
        ▼
scripts/build_multibook_training_data.py   (adds multi-bookmaker odds + 6 line-relative features)
        │
        ▼
data/processed/multibook_classification_training_data.parquet   (137 columns)
        │
        ▼
scripts/optimize_features.py   (adds 18 engineered features, selects the 114-feature config)
        │
        ▼
scripts/tune_hyperparameters.py   (searches hyperparameters over the 114-feature set)
        │
        ▼
models/trained/tuned_v1_20260201_155204/   ← CURRENT PRODUCTION MODEL
```

**One remaining caveat**: `extract_historical_odds.py` *updates* `betting_lines.json` rather than creating it — it expects the file to already exist with game/goalie entries (no odds yet), which it does today (`data/raw/betting_lines/betting_lines.json`, dated Jan 11, 2026). The script that originally built that base file from scratch, `scripts/fetch_all_betting_lines.py` (840 lines), was deleted in the same `8b1f3ab` commit and has **not** been recovered — it's only needed if you're extending this pipeline to a season whose `betting_lines.json` doesn't already have base entries. Recoverable the same way: `git show 8b1f3ab~1:scripts/fetch_all_betting_lines.py`.

### At inference time (live predictions)

The training-time pipeline above only matters for building `multibook_classification_training_data.parquet`. Live predictions use a **separate, parallel implementation**: `src/betting/feature_calculator.py`. It has to be separate because training-time feature engineering works from a static historical dataset, while inference has to hit live NHL API endpoints for a goalie who's about to play. The two are kept in sync by construction — `feature_calculator.py`'s `_add_engineered_features()` docstring says outright: *"Matches the exact logic used in training (scripts/optimize_features.py)"* — and every formula in [§6](#6-the-114-production-features-in-full) below has been checked against both sides.

Data leakage prevention at inference time works differently than the training-time `.shift(1)` pattern, but achieves the same guarantee: `feature_calculator.py` explicitly filters out any game matching the current prediction date before computing rolling stats —

```python
filtered_games = [g for g in recent_games if g.get('gameDate', '') != game_date]
```

— so leakage is structurally impossible (the current game's data is never in the input list) rather than corrected after the fact.

---

## 5. Feature Engineering — The Dead Pipeline (Read This Before You Touch `src/features/`)

`src/features/` contains ten Python files. **Nine of them are entirely dead code**, and the tenth (`feature_engineering.py`) is about 99% dead but exports exactly one function that's genuinely load-bearing. Verified by grepping every `.py` file in the repo for actual importers of each module — not by reading docstrings or assuming from naming.

```bash
# Who actually imports each src/features/ module, outside of the package itself?
for f in base_features rolling_features team_rolling_features rest_fatigue_features \
         shot_quality_features corsi_fenwick_features advanced_rolling_features \
         matchup_features interaction_features feature_engineering; do
  echo "=== $f ==="
  grep -rln "from features.$f import\|from \.$f import" scripts/ src/ --include="*.py" | grep -v "src/features/$f.py"
done
```

**Fully dead** (only ever imported by `feature_engineering.py`'s own `FeatureEngineeringPipeline` class, which is itself only reachable via `scripts/create_features.py` — nothing live touches any of these):
- `base_features.py`, `rolling_features.py`, `team_rolling_features.py`, `rest_fatigue_features.py` — the original base feature computation, superseded by the (deleted) `create_clean_features.py` on Jan 4 (§2, Phase 1) and never used again
- `shot_quality_features.py` — genuinely parses play-by-play data (`data/raw/play_by_play/*.json`, 5,248 files, real data that exists) for danger-zone/rebound shot quality metrics
- `corsi_fenwick_features.py` — despite talking about play-by-play in its docstring, actually approximates Corsi/Fenwick from **boxscore** fields only (`team_shots`, `opp_shots`, `team_blocked_shots`). This is the module at the center of the Jan 1 leakage bug (§2, Phase 0).
- `advanced_rolling_features.py` — rolls up the two modules above into rolling windows
- `matchup_features.py` and `interaction_features.py` — imported into `feature_engineering.py` but **never called** by its own `calculate_all_features()` method — dead even within the dead pipeline

**Mostly dead, but not entirely**: `feature_engineering.py` contains the dead `FeatureEngineeringPipeline` class and `create_training_dataset()` function — but also a small, fully self-contained function, `compute_line_relative_features()` (lines 438-471), that **is** live: `scripts/build_multibook_training_data.py` imports it directly (`from features.feature_engineering import compute_line_relative_features`) to compute the 6 line-relative features described in §6. This function only touches `betting_line` and `saves_rolling_*` columns already present in its input DataFrame — it has no dependency on any of the other nine files, dead or otherwise.

**How to confirm the dead pipeline's output is truly unused:**

```bash
grep -rn "training_data.parquet" scripts/ src/ --include="*.py"
# -> only scripts/create_features.py (produces it) and feature_engineering.py (defines the function that builds it)
```

`scripts/create_features.py` is the only entry point into the dead orchestrator (`python scripts/create_features.py` → `create_training_dataset()` → `data/processed/training_data.parquet`, a 357-column file with ~140 danger/xG/Corsi/Fenwick columns). That file is never consumed by anything downstream — not `build_multibook_training_data.py`, not `optimize_features.py`, not `tune_hyperparameters.py`. It's a sibling of the real pipeline's `classification_training_data.parquet`, not an ancestor of it, despite what an earlier version of this doc implied.

**If you're extending this model and considering adding shot-quality or Corsi/Fenwick features**: the code to compute them already exists and evidently worked well enough to produce a 357-column dataset at some point. The reason they're not in the production 114 isn't that they were tried and rejected — it's that the team moved to `create_clean_features.py` specifically *because* the original pipeline (this one) was where the Jan 1 leakage bug happened (see §2, Phase 0), and the replacement pipeline was deliberately simpler and never grew this functionality back in. There's no backtest evidence either way on whether shot quality would help the current architecture — it's an open, unexplored direction, not a dead end.

---

## 6. The 114 Production Features, In Full

Every feature in `models/trained/tuned_v1_20260201_155204/classifier_feature_names.json`, with its formula and where it's computed on both the training side and the live inference side (`src/betting/feature_calculator.py`).

### Context (1 feature)

- `is_home` — 1 if the goalie's team is home, 0 if away. Passed directly by the caller on both sides.

### Goalie basic rolling stats (24 features)

Stats: `saves`, `shots_against`, `goals_against`, `save_percentage` × windows `{3, 5, 10}` × `{mean, std}` = 4 × 3 × 2 = 24.

- Training: `df.groupby('goalie_id')[stat].transform(lambda x: x.rolling(window=w, min_periods=1).mean().shift(1))` (and `.std().shift(1)` for the std variants)
- Inference (`feature_calculator.py:82-108`): the NHL game-log API has no `saves` field, so it's derived as `shots_against - goals_against` per game; `_compute_rolling()` takes the first N entries of a most-recent-first game list and computes `np.mean`/`np.std` directly (no shift needed — see §4's inference-time leakage note)

### Goalie situation-specific rolling stats (54 features)

Stats: `even_strength_{saves, shots_against, goals_against}`, `power_play_{...}`, `short_handed_{...}` (9 stats) × windows `{3, 5, 10}` × `{mean, std}` = 9 × 3 × 2 = 54.

These **cannot** come from the game-log API — it lacks situational splits entirely. Both training and inference pull them from boxscores, parsing fields like `evenStrengthShotsAgainst` which arrive as `"saves/shots"` strings (e.g. `"18/20"`):

```python
def _parse_situation_stat(self, stat_str, stat_type):
    if '/' not in str(stat_str):
        return 0
    saves_str, shots_str = str(stat_str).split('/')
    return int(saves_str) if stat_type == 'saves' else int(shots_str)
```

This exact function exists independently in both `nhl_fetcher.py` (inference) and the training-side boxscore parser, confirmed byte-identical logic.

### Team/opponent rolling stats (8 features)

`opp_goals_rolling_{5,10}`, `opp_shots_rolling_{5,10}` (the opponent's own offensive output, from their own recent games), `team_goals_against_rolling_{5,10}`, `team_shots_against_rolling_{5,10}` (the goalie's own team's defense, from the goalie's own recent games) — windows `{5, 10}` only, no `{3}` window for team-level stats.

### Rest/fatigue (2 features)

- `goalie_days_rest` = `(current_game_date - last_game_date).days`
- `goalie_is_back_to_back` = `1 if goalie_days_rest == 1 else 0`

### Betting line (1 feature)

- `betting_line` — passed directly. Deliberately **not** excluded from training (unlike every other market-derived value) because it's known before the game, unlike the actual outcome.

### Line-relative (6 features) — added Feb 1 to fix the line-insensitivity bug (§2, Phase 3)

Windows `{3, 5, 10}`:
- `line_vs_rolling_{w}` = `betting_line - saves_rolling_{w}`
- `line_z_score_{w}` = `(betting_line - saves_rolling_{w}) / saves_rolling_std_{w}` if `std > 0.01` else `0`

### Engineered features (18 total) — added same day, by `optimize_features.py`

Defined identically (confirmed byte-for-byte) in `scripts/optimize_features.py`'s `add_all_engineered_features()`, `scripts/tune_hyperparameters.py`'s copy of the same function, and `feature_calculator.py`'s `_add_engineered_features()`.

**Interaction (7):**
- `save_efficiency_{3,5,10}` = `saves_rolling_w / max(shots_against_rolling_w, 1)`
- `es_saves_proportion_{5,10}` = `even_strength_saves_rolling_w / max(saves_rolling_w, 1)`
- `opp_vs_team_shots_{5,10}` = `opp_shots_rolling_w - team_shots_against_rolling_w`

**Volatility (4):**
- `saves_cv_{5,10}` = `saves_rolling_std_w / max(saves_rolling_w, 1)`
- `volatility_vs_line_{5,10}` = `saves_rolling_std_w / max(betting_line, 1)`

**Momentum (4):**
- `saves_momentum` = `saves_rolling_3 - saves_rolling_10`
- `shots_against_momentum` = `shots_against_rolling_3 - shots_against_rolling_10`
- `goals_against_momentum` = `goals_against_rolling_3 - goals_against_rolling_10`
- `save_pct_momentum` = `save_percentage_rolling_3 - save_percentage_rolling_10`

**Matchup context (3):**
- `expected_workload_diff` = `opp_shots_rolling_5 - shots_against_rolling_5`
- `line_vs_opp_implied_saves` = `betting_line - (opp_shots_rolling_5 - opp_goals_rolling_5)`
- `rest_x_performance` = `min(goalie_days_rest, 7) * saves_rolling_5`

7 + 4 + 4 + 3 = 18. **1 + 24 + 54 + 8 + 2 + 1 + 6 + 18 = 114.**

(Training code uses pandas' `.clip(lower=1)`; inference code uses Python's `max(x, 1.0)` — numerically identical, just different idioms on each side.)

---

## 7. Multi-Book Training Data

```bash
python scripts/build_multibook_training_data.py
```

Reads `data/processed/classification_training_data.parquet` (see the reproducibility gap in §14) plus cached historical odds from The-Odds-API (`data/raw/betting_lines/cache/`, filtered to `market.key == 'player_total_saves'`). Matches odds to base feature rows via `game_date + team_abbrev` (tried against both home and away, with a +/-1-day tolerance because the odds cache's `commence_time` is UTC while base `game_date` is local), gated by **two mandatory checks** (both added 2026-07-07 -- see `OFFSEASON_OPTIMIZATION_PLAN.md` section 2 for the corruption bug this fixed): the goalie's last name must match (accent-insensitive, and a missing name on either side rejects the match rather than silently passing), and the base row's `opponent_team` must equal the odds event's other team. Records for `prizepicks`/`manual`/`unknown` books are dropped before matching (placeholder or unreliable odds), and the final output is deduplicated on `(game_id, goalie_id, book_key, betting_line)`. Produces **one row per (goalie, game, bookmaker, line) combination** rather than one row per game — the whole point being to give the model multiple line values to learn from per goalie-game, which is what actually fixed the line-insensitivity bug from §2 Phase 3 (combined with the 6 line-relative features, computed via `compute_line_relative_features()` — imported from `src/features/feature_engineering.py`, the one live export of an otherwise dead module, see §5).

Output: `data/processed/multibook_classification_training_data.parquet` (137 columns).

> **Update (2026-07-24)**: this parquet has since been extended to **three seasons**. The owned 2023-24 bet-time saves lines were folded in as a new season (13,192 -> 20,799 rows; `classification_training_data.parquet` 4,755 -> 6,714), strictly additively (existing 2024-25/2025-26 rows byte-identical, backup in `data/processed/backup_20260724/`). Details and method (append-only, because the multibook build is no longer reproducible from current code): `CURRENT_HISTORICAL_DATA.md` section 4.4. **The production model in this guide was NOT retrained** — it remains the 2-season `tuned_v1_20260201_155204`. A walk-forward evaluation of the recipe on the new three-season data was preregistered as `PREREGISTRATION_NO_CREDIT_ABLATIONS.md` section 21 and **has now been run (2026-07-24): it FAILED** — pooled out-of-sample ROI -7.72% over 3,258 bets, 95% CI [-13.48%, -2.16%], AUC below 0.5 on both unseen seasons (result: section 21.9; synthesis: `HISTORICAL_DATA_ANALYSIS.md` section 10). Note 2023-24 is sportsbook-only (no DFS books). If you retrain on the three-season data, re-read §14 (reproducibility gaps) and §8 (the frozen split logic) first.

| Bookmaker key | Name |
|---|---|
| `draftkings` | DraftKings |
| `fanduel` | FanDuel |
| `betmgm` | BetMGM |
| `betonlineag` | BetOnline.ag |
| `williamhill_us` | Caesars |
| `pinnacle` | Pinnacle |

---

## 8. Model Training

The current production model was **not** trained by `scripts/train_production_multibook.py`, despite that script's name and its README billing as "Step 3: Train Multi-Book Model." That script:
- hardcodes the *old* Config #4398 hyperparameters (`max_depth=4, learning_rate=0.02, ...`, not the production model's `max_depth=6, learning_rate=0.05, ...`)
- only builds the 96-feature set (base + line-relative, no engineered features)
- always saves to `models/trained/multibook_v1_{timestamp}/` — it can never produce anything named `optimized_v1_*` or `tuned_v1_*`

It's exactly what produced the **Multibook V1** archived model (§13), which was superseded same-day by `optimize_features.py` and then `tune_hyperparameters.py`. **The current production model was saved directly by `tune_hyperparameters.py`'s own save logic** — there is no separate "final training" step; the hyperparameter search script *is* the training script for the winning config.

### Data split — chronological 60/20/20

Identical inline logic, copy-pasted across `optimize_features.py`, `tune_hyperparameters.py`, and `train_production_multibook.py` (not centralized in a shared function):

```python
df = df.sort_values('game_date')
n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train_idx = np.arange(0, train_end)      # earliest 60%
val_idx = np.arange(train_end, val_end)  # next 20%
test_idx = np.arange(val_end, n)         # most recent 20%
```

`train_production_multibook.py` additionally asserts no date overlap between splits (`train_dates.max() <= val_dates.min()`, etc.) as a runtime sanity check.

**Inconsistency worth knowing about**: `src/models/classifier_trainer.py`'s own `split_data()` method defaults to `test_size=0.2, val_size=0.15` — i.e. 65/15/20, not 60/20/20 — but this method is **never called** by any of the three real training scripts, all of which reimplement the split inline. `config/config.yaml`'s `model.test_size: 0.15` / `validation_size: 0.15` don't match either the real 60/20/20 split or this unused method's 65/15/20 default. If you're touching `config.yaml`'s `model` section, know that nothing currently reads it — it appears to be a leftover from the regression era.

### Excluded columns (data leakage prevention)

Byte-identical list, hardcoded independently in all three live training scripts (variable named `excluded_cols` in `train_production_multibook.py`, `EXCLUDED_BASE` in `optimize_features.py`, `EXCLUDED` in `tune_hyperparameters.py` — same content, different names):

```python
[
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
```

The `saves`/`shots_against`/`goals_against`/etc. entries here are the **current game's actual results** — obviously not knowable before the game, excluded to prevent the model from just reading the answer off the target row. (The *rolling* versions of these same stats, e.g. `saves_rolling_5`, are fine and are the whole point of the model — only the unrolled, current-game values are excluded.)

**Before** this list is applied, a second set of "market-derived" columns is dropped — these exist in the multibook parquet (added by the Jan 12 PR discussed in §2 Phase 2) but aren't available at inference time, since they're derived from the *odds themselves* rather than from pre-game knowledge:

```python
['line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
 'market_vig', 'impl_prob_over', 'impl_prob_under',
 'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
 'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low']
```

**A second, different, unused exclusion list exists**: `src/models/classifier_trainer.py`'s `prepare_features()` method has a materially broader list (also excludes `is_starter`, `high_danger_*`/`mid_danger_*`/`low_danger_*`, `total_xg_against`, `team_corsi_*`, `team_fenwick_*`, `is_win`/`is_loss`) — this is legacy code from the dead Corsi/Fenwick-aware pipeline (§5) and isn't called by anything live.

### `ClassifierTrainer`'s actual role today

Of everything in `src/models/classifier_trainer.py`, only **`evaluate_profitability()`** is live — used identically by all three training scripts to backtest ROI (see §11). `prepare_features()`, `split_data()`, `train()`, `_recalculate_rolling_features()`, and `calculate_sample_weights()` are all unused by the current pipeline; `calculate_sample_weights()` in particular is explicitly disabled everywhere (`use_sample_weights: False`).

---

## 9. Why 114 Features and High Regularization — The Actual Experiment

`scripts/optimize_features.py` didn't arrive at 114 features by importance-based pruning — it tested 8 explicit configurations against each other and kept whichever won, all evaluated with the same `evaluate_profitability()` backtest at a fixed 12% EV threshold:

| # | Config | What it changed | Result |
|---|---|---|---|
| 1 | Baseline (96 features) | The multibook feature set as-is | Starting point |
| 2 | Baseline + all engineered | +18 engineered features → 114 total | Strong improvement |
| 3 | Baseline + interactions only | 7 of the 18 engineered features | Partial improvement |
| 4 | Baseline + trends only | 4 of the 18 engineered features | Partial improvement |
| 5 | Top 50 by importance | Pruned baseline down to its 50 highest-gain features | **Underperformed** |
| 6 | Top 30 by importance | Pruned further to 30 | **Underperformed further** |
| 7 | All engineered + deeper trees | `max_depth=5, min_child_weight=20, n_estimators=1000` — more model capacity | Improvement, but not the best |
| 8 | **All engineered + high regularization** | `reg_alpha=20, reg_lambda=60, gamma=2.0, n_estimators=1000` — same 114 features as #2, but heavily regularized | **Winner** — became Optimized V1, +15.24% combined ROI |

The takeaway that shaped the rest of the pipeline: **feature pruning hurt** (configs 5 and 6 both underperformed the full feature set), and **regularization beat raw capacity** (config 8's heavy regularization beat config 7's deeper trees, using the same feature set). This is why the subsequent hyperparameter search (§10) explored a search space skewed toward higher `reg_alpha`/`reg_lambda`/`gamma` values rather than deeper trees, and why the winning production config (`reg_alpha=20, reg_lambda=60, gamma=2.0`) looks the way it does — it's a direct descendant of config 8's finding, further tuned.

---

## 10. Hyperparameter Tuning

```bash
python scripts/tune_hyperparameters.py
```

Takes the 114-feature dataset as a given (no feature re-selection happens here) and searches:

```python
param_grid = {
    'max_depth':        [3, 4, 5, 6],
    'learning_rate':    [0.01, 0.02, 0.05],
    'min_child_weight': [10, 15, 20, 30],
    'gamma':            [0.5, 1.0, 2.0],
    'reg_alpha':        [5, 10, 20],
    'reg_lambda':       [20, 40, 60],
    'n_estimators':     [600, 800, 1200],
    'subsample':        [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
}
```

Full grid = 15,552 combinations. Actual search: **40 randomly sampled configs** (`np.random.seed(42)`) plus **2 hand-seeded configs** (the winning "Optimized V1" config, and the original Config #4398 baseline, for comparison) = 42 configs, each evaluated at **4 EV thresholds** (`[0.08, 0.10, 0.12, 0.15]`) = **168 total evaluations**.

The current production model is `"Random #30"` — the 30th randomly-sampled configuration, not one of the 2 seeded ones — at `ev_threshold=0.12`.

**Selection filter**: results are filtered to configs whose **test-set bet rate falls between 15% and 35%**, then ranked by combined ROI, to avoid picking a config that only looks good because it bet on 3 games all season. (There's a minor, harmless inconsistency in this script worth knowing about if you're reading its output: the docstring and one print statement describe the target as "20-30%", an intermediate per-EV-threshold selection step at line 283 actually uses a 20-35% cutoff, and the final ranking at line 297 uses 15-35% — three slightly different numbers for what's conceptually one target. The final selection — the one that actually determined the production model — uses 15-35%, which is also what this doc states throughout.)

**Doc correction**: an earlier version of this table omitted `subsample` and `colsample_bytree` as search dimensions — both are real, both varied `{0.7, 0.8}` in the search above.

---

## 11. Model Evaluation Methodology

The single source of truth for "is this model good" in this codebase is `ClassifierTrainer.evaluate_profitability()`. It does **not** report accuracy, AUC, or log-loss as the primary metric — it backtests actual betting decisions:

```
For each row in the evaluation set:
  1. prob_over = model.predict(X)
  2. Calculate EV for both OVER and UNDER using the row's real historical American odds
  3. If EV >= threshold: place a hypothetical bet on that side (whichever side clears
     the threshold; if both do, take the higher-EV side)
  4. Track profit/loss at the actual historical odds (not a flat -110 assumption)
  5. ROI = total_profit / total_units_wagered * 100
```

This is why the "false audit" from §2 Phase 1 was so misleading — 97.9% classification accuracy tells you almost nothing about whether a betting strategy built on that model would make money, especially when the accuracy number itself turned out to be a leakage artifact rather than real signal.

**What this backtest does not tell you**: whether historical odds availability/liquidity would hold up for real-time betting, whether the specific bookmakers in the training data still offer these markets the same way, or how the model performs on a book/line combination it's never seen. It's a rigorous backtest on historical data, not a live-trading guarantee.

---

## 12. Bugs Found and Fixed Along The Way

A consolidated list, cross-referenced to §2 for full context:

| Bug | Found | Fixed | Nature |
|---|---|---|---|
| Corsi/Fenwick computed from current (unshifted) game | Jan 1, `a35e285` | Same day (`8019144`), fully fixed `8d89b56` | Classic same-game data leakage; RMSE artificially dropped 6.66→1.41 |
| "MODEL_AUDIT_REPORT.md" falsely declared the model leak-free at 97.9% accuracy | Jan 3, `c8b7922` | Jan 3-4, full pipeline rewrite (`c75c061`, `9c209a0`) | The audit itself was wrong — real val accuracy was 52.7% vs 87.6% train |
| Market-derived features (`market_vig`, etc.) briefly included as training inputs | Jan 12, `8c77154` | Same PR, before shipping | Caught and dropped within the same commit — never reached production |
| Live inference `feature_calculator.py` used wrong API fields (`saves`, situation stats) | — (present since inception) | Feb 1, `8f93fe6 "huge bug fix"` | Production-correctness bug, not training leakage — affected live predictions, not backtests |
| Model's P(over) was nearly insensitive to the actual betting line | Jan 31, `8a5c9d5` (`MODEL_ISSUES_AND_FIXES.md`) | Feb 1 (multibook data + line-relative features) | Structural: single-line-per-game training data meant the model never learned line sensitivity in isolation |

---

## 13. Full Model Generation History

Every `classifier_metadata.json` in the repo, verified directly (not from a summary table):

| Model | Location | Features | EV threshold | Hyperparameters (depth/lr/mcw/γ/α/λ/n_est/sub/col) | Val ROI (bets) | Test ROI (bets) | Combined ROI (bets) |
|---|---|---|---|---|---|---|---|
| Config #4398 | `models/archive/config_4398_ev2pct_20260115_103430/` | 90 | 2% | 4 / 0.02 / 15 / 1.0 / 10 / 40 / 800 / 0.8 / 0.8 | +2.54% (363) | +0.62% (336) | +1.62% (699) |
| Multibook V1 (2% run) | `models/archive/multibook_v1_20260201_100441/` | 96 | 2% | 4 / 0.02 / 15 / 1.0 / 10 / 40 / 800 / 0.8 / 0.8 | +3.13% (764) | +11.29% (807) | +7.32% (1,571) |
| Multibook V1 (12% run) | `models/archive/multibook_v1_20260201_112500/` | 96 | 12% | 4 / 0.02 / 15 / 1.0 / 10 / 40 / 800 / 0.8 / 0.8 | +8.00% (313) | +10.07% (402) | +9.16% (715) |
| Multibook V1 (dup rerun) | `models/archive/multibook_v1_20260201_152947/` | 96 | 12% | identical to above | +8.00% (313) | +10.07% (402) | +9.16% (715) |
| Optimized V1 | `models/archive/optimized_v1_20260201_153838/` | 114 | 12% | 4 / 0.02 / 15 / 2.0 / 20 / 60 / 1000 / 0.8 / 0.8 | +21.01% (199) | +10.68% (252) | +15.24% (451) |
| **Tuned V1 (PRODUCTION)** | `models/trained/tuned_v1_20260201_155204/` | **114** | **12%** | **6 / 0.05 / 30 / 2.0 / 20 / 60 / 600 / 0.7 / 0.8** | **+27.05% (191)** | **+20.45% (250)** | **+23.31% (441)** [^wf] |

[^wf]: **2026-07-24:** this recipe's walk-forward out-of-sample ROI is **-7.72%** (3,258 bets, game-level 95% CI [-13.48%, -2.16%], AUC < 0.5 on both unseen seasons). Every backtest ROI in this table is a same-methodology in-sample-selection number and none of them should be read as evidence of a tradable edge. See [HISTORICAL_DATA_ANALYSIS.md section 10](HISTORICAL_DATA_ANALYSIS.md).

Note the README previously stated Multibook V1 as "+7.32% ROI" at "12% EV" — that conflates the two rows above; +7.32% belongs to the 2% EV run, and +9.16% is the correct figure for the 12%-threshold config that fed into the rest of the pipeline. Config #4398's combined ROI is precisely 1.6176...%, which is where the commonly-quoted "+1.60%" figure (also this script's own rounding) comes from.

Old, superseded model directories (config_4398, both multibook_v1 runs, optimized_v1) were moved to `models/archive/` (gitignored) during a repo cleanup — they're kept for reference but are not loaded by any live code. Only `models/trained/tuned_v1_20260201_155204/` is referenced by `predictor.py`/`feature_calculator.py`.

---

## 14. Known Gaps If You Need To Retrain From Scratch

As of this revision, **the pipeline is fully reproducible end to end** — all four previously-deleted scripts (`create_clean_features.py`, `extract_historical_odds.py`, `merge_betting_lines.py`, `add_market_features.py`) have been recovered from git history and restored to `scripts/`. See §4 for the full data flow.

1. `scripts/collect_historical_data.py` → raw data. **Works.**
2. `scripts/create_features.py` → `training_data.parquet`. **Works, but this is the dead pipeline (§5) — its output is not what you want.** Don't run this one.
3. `scripts/create_clean_features.py` → `scripts/extract_historical_odds.py` → `scripts/merge_betting_lines.py` → `scripts/add_market_features.py`. **Recovered, works** — see §4 for exact inputs/outputs at each step.
4. `scripts/build_multibook_training_data.py` onward — **works** as documented in §7-§10.

**One remaining gap**: `extract_historical_odds.py` updates `data/raw/betting_lines/betting_lines.json` rather than creating it from scratch — it expects base game/goalie entries to already exist. The script that originally built that base file, `scripts/fetch_all_betting_lines.py`, is still deleted (same commit, `8b1f3ab`) and hasn't been recovered. This only matters if you're extending the pipeline to a season whose `betting_lines.json` doesn't already have entries — recover it the same way:
```bash
git show 8b1f3ab~1:scripts/fetch_all_betting_lines.py > scripts/fetch_all_betting_lines.py
```
Read it before running — it's 840 lines and may need adapting to any API changes since Jan 2026.

If you're regenerating from truly nothing (deleted `data/processed/` and `data/raw/betting_lines/`), also expect to adapt seasons and possibly API response shapes in the recovered scripts — they were written and run once in January 2026 and haven't been exercised since.

Also present in `data/processed/` but not part of any documented pipeline (kept for reference, not required for anything): `classification_training_data_with_opp_line.parquet` (stale artifact — unclear provenance, no producer script found), `classification_data_summary.json` (produced by `merge_betting_lines.py`, informational only), `clean_features_metadata.json` (produced by `create_clean_features.py`, informational only).

---

## 15. Model Deployment

### Artifacts

```
models/trained/tuned_v1_20260201_155204/
├── classifier_model.json          # XGBoost Booster JSON
├── classifier_feature_names.json  # 114 feature names, exact training order
└── classifier_metadata.json       # hyperparameters + performance (quoted in full in §1)
```

### Updating the deployed model

If you train a new candidate that beats production on the same backtest, update the hardcoded path in exactly two places:

- `src/betting/predictor.py` — `BettingPredictor.__init__`'s default `model_path`/`feature_order_path`
- `src/betting/feature_calculator.py` — `BettingFeatureCalculator.__init__`'s `feature_file` path

There's no config-driven model selection — both are literal hardcoded default arguments.

### Train/serve parity

The live feature calculator (`feature_calculator.py`) has to reproduce every formula in §6 exactly, computed from live API data instead of a static parquet. The specific gotchas that have bitten this before (§2, §12):
1. The game-log API has no `saves` field — derive as `shots_against - goals_against`.
2. Situation-specific stats aren't in the game log at all — must come from boxscores, parsed as `"saves/shots"` strings.
3. Team/opponent stats come from boxscores (team defense) and the opponent's own schedule + boxscores (opponent offense) — two different API call patterns.
4. Boxscores are cached in-memory per prediction run (`_boxscore_cache` in `nhl_fetcher.py`) to avoid redundant calls across multiple goalies sharing a game.

---

## 16. Quick Reference Commands

### Full pipeline from scratch

```bash
python scripts/collect_historical_data.py --seasons 20242025 20252026
python scripts/create_clean_features.py         # -> clean_training_data.parquet
# ensure data/raw/betting_lines/betting_lines.json has base entries for your seasons
# (recover scripts/fetch_all_betting_lines.py per §14 if it doesn't)
python scripts/extract_historical_odds.py       # enriches betting_lines.json with odds
python scripts/merge_betting_lines.py           # -> classification_training_data.parquet
python scripts/add_market_features.py           # enriches classification_training_data.parquet in place
python scripts/build_multibook_training_data.py # -> multibook_classification_training_data.parquet
python scripts/optimize_features.py       # optional — only needed if re-deciding the feature set
python scripts/tune_hyperparameters.py    # trains AND saves the final model
# then update the two hardcoded paths in §15
```

### Incremental retraining with the existing feature set

```bash
python scripts/collect_historical_data.py --seasons 20252026  # new season's data only
python scripts/create_clean_features.py
python scripts/extract_historical_odds.py
python scripts/merge_betting_lines.py
python scripts/add_market_features.py
python scripts/build_multibook_training_data.py
python scripts/tune_hyperparameters.py    # retune and save
# compare new val/test ROI against §13's table before deploying
```

### Daily operations (unrelated to training — see README for the full daily workflow)

```bash
python scripts/fetch_and_predict.py --verbose
python scripts/update_betting_results.py
python scripts/betting_dashboard.py
```

---

## 17. Appendix: Directory Structure

```
saves-model-v3/
├── config/
│   └── config.yaml                 # has known stale values, see §3 and §8
├── data/
│   ├── raw/
│   │   ├── boxscores/               # used by both pipelines
│   │   ├── play_by_play/            # 5,248 files, only used by the dead pipeline (§5)
│   │   └── betting_lines/cache/     # The-Odds-API historical odds cache
│   ├── processed/
│   │   ├── training_data.parquet                          # DEAD PIPELINE output — not used downstream
│   │   ├── clean_training_data.parquet                     # live pipeline, step 1 output
│   │   ├── classification_training_data.parquet            # live pipeline, step 2 output — NO CURRENT PRODUCER (§14)
│   │   ├── classification_training_data_with_opp_line.parquet  # stale artifact, unused
│   │   ├── multibook_classification_training_data.parquet  # live pipeline, current training data
│   │   ├── classification_data_summary.json                 # stale artifact
│   │   └── clean_features_metadata.json                     # stale artifact
│   └── cache/
│       └── api_cache.db             # NHL API response cache
├── models/
│   ├── trained/
│   │   └── tuned_v1_20260201_155204/    # ACTIVE PRODUCTION MODEL
│   └── archive/                          # superseded model generations, gitignored, kept for reference
│       ├── config_4398_ev2pct_20260115_103430/
│       ├── multibook_v1_20260201_100441/
│       ├── multibook_v1_20260201_112500/
│       ├── multibook_v1_20260201_152947/
│       └── optimized_v1_20260201_153838/
├── scripts/
│   ├── collect_historical_data.py   # LIVE — data collection
│   ├── create_features.py           # DEAD PIPELINE entry point (§5) — do not use for production features
│   ├── create_clean_features.py     # RECOVERED (§4/§14) — boxscores -> clean_training_data.parquet
│   ├── extract_historical_odds.py   # RECOVERED (§4/§14) — enriches betting_lines.json with odds
│   ├── merge_betting_lines.py       # RECOVERED (§4/§14) — -> classification_training_data.parquet
│   ├── add_market_features.py       # RECOVERED (§4/§14) — enriches classification_training_data.parquet
│   ├── build_multibook_training_data.py  # LIVE
│   ├── optimize_features.py         # LIVE — feature selection experiment (§9)
│   ├── tune_hyperparameters.py      # LIVE — the script that actually produced the current model
│   ├── train_production_multibook.py # SUPERSEDED — produced Multibook V1, not current production
│   ├── fetch_and_predict.py         # daily operations, not training
│   ├── record_bet.py                # daily operations, not training
│   ├── update_betting_results.py    # daily operations, not training
│   └── betting_dashboard.py         # daily operations, not training
├── src/
│   ├── betting/
│   │   ├── predictor.py             # loads the model, hardcodes its path (§15)
│   │   ├── feature_calculator.py    # LIVE inference-time feature builder — parallel to training pipeline
│   │   ├── nhl_fetcher.py           # NHL API + boxscore caching for live predictions
│   │   ├── db_manager.py            # betting tracker database (unrelated to model training)
│   │   └── excel_export.py          # betting tracker Excel snapshot (unrelated to model training)
│   ├── data/
│   │   └── api_client.py            # NHL API client
│   ├── features/
│   │   ├── feature_engineering.py   # 99% DEAD orchestrator, but exports the LIVE compute_line_relative_features() (§5)
│   │   ├── shot_quality_features.py # DEAD (§5) — but functional, real pbp-parsing code
│   │   ├── corsi_fenwick_features.py # DEAD (§5) — site of the Jan 1 leakage bug (§2 Phase 0)
│   │   ├── advanced_rolling_features.py # DEAD (§5)
│   │   ├── matchup_features.py      # DEAD, unreferenced even by the dead pipeline (§5)
│   │   ├── interaction_features.py  # DEAD, unreferenced even by the dead pipeline (§5)
│   │   ├── rolling_features.py      # DEAD (§5) — superseded by the deleted create_clean_features.py
│   │   ├── team_rolling_features.py # DEAD (§5)
│   │   └── rest_fatigue_features.py # DEAD (§5)
│   └── models/
│       └── classifier_trainer.py    # only evaluate_profitability() is live (§8)
└── docs/
    └── MODEL_TRAINING_GUIDE.md      # this file
```
