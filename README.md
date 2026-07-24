# NHL Goalie Saves Prediction Model

An XGBoost classifier that predicts NHL goalie saves over/under betting lines. Automatically fetches lines from multiple sportsbooks and calculates expected value (EV) for betting recommendations.

## Features

- **Multi-book line fetching** from Underdog Fantasy and BetOnline (via The-Odds-API)
- **114 predictive features** including rolling averages, situation-specific stats, team/opponent stats, and engineered features
- **EV-based recommendations** with 12% minimum threshold
- **Boxscore-powered inference** fetching real situation-specific stats from NHL API
- **SQLite-backed betting tracker** (`data/betting.db`) as the single source of truth, with a read-only `betting_tracker.xlsx` snapshot regenerated after every write
- **Line-snapshot and closing-line-value (CLV) tracking** -- every fetched line is snapshotted with a UTC timestamp, tickets record real stake/payout/reason economics, and CLV is computed per leg plus a rec-level shadow run of every recommendation
- **Phone-first GitHub Actions workflow** for fetching predictions, recording bets/tickets, and updating results (with CLV) without needing a PC

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize Betting Tracker

```bash
python scripts/init_betting_tracker.py
```

### 3. Fetch Lines and Generate Predictions

```bash
python scripts/fetch_and_predict.py --verbose
```

This will:
1. Fetch today's NHL schedule
2. Fetch goalie saves lines from Underdog and BetOnline
3. Match lines to games and look up goalie IDs
4. Calculate 114 features per goalie-line (including boxscore data)
5. Generate predictions with EV calculations
6. Display all lines sorted by EV

Example output:
```
ALL LINES FOR 2026-02-01:
----------------------------------------------------------------------
  Hill         (VGK) @ Underdog   | Line: 23.5  (O:+101 /U:-124 ) | Pred: 23.6  | P(Over): 52.1%  | NO BET | EV: +2.3%
  Vasilevskiy  (TBL) @ BetOnline  | Line: 20.5  (O:-135 /U:+104 ) | Pred: 20.5  | P(Over): 49.3%  | NO BET | EV: +1.7%
  Swayman      (BOS) @ Underdog   | Line: 25.5  (O:+100 /U:-122 ) | Pred: 25.4  | P(Over): 47.7%  | NO BET | EV: -2.3%
```

## Scripts

| Script | Description |
|--------|-------------|
| `fetch_and_predict.py` | Main script - fetches lines from Underdog/BetOnline and generates predictions; also snapshots every matched line to `line_snapshots` for CLV |
| `record_bet.py` | Record a bet you placed, independent of the model's recommendation |
| `record_ticket.py` | Record a ticket (1-3 legs) with full ticket economics -- stake, payout, required reason code |
| `update_betting_results.py` | Update actual saves and P/L after games complete |
| `compute_closing_clv.py` | Compute closing lines/CLV for ticket legs, settle completed tickets, summarize shadow-run CLV |
| `clv_report.py` | Print the CLV report -- per-leg and aggregate CLV, straight vs parlay, by reason_code, plus rec-level shadow CLV |
| `betting_dashboard.py` | Display performance metrics and refresh the Excel snapshot |
| `init_betting_tracker.py` | Create a new betting database and its Excel snapshot |
| `add_tracking_tables.py` | Idempotent migration: add `line_snapshots`/`tickets`/`ticket_legs` and `bets.model_version` -- run once before the season's first fetch |
| `migrate_to_sqlite.py` | One-off: fold an existing xlsx/CSV history into `data/betting.db` |
| `optimize_features.py` | Test feature engineering configurations |
| `tune_hyperparameters.py` | Hyperparameter tuning with randomized search |
| `build_multibook_training_data.py` | Build training data with multiple bookmaker lines per game |
| `collect_historical_data.py` | Pull raw historical goalie game logs from the NHL API |
| `extract_historical_odds.py` | Extract historical odds from cached Odds-API responses (`data/raw/betting_lines/`) |
| `merge_betting_lines.py` | Merge betting lines with training data to build the classification dataset (implied-probability odds averaging -- see [docs/HISTORICAL_DATA_ANALYSIS.md](docs/HISTORICAL_DATA_ANALYSIS.md)) |
| `create_clean_features.py` | Clean feature engineering pipeline for the classification model |
| `add_market_features.py` | Add market-disagreement features to the classification training data |

Retired one-off scripts (superseded, kept for reference) live in `scripts/archive/`, e.g. `train_production_multibook.py`.

## Usage

### Daily Workflow

```bash
# One-time, before the season's first fetch: add the ticket/CLV tracking tables
python scripts/add_tracking_tables.py

# Fetch lines and predictions (run multiple times as lines update) --
# also snapshots every matched line to line_snapshots for later CLV
python scripts/fetch_and_predict.py --verbose

# Right after placing a single line against the model's recommendation
python scripts/record_bet.py --goalie_name Shesterkin --book Underdog \
    --bet_selection OVER --bet_amount 2

# Or, to record full ticket economics (stake, payout, required reason) for
# a straight bet or a 2-3 leg parlay:
python scripts/record_ticket.py --book Underdog --ticket_type straight \
    --stake 2 --payout_multiplier 1.91 --reason_code "market-anchor model edge" \
    --legs "Shesterkin:NYR:OVER:24.5"

# After games complete, update results, then compute closing lines/CLV
python scripts/update_betting_results.py
python scripts/compute_closing_clv.py

# View performance and CLV
python scripts/betting_dashboard.py
python scripts/clv_report.py
```

`data/betting.db` (SQLite) is the source of truth for all of the above. `betting_tracker.xlsx` is a read-only snapshot regenerated after every write -- open it to browse, but never edit it directly; edits won't persist. (The xlsx snapshot covers the original `bets` table only -- `line_snapshots`/`tickets`/`ticket_legs` are queried directly from `data/betting.db`, e.g. via `clv_report.py`.)

### GitHub Actions (phone-first workflow)

All four daily steps are available as `workflow_dispatch` Actions, and all four commit directly to `main` (each pushes with a fetch/rebase retry loop, so a rare race between two runs fails loudly and cleanly instead of overwriting anything -- see the "Concurrent writes" note below). The full loop -- fetch, bet/ticket, check results -- works from a phone without ever needing a PC:

1. **Fetch Predictions** -- fetches lines, runs predictions, snapshots every line to `line_snapshots`, commits directly to `main`
2. **Record Bet** -- fill in goalie/book/side/amount against the model's recommendation, commits directly to `main`
3. **Record Ticket** -- fill in book/stake/payout/reason plus a compact 1-3 leg string (syntax documented on the workflow's `legs` input), commits directly to `main`
4. **Update Betting Results** -- fetches completed game results, then computes closing lines/CLV and settles completed tickets, commits directly to `main`

**Concurrent writes:** `data/betting.db` is a single SQLite file, so git can't merge two commits that both touch it -- a genuine collision (e.g. `record_bet` and `fetch_predictions` racing within the same few seconds) makes the losing workflow run fail with a clear "rebase conflict" error rather than silently overwriting data. Just re-run the failed workflow; it'll start from the now-current `main` and won't conflict.

## Documentation

Deep-dive reference material lives in `docs/` and is kept up to date across sessions rather than treated as a one-off writeup:

| Doc | What it covers |
|-----|-----------------|
| [docs/HISTORICAL_DATA_ANALYSIS.md](docs/HISTORICAL_DATA_ANALYSIS.md) | The authoritative synthesis of every historical-data finding -- what actually has edge (a directional UNDER-vs-OVER signal, not a broad EV-threshold edge) and what doesn't |
| [docs/CURRENT_HISTORICAL_DATA.md](docs/CURRENT_HISTORICAL_DATA.md) | Audit of every historical data source, how they relate, and how much data the model actually has to train/evaluate on |
| [docs/MODEL_TRAINING_GUIDE.md](docs/MODEL_TRAINING_GUIDE.md) | Complete account of how the current production model was built -- pipeline, features, hyperparameters, bugs found along the way |

## Project Structure

```
saves-model-v3/
├── .github/workflows/
│   ├── fetch_predictions.yml        # Fetch lines + predictions, snapshot every line, direct commit
│   ├── record_bet.yml               # Record a placed bet, direct commit
│   ├── record_ticket.yml            # Record a 1-3 leg ticket with stake/payout/reason, direct commit
│   └── update_results.yml           # Update results, compute closing lines/CLV, direct commit
├── models/trained/
│   └── tuned_v1_20260201_155204/    # Active production model (114 features)
├── src/
│   ├── betting/                     # Betting module
│   │   ├── predictor.py            # XGBoost prediction interface
│   │   ├── feature_calculator.py   # Real-time feature calculation (114 features)
│   │   ├── db_manager.py           # SQLite read/write layer for `bets` (source of truth)
│   │   ├── tracking_db.py          # Schema + storage for line_snapshots/tickets/ticket_legs and CLV math
│   │   ├── excel_export.py         # Regenerates the read-only xlsx snapshot
│   │   ├── nhl_fetcher.py          # NHL API data fetching + boxscore caching
│   │   ├── odds_fetcher.py         # Underdog + BetOnline line fetching
│   │   ├── odds_utils.py           # EV calculation utilities
│   │   └── metrics.py              # Performance/ROI metrics used by the dashboard
│   ├── data/
│   │   └── api_client.py           # NHL API client
│   ├── features/
│   │   └── feature_engineering.py  # Training feature pipeline
│   └── models/
│       └── classifier_trainer.py   # XGBoost training wrapper
├── scripts/                         # CLI scripts (see table above)
│   └── archive/                     # Retired one-off scripts, kept for reference
├── data/
│   ├── betting.db                   # Betting tracker database (source of truth, tracked in git)
│   └── processed/                   # Training parquets (gitignored, regenerate via scripts/)
├── docs/                            # Living reference docs (see Documentation section above)
├── HANDOVER/                        # Cross-session handover notes (gitignored, local only)
├── betting_tracker.xlsx             # Read-only Excel snapshot (do not hand-edit)
└── requirements.txt
```

## Model Details

### Architecture

- **Model**: XGBoost Classifier (Booster format)
- **Features**: 114 predictive features (96 base + 18 engineered)
- **Output**: Probability of going OVER the betting line
- **EV Threshold**: 12% minimum for bet recommendations
- **Training Data**: Multi-book (multiple bookmaker lines per goalie-game)

### Current Production Model

- **Location**: `models/trained/tuned_v1_20260201_155204/`
- **Hyperparameters**: depth=6, lr=0.05, mcw=30, gamma=2.0, alpha=20, lambda=60, n_estimators=600
- **Performance**:

| Metric | Validation | Test | Combined |
|--------|------------|------|----------|
| ROI | +27.05% | +20.45% | +23.31% |
| Bets | 191 (18%) | 250 (23%) | 441 |

> **These are backtest numbers and they do NOT hold out-of-sample (2026-07-24).**
> A preregistered walk-forward test of this recipe — retrained per fold, evaluated
> forward-in-time on two unseen seasons — returned **pooled ROI -7.72%** over
> 3,258 bets (95% CI [-13.48%, -2.16%]), with AUC below 0.5 on both folds. Do not
> cite the table above as evidence of edge. See
> [docs/HISTORICAL_DATA_ANALYSIS.md](docs/HISTORICAL_DATA_ANALYSIS.md) section 10.

### Feature Categories (114 total)

| Category | Features | Count |
|----------|----------|-------|
| Context | `is_home` | 1 |
| Goalie basic rolling | `saves`, `shots_against`, `goals_against`, `save_percentage` x 3 windows x (mean + std) | 24 |
| Goalie situation rolling | `even_strength_*`, `power_play_*`, `short_handed_*` x 3 windows x (mean + std) | 54 |
| Team/opponent rolling | `opp_goals/shots`, `team_goals_against/shots_against` x 2 windows | 8 |
| Rest/fatigue | `goalie_days_rest`, `goalie_is_back_to_back` | 2 |
| Betting line | `betting_line` | 1 |
| Line-relative | `line_vs_rolling_*`, `line_z_score_*` x 3 windows | 6 |
| Engineered: interaction | `save_efficiency_*`, `es_saves_proportion_*`, `opp_vs_team_shots_*` | 7 |
| Engineered: volatility | `saves_cv_*`, `volatility_vs_line_*` | 4 |
| Engineered: momentum | `saves_momentum`, `shots_against_momentum`, `goals_against_momentum`, `save_pct_momentum` | 4 |
| Engineered: matchup | `expected_workload_diff`, `line_vs_opp_implied_saves`, `rest_x_performance` | 3 |
| **Total** | | **114** |

### Inference Pipeline

At prediction time, the feature calculator:
1. Fetches goalie game log from NHL API (basic stats)
2. Fetches boxscores for recent games (situation-specific stats: even strength, power play, shorthanded)
3. Fetches opponent team schedule + boxscores (opponent offensive rolling stats)
4. Computes rolling means and standard deviations for 3, 5, and 10 game windows
5. Computes line-relative features (line vs recent saves average, z-score)
6. Computes 18 engineered features (interactions, volatility, momentum, matchup context)

Boxscores are cached in-memory to avoid redundant API calls within a session.

### EV Calculation

```
implied_prob = american_odds_to_prob(odds)
edge = model_prob - implied_prob
ev = edge * potential_profit
```

A bet is recommended when EV >= 12%.

## Betting Tracker

`data/betting.db` (SQLite) is the source of truth for every line, prediction, bet, ticket, and result. It's committed to git, so full history lives in `git log`. Nothing hand-edits it directly -- all writes go through the scripts in the table above.

Four tables:

- **`bets`** -- the original recommendation log: one row per (game, goalie, book, line), whether or not it was ever staked. `predicted_saves`/`prob_over`/`recommendation`/`ev` are written for every line the model sees (this is also the "shadow run" log -- see `model_version` below), and `bet_amount`/`bet_selection`/`bet_placed_at` are filled in independently by `record_bet.py` for a single line you actually bet. `model_version` (added by `add_tracking_tables.py`) attributes each recommendation row to the model that produced it, so future model swaps stay comparable.
- **`line_snapshots`** -- the CLV backbone. One row per (fetch run, book, goalie, market line), appended by `fetch_and_predict.py` on every run regardless of whether the line changed. Carries a UTC fetch timestamp and, when the book reports one, the scheduled game start in UTC -- both needed to reconstruct closing lines and line-move history after the fact.
- **`tickets`** / **`ticket_legs`** -- actual ticket economics, independent of `bets`' one-row-per-line model: stake, payout multiplier/potential payout, status, actual payout, and a *required* `reason_code` per ticket (`record_ticket.py` enforces this). Each leg auto-matches the nearest `line_snapshots` row at bet time and, once `compute_closing_clv.py` runs, is filled in with `closing_line`/`closing_odds`/`result` plus two independently-computed CLV columns: `clv_saves` (pure line movement, signed so positive = the bettor got the better number for their side) and `clv_prob_novig` (de-vigged implied-probability movement for that side). American odds are never arithmetically averaged anywhere in this pipeline -- see the odds-averaging bug in `docs/HISTORICAL_DATA_ANALYSIS.md` section 1.

`betting_tracker.xlsx` is a read-only snapshot of the `bets` table, regenerated from the database after every write to `bets`, for browsing:

- **Summary** - Overall performance metrics
- **Settings** - Configuration reference
- **Date sheets** - One sheet per date with all lines, predictions, bets, and results

`line_snapshots`/`tickets`/`ticket_legs` are not part of the xlsx snapshot -- query `data/betting.db` directly, or run `clv_report.py` for a formatted view.

Columns tracked in `bets`:
- Game info (date, teams, goalie)
- Book and line info (book, line, over/under odds)
- Predictions (predicted saves, prob_over, recommendation, EV, model_version)
- Your bet (bet_amount, bet_selection, bet_placed_at, notes) -- independent of the model's recommendation
- Results (actual saves, result, profit/loss)

## Configuration

Key settings in `src/betting/predictor.py`:
- Model path: `models/trained/tuned_v1_20260201_155204/`
- EV threshold: 12%

## Requirements

- Python 3.12+
- pandas, numpy, xgboost, scikit-learn
- requests (for API calls)
- openpyxl (for Excel handling)

## Disclaimer

This model is for educational and entertainment purposes only. Past performance does not guarantee future results. Please gamble responsibly.
