# NHL Goalie Saves Prediction Model

An XGBoost classifier that predicts NHL goalie saves over/under betting lines. Automatically fetches lines from multiple sportsbooks and calculates expected value (EV) for betting recommendations.

## Features

- **Multi-book line fetching** from Underdog Fantasy and BetOnline (via The-Odds-API)
- **114 predictive features** including rolling averages, situation-specific stats, team/opponent stats, and engineered features
- **EV-based recommendations** with 12% minimum threshold
- **Boxscore-powered inference** fetching real situation-specific stats from NHL API
- **SQLite-backed betting tracker** (`data/betting.db`) as the single source of truth, with a read-only `betting_tracker.xlsx` snapshot regenerated after every write
- **Phone-first GitHub Actions workflow** for fetching predictions, recording bets, and updating results without needing a PC

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
| `fetch_and_predict.py` | Main script - fetches lines from Underdog/BetOnline and generates predictions |
| `record_bet.py` | Record a bet you placed, independent of the model's recommendation |
| `update_betting_results.py` | Update actual saves and P/L after games complete |
| `betting_dashboard.py` | Display performance metrics and refresh the Excel snapshot |
| `init_betting_tracker.py` | Create a new betting database and its Excel snapshot |
| `migrate_to_sqlite.py` | One-off: fold an existing xlsx/CSV history into `data/betting.db` |
| `optimize_features.py` | Test feature engineering configurations |
| `tune_hyperparameters.py` | Hyperparameter tuning with randomized search |
| `train_production_multibook.py` | Train model on multi-book training data |
| `build_multibook_training_data.py` | Build training data with multiple bookmaker lines per game |

## Usage

### Daily Workflow

```bash
# Fetch lines and predictions (run multiple times as lines update)
python scripts/fetch_and_predict.py --verbose

# Right after placing a bet on your sportsbook app
python scripts/record_bet.py --goalie_name Shesterkin --book Underdog \
    --bet_selection OVER --bet_amount 2

# After games complete, update results
python scripts/update_betting_results.py

# View performance
python scripts/betting_dashboard.py
```

`data/betting.db` (SQLite) is the source of truth for all of the above. `betting_tracker.xlsx` is a read-only snapshot regenerated after every write -- open it to browse, but never edit it directly; edits won't persist.

### GitHub Actions (phone-first workflow)

All three daily steps are available as `workflow_dispatch` Actions, and all three commit directly to `main` (each pushes with a fetch/rebase retry loop, so a rare race between two runs fails loudly and cleanly instead of overwriting anything -- see the "Concurrent writes" note below). The full loop -- fetch, bet, record, check results -- works from a phone without ever needing a PC:

1. **Fetch Predictions** -- fetches lines, runs predictions, commits directly to `main`
2. **Record Bet** -- fill in goalie/book/side/amount, commits directly to `main`
3. **Update Betting Results** -- fetches completed game results, commits directly to `main`

**Concurrent writes:** `data/betting.db` is a single SQLite file, so git can't merge two commits that both touch it -- a genuine collision (e.g. `record_bet` and `fetch_predictions` racing within the same few seconds) makes the losing workflow run fail with a clear "rebase conflict" error rather than silently overwriting data. Just re-run the failed workflow; it'll start from the now-current `main` and won't conflict.

See [docs/WORKFLOW_MODERNIZATION_PROPOSAL.md](docs/WORKFLOW_MODERNIZATION_PROPOSAL.md) for the full design.

## Project Structure

```
saves-model-v3/
├── .github/workflows/
│   ├── fetch_predictions.yml        # Fetch lines + predictions, direct commit
│   ├── record_bet.yml               # Record a placed bet, direct commit
│   └── update_results.yml           # Update completed game results, direct commit
├── models/trained/
│   └── tuned_v1_20260201_155204/    # Active production model (114 features)
├── src/
│   ├── betting/                     # Betting module
│   │   ├── predictor.py            # XGBoost prediction interface
│   │   ├── feature_calculator.py   # Real-time feature calculation (114 features)
│   │   ├── db_manager.py           # SQLite read/write layer (source of truth)
│   │   ├── excel_export.py         # Regenerates the read-only xlsx snapshot
│   │   ├── nhl_fetcher.py          # NHL API data fetching + boxscore caching
│   │   ├── odds_fetcher.py         # Underdog + BetOnline line fetching
│   │   └── odds_utils.py           # EV calculation utilities
│   ├── data/
│   │   └── api_client.py           # NHL API client
│   ├── features/
│   │   └── feature_engineering.py  # Training feature pipeline
│   └── models/
│       └── classifier_trainer.py   # XGBoost training wrapper
├── scripts/                         # CLI scripts (see table above)
├── data/
│   ├── betting.db                   # Betting tracker database (source of truth)
│   └── processed/
│       └── multibook_classification_training_data.parquet  # Training data
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

`data/betting.db` (SQLite, single `bets` table) is the source of truth for every line, prediction, bet, and result. It's committed to git, so full history lives in `git log`. Nothing hand-edits it directly -- all writes go through `fetch_and_predict.py`, `record_bet.py`, or `update_betting_results.py`.

`betting_tracker.xlsx` is a read-only snapshot regenerated from the database after every write, for browsing:

- **Summary** - Overall performance metrics
- **Settings** - Configuration reference
- **Date sheets** - One sheet per date with all lines, predictions, bets, and results

Columns tracked:
- Game info (date, teams, goalie)
- Book and line info (book, line, over/under odds)
- Predictions (predicted saves, prob_over, recommendation, EV)
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
