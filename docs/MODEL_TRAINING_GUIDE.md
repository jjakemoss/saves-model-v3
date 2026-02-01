# NHL Goalie Saves Model - Training Guide

This document describes the complete process for training the NHL goalie saves prediction model, from data collection through model deployment.

## Table of Contents

1. [Overview](#overview)
2. [Data Collection](#data-collection)
3. [Feature Engineering](#feature-engineering)
4. [Historical Betting Lines](#historical-betting-lines)
5. [Model Training](#model-training)
6. [Feature Optimization](#feature-optimization)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Model Evaluation](#model-evaluation)
9. [Model Deployment](#model-deployment)
10. [Quick Reference Commands](#quick-reference-commands)

---

## Overview

### Model Type
- **Algorithm**: XGBoost Binary Classifier (saved as Booster JSON format)
- **Objective**: Predict probability of OVER/UNDER on goalie saves betting lines
- **Target**: `over_hit` (1 if actual_saves > betting_line, 0 otherwise)

### Current Production Model
- **Name**: Tuned V1 (hyperparameter-optimized with engineered features)
- **EV Threshold**: 12% minimum
- **Features**: 114 (96 base + 18 engineered)
- **Performance**: +23.31% ROI combined (441 bets across validation + test)
- **Location**: `models/trained/tuned_v1_20260201_155204/`

### Model Evolution

| Model | Features | EV Threshold | Combined ROI | Combined Bets |
|-------|----------|-------------|--------------|---------------|
| Config #4398 (single-book) | 90 | 2% | +1.60% | 699 |
| Multibook V1 (line-relative) | 96 | 12% | +7.32% | 700 |
| Optimized V1 (engineered) | 114 | 12% | +15.24% | 451 |
| **Tuned V1 (current)** | **114** | **12%** | **+23.31%** | **441** |

### Key Files

| File | Purpose |
|------|---------|
| `scripts/collect_historical_data.py` | Collect NHL game data |
| `scripts/create_features.py` | Generate training features |
| `scripts/build_multibook_training_data.py` | Build multi-book training data |
| `scripts/train_production_multibook.py` | Train multi-book model |
| `scripts/optimize_features.py` | Test feature engineering configurations |
| `scripts/tune_hyperparameters.py` | Hyperparameter tuning with randomized search |
| `src/features/feature_engineering.py` | Feature calculation pipeline |
| `src/models/classifier_trainer.py` | XGBoost training wrapper |
| `src/betting/predictor.py` | Model inference |
| `src/betting/feature_calculator.py` | Real-time feature calculation (114 features) |

---

## Data Collection

### Step 1: Run Data Collection

```bash
python scripts/collect_historical_data.py --seasons 20222023 20232024 20242025 20252026
```

This collects:
- **Schedules**: Game dates and matchups for all NHL teams
- **Boxscores**: Game statistics (saves, shots, goals, situation-specific stats)
- **Goalie game logs**: Individual goalie performance history

### Data Sources

| Data Type | Source | API Endpoint |
|-----------|--------|--------------|
| Schedules | NHL API | `/v1/schedule/{date}` |
| Boxscores | NHL API | `/v1/gamecenter/{gameId}/boxscore` |
| Goalie logs | NHL API | `/v1/player/{playerId}/game-log/{season}/{gameType}` |
| Team schedules | NHL API | `/v1/club-schedule-season/{team}/{season}` |

### Data Storage

```
data/
├── raw/
│   ├── boxscores/          # JSON files: {game_id}.json
│   └── betting_lines/
│       └── cache/          # Historical odds from The-Odds-API
├── processed/
│   ├── training_data.parquet                          # Base features (no odds)
│   ├── classification_training_data.parquet           # Single-book with odds
│   └── multibook_classification_training_data.parquet # Multi-book with odds
└── cache/
    └── api_cache.db        # SQLite cache for NHL API (24-hour TTL)
```

---

## Feature Engineering

### Step 2: Generate Features

```bash
python scripts/create_features.py
```

This creates `data/processed/training_data.parquet` with all calculated features.

### Feature Pipeline

The pipeline in `src/features/feature_engineering.py`:

1. **Extract base features** from boxscores (including situation-specific stats)
2. **Filter to starting goalies only** (exclude relief appearances)
3. **Sort chronologically** (CRITICAL for preventing data leakage)
4. **Calculate rolling features** with `shift(1)` to exclude current game
5. **Add rest/fatigue features** (days rest, back-to-back)
6. **Add team rolling features** (goals against, shots against per team)
7. **Add opponent rolling features** (goals scored, shots per opponent team)
8. **Fill missing values** with season averages

### 114 Production Features

The model uses **114 features** across these categories:

| Category | Features | Count |
|----------|----------|-------|
| Context | `is_home` | 1 |
| Goalie basic rolling | `saves`, `shots_against`, `goals_against`, `save_percentage` x 3 windows x (mean + std) | 24 |
| Goalie situation-specific | `even_strength_*`, `power_play_*`, `short_handed_*` (saves, shots_against, goals_against) x 3 windows x (mean + std) | 54 |
| Team defensive rolling | `team_goals_against`, `team_shots_against` x 2 windows | 4 |
| Opponent offensive rolling | `opp_goals`, `opp_shots` x 2 windows | 4 |
| Rest/fatigue | `goalie_days_rest`, `goalie_is_back_to_back` | 2 |
| Betting line | `betting_line` | 1 |
| Line-relative | `line_vs_rolling_*`, `line_z_score_*` x 3 windows | 6 |
| Engineered: interaction | `save_efficiency_{3,5,10}`, `es_saves_proportion_{5,10}`, `opp_vs_team_shots_{5,10}` | 7 |
| Engineered: volatility | `saves_cv_{5,10}`, `volatility_vs_line_{5,10}` | 4 |
| Engineered: momentum | `saves_momentum`, `shots_against_momentum`, `goals_against_momentum`, `save_pct_momentum` | 4 |
| Engineered: matchup | `expected_workload_diff`, `line_vs_opp_implied_saves`, `rest_x_performance` | 3 |
| **Total** | | **114** |

### Rolling Windows

All rolling features are computed for 3 windows: **3, 5, and 10 games**. Team/opponent features use 5 and 10 game windows.

For each rolling window, both **mean** and **standard deviation** are calculated:
- `saves_rolling_5` = mean of saves over last 5 games
- `saves_rolling_std_5` = std of saves over last 5 games

### Situation-Specific Stats

The model includes separate rolling stats for different game situations:
- **Even strength**: `even_strength_saves`, `even_strength_shots_against`, `even_strength_goals_against`
- **Power play**: `power_play_saves`, `power_play_shots_against`, `power_play_goals_against`
- **Shorthanded**: `short_handed_saves`, `short_handed_shots_against`, `short_handed_goals_against`

These are sourced from boxscores (not game logs, which lack this data).

### Engineered Features

The 18 engineered features are derived from base features:

**Interaction features:**
- `save_efficiency_{w}` = `saves_rolling_{w}` / `shots_against_rolling_{w}` (goalie efficiency)
- `es_saves_proportion_{w}` = `even_strength_saves_rolling_{w}` / `saves_rolling_{w}` (even strength proportion)
- `opp_vs_team_shots_{w}` = `opp_shots_rolling_{w}` - `team_shots_against_rolling_{w}` (opponent firepower vs team defense)

**Volatility features:**
- `saves_cv_{w}` = `saves_rolling_std_{w}` / `saves_rolling_{w}` (coefficient of variation)
- `volatility_vs_line_{w}` = `saves_rolling_std_{w}` / `betting_line` (volatility relative to line)

**Momentum features:**
- `saves_momentum` = `saves_rolling_3` - `saves_rolling_10` (short-term vs long-term trend)
- `shots_against_momentum` = `shots_against_rolling_3` - `shots_against_rolling_10`
- `goals_against_momentum` = `goals_against_rolling_3` - `goals_against_rolling_10`
- `save_pct_momentum` = `save_percentage_rolling_3` - `save_percentage_rolling_10`

**Matchup context features:**
- `expected_workload_diff` = `opp_shots_rolling_5` - `shots_against_rolling_5`
- `line_vs_opp_implied_saves` = `betting_line` - (`opp_shots_rolling_5` - `opp_goals_rolling_5`)
- `rest_x_performance` = `min(goalie_days_rest, 7)` * `saves_rolling_5`

### Data Leakage Prevention

**CRITICAL**: All rolling features must use `shift(1)` to exclude current game:

```python
# CORRECT: Excludes current game
df[col] = df.groupby('goalie_id')[stat].transform(
    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
)

# WRONG: Includes current game (data leakage!)
df[col] = df.groupby('goalie_id')[stat].transform(
    lambda x: x.rolling(window=window, min_periods=1).mean()
)
```

---

## Historical Betting Lines

### Multi-Book Training Data

The model is trained on multi-book data where each goalie-game can have lines from multiple bookmakers. This provides more training samples and line-relative feature variation.

```bash
python scripts/build_multibook_training_data.py
```

Output: `data/processed/multibook_classification_training_data.parquet`

### Classification Training Data Requirements

The training data must contain:

| Column | Description | Example |
|--------|-------------|---------|
| `betting_line` | Saves over/under line | 25.5 |
| `odds_over_american` | American odds for OVER | -115 |
| `odds_under_american` | American odds for UNDER | -105 |
| `over_hit` | Target: 1 if saves > line, 0 otherwise | 1 |
| `book_key` | Bookmaker identifier | `draftkings` |

### Historical Odds Cache (The-Odds-API)

Historical betting lines were collected via The-Odds-API and cached locally.

**Cache Location**: `data/raw/betting_lines/cache/`

**Available Bookmakers**:

| Bookmaker Key | Name |
|---------------|------|
| `draftkings` | DraftKings |
| `fanduel` | FanDuel |
| `betmgm` | BetMGM |
| `betonlineag` | BetOnline.ag |
| `williamhill_us` | Caesars |
| `pinnacle` | Pinnacle |

---

## Model Training

### Step 3: Train Multi-Book Model

```bash
python scripts/train_production_multibook.py
```

### Current Hyperparameters (Tuned V1)

```python
TUNED_V1 = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_child_weight': 30,
    'gamma': 2.0,
    'reg_alpha': 20,
    'reg_lambda': 60,
    'n_estimators': 600,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
}
```

### Data Split

**Chronological split** (prevents temporal leakage):
- **Train**: 60% (earliest games)
- **Validation**: 20% (mid-season)
- **Test**: 20% (most recent games)

```python
n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train_idx = np.arange(0, train_end)
val_idx = np.arange(train_end, val_end)
test_idx = np.arange(val_end, n)
```

### Training Process

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric=['logloss', 'auc'],
    **TUNED_V1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True
)

# Save as Booster (compatible with predictor.py which loads as Booster)
model.get_booster().save_model('classifier_model.json')
```

### Features Excluded from Training

These columns are removed before training:

```python
excluded_cols = [
    # Metadata
    'game_id', 'goalie_id', 'game_date', 'season', 'team_abbrev',
    'opponent_team', 'toi', 'goalie_name', 'team_id',
    'book_key', 'decision',

    # Target variables
    'over_hit', 'saves_margin', 'over_line', 'line_margin',

    # Odds (used for evaluation, not features)
    'odds_over_american', 'odds_under_american',
    'odds_over_decimal', 'odds_under_decimal', 'num_books',

    # Current-game stats (data leakage)
    'saves', 'shots_against', 'goals_against', 'save_percentage',
    'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
    'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
    'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
    'team_goals', 'team_shots', 'opp_goals', 'opp_shots',

    # Market-derived features (not available at inference)
    'line_vs_recent_avg', 'line_vs_season_avg', 'line_surprise_score',
    'market_vig', 'impl_prob_over', 'impl_prob_under',
    'fair_prob_over', 'fair_prob_under', 'line_vs_opp_shots',
    'line_is_half', 'line_is_extreme_high', 'line_is_extreme_low',
]
```

---

## Feature Optimization

### Testing Feature Configurations

```bash
python scripts/optimize_features.py
```

This script tests multiple feature configurations:
1. Baseline (96 features)
2. Baseline + engineered features (114 features)
3. High regularization variants
4. Feature selection (dropping low-importance features)

Each configuration is evaluated on validation and test ROI. The best configuration found was 114 features with high regularization.

---

## Hyperparameter Tuning

### Randomized Search

```bash
python scripts/tune_hyperparameters.py
```

This script:
1. Tests 42 hyperparameter configurations x 4 EV thresholds (0.08, 0.10, 0.12, 0.15)
2. Uses the 114 engineered features from the optimization step
3. Filters results to 15-35% test bet rate for practical volume
4. Saves the best model automatically

### Hyperparameter Search Space

| Parameter | Values |
|-----------|--------|
| `max_depth` | 3, 4, 5, 6 |
| `learning_rate` | 0.01, 0.02, 0.05 |
| `min_child_weight` | 10, 15, 20, 30 |
| `gamma` | 0.5, 1.0, 2.0 |
| `reg_alpha` | 5, 10, 20 |
| `reg_lambda` | 20, 40, 60 |
| `n_estimators` | 600, 800, 1200 |

### EV Threshold Selection

| Threshold | Effect |
|-----------|--------|
| 8% | More bets, lower selectivity |
| 10% | Moderate selectivity |
| **12%** | **Current production (good balance)** |
| 15% | Fewer bets, higher selectivity |

The 12% threshold was selected for the current model, producing a bet rate of ~20-25% of available lines.

---

## Model Evaluation

### Profitability Metrics

The model is evaluated on betting profitability, not just accuracy:

```python
def evaluate_profitability(X, y, df, split_idx, ev_threshold=0.12):
    """
    For each game:
    1. Get model probability: prob_over = model.predict(X)
    2. Calculate EV for both OVER and UNDER using actual odds
    3. If EV >= 12%: place bet on that side
    4. Track profit/loss at actual odds
    """
```

### Tuned V1 Results (Current Production)

| Metric | Validation | Test | Combined |
|--------|------------|------|----------|
| ROI | +27.05% | +20.45% | +23.31% |
| Bets | 191 (18%) | 250 (23%) | 441 |

### Previous Model Results (for comparison)

| Model | Val ROI | Test ROI | Combined ROI | Bets |
|-------|---------|----------|-------------|------|
| Config #4398 (2% EV) | +2.54% | +0.62% | +1.60% | 699 |
| Multibook V1 (12% EV) | +3.13% | +11.29% | +7.32% | 700 |
| Optimized V1 (12% EV) | +21.01% | +10.68% | +15.24% | 451 |
| **Tuned V1 (12% EV)** | **+27.05%** | **+20.45%** | **+23.31%** | **441** |

---

## Model Deployment

### Model Artifacts

After training, these files are saved:

```
models/trained/tuned_v1_{timestamp}/
├── classifier_model.json          # XGBoost Booster model in JSON format
├── classifier_feature_names.json  # List of 114 features in exact order
└── classifier_metadata.json       # Hyperparameters and performance metrics
```

### Update Predictor Path

After training a new model, update the default path in `src/betting/predictor.py`:

```python
class BettingPredictor:
    def __init__(
        self,
        model_path='models/trained/tuned_v1_20260201_155204/classifier_model.json',
        feature_order_path='models/trained/tuned_v1_20260201_155204/classifier_feature_names.json'
    ):
```

Also update the feature names path in `src/betting/feature_calculator.py`:

```python
class BettingFeatureCalculator:
    def __init__(self):
        feature_file = Path('models/trained/tuned_v1_20260201_155204/classifier_feature_names.json')
```

### Train/Serve Parity

The inference pipeline (`feature_calculator.py`) must compute features identically to training. Key considerations:

1. **Saves field**: The NHL game log API does not provide a `saves` field. Compute as `shotsAgainst - goalsAgainst`.
2. **Shots against**: Use `shotsAgainst` from game log (not `shots`).
3. **Situation-specific stats**: Not available from game log API. Must fetch individual boxscores via `/v1/gamecenter/{gameId}/boxscore` and parse situation strings (e.g., `evenStrengthShotsAgainst: "18/20"` means 18 saves on 20 shots).
4. **Team/opponent stats**: Fetch from boxscores (team defense) and opponent schedule + boxscores (opponent offense).
5. **Boxscore caching**: In-memory cache (`_boxscore_cache`) avoids re-fetching the same boxscore for multiple goalies.

### Making Predictions

```python
from betting import BettingPredictor, BettingFeatureCalculator, NHLBettingData

# Initialize
predictor = BettingPredictor()
feature_calc = BettingFeatureCalculator()
nhl_data = NHLBettingData()

# Get goalie's recent games
recent_games = nhl_data.get_goalie_recent_games(goalie_id, n_games=15)

# Calculate features (passes nhl_fetcher for boxscore data)
features_df = feature_calc.prepare_prediction_features(
    goalie_id=goalie_id,
    team='TOR',
    opponent='BOS',
    is_home=1,
    game_date='2026-02-01',
    recent_games=recent_games,
    betting_line=25.5,
    nhl_fetcher=nhl_data
)

# Generate prediction
prediction = predictor.predict(
    features_df,
    betting_line=25.5,
    line_over_odds=-115,
    line_under_odds=-105
)

# Result:
# {
#     'predicted_saves': 26.2,
#     'prob_over': 0.58,
#     'confidence_pct': 16.0,
#     'confidence_bucket': '55-60%',
#     'recommendation': 'OVER',
#     'ev_over': 0.15,
#     'ev_under': -0.04,
#     'recommended_ev': 0.15
# }
```

---

## Quick Reference Commands

### Full Training Pipeline

```bash
# 1. Collect data (if needed)
python scripts/collect_historical_data.py --seasons 20242025 20252026

# 2. Generate features
python scripts/create_features.py

# 3. Build multi-book training data (requires historical odds in cache)
python scripts/build_multibook_training_data.py

# 4. (Optional) Test feature configurations
python scripts/optimize_features.py

# 5. (Optional) Hyperparameter tuning
python scripts/tune_hyperparameters.py

# 6. Train production model
python scripts/train_production_multibook.py

# 7. Update predictor paths in src/betting/predictor.py and src/betting/feature_calculator.py
```

### Daily Operations

```bash
# Fetch lines and generate predictions
python scripts/fetch_and_predict.py --verbose

# Update results after games complete
python scripts/update_betting_results.py

# View dashboard
python scripts/betting_dashboard.py
```

### Incremental Retraining

When you have more data:

1. Run data collection for new dates
2. Regenerate features
3. Rebuild multi-book training data with new odds
4. Run hyperparameter tuning (or retrain with existing params)
5. Compare validation/test ROI to previous model
6. Deploy if improved (update predictor paths)

---

## Appendix: Directory Structure

```
saves-model-v3/
├── .github/workflows/
│   └── fetch_predictions.yml       # GitHub Action for predictions
├── config/
│   └── config.yaml                 # Configuration settings
├── data/
│   ├── raw/
│   │   ├── boxscores/              # Raw game data
│   │   └── betting_lines/
│   │       └── cache/              # The-Odds-API historical odds cache
│   ├── processed/
│   │   ├── training_data.parquet
│   │   ├── classification_training_data.parquet
│   │   └── multibook_classification_training_data.parquet
│   └── cache/
│       └── api_cache.db            # NHL API cache
├── models/
│   └── trained/
│       └── tuned_v1_20260201_155204/
│           ├── classifier_model.json
│           ├── classifier_feature_names.json
│           └── classifier_metadata.json
├── scripts/
│   ├── collect_historical_data.py
│   ├── create_features.py
│   ├── build_multibook_training_data.py
│   ├── train_production_multibook.py
│   ├── optimize_features.py
│   ├── tune_hyperparameters.py
│   ├── fetch_and_predict.py
│   ├── generate_predictions.py
│   ├── update_betting_results.py
│   ├── betting_dashboard.py
│   ├── add_manual_lines.py
│   └── init_betting_tracker.py
├── src/
│   ├── betting/
│   │   ├── predictor.py            # Model inference (loads Booster)
│   │   ├── feature_calculator.py   # Real-time 114-feature computation
│   │   ├── nhl_fetcher.py          # NHL API + boxscore caching
│   │   ├── odds_fetcher.py         # Underdog + BetOnline fetching
│   │   ├── excel_manager.py        # Excel tracker I/O
│   │   └── odds_utils.py           # EV calculation
│   ├── data/
│   │   ├── api_client.py           # NHL API client
│   │   └── cache_manager.py        # API response caching
│   ├── features/
│   │   ├── feature_engineering.py  # Training feature pipeline
│   │   ├── rolling_features.py     # Rolling stat calculations
│   │   └── team_rolling_features.py # Team-level rolling stats
│   └── models/
│       └── classifier_trainer.py   # XGBoost training + evaluation
└── betting_tracker.xlsx            # Excel tracker for daily bets
```
