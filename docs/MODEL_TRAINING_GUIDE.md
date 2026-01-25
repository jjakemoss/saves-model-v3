# NHL Goalie Saves Model - Training Guide

This document describes the complete process for training the NHL goalie saves prediction model, from data collection through model deployment.

## Table of Contents

1. [Overview](#overview)
2. [Data Collection](#data-collection)
3. [Feature Engineering](#feature-engineering)
4. [Historical Betting Lines](#historical-betting-lines)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Model Deployment](#model-deployment)
8. [Quick Reference Commands](#quick-reference-commands)

---

## Overview

### Model Type
- **Algorithm**: XGBoost Binary Classifier (`XGBClassifier`)
- **Objective**: Predict probability of OVER/UNDER on goalie saves betting lines
- **Target**: `over_hit` (1 if actual_saves > betting_line, 0 otherwise)

### Current Production Model
- **Config**: #4398
- **EV Threshold**: 2% minimum
- **Performance**: +1.60% ROI combined (699 bets across validation + test)
- **Location**: `models/trained/config_4398_ev2pct_20260115_103430/`

### Key Files

| File | Purpose |
|------|---------|
| `scripts/collect_historical_data.py` | Collect NHL game data |
| `scripts/create_features.py` | Generate training features |
| `scripts/train_production_4398.py` | Train production model |
| `src/features/feature_engineering.py` | Feature calculation pipeline |
| `src/models/classifier_trainer.py` | XGBoost training wrapper |
| `src/betting/predictor.py` | Model inference |

---

## Data Collection

### Step 1: Run Data Collection

```bash
python scripts/collect_historical_data.py --seasons 20222023 20232024 20242025 20252026
```

This collects:
- **Schedules**: Game dates and matchups for all NHL teams
- **Boxscores**: Game statistics (saves, shots, goals, etc.)
- **Play-by-play**: Shot locations and timing (for shot quality features)
- **Goalie game logs**: Individual goalie performance history

### Data Sources

| Data Type | Source | API Endpoint |
|-----------|--------|--------------|
| Schedules | NHL API | `/v1/schedule/{date}` |
| Boxscores | NHL API | `/v1/gamecenter/{gameId}/boxscore` |
| Play-by-play | NHL API | `/v1/gamecenter/{gameId}/play-by-play` |
| Goalie logs | NHL API | `/v1/player/{playerId}/game-log/{season}/{gameType}` |

### Data Storage

```
data/
├── raw/
│   ├── boxscores/          # JSON files: {game_id}.json
│   └── play_by_play/       # JSON files: {game_id}.json
├── processed/
│   ├── training_data.parquet                    # Base features (no odds)
│   └── classification_training_data.parquet    # With historical odds
└── cache/
    └── api_cache.db        # SQLite cache for NHL API (24-hour TTL)
```

### Cache Management

The NHL API responses are cached in `data/cache/api_cache.db`:
- **TTL**: 24 hours (configurable in `config/config.yaml`)
- **Format**: SQLite database with MD5 hash keys
- **Size**: ~750 MB for 2+ seasons of data

---

## Feature Engineering

### Step 2: Generate Features

```bash
python scripts/create_features.py
```

This creates `data/processed/training_data.parquet` with all calculated features.

### Feature Pipeline

The pipeline in `src/features/feature_engineering.py`:

1. **Extract base features** from boxscores and play-by-play
2. **Filter to starting goalies only** (exclude relief appearances)
3. **Sort chronologically** (CRITICAL for preventing data leakage)
4. **Calculate rolling features** with `shift(1)` to exclude current game
5. **Add rest/fatigue features** (days rest, back-to-back)
6. **Add team rolling features** (goals against, shots against)
7. **Fill missing values** with season averages

### 90 Production Features

The model uses exactly **90 features**:

| Category | Features | Count |
|----------|----------|-------|
| Context | `is_home` | 1 |
| Goalie rolling stats | `{stat}_rolling_{3,5,10}` + `_std` variants | 72 |
| Team/opponent rolling | `opp_*_rolling_{5,10}`, `team_*_rolling_{5,10}` | 8 |
| Rest/fatigue | `goalie_days_rest`, `goalie_is_back_to_back` | 2 |
| Betting line | `betting_line` | 1 |
| **Total** | | **90** |

### Goalie Rolling Stats (72 features)

For each stat below, calculate `_rolling_{3,5,10}` mean and `_rolling_std_{3,5,10}` std:

- `saves`
- `shots_against`
- `goals_against`
- `save_percentage`
- `even_strength_saves`
- `even_strength_shots_against`
- `even_strength_goals_against`
- `power_play_saves`
- `power_play_shots_against`
- `power_play_goals_against`
- `short_handed_saves`
- `short_handed_shots_against`
- `short_handed_goals_against`

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

### Classification Training Data

The model requires historical betting lines with odds for training. The final training file `data/processed/classification_training_data.parquet` must contain:

| Column | Description | Example |
|--------|-------------|---------|
| `betting_line` | Saves over/under line | 25.5 |
| `odds_over_american` | American odds for OVER | -115 |
| `odds_under_american` | American odds for UNDER | -105 |
| `over_hit` | Target: 1 if saves > line, 0 otherwise | 1 |

### Data Summary

Current training data (as of Jan 2026):
- **Samples**: 3,757 goalie games
- **Date range**: Oct 4, 2024 to Jan 3, 2026
- **Seasons**: 20242025 (2,511), 20252026 (1,246)
- **Over/Under split**: 47.6% OVER, 52.4% UNDER

### Historical Odds Cache (The-Odds-API)

Historical betting lines were collected via The-Odds-API and cached locally.

**Cache Location**: `data/raw/betting_lines/cache/`

**Cache Contents**:
- **275 events files**: Daily game schedules (`events_date=YYYY-MM-DDTHH_MM_SSZ.json`)
- **1,976 odds files**: Per-game betting lines (`odds_{eventId}_date=...json`)
- **Date range**: Oct 4, 2024 to Jan 16, 2026

**File Naming Convention**:
```
# Events file (daily schedule)
events_date=2024-10-04T18_00_00Z.json

# Odds file (per-game lines)
odds_{eventId}_date={gameTime}_markets=player_total_saves_regions=us.json
```

**Events File Structure**:
```json
{
  "timestamp": "2024-10-04T17:55:39Z",
  "data": [
    {
      "id": "e1dd2bc0fa38ee53116f047cf3d0327e",
      "sport_key": "icehockey_nhl",
      "commence_time": "2024-10-04T17:14:42Z",
      "home_team": "Buffalo Sabres",
      "away_team": "New Jersey Devils"
    }
  ]
}
```

**Odds File Structure**:
```json
{
  "timestamp": "2024-10-04T17:10:39Z",
  "data": {
    "id": "e1dd2bc0fa38ee53116f047cf3d0327e",
    "home_team": "Buffalo Sabres",
    "away_team": "New Jersey Devils",
    "bookmakers": [
      {
        "key": "williamhill_us",
        "title": "Caesars",
        "markets": [
          {
            "key": "player_total_saves",
            "outcomes": [
              {
                "name": "Over",
                "description": "Ukko-Pekka Luukkonen",
                "price": 1.84,
                "point": 26.5
              },
              {
                "name": "Under",
                "description": "Ukko-Pekka Luukkonen",
                "price": 1.81,
                "point": 26.5
              }
            ]
          }
        ]
      }
    ]
  }
}
```

### Available Bookmakers

The-Odds-API provides goalie saves lines from these bookmakers:

| Bookmaker Key | Name | Notes |
|---------------|------|-------|
| `draftkings` | DraftKings | Major US book |
| `fanduel` | FanDuel | Major US book |
| `betmgm` | BetMGM | Major US book |
| `betonlineag` | BetOnline.ag | Used in current integration |
| `williamhill_us` | Caesars | In historical cache |
| `espnbet` | ESPN Bet | US book |
| `hardrockbet` | Hard Rock Bet | US book |
| `pinnacle` | Pinnacle | Sharp book (reference lines) |

### Sources for Historical Odds

Options for obtaining historical betting lines:

1. **The-Odds-API** (used for training data)
   - Endpoint: `/v4/sports/icehockey_nhl/events/{eventId}/odds`
   - Market: `player_total_saves`
   - Historical data cached in `data/raw/betting_lines/cache/`

2. **Manual collection** from betting sites
   - Underdog Fantasy
   - PrizePicks
   - BetOnline

3. **Third-party historical data providers**

### Creating Classification Training Data

To process the cached historical odds and merge with features:

```python
import pandas as pd
import json
from pathlib import Path

# Load base features
features_df = pd.read_parquet('data/processed/training_data.parquet')

# Parse cached odds files
cache_dir = Path('data/raw/betting_lines/cache')
odds_records = []

for odds_file in cache_dir.glob('odds_*.json'):
    with open(odds_file) as f:
        data = json.load(f)

    event_data = data.get('data', {})
    game_time = event_data.get('commence_time')
    home_team = event_data.get('home_team')
    away_team = event_data.get('away_team')

    for bookmaker in event_data.get('bookmakers', []):
        for market in bookmaker.get('markets', []):
            if market.get('key') != 'player_total_saves':
                continue

            # Group outcomes by player
            player_lines = {}
            for outcome in market.get('outcomes', []):
                player = outcome.get('description')
                if not player:
                    continue

                if player not in player_lines:
                    player_lines[player] = {'line': outcome.get('point')}

                if outcome.get('name') == 'Over':
                    player_lines[player]['odds_over'] = outcome.get('price')
                else:
                    player_lines[player]['odds_under'] = outcome.get('price')

            for player, line_data in player_lines.items():
                odds_records.append({
                    'game_date': game_time[:10],
                    'home_team': home_team,
                    'away_team': away_team,
                    'goalie_name': player,
                    'betting_line': line_data.get('line'),
                    'odds_over_decimal': line_data.get('odds_over'),
                    'odds_under_decimal': line_data.get('odds_under'),
                })

odds_df = pd.DataFrame(odds_records)

# Convert decimal to American odds
def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))

odds_df['odds_over_american'] = odds_df['odds_over_decimal'].apply(decimal_to_american)
odds_df['odds_under_american'] = odds_df['odds_under_decimal'].apply(decimal_to_american)

# Merge with features (requires matching goalie names to IDs)
# ... additional matching logic needed ...

# Create target variable
merged_df['over_hit'] = (merged_df['saves'] > merged_df['betting_line']).astype(int)

# Save
merged_df.to_parquet('data/processed/classification_training_data.parquet')
```

---

## Model Training

### Step 3: Train Production Model

```bash
python scripts/train_production_4398.py
```

### Hyperparameters (Config #4398)

```python
CONFIG_4398 = {
    'max_depth': 4,
    'learning_rate': 0.02,
    'min_child_weight': 15,
    'gamma': 1.0,
    'reg_alpha': 10,
    'reg_lambda': 40,
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'use_sample_weights': False
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
    **CONFIG_4398
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True
)
```

### Features Excluded from Training

These columns are removed before training:

```python
excluded_cols = [
    # Metadata
    'game_id', 'goalie_id', 'game_date', 'season', 'team_abbrev', 'opponent_team', 'toi',

    # Target variables
    'over_hit', 'saves', 'line_margin',

    # Odds (used for evaluation, not features)
    'odds_over_american', 'odds_under_american', 'odds_over_decimal', 'odds_under_decimal', 'num_books',

    # Current-game stats (data leakage)
    'shots_against', 'goals_against', 'save_percentage',
    'even_strength_saves', 'even_strength_shots_against', 'even_strength_goals_against',
    'power_play_saves', 'power_play_shots_against', 'power_play_goals_against',
    'short_handed_saves', 'short_handed_shots_against', 'short_handed_goals_against',
    'team_goals', 'team_shots', 'opp_goals', 'opp_shots'
]
```

---

## Model Evaluation

### Profitability Metrics

The model is evaluated on betting profitability, not just accuracy:

```python
def evaluate_profitability(X, y, df, split_idx, ev_threshold=0.02):
    """
    For each game:
    1. Get model probability: prob_over = model.predict_proba(X)[:, 1]
    2. Calculate EV: ev_over = (prob_over * payout) - 1
    3. If ev >= threshold: place bet
    4. Track profit/loss using actual odds
    """
```

### Config #4398 Results

| Metric | Validation | Test | Combined |
|--------|------------|------|----------|
| ROI | +2.54% | +0.62% | +1.60% |
| Win Rate | 54.0% | 52.7% | 53.4% |
| Total Bets | 363 | 336 | 699 |

### EV Threshold Selection

The model was tested at multiple EV thresholds:

| Threshold | Bets | Win Rate | ROI |
|-----------|------|----------|-----|
| 1% | More | Lower | Lower |
| **2%** | **699** | **53.4%** | **+1.60%** |
| 3% | Fewer | Higher | Variable |
| 4% | 581 | Higher | +1.8% |

2% was chosen for balance between volume and profitability.

---

## Model Deployment

### Model Artifacts

After training, save these files:

```
models/trained/config_4398_ev2pct_{timestamp}/
├── classifier_model.json          # XGBoost model in JSON format
├── classifier_feature_names.json  # List of 90 features in exact order
└── classifier_metadata.json       # Hyperparameters and performance metrics
```

### Update Predictor Path

After training a new model, update the default path in `src/betting/predictor.py`:

```python
class BettingPredictor:
    def __init__(
        self,
        model_path='models/trained/config_4398_ev2pct_20260115_103430/classifier_model.json',
        feature_order_path='models/trained/config_4398_ev2pct_20260115_103430/classifier_feature_names.json'
    ):
```

### Making Predictions

```python
from betting import BettingPredictor, BettingFeatureCalculator, NHLBettingData

# Initialize
predictor = BettingPredictor()
feature_calc = BettingFeatureCalculator()
nhl_data = NHLBettingData()

# Get goalie's recent games
recent_games = nhl_data.get_goalie_recent_games(goalie_id, n_games=15)

# Calculate features
features_df = feature_calc.prepare_prediction_features(
    goalie_id=goalie_id,
    team='TOR',
    opponent='BOS',
    is_home=1,
    game_date='2026-01-25',
    recent_games=recent_games,
    betting_line=25.5
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
#     'ev_over': 0.035,
#     'ev_under': -0.02,
#     'recommended_ev': 0.035
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

# 3. (Manual) Add historical betting lines to create classification_training_data.parquet

# 4. Train model
python scripts/train_production_4398.py

# 5. Update predictor paths in src/betting/predictor.py
```

### Daily Operations

```bash
# Fetch lines and generate predictions
python scripts/fetch_and_predict.py

# Update results after games complete
python scripts/update_betting_results.py

# View dashboard
python scripts/betting_dashboard.py
```

### Incremental Retraining

When you have more data:

1. Run data collection for new dates
2. Regenerate features
3. Merge new odds data
4. Retrain with same hyperparameters
5. Compare validation/test ROI to previous model
6. Deploy if improved

---

## Appendix: Directory Structure

```
saves-model-v3/
├── config/
│   └── config.yaml              # Configuration settings
├── data/
│   ├── raw/
│   │   ├── boxscores/           # Raw game data
│   │   └── play_by_play/        # Shot-by-shot data
│   ├── processed/
│   │   ├── training_data.parquet
│   │   └── classification_training_data.parquet
│   ├── cache/
│   │   ├── api_cache.db         # NHL API cache
│   │   └── odds_api/            # The-Odds-API cache
│   └── betting_history/         # CSV backups of tracker
├── models/
│   └── trained/
│       └── config_4398_ev2pct_*/
│           ├── classifier_model.json
│           ├── classifier_feature_names.json
│           └── classifier_metadata.json
├── scripts/
│   ├── collect_historical_data.py
│   ├── create_features.py
│   ├── train_production_4398.py
│   ├── fetch_and_predict.py
│   └── update_betting_results.py
├── src/
│   ├── betting/
│   │   ├── predictor.py
│   │   ├── feature_calculator.py
│   │   ├── odds_fetcher.py
│   │   └── odds_utils.py
│   ├── data/
│   │   ├── api_client.py
│   │   ├── cache_manager.py
│   │   └── collectors.py
│   ├── features/
│   │   └── feature_engineering.py
│   └── models/
│       └── classifier_trainer.py
└── betting_tracker.xlsx          # Excel tracker for daily bets
```
