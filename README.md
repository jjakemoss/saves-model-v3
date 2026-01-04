# NHL Goalie Saves Prediction Model

An automated machine learning pipeline for predicting NHL goalie saves using XGBoost regression. Fetches real-time data from the NHL API and generates predictions with betting recommendations.

## Features

- ✅ **Real-time predictions** using live NHL API data
- ✅ **36 predictive features** including rolling averages, team stats, opponent stats
- ✅ **No data leakage** - proper time-based train/test splits
- ✅ **Interactive CLI** for step-by-step game predictions
- ✅ **Automated updates** - daily data collection and weekly retraining
- ✅ **Model versioning** - automatic backups and performance tracking

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Historical Data

```bash
# Collect 3 seasons of NHL data (~15,000 goalie performances)
python scripts/collect_historical_data.py
```

This will take 1-2 hours with API rate limiting and caching.

### 3. Engineer Features

```bash
# Generate training features from raw data
python scripts/engineer_features.py
```

### 4. Train Model

```bash
# Train XGBoost regression model
python scripts/train_model.py
```

Expected performance:
- RMSE: ~2.5 saves
- MAE: ~1.0 saves
- R²: ~0.96

### 5. Make Predictions

```bash
# Interactive prediction tool
python scripts/predict_games.py
```

Example session:
```
Enter the home team: MIN
Enter the away team: SJ
Enter the home goalie: Filip Gustavsson
Enter the away goalie: Mackenzie Blackwood
Enter betting line for Filip Gustavsson: 27.5
Enter betting line for Mackenzie Blackwood: 32.5

=== Predictions ===
Filip Gustavsson (MIN - Home)
  Predicted Saves: 26.3
  Betting Line: 27.5
  Difference: -1.2 saves
  Recommendation: UNDER

Mackenzie Blackwood (SJ - Away)
  Predicted Saves: 34.1
  Betting Line: 32.5
  Difference: +1.6 saves
  Recommendation: OVER
```

## Project Structure

```
saves-model-v3/
├── config/
│   └── config.yaml              # Configuration settings
├── data/
│   ├── raw/                     # Raw NHL API data
│   ├── processed/               # Training data with features
│   └── cache/                   # API response cache
├── src/
│   ├── data/                    # Data collection modules
│   │   ├── api_client.py       # NHL API wrapper
│   │   ├── cache_manager.py    # SQLite cache
│   │   └── collectors.py       # Game/player data collectors
│   ├── features/                # Feature engineering
│   │   ├── base_features.py    # Basic stats extraction
│   │   ├── rolling_features.py # Rolling averages
│   │   └── feature_engineering.py  # Main pipeline
│   ├── models/                  # Model training & evaluation
│   │   ├── trainer.py          # XGBoost trainer
│   │   ├── evaluator.py        # Metrics & plots
│   │   └── predictor.py        # Prediction interface
│   └── pipeline/                # Real-time prediction
│       └── realtime_features.py # Live feature collection
├── scripts/
│   ├── collect_historical_data.py  # Initial data collection
│   ├── engineer_features.py        # Feature generation
│   ├── train_model.py              # Model training
│   ├── predict_games.py            # Interactive predictions
│   ├── update_daily_data.py        # Daily automation
│   └── retrain_model.py            # Weekly retraining
├── models/
│   └── trained/                 # Saved models & metadata
├── logs/                        # Application logs
└── README.md                    # NHL API documentation
```

## How It Works

### Data Collection

1. **Historical Data**: Collects 3 seasons (2022-2025) from NHL API
   - Game schedules (15,000+ games)
   - Boxscores (saves, shots, goalies)
   - Play-by-play (shot locations, situations)
   - Goalie game logs
   - Team stats

2. **Real-time Data**: For predictions, fetches:
   - Last 15 games for each goalie
   - Current season team stats
   - Current season opponent stats

### Feature Engineering

**36 predictive features** across 7 categories:

1. **Rolling Averages** (18 features)
   - Saves, save %, shots against (3, 5, 10 game windows)
   - Situational save % (even strength, power play)

2. **Opponent Stats** (4 features)
   - Shots/game, goals/game
   - Power play opportunities & goals

3. **Team Stats** (11 features)
   - Shots, goals, shooting %
   - Power play %, faceoff win %
   - Blocked shots, hits, penalties

4. **Contextual** (3 features)
   - Home/away indicator
   - Starter status
   - Season trends

**Data Leakage Prevention:**
- Rolling features exclude current game (`.shift(1)`)
- Time-based train/test split (no random shuffle)
- Current-game outcomes excluded from features
- Only "knowable before game" features used

### Model Architecture

**XGBoost Regressor** (not classification):
- Predicts actual saves (e.g., 27.3 saves)
- Compare to betting line at prediction time
- Output: OVER/UNDER recommendation

**Hyperparameters:**
```python
{
    'objective': 'reg:squarederror',
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

**Training Split:**
- Training: 70% (oldest games)
- Validation: 15% (middle games)
- Test: 15% (most recent games)

### Prediction Pipeline

```
User Input (teams, goalies, betting lines)
    ↓
Look up goalie IDs from team rosters
    ↓
Fetch last 15 games for each goalie
    ↓
Calculate rolling averages (3, 5, 10 games)
    ↓
Fetch team/opponent season stats
    ↓
Generate 36 features per goalie
    ↓
Load trained XGBoost model
    ↓
Predict saves for both goalies
    ↓
Compare to betting lines
    ↓
Output: OVER/UNDER recommendations
```

## Automation

### Daily Data Updates

Fetches yesterday's completed games and updates training data.

```bash
# Manual run
python scripts/update_daily_data.py

# Automated (6 AM daily via Task Scheduler/cron)
0 6 * * * cd /path/to/saves-model-v3 && python scripts/update_daily_data.py
```

### Weekly Retraining

Retrains model with latest data, deploys only if improved.

```bash
# Manual run
python scripts/retrain_model.py

# Automated (Sunday 2 AM via Task Scheduler/cron)
0 2 * * 0 cd /path/to/saves-model-v3 && python scripts/retrain_model.py
```

See [AUTOMATION.md](AUTOMATION.md) for detailed setup guide.

## Model Performance

### Current Results

- **RMSE**: 2.539 saves (predicts within ~2.5 saves on average)
- **MAE**: 1.042 saves (typical error is ~1 save)
- **R²**: 0.9627 (explains 96.3% of variance)

### Interpretation

- **RMSE of 2.5** means predictions are typically within +/- 2.5 saves
- For a betting line of 27.5:
  - Prediction of 30+ → OVER (confident)
  - Prediction of 25- → UNDER (confident)
  - Prediction of 26-29 → NO BET (too close)

### Top Predictive Features

1. `shots_against_rolling_10` (54% importance) - Opponent shot volume
2. `save_percentage_rolling_5` (12%) - Recent goalie form
3. `opp_shots` (8%) - Opponent offensive tendency
4. `saves_rolling_10` (6%) - Goalie workload trend
5. `is_home` (5%) - Home ice advantage

## Configuration

Edit [config/config.yaml](config/config.yaml):

```yaml
api:
  rate_limit: 10  # Requests per second
  timeout: 30

data:
  seasons: ["20222023", "20232024", "20242025"]
  game_type: 2  # Regular season

features:
  rolling_windows: [3, 5, 10]
  min_games_for_rolling: 3

model:
  random_state: 42
  test_size: 0.15
  validation_size: 0.15
  target_rmse: 5.0  # Acceptable RMSE threshold
```

## Troubleshooting

### "Model not found" error

Train the model first:
```bash
python scripts/train_model.py
```

### "Training data not found" error

Run feature engineering:
```bash
python scripts/engineer_features.py
```

### "No raw data" error

Collect historical data:
```bash
python scripts/collect_historical_data.py
```

### Goalie not found in roster

- Check spelling (case-insensitive)
- Verify team abbreviation (MIN, TOR, BOS, etc.)
- Ensure goalie is on current roster

### API rate limiting errors

- Default limit: 10 requests/second
- Increase cache TTL in config to reduce API calls
- Wait and retry

## Data Privacy & Ethics

- All data sourced from public NHL API
- No personal information collected
- Model for educational/entertainment purposes
- Use responsibly - gambling has risks

## Technical Details

### Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **xgboost**: Gradient boosting model
- **scikit-learn**: Metrics and preprocessing
- **requests**: API client
- **pyyaml**: Configuration management

### Data Storage

- **Raw data**: Parquet format (~500 MB for 3 seasons)
- **Processed data**: Parquet with features (~50 MB)
- **Cache**: SQLite database (self-cleaning, 24hr TTL)
- **Models**: Pickle format (~10 MB)

### Performance

- **Data collection**: 1-2 hours (one-time)
- **Feature engineering**: 2-3 minutes
- **Model training**: 30-60 seconds
- **Prediction**: < 2 seconds per game

## Future Enhancements

Potential improvements:
- [ ] Ensemble models (XGBoost + Random Forest)
- [ ] Expected goals (xG) features from shot locations
- [ ] Injury/lineup data integration
- [ ] REST API for predictions
- [ ] Web dashboard for monitoring
- [ ] Confidence intervals on predictions
- [ ] Goalie fatigue modeling (back-to-back games)

## Contributing

This is a personal project, but suggestions welcome!

## License

Educational/personal use. NHL data belongs to the NHL.

## Acknowledgments

- NHL for public API access
- Community NHL API documentation
- XGBoost team for excellent ML library

## Contact

For questions or issues, open a GitHub issue.

---

**Disclaimer**: This model is for educational and entertainment purposes only. Past performance does not guarantee future results. Please gamble responsibly.
