# NHL Goalie Saves Prediction - Quick Start Guide

Get up and running with the NHL Goalie Saves prediction model in minutes.

## Project Overview

This model predicts the number of saves an NHL goalie will make in a game using:
- **Machine Learning**: XGBoost regression trained on 3 seasons of NHL data
- **36 Predictive Features**: Rolling averages, team stats, opponent stats
- **Real-time Predictions**: Fetches live data from NHL API
- **No Data Leakage**: Proper time-based splits and feature engineering

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## First Time Setup (3 Steps)

### Step 1: Collect Historical Data (~1-2 hours)

```bash
python scripts/collect_historical_data.py
```

Fetches 3 seasons of NHL data from the API. You only need to do this once.

### Step 2: Generate Features (~3 minutes)

```bash
python scripts/engineer_features.py
```

Creates training dataset with 36 features per goalie performance.

### Step 3: Train Model (~1 minute)

```bash
python scripts/train_model.py
```

Trains XGBoost regression model and evaluates performance.

Expected performance:
- RMSE: ~2.5 saves
- MAE: ~1.0 saves
- R²: ~0.96

## Making Predictions

```bash
python scripts/predict_games.py
```

The script will prompt you interactively:

```
Enter the home team: MIN
Enter the away team: CHI
Enter the home goalie: Marc-Andre Fleury
Enter the away goalie: Petr Mrazek
Enter betting line for Marc-Andre Fleury: 28.5
Enter betting line for Petr Mrazek: 26.5
```

Output:
```
=== MIN vs CHI - GOALIE SAVES PREDICTION ===

Marc-Andre Fleury (MIN - Home)
  Predicted Saves: 27.3
  Betting Line: 28.5
  Difference: -1.2 saves
  Recommendation: UNDER

Petr Mrazek (CHI - Away)
  Predicted Saves: 29.1
  Betting Line: 26.5
  Difference: +2.6 saves
  Recommendation: OVER
```

## Common Commands

### Update with Latest Games

```bash
# Fetch yesterday's games
python scripts/update_daily_data.py

# Fetch specific date
python scripts/update_daily_data.py --date 2025-01-15
```

### Retrain Model

```bash
# Retrain with latest data (deploys if improved)
python scripts/retrain_model.py

# Force deployment even if worse
python scripts/retrain_model.py --force
```

### Check Model Performance

```bash
# View metadata
cat models/trained/xgboost_goalie_model_metadata.json

# View feature importance
head -20 models/trained/xgboost_goalie_model_feature_importance.csv
```

### View Logs

```bash
# Training logs
tail -50 logs/train.log

# Daily update logs
tail -50 logs/daily_update.log

# Retraining logs
tail -50 logs/retrain.log
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

## Interpreting Predictions

### Confidence Levels

Based on difference between prediction and line:

- **High confidence (±2+ saves)**
  - Example: Predicted 30, Line 27.5 → OVER (+2.5)
  - Recommendation: Strong bet

- **Medium confidence (±1-2 saves)**
  - Example: Predicted 28.3, Line 27.5 → OVER (+0.8)
  - Recommendation: Moderate bet

- **Low confidence (< ±1 save)**
  - Example: Predicted 27.8, Line 27.5 → OVER (+0.3)
  - Recommendation: NO BET (too close)

### Model Accuracy

- **RMSE: 2.5 saves** - Average error is ±2.5 saves
- **MAE: 1.0 save** - Typical error is ±1 save
- **R²: 0.96** - Explains 96% of variance

This means:
- If prediction is 28.0, actual likely between 25.5-30.5
- Model is most accurate when difference > 1.5 saves
- Avoid betting when prediction within ±1 save of line

## Next Steps

1. ✅ Set up daily automation ([AUTOMATION.md](AUTOMATION.md))
2. ✅ Make your first predictions
3. ✅ Track accuracy over time
4. ✅ Refine betting strategy based on results

## Need Help?

- **Full documentation**: [PROJECT_README.md](PROJECT_README.md)
- **Automation guide**: [AUTOMATION.md](AUTOMATION.md)
- **NHL API docs**: [README.md](README.md)

---

**Remember:** This model is for educational purposes. Gamble responsibly!
