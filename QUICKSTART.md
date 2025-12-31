# NHL Goalie Expected Saves Model - Quick Start Guide

This guide will help you get started with the NHL Goalie Expected Saves prediction model.

## Project Overview

This model predicts the probability that an NHL goalie exceeds their betting line for total saves in a game using:
- **Machine Learning**: XGBoost classifier trained on historical data
- **Advanced Hockey Analytics**: Expected Goals (xG), rebound control, shot quality
- **60-80 Optimized Features**: Exponentially weighted averages, interaction effects

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Configuration

Check `config/config.yaml` for API settings and paths.

## Data Collection

### Collect Historical Data (2-3 seasons)

This will take approximately **1-2 hours** depending on your internet connection:

```bash
python scripts/collect_historical_data.py
```

The script will:
1. Collect schedules for all 32 NHL teams
2. Download boxscores and play-by-play for ~4,000 games
3. Gather goalie game logs and NHL Edge stats
4. Cache API responses to reduce future load

**Options:**
```bash
# Collect specific seasons
python scripts/collect_historical_data.py --seasons 20222023 20232024

# Collect only schedules (fast)
python scripts/collect_historical_data.py --schedules-only

# Collect only games
python scripts/collect_historical_data.py --games-only

# Collect only goalie stats
python scripts/collect_historical_data.py --goalies-only
```

### Expected Data Volume

- ~4,000 games (3 seasons × ~1,350 games/season)
- ~100 unique goalies
- ~500 MB raw JSON data
- ~50 MB processed parquet files

## Next Steps

After data collection completes:

### 2. Feature Engineering (Coming Next)

```bash
python scripts/engineer_features.py
```

This will generate ~60-80 optimized features per goalie-game including:
- Exponentially weighted recent performance
- Expected goals against (xG)
- Shot quality and rebound control metrics
- Opponent shooting skill and game state tendencies
- Interaction features (Defense × Offense, Form × Rest)

### 3. Model Training (Coming Next)

```bash
python scripts/train_model.py
```

Trains XGBoost model with:
- Time-based train/validation/test split
- Hyperparameter tuning
- Feature selection
- Calibration analysis

### 4. Make Predictions (Coming Next)

```bash
python scripts/predict_games.py \
  --home MIN --away SJ \
  --home-goalie "Filip Gustavsson" \
  --away-goalie "Mackenzie Blackwood" \
  --home-line 27.5 --away-line 32.5
```

## Project Structure

```
saves-model-v3/
├── config/                 # Configuration files
│   ├── config.yaml         # API and data settings
│   ├── feature_config.yaml # Feature definitions
│   └── model_config.yaml   # Model hyperparameters
├── data/
│   ├── raw/                # Raw API data (gitignored)
│   ├── processed/          # Processed training data
│   └── cache/              # API response cache
├── src/
│   ├── data/               # Data collection modules
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and prediction
│   └── utils/              # Utilities
├── models/
│   └── trained/            # Saved models (gitignored)
├── scripts/                # Executable scripts
└── notebooks/              # Jupyter notebooks for analysis
```

## Key Features

### Hockey Analytics Improvements

1. **Expected Goals (xG)** - More predictive than Corsi/Fenwick
2. **Exponential Weighted Averages** - Recent games matter more
3. **Rebound Control Metrics** - High rebounds = more saves
4. **Opponent Shooting Skill** - Separates volume from conversion
5. **Goalie Home/Away Splits** - Individual performance context
6. **Interaction Features** - Capture synergies (Defense × Offense)
7. **Volatility Metrics** - Save consistency and over-line frequency
8. **Travel & Fatigue** - Multi-game road trips, consecutive starts
9. **Game State Tendencies** - Teams that trail generate more shots
10. **Defensive System Metrics** - Shot distance indicates play style

### No Data Leakage

- **Critical**: Features for game N only use data from games 1 to N-1
- Rolling averages exclude the current game being predicted
- Time-based train/test split (no random shuffle)
- All features "knowable" before game starts

## Troubleshooting

### API Rate Limiting

The client automatically handles rate limiting (10 req/sec). If you encounter errors:
- Check `logs/data_collection.log`
- Increase `retry_delay` in `config/config.yaml`
- The cache will prevent redundant requests

### Missing Data

Some games or players may not have complete data:
- This is normal for recent/ongoing seasons
- The model handles missing values with season averages
- Check `data/processed/collection_summary.json` for statistics

### Cache Issues

Clear the cache if you encounter stale data:
```python
from src.data.cache_manager import CacheManager
cache = CacheManager()
cache.clear_all()
```

## Expected Model Performance

**Target Metrics:**
- Log Loss: < 0.58 (vs 0.62-0.65 without hockey analytics improvements)
- ROC-AUC: > 0.62 (vs 0.56-0.58 baseline)
- Calibration: Predicted probabilities within 5% of actual frequencies

**Estimated Improvement:** 20-30% better than traditional approaches

## Support

For issues or questions:
- Check the main [README.md](README.md) for NHL API documentation
- Review implementation plan: `.claude/plans/fluffy-exploring-narwhal.md`
- Submit issues on GitHub

## License

MIT License - See [LICENSE](LICENSE) file
