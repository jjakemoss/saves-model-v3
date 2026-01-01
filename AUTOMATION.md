# Automation Guide

This guide explains how to set up and use the automation scripts for keeping your NHL Goalie Saves model up-to-date.

## Overview

The automation system consists of two main scripts:

1. **Daily Data Update** (`update_daily_data.py`) - Fetches completed games and updates training data
2. **Model Retraining** (`retrain_model.py`) - Retrains model with latest data and deploys if improved

## Daily Data Update

### Purpose

Fetches completed NHL games from yesterday (or a specified date) and appends them to your training dataset.

### Usage

```bash
# Update with yesterday's games (default)
python scripts/update_daily_data.py

# Update with specific date
python scripts/update_daily_data.py --date 2025-01-15

# Use custom config file
python scripts/update_daily_data.py --config config/my_config.yaml
```

### What It Does

1. Fetches completed games for the target date from NHL API
2. Collects boxscore and play-by-play data for each game
3. Extracts goalie performances
4. Runs feature engineering on new data
5. Appends to existing training dataset
6. Logs results to `logs/daily_update.log`

### Scheduling (Windows Task Scheduler)

To run automatically every day at 6 AM:

1. Open Task Scheduler
2. Create Basic Task
3. Name: "NHL Model Daily Update"
4. Trigger: Daily, 6:00 AM
5. Action: Start a program
   - Program: `python`
   - Arguments: `s:\Documents\GitHub\saves-model-v3\scripts\update_daily_data.py`
   - Start in: `s:\Documents\GitHub\saves-model-v3`

### Scheduling (Linux/Mac cron)

Add to crontab (`crontab -e`):

```bash
0 6 * * * cd /path/to/saves-model-v3 && python scripts/update_daily_data.py >> logs/daily_update.log 2>&1
```

This runs every day at 6:00 AM.

## Model Retraining

### Purpose

Retrains the model with the latest data and deploys only if performance improves.

### Usage

```bash
# Retrain and deploy only if improved
python scripts/retrain_model.py

# Force deployment even if worse
python scripts/retrain_model.py --force

# Skip backing up current model
python scripts/retrain_model.py --skip-backup

# Custom config
python scripts/retrain_model.py --config config/my_config.yaml
```

### What It Does

1. Loads all training data (including recent updates)
2. Trains new XGBoost model
3. Evaluates on test set
4. Compares to previous model:
   - RMSE (lower is better)
   - MAE (lower is better)
   - R² (higher is better)
5. If improved (2/3 metrics better):
   - Backs up current model to `models/trained/backups/`
   - Deploys new model
6. If not improved:
   - Skips deployment (unless `--force` used)
   - Logs comparison results

### Scheduling

**Weekly retraining (recommended):**

Windows Task Scheduler:
- Trigger: Weekly, Sunday at 2:00 AM
- Action: `python scripts\retrain_model.py`

Linux/Mac cron:
```bash
0 2 * * 0 cd /path/to/saves-model-v3 && python scripts/retrain_model.py >> logs/retrain.log 2>&1
```

**Monthly retraining (less aggressive):**

Linux/Mac cron:
```bash
0 2 1 * * cd /path/to/saves-model-v3 && python scripts/retrain_model.py >> logs/retrain.log 2>&1
```

## Recommended Automation Setup

### Development/Testing

- **Daily updates**: Manual (run when needed)
- **Retraining**: Manual (after significant data collection)

### Production

- **Daily updates**: Automated, runs at 6:00 AM daily
- **Retraining**: Automated, runs weekly on Sundays at 2:00 AM

### Workflow

```
Daily (6 AM):
  update_daily_data.py
    ↓
  Fetches yesterday's games
    ↓
  Appends to training data

Weekly (Sunday 2 AM):
  retrain_model.py
    ↓
  Trains new model with all data
    ↓
  Compares to current model
    ↓
  Deploys if improved
```

## Monitoring

### Log Files

All automation scripts log to:
- `logs/daily_update.log` - Daily data updates
- `logs/retrain.log` - Model retraining

### Check Logs

```bash
# View recent daily updates
tail -n 50 logs/daily_update.log

# View recent retraining
tail -n 100 logs/retrain.log

# Search for errors
grep -i error logs/*.log
```

### Model Backups

Old models are backed up to `models/trained/backups/` with timestamps:
```
models/trained/backups/
  xgboost_goalie_model_20250115_020530/
    xgboost_goalie_model.pkl
    xgboost_goalie_model_features.json
    xgboost_goalie_model_metadata.json
```

### Metrics Tracking

Each model's `metadata.json` contains:
- Training date
- Test metrics (RMSE, MAE, R²)
- Comparison to previous model
- Feature count

Example:
```json
{
  "model_name": "xgboost_goalie_model",
  "training_date": "2025-01-15T02:05:30",
  "test_metrics": {
    "rmse": 2.539,
    "mae": 1.042,
    "r2": 0.9627
  },
  "comparison_to_previous": {
    "is_improvement": true,
    "rmse_change": -0.123,
    "mae_change": -0.089,
    "r2_change": 0.0015
  }
}
```

## Troubleshooting

### Daily Update Fails

**Symptom:** No data collected for a date

**Possible Causes:**
1. No games scheduled that day (check NHL schedule)
2. Games not yet final (still in progress)
3. API rate limiting (wait and retry)
4. Network issues

**Solution:**
```bash
# Manually run for specific date
python scripts/update_daily_data.py --date 2025-01-15

# Check logs for details
tail -n 100 logs/daily_update.log
```

### Retraining Always Fails

**Symptom:** Model never deploys, always shows "no improvement"

**Possible Causes:**
1. Not enough new data (model hasn't changed significantly)
2. New data is noisy (recent games have unusual patterns)
3. Model has plateaued (reached performance ceiling)

**Solutions:**
```bash
# Force deployment to test
python scripts/retrain_model.py --force

# Check if training data has grown
python -c "import pandas as pd; df = pd.read_parquet('data/processed/training_data.parquet'); print(f'Dataset size: {len(df)} rows')"

# Compare old vs new metrics manually
cat models/trained/xgboost_goalie_model_metadata.json
```

### Disk Space Issues

**Symptom:** Scripts fail with disk space errors

**Solution:**
```bash
# Check disk usage
du -sh data/ models/

# Clean old backups (keep last 5)
ls -t models/trained/backups/ | tail -n +6 | xargs rm -rf

# Clean cache older than 7 days
find data/cache -mtime +7 -delete
```

## Performance Monitoring

### Track Model Drift

Monitor these metrics over time:
- **RMSE trending up** → Model performance degrading
- **RMSE stable** → Model generalizing well
- **RMSE trending down** → Model improving with more data

### Expected Behavior

- **First month:** Model improves as more data collected
- **Months 2-3:** Model stabilizes, improvements plateau
- **Ongoing:** Small fluctuations (+/- 0.1 RMSE) are normal

### When to Intervene

1. **RMSE increases > 0.5:** Investigate data quality or NHL rule changes
2. **R² drops below 0.90:** Model losing predictive power
3. **Consistent deployment failures:** Need to retune hyperparameters

## Advanced Configuration

### Adjust Update Frequency

Edit `config/config.yaml`:

```yaml
data:
  cache_ttl: 43200  # 12 hours (more frequent updates)
```

### Adjust Retraining Threshold

Modify `retrain_model.py` to require larger improvements:

```python
# In compare_performance function
is_improvement = (
    improvements >= 2 and
    abs(rmse_change) > 0.1  # Require at least 0.1 RMSE improvement
)
```

### Notification on Deployment

Add email/Slack notification when model deployed:

```python
# In retrain_model.py, after deploy_model()
if should_deploy:
    send_notification(
        f"New model deployed!\n"
        f"RMSE: {test_metrics['rmse']:.3f}\n"
        f"Improvement: {comparison['rmse_change']:+.3f}"
    )
```

## Summary

### Daily Maintenance
- ✅ Automated data collection keeps model current
- ✅ Feature engineering runs automatically on new data
- ✅ Logs provide visibility into updates

### Weekly Maintenance
- ✅ Automated retraining with performance validation
- ✅ Only deploys if model improves
- ✅ Automatic backups prevent data loss

### Manual Monitoring
- 📊 Check logs weekly for errors
- 📊 Review model performance monthly
- 📊 Clean old backups quarterly
