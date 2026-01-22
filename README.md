# NHL Goalie Saves Prediction Model

An XGBoost classifier that predicts NHL goalie saves over/under betting lines. Automatically fetches lines from Underdog Fantasy and calculates expected value (EV) for betting recommendations.

## Features

- **Automated line fetching** from Underdog Fantasy API
- **90 predictive features** including rolling averages, team stats, opponent stats
- **EV-based recommendations** with configurable threshold (default 2%)
- **Excel-based betting tracker** for tracking bets and performance
- **GitHub Actions** for running predictions on-demand

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
2. Fetch goalie saves lines from Underdog
3. Match lines to games and look up goalie IDs
4. Generate predictions with EV calculations
5. Display all lines sorted by EV

Example output:
```
ALL LINES FOR 2026-01-21:
----------------------------------------------------------------------
  Dostal       (ANA) @ Underdog   | Line: 27.5  (O:-125 /U:+102 ) | Pred: 27.3  | P(Over): 46.5%  | UNDER  | EV: +4.0%
  Wedgewood    (COL) @ Underdog   | Line: 23.5  (O:-107 /U:-115 ) | Pred: 23.7  | P(Over): 53.6%  | NO BET | EV: +1.9%
  ...
```

## Scripts

| Script | Description |
|--------|-------------|
| `fetch_and_predict.py` | Main script - fetches lines from Underdog and generates predictions |
| `add_manual_lines.py` | Add blank rows for manual line entry from other sportsbooks |
| `generate_predictions.py` | Generate predictions for rows with lines but no predictions |
| `update_betting_results.py` | Update actual saves and P/L after games complete |
| `betting_dashboard.py` | Display performance metrics and update Summary sheet |
| `init_betting_tracker.py` | Create a new betting tracker Excel file |

## Usage

### Daily Workflow

```bash
# Fetch lines and predictions (run multiple times as lines update)
python scripts/fetch_and_predict.py --verbose

# After games complete, update results
python scripts/update_betting_results.py

# View performance
python scripts/betting_dashboard.py
```

### Adding Lines from Other Sportsbooks

```bash
# Add blank rows for a book (e.g., PrizePicks)
python scripts/add_manual_lines.py --book "PrizePicks"

# Fill in lines manually in Excel, then generate predictions
python scripts/generate_predictions.py
```

### GitHub Actions

Run predictions remotely via GitHub Actions:
1. Go to Actions tab
2. Select "Fetch Predictions" workflow
3. Click "Run workflow"

## Project Structure

```
saves-model-v3/
├── .github/workflows/
│   └── fetch_predictions.yml    # GitHub Action for predictions
├── models/trained/
│   └── config_4398_ev2pct_.../  # Active model (90 features)
├── src/
│   ├── betting/                 # Betting module
│   │   ├── predictor.py        # XGBoost prediction interface
│   │   ├── feature_calculator.py # Real-time feature calculation
│   │   ├── excel_manager.py    # Excel tracker management
│   │   ├── nhl_fetcher.py      # NHL API data fetching
│   │   └── odds_fetcher.py     # Underdog API fetching
│   └── data/
│       └── api_client.py       # NHL API client
├── scripts/                     # CLI scripts (see table above)
├── betting_tracker.xlsx         # Excel tracker (created by init script)
└── requirements.txt
```

## Model Details

### Architecture

- **Model**: XGBoost Classifier
- **Features**: 90 predictive features
- **Output**: Probability of going OVER the betting line
- **Recommendation Logic**:
  - Calculate EV for both OVER and UNDER based on odds
  - Recommend if EV >= 2% threshold
  - Otherwise, NO BET

### Feature Categories

1. **Rolling Averages** - Saves, shots against, save % (3, 5, 10, 15 game windows)
2. **Opponent Stats** - Shots/game, goals/game, shooting %
3. **Team Stats** - Goals for, shots, power play %
4. **Contextual** - Home/away, rest days, season trends
5. **Line-based** - Betting line relative to recent performance

### EV Calculation

```
implied_prob = american_odds_to_prob(odds)
edge = model_prob - implied_prob
ev = edge * potential_profit
```

## Betting Tracker

The Excel tracker (`betting_tracker.xlsx`) contains:

- **Summary** - Overall performance metrics
- **Settings** - Configuration
- **Date sheets** - One sheet per date with all lines and predictions

Columns tracked:
- Game info (date, teams, goalie)
- Book and line info (book, line, over/under odds)
- Predictions (predicted saves, prob_over, recommendation, EV)
- Results (actual saves, result, profit/loss)

## Configuration

Key settings in `src/betting/predictor.py`:
- Model path: `models/trained/config_4398_ev2pct_20260115_103430/`
- EV threshold: 2% (adjustable)

## Requirements

- Python 3.12+
- pandas, numpy, xgboost, scikit-learn
- requests (for API calls)
- openpyxl (for Excel handling)

## Disclaimer

This model is for educational and entertainment purposes only. Past performance does not guarantee future results. Please gamble responsibly.
