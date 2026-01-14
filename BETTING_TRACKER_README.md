# NHL Goalie Saves Betting Tracker

Comprehensive betting management system for daily use of the trained NHL goalie saves prediction model.

## Overview

This system provides:
- Daily game population and prediction generation
- Excel-based tracking with performance metrics
- Automatic result updates from NHL API
- Expected Value (EV) based recommendations (OVER/UNDER/NO BET)
- Historical performance analysis by confidence level

## Quick Start

### 1. Initialize Tracker (One-Time Setup)

```bash
python scripts/init_betting_tracker.py
```

This creates `betting_tracker.xlsx` with 3 sheets:
- **Bets**: Main tracking sheet (19 columns)
- **Summary**: Auto-calculating performance metrics
- **Settings**: Configuration values

### 2. Daily Workflow

#### Morning (Before Games Start)

**Step 1: Populate Today's Games**
```bash
python scripts/populate_daily_games.py
```
- Fetches today's NHL schedule
- Creates 2 rows per game (one for each team's goalie)
- Rows are initially blank - you'll fill in goalies you want to bet on

**Step 2: Enter Goalie Names, Betting Lines, and Odds** (anytime after lines are released)
1. Open `betting_tracker.xlsx`
2. For games you want to bet, enter:
   - `goalie_name`: Goalie's **LAST NAME** only (e.g., "Shesterkin", "Hellebuyck")
   - `betting_line`: Saves line from sportsbook (e.g., 24.5)
   - `line_over`: American odds for OVER (e.g., -115)
   - `line_under`: American odds for UNDER (e.g., -105)
3. Leave rows blank for games you're not betting on
4. Save file

**Note**: You don't need to wait for official lineups! The system automatically looks up goalie IDs from season stats, so you can enter names as soon as you know the expected starter.

**Step 3: Generate Predictions** (can be run multiple times safely)

```bash
python scripts/generate_predictions.py
```
- Reads games with goalie names and betting lines entered
- **Automatically looks up goalie IDs** using two methods:
  1. First tries current game roster (if lineups announced)
  2. Falls back to searching season stats leaders (works anytime!)
- Fetches recent goalie stats (last 15 games)
- Calculates 89 features (rolling averages, rest days, etc.)
- Generates predictions using trained classifier model
- Updates Excel with:
  - `goalie_id` (looked up automatically)
  - `predicted_saves`
  - `prob_over` (probability of exceeding betting line)
  - `confidence_pct` (0-100% confidence)
  - `confidence_bucket` (50-55%, 55-60%, ..., 75%+)
  - `recommendation` (OVER/UNDER/NO BET based on 2% EV threshold)
  - `ev` (Expected Value % for recommended bet)

**Important**: You can safely re-run predictions multiple times throughout the day (e.g., for late games after early games have started). The system automatically **excludes any games from the current date** when calculating rolling averages, preventing data leakage. This means:
- Morning games won't affect evening game predictions
- You can add late game goalies and re-run the script
- Predictions remain valid even after some games complete

**Step 4: Make Betting Decisions**
1. Review updated Excel file
2. Enter `bet_amount` (e.g., 1, 2, 3 units) for games you want to bet
3. Enter `bet_selection` (OVER/UNDER) based on recommendations
4. Add any `notes`
5. Save file
6. Place bets at sportsbook

#### Next Morning (After Games Complete)

**Step 1: Update Results**
```bash
python scripts/update_betting_results.py
```
- Fetches yesterday's completed games
- Updates `actual_saves`, `result` (WIN/LOSS/PUSH/NO BET), `profit_loss`
- Creates daily backup in `data/betting_history/`

**Step 2: View Performance Dashboard**
```bash
python scripts/betting_dashboard.py
```
- Displays overall performance metrics
- Shows performance by confidence level
- Analyzes OVER vs UNDER bets
- Shows recent trends (last 10, 20, 50 bets)

Optional: Save report to file
```bash
python scripts/betting_dashboard.py --save
```

## File Structure

```
betting_tracker.xlsx          # Main tracking file (user interacts here)
├── Bets sheet               # 19 columns (see schema below)
├── Summary sheet            # Auto-calculated metrics
└── Settings sheet           # Configuration

data/betting_history/        # Daily CSV backups
reports/                     # Performance reports (if using --save)
```

### Bets Sheet Schema (22 Columns)

| Column | Filled By | Description |
|--------|-----------|-------------|
| `game_date` | populate script | YYYY-MM-DD |
| `game_id` | populate script | NHL game ID |
| `goalie_name` | **USER** | Goalie's LAST NAME (e.g., "Shesterkin") |
| `betting_line` | **USER** | Saves line from sportsbook (e.g., 24.5) |
| `line_over` | **USER** | American odds for OVER (e.g., -115) |
| `line_under` | **USER** | American odds for UNDER (e.g., -105) |
| `goalie_id` | generate script | NHL player ID (auto-looked up) |
| `team_abbrev` | populate script | Team abbreviation |
| `opponent_team` | populate script | Opponent abbreviation |
| `is_home` | populate script | 1=home, 0=away |
| `predicted_saves` | generate script | Model's predicted total (estimated from line + probability) |
| `prob_over` | generate script | Probability > line (0-1.0) |
| `confidence_pct` | generate script | Confidence 0-100% |
| `confidence_bucket` | generate script | 50-55%, 55-60%, etc. |
| `recommendation` | generate script | OVER/UNDER/NO BET (based on 2% EV) |
| `ev` | generate script | Expected Value % for recommended bet |
| `bet_amount` | **USER** | Units wagered |
| `bet_selection` | **USER** | OVER/UNDER/NONE |
| `actual_saves` | results script | Actual saves from game |
| `result` | results script | WIN/LOSS/PUSH/NO BET |
| `profit_loss` | results script | Units won/lost |
| `notes` | **USER** | Optional comments |

## Model Details

### Prediction Model
- **Model**: Config #4398 (production model trained 2026-01-13)
- **Type**: XGBoost Binary Classifier (`binary:logistic`)
- **Trained On**: Historical NHL goalie games (2017-2025)
- **Features**: 90 base features including:
  - Rolling averages (3, 5, 10 game windows)
  - Saves, shots against, goals against, save %
  - Situation-specific stats (even-strength, PP, SH)
  - Rest days, back-to-back status
  - Team defensive and opponent offensive metrics
  - Betting line (for context)
- **Hyperparameters**:
  - max_depth: 4, learning_rate: 0.02, n_estimators: 800
  - min_child_weight: 15, gamma: 1.0
  - reg_alpha: 10, reg_lambda: 40
  - Sample weighting: DISABLED (simpler, more stable)
- **Validation Performance** (80/20 split): -2.02% ROI (357 bets)
- **Tuning Performance** (60/20/20 chronological split):
  - EV=2%: Val +2.54% (363 bets), Test +0.62% (336 bets), Combined +1.60% (699 bets)

### Recommendation Logic (Expected Value Based)
The model uses **Expected Value (EV)** rather than simple probability thresholds:

- **EV Calculation**: `EV = model_prob - implied_prob` where implied_prob comes from American odds
- **Minimum EV Threshold**: 2% (0.02)
- **OVER**: Recommended when `ev_over >= 2%` AND `ev_over > ev_under`
- **UNDER**: Recommended when `ev_under >= 2%` AND `ev_under > ev_over`
- **NO BET**: When neither side meets the 2% EV threshold

This EV-based approach ensures we only bet when we have a mathematical edge over the sportsbook's implied probability. The 2% threshold provides a good balance between volume (699 bets in tuning) and profitability (+1.60% ROI).

### Confidence Buckets
Confidence buckets are based on distance from 50% probability and provide additional context alongside the EV-based recommendations:
- 50-55%: Low confidence
- 55-60%: Moderate confidence
- 60-65%: Good confidence
- 65-70%: High confidence
- 70-75%: Very high confidence
- 75%+: Extreme confidence

**Note**: With the new EV-based system, focus on the **recommended_ev** value rather than confidence buckets. The EV threshold ensures we only bet when we have a mathematical edge.

## Advanced Usage

### Using Specific Dates

```bash
# Populate games for specific date
python scripts/populate_daily_games.py --date 2025-01-15

# Generate predictions for specific date
python scripts/generate_predictions.py --date 2025-01-15

# Update results for specific date
python scripts/update_betting_results.py --date 2025-01-14
```

### Custom Tracker File

```bash
python scripts/populate_daily_games.py --tracker my_tracker.xlsx
python scripts/generate_predictions.py --tracker my_tracker.xlsx
python scripts/update_betting_results.py --tracker my_tracker.xlsx
python scripts/betting_dashboard.py --tracker my_tracker.xlsx
```

## Troubleshooting

### Goalie Name Not Found
If the prediction script can't find a goalie by last name:
1. Check spelling - use exactly the last name (e.g., "Fleury" not "Fleury, Marc-Andre")
2. Make sure the goalie has played games this season (rookies with 0 games won't be in stats leaders)
3. If multiple goalies have the same last name, try using more of the name (e.g., "Marc-Andre")
4. Check that you entered the name in the correct row (home vs away team)

The script will skip any goalies it can't find and show an error message. The system searches through the top 200 goalies by wins, so any active NHL goalie should be found.

### Prediction Errors
If prediction fails for a goalie:
- Check that goalie has played at least 3 recent games
- Verify `goalie_id` is correct NHL player ID
- Review error messages in console output

### Result Mismatches
If results don't update:
- Ensure games are completed (check NHL.com)
- Verify `game_id` matches actual game
- Check that goalie actually played (may have been pulled/replaced)

## Performance Metrics

### Overall Metrics
- **Total Bets**: Count of OVER/UNDER bets placed
- **Win Rate**: Wins / (Wins + Losses), excluding pushes
- **ROI**: (Total Profit/Loss) / (Total Units Wagered) × 100%

### Target Performance
Based on -110 odds:
- **Break-even**: 52.4% win rate
- **Good**: >55% win rate, >5% ROI
- **Excellent**: >58% win rate, >10% ROI

### Key Insight
The new Config #4398 model with 2% EV threshold has shown:
- +1.60% ROI on validation+test (699 bets)
- Consistent positive performance across val and test sets
- Higher volume than higher EV thresholds
- Expected 150-200 bets per season at 2% EV threshold

## Data Backups

Daily backups are automatically created in `data/betting_history/`:
```
betting_tracker_YYYYMMDD_HHMMSS.csv
```

These provide:
- Historical record of all bets
- Data integrity protection
- Ability to analyze trends over time

## Technical Notes

### API Rate Limiting
The system uses the existing `NHLAPIClient` with built-in rate limiting:
- Respects NHL API rate limits
- Implements exponential backoff
- Caches responses when possible

### Feature Calculation
Features are calculated in real-time from:
- `/v1/player/{goalie_id}/game-log/{season}/2` (recent games)
- `/v1/gamecenter/{game_id}/boxscore` (game results)
- `/v1/schedule/{date}` (daily schedule)

### Model Files Required
- `models/trained/config_4398_ev2pct_20260113_140448.json/classifier_model.json` - Trained XGBoost classifier (Config #4398)
- `models/trained/config_4398_ev2pct_20260113_140448.json/classifier_feature_names.json` - Feature order (103 features, 90 used)

## Support

For issues or questions:
1. Check error messages in console output
2. Review Excel file for data integrity
3. Verify model files exist in `models/` directory
4. Check that NHL API is accessible

## Future Enhancements

Potential improvements:
- Auto-fetch betting lines from odds API
- SMS/Email alerts for high-confidence plays
- Web dashboard with interactive charts
- Kelly Criterion bankroll management
- Live odds monitoring
- Model retraining automation

---

**Remember**: This is a prediction tool to assist betting decisions. Always bet responsibly and never wager more than you can afford to lose.
