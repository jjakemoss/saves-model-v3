# Date-Based Sheet Migration

## Overview

The betting tracker has been updated to organize games by date in separate Excel sheets instead of a single "Bets" sheet. This provides better organization and easier navigation.

## New Structure

### Sheet Order
1. **Summary** (first tab) - Performance metrics aggregated across all dates
2. **Settings** (second tab) - Configuration settings
3. **Date sheets** (one per day, newest first) - e.g., "2026-01-05", "2026-01-04", etc.

### Benefits
- Easier to navigate and review specific days
- Cleaner organization as data accumulates over time
- Each date is self-contained on its own sheet
- Summary tab aggregates across all dates

## Migration

Your existing `betting_tracker.xlsx` has been automatically migrated to the new format.

### What Happened
1. **Backup Created**: `betting_tracker_backup_YYYYMMDD_HHMMSS.xlsx` saved
2. **New Structure**: Summary and Settings tabs created at positions 1 and 2
3. **Data Migrated**: All games from the old "Bets" sheet distributed to date-specific sheets
4. **Sheet Ordering**: Date sheets sorted with newest first (after Summary and Settings)

### Migration Script
```bash
python scripts/migrate_tracker_to_date_sheets.py
```

This script:
- Creates a timestamped backup
- Reads all data from the old "Bets" sheet
- Creates new tracker structure
- Distributes games to appropriate date sheets
- Preserves all data (predictions, results, bets, etc.)

## Functionality

### All Scripts Updated

All scripts now work seamlessly with the date-based format:

1. **populate_daily_games.py** - Creates new date sheet if needed, appends games
2. **generate_predictions.py** - Reads from and updates the appropriate date sheet
3. **update_betting_results.py** - Updates results in the appropriate date sheet
4. **betting_dashboard.py** - Aggregates across all date sheets (if exists)

### Under the Hood

The `BettingTracker` class (in `src/betting/excel_manager.py`) now:
- Automatically creates new date sheets as needed
- Maintains correct sheet ordering (newest dates first)
- Routes operations to the correct date sheet
- Aggregates data across sheets for backups and reporting

### Key Methods
- `_get_or_create_date_sheet(date_str)` - Creates date sheet if missing, maintains order
- `append_games(games_df)` - Groups by date, appends to appropriate sheets
- `update_predictions(predictions_df)` - Updates predictions by date
- `update_results(results_df)` - Updates results by date
- `backup_to_csv()` - Combines all date sheets into single CSV

## User Experience

### Daily Workflow
1. Run `populate_daily_games.py` → New date sheet created automatically
2. Enter betting lines in Excel (navigate to today's date tab)
3. Run `generate_predictions.py` → Updates today's date sheet
4. Run `update_betting_results.py` → Updates yesterday's date sheet with results

### Navigation
- Click on specific date tabs to see games for that day
- Summary tab shows aggregate performance
- Settings tab for configuration
- Newest games are in the leftmost date tabs (after Summary and Settings)

## Technical Details

### Column Order (unchanged)
```
game_date, game_id, goalie_name, betting_line, goalie_id,
team_abbrev, opponent_team, is_home, predicted_saves,
prob_over, confidence_pct, confidence_bucket, recommendation,
bet_amount, bet_selection, actual_saves, result,
profit_loss, notes
```

### Date Sheet Creation
- Sheets named in YYYY-MM-DD format (e.g., "2026-01-05")
- Created on-demand when games are populated
- Inserted in correct position (sorted newest first)
- Includes same headers and formatting as before

### Backward Compatibility
- All existing scripts work without modification
- Data preserved during migration
- Backup created before any changes
- Can re-run migration script safely (checks if already migrated)

## Files Modified

### Core Changes
1. `scripts/init_betting_tracker.py` - Creates Summary and Settings tabs first
2. `src/betting/excel_manager.py` - Added date sheet routing logic
3. `scripts/generate_predictions.py` - Includes `game_date` in predictions
4. `scripts/update_betting_results.py` - Includes `game_date` in results

### New Files
- `scripts/migrate_tracker_to_date_sheets.py` - One-time migration script
- `DATE_SHEET_MIGRATION.md` - This documentation

## End Result

The functionality remains exactly the same - you can run all scripts as before. The only difference is the improved organization in Excel with date-based tabs instead of a single growing sheet.
