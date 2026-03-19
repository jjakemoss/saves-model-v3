Parse one or more lines from the `fetch_and_predict --verbose` output and add them to `betting_tracker.xlsx`, with all prediction columns pre-filled.

Each input line follows this format (leading whitespace ignored):
```
  Oettinger    (DAL vs COL) @ Underdog   | Line: 27.5  (O:+100 /U:-124 ) | Pred: 25.8  | P(Over): 15.6%  | UNDER  | EV: +29.0%
```

**Input to process:**
$ARGUMENTS

---

**Steps:**

1. Parse every line from the input. For each line extract:
   - `goalie_name` — first token (e.g. `Oettinger`)
   - `team` — inside parentheses before `vs` (e.g. `DAL`)
   - `opponent` — inside parentheses after `vs` (e.g. `COL`)
   - `book` — after `@` and before `|` (e.g. `Underdog`)
   - `line` — float after `Line:` and before `(` (e.g. `27.5`)
   - `line_over` — integer inside `(O:...` stripping `+` (e.g. `100`)
   - `line_under` — integer inside `/U:...` stripping `+` (e.g. `-124`)
   - `predicted_saves` — float after `Pred:` (e.g. `25.8`); use `None` if `N/A`
   - `prob_over` — float after `P(Over):` divided by 100 (e.g. `0.156`); use `None` if `N/A`
   - `recommendation` — token after the last `|` before `|` containing `EV:` — one of `OVER`, `UNDER`, or empty
   - `ev` — float after `EV:` divided by 100 (e.g. `0.290`); use `None` if `N/A`
   - Skip any line tagged `[OUTDATED]` at the end

2. Write and run a Python script (using the `.venv` interpreter at `.venv/Scripts/python.exe`) that:
   - Adds `src` to `sys.path`
   - Imports `NHLBettingData` and `BettingTracker` from `betting`
   - Gets today's date with `datetime.now().strftime('%Y-%m-%d')`
   - Fetches today's schedule via `nhl_data.get_todays_games(date)` to resolve `game_id` and `is_home` for each parsed team/opponent pair
   - Looks up each `goalie_id` via `nhl_data.get_goalie_id_by_name(goalie_name)`
   - Checks for duplicates against the existing sheet before inserting: skip a row if a row already exists with the same `game_id` + `goalie_name` (case-insensitive) + `book` + `betting_line`
   - Builds a `pd.DataFrame` of new rows with all columns populated including prediction columns: `predicted_saves`, `prob_over`, `recommendation`, `ev`; leave `confidence_pct`, `confidence_bucket`, `bet_amount`, `bet_selection`, `actual_saves`, `result`, `profit_loss`, `notes` empty (or `NONE` for `bet_selection`)
   - Calls `tracker.append_games(df)` — note this method only writes the base game columns, so after appending you must also call `tracker.update_predictions(predictions_df)` with a DataFrame containing `game_id`, `game_date`, `goalie_id`, `goalie_name`, `book`, `betting_line`, `line_over`, `line_under`, `predicted_saves`, `prob_over`, `recommendation`, `recommended_ev` to fill in the prediction columns on the newly written rows

3. Print what was added and what was skipped as duplicates.

4. If any goalie_id or game_id cannot be resolved, print a clear warning for that row and skip it rather than erroring out.
