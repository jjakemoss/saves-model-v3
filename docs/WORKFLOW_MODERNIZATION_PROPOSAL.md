# Betting Tracker Workflow Modernization — Proposal

Status: **Implemented.** `data/betting.db` is the live source of truth, `betting_tracker.xlsx` is regenerated from it after every write, and all three GitHub Actions (`fetch_predictions.yml`, `record_bet.yml`, `update_results.yml`) commit directly to `main` -- no PR-based workflows remain. See the README's Betting Tracker section for the current day-to-day usage.

## 1. Why this exists

The current system works, but the day-to-day loop has two structural problems:

1. **`betting_tracker.xlsx` is a binary file.** GitHub's mobile web/app can view and merge it, but cannot edit it. That forces a hard split between "see recommendations on phone" and "record what I actually bet," which can only happen back at a PC — sometimes hours after the bet was placed, when exact odds/amounts are harder to recall.
2. **The spreadsheet is both the source of truth and the thing you hand-edit.** Every manual touch (entering `bet_amount`, `bet_selection`, fixing a row) is a chance for a typo, a misplaced row, or — since it's a binary file — a git merge conflict that can't be resolved with a normal diff.

Everything below is designed to fix those two problems without changing anything about the model, the feature pipeline, or the odds-fetching logic.

## 2. Your actual workflow today (for reference)

1. Trigger `fetch_predictions.yml` from your phone (Actions tab → Run workflow).
2. The Action fetches the schedule + lines, runs predictions, writes new rows into `betting_tracker.xlsx`, opens a PR.
3. You review the recommendations (console output / PR diff) on your phone.
4. You merge the PR to `main` — this is your record-keeping step.
5. You place bets on your sportsbook app(s) — completely outside this system.
6. **You wait until you're at your PC.**
7. You open `betting_tracker.xlsx` and manually fill in `bet_amount`, `bet_selection`, `notes` for whichever rows you actually bet — including rows below the 12% EV threshold, since you sometimes bet those too.
8. Next day, `update_results_pr.yml` runs, fetches results, opens a PR, you merge it.
9. You occasionally run `betting_dashboard.py` to check performance.

Step 6 is the entire problem this proposal targets. Steps 1–4, 8, and 9 already work fine and are not changing in spirit.

## 3. Proposed architecture

**Core idea:** separate the *system of record* (durable, structured, script-owned data) from the *human interaction surface* (what you actually touch on your phone). Right now those are the same file. They shouldn't be.

```
  fetch_predictions ──┐
  record_bet ─────────┤  all three commit
  update_results ─────┤  directly to main
                       ▼
                  ┌─────────────────────┐
                  │   data/betting.db    │
                  │      (SQLite)        │
                  │  committed to git,    │
                  │  single source of     │
                  │       truth           │
                  └──────────┬───────────┘
                             │ regenerated every run
                             ▼
                   betting_tracker.xlsx
                (read-only snapshot for
                 browsing on your PC —
                   never hand-edited)
```

| Today | Proposed |
|---|---|
| `betting_tracker.xlsx` is read **and written** by every script | `data/betting.db` (SQLite) is read/written by every script |
| You hand-edit the xlsx to record a bet | You fill out a form (GitHub Action `workflow_dispatch` inputs) from your phone, which commits directly to `main` |
| xlsx is the git-tracked record | Both `data/betting.db` and the generated `betting_tracker.xlsx` are git-tracked — full history of every prediction, bet, and result lives in `git log`, but you never hand-edit either file directly |
| Merge conflicts possible on binary file | Nothing hand-edits the DB, so there's nothing to conflict — every write goes through a script. `record_bet`, `fetch_predictions`, and `update_results` all push directly to `main` with a fetch-and-rebase retry loop; a genuine race (two runs within seconds of each other) makes the losing run fail loudly with a clear error rather than silently overwriting anything — see §10 for why this is safe even without the PR review buffer originally planned for fetch/update |

## 4. The new daily workflow

1. Trigger `fetch_predictions.yml` from your phone.
2. Action fetches schedule + lines, runs predictions, writes rows into `data/betting.db`, regenerates `betting_tracker.xlsx` from the DB, and **commits directly to `main`**. No PR, no separate merge tap.
3. You review recommendations on your phone (console/job output, or just browse the updated `betting_tracker.xlsx` on GitHub).
4. You place a bet on your sportsbook app.
5. Right then, still on your phone — Actions tab → `record_bet.yml` → Run workflow. Fill in a short form:
   - `date` (defaults to today)
   - `goalie_name`
   - `book` (dropdown: Underdog / BetOnline / BetMGM / Caesars / PrizePicks / other)
   - `bet_selection` (dropdown: OVER / UNDER)
   - `bet_amount`
   - `notes` (optional)
   - This works **identically** whether the model recommended the bet, recommended the other side, or said NO BET — it's just writing your decision, completely independent of the model's. This directly covers your "I sometimes bet below the EV threshold" case: the form doesn't know or care what the recommendation was.
   - If more than one row matches (e.g. the line moved during the day and got re-fetched), the script matches the **most recently fetched row** automatically — no extra input needed from you.
6. The workflow finds the matching row in the DB, fills in your bet info, regenerates the xlsx snapshot, and commits directly to `main`. No PC required, no risk of forgetting the exact odds by the time you get home.
7. Results update — same shape as fetch: `update_results.yml` fetches completed game results and commits directly to `main`.
8. Dashboard — reads from SQLite instead of xlsx (faster, and lets you run ad-hoc SQL queries on your own betting history if you ever want to).

Net change to your loop: **there's no more "wait for PC, hand-edit spreadsheet" step, and no more PR-merge taps anywhere in the daily loop.** Fetch, bet, record, and check results are each a single phone action.

## 5. Database schema

One table, deliberately mirroring the current Excel columns so the mental model doesn't change — just where it lives:

```sql
CREATE TABLE bets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date         TEXT NOT NULL,
    game_id           INTEGER NOT NULL,
    book              TEXT NOT NULL,
    goalie_name       TEXT NOT NULL,
    goalie_id         INTEGER,
    team_abbrev       TEXT,
    opponent_team     TEXT,
    is_home           INTEGER,
    betting_line      REAL,
    line_over         INTEGER,
    line_under        INTEGER,
    predicted_saves   REAL,
    prob_over         REAL,
    confidence_pct    REAL,
    confidence_bucket TEXT,
    recommendation    TEXT,        -- model's call: OVER/UNDER/NO BET
    ev                REAL,
    bet_amount        REAL,        -- your call (independent of recommendation)
    bet_selection     TEXT,        -- your call: OVER/UNDER/NONE
    bet_placed_at     TEXT,        -- NEW: timestamp from record_bet, for audit/recall
    actual_saves      REAL,
    result            TEXT,        -- WIN/LOSS/PUSH/NO BET
    profit_loss       REAL,
    notes             TEXT,
    UNIQUE(game_id, goalie_name, book, betting_line, line_over, line_under)
);
```

The `UNIQUE` constraint is what `fetch_and_predict.py` uses to dedupe new lines — same key its current pandas-mask logic already uses, just enforced by the DB.

`record_bet` matches more loosely, on `game_date` + `goalie_name` + `book` only (not the full unique key, since you won't know the exact line/odds when filling out the form from memory). If that match is ambiguous — multiple rows because the line moved and got re-fetched during the day — it resolves to the row with the highest `id` (i.e. the most recently inserted / most current line), with no extra input required from you.

## 6. Component-level changes

**New files:**
- `data/betting.db` — SQLite database, **committed to git** (full history of every prediction, bet, and result via `git log`, same way the xlsx is today)
- `src/betting/db_manager.py` — replaces `excel_manager.py`'s role as the read/write layer; same method names (`append_games`, `update_predictions`, `update_results`, `get_todays_games`) so the calling scripts barely change
- `src/betting/excel_export.py` — new, regenerates `betting_tracker.xlsx` from the DB (formatting/coloring logic lifted from the current `excel_manager.py`)
- `scripts/record_bet.py` — the script the new Action calls; takes the form inputs, matches the most recent matching row, updates it
- `.github/workflows/record_bet.yml` — `workflow_dispatch` with the form inputs from §4 step 5; commits directly to `main` (no PR)

**Modified files:**
- `scripts/fetch_and_predict.py` — swap `BettingTracker` (Excel) for the new DB manager; add an export call at the end
- `scripts/update_betting_results.py` — same swap
- `scripts/betting_dashboard.py` — read from SQLite instead of `pd.read_excel`
- `scripts/init_betting_tracker.py` — becomes "create the SQLite schema" instead of "create the xlsx workbook"
- `.github/workflows/fetch_predictions.yml` — replaces the old dead-end version that ran the script but never persisted anything; now commits directly to `main`
- `.github/workflows/update_results.yml` — renamed from `update_results_pr.yml`; switched from PR-based to direct commit (see §10 for why this ended up simpler than keeping PRs)

**Retired files:**
- `.github/workflows/fetch_predictions_pr.yml`, `.github/workflows/update_results_pr.yml` — superseded by the direct-commit versions above

**Unchanged — this is purely a tracking-layer change:**
- `src/betting/predictor.py`, `feature_calculator.py`, `nhl_fetcher.py`, `odds_fetcher.py`, `odds_utils.py`, `metrics.py`
- The model, the 114 features, the EV math, the 12% threshold — none of it moves

## 7. Migration plan

1. One-off script: load `data/betting_history/season_20252026.csv` (the consolidated archive from the cleanup) plus the current `betting_tracker.xlsx` sheets into the new `bets` table.
2. Verify row counts match and spot-check a handful of known results (a couple of WINs, a PUSH, a NO BET).
3. Generate the first Excel snapshot from the DB and diff it visually against the current file to confirm formatting parity.
4. Cut over: `fetch_and_predict.py` and `update_betting_results.py` start writing to the DB; old Excel-writing code path retired.

## 8. Decisions

1. **SQLite file is committed to git.** Full history of every prediction, bet, and result lives in `git log`, same as the xlsx does today.
2. **`record_bet` commits directly to `main`.** No PR step — it's low-stakes data, and if you fat-finger the form, you just run it again to correct it.
3. **Ambiguous matches resolve to the most recent row.** If a goalie has multiple lines from the same book on the same day (line moved, re-fetched), `record_bet` automatically picks the most recently fetched one — no extra input needed from you.
4. **Out of scope:** betting on something `fetch_and_predict` never fetched a line for. Not a real-world case worth designing for — every line you'd bet on already comes through the normal fetch.

## 9. What does NOT change

- The model, the 114 features, train/serve parity, the 12% EV threshold
- Underdog + BetMGM/Caesars line fetching
- The shape of your phone-first daily loop (trigger → review → merge → bet → record → results)
- Your ability to bet independently of what the model recommends, at any EV level

## 10. Implementation notes (deviations from the plan above)

- **Historical data spans two sources.** `betting_tracker.xlsx`'s January sheets were lost when the workbook was recreated mid-season, so the migration pulls January rows from the CSV archive and Feb–Apr rows from the live xlsx (which is authoritative for that range — it reflects final state, while the CSV is a set of point-in-time daily snapshots). See `scripts/migrate_to_sqlite.py` for the exact logic.
- **Blank template rows were dropped, not migrated.** Rows with no `goalie_name` or no `betting_line` (leftover blanks from the now-deleted `populate_daily_games.py` / `add_manual_lines.py` workflow) carry no information and were excluded.
- **Pre-multibook rows (Jan 4–20) had no `book` value.** Since `book` is `NOT NULL` in the new schema, these were backfilled as `'Unknown'` rather than guessed.
- **The live xlsx had ~160 literal duplicate rows** from a latent bug in the old dedup check (comparing `NaN == NaN` in pandas, which is always `False`, so a row with no `line_under` could never register as an existing duplicate). The new schema's `IS`-based NULL-safe comparisons fix this going forward; duplicates were collapsed during migration.
- **`record_bet.yml`'s commit step uses job-level `env:` for all form inputs**, including in the commit message, rather than interpolating `${{ inputs.* }}` directly into shell commands — avoids shell-injection/quoting issues from arbitrary form text.
- **The push-retry loop aborts cleanly on a rebase conflict** (rare: two workflow runs modifying `data/betting.db` within seconds of each other) rather than leaving the runner in a broken rebase state — it fails the job with a clear message to re-run.
- **`PrizePicks` was added to the `record_bet` book dropdown**, since it appears in real historical data even though `fetch_and_predict.py` doesn't currently fetch it automatically (its fetcher exists in `odds_fetcher.py` but is commented out).
- **`fetch_predictions.yml` and `update_results.yml` were switched from PR-based to direct-commit**, after the fact, in favor of simplicity. The original reasoning for keeping them PR-based was that a git conflict on `data/betting.db` would be more expensive to recover from for these two (both call rate-limited/metered external APIs before ever touching git, so redoing a failed direct-push would burn API quota on a retry). That's still true in principle, but the risk is low probability, and both `fetch_predictions` and `record_bet` already share the same fetch-and-rebase retry loop that fails cleanly (not silently) on a genuine conflict — see the README's "Concurrent writes" note. Net effect: one less kind of tap (no more PR merges anywhere in the daily loop), traded for a small chance of needing to re-run a workflow if two runs race within the same few seconds.
