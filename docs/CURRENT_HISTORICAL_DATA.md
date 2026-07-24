# Current Historical Data

Written 2026-07-02. This is a full audit of every historical data source this
project has, how they relate to each other, what was done to extend the
training data through the end of the 2025-26 season, and an honest read on
whether the current volume of data is enough to train and evaluate the model
well. See [MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md) for how the
pipeline scripts work; this document is about the data itself, not the code.

## 1. The three data sources

There are three genuinely separate pools of historical data in this repo.
Before 2026-07-02 they had never been combined.

### A. Cached Odds-API historical data (`data/raw/betting_lines/`)

A one-time historical backfill of betting lines, fetched via a now-deleted
script (`scripts/fetch_all_betting_lines.py`, removed in commit `8b1f3ab` and
not recovered — only the script that *reads* this cache still exists).

- `betting_lines.json` — 1,966 games, one row per game with home/away goalie
  lines pre-averaged across books. Covers **2024-10-04 to 2026-01-03** (January
  2026 is partial; collection stopped mid-month).
- `data/raw/betting_lines/cache/odds_*.json` — 1,976 raw per-event,
  per-bookmaker responses from The Odds API (`player_total_saves` market).
  Same date range. Six bookmakers appear: BetMGM, BetOnline (`betonlineag`),
  Bovada, DraftKings, Caesars (`williamhill_us`), Fanatics.
- This is a frozen historical snapshot. Nothing updates it anymore — the
  script that built it is gone, and the live daily workflow (`fetch_and_predict.py`)
  writes to `data/betting.db` instead, not to this cache.

### B. Betting tracker (`data/betting.db`)

The live, phone-driven daily workflow's system of record (see
`src/betting/db_manager.py`). Every line the model has seen and every bet
placed, one row per (game, goalie, book).

- **1,789 rows**, 566 distinct games, 78 distinct goalies.
- Covers **2026-01-04 to 2026-04-13** — starts the day immediately after the
  Odds-API cache above stops. This is not a coincidence: it's the exact
  moment the project switched from the offline historical-backfill pipeline
  to the live GitHub-Actions-triggered workflow.
- Book breakdown: Underdog 1,103, BetOnline 274, Unknown 187 (early-season
  rows before book attribution was being captured), PrizePicks 128, BetMGM 95,
  Manual 2.
- Unlike the Odds-API cache, this has **real outcomes attached** —
  `actual_saves` and `result` (WIN/LOSS/PUSH) are populated for ~98% of rows,
  because `update_betting_results.py` fills them in the day after each game.

### C. Boxscore / game-stats cache (`data/raw/boxscores/`, `data/cache/api_cache.db`)

The underlying per-game goalie and team stats (saves, shots against, goals
against, etc.) that every rolling feature is built from. This is separate
from both betting sources above — it's just "what actually happened in the
game," collected via `scripts/collect_historical_data.py` from the NHL's
public API.

- 5,248 boxscore JSON files, covering the 2022-23 through 2025-26 seasons in
  full (2022-10-07 onward).
- As of 2026-07-02 (before this session's refresh), 656 of the 2025-26
  season's 1,312 games were stuck as stale `"gameState": "FUT"` (future/
  unplayed) placeholders — they'd been pre-fetched before those games were
  played and never refreshed with final results. This matches the same
  2026-01-03/04 boundary as everything else: the feature pipeline froze at
  exactly the point the live tracker took over.

## 2. Why everything froze at the same date

This is the throughline: **the Odds-API cache, the boxscore cache, and all
three training parquets all stopped on 2026-01-03/04**, the same day
`data/betting.db` picks up. Nobody was doing anything wrong — the offline
historical pipeline (`create_clean_features.py` → `merge_betting_lines.py` →
`add_market_features.py` → `build_multibook_training_data.py`) simply never
got re-run after the live daily tracker workflow took over day-to-day
operation. The two systems ran in parallel with no bridge between them for
about 3.5 months (566 real, played-out games) before this session connected
them.

## 3. What was done to extend the data (2026-07-02)

Backed up the three training parquets first, then:

1. **Refreshed all 1,312 boxscores for the 2025-26 season** via
   `python scripts/collect_historical_data.py --games-only --seasons 20252026`.
   This hit the live NHL API and converted all 656 stale `FUT` placeholders to
   final results — 100% success rate, 0 failures. Also recovered 2 games
   (`2025020655`, `2025020656`) that had been silently missing from the
   dataset entirely, and picked up 4 genuine NHL official stat corrections
   (games where the league revised a save count by ±1 after initial posting —
   verified against the internally-consistent `saves`/`shotsAgainst`/
   `goalsAgainst` fields in the corrected boxscore JSON, not a parsing bug).
2. **Regenerated `clean_training_data.parquet`** from the refreshed
   boxscores — now 10,496 rows (up from 9,180), extending through
   **2026-04-16**.
3. **Extended `scripts/merge_betting_lines.py`** with a new
   `load_betting_lines_from_tracker()` function that pulls a consensus
   betting line (averaged across books) from `data/betting.db` for any date
   past the Odds-API cache's coverage, then concatenates it with the existing
   JSON-cache loader. Re-ran it plus `add_market_features.py` — now 4,755
   rows (up from 3,757), extending through **2026-04-13**.
4. **Extended `scripts/build_multibook_training_data.py`** with a new
   `parse_betting_db()` function that pulls genuine per-bookmaker odds
   directly from `data/betting.db` (one record per game/goalie/book row,
   preserving true multi-book granularity — the tracker's own multi-book
   coverage carries straight through) for dates past the raw cache's
   coverage. Re-ran it — now 6,916 rows (up from 5,408), extending through
   **2026-04-13**.

### Verification performed before trusting any of it

- Every pre-2026-01-04 row in all three parquets was diffed against the
  pre-change backups, key-by-key. Only the 4 rows affected by the genuine NHL
  stat corrections changed (plus their downstream `.shift(1)` rolling-window
  ripples, on the order of 40 rows total, exactly as expected — no more, no
  less). Zero unexpected drift.
- Spot-checked a real, resolved bet (goalie Wolf, 2026-02-02, OVER bet,
  LOSS) end-to-end from `betting.db` through to the final multibook parquet —
  `saves=18`, `over_hit=0` matched correctly across every bookmaker line for
  that game.
- Confirmed a pre-existing duplicate-row quirk in the multibook dataset
  (~7-8% duplicate rate on `(game_id, goalie_id, book_key, betting_line)`,
  mostly from `book='unknown'` early-tracker rows and the same bookmaker
  appearing across multiple raw cache files) exists at the same rate in the
  old and new data — not something introduced by this extension, and out of
  scope to fix here.
- No new NaNs introduced in `saves`, `betting_line`, `over_hit`, or odds
  columns anywhere in the newly added rows.

Nothing here changed the model itself — no retraining or re-evaluation was
performed. The extended parquets are just sitting there, ready for whenever
that's the next step.

## 4. Current state (as of 2026-07-02)

| Dataset | Rows | Columns | Date range | Notes |
|---|---|---|---|---|
| `clean_training_data.parquet` | 10,496 | 114 | 2022-10-07 to 2026-04-16 | All 4 seasons of raw game features, no betting lines required (now carries `goalie_name`) |
| `classification_training_data.parquet` | 6,714 | 130 | 2023-11-02 to 2026-04-13 | One row per goalie-game, single consensus line, `over_hit` target (2023-24 added 2026-07-24, see 4.4) |
| `multibook_classification_training_data.parquet` | 20,799 | 138 | 2023-11-02 to 2026-04-13 | One row per (goalie, game, bookmaker, line) — **this is what feeds the production model** (2023-24 added 2026-07-24, see 4.4) |

**Updated 2026-07-07**: the multibook dataset was regenerated after fixing the
line-misattribution bug and a UTC/local date-shift bug in
`scripts/build_multibook_training_data.py` (see
[OFFSEASON_OPTIMIZATION_PLAN.md](OFFSEASON_OPTIMIZATION_PLAN.md) section 2).
Rows went from 6,916 (79.7% home goalies, 1,413 unique goalie-games, ~44% of
tracker-era rows carrying the opposing goalie's line) to 13,192 clean rows
(51.4% home, 4,580 unique goalie-games, 2,182 of 2,398 games with both
goalies present, zero duplicates, zero misattribution). Pre-fix parquets are
backed up in `data/processed/backup_20260707/`. The production model has NOT
been retrained on this yet.

Season breakdown of the multibook dataset (the one that matters for
training): `20232024` = 7,607 rows, `20242025` = 7,463 rows, `20252026` = 5,729
rows (the 2023-24 slice added 2026-07-24, section 4.4). So in practice, usable
betting-line-labeled data is now **three NHL seasons** (it was just under two
before the 2023-24 fold). Note the 2023-24 slice is sportsbook-only (no DFS
books existed for that season), a disclosed structural difference from the
other two seasons.

`data/raw/boxscores/` now shows all 5,248 games (all 4 seasons) as finalized
(`OFF` state) — no stale placeholders remain anywhere.

### 4.1 W1 market-coverage probe (Codex-authored, 2026-07-13)

The repository now also holds 24 append-only historical event-odds probe
responses under `data/raw/betting_lines/probes/w1_market_coverage/` (eight
games per season across 2023-24, 2024-25, and 2025-26). They contain standard
and alternate goalie saves plus player SOG for a nine-book named set at the
bettime anchor. The probe cost 800 credits and left 51,465. (The season-scale
purchase it gated has since been executed -- see section 4.2.)

Coverage implications: sportsbook SOG is broad enough for the planned W1
development pass; PrizePicks goalie-saves history is available from 2024-25
onward; Underdog historical goalie saves and all 2023-24 DFS props were absent
from the sample; and alternate saves after 2023-24 are over-only rather than a
paired probability curve. See `HISTORICAL_DATA_ANALYSIS.md` section 9 for the
verified counts and schema cautions.

### 4.2 Core bet-time passes (Claude-authored, 2026-07-14)

`data/raw/betting_lines/passes/core_bettime_202607/` holds 2,626 append-only
event-odds records from the two authorized core purchases (script:
`scripts/purchase_core_bettime_passes.py`; independent audit:
`scripts/audit_core_bettime_passes.py` and `audit_summary.json` alongside the
records). Total spend 38,570 credits, balance 51,465 -> 12,895, every credit
reconciled against response headers.

- `combined-2024-25`: 1,313 events at the bettime anchor, markets
  `player_total_saves,player_shots_on_goal`, nine named books. 1,301 events
  have SOG, 1,244 have saves (betonlineag saves 1,050; prizepicks saves
  1,139; underdog saves 0 -- SOG only). 1,233 of the saves events intersect
  the existing 2024-25 closing archive: that is the CLV/W6-usable set.
- `sog-2023-24`: 1,313 events, `player_shots_on_goal` only. 1,312 have SOG
  with at least two paired books on essentially every event; every existing
  2023-24 bettime-saves event now has matching SOG.

Ingestion is complete under `PREREGISTRATION_NO_CREDIT_ABLATIONS.md` section
14.5. `scripts/build_core_bettime_pass_snapshots.py` produced
`data/processed/core_bettime_202607_snapshots.parquet` without mutating the
existing snapshots parquet. Final contents: 413,758 rows / 23 columns --
2023-24 SOG 214,252 rows across 1,310 events; 2024-25 SOG 182,686 / 1,301;
2024-25 saves 16,820 / 1,244. Saves-row goalie matching is 98.93%.

Applied exclusions and checks:

- Three events failed the registered `effective_gap_minutes >= 10` rule:
  BUF@NJD 2024-10-05 (-24.8), BOS@PHI 2024-01-27 (5.0), and PHI@OTT
  2023-10-14 (9.2). The audit's 80 cached-to-true drift events measured a
  different quantity and were not blanket-excluded.
- A price-aware outcome key finds 5,293 exact duplicate extras across all
  200 responses and 5,282 after drift exclusion. The audit's 5,296 total
  omitted price from its grouping key and therefore included three
  conflicting-price groups.
- The three FanDuel conflicts contain six rows and were excluded entirely;
  no tie-break was used. Fanatics remained absent.
- There are no null lines, null prices, invalid sides, or surviving
  sub-10-minute rows. An independent raw-record reconstruction matched every
  one of the 413,758 final outcome keys.
- Eleven new 2024-25 event ids overlap the old 21-event bettime fragment;
  downstream joins must deduplicate them.

The two 404 records (wildfire-postponed CGY@LAK 2025-01-08 and dead CHI@BUF
2024-01-18) remain covered under reissued ids elsewhere in the raw pass.
The processed parquet is data-ready for registered Experiments 11-13.
Experiment 11 has now consumed only its 2024-25 saves slice and produced the
verified PASS recorded in `HISTORICAL_DATA_ANALYSIS.md` section 9.6;
Experiment 13 has now consumed its PrizePicks saves slice and closed as the
verified null recorded in section 9.7. Experiment 12 consumed both its
original and registered recovery touches. The persisted recovery calculation
was numerically encouraging, but a post-touch metadata failure leaves the
official verdict `NO VERDICT -- INFRASTRUCTURE FAILURE`; see section 9.8 and
preregistration section 15.11. No further historical W1 touch is allowed.

### 4.3 2025-26 bet-time saves completion buy (2026-07-24)

`data/raw/betting_lines/passes/saves_fill_2526_202607/` holds 481 append-only
event-odds records from a durable-data purchase that completed and re-anchored
the 2025-26 bet-time `player_total_saves` archive (script:
`scripts/purchase_2526_bettime_saves_fill.py`; independent audit:
`scripts/audit_2526_bettime_saves_fill.py` and `audit_summary.json`;
preregistration section 20; not a hypothesis test -- pure acquisition).

Before this buy, 2025-26 bet-time saves coverage in
`saves_lines_snapshots.parquet` was only 781 of 1,232 in-window cached events
(~60%), versus ~95% for 2024-25 and ~86% for 2023-24, and 49 of the owned
events were mis-anchored. The buy set was every in-window 2025-26 cached event
NOT already owned at the anchor the purchase would request -- defined as
correctly-anchored iff at least one owned bet-time snapshot is within 300s of
`compute_bettime_ts(commence_time)` computed from the CACHE `commence_time`
(NOT the snapshot's own, which disagrees by up to 30 min on 85 events), a
min-gap-over-all-snapshots test (68 events carry two snapshots, so no
single-row dedup is valid). That yields exactly **481** events (451
truly-missing + 30 mis-anchored; sha256 of the sorted buy-set `96163617c977a9c5`).

Actuals: 481/481 calls, `player_total_saves` only at the bettime anchor, nine
named books. **439 events returned a saves line** (billed 10 credits each), 41
returned zero markets (free), 1 was a free 404. Total spend **4,390 credits**,
balance **11,055 -> 6,665**, every credit reconciled against response headers;
audit VERDICT CLEAN, independently re-verified by the lead (see section 20.9).

Ingested (2026-07-24, zero credits): `scripts/build_saves_fill_2526_snapshots.py`
(a pure parser mirroring `build_core_bettime_pass_snapshots.py`, reusing
`build_odds_snapshots.py`'s 15-column schema and goalie-matching helpers)
parsed the 439 saves-market records into a SIBLING parquet,
`data/processed/saves_fill_2526_202607_snapshots.parquet` (7,357 rows x 15
cols, schema/dtypes byte-identical to `saves_lines_snapshots.parquet`, a
drop-in `pd.concat`; the existing archive was NOT mutated, same pattern as the
core-pass sibling). On the union, the registered min-gap-over-all-snapshots
test moves 2025-26 bet-time saves correctly-anchored coverage from **751 /
1,232 (60.96%) to 1,190 / 1,232 (96.59%)** -- now the best-anchored of the
three owned seasons. Of the 30 previously mis-anchored events, 24 were fixed;
6 remain mis-anchored (they returned zero bookmakers on the re-buy). Full
detail and independent verification: PREREGISTRATION section 20.10. These new
2025-26 bet-time lines were deliberately NOT folded into the training parquets
(they overlap tracker-sourced 2025-26 rows already in training; low marginal
value, real duplication risk -- see 4.4). **6,665 credits remain, expiring
2026-07-31, with no further planned use.**

### 4.4 2023-24 saves folded into the training parquets (2026-07-24)

The 2023-24 bet-time saves lines -- owned since the core passes but never in the
training pipeline (PREREGISTRATION 20.1 item 3 named this a separate follow-on)
-- were folded into the two betting-line-carrying training parquets as a NEW
third season, giving the production training set three seasons for the first
time (the substrate sections 5-6 name as the binding constraint on walk-forward
validation). User-authorized; 2023-24 only, bet-time only, all sportsbooks,
strictly additive, NO model retrain.

Method (append-only, NOT a pipeline rerun): a Stage-1 reproducibility test
found `build_multibook_training_data.py` no longer reproduces the 2026-07-07
multibook parquet from current code -- the raw cache has grown since, shifting
the cache-vs-tracker boundary and dropping the tracker-sourced `underdog` (927)
and `betonline` (174) rows on any rerun. So a full rerun was rejected; instead
the 2023-24 rows were built in isolation (sourcing bet-time lines from
`saves_lines_snapshots.parquet`, collapsing the double-bettime-snapshot events
via the repo's own `clv_audit_pace_policy.clean_bettime_pass` earliest-
`requested_ts` rule, reusing `merge_betting_lines`/`add_market_features`/
`compute_line_relative_features` unchanged so column semantics match exactly)
and appended onto the existing parquets. Pre-change parquets backed up to
`data/processed/backup_20260724/`.

Result: `classification_training_data.parquet` 4,755 -> 6,714 (+1,959 goalie-
games); `multibook_classification_training_data.parquet` 13,192 -> 20,799
(+7,607 per-book rows across 1,123 events / 1,959 goalie-nights, books
`williamhill_us`/`draftkings`/`fanduel`/`bovada`/`betmgm`). Verified from disk:
every existing 2024-25/2025-26 row byte-identical to the backup, only 2023-24
rows added, zero new nulls in saves/betting_line/over_hit/odds, schema/dtypes
unchanged (130 and 138 cols), `clean_training_data.parquet` untouched. The
2023-24 goalie-night count (1,959) cross-checked identically via two
independent match paths. Caveat: 2023-24 is sportsbook-only (no DFS books
existed then), a disclosed structural difference from the other two seasons. As
in section 3, this only extends the parquets -- the production model has NOT
been retrained or re-evaluated on the three-season data; a walk-forward
evaluation is now possible (substrate-wise) but remains a separate future
decision.

## 5. Is this enough data?

Workable, but thin. Two full seasons of betting-line-labeled data (4,755-6,916
rows depending on granularity) is on the low side for an XGBoost classifier
carrying 114 features. The production model already leans on heavy
regularization (`reg_alpha=20, reg_lambda=60, min_child_weight=30`) — that's
not a stylistic choice, it's a direct response to overfitting risk at this
sample size.

The bigger constraint isn't the training set, it's the **test set**. With a
chronological 60/20/20 split, the held-out test fold is only ~950-1,380 rows.
For a bet with a modest real edge (model says 55% vs. the market's implied
52%), the standard error on a sample that size is roughly 1.5-2 percentage
points — meaning a single headline ROI number from one test window carries
real uncertainty that's easy to forget once it's reported as a single
figure.

**Another season or two would meaningfully help**, mainly by making the
out-of-sample performance estimate trustworthy rather than by adding raw
training volume. Two caveats:

- Older seasons are still useful even as specific goalies retire or change
  teams, because the model trains on engineered features (rolling averages,
  line-relative features), not goalie identity — the signal generalizes.
- There's a ceiling. Go back far enough (pre-2021ish) and you hit
  COVID-shortened seasons, rule changes, and general "meta drift" in how the
  game is played — noise, not signal. Three to four seasons total is
  probably close to the sweet spot. It's also worth checking whether The Odds
  API's historical archive even extends further back than what's already
  cached here — historical odds access is often tiered/paywalled, so growing
  this further may only be possible going forward (~one more season's worth
  of games each year) rather than backward.

## 6. Chronological vs. random train/test split

Chronological — not a close call for this problem, for two reasons:

1. **Leakage via correlated rows, not just literal future data.** Even
   though every rolling feature already uses `.shift(1)` to exclude the
   current game, a random split can still put a goalie's Tuesday start in
   train and their Friday start (3 days later, nearly identical rolling-window
   values) in test. That's effectively a near-duplicate row split across
   train/test — it inflates test accuracy without meaning anything.
2. **It's the only split that matches how the model is actually used.**
   In production, you're always betting on games that happen after
   everything you've trained on. A random split answers "can this model
   interpolate within a period it's already partly seen," which is a
   different and less useful question than "would this have made money on
   games it had never seen." Random splitting would almost certainly
   overstate real-world ROI.

The real cost of a strict chronological split is that a single cut is at the
mercy of whatever happened to occur in that particular test window (an
injury run, trade deadline chaos, etc. could make the held-out period
unusually easy or hard, independent of the model's actual quality). The
standard mitigation is **walk-forward validation**: instead of one
chronological train/test cut, do several rolling chronological splits (train
on season 1, test on season 2; train on seasons 1-2, test on season 3; etc.)
and average the results. **As of 2026-07-24 the training parquets carry three
seasons** (2023-24 was folded in, section 4.4), so a walk-forward evaluation is
now runnable for the first time — train 2023-24 -> test 2024-25 -> test 2025-26.
It is now **preregistered as `PREREGISTRATION_NO_CREDIT_ABLATIONS.md` section 21**
(frozen production recipe, retrained per fold; game-level bootstrap CIs; PASS/FAIL
bar fixed in advance) but **not yet run**. The per-window test folds remain thin
(section 5's sampling-uncertainty caveat applies to each individual cut), which
is why the registered bar leans on a pooled bootstrap CI plus per-fold
positivity; but the two-season blocker is resolved.

## 7. Open follow-ups

- Historical player props are vendor-available after 2023-05-03, but the W1
  probe found no Underdog/PrizePicks data in the sampled 2023-24 games. Treat
  2024-25 as the practical start of the DFS archive unless a later targeted
  probe demonstrates otherwise.
- As each new NHL season completes, re-run the same extension pattern used
  in §3 (refresh boxscores → `create_clean_features.py` →
  `merge_betting_lines.py` → `add_market_features.py` →
  `build_multibook_training_data.py`) to keep growing the labeled dataset.
- Three seasons of betting-line data now exist (2023-24 folded in 2026-07-24,
  section 4.4), so the deferred step is now actionable and **has been registered
  as `PREREGISTRATION_NO_CREDIT_ABLATIONS.md` section 21** (walk-forward
  validation of the frozen production classifier recipe; not yet run). Mind that
  the 2023-24 season is sportsbook-only (no DFS books) when interpreting per-book
  or DFS-specific results across the walk-forward windows.
